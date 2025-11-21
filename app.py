# Backend/app.py
import os
import re
import sys
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ---- optional but recommended (.env support) ----
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_kw):  # no-op if package not installed
        pass

# ---------------------------
# Load env (.env in Backend/)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": cors_origins or "*"}})

# ---------------------------
# Ensure PyMySQL present
# ---------------------------
try:
    import pymysql  # noqa: F401
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyMySQL is not installed in this Python interpreter.\n"
        f"Interpreter: {sys.executable}\n"
        "Fix: activate your venv and run:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install PyMySQL\n"
    )

# ---------------------------
# MySQL via SQLAlchemy
# ---------------------------
from sqlalchemy.engine import URL

MYSQL_USER = os.getenv("MYSQL_USER", "autoplus")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB", "autoplus")

if "@" in MYSQL_HOST:
    raise ValueError(
        f"MYSQL_HOST is invalid: {MYSQL_HOST!r}. It must NOT contain '@'. "
        "If your password contains '@', keep it only in MYSQL_PASSWORD."
    )

db_url = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    database=MYSQL_DB,
    query={"charset": "utf8mb4"},
)

safe_url = str(db_url)
if MYSQL_PASSWORD:
    safe_url = safe_url.replace(MYSQL_PASSWORD, "********")
print(f"[DB] Python: {sys.executable}")
print(f"[DB] Connecting to: {safe_url}")

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------------------
# Auth (JWT)
# ---------------------------
import jwt  # PyJWT


def make_token(user_id: int) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")


def decode_token(token: str):
    try:
        return jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
    except Exception:
        return None


# ---------------------------
# User model
# ---------------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    mobile = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_public(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "mobile": self.mobile,
            "created_at": self.created_at.isoformat(),
        }


with app.app_context():
    db.create_all()

# ---------------------------
# Auth routes
# ---------------------------
@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    mobile = (data.get("mobile") or "").strip()
    password = data.get("password") or ""

    if len(name) < 2:
        return jsonify(ok=False, error="Name must be at least 2 characters"), 400
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify(ok=False, error="Invalid email"), 400
    if len(password) < 8:
        return jsonify(ok=False, error="Password must be at least 8 characters"), 400
    if User.query.filter_by(email=email).first():
        return jsonify(ok=False, error="Email already in use"), 409

    user = User(
        name=name,
        email=email,
        mobile=re.sub(r"\D", "", mobile),
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()

    token = make_token(user.id)
    return jsonify(ok=True, token=token, user=user.to_public())


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify(ok=False, error="Invalid email or password"), 401

    token = make_token(user.id)
    return jsonify(ok=True, token=token, user=user.to_public())


@app.get("/api/auth/me")
def me():
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "", 1) if auth else ""
    payload = decode_token(token)
    if not payload:
        return jsonify(ok=False, error="Invalid token"), 401
    user = db.session.get(User, payload["sub"])
    if not user:
        return jsonify(ok=False, error="User not found"), 404
    return jsonify(ok=True, user=user.to_public())


# ---------------------------
# CLIP model / Upload
# ---------------------------
CLIP_AVAILABLE = True
CLIP_ERROR = ""

try:
    import torch
    import pandas as pd
    from PIL import Image
    import clip  # clip-anytorch
except Exception as e:
    CLIP_AVAILABLE = False
    CLIP_ERROR = f"{type(e).__name__}: {e}"
    print("[CLIP] Not available:", CLIP_ERROR)

# --- enable AVIF/HEIC decoding in Pillow ---
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    try:
        import pillow_avif  # noqa: F401
    except Exception:
        pass

if CLIP_AVAILABLE:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"[CLIP] Loaded ViT-B/32 on {device}")

    labels_path = os.path.join(DATA_DIR, "car_labels.txt")
    car_labels = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            car_labels = [line.strip() for line in f if line.strip()]

    excel_path = os.path.join(DATA_DIR, "car_dataset.xlsx")
    csv_path = os.path.join(DATA_DIR, "car_dataset.csv")
    if os.path.exists(excel_path):
        car_data = pd.read_excel(excel_path)
    elif os.path.exists(csv_path):
        car_data = pd.read_csv(csv_path)
    else:
        car_data = pd.DataFrame()

    DATASET_ROWS = []
    DATASET_TEXTS = []
    DATASET_TEXT_FEATURES = None

    def _row_label_text(rowdict: dict) -> str:
        rvn = str(rowdict.get("Recognised Vehicle Name", "")).strip()
        if rvn:
            return rvn
        brand = str(rowdict.get("Brand", "")).strip()
        model_name = str(rowdict.get("Model", "")).strip()
        year = str(rowdict.get("Year", "")).strip()
        parts = [p for p in [brand, model_name, year] if p]
        return " ".join(parts) if parts else (model_name or "Unknown Vehicle")
else:
    car_data = None
    DATASET_ROWS = []
    DATASET_TEXTS = []
    DATASET_TEXT_FEATURES = None


def _build_dataset_text_features():
    """Precompute CLIP text features for all vehicles (run once at startup)."""
    global DATASET_ROWS, DATASET_TEXTS, DATASET_TEXT_FEATURES, car_data

    if not CLIP_AVAILABLE or car_data is None or car_data.empty:
        DATASET_ROWS, DATASET_TEXTS, DATASET_TEXT_FEATURES = [], [], None
        print("[CLIP] No dataset available for text features.")
        return

    print("[CLIP] Building dataset text features...")
    DATASET_ROWS = []
    DATASET_TEXTS = []

    for _, row in car_data.iterrows():
        r = row.to_dict()
        DATASET_ROWS.append(r)
        DATASET_TEXTS.append(_row_label_text(r))

    feats = []
    with torch.no_grad():
        bs = 64
        for i in range(0, len(DATASET_TEXTS), bs):
            toks = clip.tokenize(DATASET_TEXTS[i:i + bs]).to(device)
            f = model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)

    DATASET_TEXT_FEATURES = torch.cat(feats, dim=0) if feats else None
    print(f"[CLIP] Dataset text features built for {len(DATASET_TEXTS)} entries.")


if CLIP_AVAILABLE:
    _build_dataset_text_features()

# ---------- helpers ----------
def _norm_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def pick_flexible(row_dict: dict, candidates):
    cols = list(row_dict.keys())
    for cand in candidates:
        if isinstance(cand, re.Pattern):
            for col in cols:
                if cand.search(col):
                    val = row_dict[col]
                    if val is not None and str(val).strip():
                        return val, col
        else:
            for col in cols:
                if str(col).lower().strip() == str(cand).lower().strip():
                    val = row_dict[col]
                    if val is not None and str(val).strip():
                        return val, col

    norm_cands = [(_norm_key(c.pattern) if isinstance(c, re.Pattern) else _norm_key(c)) for c in candidates]
    for col in cols:
        ncol = _norm_key(col)
        for nc in norm_cands:
            if isinstance(nc, str) and ncol == nc:
                val = row_dict[col]
                if val is not None and str(val).strip():
                    return val, col

    for col in cols:
        ncol = _norm_key(col)
        for nc in norm_cands:
            if isinstance(nc, str) and (nc in ncol or ncol in nc):
                val = row_dict[col]
                if val is not None and str(val).strip():
                    return val, col
    return None, None


def split_list(val):
    if val is None:
        return []
    if not isinstance(val, str):
        val = str(val)
    parts = re.split(r"[;,\|\n/]+", val)
    seen, out = set(), []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def with_units(value, matched_col):
    if value is None:
        return None
    txt = str(value).strip()
    lc = str(matched_col or "").lower()
    if 'boot space' in lc or ' cargo ' in f" {lc} " or 'cargo space' in lc:
        return f"{txt} L" if ' l' in lc or lc.endswith(' l') else txt
    if 'fuel tank capacity' in lc or 'fueltank' in lc or 'tank capacity' in lc:
        return f"{txt} L" if ' l' in lc or lc.endswith(' l') else txt
    if 'km per l' in lc or 'km/l' in lc or 'kmperl' in _norm_key(lc):
        return f"{txt} km/L"
    if 'km per full tank' in lc:
        return f"{txt} km"
    return txt


ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "avif"}

# similarity threshold for "unknown" images
SIM_THRESHOLD = float(os.getenv("CLIP_SIM_THRESHOLD", "0.28"))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _unknown_result():
    return {
        "predicted_label": "Unknown Vehicle",
        "car_info": {
            "Model": "Unknown",
            "Year": "Unknown",
            "Price": "Unknown",
            "Hybrid": "Unknown",
        },
        "spec": {
            "Category": "Unknown",
            "Engine": "Unknown",
            "Transmission": "Unknown",
            "Mileage": "Unknown",
            "Fuel Type": "Unknown",
            "Oil Consumption": "Unknown",
            "Boot Space": "Unknown",
            "Airbags": "Unknown",
            "Seating Capacity": "Unknown",
            "Fuel Tank Capacity": "Unknown",
        },
        "common_errors": [],
        "market_price": {"Estimated Price": "Unknown"},
        "similar": [],
    }


def predict_car(image_path: str):
    if not CLIP_AVAILABLE:
        raise RuntimeError(f"CLIP/Torch not available: {CLIP_ERROR}")

    img = Image.open(image_path).convert("RGB")
    MAX_SIZE = 512
    img.thumbnail((MAX_SIZE, MAX_SIZE))

    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    matched_row = None
    predicted_label = None
    similar_from_clip = []
    best_score = None

    # ---------- preferred path: dataset text features ----------
    if DATASET_TEXT_FEATURES is not None and len(DATASET_TEXT_FEATURES) > 0:
        with torch.no_grad():
            sims = image_features @ DATASET_TEXT_FEATURES.T
            k = min(4, len(DATASET_TEXT_FEATURES))
            topk_vals, topk_idx = sims.topk(k=k, dim=-1)
            idxs = topk_idx[0].tolist()
            best_score = topk_vals[0, 0].item()

        print(f"[CLIP] Best similarity score: {best_score:.3f}")

        if best_score < SIM_THRESHOLD:
            print("[CLIP] Below threshold – treating as unknown image.")
            return _unknown_result()

        best_idx = idxs[0]
        matched_row = DATASET_ROWS[best_idx]
        predicted_label = DATASET_TEXTS[best_idx]
        similar_from_clip = [DATASET_TEXTS[i] for i in idxs[1:]]
    else:
        # ---------- fallback path: label list only ----------
        car_labels_local = globals().get("car_labels", []) or []
        if car_labels_local:
            toks = clip.tokenize(car_labels_local).to(device)
            with torch.no_grad():
                text_features = model.encode_text(toks)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sims = image_features @ text_features.T
                best_vals, best_idx_tensor = sims.max(dim=-1)
                best_score = best_vals.item()
                best_idx = best_idx_tensor.item()

            print(f"[CLIP] Best label similarity: {best_score:.3f}")

            # slightly different default threshold here
            if best_score < 0.30:
                print("[CLIP] Below label threshold – treating as unknown image.")
                return _unknown_result()

            predicted_label = car_labels_local[best_idx]
        else:
            return _unknown_result()

        cd = globals().get("car_data")
        if cd is not None and not cd.empty:
            pr_norm = predicted_label.lower().strip()
            for _, row in cd.iterrows():
                rd = row.to_dict()
                name = str(rd.get("Model", "")).lower().strip()
                if pr_norm in name or name in pr_norm:
                    matched_row = rd
                    break

    if matched_row is None:
        # if we couldn't map to a concrete dataset row, still show label only
        base = _unknown_result()
        base["predicted_label"] = predicted_label or "Unknown Vehicle"
        base["similar"] = similar_from_clip
        return base

    cat_val, cat_col = pick_flexible(matched_row, ["Category", "Vehicle Category", "Type"])
    eng_val, eng_col = pick_flexible(matched_row, ["Engine", "Engine Type", "Engine Size"])
    trans_val, trans_col = pick_flexible(matched_row, ["Transmission", "Gearbox"])
    mileage_val, mileage_col = pick_flexible(
        matched_row,
        ["Mileage", "Fuel Economy", "Fuel Efficiency",
         "Mileage km per full tank", re.compile(r"mileage.*full tank", re.I)],
    )
    fuel_val, fuel_col = pick_flexible(matched_row, ["Fuel Type", "FuelType"])
    oil_val, oil_col = pick_flexible(
        matched_row,
        ["Oil Consumption", "OilCapacity", "Oil Capacity",
         "Oil Consumption Km per L", re.compile(r"oil.*(consumption|capacity)", re.I)],
    )
    boot_val, boot_col = pick_flexible(
        matched_row,
        ["Boot Space", "Boot Space L", "Cargo", "Cargo Space",
         re.compile(r"boot\s*space.*", re.I)],
    )
    airbags_val, airbags_col = pick_flexible(matched_row, ["Airbags"])
    seats_val, seats_col = pick_flexible(matched_row, ["Seating Capacity", "Seats", "Seating"])
    tank_val, tank_col = pick_flexible(
        matched_row,
        ["Fuel Tank Capacity", "Fuel Tank Capacity L", "Tank Capacity", "FuelTank",
         re.compile(r"fuel\s*tank.*", re.I)],
    )

    spec = {
        "Category": with_units(cat_val, cat_col) or "Unknown",
        "Engine": with_units(eng_val, eng_col) or "Unknown",
        "Transmission": with_units(trans_val, trans_col) or "Unknown",
        "Mileage": with_units(mileage_val, mileage_col) or "Unknown",
        "Fuel Type": with_units(fuel_val, fuel_col) or "Unknown",
        "Oil Consumption": with_units(oil_val, oil_col) or "Unknown",
        "Boot Space": with_units(boot_val, boot_col) or "Unknown",
        "Airbags": with_units(airbags_val, airbags_col) or "Unknown",
        "Seating Capacity": with_units(seats_val, seats_col) or "Unknown",
        "Fuel Tank Capacity": with_units(tank_val, tank_col) or "Unknown",
    }

    errors_raw, _ = pick_flexible(matched_row, ["CommonErrors", "Common Errors", "Issues", "Common Issues"])
    common_errors = split_list(errors_raw)

    market_price = {}
    for col, val in matched_row.items():
        if re.search(r"price", str(col), flags=re.I) and str(val).strip():
            market_price[col] = str(val).strip()
    if not market_price:
        one_price, _ = pick_flexible(matched_row, ["Price", "Market Price", "Estimated Price"])
        if one_price:
            market_price = {"Estimated Price": str(one_price).strip()}

    similar_col_raw, _ = pick_flexible(matched_row, ["Similar", "Similar Vehicles", "Alternatives"])
    similar_from_col = split_list(similar_col_raw)

    seen, similar = set(), []
    for name in similar_from_col + similar_from_clip:
        key = str(name).strip().lower()
        if key and key not in seen:
            seen.add(key)
            similar.append(str(name).strip())

    if not matched_row.get("Model"):
        matched_row["Model"] = predicted_label

    return {
        "predicted_label": predicted_label,
        "car_info": matched_row,
        "spec": spec,
        "common_errors": common_errors,
        "market_price": market_price if market_price else {},
        "similar": similar,
    }


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({"ok": True, "service": "AutoPlus backend (MySQL)"}), 200


@app.post("/upload")
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    save_path = os.path.join(UPLOADS_DIR, f.filename)
    f.save(save_path)

    try:
        result = predict_car(save_path)
        return jsonify(result)
    except Exception as e:
        print("[UPLOAD] Inference failed:", e)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
