document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const username = loginForm.querySelector("input[type='text']").value;
            const password = loginForm.querySelector("input[type='password']").value;

            if (username === "admin" && password === "1234") {
                alert("Login Successful! Redirecting...");
                window.location.href = "upload.html"; // go to Upload page
            } else {
                alert("Invalid Username or Password. Try again!");
            }
        });
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const uploadBtn = document.getElementById("uploadBtn");
    if (uploadBtn) {
        uploadBtn.addEventListener("click", (e) => {
            e.preventDefault();
            alert("Image Uploaded Successfully! Redirecting to Specification page...");
            window.location.href = "specification.html";  // redirect
        });
    }
});