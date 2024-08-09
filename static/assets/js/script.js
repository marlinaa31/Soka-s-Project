// Ambil semua elemen dengan class "dropdown-toggle"
var dropdownToggleList = document.querySelectorAll(".dropdown-toggle");

// Tambahkan event listener untuk setiap elemen
dropdownToggleList.forEach(function(dropdownToggle) {
    dropdownToggle.addEventListener("click", function(event) {
        // Hentikan perilaku default (menghilangkan dropdown)
        event.preventDefault();
        
        // Temukan dropdown-menu terkait
        var dropdownMenu = this.nextElementSibling;
        
        // Toggle class "show" pada dropdown-menu untuk menampilkan/sembunyikannya
        dropdownMenu.classList.toggle("show");
    });
});

// Menutup dropdown saat mengklik di luar dropdown
document.addEventListener("click", function(event) {
    if (!event.target.matches('.dropdown-toggle')) {
        var dropdowns = document.querySelectorAll(".dropdown-menu");
        dropdowns.forEach(function(dropdown) {
            if (dropdown.classList.contains("show")) {
                dropdown.classList.remove("show");
            }
        });
    }
});
