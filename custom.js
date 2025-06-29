AOS.init();
var swiper = new Swiper("#testimonails-slider", {
  effect: "slide",
  loop: false,
  slidesPerView: 1,
  speed: 900,
  spaceBetween: 15,
  // autoHeight: true,
  //   pagination: {
  //     el: ".swiper-pagination",
  //     type: "progressbar",
  //   },
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
  breakpoints: {
    768: {
      slidesPerView: 1,
      spaceBetween: 30,
    },
  },
});
$(window).scroll(function () {
  if ($(this).scrollTop() > 50) {
    $(".main-header").addClass("scroll");
  } else {
    $(".main-header").removeClass("scroll");
  }
});
