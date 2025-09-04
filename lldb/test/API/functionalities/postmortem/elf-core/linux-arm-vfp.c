// commandline: -march=armv7+fp -nostdlib -static -Wl,--build-id=none linux-arm-vfp.c

static void foo(char *boom) {
  asm volatile(R"(
    vmov.f64  d0,  #0.5
    vmov.f64  d1,  #1.5
    vmov.f64  d2,  #2.5
    vmov.f64  d3,  #3.5
    vmov.f32  s8,  #4.5
    vmov.f32  s9,  #5.5
    vmov.f32  s10, #6.5
    vmov.f32  s11, #7.5
    vcmp.f32  s9,  s8
  )");

  *boom = 47;
}

void _start(void) {
  foo(0);
}
