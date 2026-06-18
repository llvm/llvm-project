// linux-arm-vfp.core was generated with:
// > gcc-12 -march=armv7+fp -nostdlib -static -Wl,--build-id=none \
//   linux-arm-vfp.c -o linux-arm-vfp.out
// > ulimit -c 1000
// > ulimit -s 8
// > env -i ./linux-arm-vfp.out

static void foo(char *boom) {
  asm volatile(R"(
    vmov.f64  d0,  #0.5
    vmov.f64  d1,  #1.5
    vmov.f64  d14, #14.5
    vmov.f64  d15, #15.5
    vmov.f32  s4,  #4.5
    vmov.f32  s5,  #5.5
    vmov.f32  s6,  #6.5
    vmov.f32  s7,  #7.5
    vcmp.f32  s7,  s6
  )");

  *boom = 47;
}

void _start(void) { foo(0); }
