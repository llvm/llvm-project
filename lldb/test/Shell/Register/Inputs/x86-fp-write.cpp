#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

struct alignas(16) float80_raw {
  uint8_t data[10];
};

int main() {
  float80_raw st[8];
  uint16_t env[14];
  union alignas(16) {
    uint16_t i16[256];
    uint32_t i32[128];
    uint64_t i64[64];
  } fxsave;

  asm volatile(
    "finit\n\t"
    "int3\n\t"
#if defined(__x86_64__)
    "fxsave64 %2\n\t"
#else
    "fxsave %2\n\t"
#endif
    "fnstenv %1\n\t"
    "fnclex\n\t"
    "fstpt 0x00(%0)\n\t"
    "fstpt 0x10(%0)\n\t"
    "fstpt 0x20(%0)\n\t"
    "fstpt 0x30(%0)\n\t"
    "fstpt 0x40(%0)\n\t"
    "fstpt 0x50(%0)\n\t"
    "fstpt 0x60(%0)\n\t"
    "fstpt 0x70(%0)\n\t"
    :
    : "a"(st), "m"(env), "m"(fxsave)
    : "st"
  );

  assert(env[0] == fxsave.i16[0]);
  assert(env[2] == fxsave.i16[1]);

  printf("fctrl = 0x%04" PRIx16 "\n", env[0]);
  printf("fstat = 0x%04" PRIx16 "\n", env[2]);
  printf("ftag = 0x%04" PRIx16 "\n", env[4]);
  printf("fop = 0x%04" PRIx16 "\n", fxsave.i16[3]);
#if defined(__x86_64__)
  printf("fip = 0x%016" PRIx64 "\n", fxsave.i64[1]);
  printf("fdp = 0x%016" PRIx64 "\n", fxsave.i64[2]);
#else
  printf("fip = 0x%08" PRIx32 "\n", fxsave.i32[2]);
  printf("fcs = 0x%04" PRIx16 "\n", fxsave.i16[6]);
  printf("fdp = 0x%08" PRIx32 "\n", fxsave.i32[4]);
  printf("fds = 0x%04" PRIx16 "\n", fxsave.i16[10]);
#endif
  printf("mxcsr = 0x%08" PRIx32 "\n", fxsave.i32[6]);
  printf("mxcsr_mask = 0x%08" PRIx32 "\n", fxsave.i32[7]);

  for (int i = 0; i < 8; ++i) {
    printf("st%d = { ", i);
    for (int j = 0; j < sizeof(st->data); ++j)
      printf("0x%02" PRIx8 " ", st[i].data[j]);
    printf("}\n");
  }

  return 0;
}
