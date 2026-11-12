#include <cinttypes>
#include <cstdint>
#include <cstdio>

union alignas(32) ymm_t {
  uint64_t as_uint64[4];
  uint8_t as_uint8[32];
};

int main() {
  constexpr ymm_t ymm_fill = {
    .as_uint64 = { 0, 0, 0, 0 }
  };

  ymm_t ymm[16];

  asm volatile(
    "vmovaps  %1, %%ymm0\n\t"
    "vmovaps  %1, %%ymm1\n\t"
    "vmovaps  %1, %%ymm2\n\t"
    "vmovaps  %1, %%ymm3\n\t"
    "vmovaps  %1, %%ymm4\n\t"
    "vmovaps  %1, %%ymm5\n\t"
    "vmovaps  %1, %%ymm6\n\t"
    "vmovaps  %1, %%ymm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  %1, %%ymm8\n\t"
    "vmovaps  %1, %%ymm9\n\t"
    "vmovaps  %1, %%ymm10\n\t"
    "vmovaps  %1, %%ymm11\n\t"
    "vmovaps  %1, %%ymm12\n\t"
    "vmovaps  %1, %%ymm13\n\t"
    "vmovaps  %1, %%ymm14\n\t"
    "vmovaps  %1, %%ymm15\n\t"
#endif
    "\n\t"
    "int3\n\t"
    "\n\t"
    "vmovaps %%ymm0,  0x000(%0)\n\t"
    "vmovaps %%ymm1,  0x020(%0)\n\t"
    "vmovaps %%ymm2,  0x040(%0)\n\t"
    "vmovaps %%ymm3,  0x060(%0)\n\t"
    "vmovaps %%ymm4,  0x080(%0)\n\t"
    "vmovaps %%ymm5,  0x0A0(%0)\n\t"
    "vmovaps %%ymm6,  0x0C0(%0)\n\t"
    "vmovaps %%ymm7,  0x0E0(%0)\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps %%ymm8,  0x100(%0)\n\t"
    "vmovaps %%ymm9,  0x120(%0)\n\t"
    "vmovaps %%ymm10, 0x140(%0)\n\t"
    "vmovaps %%ymm11, 0x160(%0)\n\t"
    "vmovaps %%ymm12, 0x180(%0)\n\t"
    "vmovaps %%ymm13, 0x1A0(%0)\n\t"
    "vmovaps %%ymm14, 0x1C0(%0)\n\t"
    "vmovaps %%ymm15, 0x1E0(%0)\n\t"
#endif
    :
    : "b"(ymm), "m"(ymm_fill)
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7"
#if defined(__x86_64__) || defined(_M_X64)
    , "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
      "%ymm15"
#endif
  );

  for (int i = 0; i < 16; ++i) {
    printf("ymm%d = { ", i);
    for (int j = 0; j < sizeof(ymm->as_uint8); ++j)
      printf("0x%02x ", ymm[i].as_uint8[j]);
    printf("}\n");
  }

  return 0;
}
