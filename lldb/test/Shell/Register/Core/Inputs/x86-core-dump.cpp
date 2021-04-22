// This program is used to generate a core dump for testing register dumps.
// The exact set of registers dumped depends on the instruction sets enabled
// via compiler flags.

#include <cstdint>

struct alignas(64) zmm_t {
  uint64_t a, b, c, d, e, f, g, h;
};

struct alignas(16) float80_raw {
  uint64_t mantissa;
  uint16_t sign_exp;
};

int main() {
  // test data for xmm, ymm and zmm registers
  constexpr zmm_t zmm[] = {
    { 0x0706050403020100, 0x0F0E0D0C0B0A0908,
      0x1716151413121110, 0x1F1E1D1C1B1A1918,
      0x2726252423222120, 0x2F2E2D2C2B2A2928,
      0x3736353433323130, 0x3F3E3D3C3B3A3938, },
    { 0x0807060504030201, 0x100F0E0D0C0B0A09,
      0x1817161514131211, 0x201F1E1D1C1B1A19,
      0x2827262524232221, 0x302F2E2D2C2B2A29,
      0x3837363534333231, 0x403F3E3D3C3B3A39, },
    { 0x0908070605040302, 0x11100F0E0D0C0B0A,
      0x1918171615141312, 0x21201F1E1D1C1B1A,
      0x2928272625242322, 0x31302F2E2D2C2B2A,
      0x3938373635343332, 0x41403F3E3D3C3B3A, },
    { 0x0A09080706050403, 0x1211100F0E0D0C0B,
      0x1A19181716151413, 0x2221201F1E1D1C1B,
      0x2A29282726252423, 0x3231302F2E2D2C2B,
      0x3A39383736353433, 0x4241403F3E3D3C3B, },
    { 0x0B0A090807060504, 0x131211100F0E0D0C,
      0x1B1A191817161514, 0x232221201F1E1D1C,
      0x2B2A292827262524, 0x333231302F2E2D2C,
      0x3B3A393837363534, 0x434241403F3E3D3C, },
    { 0x0C0B0A0908070605, 0x14131211100F0E0D,
      0x1C1B1A1918171615, 0x24232221201F1E1D,
      0x2C2B2A2928272625, 0x34333231302F2E2D,
      0x3C3B3A3938373635, 0x44434241403F3E3D, },
    { 0x0D0C0B0A09080706, 0x1514131211100F0E,
      0x1D1C1B1A19181716, 0x2524232221201F1E,
      0x2D2C2B2A29282726, 0x3534333231302F2E,
      0x3D3C3B3A39383736, 0x4544434241403F3E, },
    { 0x0E0D0C0B0A090807, 0x161514131211100F,
      0x1E1D1C1B1A191817, 0x262524232221201F,
      0x2E2D2C2B2A292827, 0x363534333231302F,
      0x3E3D3C3B3A393837, 0x464544434241403F, },
#if defined(__x86_64__) || defined(_M_X64)
    { 0x0F0E0D0C0B0A0908, 0x1716151413121110,
      0x1F1E1D1C1B1A1918, 0x2726252423222120,
      0x2F2E2D2C2B2A2928, 0x3736353433323130,
      0x3F3E3D3C3B3A3938, 0x4746454443424140, },
    { 0x100F0E0D0C0B0A09, 0x1817161514131211,
      0x201F1E1D1C1B1A19, 0x2827262524232221,
      0x302F2E2D2C2B2A29, 0x3837363534333231,
      0x403F3E3D3C3B3A39, 0x4847464544434241, },
    { 0x11100F0E0D0C0B0A, 0x1918171615141312,
      0x21201F1E1D1C1B1A, 0x2928272625242322,
      0x31302F2E2D2C2B2A, 0x3938373635343332,
      0x41403F3E3D3C3B3A, 0x4948474645444342, },
    { 0x1211100F0E0D0C0B, 0x1A19181716151413,
      0x2221201F1E1D1C1B, 0x2A29282726252423,
      0x3231302F2E2D2C2B, 0x3A39383736353433,
      0x4241403F3E3D3C3B, 0x4A49484746454443, },
    { 0x131211100F0E0D0C, 0x1B1A191817161514,
      0x232221201F1E1D1C, 0x2B2A292827262524,
      0x333231302F2E2D2C, 0x3B3A393837363534,
      0x434241403F3E3D3C, 0x4B4A494847464544, },
    { 0x14131211100F0E0D, 0x1C1B1A1918171615,
      0x24232221201F1E1D, 0x2C2B2A2928272625,
      0x34333231302F2E2D, 0x3C3B3A3938373635,
      0x44434241403F3E3D, 0x4C4B4A4948474645, },
    { 0x1514131211100F0E, 0x1D1C1B1A19181716,
      0x2524232221201F1E, 0x2D2C2B2A29282726,
      0x3534333231302F2E, 0x3D3C3B3A39383736,
      0x4544434241403F3E, 0x4D4C4B4A49484746, },
    { 0x161514131211100F, 0x1E1D1C1B1A191817,
      0x262524232221201F, 0x2E2D2C2B2A292827,
      0x363534333231302F, 0x3E3D3C3B3A393837,
      0x464544434241403F, 0x4E4D4C4B4A494847, },
    { 0x1716151413121110, 0x1F1E1D1C1B1A1918,
      0x2726252423222120, 0x2F2E2D2C2B2A2928,
      0x3736353433323130, 0x3F3E3D3C3B3A3938,
      0x4746454443424140, 0x4F4E4D4C4B4A4948, },
    { 0x1817161514131211, 0x201F1E1D1C1B1A19,
      0x2827262524232221, 0x302F2E2D2C2B2A29,
      0x3837363534333231, 0x403F3E3D3C3B3A39,
      0x4847464544434241, 0x504F4E4D4C4B4A49, },
    { 0x1918171615141312, 0x21201F1E1D1C1B1A,
      0x2928272625242322, 0x31302F2E2D2C2B2A,
      0x3938373635343332, 0x41403F3E3D3C3B3A,
      0x4948474645444342, 0x51504F4E4D4C4B4A, },
    { 0x1A19181716151413, 0x2221201F1E1D1C1B,
      0x2A29282726252423, 0x3231302F2E2D2C2B,
      0x3A39383736353433, 0x4241403F3E3D3C3B,
      0x4A49484746454443, 0x5251504F4E4D4C4B, },
    { 0x1B1A191817161514, 0x232221201F1E1D1C,
      0x2B2A292827262524, 0x333231302F2E2D2C,
      0x3B3A393837363534, 0x434241403F3E3D3C,
      0x4B4A494847464544, 0x535251504F4E4D4C, },
    { 0x1C1B1A1918171615, 0x24232221201F1E1D,
      0x2C2B2A2928272625, 0x34333231302F2E2D,
      0x3C3B3A3938373635, 0x44434241403F3E3D,
      0x4C4B4A4948474645, 0x54535251504F4E4D, },
    { 0x1D1C1B1A19181716, 0x2524232221201F1E,
      0x2D2C2B2A29282726, 0x3534333231302F2E,
      0x3D3C3B3A39383736, 0x4544434241403F3E,
      0x4D4C4B4A49484746, 0x5554535251504F4E, },
    { 0x1E1D1C1B1A191817, 0x262524232221201F,
      0x2E2D2C2B2A292827, 0x363534333231302F,
      0x3E3D3C3B3A393837, 0x464544434241403F,
      0x4E4D4C4B4A494847, 0x565554535251504F, },
    { 0x1F1E1D1C1B1A1918, 0x2726252423222120,
      0x2F2E2D2C2B2A2928, 0x3736353433323130,
      0x3F3E3D3C3B3A3938, 0x4746454443424140,
      0x4F4E4D4C4B4A4948, 0x5756555453525150, },
    { 0x201F1E1D1C1B1A19, 0x2827262524232221,
      0x302F2E2D2C2B2A29, 0x3837363534333231,
      0x403F3E3D3C3B3A39, 0x4847464544434241,
      0x504F4E4D4C4B4A49, 0x5857565554535251, },
    { 0x21201F1E1D1C1B1A, 0x2928272625242322,
      0x31302F2E2D2C2B2A, 0x3938373635343332,
      0x41403F3E3D3C3B3A, 0x4948474645444342,
      0x51504F4E4D4C4B4A, 0x5958575655545352, },
    { 0x2221201F1E1D1C1B, 0x2A29282726252423,
      0x3231302F2E2D2C2B, 0x3A39383736353433,
      0x4241403F3E3D3C3B, 0x4A49484746454443,
      0x5251504F4E4D4C4B, 0x5A59585756555453, },
    { 0x232221201F1E1D1C, 0x2B2A292827262524,
      0x333231302F2E2D2C, 0x3B3A393837363534,
      0x434241403F3E3D3C, 0x4B4A494847464544,
      0x535251504F4E4D4C, 0x5B5A595857565554, },
    { 0x24232221201F1E1D, 0x2C2B2A2928272625,
      0x34333231302F2E2D, 0x3C3B3A3938373635,
      0x44434241403F3E3D, 0x4C4B4A4948474645,
      0x54535251504F4E4D, 0x5C5B5A5958575655, },
    { 0x2524232221201F1E, 0x2D2C2B2A29282726,
      0x3534333231302F2E, 0x3D3C3B3A39383736,
      0x4544434241403F3E, 0x4D4C4B4A49484746,
      0x5554535251504F4E, 0x5D5C5B5A59585756, },
    { 0x262524232221201F, 0x2E2D2C2B2A292827,
      0x363534333231302F, 0x3E3D3C3B3A393837,
      0x464544434241403F, 0x4E4D4C4B4A494847,
      0x565554535251504F, 0x5E5D5C5B5A595857, },
#endif
  };

  // test data for FPU registers
  float80_raw st[] = {
    {0x8000000000000000, 0x4000},  // +2.0
    {0x3f00000000000000, 0x0000},  // 1.654785e-4932 (denormal)
    {0x0000000000000000, 0x0000},  // +0
    {0x0000000000000000, 0x8000},  // -0
    {0x8000000000000000, 0x7fff},  // +inf
    {0x8000000000000000, 0xffff},  // -inf
    {0xc000000000000000, 0xffff},  // nan
    // st7 will be freed to test tag word better
    {0x0000000000000000, 0x0000},  // +0
  };

  // unmask divide-by-zero exception
  uint16_t cw = 0x037b;
  // used as single-precision float
  uint32_t zero = 0;

  // test data for GP registers
  const uint64_t gpr[] = {
    0x2726252423222120,
    0x2827262524232221,
    0x2928272625242322,
    0x2A29282726252423,
    0x2B2A292827262524,
    0x2C2B2A2928272625,
    0x2D2C2B2A29282726,
    0x2E2D2C2B2A292827,
    0x2F2E2D2C2B2A2928,
    0x302F2E2D2C2B2A29,
    0x31302F2E2D2C2B2A,
    0x3231302F2E2D2C2B,
    0x333231302F2E2D2C,
    0x34333231302F2E2D,
    0x3534333231302F2E,
    0x363534333231302F,
  };

  asm volatile(
    // fill the highest register set supported -- ZMM, YMM or XMM
#if defined(__AVX512F__)
    "vmovaps  0x000(%0), %%zmm0\n\t"
    "vmovaps  0x040(%0), %%zmm1\n\t"
    "vmovaps  0x080(%0), %%zmm2\n\t"
    "vmovaps  0x0C0(%0), %%zmm3\n\t"
    "vmovaps  0x100(%0), %%zmm4\n\t"
    "vmovaps  0x140(%0), %%zmm5\n\t"
    "vmovaps  0x180(%0), %%zmm6\n\t"
    "vmovaps  0x1C0(%0), %%zmm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  0x200(%0), %%zmm8\n\t"
    "vmovaps  0x240(%0), %%zmm9\n\t"
    "vmovaps  0x280(%0), %%zmm10\n\t"
    "vmovaps  0x2C0(%0), %%zmm11\n\t"
    "vmovaps  0x300(%0), %%zmm12\n\t"
    "vmovaps  0x340(%0), %%zmm13\n\t"
    "vmovaps  0x380(%0), %%zmm14\n\t"
    "vmovaps  0x3C0(%0), %%zmm15\n\t"
    "vmovaps  0x400(%0), %%zmm16\n\t"
    "vmovaps  0x440(%0), %%zmm17\n\t"
    "vmovaps  0x480(%0), %%zmm18\n\t"
    "vmovaps  0x4C0(%0), %%zmm19\n\t"
    "vmovaps  0x500(%0), %%zmm20\n\t"
    "vmovaps  0x540(%0), %%zmm21\n\t"
    "vmovaps  0x580(%0), %%zmm22\n\t"
    "vmovaps  0x5C0(%0), %%zmm23\n\t"
    "vmovaps  0x600(%0), %%zmm24\n\t"
    "vmovaps  0x640(%0), %%zmm25\n\t"
    "vmovaps  0x680(%0), %%zmm26\n\t"
    "vmovaps  0x6C0(%0), %%zmm27\n\t"
    "vmovaps  0x700(%0), %%zmm28\n\t"
    "vmovaps  0x740(%0), %%zmm29\n\t"
    "vmovaps  0x780(%0), %%zmm30\n\t"
    "vmovaps  0x7C0(%0), %%zmm31\n\t"
#endif // defined(__x86_64__) || defined(_M_X64)
#elif defined(__AVX__)
    "vmovaps  0x000(%0), %%ymm0\n\t"
    "vmovaps  0x040(%0), %%ymm1\n\t"
    "vmovaps  0x080(%0), %%ymm2\n\t"
    "vmovaps  0x0C0(%0), %%ymm3\n\t"
    "vmovaps  0x100(%0), %%ymm4\n\t"
    "vmovaps  0x140(%0), %%ymm5\n\t"
    "vmovaps  0x180(%0), %%ymm6\n\t"
    "vmovaps  0x1C0(%0), %%ymm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  0x200(%0), %%ymm8\n\t"
    "vmovaps  0x240(%0), %%ymm9\n\t"
    "vmovaps  0x280(%0), %%ymm10\n\t"
    "vmovaps  0x2C0(%0), %%ymm11\n\t"
    "vmovaps  0x300(%0), %%ymm12\n\t"
    "vmovaps  0x340(%0), %%ymm13\n\t"
    "vmovaps  0x380(%0), %%ymm14\n\t"
    "vmovaps  0x3C0(%0), %%ymm15\n\t"
#endif // defined(__x86_64__) || defined(_M_X64)
#else // !defined(__AVX__)
    "movaps   0x000(%0), %%xmm0\n\t"
    "movaps   0x040(%0), %%xmm1\n\t"
    "movaps   0x080(%0), %%xmm2\n\t"
    "movaps   0x0C0(%0), %%xmm3\n\t"
    "movaps   0x100(%0), %%xmm4\n\t"
    "movaps   0x140(%0), %%xmm5\n\t"
    "movaps   0x180(%0), %%xmm6\n\t"
    "movaps   0x1C0(%0), %%xmm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "movaps   0x200(%0), %%xmm8\n\t"
    "movaps   0x240(%0), %%xmm9\n\t"
    "movaps   0x280(%0), %%xmm10\n\t"
    "movaps   0x2C0(%0), %%xmm11\n\t"
    "movaps   0x300(%0), %%xmm12\n\t"
    "movaps   0x340(%0), %%xmm13\n\t"
    "movaps   0x380(%0), %%xmm14\n\t"
    "movaps   0x3C0(%0), %%xmm15\n\t"
#endif // defined(__x86_64__) || defined(_M_X64)
#endif
    "\n\t"

    // fill FPU registers
    "finit\n\t"
    "fldcw %2\n\t"
    // load on stack in reverse order to make the result easier to read
    "fldt 0x70(%1)\n\t"
    "fldt 0x60(%1)\n\t"
    "fldt 0x50(%1)\n\t"
    "fldt 0x40(%1)\n\t"
    "fldt 0x30(%1)\n\t"
    "fldt 0x20(%1)\n\t"
    "fldt 0x10(%1)\n\t"
    "fldt 0x00(%1)\n\t"
    // free st7
    "ffree %%st(7)\n\t"
    // this should trigger a divide-by-zero
    "fdivs (%3)\n\t"
    "\n\t"

    // fill GP registers
    // note that this invalidates all parameters
#if defined(__x86_64__) || defined(_M_X64)
    "movq 0x78(%4), %%r15\n\t"
    "movq 0x70(%4), %%r14\n\t"
    "movq 0x68(%4), %%r13\n\t"
    "movq 0x60(%4), %%r12\n\t"
    "movq 0x58(%4), %%r11\n\t"
    "movq 0x50(%4), %%r10\n\t"
    "movq 0x48(%4), %%r9\n\t"
    "movq 0x40(%4), %%r8\n\t"
    "movq 0x38(%4), %%rdi\n\t"
    "movq 0x30(%4), %%rsi\n\t"
    "movq 0x28(%4), %%rbp\n\t"
    "movq 0x20(%4), %%rsp\n\t"
    "movq 0x18(%4), %%rdx\n\t"
    "movq 0x10(%4), %%rcx\n\t"
    "movq 0x08(%4), %%rbx\n\t"
    "movq 0x00(%4), %%rax\n\t"
#else // !(defined(__x86_64__) || defined(_M_X64))
    "movl 0x38(%4), %%edi\n\t"
    "movl 0x30(%4), %%esi\n\t"
    "movl 0x28(%4), %%ebp\n\t"
    "movl 0x20(%4), %%esp\n\t"
    "movl 0x18(%4), %%edx\n\t"
    "movl 0x10(%4), %%ecx\n\t"
    "movl 0x08(%4), %%ebx\n\t"
    "movl 0x00(%4), %%eax\n\t"
#endif
    "\n\t"

    // trigger SEGV
    "movl %%eax, 0\n\t"
    :
    : "b"(zmm), "c"(st), "m"(cw), "d"(&zero), "a"(gpr)
    : // clobbers do not really matter since we crash
  );

  return 0;
}
