#include <cstdint>

struct alignas(16) float80_raw {
  uint64_t mantissa;
  uint16_t sign_exp;
};

int main() {
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

  asm volatile(
    "finit\n\t"
    "fldcw %1\n\t"
    // load on stack in reverse order to make the result easier to read
    "fldt 0x70(%0)\n\t"
    "fldt 0x60(%0)\n\t"
    "fldt 0x50(%0)\n\t"
    "fldt 0x40(%0)\n\t"
    "fldt 0x30(%0)\n\t"
    "fldt 0x20(%0)\n\t"
    "fldt 0x10(%0)\n\t"
    "fldt 0x00(%0)\n\t"
    // free st7
    "ffree %%st(7)\n\t"
    // this should trigger a divide-by-zero
    "fdivs (%2)\n\t"
    "int3\n\t"
    :
    : "a"(st), "m"(cw), "b"(&zero)
    : "st"
  );

  return 0;
}
