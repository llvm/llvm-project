#include <sys/auxv.h>

// If this program receives 0 arguments, it will use non-streaming SVE
// registers. If the number of arguments is >= 1, it will use streaming SVE
// registers.

#ifndef HWCAP2_SME
#define HWCAP2_SME (1 << 23)
#endif

void set_sve_registers() {
  // AArch64 SVE extension ISA adds a new set of vector and predicate registers:
  // 32 Z registers, 16 P registers, and 1 FFR register.
  // Code below populates SVE registers to be read back by the debugger via
  // ptrace interface at runtime.
  // The P registers are predicate registers and hold one bit for each byte
  // available in a Z vector register. For example, an SVE implementation with
  // 1024-bit Z registers has 128-bit predicate registers.
  // ptrue/pfalse instruction is used to set a predicate lane with a pattern.
  // pattern is decided based on size specifier, b, h, s and d. if size
  // specified is b all lanes will be set to 1. which is needed to set all bytes
  // in a Z registers to the specified value.
  asm volatile("setffr\n\t");
  asm volatile("ptrue p0.b\n\t");
  asm volatile("ptrue p1.h\n\t");
  asm volatile("ptrue p2.s\n\t");
  asm volatile("ptrue p3.d\n\t");
  asm volatile("pfalse p4.b\n\t");
  asm volatile("ptrue p5.b\n\t");
  asm volatile("ptrue p6.h\n\t");
  asm volatile("ptrue p7.s\n\t");
  asm volatile("ptrue p8.d\n\t");
  asm volatile("pfalse p9.b\n\t");
  asm volatile("ptrue p10.b\n\t");
  asm volatile("ptrue p11.h\n\t");
  asm volatile("ptrue p12.s\n\t");
  asm volatile("ptrue p13.d\n\t");
  asm volatile("pfalse p14.b\n\t");
  asm volatile("ptrue p15.b\n\t");

  asm volatile("cpy  z0.b, p0/z, #1\n\t");
  asm volatile("cpy  z1.b, p5/z, #2\n\t");
  asm volatile("cpy  z2.b, p10/z, #3\n\t");
  asm volatile("cpy  z3.b, p15/z, #4\n\t");
  asm volatile("cpy  z4.b, p0/z, #5\n\t");
  asm volatile("cpy  z5.b, p5/z, #6\n\t");
  asm volatile("cpy  z6.b, p10/z, #7\n\t");
  asm volatile("cpy  z7.b, p15/z, #8\n\t");
  asm volatile("cpy  z8.b, p0/z, #9\n\t");
  asm volatile("cpy  z9.b, p5/z, #10\n\t");
  asm volatile("cpy  z10.b, p10/z, #11\n\t");
  asm volatile("cpy  z11.b, p15/z, #12\n\t");
  asm volatile("cpy  z12.b, p0/z, #13\n\t");
  asm volatile("cpy  z13.b, p5/z, #14\n\t");
  asm volatile("cpy  z14.b, p10/z, #15\n\t");
  asm volatile("cpy  z15.b, p15/z, #16\n\t");
  asm volatile("cpy  z16.b, p0/z, #17\n\t");
  asm volatile("cpy  z17.b, p5/z, #18\n\t");
  asm volatile("cpy  z18.b, p10/z, #19\n\t");
  asm volatile("cpy  z19.b, p15/z, #20\n\t");
  asm volatile("cpy  z20.b, p0/z, #21\n\t");
  asm volatile("cpy  z21.b, p5/z, #22\n\t");
  asm volatile("cpy  z22.b, p10/z, #23\n\t");
  asm volatile("cpy  z23.b, p15/z, #24\n\t");
  asm volatile("cpy  z24.b, p0/z, #25\n\t");
  asm volatile("cpy  z25.b, p5/z, #26\n\t");
  asm volatile("cpy  z26.b, p10/z, #27\n\t");
  asm volatile("cpy  z27.b, p15/z, #28\n\t");
  asm volatile("cpy  z28.b, p0/z, #29\n\t");
  asm volatile("cpy  z29.b, p5/z, #30\n\t");
  asm volatile("cpy  z30.b, p10/z, #31\n\t");
  asm volatile("cpy  z31.b, p15/z, #32\n\t");
}

int main(int argc, char *argv[]) {
  if (argc > 1) {
    // Enable streaming mode SVE and the ZA array storage.
    asm volatile("msr  s0_3_c4_c7_3, xzr" /*smstart*/);
  }

  // If we have SVE or SME, set the SVE registers.
  if ((getauxval(AT_HWCAP) & HWCAP_SVE) || (getauxval(AT_HWCAP2) & HWCAP2_SME))
    set_sve_registers();

  return 0; // Set a break point here.
}
