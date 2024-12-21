///  BUILT with
///    xcrun -sdk macosx.internal clang -mcpu=apple-m4 -g sme.c -o sme

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void write_sve_regs() {
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

#define MAX_VL_BYTES 256
void set_za_register(int svl, int value_offset) {
  uint8_t data[MAX_VL_BYTES];

  // ldr za will actually wrap the selected vector row, by the number of rows
  // you have. So setting one that didn't exist would actually set one that did.
  // That's why we need the streaming vector length here.
  for (int i = 0; i < svl; ++i) {
    // This may involve instructions that require the smefa64 extension.
    for (int j = 0; j < MAX_VL_BYTES; j++)
      data[j] = i + value_offset;
    // Each one of these loads a VL sized row of ZA.
    asm volatile("mov w12, %w0\n\t"
                 "ldr za[w12, 0], [%1]\n\t" ::"r"(i),
                 "r"(&data)
                 : "w12");
  }
}

static uint16_t arm_sme_svl_b(void) {
  uint64_t ret = 0;
  asm volatile("rdsvl  %[ret], #1" : [ret] "=r"(ret));
  return (uint16_t)ret;
}

void arm_sme2_set_zt0() {
#define ZTO_LEN (512 / 8)
  uint8_t data[ZTO_LEN];
  for (unsigned i = 0; i < ZTO_LEN; ++i)
    data[i] = i + 0;

  asm volatile("ldr zt0, [%0]" ::"r"(&data));
#undef ZT0_LEN
}

int main() {
  printf("Enable SME mode\n"); // break before sme

  asm volatile("smstart");

  write_sve_regs();

  set_za_register(arm_sme_svl_b(), 4);

  arm_sme2_set_zt0();

  int c = 10; // break while sme
  c += 5;
  c += 5;

  asm volatile("smstop");

  printf("SME mode disabled\n"); // break after sme
}
