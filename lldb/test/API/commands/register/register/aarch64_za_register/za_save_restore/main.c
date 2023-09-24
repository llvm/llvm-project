#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>

// Important details for this program:
// * Making a syscall will disable streaming mode if it is active.
// * Changing the vector length will make streaming mode and ZA inactive.
// * ZA can be active independent of streaming mode.
// * ZA's size is the streaming vector length squared.

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif

#ifndef PR_SME_VL_LEN_MASK
#define PR_SME_VL_LEN_MASK 0xffff
#endif

#define SM_INST(c) asm volatile("msr s0_3_c4_c" #c "_3, xzr")
#define SMSTART SM_INST(7)
#define SMSTART_SM SM_INST(3)
#define SMSTART_ZA SM_INST(5)
#define SMSTOP SM_INST(6)
#define SMSTOP_SM SM_INST(2)
#define SMSTOP_ZA SM_INST(4)

int start_vl = 0;
int other_vl = 0;

void write_sve_regs() {
  // We assume the smefa64 feature is present, which allows ffr access
  // in streaming mode.
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

// Write something different so we will know if we didn't restore them
// correctly.
void write_sve_regs_expr() {
  asm volatile("pfalse p0.b\n\t");
  asm volatile("wrffr p0.b\n\t");
  asm volatile("pfalse p1.b\n\t");
  asm volatile("pfalse p2.b\n\t");
  asm volatile("pfalse p3.b\n\t");
  asm volatile("ptrue p4.b\n\t");
  asm volatile("pfalse p5.b\n\t");
  asm volatile("pfalse p6.b\n\t");
  asm volatile("pfalse p7.b\n\t");
  asm volatile("pfalse p8.b\n\t");
  asm volatile("ptrue p9.b\n\t");
  asm volatile("pfalse p10.b\n\t");
  asm volatile("pfalse p11.b\n\t");
  asm volatile("pfalse p12.b\n\t");
  asm volatile("pfalse p13.b\n\t");
  asm volatile("ptrue p14.b\n\t");
  asm volatile("pfalse p15.b\n\t");

  asm volatile("cpy  z0.b, p0/z, #2\n\t");
  asm volatile("cpy  z1.b, p5/z, #3\n\t");
  asm volatile("cpy  z2.b, p10/z, #4\n\t");
  asm volatile("cpy  z3.b, p15/z, #5\n\t");
  asm volatile("cpy  z4.b, p0/z, #6\n\t");
  asm volatile("cpy  z5.b, p5/z, #7\n\t");
  asm volatile("cpy  z6.b, p10/z, #8\n\t");
  asm volatile("cpy  z7.b, p15/z, #9\n\t");
  asm volatile("cpy  z8.b, p0/z, #10\n\t");
  asm volatile("cpy  z9.b, p5/z, #11\n\t");
  asm volatile("cpy  z10.b, p10/z, #12\n\t");
  asm volatile("cpy  z11.b, p15/z, #13\n\t");
  asm volatile("cpy  z12.b, p0/z, #14\n\t");
  asm volatile("cpy  z13.b, p5/z, #15\n\t");
  asm volatile("cpy  z14.b, p10/z, #16\n\t");
  asm volatile("cpy  z15.b, p15/z, #17\n\t");
  asm volatile("cpy  z16.b, p0/z, #18\n\t");
  asm volatile("cpy  z17.b, p5/z, #19\n\t");
  asm volatile("cpy  z18.b, p10/z, #20\n\t");
  asm volatile("cpy  z19.b, p15/z, #21\n\t");
  asm volatile("cpy  z20.b, p0/z, #22\n\t");
  asm volatile("cpy  z21.b, p5/z, #23\n\t");
  asm volatile("cpy  z22.b, p10/z, #24\n\t");
  asm volatile("cpy  z23.b, p15/z, #25\n\t");
  asm volatile("cpy  z24.b, p0/z, #26\n\t");
  asm volatile("cpy  z25.b, p5/z, #27\n\t");
  asm volatile("cpy  z26.b, p10/z, #28\n\t");
  asm volatile("cpy  z27.b, p15/z, #29\n\t");
  asm volatile("cpy  z28.b, p0/z, #30\n\t");
  asm volatile("cpy  z29.b, p5/z, #31\n\t");
  asm volatile("cpy  z30.b, p10/z, #32\n\t");
  asm volatile("cpy  z31.b, p15/z, #33\n\t");
}

void set_za_register(int svl, int value_offset) {
#define MAX_VL_BYTES 256
  uint8_t data[MAX_VL_BYTES];

  // ldr za will actually wrap the selected vector row, by the number of rows
  // you have. So setting one that didn't exist would actually set one that did.
  // That's why we need the streaming vector length here.
  for (int i = 0; i < svl; ++i) {
    memset(data, i + value_offset, MAX_VL_BYTES);
    // Each one of these loads a VL sized row of ZA.
    asm volatile("mov w12, %w0\n\t"
                 "ldr za[w12, 0], [%1]\n\t" ::"r"(i),
                 "r"(&data)
                 : "w12");
  }
}

void expr_disable_za() {
  SMSTOP_ZA;
  write_sve_regs_expr();
}

void expr_enable_za() {
  SMSTART_ZA;
  set_za_register(start_vl, 2);
  write_sve_regs_expr();
}

void expr_start_vl() {
  prctl(PR_SME_SET_VL, start_vl);
  SMSTART_ZA;
  set_za_register(start_vl, 4);
  write_sve_regs_expr();
}

void expr_other_vl() {
  prctl(PR_SME_SET_VL, other_vl);
  SMSTART_ZA;
  set_za_register(other_vl, 5);
  write_sve_regs_expr();
}

void expr_enable_sm() {
  SMSTART_SM;
  write_sve_regs_expr();
}

void expr_disable_sm() {
  SMSTOP_SM;
  write_sve_regs_expr();
}

int main(int argc, char *argv[]) {
  // We expect to get:
  // * whether to enable streaming mode
  // * whether to enable ZA
  // * what the starting VL should be
  // * what the other VL should be
  if (argc != 5)
    return 1;

  bool ssve = argv[1][0] == '1';
  bool za = argv[2][0] == '1';
  start_vl = atoi(argv[3]);
  other_vl = atoi(argv[4]);

  prctl(PR_SME_SET_VL, start_vl);

  if (ssve)
    SMSTART_SM;

  if (za) {
    SMSTART_ZA;
    set_za_register(start_vl, 1);
  }

  write_sve_regs();

  return 0; // Set a break point here.
}
