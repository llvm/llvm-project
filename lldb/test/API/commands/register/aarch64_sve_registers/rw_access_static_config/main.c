#include <stdbool.h>
#include <sys/prctl.h>

// If START_SSVE is defined, this program will start in streaming SVE mode
// (it will later enter and exit streaming mode a few times). Otherwise, it
// will start in non-streaming SVE mode.

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

#define SMSTART_SM() asm volatile("msr  s0_3_c4_c3_3, xzr" /*smstart sm*/)

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

// Set some different values so we can tell if lldb correctly returns to the set
// above after the expression is finished.
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

// This function will be called using jitted expression call. We change vector
// length and write SVE registers. Our program context should restore to
// orignal vector length and register values after expression evaluation.
int expr_eval_func(bool streaming) {
  int SET_VL_OPT = streaming ? PR_SME_SET_VL : PR_SVE_SET_VL;
  prctl(SET_VL_OPT, 8 * 2);
  // Note that doing a syscall brings you back to non-streaming mode, so we
  // don't need to SMSTOP here.
  if (streaming)
    SMSTART_SM();
  write_sve_regs_expr();
  prctl(SET_VL_OPT, 8 * 4);
  if (streaming)
    SMSTART_SM();
  write_sve_regs_expr();
  return 1;
}

int main() {
#ifdef START_SSVE
  SMSTART_SM();
#endif
  write_sve_regs();

  return 0; // Set a break point here.
}
