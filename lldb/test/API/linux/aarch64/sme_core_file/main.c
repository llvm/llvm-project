// clang-format off
// Compile with:
// clang -target aarch64-unknown-linux-gnu main.c -o a.out -g -march=armv8.6-a+sve+sme
//
// For minimal corefile size, do this before running the program:
// echo 0x20 > /proc/self/coredeump_filter
//
// Must be run on a system that has SVE and SME, including the smefa64
// extension. Example command:
// main 0 32 64 1
//
// This would not enter streaming mode, set non-streaming VL to 32
// bytes, streaming VL to 64 bytes and enable ZA.
// clang-format on

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

#define SM_INST(c) asm volatile("msr s0_3_c4_c" #c "_3, xzr")
#define SMSTART_SM SM_INST(3)
#define SMSTART_ZA SM_INST(5)

void set_sve_registers() {
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

void set_za_register(int streaming_vl) {
#define MAX_VL_BYTES 256
  uint8_t data[MAX_VL_BYTES];

  for (unsigned i = 0; i < streaming_vl; ++i) {
    for (unsigned j = 0; j < MAX_VL_BYTES; ++j)
      data[j] = i + 1;
    asm volatile("mov w12, %w0\n\t"
                 "ldr za[w12, 0], [%1]\n\t" ::"r"(i),
                 "r"(&data)
                 : "w12");
  }
}

void set_tpidr2(uint64_t value) {
  __asm__ volatile("msr S3_3_C13_C0_5, %0" ::"r"(value));
}

int main(int argc, char **argv) {
  // Arguments:
  //                      SVE mode: 1 for streaming SVE (SSVE), any other value
  //                      for non-streaming SVE mode.
  //   Non-Streaming Vector length: In bytes, an integer e.g. "32".
  //       Streaming Vector length: As above, but for streaming mode.
  //                       ZA mode: 1 for enabled, any other value for disabled.
  if (argc != 5)
    return 1;

  // We assume this is run on a system with SME, so tpidr2 can always be
  // accessed.
  set_tpidr2(0x1122334455667788);

  // Streaming mode or not?
  bool streaming_mode = strcmp(argv[1], "1") == 0;

  // Set vector length (is a syscall, resets modes).
  int non_streaming_vl = atoi(argv[2]);
  prctl(PR_SVE_SET_VL, non_streaming_vl);
  int streaming_vl = atoi(argv[3]);
  prctl(PR_SME_SET_VL, streaming_vl);

  if (streaming_mode)
    SMSTART_SM;

  set_sve_registers();

  // ZA enabled or disabled?
  if (strcmp(argv[4], "1") == 0) {
    SMSTART_ZA;
    set_za_register(streaming_vl);
  }

  *(volatile char *)(0) = 0; // Crashes here.

  return 0;
}
