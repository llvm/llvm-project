#include <asm/hwcap.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif

#define SM_INST(c) asm volatile("msr s0_3_c4_c" #c "_3, xzr")
#define SMSTART SM_INST(7)
#define SMSTART_SM SM_INST(3)
#define SMSTART_ZA SM_INST(5)
#define SMSTOP SM_INST(6)
#define SMSTOP_SM SM_INST(2)
#define SMSTOP_ZA SM_INST(4)

#ifndef HWCAP2_SME2
#define HWCAP2_SME2 (1UL << 37)
#endif

// Doing a syscall exits streaming mode, but we need to use these during an
// expression in streaming mode. So it's set by main and we reference these
// later.
int svl_b = 0;
bool has_sme2 = 0;

#define VREG_NUM 32
#define VREG_SIZE 16

// Some of these could be statically allocated, but malloc-ing them all makes
// it simpler to write the Python side of this test.
uint8_t *expected_v_regs = NULL;
// These are treated as 32-bit but msr/mrs uses 64-bit values.
uint64_t *expected_fpcr = NULL;
uint64_t *expected_fpsr = NULL;
uint8_t *expected_sve_z = NULL;
uint8_t *expected_sve_p = NULL;
uint8_t *expected_sve_ffr = NULL;
uint8_t *expected_za = NULL;
uint8_t *expected_zt0 = NULL;
uint64_t *expected_svcr = NULL;
uint64_t *expected_svg = NULL;

static void *checked_malloc(size_t size) {
  void *ptr = malloc(size);
  if (ptr == NULL)
    exit(1);

  return ptr;
}

static int gpr_only_memcmp(uint8_t *lhs, uint8_t *rhs, size_t len) {
  // Hand written memcmp so we don't have to use
  // the compiler or library version, which would use SIMD registers
  // and corrupt registers before we can read them.
  int ret = 0;
  for (; len; ++lhs, ++rhs, --len) {
    asm volatile("cmp  %w1, %w2\n\t"
                 "cset %w0, ne \n\t"
                 : "=r"(ret)
                 : "r"(*lhs), "r"(*rhs)
                 : "cc");
    if (ret)
      return ret;
  }
  return ret;
}

// I would return bool here and check that from inside LLDB, but we cannot
// assume that expression evaluation works. So instead we exit, which is
// harder to track down but doesn't need expression evaluation to check for.
void check_register_values(bool streaming, bool za) {
  // In streaming mode, SIMD instructions are illegal. We will read the SIMD
  // portion of the streaming SVE registers later.
  if (!streaming) {
    uint64_t v_got[2];
#define VERIFY_V(NUM)                                                          \
  do {                                                                         \
    asm volatile("MOV %0, v" #NUM ".d[0]\n\t"                                  \
                 "MOV %1, v" #NUM ".d[1]\n\t"                                  \
                 : "=r"(v_got[0]), "=r"(v_got[1]));                            \
    if (gpr_only_memcmp((void *)(expected_v_regs + (NUM * VREG_SIZE)),         \
                        (void *)&v_got[0], VREG_SIZE) != 0)                    \
      exit(1);                                                                 \
  } while (0)

    VERIFY_V(0);
    VERIFY_V(1);
    VERIFY_V(2);
    VERIFY_V(3);
    VERIFY_V(4);
    VERIFY_V(5);
    VERIFY_V(6);
    VERIFY_V(7);
    VERIFY_V(8);
    VERIFY_V(9);
    VERIFY_V(10);
    VERIFY_V(11);
    VERIFY_V(12);
    VERIFY_V(13);
    VERIFY_V(14);
    VERIFY_V(15);
    VERIFY_V(16);
    VERIFY_V(17);
    VERIFY_V(18);
    VERIFY_V(19);
    VERIFY_V(20);
    VERIFY_V(21);
    VERIFY_V(22);
    VERIFY_V(23);
    VERIFY_V(24);
    VERIFY_V(25);
    VERIFY_V(26);
    VERIFY_V(27);
    VERIFY_V(28);
    VERIFY_V(29);
    VERIFY_V(30);
    VERIFY_V(31);

#undef VERIFY_V
  }

  uint64_t val = 0;
  asm volatile("mrs %0, fpcr" : "=r"(val));
  if (val != *expected_fpcr)
    exit(1);

  asm volatile("mrs %0, fpsr" : "=r"(val));
  if (val != *expected_fpsr)
    exit(1);

  // Can't read SVE registers outside of streaming mode.
  if (streaming) {
    // Not 0 init because 0 is a valid register value here.
    uint64_t got_svcr = 0xFFFFFFFFFFFFFFFFull;
    asm volatile("mrs %0, svcr" : "=r"(got_svcr));
    if (got_svcr != *expected_svcr)
      exit(1);

    // svg's unit is 8 byte granules.
    if (*expected_svg != svl_b / 8)
      exit(1);

    // We do not check FFR because we have no way to read or write it while in
    // streaming mode. Both wrffr and store value of ffr require SME_FA64, which
    // requires that you have SVE, which we don't have.

    size_t preg_size = svl_b / 8;
    uint8_t *got_sve_p = checked_malloc(preg_size);
#define VERIFY_P(NUM)                                                          \
  do {                                                                         \
    asm volatile("str p" #NUM ", [%0]" ::"r"(&got_sve_p[0]) : "memory");       \
    if (gpr_only_memcmp((void *)(expected_sve_p + (NUM * preg_size)),          \
                        (void *)got_sve_p, preg_size) != 0)                    \
      exit(1);                                                                 \
  } while (0)

    VERIFY_P(0);
    VERIFY_P(1);
    VERIFY_P(2);
    VERIFY_P(3);
    VERIFY_P(4);
    VERIFY_P(5);
    VERIFY_P(6);
    VERIFY_P(7);
    VERIFY_P(8);
    VERIFY_P(9);
    VERIFY_P(10);
    VERIFY_P(11);
    VERIFY_P(12);
    VERIFY_P(13);
    VERIFY_P(14);
    VERIFY_P(15);

#undef VERIFY_P

    uint8_t *got_sve_z = checked_malloc(svl_b);
    // Note that we are not using a p0 clobber below. We will manually restore
    // it once we have checked all the Z registers.
    // If we relied on the clobber, the compiler would only save and restore if
    // if was already using p0, which it usually is not.
#define VERIFY_Z(NUM)                                                          \
  do {                                                                         \
    asm volatile("ptrue p0.d\n\t"                                              \
                 "st1d z" #NUM ".d, p0, [%0]\n\t" ::"r"(got_sve_z)             \
                 : "memory");                                                  \
    if (gpr_only_memcmp((void *)(expected_sve_z + (NUM * svl_b)),              \
                        (void *)got_sve_z, svl_b) != 0)                        \
      exit(1);                                                                 \
  } while (0)

    VERIFY_Z(0);

    // Put back p0 value. LLDB will expect to see this.
    asm volatile("ldr p0, [%0]" ::"r"(expected_sve_p));

#undef VERIFY_Z
  }

  if (za) {
    uint8_t *got_za = checked_malloc(svl_b * svl_b);

    // Store one row of ZA at a time.
    uint8_t *za_row = got_za;
    for (int i = 0; i < svl_b; ++i, za_row += svl_b) {
      asm volatile("mov w12, %w0\n\t"
                   "str za[w12, 0], [%1]\n\t" ::"r"(i),
                   "r"(za_row)
                   : "w12");
    }

    if (gpr_only_memcmp(expected_za, got_za, svl_b * svl_b) != 0)
      exit(1);

    if (has_sme2) {
      uint8_t *got_zt0 = checked_malloc(svl_b * 2);
      asm volatile("str zt0, [%0]" ::"r"(got_zt0));

      if (gpr_only_memcmp((void *)expected_zt0, (void *)got_zt0, svl_b * 2) !=
          0)
        exit(1);
    }
  }
}

static void write_fp_control() {
  // Some of these bits won't get set, this is fine. Just needs to be
  // recongisable from inside the debugger.
  uint64_t val = 0x5555555555555555ULL;
  asm volatile("msr fpcr, %0" ::"r"(val));
  asm volatile("msr fpsr, %0" ::"r"(val));
}

static void write_sve_regs() {
  // We do not explicitly set ffr here because doing so requires the smefa64
  // extension (both for load to ffr, and the specific wrffr instruction).
  // To have that extension you have to have SVE outside of streaming
  // mode which we do not have.

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

  write_fp_control();
}

static void write_sme_regs(int svl_b) {
  uint8_t value_offset = 1;

#define MAX_VL_BYTES 256
  uint8_t data[MAX_VL_BYTES];

  // ldr za will actually wrap the selected vector row, by the number of rows
  // you have. So setting one that didn't exist would actually set one that did.
  // That's why we need the streaming vector length here.
  for (int i = 0; i < svl_b; ++i) {
    // Glibc's memset uses instructions not allowed in streaming mode, so we
    // do this, and make sure it's not optimised into memset.
    for (unsigned j = 0; j < MAX_VL_BYTES; ++j)
      data[j] = j + value_offset;

    // Each one of these loads a VL sized row of ZA.
    asm volatile("mov w12, %w0\n\t"
                 "ldr za[w12, 0], [%1]\n\t" ::"r"(i),
                 "r"(&data)
                 : "w12");
  }
#undef MAX_VL_BYTES

  if (has_sme2) {
#define ZTO_LEN (512 / 8)
    uint8_t data[ZTO_LEN];
    for (unsigned i = 0; i < ZTO_LEN; ++i)
      data[i] = i + value_offset;

    asm volatile("ldr zt0, [%0]" ::"r"(&data));
#undef ZT0_LEN
  }
}

static void write_simd_regs() {
  // base is added to each value. If base = 1, then v0 = 1, v1 = 2, etc.
  unsigned base = 1;

#define WRITE_SIMD(NUM)                                                        \
  asm volatile("MOV v" #NUM ".d[0], %0\n\t"                                    \
               "MOV v" #NUM ".d[1], %0\n\t" ::"r"((uint64_t)(base + NUM)))

  WRITE_SIMD(0);
  WRITE_SIMD(1);
  WRITE_SIMD(2);
  WRITE_SIMD(3);
  WRITE_SIMD(4);
  WRITE_SIMD(5);
  WRITE_SIMD(6);
  WRITE_SIMD(7);
  WRITE_SIMD(8);
  WRITE_SIMD(9);
  WRITE_SIMD(10);
  WRITE_SIMD(11);
  WRITE_SIMD(12);
  WRITE_SIMD(13);
  WRITE_SIMD(14);
  WRITE_SIMD(15);
  WRITE_SIMD(16);
  WRITE_SIMD(17);
  WRITE_SIMD(18);
  WRITE_SIMD(19);
  WRITE_SIMD(20);
  WRITE_SIMD(21);
  WRITE_SIMD(22);
  WRITE_SIMD(23);
  WRITE_SIMD(24);
  WRITE_SIMD(25);
  WRITE_SIMD(26);
  WRITE_SIMD(27);
  WRITE_SIMD(28);
  WRITE_SIMD(29);
  WRITE_SIMD(30);
  WRITE_SIMD(31);

  write_fp_control();
}

void expr_function(bool streaming, bool za, unsigned svl) {
  // Making this call exits streaming mode so it must be done first.
  prctl(PR_SME_SET_VL, svl);

  if (streaming) {
    SMSTART_SM;
    write_sve_regs();
  } else {
    SMSTOP_SM;
    write_simd_regs();
  }

  if (za) {
    SMSTART_ZA;
    write_sme_regs(svl_b);
  } else {
    SMSTOP_ZA;
  }
}

typedef struct {
  bool streaming;
  bool za;
  int svl;
} ProcessState;

ProcessState get_initial_state(const char *mode, const char *za,
                               const char *svl) {
  ProcessState ret;

  if (strcmp("streaming", mode) == 0)
    ret.streaming = true;
  else if (strcmp("simd", mode) == 0)
    ret.streaming = false;
  else {
    printf("Unexpected value \"%s\" for mode option.\n", mode);
    exit(1);
  }

  if (strcmp("on", za) == 0)
    ret.za = true;
  else if (strcmp("off", za) == 0)
    ret.za = false;
  else {
    printf("Unexpected value \"%s\" for za option.\n", za);
    exit(1);
  }

  ret.svl = atoi(svl);
  if (!svl) {
    printf("Unexpected svl \"%s\"\n", svl);
    exit(1);
  }

  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Expected 3 arguments, process mode, za on or off, streaming vector "
           "length\n");
    exit(1);
  }

  ProcessState initial_state = get_initial_state(argv[1], argv[2], argv[3]);
  svl_b = initial_state.svl;

  // Making a syscall exits streaming mode, so we have to set this now.
  prctl(PR_SME_SET_VL, svl_b);

  if ((getauxval(AT_HWCAP2) & HWCAP2_SME2))
    has_sme2 = true;

  expected_v_regs = checked_malloc(VREG_NUM * VREG_SIZE);
  expected_fpcr = checked_malloc(sizeof(uint64_t));
  expected_fpsr = checked_malloc(sizeof(uint64_t));
  expected_sve_z = checked_malloc(svl_b * 32);
  expected_sve_p = checked_malloc((svl_b / 8) * 16);
  expected_sve_ffr = checked_malloc(svl_b / 8);
  expected_za = checked_malloc(svl_b * svl_b);
  expected_zt0 = checked_malloc(svl_b * 2);
  expected_svcr = checked_malloc(sizeof(uint64_t));
  expected_svg = checked_malloc(sizeof(uint64_t));

  if (initial_state.streaming) {
    SMSTART_SM;
    write_sve_regs();
  } else {
    write_simd_regs();
  }

  if (initial_state.za) {
    SMSTART_ZA;
    write_sme_regs(svl_b);
  }

  // The number of these is greater than or equal to the number of "next"
  // each lldb test issues. The
  // idea is that lldb will write register values in, updated the expected
  // values, then step over. This should cause the values written via ptrace to
  // appear in this process and match the expected values in memory.
  check_register_values(initial_state.streaming,
                        initial_state.za); // Set a break point here.
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  check_register_values(initial_state.streaming, initial_state.za);
  // To catch us in case there are not enough above.
  exit(2);

  return 0;
}
