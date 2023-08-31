#include <stdint.h>
#include <sys/prctl.h>

// base is added to each value. If base = 2, then v0 = 2, v1 = 3, etc.
void write_simd_regs(unsigned base) {
#define WRITE_SIMD(NUM)                                                        \
  asm volatile("MOV v" #NUM ".d[0], %0\n\t"                                    \
               "MOV v" #NUM ".d[1], %0\n\t" ::"r"(base + NUM))

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
}

unsigned verify_simd_regs() {
  uint64_t got_low = 0;
  uint64_t got_high = 0;
  uint64_t target = 0;

#define VERIFY_SIMD(NUM)                                                       \
  do {                                                                         \
    got_low = 0;                                                               \
    got_high = 0;                                                              \
    asm volatile("MOV %0, v" #NUM ".d[0]\n\t"                                  \
                 "MOV %1, v" #NUM ".d[1]\n\t"                                  \
                 : "=r"(got_low), "=r"(got_high));                             \
    target = NUM + 1;                                                          \
    if ((got_low != target) || (got_high != target))                           \
      return 1;                                                                \
  } while (0)

  VERIFY_SIMD(0);
  VERIFY_SIMD(1);
  VERIFY_SIMD(2);
  VERIFY_SIMD(3);
  VERIFY_SIMD(4);
  VERIFY_SIMD(5);
  VERIFY_SIMD(6);
  VERIFY_SIMD(7);
  VERIFY_SIMD(8);
  VERIFY_SIMD(9);
  VERIFY_SIMD(10);
  VERIFY_SIMD(11);
  VERIFY_SIMD(12);
  VERIFY_SIMD(13);
  VERIFY_SIMD(14);
  VERIFY_SIMD(15);
  VERIFY_SIMD(16);
  VERIFY_SIMD(17);
  VERIFY_SIMD(18);
  VERIFY_SIMD(19);
  VERIFY_SIMD(20);
  VERIFY_SIMD(21);
  VERIFY_SIMD(22);
  VERIFY_SIMD(23);
  VERIFY_SIMD(24);
  VERIFY_SIMD(25);
  VERIFY_SIMD(26);
  VERIFY_SIMD(27);
  VERIFY_SIMD(28);
  VERIFY_SIMD(29);
  VERIFY_SIMD(30);
  VERIFY_SIMD(31);

  return 0;
}

int main() {
#ifdef SSVE
  asm volatile("msr  s0_3_c4_c7_3, xzr" /*smstart*/);
#elif defined SVE
  // Make the non-streaming SVE registers active.
  asm volatile("cpy  z0.b, p0/z, #1\n\t");
#endif
  // else test plain SIMD access.

  write_simd_regs(0);

  return verify_simd_regs(); // Set a break point here.
}
