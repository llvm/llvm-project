#include <asm/hwcap.h>
#include <stdint.h>
#include <sys/auxv.h>

#ifndef HWCAP2_FPMR
#define HWCAP2_FPMR (1UL << 48)
#endif

uint64_t get_fpmr(void) {
  uint64_t fpmr = 0;
  __asm__ volatile("mrs %0, s3_3_c4_c4_2" : "=r"(fpmr));
  return fpmr;
}

void set_fpmr(uint64_t value) {
  __asm__ volatile("msr s3_3_c4_c4_2, %0" ::"r"(value));
}

// Set F8S1 (bits 0-2) and LSCALE2 (bits 37-32) (to prove we treat fpmr as 64
// bit).
const uint64_t original_fpmr = (uint64_t)0b101010 << 32 | (uint64_t)0b101;

void expr_func() { set_fpmr(original_fpmr); }

int main(int argc, char *argv[]) {
  if (!(getauxval(AT_HWCAP2) & HWCAP2_FPMR))
    return 1;

  // As FPMR controls a bunch of floating point options that are quite
  // extensive, we're not going to run any floating point ops here. Instead just
  // update the value from the debugger and check it from this program, and vice
  // versa.
  set_fpmr(original_fpmr);

  // Here the debugger checks it read back the value above, then writes in a new
  // value. Note that the bits are flipped in the new value.
  uint64_t new_fpmr = get_fpmr(); // Set break point at this line.
  uint64_t expected_fpmr = ((uint64_t)0b010101 << 32) | (uint64_t)0b010;

  // If the debugger failed to update the value, exit uncleanly.
  // This also allows you to run this program standalone to create a core file.
  if (new_fpmr != expected_fpmr)
    __builtin_trap();

  return 0;
}
