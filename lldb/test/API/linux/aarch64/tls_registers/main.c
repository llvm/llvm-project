#include <stdbool.h>
#include <stdint.h>

uint64_t get_tpidr(void) {
  uint64_t tpidr = 0;
  __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tpidr));
  return tpidr;
}

uint64_t get_tpidr2(void) {
  uint64_t tpidr2 = 0;
  // S3_3_C13_C0_5 means tpidr2, and will work with older tools.
  __asm__ volatile("mrs %0, S3_3_C13_C0_5" : "=r"(tpidr2));
  return tpidr2;
}

void set_tpidr(uint64_t value) {
  __asm__ volatile("msr tpidr_el0, %0" ::"r"(value));
}

void set_tpidr2(uint64_t value) {
  __asm__ volatile("msr S3_3_C13_C0_5, %0" ::"r"(value));
}

bool use_tpidr2 = false;
const uint64_t tpidr_pattern = 0x1122334455667788;
const uint64_t tpidr2_pattern = 0x8877665544332211;

void expr_func() {
  set_tpidr(~tpidr_pattern);
  if (use_tpidr2)
    set_tpidr2(~tpidr2_pattern);
}

int main(int argc, char *argv[]) {
  use_tpidr2 = argc > 1;

  uint64_t original_tpidr = get_tpidr();
  // Accessing this on a core without it produces SIGILL. Only do this if
  // requested.
  uint64_t original_tpidr2 = 0;
  if (use_tpidr2)
    original_tpidr2 = get_tpidr2();

  set_tpidr(tpidr_pattern);

  if (use_tpidr2)
    set_tpidr2(tpidr2_pattern);

  // Set break point at this line.
  // lldb will now set its own pattern(s) for us to find.

  uint64_t new_tpidr = get_tpidr();
  volatile bool tpidr_was_set = new_tpidr == 0x1111222233334444;

  uint64_t new_tpidr2 = 0;
  volatile bool tpidr2_was_set = false;
  if (use_tpidr2) {
    new_tpidr2 = get_tpidr2();
    tpidr2_was_set = new_tpidr2 == 0x4444333322221111;
  }

  set_tpidr(original_tpidr);
  if (use_tpidr2)
    set_tpidr2(original_tpidr2);

  return 0; // Set break point 2 at this line.
}
