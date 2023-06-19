#include <stdbool.h>
#include <stdint.h>

int main() {
  // Save tpidr to restore later.
  uint64_t tpidr = 0;
  __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tpidr));

  // Set a pattern for lldb to find.
  uint64_t pattern = 0x1122334455667788;
  __asm__ volatile("msr tpidr_el0, %0" ::"r"(pattern));

  // Set break point at this line.
  // lldb will now set its own pattern for us to find.

  uint64_t new_tpidr = pattern;
  __asm__ volatile("mrs %0, tpidr_el0" : "=r"(new_tpidr));
  volatile bool tpidr_was_set = new_tpidr == 0x8877665544332211;

  // Restore the original.
  __asm__ volatile("msr tpidr_el0, %0" ::"r"(tpidr));

  return 0; // Set break point 2 at this line.
}
