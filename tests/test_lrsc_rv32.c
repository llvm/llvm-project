#include <stdint.h>

volatile uint32_t g = 0;

int cas_set_42(void) {
  uint32_t expected = 0;

  // Strong CAS. On RV32 with the 'A' extension, LLVM usually lowers this
  // to an LR/SC loop (lr.w / sc.w).
  return __atomic_compare_exchange_n(
      &g,                // ptr
      &expected,         // expected (updated on failure)
      42,                // desired
      0,                 // weak? 0 = strong
      __ATOMIC_SEQ_CST,  // success order
      __ATOMIC_SEQ_CST   // failure order
  );
}

int main(void) {
  return cas_set_42() ? 0 : 1;
}

