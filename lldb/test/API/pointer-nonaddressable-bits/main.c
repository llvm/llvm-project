#include <stdint.h>

uintptr_t get_high_bits(void *ptr) {
  uintptr_t address_bits = 56;
  uintptr_t mask = ~((1ULL << address_bits) - 1);
  uintptr_t ptrtoint = (uintptr_t)ptr;
  uintptr_t high_bits = ptrtoint & mask;
  return high_bits;
}

int main() {
  return 0; // break here
}
