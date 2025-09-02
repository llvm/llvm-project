#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int myglobal = 41;

uint64_t get_high_bits(void *ptr) {
  uint64_t mask = ~((1ULL << 48) - 1);
  uint64_t ptrtoint = (uint64_t)ptr;
  uint64_t high_bits = ptrtoint & mask;
  printf("Higher bits are = %llx\n", high_bits);
  return high_bits;
}

int main() {
  int x = 42;
  assert(0 == get_high_bits(&x));
  return 0; // break here
}
