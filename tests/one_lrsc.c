#include <stdint.h>
#include <stdbool.h>

volatile uint32_t g = 0;

bool cas_u32(uint32_t expected, uint32_t desired) {
  // Strong CAS usually lowers to an LR/SC loop on RISC-V
  return __atomic_compare_exchange_n(&g, &expected, desired,
                                     /*weak=*/false,
                                     __ATOMIC_SEQ_CST,
                                     __ATOMIC_SEQ_CST);
}

