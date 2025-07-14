#include <gpuintrin.h>
#include <stdint.h>

extern uint32_t global[64];

[[gnu::visibility("default")]]
uint32_t funky() {
  global[0] = 100;
  return 200;
}
