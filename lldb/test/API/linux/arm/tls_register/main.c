#include <stdint.h>

int main(int argc, char *argv[]) {
  uint32_t tpidruro = 0;
  __asm__ volatile("mrc p15, 0, %0, c13, c0, 3" : "=r"(tpidruro));
  return 0; // Set breakpoint here.
}
