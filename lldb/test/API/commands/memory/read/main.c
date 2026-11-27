#include <stdint.h>

int main(int argc, const char *argv[]) {
  char my_string[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 0};
  double my_double = 1234.5678;
  int my_ints[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
  uint64_t my_uint64s[] = {0, 1, 2, 3, 4, 5, 6, 7};
  // assume that 0xffffff is invalid instruction in RISC-V and AArch64,
  // so decoding it will fail
  char my_insns[] = {0xff, 0xff, 0xff};
  // 2 x the default read size of 32 bytes.
  uint8_t incrementing_bytes[64];
  for (unsigned i = 0; i < 64; ++i)
    incrementing_bytes[i] = i;
  return 0; // break here
}
