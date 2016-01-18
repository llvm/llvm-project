// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // TODO: check other sizes.
  if (Size >= 8 && memcmp(Data, "01234567", 8) == 0) {
    if (Size >= 12 && memcmp(Data + 8, "ABCD", 4) == 0) {
      if (Size >= 14 && memcmp(Data + 12, "XY", 2) == 0) {
        if (Size >= 16 && memcmp(Data + 14, "KLM", 3) == 0) {
          fprintf(stderr, "BINGO\n");
          exit(1);
        }
      }
    }
  }
  return 0;
}
