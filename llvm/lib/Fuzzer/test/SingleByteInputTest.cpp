// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer, need just one byte to crash.
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <cstdio>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 0 && Data[Size/2] == 42) {
    fprintf(stderr, "BINGO\n");
    abort();
  }
  return 0;
}

