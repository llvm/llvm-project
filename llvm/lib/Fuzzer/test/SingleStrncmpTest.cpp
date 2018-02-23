// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 64) return 0;
  char *S = (char*)Data;
  volatile auto Strncmp = &(strncmp);   // Make sure strncmp is not inlined.
  if (Size >= 6 && !Strncmp(S, "qwerty", 6)) {
    fprintf(stderr, "BINGO\n");
    exit(1);
  }
  return 0;
}
