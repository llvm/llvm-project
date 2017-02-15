// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  const char kString64Bytes[] =
      "123456789 123456789 123456789 123456789 123456789 123456789 1234";
  assert(sizeof(kString64Bytes) == 65);
  if (Size >= 64 && memcmp(Data, kString64Bytes, 64) == 0) {
    fprintf(stderr, "BINGO\n");
    exit(1);
  }
  return 0;
}
