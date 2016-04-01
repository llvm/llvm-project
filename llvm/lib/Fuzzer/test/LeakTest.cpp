// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Test with a leak.
#include <cstdint>
#include <cstddef>

static volatile void *Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  Sink = new int;
  return 0;
}

