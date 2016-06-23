// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Contains dummy functions used to avoid dependency on AFL.
#include <stdint.h>
#include <stdlib.h>

extern "C" void __afl_manual_init() {}

extern "C" int __afl_persistent_loop(unsigned int) {
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  return 0;
}
