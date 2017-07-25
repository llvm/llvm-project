// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Crash on the N-th execution.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>

static int Counter;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Counter++ == 1000) {
    std::cout << "BINGO; Found the target, exiting\n" << std::flush;
    exit(1);
  }
  return 0;
}

