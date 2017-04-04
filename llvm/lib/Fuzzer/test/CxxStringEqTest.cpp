// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer. Must find a specific string
// used in std::string operator ==.
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <string>
#include <iostream>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::string Str((const char*)Data, Size);
  bool Eq = Str == "FooBar";
  Sink = Str == "123456";   // Try to confuse the fuzzer
  if (Eq) {
    std::cout << "BINGO; Found the target, exiting\n";
    std::cout.flush();
    abort();
  }
  return 0;
}

