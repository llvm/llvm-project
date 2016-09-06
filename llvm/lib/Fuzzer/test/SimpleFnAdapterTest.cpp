// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer Fn adapter. The fuzzer has to find two non-empty
// vectors with the same content.

#include <iostream>
#include <vector>

#include "FuzzerFnAdapter.h"

static void TestFn(std::vector<uint8_t> V1, std::vector<uint8_t> V2) {
  if (V1.size() > 0 && V1 == V2) {
    std::cout << "BINGO; Found the target, exiting\n";
    exit(0);
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  fuzzer::Adapt(TestFn, Data, Size);
  return 0;
}


