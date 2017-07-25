// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Threaded test for a fuzzer. The fuzzer should find "H"
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <thread>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  auto C = [&] {
    if (Size >= 2 && Data[0] == 'H') {
        std::cout << "BINGO; Found the target, exiting\n" << std::flush;
        abort();
    }
  };
  std::thread T[] = {std::thread(C), std::thread(C), std::thread(C),
                     std::thread(C), std::thread(C), std::thread(C)};
  for (auto &X : T)
    X.join();
  return 0;
}

