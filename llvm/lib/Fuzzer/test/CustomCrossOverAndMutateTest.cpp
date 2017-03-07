// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Test that libFuzzer does not crash when LLVMFuzzerMutate called from
// LLVMFuzzerCustomCrossOver.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string.h>
#include <vector>

#include "FuzzerInterface.h"

static volatile int sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::string Str(reinterpret_cast<const char *>(Data), Size);
  if (Size && Data[0] == '0')
    sink++;
  return 0;
}

extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                            const uint8_t *Data2, size_t Size2,
                                            uint8_t *Out, size_t MaxOutSize,
                                            unsigned int Seed) {
  std::vector<uint8_t> Buffer(MaxOutSize * 10);
  LLVMFuzzerMutate(Buffer.data(), Buffer.size(), Buffer.size());
  size_t Size = std::min<size_t>(Size1, MaxOutSize);
  memcpy(Out, Data1, Size);
  return Size;
}
