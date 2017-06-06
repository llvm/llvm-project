// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer. This test may trigger two different bugs.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static volatile int *Null = 0;

void Foo() { Null[1] = 0; }
void Bar() { Null[2] = 0; }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 10 && Data[0] == 'H')
    Foo();
  if (Size >= 10 && Data[0] == 'H')
    Bar();
  return 0;
}

