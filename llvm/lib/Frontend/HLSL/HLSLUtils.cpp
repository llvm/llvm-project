//===- HLSLUtils.cpp - HLSL helper functions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Various helpers for working with HLSL shaders.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLUtils.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace hlsl {

// Parses a numthreads or wavesize attribute string in "X,Y,Z" format into three
// unsigned integers: the thread counts in each dimension for numthreads, or the
// minimum, maximum, and preferred wave sizes for wavesize.
// Validates the input only through assertions.
void parseThreadCounts(StringRef Str, unsigned &Count1, unsigned &Count2,
                       unsigned &Count3) {
  SmallVector<StringRef, 3> Counts;
  Str.split(Counts, ',');
  assert(Counts.size() == 3 && "expected three thread dimensions");

  [[maybe_unused]] bool IsInvalid;
  IsInvalid = Counts[0].getAsInteger(10, Count1);
  assert(!IsInvalid && "expected an integer thread count");
  IsInvalid = Counts[1].getAsInteger(10, Count2);
  assert(!IsInvalid && "expected an integer thread count");
  IsInvalid = Counts[2].getAsInteger(10, Count3);
  assert(!IsInvalid && "expected an integer thread count");
}

} // namespace hlsl
} // namespace llvm
