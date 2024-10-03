//===- SeedCollection.cpp  -0000000----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SeedCollector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Type.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Support/Debug.h"
#include <span>

using namespace llvm;
namespace llvm::sandboxir {

MutableArrayRef<SeedBundle::SeedList>
SeedBundle::getSlice(unsigned StartIdx, unsigned MaxVecRegBits,
                     bool ForcePowerOf2) {
  // Use uint32_t here for compatibility with IsPowerOf2_32

  // BitCount tracks the size of the working slice. From that we can tell
  // when the working slice's size is a power-of-two and when it exceeds
  // the legal size in MaxVecBits.
  uint32_t BitCount = 0;
  uint32_t NumElements = 0;
  // Can't start a slice with a used instruction.
  assert(!isUsed(StartIdx) && "Expected unused at StartIdx");
  for (auto S : make_range(Seeds.begin() + StartIdx, Seeds.end())) {
    uint32_t InstBits = Utils::getNumBits(S);
    // Stop if this instruction is used, or if adding it puts the slice over
    // the limit.
    if (isUsed(StartIdx + NumElements) || BitCount + InstBits > MaxVecRegBits)
      break;
    NumElements++;
    BitCount += Utils::getNumBits(S);
  }
  // Most slices will already be power-of-two-sized. But this one isn't, remove
  // instructions until it is. This could be tracked in the loop above but the
  // logic is harder to follow. TODO: Move if performance is unacceptable.
  if (ForcePowerOf2) {
    while (!isPowerOf2_32(BitCount) && NumElements > 1) {
      BitCount -= Utils::getNumBits(Seeds[StartIdx + NumElements - 1]);
      NumElements--;
    }
  }

  // Return any non-empty slice
  if (NumElements > 1)
    return MutableArrayRef<SeedBundle::SeedList>(&Seeds + StartIdx,
                                                 NumElements);
  else
    return {};
}

} // namespace llvm::sandboxir
