//===- SeedCollector.cpp  -0000000-----------------------------------------===//
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

using namespace llvm;
namespace llvm::sandboxir {

MutableArrayRef<Instruction *> SeedBundle::getSlice(unsigned StartIdx,
                                                    unsigned MaxVecRegBits,
                                                    bool ForcePowerOf2) {
  // Use uint32_t here for compatibility with IsPowerOf2_32

  // BitCount tracks the size of the working slice. From that we can tell
  // when the working slice's size is a power-of-two and when it exceeds
  // the legal size in MaxVecBits.
  uint32_t BitCount = 0;
  uint32_t NumElements = 0;
  // Tracks the most recent slice where NumElements gave a power-of-2 BitCount
  uint32_t NumElementsPowerOfTwo = 0;
  uint32_t BitCountPowerOfTwo = 0;
  // Can't start a slice with a used instruction.
  assert(!isUsed(StartIdx) && "Expected unused at StartIdx");
  for (auto S : make_range(Seeds.begin() + StartIdx, Seeds.end())) {
    uint32_t InstBits = Utils::getNumBits(S);
    // Stop if this instruction is used, or if adding it puts the slice over
    // the limit.
    if (isUsed(StartIdx + NumElements) || BitCount + InstBits > MaxVecRegBits)
      break;
    NumElements++;
    BitCount += InstBits;
    if (ForcePowerOf2 && isPowerOf2_32(BitCount)) {
      NumElementsPowerOfTwo = NumElements;
      BitCountPowerOfTwo = BitCount;
    }
  }
  if (ForcePowerOf2) {
    NumElements = NumElementsPowerOfTwo;
    BitCount = BitCountPowerOfTwo;
  }

  assert((!ForcePowerOf2 || isPowerOf2_32(BitCount)) &&
         "Must be a power of two");
  // Return any non-empty slice
  if (NumElements > 1)
    return MutableArrayRef<Instruction *>(&Seeds[StartIdx], NumElements);
  else
    return {};
}

} // namespace llvm::sandboxir
