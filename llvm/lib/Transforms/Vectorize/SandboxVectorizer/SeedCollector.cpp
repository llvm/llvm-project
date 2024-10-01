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

MutableArrayRef<sandboxir::SeedBundle::SeedList>
sandboxir::SeedBundle::getSlice(unsigned StartIdx, unsigned MaxVecRegBits,
                                bool ForcePowerOf2) {
  // Use uint32_t for counts to make it clear we are also using the proper
  // isPowerOf2_[32|64].

  // Count both the bits and the elements of the slice we are about to build.
  // The bits tell us whether this is a legal slice (that is <= MaxVecRegBits),
  // and the num of elements help us do the actual slicing.
  uint32_t BitsSum = 0;
  // As we are collecting slice elements we may go over the limit, so we need to
  // remember the last legal one. This is used for the creation of the slice.
  uint32_t LastGoodBitsSum = 0;
  uint32_t LastGoodNumSliceElements = 0;
  // Skip any used elements (which have already been handled) and all below
  // `StartIdx`.
  assert(StartIdx >= getFirstUnusedElementIdx() &&
         "Expected unused at StartIdx");
  uint32_t FirstGoodElementIdx = StartIdx;
  // Go through elements starting at FirstGoodElementIdx.
  for (auto [ElementCnt, S] : enumerate(make_range(
           std::next(Seeds.begin(), FirstGoodElementIdx), Seeds.end()))) {
    // Stop if we found a used element.
    if (isUsed(FirstGoodElementIdx + ElementCnt))
      break;
    BitsSum += sandboxir::Utils::getNumBits(S);
    // Stop if the bits sum is over the limit.
    if (BitsSum > MaxVecRegBits)
      break;
    // If forcing a power-of-2 bit-size we check if this bit size is accepted.
    if (ForcePowerOf2 && !isPowerOf2_32(BitsSum))
      continue;
    LastGoodBitsSum = BitsSum;
    LastGoodNumSliceElements = ElementCnt + 1;
  }
  if (LastGoodNumSliceElements < 2)
    return {};
  if (LastGoodBitsSum == 0)
    return {};
  return MutableArrayRef<sandboxir::SeedBundle::SeedList>(
      &Seeds + FirstGoodElementIdx, LastGoodNumSliceElements);
}
