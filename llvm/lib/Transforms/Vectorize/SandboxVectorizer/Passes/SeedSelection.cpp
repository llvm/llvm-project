//===- SeedSelection.cpp - Seed selection pass of the bottom-up vectorizer ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/SeedSelection.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SeedCollector.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm {

static cl::opt<unsigned>
    OverrideVecRegBits("sbvec-vec-reg-bits", cl::init(0), cl::Hidden,
                       cl::desc("Override the vector register size in bits, "
                                "which is otherwise found by querying TTI."));
static cl::opt<bool>
    AllowNonPow2("sbvec-allow-non-pow2", cl::init(false), cl::Hidden,
                 cl::desc("Allow non-power-of-2 vectorization."));

namespace sandboxir {
SeedSelection::SeedSelection(StringRef Pipeline)
    : FunctionPass("seed-selection"),
      RPM("rpm", Pipeline, SandboxVectorizerPassBuilder::createRegionPass) {}

bool SeedSelection::runOnFunction(Function &F, const Analyses &A) {
  bool Change = false;
  const auto &DL = F.getParent()->getDataLayout();
  unsigned VecRegBits =
      OverrideVecRegBits != 0
          ? OverrideVecRegBits
          : A.getTTI()
                .getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
                .getFixedValue();

  // TODO: Start from innermost BBs first
  for (auto &BB : F) {
    SeedCollector SC(&BB, A.getScalarEvolution());
    for (SeedBundle &Seeds : SC.getStoreSeeds()) {
      unsigned ElmBits =
          Utils::getNumBits(VecUtils::getElementType(Utils::getExpectedType(
                                Seeds[Seeds.getFirstUnusedElementIdx()])),
                            DL);

      auto DivideBy2 = [](unsigned Num) {
        auto Floor = VecUtils::getFloorPowerOf2(Num);
        if (Floor == Num)
          return Floor / 2;
        return Floor;
      };
      // Try to create the largest vector supported by the target. If it fails
      // reduce the vector size by half.
      for (unsigned SliceElms = std::min(VecRegBits / ElmBits,
                                         Seeds.getNumUnusedBits() / ElmBits);
           SliceElms >= 2u; SliceElms = DivideBy2(SliceElms)) {
        if (Seeds.allUsed())
          break;
        // Keep trying offsets after FirstUnusedElementIdx, until we vectorize
        // the slice. This could be quite expensive, so we enforce a limit.
        for (unsigned Offset = Seeds.getFirstUnusedElementIdx(),
                      OE = Seeds.size();
             Offset + 1 < OE; Offset += 1) {
          // Seeds are getting used as we vectorize, so skip them.
          if (Seeds.isUsed(Offset))
            continue;
          if (Seeds.allUsed())
            break;

          auto SeedSlice =
              Seeds.getSlice(Offset, SliceElms * ElmBits, !AllowNonPow2);
          if (SeedSlice.empty())
            continue;

          assert(SeedSlice.size() >= 2 && "Should have been rejected!");

          // Create a region containing the seed slice.
          auto &Ctx = F.getContext();
          Region Rgn(Ctx, A.getTTI());
          Rgn.setAux(SeedSlice);
          // Run the region pass pipeline.
          Change |= RPM.runOnRegion(Rgn, A);
        }
      }
    }
  }
  return Change;
}
} // namespace sandboxir
} // namespace llvm
