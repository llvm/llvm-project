//===----- RippleSESE.h - Update CFG to satisfy Ripple's SESE criterion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// According to ripple's semantics, any sub-CFG constructed from a basic block
// containing a branch instruction dependent on ripple-id to its post-dominator
// must have a single-entry and single-exit. This requirement is known as the
// single-entry, single-exit (SESE) criterion. When this criterion is violated,
// RippleSESE is triggered to clone certain basic block paths, ensuring that the
// resulting CFG satisfies the SESE criterion.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_RIPPLE_SESE_H
#define LLVM_TRANSFORMS_VECTORIZE_RIPPLE_SESE_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Ripple/Ripple.h"

namespace llvm {

template <typename ValueTy> class AssertingVH;
class Function;
class TargetMachine;

class RippleSESEPass : public PassInfoMixin<RippleSESEPass> {
  TargetMachine *TM;
  Ripple::ProcessingStatus &PS;
  DenseSet<AssertingVH<Function>> &SpecializationsPending,
      &SpecializationsAvailable;

public:
  RippleSESEPass(TargetMachine *TM, Ripple::ProcessingStatus &PS,
                 DenseSet<AssertingVH<Function>> &SpecializationsPending,
                 DenseSet<AssertingVH<Function>> &SpecializationsAvailable)
      : TM(TM), PS(PS), SpecializationsPending(SpecializationsPending),
        SpecializationsAvailable(SpecializationsAvailable) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  // Run RippleSESE when optnone is set
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_RIPPLE_SESE_H
