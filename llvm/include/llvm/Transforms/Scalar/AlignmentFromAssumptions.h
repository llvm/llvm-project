//===---- AlignmentFromAssumptions.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a ScalarEvolution-based transformation to set
// the alignments of load, stores and memory intrinsics based on the truth
// expressions of assume intrinsics. The primary motivation is to handle
// complex alignment assumptions that apply to vector loads and stores that
// appear after vectorization and unrolling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_ALIGNMENTFROMASSUMPTIONS_H
#define LLVM_TRANSFORMS_SCALAR_ALIGNMENTFROMASSUMPTIONS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class AssumptionCache;
class CallInst;
class DominatorTree;
class ScalarEvolution;
class SCEV;
class Value;

struct AlignmentFromAssumptionsPass
    : public PassInfoMixin<AlignmentFromAssumptionsPass> {
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  // Glue for old PM.
  LLVM_ABI bool runImpl(Function &F, AssumptionCache &AC, ScalarEvolution *SE_,
                        DominatorTree *DT_);

  ScalarEvolution *SE = nullptr;
  DominatorTree *DT = nullptr;

  LLVM_ABI bool extractAlignmentInfo(CallInst *I, unsigned Idx, Value *&AAPtr,
                                     const SCEV *&AlignSCEV,
                                     const SCEV *&OffSCEV);
  LLVM_ABI bool processAssumption(CallInst *I, unsigned Idx);
};
}

#endif // LLVM_TRANSFORMS_SCALAR_ALIGNMENTFROMASSUMPTIONS_H
