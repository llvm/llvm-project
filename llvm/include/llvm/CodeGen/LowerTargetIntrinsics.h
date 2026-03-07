//===- LowerTargetIntrinsics.h - Lower feature intrinsics -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers 'llvm.target.has.feature' and 'llvm.target.is.cpu'
// intrinsics into constants by querying the TargetMachine's subtarget for the
// enclosing function. It then propagates the resulting constants, folds
// branches, and removes dead blocks.
//
// This is a correctness requirement, code guarded by these intrinsics may
// contain instructions illegal on the current target. The dead code must be
// eliminated before ISel. The pass must run even on optnone functions at -O0.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LOWERTARGETINTRINSICS_H
#define LLVM_CODEGEN_LOWERTARGETINTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TargetMachine;

/// Lower all llvm.target.has.feature and llvm.target.is.cpu calls in \p F to
/// constants by querying \p TM.
bool lowerTargetIntrinsics(Function &F, const TargetMachine &TM);

class LowerTargetIntrinsicsPass
    : public PassInfoMixin<LowerTargetIntrinsicsPass> {
  const TargetMachine *TM;

public:
  LowerTargetIntrinsicsPass(const TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_CODEGEN_LOWERTARGETINTRINSICS_H
