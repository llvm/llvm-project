//===---- TapirToTarget.h - Lower Tapir to target ABI -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers Tapir construct to a specified runtime ABI.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_TAPIRTOTARGET_H
#define LLVM_TRANSFORMS_TAPIR_TAPIRTOTARGET_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

/// The TapirToTarget Pass.
struct TapirToTargetPass : public PassInfoMixin<TapirToTargetPass> {
  TapirToTargetPass(TapirTargetID TargetID = TapirTargetID::Last_TapirTargetID)
      : TargetID(TargetID) {}

  /// \brief Run the pass over the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  TapirTargetID TargetID;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_TAPIRTOTARGET_H
