//===- TapirToTarget.h - Lower Tapir to target ABI --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
