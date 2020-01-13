//===- TapirCleanup - Cleanup leftover Tapir tasks for code generation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass serializes any remaining Tapir instructions before code generation.
// Typically this pass should have no effect, because Tapir instructions should
// have been lowered already to a particular parallel runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapircleanup"

STATISTIC(NumTasksSerialized, "Number of Tapir tasks serialized");

namespace {
class TapirCleanup : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.

  TapirCleanup() : FunctionPass(ID) {}

  bool runOnFunction(Function &Fn) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  StringRef getPassName() const override {
    return "Tapir last-minute cleanup for CodeGen";
  }
};
} // end anonymous namespace

char TapirCleanup::ID = 0;

INITIALIZE_PASS_BEGIN(TapirCleanup, DEBUG_TYPE,
                      "Cleanup Tapir", false, false)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(TapirCleanup, DEBUG_TYPE,
                    "Cleanup Tapir", false, false)

FunctionPass *llvm::createTapirCleanupPass() { return new TapirCleanup(); }

void TapirCleanup::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TaskInfoWrapperPass>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
}

bool TapirCleanup::runOnFunction(Function &F) {
  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  if (TI.isSerial())
    return false;
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  // If we haven't lowered the Tapir task to a particular parallel runtime by
  // this point, simply serialize the task.
  for (Task *T : post_order(TI.getRootTask())) {
    if (T->isRootTask())
      continue;
    ORE.emit(DiagnosticInfoOptimizationFailure(DEBUG_TYPE, "CleanedUpTapir",
                                               T->getDetach()->getDebugLoc(),
                                               T->getDetach()->getParent())
             << "CodeGen found Tapir instructions to serialize.  Specify a "
                "Tapir back-end to lower Tapir instructions to a parallel "
                "runtime.");

    SerializeDetach(T->getDetach(), T);
    NumTasksSerialized++;
  }

  return true;
}
