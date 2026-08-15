//===- RemoveOracleFunctions.cpp - Remove unused oracle functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes functions annotated with the "oracle-function" attribute by
// changing their linkage to available_externally. Oracle functions must have no
// uses by the time this pass runs (after ISel).
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "remove-oracle-functions"

namespace {

class RemoveOracleFunctions : public MachineFunctionPass {
public:
  static char ID;
  RemoveOracleFunctions() : MachineFunctionPass(ID) {
    initializeRemoveOracleFunctionsPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    Function &F = MF.getFunction();
    if (!F.hasFnAttribute("oracle-function"))
      return false;

    assert(F.use_empty() &&
           "oracle-function must have no uses by the time it is removed");

    // Change linkage to available_externally so that all subsequent
    // MachineFunctionPasses skip this function (MachineFunctionPass::
    // runOnFunction checks this). The AsmPrinter will also skip it,
    // so the function produces no output.
    F.setLinkage(GlobalValue::AvailableExternallyLinkage);
    return true;
  }
};

} // namespace

char RemoveOracleFunctions::ID = 0;

INITIALIZE_PASS(RemoveOracleFunctions, DEBUG_TYPE,
                "Remove unused oracle functions", false, false)

MachineFunctionPass *llvm::createRemoveOracleFunctionsPass() {
  return new RemoveOracleFunctions();
}
