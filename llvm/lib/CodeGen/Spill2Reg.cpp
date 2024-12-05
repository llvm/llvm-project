//===- Spill2Reg.cpp - Spill To Register Optimization ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements Spill2Reg, an optimization which selectively
/// replaces spills/reloads to/from the stack with register copies to/from other
/// registers. This works even on targets where load/stores have similar latency
/// to register copies because it can free up memory units which helps avoid
/// stalls in the pipeline.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

class Spill2Reg : public MachineFunctionPass {
public:
  static char ID;
  Spill2Reg() : MachineFunctionPass(ID) {
    initializeSpill2RegPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnMachineFunction(MachineFunction &) override;
};

} // namespace

void Spill2Reg::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void Spill2Reg::releaseMemory() {}

bool Spill2Reg::runOnMachineFunction(MachineFunction &MFn) {
  llvm_unreachable("Unimplemented");
}

char Spill2Reg::ID = 0;

char &llvm::Spill2RegID = Spill2Reg::ID;

INITIALIZE_PASS_BEGIN(Spill2Reg, "spill2reg", "Spill2Reg", false, false)
INITIALIZE_PASS_END(Spill2Reg, "spill2reg", "Spill2Reg", false, false)
