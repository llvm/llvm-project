//===- AMDGPUNewInsertWaitcnts.cpp - Wait Instruction Insertion Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A wait insertion pass for AMDGPU.
//
// NOTE: This is currently work in progress so don't expect this to work
//       correctly! This new pass will eventually replace SIInsertWaitcnt.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNewInsertWaitcnts.h"
#include "AMDGPU.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#define DEBUG_TYPE "amdgpu-new-insert-waitcnts"

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

constexpr const StringRef PassName = "AMDGPU New Insert Waitcnts";

class AMDGPUNewInsertWaitcntsLegacy : public MachineFunctionPass {
  InsertWaitcnts IW;

public:
  static char ID;

  AMDGPUNewInsertWaitcntsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override { return IW.run(MF); }

  StringRef getPassName() const override { return PassName; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUNewInsertWaitcntsLegacy, DEBUG_TYPE, PassName, false,
                false)

char AMDGPUNewInsertWaitcntsLegacy::ID = 0;

char &llvm::AMDGPUNewInsertWaitcntsID = AMDGPUNewInsertWaitcntsLegacy::ID;

FunctionPass *llvm::createAMDGPUNewInsertWaitcntsPass() {
  return new AMDGPUNewInsertWaitcntsLegacy();
}

PreservedAnalyses
AMDGPUNewInsertWaitcntsPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (!IW.run(MF))
    return PreservedAnalyses::all();
  // TODO: Preserve analyses.
  return PreservedAnalyses::none();
}

bool InsertWaitcnts::run(MachineFunction &MF) {
  bool Change = false;
  return Change;
}
