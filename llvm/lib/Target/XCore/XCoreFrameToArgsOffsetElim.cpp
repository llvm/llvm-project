//===-- XCoreFrameToArgsOffsetElim.cpp ----------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace Pseudo FRAME_TO_ARGS_OFFSET with the appropriate real offset.
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "XCoreInstrInfo.h"
#include "XCoreSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

namespace {
struct XCoreFrameToArgsOffsetEliminationLegacy : public MachineFunctionPass {
  static char ID;
  XCoreFrameToArgsOffsetEliminationLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &Fn) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override {
    return "XCore FRAME_TO_ARGS_OFFSET Elimination";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
char XCoreFrameToArgsOffsetEliminationLegacy::ID = 0;
}

/// createXCoreFrameToArgsOffsetEliminationPass - returns an instance of the
/// Frame to args offset elimination pass
FunctionPass *llvm::createXCoreFrameToArgsOffsetEliminationLegacyPass() {
  return new XCoreFrameToArgsOffsetEliminationLegacy();
}

static bool runFrameToArgOffsetElimination(MachineFunction &MF) {
  const XCoreInstrInfo &TII =
      *static_cast<const XCoreInstrInfo *>(MF.getSubtarget().getInstrInfo());
  unsigned StackSize = MF.getFrameInfo().getStackSize();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator MBBI = MBB.begin(), EE = MBB.end();
         MBBI != EE; ++MBBI) {
      if (MBBI->getOpcode() == XCore::FRAME_TO_ARGS_OFFSET) {
        MachineInstr &OldInst = *MBBI;
        Register Reg = OldInst.getOperand(0).getReg();
        MBBI = TII.loadImmediate(MBB, MBBI, Reg, StackSize);
        OldInst.eraseFromParent();
      }
    }
  }
  return true;
}

bool XCoreFrameToArgsOffsetEliminationLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return runFrameToArgOffsetElimination(MF);
}

PreservedAnalyses XCoreFrameToArgsOffsetEliminationPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  return runFrameToArgOffsetElimination(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
