//=== LoongArchDeadRegisterDefinitions.cpp - Replace dead defs w/ zero reg ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass rewrites Rd to r0 for instrs whose return values are unused.
//
//===---------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;
#define DEBUG_TYPE "loongarch-dead-register-definitions"
#define LoongArch_DEAD_REG_DEF_NAME "LoongArch Dead register definitions"

STATISTIC(NumDeadDefsReplaced, "Number of dead definitions replaced");

namespace {
class LoongArchDeadRegisterDefinitionsLegacy : public MachineFunctionPass {
public:
  static char ID;

  LoongArchDeadRegisterDefinitionsLegacy() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveDebugVariablesWrapperLegacy>();
    AU.addPreserved<LiveStacksWrapperLegacy>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return LoongArch_DEAD_REG_DEF_NAME; }
};
} // end anonymous namespace

char LoongArchDeadRegisterDefinitionsLegacy::ID = 0;
INITIALIZE_PASS(LoongArchDeadRegisterDefinitionsLegacy, DEBUG_TYPE,
                LoongArch_DEAD_REG_DEF_NAME, false, false)

FunctionPass *llvm::createLoongArchDeadRegisterDefinitionsLegacyPass() {
  return new LoongArchDeadRegisterDefinitionsLegacy();
}

static bool runDeadRegisterDefinitions(MachineFunction &MF,
                                       LiveIntervals &LIS) {
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  LLVM_DEBUG(dbgs() << "***** LoongArchDeadRegisterDefinitions *****\n");

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // We only handle non-computational instructions.
      const MCInstrDesc &Desc = MI.getDesc();
      if (!Desc.mayLoad() && !Desc.mayStore() &&
          !Desc.hasUnmodeledSideEffects())
        continue;
      for (int I = 0, E = Desc.getNumDefs(); I != E; ++I) {
        MachineOperand &MO = MI.getOperand(I);
        if (!MO.isReg() || !MO.isDef() || MO.isEarlyClobber())
          continue;
        // Be careful not to change the register if it's a tied operand.
        if (MI.isRegTiedToUseOperand(I)) {
          LLVM_DEBUG(dbgs() << "    Ignoring, def is tied operand.\n");
          continue;
        }
        Register Reg = MO.getReg();
        if (!Reg.isVirtual() || !MO.isDead())
          continue;
        LLVM_DEBUG(dbgs() << "    Dead def operand #" << I << " in:\n      ";
                   MI.print(dbgs()));
        const TargetRegisterClass *RC = TII->getRegClass(Desc, I);
        if (!(RC && RC->contains(LoongArch::R0))) {
          LLVM_DEBUG(dbgs() << "    Ignoring, register is not a GPR.\n");
          continue;
        }
        assert(LIS.hasInterval(Reg));
        LIS.removeInterval(Reg);
        MO.setReg(LoongArch::R0);
        LLVM_DEBUG(dbgs() << "    Replacing with zero register. New:\n      ";
                   MI.print(dbgs()));
        ++NumDeadDefsReplaced;
        MadeChange = true;
      }
    }
  }

  return MadeChange;
}

bool LoongArchDeadRegisterDefinitionsLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return runDeadRegisterDefinitions(
      MF, getAnalysis<LiveIntervalsWrapperPass>().getLIS());
}

PreservedAnalyses LoongArchDeadRegisterDefinitionsPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  bool Changed =
      runDeadRegisterDefinitions(MF, MFAM.getResult<LiveIntervalsAnalysis>(MF));
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<LiveIntervalsAnalysis>()
      .preserve<SlotIndexesAnalysis>()
      .preserve<LiveDebugVariablesAnalysis>()
      .preserve<LiveStacksAnalysis>();
}
