//=== AArch64PostSelectOptimize.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does post-instruction-selection optimizations in the GlobalISel
// pipeline, before the rest of codegen runs.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64TargetMachine.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-post-select-optimize"

using namespace llvm;

namespace {
class AArch64PostSelectOptimize : public MachineFunctionPass {
public:
  static char ID;

  AArch64PostSelectOptimize();

  StringRef getPassName() const override {
    return "AArch64 Post Select Optimizer";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool optimizeNZCVDefs(MachineBasicBlock &MBB);
};
} // end anonymous namespace

void AArch64PostSelectOptimize::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

AArch64PostSelectOptimize::AArch64PostSelectOptimize()
    : MachineFunctionPass(ID) {
  initializeAArch64PostSelectOptimizePass(*PassRegistry::getPassRegistry());
}

unsigned getNonFlagSettingVariant(unsigned Opc) {
  switch (Opc) {
  default:
    return 0;
  case AArch64::SUBSXrr:
    return AArch64::SUBXrr;
  case AArch64::SUBSWrr:
    return AArch64::SUBWrr;
  case AArch64::SUBSXrs:
    return AArch64::SUBXrs;
  case AArch64::SUBSXri:
    return AArch64::SUBXri;
  case AArch64::SUBSWri:
    return AArch64::SUBWri;
  }
}

bool AArch64PostSelectOptimize::optimizeNZCVDefs(MachineBasicBlock &MBB) {
  // Consider the following code:
  //  FCMPSrr %0, %1, implicit-def $nzcv
  //  %sel1:gpr32 = CSELWr %_, %_, 12, implicit $nzcv
  //  %sub:gpr32 = SUBSWrr %_, %_, implicit-def $nzcv
  //  FCMPSrr %0, %1, implicit-def $nzcv
  //  %sel2:gpr32 = CSELWr %_, %_, 12, implicit $nzcv
  // This kind of code where we have 2 FCMPs each feeding a CSEL can happen
  // when we have a single IR fcmp being used by two selects. During selection,
  // to ensure that there can be no clobbering of nzcv between the fcmp and the
  // csel, we have to generate an fcmp immediately before each csel is
  // selected.
  // However, often we can essentially CSE these together later in MachineCSE.
  // This doesn't work though if there are unrelated flag-setting instructions
  // in between the two FCMPs. In this case, the SUBS defines NZCV
  // but it doesn't have any users, being overwritten by the second FCMP.
  //
  // Our solution here is to try to convert flag setting operations between
  // a interval of identical FCMPs, so that CSE will be able to eliminate one.
  bool Changed = false;
  auto &MF = *MBB.getParent();
  auto &Subtarget = MF.getSubtarget();
  const auto &TII = Subtarget.getInstrInfo();
  auto TRI = Subtarget.getRegisterInfo();
  auto RBI = Subtarget.getRegBankInfo();
  auto &MRI = MF.getRegInfo();

  // The first step is to find the first and last FCMPs. If we have found
  // at least two, then set the limit of the bottom-up walk to the first FCMP
  // found since we're only interested in dealing with instructions between
  // them.
  MachineInstr *FirstCmp = nullptr, *LastCmp = nullptr;
  for (auto &MI : instructionsWithoutDebug(MBB.begin(), MBB.end())) {
    if (MI.getOpcode() == AArch64::FCMPSrr ||
        MI.getOpcode() == AArch64::FCMPDrr) {
      if (!FirstCmp)
        FirstCmp = &MI;
      else
        LastCmp = &MI;
    }
  }

  // In addition to converting flag-setting ops in fcmp ranges into non-flag
  // setting ops, across the whole basic block we also detect when nzcv
  // implicit-defs are dead, and mark them as dead. Peephole optimizations need
  // this information later.

  LiveRegUnits LRU(*MBB.getParent()->getSubtarget().getRegisterInfo());
  LRU.addLiveOuts(MBB);
  bool NZCVDead = LRU.available(AArch64::NZCV);
  bool InsideCmpRange = false;
  for (auto &II : instructionsWithoutDebug(MBB.rbegin(), MBB.rend())) {
    LRU.stepBackward(II);

    if (LastCmp) { // There's a range present in this block.
      // If we're inside an fcmp range, look for begin instruction.
      if (InsideCmpRange && &II == FirstCmp)
        InsideCmpRange = false;
      else if (&II == LastCmp)
        InsideCmpRange = true;
    }

    // Did this instruction define NZCV?
    bool NZCVDeadAtCurrInstr = LRU.available(AArch64::NZCV);
    if (NZCVDead && NZCVDeadAtCurrInstr && II.definesRegister(AArch64::NZCV)) {
      // If we have a def and NZCV is dead, then we may convert this op.
      unsigned NewOpc = getNonFlagSettingVariant(II.getOpcode());
      int DeadNZCVIdx = II.findRegisterDefOperandIdx(AArch64::NZCV);
      if (DeadNZCVIdx != -1) {
        // If we're inside an fcmp range, then convert flag setting ops.
        if (InsideCmpRange && NewOpc) {
          LLVM_DEBUG(dbgs() << "Post-select optimizer: converting flag-setting "
                               "op in fcmp range: "
                            << II);
          II.setDesc(TII->get(NewOpc));
          II.RemoveOperand(DeadNZCVIdx);
          // Changing the opcode can result in differing regclass requirements,
          // e.g. SUBSWri uses gpr32 for the dest, whereas SUBWri uses gpr32sp.
          // Constrain the regclasses, possibly introducing a copy.
          constrainOperandRegClass(MF, *TRI, MRI, *TII, *RBI, II, II.getDesc(),
                                   II.getOperand(0), 0);
          Changed |= true;
        } else {
          // Otherwise, we just set the nzcv imp-def operand to be dead, so the
          // peephole optimizations can optimize them further.
          II.getOperand(DeadNZCVIdx).setIsDead();
        }
      }
    }

    NZCVDead = NZCVDeadAtCurrInstr;
  }
  return Changed;
}

bool AArch64PostSelectOptimize::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Selected) &&
         "Expected a selected MF");

  bool Changed = false;
  for (auto &BB : MF)
    Changed |= optimizeNZCVDefs(BB);
  return Changed;
}

char AArch64PostSelectOptimize::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PostSelectOptimize, DEBUG_TYPE,
                      "Optimize AArch64 selected instructions",
                      false, false)
INITIALIZE_PASS_END(AArch64PostSelectOptimize, DEBUG_TYPE,
                    "Optimize AArch64 selected instructions", false,
                    false)

namespace llvm {
FunctionPass *createAArch64PostSelectOptimize() {
  return new AArch64PostSelectOptimize();
}
} // end namespace llvm
