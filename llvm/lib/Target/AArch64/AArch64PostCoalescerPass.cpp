//===- AArch64PostCoalescerPass.cpp - AArch64 Post Coalescer pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64MachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-post-coalescer"

namespace {

/// Expands FORM_TRANSPOSED_REG_TUPLE_{X2|X4}_PSEUDO instructions into copy
/// sequences. Note: This expansion occurs immediately before greedy regalloc
/// and after the coalescer and pre-RA scheduler.
///
/// Example:
///
///   %v2:zpr2 = FORM_TRANSPOSED_REG_TUPLE_X2_PSEUDO %v0.zsub0, %v1.zsub0
///
/// Expands to:
///
///   undef %v2.zsub0:zpr2 = COPY_INTO_TRANSPOSED_TUPLE %v0.zsub0, 2
///   %v2.zsub1:zpr2 = COPY_INTO_TRANSPOSED_TUPLE %v1.zsub0, 2
static bool expandFormTransposedRegTuple(MachineBasicBlock &MBB,
                                         MachineInstr &MI, LiveIntervals *LIS) {
  const TargetInstrInfo *TII =
      MBB.getParent()->getSubtarget<AArch64Subtarget>().getInstrInfo();
  unsigned TupleSize =
      MI.getOpcode() == AArch64::FORM_TRANSPOSED_REG_TUPLE_X2_PSEUDO ? 2 : 4;

  DebugLoc DL = MI.getDebugLoc();
  Register TupleReg = MI.getOperand(0).getReg();
  SmallVector<Register, 5> OrigRegs{TupleReg};
  MachineBasicBlock::iterator FirstCopyMBBI;

  for (unsigned I = 0; I < TupleSize; ++I) {
    MachineOperand &SrcOp = MI.getOperand(I + 1);
    OrigRegs.push_back(SrcOp.getReg());

    // Ensure that if operand is killed, the kill flag is placed on the final
    // copy for that operand.
    if (SrcOp.isKill()) {
      for (unsigned J = I + 2; J < MI.getNumOperands(); ++J) {
        MachineOperand &LaterOp = MI.getOperand(J);
        if (LaterOp.getReg() == SrcOp.getReg()) {
          LaterOp.setIsKill();
          SrcOp.setIsKill(false);
        }
      }
    }

    RegState DefState = I == 0 ? RegState::Undef : RegState::NoFlags;
    MachineInstr *CopyMI =
        BuildMI(MBB, MI, DL, TII->get(AArch64::COPY_INTO_TRANSPOSED_TUPLE))
            .addDef(TupleReg, DefState, AArch64::zsub0 + I)
            .add(SrcOp)
            .addImm(TupleSize);

    if (I == 0)
      FirstCopyMBBI = CopyMI;
  }

  MachineBasicBlock::iterator EndMBBI = std::next(MI.getIterator());
  if (LIS)
    LIS->RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();

  if (LIS)
    LIS->repairIntervalsInRange(&MBB, FirstCopyMBBI, EndMBBI, OrigRegs);
  return true;
}

bool runAArch64PostCoalescer(MachineFunction &MF, LiveIntervals *LIS) {
  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  if (!FuncInfo->hasStreamingModeChanges() &&
      !MF.getSubtarget<AArch64Subtarget>().isStreaming())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      switch (MI.getOpcode()) {
      default:
        break;
      case AArch64::FORM_TRANSPOSED_REG_TUPLE_X2_PSEUDO:
      case AArch64::FORM_TRANSPOSED_REG_TUPLE_X4_PSEUDO:
        Changed |= expandFormTransposedRegTuple(MBB, MI, LIS);
        break;
      case AArch64::COALESCER_BARRIER_FPR16:
      case AArch64::COALESCER_BARRIER_FPR32:
      case AArch64::COALESCER_BARRIER_FPR64:
      case AArch64::COALESCER_BARRIER_FPR128: {
        Register Src = MI.getOperand(1).getReg();
        Register Dst = MI.getOperand(0).getReg();
        if (Src != Dst)
          MRI.replaceRegWith(Dst, Src);

        if (MI.getOperand(1).isUndef())
          for (MachineOperand &MO : MRI.use_operands(Dst))
            MO.setIsUndef();

        // MI must be erased from the basic block before recalculating the live
        // interval.
        if (LIS)
          LIS->RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();

        if (LIS) {
          LIS->removeInterval(Src);
          LIS->createAndComputeVirtRegInterval(Src);
        }

        Changed = true;
        break;
      }
      }
    }
  }

  return Changed;
}

struct AArch64PostCoalescerLegacy : public MachineFunctionPass {
  static char ID;

  AArch64PostCoalescerLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Post Coalescer pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addUsedIfAvailable<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char AArch64PostCoalescerLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AArch64PostCoalescerLegacy, "aarch64-post-coalescer",
                      "AArch64 Post Coalescer Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AArch64PostCoalescerLegacy, "aarch64-post-coalescer",
                    "AArch64 Post Coalescer Pass", false, false)

bool AArch64PostCoalescerLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
  auto *LIS = LISWrapper ? &LISWrapper->getLIS() : nullptr;
  return runAArch64PostCoalescer(MF, LIS);
}

PreservedAnalyses
AArch64PostCoalescerPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  auto *LIS = MFAM.getCachedResult<LiveIntervalsAnalysis>(MF);
  const bool Changed = runAArch64PostCoalescer(MF, LIS);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserve<SlotIndexesAnalysis>();
  return PA;
}

FunctionPass *llvm::createAArch64PostCoalescerPass() {
  return new AArch64PostCoalescerLegacy();
}
