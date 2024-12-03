//===- SMEPeepholeOpt.cpp - SME peephole optimization pass-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass tries to remove back-to-back (smstart, smstop) and
// (smstop, smstart) sequences. The pass is conservative when it cannot
// determine that it is safe to remove these sequences.
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-peephole-opt"

namespace {

struct SMEPeepholeOpt : public MachineFunctionPass {
  static char ID;

  SMEPeepholeOpt() : MachineFunctionPass(ID) {
    initializeSMEPeepholeOptPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SME Peephole Optimization pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool optimizeStartStopPairs(MachineBasicBlock &MBB,
                              bool &HasRemovedAllSMChanges) const;
};

char SMEPeepholeOpt::ID = 0;

} // end anonymous namespace

static bool isConditionalStartStop(const MachineInstr *MI) {
  return MI->getOpcode() == AArch64::MSRpstatePseudo;
}

static bool isMatchingStartStopPair(const MachineInstr *MI1,
                                    const MachineInstr *MI2) {
  // We only consider the same type of streaming mode change here, i.e.
  // start/stop SM, or start/stop ZA pairs.
  if (MI1->getOperand(0).getImm() != MI2->getOperand(0).getImm())
    return false;

  // One must be 'start', the other must be 'stop'
  if (MI1->getOperand(1).getImm() == MI2->getOperand(1).getImm())
    return false;

  bool IsConditional = isConditionalStartStop(MI2);
  if (isConditionalStartStop(MI1) != IsConditional)
    return false;

  if (!IsConditional)
    return true;

  // Check to make sure the conditional start/stop pairs are identical.
  if (MI1->getOperand(2).getImm() != MI2->getOperand(2).getImm())
    return false;

  // Ensure reg masks are identical.
  if (MI1->getOperand(4).getRegMask() != MI2->getOperand(4).getRegMask())
    return false;

  // This optimisation is unlikely to happen in practice for conditional
  // smstart/smstop pairs as the virtual registers for pstate.sm will always
  // be different.
  // TODO: For this optimisation to apply to conditional smstart/smstop,
  // this pass will need to do more work to remove redundant calls to
  // __arm_sme_state.

  // Only consider conditional start/stop pairs which read the same register
  // holding the original value of pstate.sm, as some conditional start/stops
  // require the state on entry to the function.
  if (MI1->getOperand(3).isReg() && MI2->getOperand(3).isReg()) {
    Register Reg1 = MI1->getOperand(3).getReg();
    Register Reg2 = MI2->getOperand(3).getReg();
    if (Reg1.isPhysical() || Reg2.isPhysical() || Reg1 != Reg2)
      return false;
  }

  return true;
}

static bool ChangesStreamingMode(const MachineInstr *MI) {
  assert((MI->getOpcode() == AArch64::MSRpstatesvcrImm1 ||
          MI->getOpcode() == AArch64::MSRpstatePseudo) &&
         "Expected MI to be a smstart/smstop instruction");
  return MI->getOperand(0).getImm() == AArch64SVCR::SVCRSM ||
         MI->getOperand(0).getImm() == AArch64SVCR::SVCRSMZA;
}

static bool isSVERegOp(const TargetRegisterInfo &TRI,
                       const MachineRegisterInfo &MRI,
                       const MachineOperand &MO) {
  if (!MO.isReg())
    return false;

  Register R = MO.getReg();
  if (R.isPhysical())
    return llvm::any_of(TRI.subregs_inclusive(R), [](const MCPhysReg &SR) {
      return AArch64::ZPRRegClass.contains(SR) ||
             AArch64::PPRRegClass.contains(SR);
    });

  const TargetRegisterClass *RC = MRI.getRegClass(R);
  return TRI.getCommonSubClass(&AArch64::ZPRRegClass, RC) ||
         TRI.getCommonSubClass(&AArch64::PPRRegClass, RC);
}

bool SMEPeepholeOpt::optimizeStartStopPairs(
    MachineBasicBlock &MBB, bool &HasRemovedAllSMChanges) const {
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterInfo &TRI =
      *MBB.getParent()->getSubtarget().getRegisterInfo();

  bool Changed = false;
  MachineInstr *Prev = nullptr;
  SmallVector<MachineInstr *, 4> ToBeRemoved;

  // Convenience function to reset the matching of a sequence.
  auto Reset = [&]() {
    Prev = nullptr;
    ToBeRemoved.clear();
  };

  // Walk through instructions in the block trying to find pairs of smstart
  // and smstop nodes that cancel each other out. We only permit a limited
  // set of instructions to appear between them, otherwise we reset our
  // tracking.
  unsigned NumSMChanges = 0;
  unsigned NumSMChangesRemoved = 0;
  for (MachineInstr &MI : make_early_inc_range(MBB)) {
    switch (MI.getOpcode()) {
    case AArch64::MSRpstatesvcrImm1:
    case AArch64::MSRpstatePseudo: {
      if (ChangesStreamingMode(&MI))
        NumSMChanges++;

      if (!Prev)
        Prev = &MI;
      else if (isMatchingStartStopPair(Prev, &MI)) {
        // If they match, we can remove them, and possibly any instructions
        // that we marked for deletion in between.
        Prev->eraseFromParent();
        MI.eraseFromParent();
        for (MachineInstr *TBR : ToBeRemoved)
          TBR->eraseFromParent();
        ToBeRemoved.clear();
        Prev = nullptr;
        Changed = true;
        NumSMChangesRemoved += 2;
      } else {
        Reset();
        Prev = &MI;
      }
      continue;
    }
    default:
      if (!Prev)
        // Avoid doing expensive checks when Prev is nullptr.
        continue;
      break;
    }

    // Test if the instructions in between the start/stop sequence are agnostic
    // of streaming mode. If not, the algorithm should reset.
    switch (MI.getOpcode()) {
    default:
      Reset();
      break;
    case AArch64::COALESCER_BARRIER_FPR16:
    case AArch64::COALESCER_BARRIER_FPR32:
    case AArch64::COALESCER_BARRIER_FPR64:
    case AArch64::COALESCER_BARRIER_FPR128:
    case AArch64::COPY:
      // These instructions should be safe when executed on their own, but
      // the code remains conservative when SVE registers are used. There may
      // exist subtle cases where executing a COPY in a different mode results
      // in different behaviour, even if we can't yet come up with any
      // concrete example/test-case.
      if (isSVERegOp(TRI, MRI, MI.getOperand(0)) ||
          isSVERegOp(TRI, MRI, MI.getOperand(1)))
        Reset();
      break;
    case AArch64::ADJCALLSTACKDOWN:
    case AArch64::ADJCALLSTACKUP:
    case AArch64::ANDXri:
    case AArch64::ADDXri:
      // We permit these as they don't generate SVE/NEON instructions.
      break;
    case AArch64::VGRestorePseudo:
    case AArch64::VGSavePseudo:
      // When the smstart/smstop are removed, we should also remove
      // the pseudos that save/restore the VG value for CFI info.
      ToBeRemoved.push_back(&MI);
      break;
    case AArch64::MSRpstatesvcrImm1:
    case AArch64::MSRpstatePseudo:
      llvm_unreachable("Should have been handled");
    }
  }

  HasRemovedAllSMChanges =
      NumSMChanges && (NumSMChanges == NumSMChangesRemoved);
  return Changed;
}

INITIALIZE_PASS(SMEPeepholeOpt, "aarch64-sme-peephole-opt",
                "SME Peephole Optimization", false, false)

bool SMEPeepholeOpt::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  if (!MF.getSubtarget<AArch64Subtarget>().hasSME())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  bool Changed = false;
  bool FunctionHasAllSMChangesRemoved = false;

  // Even if the block lives in a function with no SME attributes attached we
  // still have to analyze all the blocks because we may call a streaming
  // function that requires smstart/smstop pairs.
  for (MachineBasicBlock &MBB : MF) {
    bool BlockHasAllSMChangesRemoved;
    Changed |= optimizeStartStopPairs(MBB, BlockHasAllSMChangesRemoved);
    FunctionHasAllSMChangesRemoved |= BlockHasAllSMChangesRemoved;
  }

  AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
  if (FunctionHasAllSMChangesRemoved)
    AFI->setHasStreamingModeChanges(false);

  return Changed;
}

FunctionPass *llvm::createSMEPeepholeOptPass() { return new SMEPeepholeOpt(); }
