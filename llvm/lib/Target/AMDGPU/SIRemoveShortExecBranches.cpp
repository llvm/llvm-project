//===-- SIRemoveShortExecBranches.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass optmizes the s_cbranch_execz instructions.
/// The pass removes this skip instruction for short branches,
/// if there is no unwanted sideeffect in the fallthrough code sequence.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-remove-short-exec-branches"

static unsigned SkipThreshold;

static cl::opt<unsigned, true> SkipThresholdFlag(
    "amdgpu-skip-threshold", cl::Hidden,
    cl::desc(
        "Number of instructions before jumping over divergent control flow"),
    cl::location(SkipThreshold), cl::init(12));

namespace {

class SIRemoveShortExecBranches : public MachineFunctionPass {
private:
  const SIInstrInfo *TII = nullptr;
  bool getBlockDestinations(MachineBasicBlock &SrcMBB,
                            MachineBasicBlock *&TrueMBB,
                            MachineBasicBlock *&FalseMBB,
                            SmallVectorImpl<MachineOperand> &Cond);
  bool mustRetainExeczBranch(const MachineBasicBlock &From,
                             const MachineBasicBlock &To) const;
  bool removeExeczBranch(MachineInstr &MI, MachineBasicBlock &SrcMBB);

public:
  static char ID;

  SIRemoveShortExecBranches() : MachineFunctionPass(ID) {
    initializeSIRemoveShortExecBranchesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // End anonymous namespace.

INITIALIZE_PASS(SIRemoveShortExecBranches, DEBUG_TYPE,
                "SI remove short exec branches", false, false)

char SIRemoveShortExecBranches::ID = 0;

char &llvm::SIRemoveShortExecBranchesID = SIRemoveShortExecBranches::ID;

bool SIRemoveShortExecBranches::getBlockDestinations(
    MachineBasicBlock &SrcMBB, MachineBasicBlock *&TrueMBB,
    MachineBasicBlock *&FalseMBB, SmallVectorImpl<MachineOperand> &Cond) {
  if (TII->analyzeBranch(SrcMBB, TrueMBB, FalseMBB, Cond))
    return false;

  if (!FalseMBB)
    FalseMBB = SrcMBB.getNextNode();

  return true;
}

bool SIRemoveShortExecBranches::mustRetainExeczBranch(
    const MachineBasicBlock &From, const MachineBasicBlock &To) const {
  unsigned NumInstr = 0;
  const MachineFunction *MF = From.getParent();

  for (MachineFunction::const_iterator MBBI(&From), ToI(&To), End = MF->end();
       MBBI != End && MBBI != ToI; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;

    for (MachineBasicBlock::const_iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      // When a uniform loop is inside non-uniform control flow, the branch
      // leaving the loop might never be taken when EXEC = 0.
      // Hence we should retain cbranch out of the loop lest it become infinite.
      if (I->isConditionalBranch())
        return true;

      if (TII->hasUnwantedEffectsWhenEXECEmpty(*I))
        return true;

      if (TII->isKillTerminator(I->getOpcode()))
        return true;

      // These instructions are potentially expensive even if EXEC = 0.
      if (TII->isSMRD(*I) || TII->isVMEM(*I) || TII->isFLAT(*I) ||
          I->getOpcode() == AMDGPU::S_WAITCNT)
        return true;

      ++NumInstr;
      if (NumInstr >= SkipThreshold)
        return true;
    }
  }

  return false;
}

// Returns true if the skip branch instruction is removed.
bool SIRemoveShortExecBranches::removeExeczBranch(MachineInstr &MI,
                                                  MachineBasicBlock &SrcMBB) {
  MachineBasicBlock *TrueMBB = nullptr;
  MachineBasicBlock *FalseMBB = nullptr;
  SmallVector<MachineOperand, 1> Cond;

  if (!getBlockDestinations(SrcMBB, TrueMBB, FalseMBB, Cond))
    return false;

  // Consider only the forward branches.
  if ((SrcMBB.getNumber() >= TrueMBB->getNumber()) ||
      mustRetainExeczBranch(*FalseMBB, *TrueMBB))
    return false;

  LLVM_DEBUG(dbgs() << "Removing the execz branch: " << MI);
  MI.eraseFromParent();
  SrcMBB.removeSuccessor(TrueMBB);

  return true;
}

bool SIRemoveShortExecBranches::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MF.RenumberBlocks();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();
    if (MBBI == MBB.end())
      continue;

    MachineInstr &MI = *MBBI;
    switch (MI.getOpcode()) {
    case AMDGPU::S_CBRANCH_EXECZ:
      Changed = removeExeczBranch(MI, MBB);
      break;
    default:
      break;
    }
  }

  return Changed;
}
