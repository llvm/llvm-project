//===-AArch64BiasedBranchOpt.cpp - Optimize strongly biased branch --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Huawei Technologies Co., Ltd. All rights reserved.
//
//===---------------------------------------------------------------------===//
//
// Optimize strongly biased conditional branches by turning them into their
// Hinted Conditional Branch (HBC) forms, i.e.:
// 1) b.cond => bc.cond
// 2) cbz/cbnz => cmp + bc.cond
//
// The branch bias is derived from the machine basic block edge probabilities
// (MachineBranchProbabilityInfo), which in turn come from the branch weights
// annotated by the SampleProfileLoader (CSSPGO). A branch is converted only
// when max(P(taken), 1 - P(taken)) >= -biased-branch-threshold, so unbiased
// branches (which would not benefit from the HBC hint) are left unchanged.
//
//===---------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>

#define DEBUG_TYPE "aarch64-biased-branch-opt"

using namespace llvm;

static cl::opt<bool> EnableHBC("enable-hbc", cl::init(false),
                               cl::desc("Enable HBC"), cl::Hidden);

static cl::opt<double> BiasedBranchThreshold(
    "biased-branch-threshold", cl::init(0.9), cl::value_desc("ratio"),
    cl::desc("Convert a conditional branch to HBC only when its taken "
             "probability bias (max(P(taken), 1-P(taken))) is at least this "
             "value"),
    cl::Hidden);

STATISTIC(NumBiasedBranchConverted, "Number of biased branches converted");
STATISTIC(NumBiasedBranchConvertedBCond,
          "Number of biased branches converted that are b.cond");
STATISTIC(NumBiasedBranchConvertedCBZ,
          "Number of biased branches converted that are cbz/cbnz");

namespace {

class AArch64BiasedBranchOpt : public MachineFunctionPass {
public:
  static char ID;
  AArch64BiasedBranchOpt();
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return "AArch64 Biased Branch Optimization";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  }

private:
  bool transformBcond(MachineInstr &MI, MachineBasicBlock &MBB,
                      MachineBasicBlock *TargetMBB, const DebugLoc &DL,
                      const TargetInstrInfo *TII);
  bool transformCBZ(MachineInstr &MI, MachineBasicBlock &MBB,
                    MachineBasicBlock *TargetMBB, const DebugLoc &DL,
                    const TargetInstrInfo *TII);
};
} // end anonymous namespace

char AArch64BiasedBranchOpt::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64BiasedBranchOpt, "aarch64-biased-branch-opt",
                      "AArch64 Biased Branch Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(AArch64BiasedBranchOpt, "aarch64-biased-branch-opt",
                    "AArch64 Biased Branch Optimization", false, false)

AArch64BiasedBranchOpt::AArch64BiasedBranchOpt() : MachineFunctionPass(ID) {}

bool AArch64BiasedBranchOpt::transformBcond(MachineInstr &MI,
                                            MachineBasicBlock &MBB,
                                            MachineBasicBlock *TargetMBB,
                                            const DebugLoc &DL,
                                            const TargetInstrInfo *TII) {
  assert(MI.getOpcode() == AArch64::Bcc && "opcode should be b.cond!");
  AArch64CC::CondCode CC = (AArch64CC::CondCode)MI.getOperand(0).getImm();
  MachineInstr *NewMI = BuildMI(MBB, MI, DL, TII->get(AArch64::BCcc))
                            .addImm(CC)
                            .addMBB(TargetMBB);
  unsigned ImplicitOpsIdx = 2;
  MachineFunction *MF = MBB.getParent();
  NewMI->removeOperand(ImplicitOpsIdx);
  NewMI->copyImplicitOps(*MF, MI);
  LLVM_DEBUG(dbgs() << ": B.cond -> BC.Cond New MI: ");
  LLVM_DEBUG(NewMI->dump());
  MI.eraseFromParent();
  ++NumBiasedBranchConvertedBCond;
  ++NumBiasedBranchConverted;
  return true;
}

bool AArch64BiasedBranchOpt::transformCBZ(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          MachineBasicBlock *TargetMBB,
                                          const DebugLoc &DL,
                                          const TargetInstrInfo *TII) {
  assert((MI.getOpcode() == AArch64::CBZW || MI.getOpcode() == AArch64::CBZX ||
          MI.getOpcode() == AArch64::CBNZW ||
          MI.getOpcode() == AArch64::CBNZX) &&
         "opcode should be cbz/cbnz!");

  // Make sure current change won't clobber previous NZCV
  // that has been defined but has not been used
  const TargetRegisterInfo *TRI =
      MBB.getParent()->getSubtarget().getRegisterInfo();
  LivePhysRegs LiveRegs(*TRI);
  LiveRegs.addLiveOuts(MBB);
  for (MachineInstr &PrevMI : reverse(MBB)) {
    if (&PrevMI == &MI)
      break;
    LiveRegs.stepBackward(PrevMI);
  }
  if (LiveRegs.contains(AArch64::NZCV))
    return false;

  Register RegCmp = MI.getOperand(0).getReg();
  RegState NewRegState = getRegState(MI.getOperand(0));
  bool isGPR64 = AArch64::GPR64RegClass.contains(RegCmp);
  bool isGPR32 = AArch64::GPR32RegClass.contains(RegCmp);
  if (!isGPR32 && !isGPR64)
    return false;

  unsigned CmpOp = isGPR32 ? AArch64::SUBSWri : AArch64::SUBSXri;
  unsigned ZReg = isGPR32 ? AArch64::WZR : AArch64::XZR;
  MachineInstr *CmpInstr = BuildMI(MBB, MI, DL, TII->get(CmpOp), ZReg)
                               .addReg(RegCmp, NewRegState)
                               .addImm(0)
                               .addImm(0);
  AArch64CC::CondCode CC =
      (MI.getOpcode() == AArch64::CBZW || MI.getOpcode() == AArch64::CBZX)
          ? AArch64CC::EQ
          : AArch64CC::NE;
  MachineInstr *NewMI = BuildMI(MBB, MI, DL, TII->get(AArch64::BCcc))
                            .addImm(CC)
                            .addMBB(TargetMBB);

  LLVM_DEBUG(dbgs() << ": CBZ -> CMP New MI: ");
  LLVM_DEBUG(CmpInstr->dump());
  LLVM_DEBUG(dbgs() << ": CBZ -> BC.cond New MI: ");
  LLVM_DEBUG(NewMI->dump());
  MI.eraseFromParent();
  ++NumBiasedBranchConverted;
  ++NumBiasedBranchConvertedCBZ;
  return true;
}

bool AArch64BiasedBranchOpt::runOnMachineFunction(MachineFunction &MF) {
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  if (!EnableHBC)
    return false;

  if (!MF.getFunction().hasProfileData())
    return false;
  else
    LLVM_DEBUG(dbgs() << "Function Name: " << MF.getFunction().getName()
                      << "\n");
  if (!Subtarget.hasHBC())
    return false;

  auto &MBPI = getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();

  bool Changed = false;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (auto &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB.terminators())) {
      switch (MI.getOpcode()) {
      case AArch64::Bcc:
      case AArch64::CBZW:
      case AArch64::CBZX:
      case AArch64::CBNZW:
      case AArch64::CBNZX:
        break;
      default:
        continue;
      }

      const BasicBlock *BB = MBB.getBasicBlock();
      if (!BB)
        continue;
      const Instruction *TI = BB->getTerminator();
      if (!TI || TI->getNumSuccessors() < 2)
        continue;
      if (!hasBranchWeightMD(*TI))
        continue;

      MachineBasicBlock *TargetMBB = TII->getBranchDestBlock(MI);
      BranchProbability TakenProb = MBPI.getEdgeProbability(&MBB, TargetMBB);
      double PTaken =
          (double)TakenProb.getNumerator() / TakenProb.getDenominator();
      double Bias = std::max(PTaken, 1.0 - PTaken);
      LLVM_DEBUG(dbgs() << ": Bias=" << Bias << " P(taken)=" << PTaken << "\n");
      if (Bias < BiasedBranchThreshold)
        continue;

      LLVM_DEBUG(dbgs() << ": Bias=" << Bias << " P(taken)=" << PTaken
                        << " -> HBC\n");
      LLVM_DEBUG(dbgs() << ": Old MI: ");
      LLVM_DEBUG(MI.dump());
      const DebugLoc &DL = MI.getDebugLoc();
      if (MI.getOpcode() == AArch64::Bcc)
        Changed |= transformBcond(MI, MBB, TargetMBB, DL, TII);
      else
        Changed |= transformCBZ(MI, MBB, TargetMBB, DL, TII);
    }
  }
  return Changed;
}

FunctionPass *llvm::createAArch64BiasedBranchOptPass() {
  return new AArch64BiasedBranchOpt();
}
