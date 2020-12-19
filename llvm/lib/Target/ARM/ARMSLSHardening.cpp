//===- ARMSLSHardening.cpp - Harden Straight Line Missspeculation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass to insert code to mitigate against side channel
// vulnerabilities that may happen under straight line miss-speculation.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/DebugLoc.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "arm-sls-hardening"

#define ARM_SLS_HARDENING_NAME "ARM sls hardening pass"

namespace {

class ARMSLSHardening : public MachineFunctionPass {
public:
  const TargetInstrInfo *TII;
  const ARMSubtarget *ST;

  static char ID;

  ARMSLSHardening() : MachineFunctionPass(ID) {
    initializeARMSLSHardeningPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return ARM_SLS_HARDENING_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool hardenReturnsAndBRs(MachineBasicBlock &MBB) const;
};

} // end anonymous namespace

char ARMSLSHardening::ID = 0;

INITIALIZE_PASS(ARMSLSHardening, "arm-sls-hardening",
                ARM_SLS_HARDENING_NAME, false, false)

static void insertSpeculationBarrier(const ARMSubtarget *ST,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     DebugLoc DL,
                                     bool AlwaysUseISBDSB = false) {
  assert(MBBI != MBB.begin() &&
         "Must not insert SpeculationBarrierEndBB as only instruction in MBB.");
  assert(std::prev(MBBI)->isBarrier() &&
         "SpeculationBarrierEndBB must only follow unconditional control flow "
         "instructions.");
  assert(std::prev(MBBI)->isTerminator() &&
         "SpeculationBarrierEndBB must only follow terminators.");
  const TargetInstrInfo *TII = ST->getInstrInfo();
  assert(ST->hasDataBarrier() || ST->hasSB());
  bool ProduceSB = ST->hasSB() && !AlwaysUseISBDSB;
  unsigned BarrierOpc =
      ProduceSB ? (ST->isThumb() ? ARM::t2SpeculationBarrierSBEndBB
                                 : ARM::SpeculationBarrierSBEndBB)
                : (ST->isThumb() ? ARM::t2SpeculationBarrierISBDSBEndBB
                                 : ARM::SpeculationBarrierISBDSBEndBB);
  if (MBBI == MBB.end() || !isSpeculationBarrierEndBBOpcode(MBBI->getOpcode()))
    BuildMI(MBB, MBBI, DL, TII->get(BarrierOpc));
}

bool ARMSLSHardening::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<ARMSubtarget>();
  TII = MF.getSubtarget().getInstrInfo();

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= hardenReturnsAndBRs(MBB);

  return Modified;
}

bool ARMSLSHardening::hardenReturnsAndBRs(MachineBasicBlock &MBB) const {
  if (!ST->hardenSlsRetBr())
    return false;
  assert(!ST->isThumb1Only());
  bool Modified = false;
  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator(), E = MBB.end();
  MachineBasicBlock::iterator NextMBBI;
  for (; MBBI != E; MBBI = NextMBBI) {
    MachineInstr &MI = *MBBI;
    NextMBBI = std::next(MBBI);
    if (isIndirectControlFlowNotComingBack(MI)) {
      assert(MI.isTerminator());
      assert(!TII->isPredicated(MI));
      insertSpeculationBarrier(ST, MBB, std::next(MBBI), MI.getDebugLoc());
      Modified = true;
    }
  }
  return Modified;
}

FunctionPass *llvm::createARMSLSHardeningPass() {
  return new ARMSLSHardening();
}
