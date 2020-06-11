//===- AArch64SLSHardening.cpp - Harden Straight Line Missspeculation -----===//
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

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "aarch64-sls-hardening"

#define AARCH64_SLS_HARDENING_NAME "AArch64 sls hardening pass"

namespace {

class AArch64SLSHardening : public MachineFunctionPass {
public:
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const AArch64Subtarget *ST;

  static char ID;

  AArch64SLSHardening() : MachineFunctionPass(ID) {
    initializeAArch64SLSHardeningPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return AARCH64_SLS_HARDENING_NAME; }

private:
  bool hardenReturnsAndBRs(MachineBasicBlock &MBB) const;
  void insertSpeculationBarrier(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                DebugLoc DL) const;
};

} // end anonymous namespace

char AArch64SLSHardening::ID = 0;

INITIALIZE_PASS(AArch64SLSHardening, "aarch64-sls-hardening",
                AARCH64_SLS_HARDENING_NAME, false, false)

void AArch64SLSHardening::insertSpeculationBarrier(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    DebugLoc DL) const {
  assert(MBBI != MBB.begin() &&
         "Must not insert SpeculationBarrierEndBB as only instruction in MBB.");
  assert(std::prev(MBBI)->isBarrier() &&
         "SpeculationBarrierEndBB must only follow unconditional control flow "
         "instructions.");
  assert(std::prev(MBBI)->isTerminator() &&
         "SpeculatoinBarrierEndBB must only follow terminators.");
  if (ST->hasSB())
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SpeculationBarrierSBEndBB));
  else
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SpeculationBarrierISBDSBEndBB));
}

bool AArch64SLSHardening::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<AArch64Subtarget>();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= hardenReturnsAndBRs(MBB);

  return Modified;
}

bool AArch64SLSHardening::hardenReturnsAndBRs(MachineBasicBlock &MBB) const {
  if (!ST->hardenSlsRetBr())
    return false;
  bool Modified = false;
  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator(), E = MBB.end();
  MachineBasicBlock::iterator NextMBBI;
  for (; MBBI != E; MBBI = NextMBBI) {
    MachineInstr &MI = *MBBI;
    NextMBBI = std::next(MBBI);
    if (MI.isReturn() || isIndirectBranchOpcode(MI.getOpcode())) {
      assert(MI.isTerminator());
      insertSpeculationBarrier(MBB, std::next(MBBI), MI.getDebugLoc());
      Modified = true;
    }
  }
  return Modified;
}

FunctionPass *llvm::createAArch64SLSHardeningPass() {
  return new AArch64SLSHardening();
}
