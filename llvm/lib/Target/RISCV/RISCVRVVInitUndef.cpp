//===- RISCVRVVInitUndef.cpp - Initialize undef vector value to pseudo ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that initializes undef vector value to
// temporary pseudo instruction and remove it in expandpseudo pass to prevent
// register allocation resulting in a constraint violated result for vector
// instruction.  It also rewrites the NoReg tied operand back to an
// IMPLICIT_DEF.
//
// RISC-V vector instruction has register overlapping constraint for certain
// instructions, and will cause illegal instruction trap if violated, we use
// early clobber to model this constraint, but it can't prevent register
// allocator allocated same or overlapped if the input register is undef value,
// so convert IMPLICIT_DEF to temporary pseudo instruction and remove it later
// could prevent that happen, it's not best way to resolve this, and it might
// change the order of program or increase the register pressure, so ideally we
// should model the constraint right, but before we model the constraint right,
// it's the only way to prevent that happen.
//
// When we enable the subregister liveness option, it will also trigger same
// issue due to the partial of register is undef. If we pseudoinit the whole
// register, then it will generate redundant COPY instruction. Currently, it
// will generate INSERT_SUBREG to make sure the whole register is occupied
// when program encounter operation that has early-clobber constraint.
//
//
// See also: https://github.com/llvm/llvm-project/issues/50157
//
// Additionally, this pass rewrites tied operands of vector instructions
// from NoReg to IMPLICIT_DEF.  (Not that this is a non-overlapping set of
// operands to the above.)  We use NoReg to side step a MachineCSE
// optimization quality problem but need to convert back before
// TwoAddressInstruction.  See pr64282 for context.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DetectDeadLanes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "riscv-init-undef"
#define RISCV_INIT_UNDEF_NAME "RISC-V init undef pass"

namespace {

class RISCVInitUndef : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const RISCVSubtarget *ST;
  const TargetRegisterInfo *TRI;

  // Newly added vregs, assumed to be fully rewritten
  SmallSet<Register, 8> NewRegs;
  SmallVector<MachineInstr *, 8> DeadInsts;

public:
  static char ID;

  RISCVInitUndef() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INIT_UNDEF_NAME; }

private:
  bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB,
                         const DeadLaneDetector &DLD);
  bool isVectorRegClass(const Register R);
  const TargetRegisterClass *
  getVRLargestSuperClass(const TargetRegisterClass *RC) const;
  bool handleSubReg(MachineFunction &MF, MachineInstr &MI,
                    const DeadLaneDetector &DLD);
  bool fixupIllOperand(MachineInstr *MI, MachineOperand &MO);
  bool handleReg(MachineInstr *MI);
};

} // end anonymous namespace

char RISCVInitUndef::ID = 0;
INITIALIZE_PASS(RISCVInitUndef, DEBUG_TYPE, RISCV_INIT_UNDEF_NAME, false, false)
char &llvm::RISCVInitUndefID = RISCVInitUndef::ID;

const TargetRegisterClass *
RISCVInitUndef::getVRLargestSuperClass(const TargetRegisterClass *RC) const {
  if (RISCV::VRM8RegClass.hasSubClassEq(RC))
    return &RISCV::VRM8RegClass;
  if (RISCV::VRM4RegClass.hasSubClassEq(RC))
    return &RISCV::VRM4RegClass;
  if (RISCV::VRM2RegClass.hasSubClassEq(RC))
    return &RISCV::VRM2RegClass;
  if (RISCV::VRRegClass.hasSubClassEq(RC))
    return &RISCV::VRRegClass;
  return RC;
}

bool RISCVInitUndef::isVectorRegClass(const Register R) {
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  return RISCV::VRRegClass.hasSubClassEq(RC) ||
         RISCV::VRM2RegClass.hasSubClassEq(RC) ||
         RISCV::VRM4RegClass.hasSubClassEq(RC) ||
         RISCV::VRM8RegClass.hasSubClassEq(RC);
}

static unsigned getUndefInitOpcode(unsigned RegClassID) {
  switch (RegClassID) {
  case RISCV::VRRegClassID:
    return RISCV::PseudoRVVInitUndefM1;
  case RISCV::VRM2RegClassID:
    return RISCV::PseudoRVVInitUndefM2;
  case RISCV::VRM4RegClassID:
    return RISCV::PseudoRVVInitUndefM4;
  case RISCV::VRM8RegClassID:
    return RISCV::PseudoRVVInitUndefM8;
  default:
    llvm_unreachable("Unexpected register class.");
  }
}

static bool isEarlyClobberMI(MachineInstr &MI) {
  return llvm::any_of(MI.defs(), [](const MachineOperand &DefMO) {
    return DefMO.isReg() && DefMO.isEarlyClobber();
  });
}

static bool findImplictDefMIFromReg(Register Reg, MachineRegisterInfo *MRI) {
  for (auto &DefMI : MRI->def_instructions(Reg)) {
    if (DefMI.getOpcode() == TargetOpcode::IMPLICIT_DEF)
      return true;
  }
  return false;
}

bool RISCVInitUndef::handleReg(MachineInstr *MI) {
  bool Changed = false;
  for (auto &UseMO : MI->uses()) {
    if (!UseMO.isReg())
      continue;
    if (UseMO.isTied())
      continue;
    if (!UseMO.getReg().isVirtual())
      continue;
    if (!isVectorRegClass(UseMO.getReg()))
      continue;

    if (UseMO.isUndef() || findImplictDefMIFromReg(UseMO.getReg(), MRI))
      Changed |= fixupIllOperand(MI, UseMO);
  }
  return Changed;
}

bool RISCVInitUndef::handleSubReg(MachineFunction &MF, MachineInstr &MI,
                                  const DeadLaneDetector &DLD) {
  bool Changed = false;

  for (MachineOperand &UseMO : MI.uses()) {
    if (!UseMO.isReg())
      continue;
    if (!UseMO.getReg().isVirtual())
      continue;
    if (UseMO.isTied())
      continue;

    Register Reg = UseMO.getReg();
    if (NewRegs.count(Reg))
      continue;
    DeadLaneDetector::VRegInfo Info =
        DLD.getVRegInfo(Register::virtReg2Index(Reg));

    if (Info.UsedLanes == Info.DefinedLanes)
      continue;

    const TargetRegisterClass *TargetRegClass =
        getVRLargestSuperClass(MRI->getRegClass(Reg));

    LaneBitmask NeedDef = Info.UsedLanes & ~Info.DefinedLanes;

    LLVM_DEBUG({
      dbgs() << "Instruction has undef subregister.\n";
      dbgs() << printReg(Reg, nullptr)
             << " Used: " << PrintLaneMask(Info.UsedLanes)
             << " Def: " << PrintLaneMask(Info.DefinedLanes)
             << " Need Def: " << PrintLaneMask(NeedDef) << "\n";
    });

    SmallVector<unsigned> SubRegIndexNeedInsert;
    TRI->getCoveringSubRegIndexes(*MRI, TargetRegClass, NeedDef,
                                  SubRegIndexNeedInsert);

    Register LatestReg = Reg;
    for (auto ind : SubRegIndexNeedInsert) {
      Changed = true;
      const TargetRegisterClass *SubRegClass =
          getVRLargestSuperClass(TRI->getSubRegisterClass(TargetRegClass, ind));
      Register TmpInitSubReg = MRI->createVirtualRegister(SubRegClass);
      BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(),
              TII->get(getUndefInitOpcode(SubRegClass->getID())),
              TmpInitSubReg);
      Register NewReg = MRI->createVirtualRegister(TargetRegClass);
      BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(),
              TII->get(TargetOpcode::INSERT_SUBREG), NewReg)
          .addReg(LatestReg)
          .addReg(TmpInitSubReg)
          .addImm(ind);
      LatestReg = NewReg;
    }

    UseMO.setReg(LatestReg);
  }

  return Changed;
}

bool RISCVInitUndef::fixupIllOperand(MachineInstr *MI, MachineOperand &MO) {

  LLVM_DEBUG(
      dbgs() << "Emitting PseudoRVVInitUndef for implicit vector register "
             << MO.getReg() << '\n');

  const TargetRegisterClass *TargetRegClass =
      getVRLargestSuperClass(MRI->getRegClass(MO.getReg()));
  unsigned Opcode = getUndefInitOpcode(TargetRegClass->getID());
  Register NewReg = MRI->createVirtualRegister(TargetRegClass);
  BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), TII->get(Opcode), NewReg);
  MO.setReg(NewReg);
  if (MO.isUndef())
    MO.setIsUndef(false);
  return true;
}

bool RISCVInitUndef::processBasicBlock(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       const DeadLaneDetector &DLD) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
    MachineInstr &MI = *I;

    // If we used NoReg to represent the passthru, switch this back to being
    // an IMPLICIT_DEF before TwoAddressInstructions.
    unsigned UseOpIdx;
    if (MI.getNumDefs() != 0 && MI.isRegTiedToUseOperand(0, &UseOpIdx)) {
      MachineOperand &UseMO = MI.getOperand(UseOpIdx);
      if (UseMO.getReg() == RISCV::NoRegister) {
        const TargetRegisterClass *RC =
          TII->getRegClass(MI.getDesc(), UseOpIdx, TRI, MF);
        Register NewDest = MRI->createVirtualRegister(RC);
        // We don't have a way to update dead lanes, so keep track of the
        // new register so that we avoid querying it later.
        NewRegs.insert(NewDest);
        BuildMI(MBB, I, I->getDebugLoc(),
                TII->get(TargetOpcode::IMPLICIT_DEF), NewDest);
        UseMO.setReg(NewDest);
        Changed = true;
      }
    }

    if (isEarlyClobberMI(MI)) {
      if (ST->enableSubRegLiveness())
        Changed |= handleSubReg(MF, MI, DLD);
      Changed |= handleReg(&MI);
    }
  }
  return Changed;
}

bool RISCVInitUndef::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  MRI = &MF.getRegInfo();
  TII = ST->getInstrInfo();
  TRI = MRI->getTargetRegisterInfo();

  bool Changed = false;
  DeadLaneDetector DLD(MRI, TRI);
  DLD.computeSubRegisterLaneBitInfo();

  for (MachineBasicBlock &BB : MF)
    Changed |= processBasicBlock(MF, BB, DLD);

  for (auto *DeadMI : DeadInsts)
    DeadMI->eraseFromParent();
  DeadInsts.clear();

  return Changed;
}

FunctionPass *llvm::createRISCVInitUndefPass() { return new RISCVInitUndef(); }
