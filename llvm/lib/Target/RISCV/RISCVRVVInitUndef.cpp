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
// instruction.
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
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
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

public:
  static char ID;

  RISCVInitUndef() : MachineFunctionPass(ID) {
    initializeRISCVInitUndefPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INIT_UNDEF_NAME; }

private:
  bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB,
                         const DeadLaneDetector &DLD);
  bool handleImplicitDef(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &Inst);
  bool isVectorRegClass(const Register R);
  const TargetRegisterClass *
  getVRLargestSuperClass(const TargetRegisterClass *RC) const;
  bool handleSubReg(MachineFunction &MF, MachineInstr &MI,
                    const DeadLaneDetector &DLD);
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

bool RISCVInitUndef::handleImplicitDef(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator &Inst) {
  const TargetRegisterInfo &TRI =
      *MBB.getParent()->getSubtarget().getRegisterInfo();

  assert(Inst->getOpcode() == TargetOpcode::IMPLICIT_DEF);

  Register Reg = Inst->getOperand(0).getReg();
  if (!Reg.isVirtual())
    return false;

  bool NeedPseudoInit = false;
  SmallVector<MachineOperand *, 1> UseMOs;
  for (MachineOperand &MO : MRI->use_nodbg_operands(Reg)) {
    MachineInstr *UserMI = MO.getParent();

    bool HasEarlyClobber = false;
    bool TiedToDef = false;
    for (MachineOperand &UserMO : UserMI->operands()) {
      if (!UserMO.isReg())
        continue;
      if (UserMO.isEarlyClobber())
        HasEarlyClobber = true;
      if (UserMO.isUse() && UserMO.isTied() &&
          TRI.regsOverlap(UserMO.getReg(), Reg))
        TiedToDef = true;
    }
    if (HasEarlyClobber && !TiedToDef) {
      NeedPseudoInit = true;
      UseMOs.push_back(&MO);
    }
  }

  if (!NeedPseudoInit)
    return false;

  LLVM_DEBUG(
      dbgs() << "Emitting PseudoRVVInitUndef for implicit vector register "
             << Reg << '\n');

  unsigned RegClassID = getVRLargestSuperClass(MRI->getRegClass(Reg))->getID();
  unsigned Opcode = getUndefInitOpcode(RegClassID);

  BuildMI(MBB, Inst, Inst->getDebugLoc(), TII->get(Opcode), Reg);

  Inst = MBB.erase(Inst);

  for (auto MO : UseMOs)
    MO->setIsUndef(false);

  return true;
}

static bool isEarlyClobberMI(MachineInstr &MI) {
  return llvm::any_of(MI.defs(), [](const MachineOperand &DefMO) {
    return DefMO.isReg() && DefMO.isEarlyClobber();
  });
}

bool RISCVInitUndef::handleSubReg(MachineFunction &MF, MachineInstr &MI,
                                  const DeadLaneDetector &DLD) {
  bool Changed = false;

  for (MachineOperand &UseMO : MI.uses()) {
    if (!UseMO.isReg())
      continue;
    if (!UseMO.getReg().isVirtual())
      continue;

    Register Reg = UseMO.getReg();
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

bool RISCVInitUndef::processBasicBlock(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       const DeadLaneDetector &DLD) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
    MachineInstr &MI = *I;
    if (ST->enableSubRegLiveness() && isEarlyClobberMI(MI))
      Changed |= handleSubReg(MF, MI, DLD);
    if (MI.isImplicitDef()) {
      auto DstReg = MI.getOperand(0).getReg();
      if (isVectorRegClass(DstReg))
        Changed |= handleImplicitDef(MBB, I);
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

  return Changed;
}

FunctionPass *llvm::createRISCVInitUndefPass() { return new RISCVInitUndef(); }
