//===- AArch64MIPeepholeOpt.cpp - AArch64 MI peephole optimization pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs below peephole optimizations on MIR level.
//
// 1. MOVi32imm + ANDWrr ==> ANDWri + ANDWri
//    MOVi64imm + ANDXrr ==> ANDXri + ANDXri
//
// 2. MOVi32imm + ADDWrr ==> ADDWRi + ADDWRi
//    MOVi64imm + ADDXrr ==> ANDXri + ANDXri
//
// 3. MOVi32imm + SUBWrr ==> SUBWRi + SUBWRi
//    MOVi64imm + SUBXrr ==> SUBXri + SUBXri
//
//    The mov pseudo instruction could be expanded to multiple mov instructions
//    later. In this case, we could try to split the constant  operand of mov
//    instruction into two immediates which can be directly encoded into
//    *Wri/*Xri instructions. It makes two AND/ADD/SUB instructions instead of
//    multiple `mov` + `and/add/sub` instructions.
//===----------------------------------------------------------------------===//

#include "AArch64ExpandImm.h"
#include "AArch64InstrInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-mi-peephole-opt"

namespace {

struct AArch64MIPeepholeOpt : public MachineFunctionPass {
  static char ID;

  AArch64MIPeepholeOpt() : MachineFunctionPass(ID) {
    initializeAArch64MIPeepholeOptPass(*PassRegistry::getPassRegistry());
  }

  const AArch64InstrInfo *TII;
  MachineLoopInfo *MLI;
  MachineRegisterInfo *MRI;

  bool checkMovImmInstr(MachineInstr &MI, MachineInstr *&MovMI,
                        MachineInstr *&SubregToRegMI);

  template <typename T>
  bool visitADDSUB(MachineInstr &MI,
                   SmallSetVector<MachineInstr *, 8> &ToBeRemoved, bool IsAdd);

  template <typename T>
  bool visitAND(MachineInstr &MI,
                SmallSetVector<MachineInstr *, 8> &ToBeRemoved);
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 MI Peephole Optimization pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char AArch64MIPeepholeOpt::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(AArch64MIPeepholeOpt, "aarch64-mi-peephole-opt",
                "AArch64 MI Peephole Optimization", false, false)

template <typename T>
static bool splitBitmaskImm(T Imm, unsigned RegSize, T &Imm1Enc, T &Imm2Enc) {
  T UImm = static_cast<T>(Imm);
  if (AArch64_AM::isLogicalImmediate(UImm, RegSize))
    return false;

  // If this immediate can be handled by one instruction, do not split it.
  SmallVector<AArch64_IMM::ImmInsnModel, 4> Insn;
  AArch64_IMM::expandMOVImm(UImm, RegSize, Insn);
  if (Insn.size() == 1)
    return false;

  // The bitmask immediate consists of consecutive ones.  Let's say there is
  // constant 0b00000000001000000000010000000000 which does not consist of
  // consecutive ones. We can split it in to two bitmask immediate like
  // 0b00000000001111111111110000000000 and 0b11111111111000000000011111111111.
  // If we do AND with these two bitmask immediate, we can see original one.
  unsigned LowestBitSet = countTrailingZeros(UImm);
  unsigned HighestBitSet = Log2_64(UImm);

  // Create a mask which is filled with one from the position of lowest bit set
  // to the position of highest bit set.
  T NewImm1 = (static_cast<T>(2) << HighestBitSet) -
              (static_cast<T>(1) << LowestBitSet);
  // Create a mask which is filled with one outside the position of lowest bit
  // set and the position of highest bit set.
  T NewImm2 = UImm | ~NewImm1;

  // If the split value is not valid bitmask immediate, do not split this
  // constant.
  if (!AArch64_AM::isLogicalImmediate(NewImm2, RegSize))
    return false;

  Imm1Enc = AArch64_AM::encodeLogicalImmediate(NewImm1, RegSize);
  Imm2Enc = AArch64_AM::encodeLogicalImmediate(NewImm2, RegSize);
  return true;
}

template <typename T>
bool AArch64MIPeepholeOpt::visitAND(
    MachineInstr &MI, SmallSetVector<MachineInstr *, 8> &ToBeRemoved) {
  // Try below transformation.
  //
  // MOVi32imm + ANDWrr ==> ANDWri + ANDWri
  // MOVi64imm + ANDXrr ==> ANDXri + ANDXri
  //
  // The mov pseudo instruction could be expanded to multiple mov instructions
  // later. Let's try to split the constant operand of mov instruction into two
  // bitmask immediates. It makes only two AND instructions intead of multiple
  // mov + and instructions.

  unsigned RegSize = sizeof(T) * 8;
  assert((RegSize == 32 || RegSize == 64) &&
         "Invalid RegSize for AND bitmask peephole optimization");

  // Perform several essential checks against current MI.
  MachineInstr *MovMI, *SubregToRegMI;
  if (!checkMovImmInstr(MI, MovMI, SubregToRegMI))
    return false;

  // Split the bitmask immediate into two.
  T UImm = static_cast<T>(MovMI->getOperand(1).getImm());
  // For the 32 bit form of instruction, the upper 32 bits of the destination
  // register are set to zero. If there is SUBREG_TO_REG, set the upper 32 bits
  // of UImm to zero.
  if (SubregToRegMI)
    UImm &= 0xFFFFFFFF;
  T Imm1Enc;
  T Imm2Enc;
  if (!splitBitmaskImm(UImm, RegSize, Imm1Enc, Imm2Enc))
    return false;

  // Create new AND MIs.
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock *MBB = MI.getParent();
  const TargetRegisterClass *ANDImmRC =
      (RegSize == 32) ? &AArch64::GPR32spRegClass : &AArch64::GPR64spRegClass;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  Register NewTmpReg = MRI->createVirtualRegister(ANDImmRC);
  unsigned Opcode = (RegSize == 32) ? AArch64::ANDWri : AArch64::ANDXri;

  MRI->constrainRegClass(NewTmpReg, MRI->getRegClass(SrcReg));
  BuildMI(*MBB, MI, DL, TII->get(Opcode), NewTmpReg)
      .addReg(SrcReg)
      .addImm(Imm1Enc);

  MRI->constrainRegClass(DstReg, ANDImmRC);
  BuildMI(*MBB, MI, DL, TII->get(Opcode), DstReg)
      .addReg(NewTmpReg)
      .addImm(Imm2Enc);

  ToBeRemoved.insert(&MI);
  if (SubregToRegMI)
    ToBeRemoved.insert(SubregToRegMI);
  ToBeRemoved.insert(MovMI);

  return true;
}

template <typename T>
static bool splitAddSubImm(T Imm, unsigned RegSize, T &Imm0, T &Imm1) {
  // The immediate must be in the form of ((imm0 << 12) + imm1), in which both
  // imm0 and imm1 are non-zero 12-bit unsigned int.
  if ((Imm & 0xfff000) == 0 || (Imm & 0xfff) == 0 ||
      (Imm & ~static_cast<T>(0xffffff)) != 0)
    return false;

  // The immediate can not be composed via a single instruction.
  SmallVector<AArch64_IMM::ImmInsnModel, 4> Insn;
  AArch64_IMM::expandMOVImm(Imm, RegSize, Insn);
  if (Insn.size() == 1)
    return false;

  // Split Imm into (Imm0 << 12) + Imm1;
  Imm0 = (Imm >> 12) & 0xfff;
  Imm1 = Imm & 0xfff;
  return true;
}

template <typename T>
bool AArch64MIPeepholeOpt::visitADDSUB(
    MachineInstr &MI, SmallSetVector<MachineInstr *, 8> &ToBeRemoved,
    bool IsAdd) {
  // Try below transformation.
  //
  // MOVi32imm + ADDWrr ==> ANDWri + ANDWri
  // MOVi64imm + ADDXrr ==> ANDXri + ANDXri
  //
  // MOVi32imm + SUBWrr ==> SUBWri + SUBWri
  // MOVi64imm + SUBXrr ==> SUBXri + SUBXri
  //
  // The mov pseudo instruction could be expanded to multiple mov instructions
  // later. Let's try to split the constant operand of mov instruction into two
  // legal add/sub immediates. It makes only two ADD/SUB instructions intead of
  // multiple `mov` + `and/sub` instructions.

  unsigned RegSize = sizeof(T) * 8;
  assert((RegSize == 32 || RegSize == 64) &&
         "Invalid RegSize for legal add/sub immediate peephole optimization");

  // Perform several essential checks against current MI.
  MachineInstr *MovMI, *SubregToRegMI;
  if (!checkMovImmInstr(MI, MovMI, SubregToRegMI))
    return false;

  // Split the immediate to Imm0 and Imm1, and calculate the Opcode.
  T Imm = static_cast<T>(MovMI->getOperand(1).getImm()), Imm0, Imm1;
  unsigned Opcode;
  if (splitAddSubImm(Imm, RegSize, Imm0, Imm1)) {
    if (IsAdd)
      Opcode = RegSize == 32 ? AArch64::ADDWri : AArch64::ADDXri;
    else
      Opcode = RegSize == 32 ? AArch64::SUBWri : AArch64::SUBXri;
  } else if (splitAddSubImm(-Imm, RegSize, Imm0, Imm1)) {
    if (IsAdd)
      Opcode = RegSize == 32 ? AArch64::SUBWri : AArch64::SUBXri;
    else
      Opcode = RegSize == 32 ? AArch64::ADDWri : AArch64::ADDXri;
  } else {
    return false;
  }

  // Create new ADD/SUB MIs.
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock *MBB = MI.getParent();
  const TargetRegisterClass *RC =
      (RegSize == 32) ? &AArch64::GPR32spRegClass : &AArch64::GPR64spRegClass;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  Register TmpReg = MRI->createVirtualRegister(RC);

  MRI->constrainRegClass(SrcReg, RC);
  BuildMI(*MBB, MI, DL, TII->get(Opcode), TmpReg)
      .addReg(SrcReg)
      .addImm(Imm0)
      .addImm(12);

  MRI->constrainRegClass(DstReg, RC);
  BuildMI(*MBB, MI, DL, TII->get(Opcode), DstReg)
      .addReg(TmpReg)
      .addImm(Imm1)
      .addImm(0);

  // Record the MIs need to be removed.
  ToBeRemoved.insert(&MI);
  if (SubregToRegMI)
    ToBeRemoved.insert(SubregToRegMI);
  ToBeRemoved.insert(MovMI);

  return true;
}

// Checks if the corresponding MOV immediate instruction is applicable for
// this peephole optimization.
bool AArch64MIPeepholeOpt::checkMovImmInstr(MachineInstr &MI,
                                            MachineInstr *&MovMI,
                                            MachineInstr *&SubregToRegMI) {
  // Check whether current MI is in loop and is loop invariant.
  MachineBasicBlock *MBB = MI.getParent();
  MachineLoop *L = MLI->getLoopFor(MBB);
  if (L && !L->isLoopInvariant(MI))
    return false;

  // Check whether current MI's operand is MOV with immediate.
  MovMI = MRI->getUniqueVRegDef(MI.getOperand(2).getReg());
  SubregToRegMI = nullptr;
  // If it is SUBREG_TO_REG, check its operand.
  if (MovMI->getOpcode() == TargetOpcode::SUBREG_TO_REG) {
    SubregToRegMI = MovMI;
    MovMI = MRI->getUniqueVRegDef(MovMI->getOperand(2).getReg());
  }

  if (MovMI->getOpcode() != AArch64::MOVi32imm &&
      MovMI->getOpcode() != AArch64::MOVi64imm)
    return false;

  // If the MOV has multiple uses, do not split the immediate because it causes
  // more instructions.
  if (!MRI->hasOneUse(MovMI->getOperand(0).getReg()))
    return false;

  if (SubregToRegMI && !MRI->hasOneUse(SubregToRegMI->getOperand(0).getReg()))
    return false;

  // It is OK to perform this peephole optimization.
  return true;
}

bool AArch64MIPeepholeOpt::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  MLI = &getAnalysis<MachineLoopInfo>();
  MRI = &MF.getRegInfo();

  if (!MRI->isSSA())
    return false;

  bool Changed = false;
  SmallSetVector<MachineInstr *, 8> ToBeRemoved;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      default:
        break;
      case AArch64::ANDWrr:
        Changed = visitAND<uint32_t>(MI, ToBeRemoved);
        break;
      case AArch64::ANDXrr:
        Changed = visitAND<uint64_t>(MI, ToBeRemoved);
        break;
      case AArch64::ADDWrr:
        Changed = visitADDSUB<uint32_t>(MI, ToBeRemoved, true);
        break;
      case AArch64::SUBWrr:
        Changed = visitADDSUB<uint32_t>(MI, ToBeRemoved, false);
        break;
      case AArch64::ADDXrr:
        Changed = visitADDSUB<uint64_t>(MI, ToBeRemoved, true);
        break;
      case AArch64::SUBXrr:
        Changed = visitADDSUB<uint64_t>(MI, ToBeRemoved, false);
        break;
      }
    }
  }

  for (MachineInstr *MI : ToBeRemoved)
    MI->eraseFromParent();

  return Changed;
}

FunctionPass *llvm::createAArch64MIPeepholeOptPass() {
  return new AArch64MIPeepholeOpt();
}
