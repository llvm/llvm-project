//===- PPCInstructionSelector.cpp --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the InstructionSelector class for
/// PowerPC.
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCInstrInfo.h"
#include "PPCRegisterBankInfo.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelectorImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-gisel"

using namespace llvm;

namespace {

#define GET_GLOBALISEL_PREDICATE_BITSET
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATE_BITSET

class PPCInstructionSelector : public InstructionSelector {
public:
  PPCInstructionSelector(const PPCTargetMachine &TM, const PPCSubtarget &STI,
                         const PPCRegisterBankInfo &RBI);

  bool select(MachineInstr &I) override;
  static const char *getName() { return DEBUG_TYPE; }

private:
  /// tblgen generated 'select' implementation that is used as the initial
  /// selector for the patterns that do not require complex C++.
  bool selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const;

  bool selectFPToInt(MachineInstr &I, MachineBasicBlock &MBB,
                  MachineRegisterInfo &MRI) const;
  bool selectIntToFP(MachineInstr &I, MachineBasicBlock &MBB,
                  MachineRegisterInfo &MRI) const;

  const PPCSubtarget &STI;
  const PPCInstrInfo &TII;
  const PPCRegisterInfo &TRI;
  const PPCRegisterBankInfo &RBI;

#define GET_GLOBALISEL_PREDICATES_DECL
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_DECL

#define GET_GLOBALISEL_TEMPORARIES_DECL
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_DECL
};

} // end anonymous namespace

#define GET_GLOBALISEL_IMPL
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_IMPL

PPCInstructionSelector::PPCInstructionSelector(const PPCTargetMachine &TM,
                                               const PPCSubtarget &STI,
                                               const PPCRegisterBankInfo &RBI)
    : STI(STI), TII(*STI.getInstrInfo()), TRI(*STI.getRegisterInfo()), RBI(RBI),
#define GET_GLOBALISEL_PREDICATES_INIT
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_INIT
#define GET_GLOBALISEL_TEMPORARIES_INIT
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_INIT
{
}

static const TargetRegisterClass *getRegClass(LLT Ty, const RegisterBank *RB) {
  if (RB->getID() == PPC::GPRRegBankID) {
    if (Ty.getSizeInBits() == 64)
      return &PPC::G8RCRegClass;
  }
  if (RB->getID() == PPC::FPRRegBankID) {
    if (Ty.getSizeInBits() == 32)
      return &PPC::F4RCRegClass;
    if (Ty.getSizeInBits() == 64)
      return &PPC::F8RCRegClass;
  }

  llvm_unreachable("Unknown RegBank!");
}

static bool selectCopy(MachineInstr &I, const TargetInstrInfo &TII,
                       MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
                       const RegisterBankInfo &RBI) {
  Register DstReg = I.getOperand(0).getReg();

  if (DstReg.isPhysical())
    return true;

  const RegisterBank *DstRegBank = RBI.getRegBank(DstReg, MRI, TRI);
  const TargetRegisterClass *DstRC =
      getRegClass(MRI.getType(DstReg), DstRegBank);

  // No need to constrain SrcReg. It will get constrained when we hit another of
  // its use or its defs.
  // Copies do not have constraints.
  if (!RBI.constrainGenericRegister(DstReg, *DstRC, MRI)) {
    LLVM_DEBUG(dbgs() << "Failed to constrain " << TII.getName(I.getOpcode())
                      << " operand\n");
    return false;
  }

  return true;
}

bool PPCInstructionSelector::selectIntToFP(MachineInstr &I,
                                           MachineBasicBlock &MBB,
                                           MachineRegisterInfo &MRI) const {
  if (!STI.hasDirectMove() || !STI.isPPC64() || !STI.hasFPCVT())
    return false;

  const DebugLoc &DbgLoc = I.getDebugLoc();
  const Register DstReg = I.getOperand(0).getReg();
  const Register SrcReg = I.getOperand(1).getReg();

  Register MoveReg = MRI.createVirtualRegister(&PPC::VSFRCRegClass);

  // For now, only handle the case for 64 bit integer.
  BuildMI(MBB, I, DbgLoc, TII.get(PPC::MTVSRD), MoveReg).addReg(SrcReg);

  bool IsSingle = MRI.getType(DstReg).getSizeInBits() == 32;
  bool IsSigned = I.getOpcode() == TargetOpcode::G_SITOFP;
  unsigned ConvOp = IsSingle ? (IsSigned ? PPC::XSCVSXDSP : PPC::XSCVUXDSP)
                             : (IsSigned ? PPC::XSCVSXDDP : PPC::XSCVUXDDP);

  MachineInstr *MI =
      BuildMI(MBB, I, DbgLoc, TII.get(ConvOp), DstReg).addReg(MoveReg);

  I.eraseFromParent();
  return constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
}

bool PPCInstructionSelector::selectFPToInt(MachineInstr &I,
                                           MachineBasicBlock &MBB,
                                           MachineRegisterInfo &MRI) const {
  if (!STI.hasDirectMove() || !STI.isPPC64() || !STI.hasFPCVT())
    return false;

  const DebugLoc &DbgLoc = I.getDebugLoc();
  const Register DstReg = I.getOperand(0).getReg();
  const Register SrcReg = I.getOperand(1).getReg();

  Register CopyReg = MRI.createVirtualRegister(&PPC::VSFRCRegClass);
  BuildMI(MBB, I, DbgLoc, TII.get(TargetOpcode::COPY), CopyReg).addReg(SrcReg);

  Register ConvReg = MRI.createVirtualRegister(&PPC::VSFRCRegClass);

  bool IsSigned = I.getOpcode() == TargetOpcode::G_FPTOSI;

  // single-precision is stored as double-precision on PPC in registers, so
  // always use double-precision convertions.
  unsigned ConvOp = IsSigned ? PPC::XSCVDPSXDS : PPC::XSCVDPUXDS;

  BuildMI(MBB, I, DbgLoc, TII.get(ConvOp), ConvReg).addReg(CopyReg);

  MachineInstr *MI =
      BuildMI(MBB, I, DbgLoc, TII.get(PPC::MFVSRD), DstReg).addReg(ConvReg);

  I.eraseFromParent();
  return constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
}

bool PPCInstructionSelector::select(MachineInstr &I) {
  auto &MBB = *I.getParent();
  auto &MF = *MBB.getParent();
  auto &MRI = MF.getRegInfo();

  if (!isPreISelGenericOpcode(I.getOpcode())) {
    if (I.isCopy())
      return selectCopy(I, TII, MRI, TRI, RBI);

    return true;
  }

  if (selectImpl(I, *CoverageInfo))
    return true;

  unsigned Opcode = I.getOpcode();

  switch (Opcode) {
  default:
    return false;
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
    return selectIntToFP(I, MBB, MRI);
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
    return selectFPToInt(I, MBB, MRI);
  }
  return false;
}

namespace llvm {
InstructionSelector *
createPPCInstructionSelector(const PPCTargetMachine &TM,
                             const PPCSubtarget &Subtarget,
                             const PPCRegisterBankInfo &RBI) {
  return new PPCInstructionSelector(TM, Subtarget, RBI);
}
} // end namespace llvm
