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
    : TII(*STI.getInstrInfo()), TRI(*STI.getRegisterInfo()), RBI(RBI),
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
