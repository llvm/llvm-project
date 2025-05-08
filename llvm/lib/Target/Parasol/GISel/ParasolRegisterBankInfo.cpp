//===- ParasolRegisterBankInfo.cpp ----------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the RegisterBankInfo class for Parasol
//===----------------------------------------------------------------------===//

#include "ParasolRegisterBankInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBank.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Threading.h"
#include <limits>

#define DEBUG_TYPE "parasol-reg-bank-info"

#define GET_TARGET_REGBANK_IMPL
#include "ParasolGenRegisterBank.inc"

// This file will be TableGen'ed at some point.
#include "ParasolGenRegisterBankInfo.def"

using namespace llvm;

const RegisterBankInfo::InstructionMapping &
ParasolRegisterBankInfo::getSameKindOfOperandsMapping(
    const MachineInstr &MI) const {
  const unsigned Opc = MI.getOpcode();
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  unsigned NumOperands = MI.getNumOperands();
  assert(NumOperands <= 3 &&
         "This code is for instructions with 3 or less operands");

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = Ty.getSizeInBits();

  return getInstructionMapping(
      DefaultMappingID, 1, Parasol::getValueMapping(Parasol::IRRegBankID, Size),
      NumOperands);
}

ParasolRegisterBankInfo::ParasolRegisterBankInfo(const TargetRegisterInfo &TRI)
    : ParasolGenRegisterBankInfo() {
  static llvm::once_flag InitializeRegisterBankFlag;

  static auto InitializeRegisterBankOnce = [&]() {};

  llvm::call_once(InitializeRegisterBankFlag, InitializeRegisterBankOnce);
}

const RegisterBank &
ParasolRegisterBankInfo::getRegBankFromRegClass(const TargetRegisterClass &RC,
                                                LLT) const {
  switch (RC.getID()) {
  case Parasol::IRRegClassID:
    return getRegBank(Parasol::IRRegBankID);
  case Parasol::PRRegClassID:
    return getRegBank(Parasol::PRRegBankID);
  default:
    llvm_unreachable("Register class not supported");
  }
}

unsigned ParasolRegisterBankInfo::copyCost(const RegisterBank &A,
                                           const RegisterBank &B,
                                           TypeSize Size) const {
  if (A.getID() == B.getID())
    return 0;

  // We really don't want to copy between different banks. Return a huge cost.
  return 4;
}

const RegisterBankInfo::InstructionMapping &
ParasolRegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
  const unsigned Opc = MI.getOpcode();

  // Try getting the default mapping.
  const RegisterBankInfo::InstructionMapping &Mapping = getInstrMappingImpl(MI);
  if (Mapping.isValid()) {
    return Mapping;
  }

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

  // SUNSCREEN TODO: Look, a specific pointer add!
  //   case TargetOpcode::G_PTR_ADD:

  switch (Opc) {
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
    return getSameKindOfOperandsMapping(MI);
  case TargetOpcode::COPY: {
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    // Check if one of the register is not a generic register.
    if ((DstReg.isPhysical() || !MRI.getType(DstReg).isValid()) ||
        (SrcReg.isPhysical() || !MRI.getType(SrcReg).isValid())) {
      const RegisterBank *DstRB = getRegBank(DstReg, MRI, TRI);
      const RegisterBank *SrcRB = getRegBank(SrcReg, MRI, TRI);
      if (!DstRB)
        DstRB = SrcRB;
      else if (!SrcRB)
        SrcRB = DstRB;
      // If both RB are null that means both registers are generic.
      // We shouldn't be here.
      assert(DstRB && SrcRB && "Both RegBank were nullptr");
      unsigned Size = getSizeInBits(DstReg, MRI, TRI);
      return getInstructionMapping(
          DefaultMappingID, copyCost(*DstRB, *SrcRB, TypeSize::getFixed(Size)),
          Parasol::getValueMapping(DstRB->getID(), Size),
          // We only care about the mapping of the destination.
          /*NumOperands*/ 1);
    }
    // Both registers are generic, use G_BITCAST.
    [[fallthrough]];
  }
  case TargetOpcode::G_BITCAST: {
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
    unsigned Size = DstTy.getSizeInBits();
    bool isPointer = DstTy.isPointer();
    const RegisterBank &DstRB =
        isPointer ? Parasol::PRRegBank : Parasol::IRRegBank;
    const RegisterBank &SrcRB =
        isPointer ? Parasol::PRRegBank : Parasol::IRRegBank;
    return getInstructionMapping(
        DefaultMappingID, copyCost(DstRB, SrcRB, TypeSize::getFixed(Size)),
        Parasol::getValueMapping(DstRB.getID(), Size),
        // We only care about the mapping of the destination for COPY.
        /*NumOperands*/ Opc == TargetOpcode::G_BITCAST ? 2 : 1);
  }
  default:
    break;
  }

  unsigned NumOperands = MI.getNumOperands();

  // Track the size and bank of each register.  We don't do partial mappings.
  SmallVector<unsigned, 4> OpSize(NumOperands);
  SmallVector<unsigned, 4> OpRegBankIdx(NumOperands);

  for (unsigned Idx = 0; Idx < NumOperands; ++Idx) {
    auto &MO = MI.getOperand(Idx);
    if (!MO.isReg() || !MO.getReg())
      continue;

    LLT Ty = MRI.getType(MO.getReg());
    if (!Ty.isValid())
      continue;
    OpSize[Idx] = Ty.getSizeInBits();

    OpRegBankIdx[Idx] =
        Ty.isPointer() ? Parasol::PRRegBankID : Parasol::IRRegBankID;
  }

  unsigned Cost = 1;

  // Finally construct the computed mapping.
  SmallVector<const ValueMapping *, 8> OpdsMapping(NumOperands);
  for (unsigned Idx = 0; Idx < NumOperands; ++Idx) {
    if (MI.getOperand(Idx).isReg() && MI.getOperand(Idx).getReg()) {
      LLT Ty = MRI.getType(MI.getOperand(Idx).getReg());
      if (!Ty.isValid())
        continue;

      auto Mapping = Parasol::getValueMapping(OpRegBankIdx[Idx], OpSize[Idx]);

      if (!Mapping->isValid()) {
        return getInvalidInstructionMapping();
      }

      OpdsMapping[Idx] = Mapping;
    }
  }

  return getInstructionMapping(DefaultMappingID, Cost,
                               getOperandsMapping(OpdsMapping), NumOperands);
}
