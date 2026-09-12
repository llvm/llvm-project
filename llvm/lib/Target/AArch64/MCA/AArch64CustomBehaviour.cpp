//===--------------- AArch64CustomBehaviour.cpp -----------------*-C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines custom behaviour for MCA that is not expressed in the
/// schedule model.
///
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "TargetInfo/AArch64TargetInfo.h"
#include "llvm-c/Visibility.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace mca;

// Same as AArch64_MC::hasExtendedReg, but for an MCA Instruction.
static bool hasExtendedReg(const mca::Instruction &Inst) {
  switch (Inst.getOpcode()) {
  case AArch64::ADDWrx:
  case AArch64::ADDXrx:
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrx:
  case AArch64::SUBWrx:
  case AArch64::SUBXrx:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrx:
  case AArch64::ADDXrx64:
  case AArch64::ADDSXrx64:
  case AArch64::SUBXrx64:
  case AArch64::SUBSXrx64:
    return Inst.getOperand(3)->getImm() != 0;
  default:
    return false;
  } // end of switch-stmt
}

// Same as AArch64_MC::hasShiftedReg, but for an MCA Instruction.
static bool hasShiftedReg(const mca::Instruction &Inst) {
  switch (Inst.getOpcode()) {
  case AArch64::ADDWrs:
  case AArch64::ADDXrs:
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs:
  case AArch64::SUBWrs:
  case AArch64::SUBXrs:
  case AArch64::SUBSWrs:
  case AArch64::SUBSXrs:
  case AArch64::ANDWrs:
  case AArch64::ANDXrs:
  case AArch64::ANDSWrs:
  case AArch64::ANDSXrs:
  case AArch64::BICWrs:
  case AArch64::BICXrs:
  case AArch64::BICSWrs:
  case AArch64::BICSXrs:
  case AArch64::EONWrs:
  case AArch64::EONXrs:
  case AArch64::EORWrs:
  case AArch64::EORXrs:
  case AArch64::ORNWrs:
  case AArch64::ORNXrs:
  case AArch64::ORRWrs:
  case AArch64::ORRXrs:
    return Inst.getOperand(3)->getImm() != 0;
  default:
    return false;
  }
}

/// AES encoding or decoding.
static bool isAESPair(const MCSubtargetInfo &STI, const mca::Instruction &First,
                      const mca::Instruction &Second) {
  switch (Second.getOpcode()) {
  // AES encode.
  case AArch64::AESMCrr: {
    if (First.getOpcode() != AArch64::AESErr)
      return false;
    MCRegister EDef = First.getOperand(0)->getReg();
    MCRegister MCDef = Second.getOperand(0)->getReg();
    MCRegister MCUse = Second.getOperand(1)->getReg();
    return (EDef == MCDef && EDef == MCUse);
  }
  // AES decode.
  case AArch64::AESIMCrr: {
    if (First.getOpcode() != AArch64::AESDrr)
      return false;

    MCRegister DDef = First.getOperand(0)->getReg();
    MCRegister IMCDef = Second.getOperand(0)->getReg();
    MCRegister IMCUse = Second.getOperand(1)->getReg();
    return (DDef == IMCDef && DDef == IMCUse);
  }
  }
  return false;
}

/// Compare and conditional select.
static bool isCmpCSelPair(const MCSubtargetInfo &STI,
                          const mca::Instruction &First,
                          const mca::Instruction &Second) {
  switch (Second.getOpcode()) {
  case AArch64::CSELWr: {
    // 32 bits
    const MCAOperand *Zero = First.getOperand(0);
    if (!Zero || !Zero->isReg() || Zero->getReg() != AArch64::WZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSWrs:
      return !hasShiftedReg(First);
    case AArch64::SUBSWrx:
      return !hasExtendedReg(First);
    case AArch64::SUBSWrr:
    case AArch64::SUBSWri:
      return true;
    }
    return false;
  }
  case AArch64::CSELXr: {
    // 64 bits
    const MCAOperand *Zero = First.getOperand(0);
    if (!Zero || !Zero->isReg() || Zero->getReg() != AArch64::XZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSXrs:
      return !hasShiftedReg(First);
    case AArch64::SUBSXrx:
    case AArch64::SUBSXrx64:
      return !hasExtendedReg(First);
    case AArch64::SUBSXrr:
    case AArch64::SUBSXri:
      return true;
    }
    return false;
  }
  }

  return false;
}

/// Compare and conditional set.
static bool isCmpCSetPair(const MCSubtargetInfo &STI,
                          const mca::Instruction &First,
                          const mca::Instruction &Second) {
  switch (Second.getOpcode()) {
  case AArch64::CSINCWr: {
    // 32 bits
    const MCAOperand *Op1 = Second.getOperand(1);
    const MCAOperand *Op2 = Second.getOperand(2);
    if (!Op1 || !Op1->isReg() || Op1->getReg() != AArch64::WZR)
      return false;
    if (!Op2 || !Op2->isReg() || Op2->getReg() != AArch64::WZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSWrs: {
      if (hasShiftedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSWrx: {
      if (hasExtendedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSWri:
    case AArch64::SUBSWrr:
      break;
    default:
      return false;
    }

    const MCAOperand *Def = First.getOperand(0);
    return (Def && Def->isReg() && Def->getReg() == AArch64::WZR);
  }
  case AArch64::CSINCXr: {
    // 64 bits
    const MCAOperand *Op1 = Second.getOperand(1);
    const MCAOperand *Op2 = Second.getOperand(2);
    if (!Op1 || !Op1->isReg() || Op1->getReg() != AArch64::XZR)
      return false;
    if (!Op2 || !Op2->isReg() || Op2->getReg() != AArch64::XZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSXrs: {
      if (hasShiftedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSXrx:
    case AArch64::SUBSXrx64: {
      if (hasExtendedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSXri:
    case AArch64::SUBSXrr:
      break;
    }

    const MCAOperand *Def = First.getOperand(0);
    return (Def && Def->isReg() && Def->getReg() == AArch64::XZR);
  }
  }
  return false;
}

/// CMN, CMP, TST followed by Bcc
static bool isArithmeticBccPair(const MCSubtargetInfo &STI,
                                const mca::Instruction &First,
                                const mca::Instruction &Second) {
  if (Second.getOpcode() != AArch64::Bcc)
    return false;

  bool CmpOnly = !STI.hasFeature(AArch64::FeatureArithmeticBccFusion);

  // If we're in CmpOnly mode, we only fuse arithmetic instructions that
  // discard their result.
  if (CmpOnly) {
    const MCAOperand *Def = First.getOperand(0);
    if (!Def || !Def->isReg())
      return false;

    if (Def->getReg() != AArch64::XZR && Def->getReg() != AArch64::WZR) {
      return false;
    }
  }

  switch (First.getOpcode()) {
  case AArch64::ADDSWri:
  case AArch64::ADDSWrr:
  case AArch64::ADDSXri:
  case AArch64::ADDSXrr:
  case AArch64::ANDSWri:
  case AArch64::ANDSWrr:
  case AArch64::ANDSXri:
  case AArch64::ANDSXrr:
  case AArch64::SUBSWri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSXri:
  case AArch64::SUBSXrr:
  case AArch64::BICSWrr:
  case AArch64::BICSXrr:
    return true;
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs:
  case AArch64::ANDSWrs:
  case AArch64::ANDSXrs:
  case AArch64::SUBSWrs:
  case AArch64::SUBSXrs:
  case AArch64::BICSWrs:
  case AArch64::BICSXrs:
    return !hasShiftedReg(First);
  }

  return false;
}

class AArch64CustomBehaviour : public CustomBehaviour {
  SmallVector<MacroFusionPredicate, 32> FusionPredicates;

public:
  AArch64CustomBehaviour(const MCSubtargetInfo &STI,
                         const mca::SourceMgr &SrcMgr, const MCInstrInfo &MCII)
      : CustomBehaviour(STI, SrcMgr, MCII) {

    if (STI.hasFeature(AArch64::FeatureFuseAES))
      FusionPredicates.push_back(isAESPair);

    if (STI.hasFeature(AArch64::FeatureFuseCmpCSel))
      FusionPredicates.push_back(isCmpCSelPair);

    if (STI.hasFeature(AArch64::FeatureFuseCmpCSet))
      FusionPredicates.push_back(isCmpCSetPair);

    if (STI.hasFeature(AArch64::FeatureCmpBccFusion) ||
        STI.hasFeature(AArch64::FeatureArithmeticBccFusion))
      FusionPredicates.push_back(isArithmeticBccPair);
  }

  ~AArch64CustomBehaviour() override = default;

  ArrayRef<MacroFusionPredicate> getFusionPredicates() override {
    return FusionPredicates;
  }
};

class AArch64InstrPostProcess : public InstrPostProcess {
public:
  AArch64InstrPostProcess(const MCSubtargetInfo &STI, const MCInstrInfo &MCII)
      : InstrPostProcess(STI, MCII) {}

  ~AArch64InstrPostProcess() override = default;

  void postProcessInstruction(mca::Instruction &Inst,
                              const MCInst &MCI) override {
    for (unsigned I = 0; I < MCI.getNumOperands(); ++I) {
      MCOperand MCO = MCI.getOperand(I);
      MCAOperand Op;
      if (MCO.isReg())
        Op = MCAOperand::createReg(MCO.getReg());
      else if (MCO.isImm())
        Op = MCAOperand::createImm(MCO.getImm());
      else if (MCO.isSFPImm())
        Op = MCAOperand::createSFPImm(MCO.getSFPImm());
      else if (MCO.isDFPImm())
        Op = MCAOperand::createDFPImm(MCO.getDFPImm());

      Op.setIndex(I);
      Inst.addOperand(Op);
    }
  }
};

static CustomBehaviour *
createAArch64CustomBehaviour(const MCSubtargetInfo &STI,
                             const mca::SourceMgr &SrcMgr,
                             const MCInstrInfo &MCII) {
  return new AArch64CustomBehaviour(STI, SrcMgr, MCII);
}

static InstrPostProcess *
createAArch64InstrPostProcess(const MCSubtargetInfo &STI,
                              const MCInstrInfo &MCII) {
  return new AArch64InstrPostProcess(STI, MCII);
}

extern "C" LLVM_C_ABI void LLVMInitializeAArch64TargetMCA() {
  Target *Targets[] = {
      &getTheAArch64leTarget(),  &getTheAArch64beTarget(),
      &getTheAArch64_32Target(), &getTheARM64Target(),
      &getTheARM64_32Target(),
  };

  for (Target *T : Targets) {
    TargetRegistry::RegisterCustomBehaviour(*T, createAArch64CustomBehaviour);
    TargetRegistry::RegisterInstrPostProcess(*T, createAArch64InstrPostProcess);
  }
}
