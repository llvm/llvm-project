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

class AArch64InstrPostProcess : public InstrPostProcess {
  mca::Instruction *PreviousInst;
  const MCInst *PreviousMCI;

public:
  AArch64InstrPostProcess(const MCSubtargetInfo &STI, const MCInstrInfo &MCII)
      : InstrPostProcess(STI, MCII), PreviousInst(nullptr),
        PreviousMCI(nullptr) {}

  ~AArch64InstrPostProcess() override = default;

  void postProcessInstruction(mca::Instruction &Inst,
                              const MCInst &MCI) override;
  void resetState() override;
};

/// AES encoding or decoding.
static bool isAESPair(const MCInst &First, const MCInst &Second) {
  switch (Second.getOpcode()) {
  // AES encode.
  case AArch64::AESMCrr: {
    if (First.getOpcode() != AArch64::AESErr)
      return false;

    MCRegister EDef = First.getOperand(0).getReg();
    MCRegister MCDef = Second.getOperand(0).getReg();
    MCRegister MCUse = Second.getOperand(1).getReg();
    return (EDef == MCDef && EDef == MCUse);
  }
  // AES decode.
  case AArch64::AESIMCrr: {
    if (First.getOpcode() != AArch64::AESDrr)
      return false;

    MCRegister DDef = First.getOperand(0).getReg();
    MCRegister IMCDef = Second.getOperand(0).getReg();
    MCRegister IMCUse = Second.getOperand(1).getReg();
    return (DDef == IMCDef && DDef == IMCUse);
  }
  }
  return false;
}

/// Compare and conditional select.
static bool isCmpCSelPair(const MCInst &First, const MCInst &Second) {
  switch (Second.getOpcode()) {
  case AArch64::CSELWr: {
    // 32 bits
    if (!First.getNumOperands())
      return false;

    MCOperand Zero = First.getOperand(0);
    if (!Zero.isReg() || Zero.getReg() != AArch64::WZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSWrs:
      return !AArch64_MC::hasShiftedReg(First);
    case AArch64::SUBSWrx:
      return !AArch64_MC::hasExtendedReg(First);
    case AArch64::SUBSWrr:
    case AArch64::SUBSWri:
      return true;
    }
    return false;
  }
  case AArch64::CSELXr: {
    // 64 bits
    if (!First.getNumOperands())
      return false;

    MCOperand Zero = First.getOperand(0);
    if (!Zero.isReg() || Zero.getReg() != AArch64::XZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSXrs:
      return !AArch64_MC::hasShiftedReg(First);
    case AArch64::SUBSXrx:
    case AArch64::SUBSXrx64:
      return !AArch64_MC::hasExtendedReg(First);
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
static bool isCmpCSetPair(const MCInst &First, const MCInst &Second) {
  switch (Second.getOpcode()) {
  case AArch64::CSINCWr: {
    // 32 bits
    MCOperand Op1 = Second.getOperand(1);
    MCOperand Op2 = Second.getOperand(2);
    if (!Op1.isReg() || Op1.getReg() != AArch64::WZR)
      return false;
    if (!Op2.isReg() || Op2.getReg() != AArch64::WZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSWrs: {
      if (AArch64_MC::hasShiftedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSWrx: {
      if (AArch64_MC::hasExtendedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSWri:
    case AArch64::SUBSWrr:
      break;
    default:
      return false;
    }

    MCOperand Def = First.getOperand(0);
    return (Def.isReg() && Def.getReg() == AArch64::WZR);
  }
  case AArch64::CSINCXr: {
    // 64 bits
    MCOperand Op1 = Second.getOperand(1);
    MCOperand Op2 = Second.getOperand(2);
    if (!Op1.isReg() || Op1.getReg() != AArch64::XZR)
      return false;
    if (!Op2.isReg() || Op2.getReg() != AArch64::XZR)
      return false;

    switch (First.getOpcode()) {
    case AArch64::SUBSXrs: {
      if (AArch64_MC::hasShiftedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSXrx:
    case AArch64::SUBSXrx64: {
      if (AArch64_MC::hasExtendedReg(First))
        return false;
      break;
    }
    case AArch64::SUBSXri:
    case AArch64::SUBSXrr:
      break;
    }

    MCOperand Def = First.getOperand(0);
    return (Def.isReg() && Def.getReg() == AArch64::XZR);
  }
  }
  return false;
}

/// CMN, CMP, TST followed by Bcc
static bool isArithmeticBccPair(const MCInst &First, const MCInst &Second,
                                bool CmpOnly) {
  if (Second.getOpcode() != AArch64::Bcc)
    return false;

  // If we're in CmpOnly mode, we only fuse arithmetic instructions that
  // discard their result.
  if (CmpOnly) {
    MCOperand Def = First.getOperand(0);
    if (!Def.isReg())
      return false;

    if (Def.getReg() != AArch64::XZR && Def.getReg() != AArch64::WZR) {
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
    return !AArch64_MC::hasShiftedReg(First);
  }

  return false;
}

// Move all writes from Second to First instruction.
static void moveWrites(mca::Instruction &First, mca::Instruction &Second) {
  for (WriteState &SecondDef : Second.getDefs()) {
    bool AlreadyDefined = false;
    for (WriteState &FirstDef : First.getDefs()) {
      if (FirstDef.getRegisterID() == SecondDef.getRegisterID()) {
        AlreadyDefined = true;
        break;
      }
    }

    if (!AlreadyDefined)
      First.getDefs().push_back(SecondDef);

    // Second instruction is going to be eliminated, so it cannot have
    // any active writes.
    SecondDef.setRegisterID(0);
  }
}

// Move all reads from Second to First instruction.
static void moveReads(mca::Instruction &First, mca::Instruction &Second) {
  for (ReadState &SecondUse : Second.getUses()) {
    bool AlreadyUsed = false;
    for (ReadState &FirstUse : First.getUses()) {
      if (SecondUse.getRegisterID() == FirstUse.getRegisterID()) {
        AlreadyUsed = true;
        break;
      }
    }

    // Ignore reads of registers which are defined by the instruction
    // we move them to.
    bool IsDef = false;
    for (WriteState &FirstDef : First.getDefs()) {
      if (SecondUse.getRegisterID() == FirstDef.getRegisterID()) {
        IsDef = true;
        break;
      }
    }

    if (!AlreadyUsed && !IsDef)
      First.getUses().push_back(SecondUse);
  }
}

static void fuseInstructions(mca::Instruction &First,
                             mca::Instruction &Second) {
  moveReads(First, Second);
  moveWrites(First, Second);
  Second.setEliminated();
}

static bool tryFuseInstructions(mca::Instruction &First, const MCInst &FirstMCI,
                                mca::Instruction &Second,
                                const MCInst &SecondMCI,
                                const MCSubtargetInfo &STI) {
  if (STI.hasFeature(AArch64::FeatureFuseAES) &&
      isAESPair(FirstMCI, SecondMCI)) {
    fuseInstructions(First, Second);
    return true;
  }

  if (STI.hasFeature(AArch64::FeatureFuseCmpCSel) &&
      isCmpCSelPair(FirstMCI, SecondMCI)) {
    fuseInstructions(First, Second);
    return true;
  }

  if (STI.hasFeature(AArch64::FeatureFuseCmpCSet) &&
      isCmpCSetPair(FirstMCI, SecondMCI)) {
    fuseInstructions(First, Second);
    return true;
  }

  if (STI.hasFeature(AArch64::FeatureCmpBccFusion) ||
      STI.hasFeature(AArch64::FeatureArithmeticBccFusion)) {
    bool CmpOnly = !STI.hasFeature(AArch64::FeatureArithmeticBccFusion);
    if (isArithmeticBccPair(FirstMCI, SecondMCI, CmpOnly)) {
      fuseInstructions(First, Second);
      return true;
    }
  }

  return false;
}

void AArch64InstrPostProcess::postProcessInstruction(mca::Instruction &Inst,
                                                     const MCInst &MCI) {
  if (!PreviousInst) {
    PreviousInst = &Inst;
    PreviousMCI = &MCI;
    return;
  }

  if (tryFuseInstructions(*PreviousInst, *PreviousMCI, Inst, MCI, STI)) {
    PreviousInst = nullptr;
    PreviousMCI = nullptr;
  } else {
    PreviousInst = &Inst;
    PreviousMCI = &MCI;
  }
}

void AArch64InstrPostProcess::resetState() { PreviousInst = nullptr; }

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
    TargetRegistry::RegisterInstrPostProcess(*T, createAArch64InstrPostProcess);
  }
}
