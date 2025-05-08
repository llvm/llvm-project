//===-- ParasolSubtarget.cpp - Parasol Subtarget Information --------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file implements the Parasol specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "ParasolSubtarget.h"
#include "Parasol.h"
#include "ParasolMachineFunction.h"
#include "ParasolRegisterInfo.h"
#include "ParasolTargetMachine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "parasol-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ParasolGenSubtargetInfo.inc"

#include "GISel/ParasolCallLowering.h"
#include "GISel/ParasolLegalizerInfo.h"
#include "GISel/ParasolRegisterBankInfo.h"

ParasolSubtarget::ParasolSubtarget(const Triple &TT, StringRef CPU,
                                   StringRef FS, const TargetMachine &TM)
    // TODO: Maybe pass in an actual TuneCPU here
    : ParasolGenSubtargetInfo(TT, CPU, StringRef(), FS), TSInfo(),
      InstrInfo(initializeSubtargetDependencies(TT, CPU, FS, TM)),
      FrameLowering(*this), TLInfo(TM, *this), RegInfo(*this) {

  CallLoweringInfo.reset(new ParasolCallLowering(*getTargetLowering()));
  Legalizer.reset(new ParasolLegalizerInfo(*this));
  auto *RBI = new ParasolRegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);

  InstSelector.reset(createParasolInstructionSelector(
      *static_cast<const ParasolTargetMachine *>(&TM), *this, *RBI));
}

ParasolSubtarget &ParasolSubtarget::initializeSubtargetDependencies(
    const Triple &TT, StringRef CPU, StringRef FS, const TargetMachine &TM) {
  if (CPU.empty())
    CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(CPU, StringRef(), FS);

  return *this;
}

const CallLowering *ParasolSubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

InstructionSelector *ParasolSubtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *ParasolSubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *ParasolSubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}
