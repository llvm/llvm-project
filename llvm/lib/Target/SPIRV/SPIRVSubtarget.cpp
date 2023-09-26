//===-- SPIRVSubtarget.cpp - SPIR-V Subtarget Information ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIR-V specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SPIRVSubtarget.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVLegalizerInfo.h"
#include "SPIRVRegisterBankInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SPIRVGenSubtargetInfo.inc"

cl::list<SPIRV::Extension::Extension> Extensions(
    "spirv-extensions", cl::desc("SPIR-V extensions"), cl::ZeroOrMore,
    cl::Hidden,
    cl::values(
        clEnumValN(SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers,
                   "SPV_INTEL_arbitrary_precision_integers",
                   "Allows generating arbitrary width integer types"),
        clEnumValN(SPIRV::Extension::SPV_INTEL_optnone, "SPV_INTEL_optnone",
                   "Adds OptNoneINTEL value for Function Control mask that "
                   "indicates a request to not optimize the function"),
        clEnumValN(SPIRV::Extension::SPV_KHR_no_integer_wrap_decoration,
                   "SPV_KHR_no_integer_wrap_decoration",
                   "Adds decorations to indicate that a given instruction does "
                   "not cause integer wrapping"),
        clEnumValN(SPIRV::Extension::SPV_KHR_bit_instructions,
                   "SPV_KHR_bit_instructions",
                   "This enables bit instructions to be used by SPIR-V modules "
                   "without requiring the Shader capability")));

// Compare version numbers, but allow 0 to mean unspecified.
static bool isAtLeastVer(uint32_t Target, uint32_t VerToCompareTo) {
  return Target == 0 || Target >= VerToCompareTo;
}

SPIRVSubtarget::SPIRVSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const SPIRVTargetMachine &TM)
    : SPIRVGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS),
      PointerSize(TM.getPointerSizeInBits(/* AS= */ 0)), SPIRVVersion(0),
      OpenCLVersion(0), InstrInfo(),
      FrameLowering(initSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      TargetTriple(TT) {
  // The order of initialization is important.
  initAvailableExtensions();
  initAvailableExtInstSets();

  GR = std::make_unique<SPIRVGlobalRegistry>(PointerSize);
  CallLoweringInfo = std::make_unique<SPIRVCallLowering>(TLInfo, GR.get());
  Legalizer = std::make_unique<SPIRVLegalizerInfo>(*this);
  RegBankInfo = std::make_unique<SPIRVRegisterBankInfo>();
  InstSelector.reset(
      createSPIRVInstructionSelector(TM, *this, *RegBankInfo.get()));
}

SPIRVSubtarget &SPIRVSubtarget::initSubtargetDependencies(StringRef CPU,
                                                          StringRef FS) {
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  if (SPIRVVersion == 0)
    SPIRVVersion = 14;
  if (OpenCLVersion == 0)
    OpenCLVersion = 22;
  return *this;
}

bool SPIRVSubtarget::canUseExtension(SPIRV::Extension::Extension E) const {
  return AvailableExtensions.contains(E);
}

bool SPIRVSubtarget::canUseExtInstSet(
    SPIRV::InstructionSet::InstructionSet E) const {
  return AvailableExtInstSets.contains(E);
}

bool SPIRVSubtarget::isAtLeastSPIRVVer(uint32_t VerToCompareTo) const {
  return isAtLeastVer(SPIRVVersion, VerToCompareTo);
}

bool SPIRVSubtarget::isAtLeastOpenCLVer(uint32_t VerToCompareTo) const {
  if (!isOpenCLEnv())
    return false;
  return isAtLeastVer(OpenCLVersion, VerToCompareTo);
}

// If the SPIR-V version is >= 1.4 we can call OpPtrEqual and OpPtrNotEqual.
bool SPIRVSubtarget::canDirectlyComparePointers() const {
  return isAtLeastVer(SPIRVVersion, 14);
}

void SPIRVSubtarget::initAvailableExtensions() {
  AvailableExtensions.clear();
  if (!isOpenCLEnv())
    return;

  for (auto Extension : Extensions)
    AvailableExtensions.insert(Extension);
}

// TODO: use command line args for this rather than just defaults.
// Must have called initAvailableExtensions first.
void SPIRVSubtarget::initAvailableExtInstSets() {
  AvailableExtInstSets.clear();
  if (!isOpenCLEnv())
    AvailableExtInstSets.insert(SPIRV::InstructionSet::GLSL_std_450);
  else
    AvailableExtInstSets.insert(SPIRV::InstructionSet::OpenCL_std);

  // Handle extended instruction sets from extensions.
  if (canUseExtension(
          SPIRV::Extension::SPV_AMD_shader_trinary_minmax_extension)) {
    AvailableExtInstSets.insert(
        SPIRV::InstructionSet::SPV_AMD_shader_trinary_minmax);
  }
}
