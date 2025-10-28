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
#include "SPIRVCommandLine.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVLegalizerInfo.h"
#include "SPIRVRegisterBankInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SPIRVGenSubtargetInfo.inc"

static cl::opt<bool>
    SPVTranslatorCompat("translator-compatibility-mode",
                        cl::desc("SPIR-V Translator compatibility mode"),
                        cl::Optional, cl::init(false));

static cl::opt<std::set<SPIRV::Extension::Extension>, false,
               SPIRVExtensionsParser>
    Extensions("spirv-ext",
               cl::desc("Specify list of enabled SPIR-V extensions"));

// Provides access to the cl::opt<...> `Extensions` variable from outside of the
// module.
void SPIRVSubtarget::addExtensionsToClOpt(
    const std::set<SPIRV::Extension::Extension> &AllowList) {
  Extensions.insert(AllowList.begin(), AllowList.end());
}

// Compare version numbers, but allow 0 to mean unspecified.
static bool isAtLeastVer(VersionTuple Target, VersionTuple VerToCompareTo) {
  return Target.empty() || Target >= VerToCompareTo;
}

SPIRVSubtarget::SPIRVSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const SPIRVTargetMachine &TM)
    : SPIRVGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS),
      PointerSize(TM.getPointerSizeInBits(/* AS= */ 0)),
      InstrInfo(initSubtargetDependencies(CPU, FS)), FrameLowering(*this),
      TLInfo(TM, *this), TargetTriple(TT) {
  switch (TT.getSubArch()) {
  case Triple::SPIRVSubArch_v10:
    SPIRVVersion = VersionTuple(1, 0);
    break;
  case Triple::SPIRVSubArch_v11:
    SPIRVVersion = VersionTuple(1, 1);
    break;
  case Triple::SPIRVSubArch_v12:
    SPIRVVersion = VersionTuple(1, 2);
    break;
  case Triple::SPIRVSubArch_v13:
    SPIRVVersion = VersionTuple(1, 3);
    break;
  case Triple::SPIRVSubArch_v14:
  default:
    SPIRVVersion = VersionTuple(1, 4);
    break;
  case Triple::SPIRVSubArch_v15:
    SPIRVVersion = VersionTuple(1, 5);
    break;
  case Triple::SPIRVSubArch_v16:
    SPIRVVersion = VersionTuple(1, 6);
    break;
  }
  OpenCLVersion = VersionTuple(2, 2);

  // Set the environment based on the target triple.
  if (TargetTriple.getOS() == Triple::Vulkan)
    Env = Shader;
  else if (TargetTriple.getEnvironment() == Triple::OpenCL)
    Env = Kernel;
  else
    Env = Unknown;

  // Set the default extensions based on the target triple.
  if (TargetTriple.getVendor() == Triple::Intel)
    Extensions.insert(SPIRV::Extension::SPV_INTEL_function_pointers);

  // The order of initialization is important.
  initAvailableExtensions(Extensions);
  initAvailableExtInstSets();

  GR = std::make_unique<SPIRVGlobalRegistry>(PointerSize);
  CallLoweringInfo = std::make_unique<SPIRVCallLowering>(TLInfo, GR.get());
  InlineAsmInfo = std::make_unique<SPIRVInlineAsmLowering>(TLInfo);
  Legalizer = std::make_unique<SPIRVLegalizerInfo>(*this);
  RegBankInfo = std::make_unique<SPIRVRegisterBankInfo>();
  InstSelector.reset(createSPIRVInstructionSelector(TM, *this, *RegBankInfo));
}

SPIRVSubtarget &SPIRVSubtarget::initSubtargetDependencies(StringRef CPU,
                                                          StringRef FS) {
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  return *this;
}

bool SPIRVSubtarget::canUseExtension(SPIRV::Extension::Extension E) const {
  return AvailableExtensions.contains(E);
}

bool SPIRVSubtarget::canUseExtInstSet(
    SPIRV::InstructionSet::InstructionSet E) const {
  return AvailableExtInstSets.contains(E);
}

SPIRV::InstructionSet::InstructionSet
SPIRVSubtarget::getPreferredInstructionSet() const {
  if (isShader())
    return SPIRV::InstructionSet::GLSL_std_450;
  else
    return SPIRV::InstructionSet::OpenCL_std;
}

bool SPIRVSubtarget::isAtLeastSPIRVVer(VersionTuple VerToCompareTo) const {
  return isAtLeastVer(SPIRVVersion, VerToCompareTo);
}

bool SPIRVSubtarget::isAtLeastOpenCLVer(VersionTuple VerToCompareTo) const {
  if (isShader())
    return false;
  return isAtLeastVer(OpenCLVersion, VerToCompareTo);
}

// If the SPIR-V version is >= 1.4 we can call OpPtrEqual and OpPtrNotEqual.
// In SPIR-V Translator compatibility mode this feature is not available.
bool SPIRVSubtarget::canDirectlyComparePointers() const {
  return !SPVTranslatorCompat && isAtLeastVer(SPIRVVersion, VersionTuple(1, 4));
}

void SPIRVSubtarget::accountForAMDShaderTrinaryMinmax() {
  if (canUseExtension(
          SPIRV::Extension::SPV_AMD_shader_trinary_minmax_extension)) {
    AvailableExtInstSets.insert(
        SPIRV::InstructionSet::SPV_AMD_shader_trinary_minmax);
  }
}

// TODO: use command line args for this rather than just defaults.
// Must have called initAvailableExtensions first.
void SPIRVSubtarget::initAvailableExtInstSets() {
  AvailableExtInstSets.clear();
  if (isShader())
    AvailableExtInstSets.insert(SPIRV::InstructionSet::GLSL_std_450);
  else
    AvailableExtInstSets.insert(SPIRV::InstructionSet::OpenCL_std);

  // Handle extended instruction sets from extensions.
  accountForAMDShaderTrinaryMinmax();
}

// Set available extensions after SPIRVSubtarget is created.
void SPIRVSubtarget::initAvailableExtensions(
    const std::set<SPIRV::Extension::Extension> &AllowedExtIds) {
  AvailableExtensions.clear();
  const std::set<SPIRV::Extension::Extension> &ValidExtensions =
      SPIRVExtensionsParser::getValidExtensions(TargetTriple);

  for (const auto &Ext : AllowedExtIds) {
    if (ValidExtensions.count(Ext))
      AvailableExtensions.insert(Ext);
  }

  accountForAMDShaderTrinaryMinmax();
}
