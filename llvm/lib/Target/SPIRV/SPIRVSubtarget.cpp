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
        clEnumValN(SPIRV::Extension::SPV_EXT_shader_atomic_float_add,
                   "SPV_EXT_shader_atomic_float_add",
                   "Adds atomic add instruction on floating-point numbers."),
        clEnumValN(
            SPIRV::Extension::SPV_EXT_shader_atomic_float16_add,
            "SPV_EXT_shader_atomic_float16_add",
            "Extends the SPV_EXT_shader_atomic_float_add extension to support "
            "atomically adding to 16-bit floating-point numbers in memory."),
        clEnumValN(
            SPIRV::Extension::SPV_EXT_shader_atomic_float_min_max,
            "SPV_EXT_shader_atomic_float_min_max",
            "Adds atomic min and max instruction on floating-point numbers."),
        clEnumValN(SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers,
                   "SPV_INTEL_arbitrary_precision_integers",
                   "Allows generating arbitrary width integer types."),
        clEnumValN(SPIRV::Extension::SPV_INTEL_optnone, "SPV_INTEL_optnone",
                   "Adds OptNoneINTEL value for Function Control mask that "
                   "indicates a request to not optimize the function."),
        clEnumValN(SPIRV::Extension::SPV_INTEL_usm_storage_classes,
                   "SPV_INTEL_usm_storage_classes",
                   "Introduces two new storage classes that are sub classes of "
                   "the CrossWorkgroup storage class "
                   "that provides additional information that can enable "
                   "optimization."),
        clEnumValN(SPIRV::Extension::SPV_INTEL_subgroups, "SPV_INTEL_subgroups",
                   "Allows work items in a subgroup to share data without the "
                   "use of local memory and work group barriers, and to "
                   "utilize specialized hardware to load and store blocks of "
                   "data from images or buffers."),
        clEnumValN(SPIRV::Extension::SPV_KHR_uniform_group_instructions,
                   "SPV_KHR_uniform_group_instructions",
                   "Allows support for additional group operations within "
                   "uniform control flow."),
        clEnumValN(SPIRV::Extension::SPV_KHR_no_integer_wrap_decoration,
                   "SPV_KHR_no_integer_wrap_decoration",
                   "Adds decorations to indicate that a given instruction does "
                   "not cause integer wrapping."),
        clEnumValN(SPIRV::Extension::SPV_KHR_expect_assume,
                   "SPV_KHR_expect_assume",
                   "Provides additional information to a compiler, similar to "
                   "the llvm.assume and llvm.expect intrinsics."),
        clEnumValN(SPIRV::Extension::SPV_KHR_bit_instructions,
                   "SPV_KHR_bit_instructions",
                   "This enables bit instructions to be used by SPIR-V modules "
                   "without requiring the Shader capability."),
        clEnumValN(
            SPIRV::Extension::SPV_KHR_linkonce_odr, "SPV_KHR_linkonce_odr",
            "Allows to use the LinkOnceODR linkage type that is to let "
            "a function or global variable to be merged with other functions "
            "or global variables of the same name when linkage occurs."),
        clEnumValN(SPIRV::Extension::SPV_KHR_subgroup_rotate,
                   "SPV_KHR_subgroup_rotate",
                   "Adds a new instruction that enables rotating values across "
                   "invocations within a subgroup."),
        clEnumValN(SPIRV::Extension::SPV_INTEL_variable_length_array,
                   "SPV_INTEL_variable_length_array",
                   "Allows to allocate local arrays whose number of elements "
                   "is unknown at compile time."),
        clEnumValN(SPIRV::Extension::SPV_INTEL_function_pointers,
                   "SPV_INTEL_function_pointers",
                   "Allows translation of function pointers.")));

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
