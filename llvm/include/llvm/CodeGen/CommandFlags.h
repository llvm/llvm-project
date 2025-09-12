//===-- CommandFlags.h - Command Line Flags Interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains codegen-specific flags that are shared between different
// command line tools. The tools "llc" and "opt" both use this file to prevent
// flag duplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COMMANDFLAGS_H
#define LLVM_CODEGEN_COMMANDFLAGS_H

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>
#include <string>
#include <vector>

namespace llvm {

class Module;
class AttrBuilder;
class Function;
class Triple;
class TargetMachine;

namespace codegen {

LLVM_ABI std::string getMArch();

LLVM_ABI std::string getMCPU();

LLVM_ABI std::vector<std::string> getMAttrs();

LLVM_ABI Reloc::Model getRelocModel();
LLVM_ABI std::optional<Reloc::Model> getExplicitRelocModel();

LLVM_ABI ThreadModel::Model getThreadModel();

LLVM_ABI CodeModel::Model getCodeModel();
LLVM_ABI std::optional<CodeModel::Model> getExplicitCodeModel();

LLVM_ABI uint64_t getLargeDataThreshold();
LLVM_ABI std::optional<uint64_t> getExplicitLargeDataThreshold();

LLVM_ABI llvm::ExceptionHandling getExceptionModel();

LLVM_ABI std::optional<CodeGenFileType> getExplicitFileType();

LLVM_ABI CodeGenFileType getFileType();

LLVM_ABI FramePointerKind getFramePointerUsage();

LLVM_ABI bool getEnableUnsafeFPMath();

LLVM_ABI bool getEnableNoInfsFPMath();

LLVM_ABI bool getEnableNoNaNsFPMath();

LLVM_ABI bool getEnableNoSignedZerosFPMath();

LLVM_ABI bool getEnableNoTrappingFPMath();

LLVM_ABI DenormalMode::DenormalModeKind getDenormalFPMath();
LLVM_ABI DenormalMode::DenormalModeKind getDenormalFP32Math();

LLVM_ABI bool getEnableHonorSignDependentRoundingFPMath();

LLVM_ABI llvm::FloatABI::ABIType getFloatABIForCalls();

LLVM_ABI llvm::FPOpFusion::FPOpFusionMode getFuseFPOps();

LLVM_ABI SwiftAsyncFramePointerMode getSwiftAsyncFramePointer();

LLVM_ABI bool getDontPlaceZerosInBSS();

LLVM_ABI bool getEnableGuaranteedTailCallOpt();

LLVM_ABI bool getEnableAIXExtendedAltivecABI();

LLVM_ABI bool getDisableTailCalls();

LLVM_ABI bool getStackSymbolOrdering();

LLVM_ABI bool getStackRealign();

LLVM_ABI std::string getTrapFuncName();

LLVM_ABI bool getUseCtors();

LLVM_ABI bool getDisableIntegratedAS();

LLVM_ABI bool getDataSections();
LLVM_ABI std::optional<bool> getExplicitDataSections();

LLVM_ABI bool getFunctionSections();
LLVM_ABI std::optional<bool> getExplicitFunctionSections();

LLVM_ABI bool getIgnoreXCOFFVisibility();

LLVM_ABI bool getXCOFFTracebackTable();

LLVM_ABI std::string getBBSections();

LLVM_ABI unsigned getTLSSize();

LLVM_ABI bool getEmulatedTLS();
LLVM_ABI std::optional<bool> getExplicitEmulatedTLS();

LLVM_ABI bool getEnableTLSDESC();
LLVM_ABI std::optional<bool> getExplicitEnableTLSDESC();

LLVM_ABI bool getUniqueSectionNames();

LLVM_ABI bool getUniqueBasicBlockSectionNames();

LLVM_ABI bool getSeparateNamedSections();

LLVM_ABI llvm::EABI getEABIVersion();

LLVM_ABI llvm::DebuggerKind getDebuggerTuningOpt();

LLVM_ABI bool getEnableStackSizeSection();

LLVM_ABI bool getEnableAddrsig();

LLVM_ABI bool getEnableCallGraphSection();

LLVM_ABI bool getEmitCallSiteInfo();

LLVM_ABI bool getEnableMachineFunctionSplitter();

LLVM_ABI bool getEnableStaticDataPartitioning();

LLVM_ABI bool getEnableDebugEntryValues();

LLVM_ABI bool getValueTrackingVariableLocations();
LLVM_ABI std::optional<bool> getExplicitValueTrackingVariableLocations();

LLVM_ABI bool getForceDwarfFrameSection();

LLVM_ABI bool getXRayFunctionIndex();

LLVM_ABI bool getDebugStrictDwarf();

LLVM_ABI unsigned getAlignLoops();

LLVM_ABI bool getJMCInstrument();

LLVM_ABI bool getXCOFFReadOnlyPointers();

/// Create this object with static storage to register codegen-related command
/// line options.
struct RegisterCodeGenFlags {
  LLVM_ABI RegisterCodeGenFlags();
};

LLVM_ABI bool getEnableBBAddrMap();

LLVM_ABI llvm::BasicBlockSection
getBBSectionsMode(llvm::TargetOptions &Options);

/// Common utility function tightly tied to the options listed here. Initializes
/// a TargetOptions object with CodeGen flags and returns it.
/// \p TheTriple is used to determine the default value for options if
///    options are not explicitly specified. If those triple dependant options
///    value do not have effect for your component, a default Triple() could be
///    passed in.
LLVM_ABI TargetOptions
InitTargetOptionsFromCodeGenFlags(const llvm::Triple &TheTriple);

LLVM_ABI std::string getCPUStr();

LLVM_ABI std::string getFeaturesStr();

LLVM_ABI std::vector<std::string> getFeatureList();

LLVM_ABI void renderBoolStringAttr(AttrBuilder &B, StringRef Name, bool Val);

/// Set function attributes of function \p F based on CPU, Features, and command
/// line flags.
LLVM_ABI void setFunctionAttributes(StringRef CPU, StringRef Features,
                                    Function &F);

/// Set function attributes of functions in Module M based on CPU,
/// Features, and command line flags.
LLVM_ABI void setFunctionAttributes(StringRef CPU, StringRef Features,
                                    Module &M);

/// Should value-tracking variable locations / instruction referencing be
/// enabled by default for this triple?
LLVM_ABI bool getDefaultValueTrackingVariableLocations(const llvm::Triple &T);

/// Creates a TargetMachine instance with the options defined on the command
/// line. This can be used for tools that do not need further customization of
/// the TargetOptions.
LLVM_ABI Expected<std::unique_ptr<TargetMachine>> createTargetMachineForTriple(
    StringRef TargetTriple,
    CodeGenOptLevel OptLevel = CodeGenOptLevel::Default);

} // namespace codegen
} // namespace llvm

#endif // LLVM_CODEGEN_COMMANDFLAGS_H
