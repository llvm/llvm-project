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

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <string>

namespace llvm {

namespace codegen {

std::string getMArch();

std::string getMCPU();

std::vector<std::string> getMAttrs();

Reloc::Model getRelocModel();
Optional<Reloc::Model> getExplicitRelocModel();

ThreadModel::Model getThreadModel();

CodeModel::Model getCodeModel();
Optional<CodeModel::Model> getExplicitCodeModel();

llvm::ExceptionHandling getExceptionModel();

CodeGenFileType getFileType();
Optional<CodeGenFileType> getExplicitFileType();

CodeGenFileType getFileType();

llvm::FramePointer::FP getFramePointerUsage();

bool getEnableUnsafeFPMath();

bool getEnableNoInfsFPMath();

bool getEnableNoNaNsFPMath();

bool getEnableNoSignedZerosFPMath();

bool getEnableNoTrappingFPMath();

DenormalMode::DenormalModeKind getDenormalFPMath();

bool getEnableHonorSignDependentRoundingFPMath();

llvm::FloatABI::ABIType getFloatABIForCalls();

llvm::FPOpFusion::FPOpFusionMode getFuseFPOps();

bool getDontPlaceZerosInBSS();

bool getEnableGuaranteedTailCallOpt();

bool getDisableTailCalls();

bool getStackSymbolOrdering();

unsigned getOverrideStackAlignment();

bool getStackRealign();

std::string getTrapFuncName();

bool getUseCtors();

bool getRelaxELFRelocations();

bool getDataSections();
Optional<bool> getExplicitDataSections();

bool getFunctionSections();
Optional<bool> getExplicitFunctionSections();

std::string getBBSections();

unsigned getTLSSize();

bool getEmulatedTLS();

bool getUniqueSectionNames();

bool getUniqueBBSectionNames();

llvm::EABI getEABIVersion();

llvm::DebuggerKind getDebuggerTuningOpt();

bool getEnableStackSizeSection();

bool getEnableAddrsig();

bool getEmitCallSiteInfo();

bool getEnableDebugEntryValues();

bool getForceDwarfFrameSection();

/// Create this object with static storage to register codegen-related command
/// line options.
struct RegisterCodeGenFlags {
  RegisterCodeGenFlags();
};

llvm::BasicBlockSection getBBSectionsMode(llvm::TargetOptions &Options);

// Common utility function tightly tied to the options listed here. Initializes
// a TargetOptions object with CodeGen flags and returns it.
TargetOptions InitTargetOptionsFromCodeGenFlags();

std::string getCPUStr();

std::string getFeaturesStr();

std::vector<std::string> getFeatureList();

void renderBoolStringAttr(AttrBuilder &B, StringRef Name, bool Val);

/// Set function attributes of function \p F based on CPU, Features, and command
/// line flags.
void setFunctionAttributes(StringRef CPU, StringRef Features, Function &F);

/// Set function attributes of functions in Module M based on CPU,
/// Features, and command line flags.
void setFunctionAttributes(StringRef CPU, StringRef Features, Module &M);
} // namespace codegen
} // namespace llvm
