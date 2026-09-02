//===-- SPIRV.h - Top-level interface for SPIR-V representation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRV_H
#define LLVM_LIB_TARGET_SPIRV_SPIRV_H

#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class SPIRVTargetMachine;
class SPIRVSubtarget;
class InstructionSelector;
class RegisterBankInfo;

ModulePass *createSPIRVPrepareFunctionsPass(const SPIRVTargetMachine &TM);
FunctionPass *createSPIRVStructurizerPass();
ModulePass *createSPIRVCBufferAccessLegacyPass();
ModulePass *createSPIRVPushConstantAccessLegacyPass(SPIRVTargetMachine *TM);
FunctionPass *createSPIRVMergeRegionExitTargetsPass();
ModulePass *createSPIRVLegalizeImplicitBindingPass();
ModulePass *createSPIRVLegalizeZeroSizeArraysPass(const SPIRVTargetMachine &TM);
FunctionPass *createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM);
FunctionPass *createSPIRVRegularizerPass();
FunctionPass *createSPIRVPreLegalizerCombiner();
FunctionPass *createSPIRVPreLegalizerPass();
FunctionPass *createSPIRVPostLegalizerPass();
ModulePass *createSPIRVEmitIntrinsicsPass(const SPIRVTargetMachine &TM);
ModulePass *createSPIRVPrepareGlobalsPass();
ModulePass *createSPIRVCtorDtorLoweringLegacyPass();
InstructionSelector *
createSPIRVInstructionSelector(const SPIRVTargetMachine &TM,
                               const SPIRVSubtarget &Subtarget,
                               const RegisterBankInfo &RBI);

void initializeSPIRVModuleAnalysisPass(PassRegistry &);
void initializeSPIRVAsmPrinterPass(PassRegistry &);
void initializeSPIRVConvergenceRegionAnalysisWrapperPassPass(PassRegistry &);
void initializeSPIRVPreLegalizerPass(PassRegistry &);
void initializeSPIRVPreLegalizerCombinerPass(PassRegistry &);
void initializeSPIRVPostLegalizerPass(PassRegistry &);
void initializeSPIRVStructurizerPass(PassRegistry &);
void initializeSPIRVCBufferAccessLegacyPass(PassRegistry &);
void initializeSPIRVPushConstantAccessLegacyPass(PassRegistry &);
void initializeSPIRVEmitIntrinsicsPass(PassRegistry &);
void initializeSPIRVLegalizePointerCastLegacyPass(PassRegistry &);
void initializeSPIRVRegularizerLegacyPass(PassRegistry &);
void initializeSPIRVMergeRegionExitTargetsLegacyPass(PassRegistry &);
void initializeSPIRVPrepareFunctionsLegacyPass(PassRegistry &);
void initializeSPIRVPrepareGlobalsLegacyPass(PassRegistry &);
void initializeSPIRVLegalizeImplicitBindingLegacyPass(PassRegistry &);
void initializeSPIRVLegalizeZeroSizeArraysLegacyPass(PassRegistry &);
void initializeSPIRVCtorDtorLoweringLegacyPass(PassRegistry &);
} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRV_H
