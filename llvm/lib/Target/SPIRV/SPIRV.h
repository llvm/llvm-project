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

class SPIRVPrepareFunctions
    : public RequiredPassInfoMixin<SPIRVPrepareFunctions> {
  const SPIRVTargetMachine &TM;

public:
  explicit SPIRVPrepareFunctions(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVPrepareFunctionsPass(const SPIRVTargetMachine &TM);

class SPIRVStructurizerWrapper
    : public RequiredPassInfoMixin<SPIRVStructurizerWrapper> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVStructurizerPass();

class SPIRVCBufferAccess : public RequiredPassInfoMixin<SPIRVCBufferAccess> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVCBufferAccessLegacyPass();

class SPIRVPushConstantAccess
    : public RequiredPassInfoMixin<SPIRVPushConstantAccess> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVPushConstantAccess(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVPushConstantAccessLegacyPass(SPIRVTargetMachine *TM);

class SPIRVMergeRegionExitTargets
    : public RequiredPassInfoMixin<SPIRVMergeRegionExitTargets> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVMergeRegionExitTargetsPass();

class SPIRVLegalizeImplicitBinding
    : public RequiredPassInfoMixin<SPIRVLegalizeImplicitBinding> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVLegalizeImplicitBindingPass();

class SPIRVLegalizeZeroSizeArrays
    : public RequiredPassInfoMixin<SPIRVLegalizeZeroSizeArrays> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVLegalizeZeroSizeArrays(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVLegalizeZeroSizeArraysPass(const SPIRVTargetMachine &TM);

class SPIRVFinalizeShaderLinkage
    : public RequiredPassInfoMixin<SPIRVFinalizeShaderLinkage> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVFinalizeShaderLinkage(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVFinalizeShaderLinkagePass(const SPIRVTargetMachine &TM);

class SPIRVLegalizePointerCast
    : public RequiredPassInfoMixin<SPIRVLegalizePointerCast> {
  const SPIRVTargetMachine &TM;

public:
  explicit SPIRVLegalizePointerCast(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM);

class SPIRVRegularizer : public RequiredPassInfoMixin<SPIRVRegularizer> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVRegularizerPass();
FunctionPass *createSPIRVPreLegalizerCombiner();
FunctionPass *createSPIRVPreLegalizerPass();
FunctionPass *createSPIRVPostLegalizerPass();

class SPIRVEmitIntrinsicsPass
    : public RequiredPassInfoMixin<SPIRVEmitIntrinsicsPass> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVEmitIntrinsicsPass(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVEmitIntrinsicsPass(const SPIRVTargetMachine &TM);

class SPIRVPrepareGlobals : public RequiredPassInfoMixin<SPIRVPrepareGlobals> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVPrepareGlobalsPass();

/// Lower llvm.global_ctors and llvm.global_dtors to special kernels.
class SPIRVCtorDtorLoweringPass
    : public RequiredPassInfoMixin<SPIRVCtorDtorLoweringPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

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
void initializeSPIRVEmitIntrinsicsLegacyPass(PassRegistry &);
void initializeSPIRVLegalizePointerCastLegacyPass(PassRegistry &);
void initializeSPIRVRegularizerLegacyPass(PassRegistry &);
void initializeSPIRVMergeRegionExitTargetsLegacyPass(PassRegistry &);
void initializeSPIRVPrepareFunctionsLegacyPass(PassRegistry &);
void initializeSPIRVPrepareGlobalsLegacyPass(PassRegistry &);
void initializeSPIRVLegalizeImplicitBindingLegacyPass(PassRegistry &);
void initializeSPIRVLegalizeZeroSizeArraysLegacyPass(PassRegistry &);
void initializeSPIRVFinalizeShaderLinkageLegacyPass(PassRegistry &);
void initializeSPIRVCtorDtorLoweringLegacyPass(PassRegistry &);
} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRV_H
