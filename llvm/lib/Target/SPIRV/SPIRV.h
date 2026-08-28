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
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class SPIRVTargetMachine;
class SPIRVSubtarget;
class InstructionSelector;
class RegisterBankInfo;

class SPIRVPrepareFunctionsPass
    : public RequiredPassInfoMixin<SPIRVPrepareFunctionsPass> {
  const SPIRVTargetMachine &TM;

public:
  explicit SPIRVPrepareFunctionsPass(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVPrepareFunctionsPass(const SPIRVTargetMachine &TM);

class SPIRVStructurizerPass
    : public RequiredPassInfoMixin<SPIRVStructurizerPass> {
public:
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVStructurizerPass();

class SPIRVCBufferAccessPass
    : public RequiredPassInfoMixin<SPIRVCBufferAccessPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVCBufferAccessLegacyPass();

class SPIRVPushConstantAccessPass
    : public RequiredPassInfoMixin<SPIRVPushConstantAccessPass> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVPushConstantAccessPass(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVPushConstantAccessLegacyPass(SPIRVTargetMachine *TM);

class SPIRVMergeRegionExitTargetsPass
    : public RequiredPassInfoMixin<SPIRVMergeRegionExitTargetsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVMergeRegionExitTargetsPass();

class SPIRVLegalizeImplicitBindingPass
    : public RequiredPassInfoMixin<SPIRVLegalizeImplicitBindingPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVLegalizeImplicitBindingPass();

class SPIRVLegalizeZeroSizeArraysPass
    : public RequiredPassInfoMixin<SPIRVLegalizeZeroSizeArraysPass> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVLegalizeZeroSizeArraysPass(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVLegalizeZeroSizeArraysPass(const SPIRVTargetMachine &TM);

class SPIRVFinalizeShaderLinkagePass
    : public RequiredPassInfoMixin<SPIRVFinalizeShaderLinkagePass> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVFinalizeShaderLinkagePass(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVFinalizeShaderLinkagePass(const SPIRVTargetMachine &TM);

class SPIRVLegalizePointerCastPass
    : public RequiredPassInfoMixin<SPIRVLegalizePointerCastPass> {
  const SPIRVTargetMachine &TM;

public:
  explicit SPIRVLegalizePointerCastPass(const SPIRVTargetMachine &TM)
      : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM);

class SPIRVRegularizerPass
    : public RequiredPassInfoMixin<SPIRVRegularizerPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

FunctionPass *createSPIRVRegularizerPass();

class SPIRVPreLegalizerPass
    : public RequiredPassInfoMixin<SPIRVPreLegalizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createSPIRVPreLegalizerLegacyPass();

class SPIRVPreLegalizerCombinerPass
    : public RequiredPassInfoMixin<SPIRVPreLegalizerCombinerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createSPIRVPreLegalizerCombinerLegacyPass();

class SPIRVPostLegalizerPass
    : public RequiredPassInfoMixin<SPIRVPostLegalizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createSPIRVPostLegalizerLegacyPass();

class SPIRVEmitIntrinsicsPass
    : public RequiredPassInfoMixin<SPIRVEmitIntrinsicsPass> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVEmitIntrinsicsPass(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createSPIRVEmitIntrinsicsPass(const SPIRVTargetMachine &TM);

class SPIRVPrepareGlobalsPass
    : public RequiredPassInfoMixin<SPIRVPrepareGlobalsPass> {
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
void initializeSPIRVPreLegalizerLegacyPass(PassRegistry &);
void initializeSPIRVPreLegalizerCombinerLegacyPass(PassRegistry &);
void initializeSPIRVPostLegalizerLegacyPass(PassRegistry &);
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
