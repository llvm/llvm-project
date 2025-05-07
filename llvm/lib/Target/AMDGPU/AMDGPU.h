//===-- AMDGPU.h - MachineFunction passes hw codegen --------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPU_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPU_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class AMDGPUTargetMachine;
class GCNTargetMachine;
class TargetMachine;

// GlobalISel passes
void initializeAMDGPUPreLegalizerCombinerPass(PassRegistry &);
FunctionPass *createAMDGPUPreLegalizeCombiner(bool IsOptNone);
void initializeAMDGPUPostLegalizerCombinerPass(PassRegistry &);
FunctionPass *createAMDGPUPostLegalizeCombiner(bool IsOptNone);
FunctionPass *createAMDGPURegBankCombiner(bool IsOptNone);
void initializeAMDGPURegBankCombinerPass(PassRegistry &);
FunctionPass *createAMDGPUGlobalISelDivergenceLoweringPass();
FunctionPass *createAMDGPURegBankSelectPass();
FunctionPass *createAMDGPURegBankLegalizePass();

// SI Passes
FunctionPass *createGCNDPPCombinePass();
FunctionPass *createSIAnnotateControlFlowLegacyPass();
FunctionPass *createSIFoldOperandsLegacyPass();
FunctionPass *createSIPeepholeSDWALegacyPass();
FunctionPass *createSILowerI1CopiesLegacyPass();
FunctionPass *createSIShrinkInstructionsLegacyPass();
FunctionPass *createSILoadStoreOptimizerLegacyPass();
FunctionPass *createSIWholeQuadModeLegacyPass();
FunctionPass *createAMDGPUBundleIdxLdStPass();
FunctionPass *createSIFixControlFlowLiveIntervalsPass();
FunctionPass *createSIOptimizeExecMaskingPreRAPass();
FunctionPass *createSIOptimizeVGPRLiveRangeLegacyPass();
FunctionPass *createSIFixSGPRCopiesLegacyPass();
FunctionPass *createLowerWWMCopiesPass();
FunctionPass *createSIMemoryLegalizerPass();
FunctionPass *createSIInsertWaitcntsPass();
FunctionPass *createSIPreAllocateWWMRegsLegacyPass();
FunctionPass *createSIFormMemoryClausesLegacyPass();

FunctionPass *createSIPostRABundlerPass();
FunctionPass *createAMDGPUImageIntrinsicOptimizerPass(const TargetMachine *);
ModulePass *createAMDGPURemoveIncompatibleFunctionsPass(const TargetMachine *);
FunctionPass *createAMDGPUCodeGenPreparePass();
FunctionPass *createAMDGPULateCodeGenPrepareLegacyPass();
FunctionPass *createAMDGPUReserveWWMRegsPass();
FunctionPass *createAMDGPURewriteOutArgumentsPass();
ModulePass *
createAMDGPULowerModuleLDSLegacyPass(const AMDGPUTargetMachine *TM = nullptr);
FunctionPass *createAMDGPUMarkPromotableLaneSharedLegacyPass();
FunctionPass *createAMDGPUMarkPromotablePrivateLegacyPass();
ModulePass *createAMDGPULowerBufferFatPointersPass();
FunctionPass *createSIModeRegisterPass();
FunctionPass *createGCNPreRAOptimizationsLegacyPass();
FunctionPass *createAMDGPUPreloadKernArgPrologLegacyPass();
FunctionPass *createAMDGPUIdxRegAllocPass();
FunctionPass *createAMDGPUPrivateObjectVGPRsPass();
FunctionPass *createAMDGPUIndexingInfoWrapperPass();

struct AMDGPUSimplifyLibCallsPass : PassInfoMixin<AMDGPUSimplifyLibCallsPass> {
  AMDGPUSimplifyLibCallsPass() {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct AMDGPUImageIntrinsicOptimizerPass
    : PassInfoMixin<AMDGPUImageIntrinsicOptimizerPass> {
  AMDGPUImageIntrinsicOptimizerPass(TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
};

struct AMDGPUUseNativeCallsPass : PassInfoMixin<AMDGPUUseNativeCallsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class SILowerI1CopiesPass : public PassInfoMixin<SILowerI1CopiesPass> {
public:
  SILowerI1CopiesPass() = default;
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

void initializeAMDGPUDAGToDAGISelLegacyPass(PassRegistry &);

void initializeAMDGPUAlwaysInlinePass(PassRegistry&);

void initializeAMDGPUAsmPrinterPass(PassRegistry &);

Pass *createAMDGPUAttributorLegacyPass();
void initializeAMDGPUAttributorLegacyPass(PassRegistry &);

// DPP/Iterative option enables the atomic optimizer with given strategy
// whereas None disables the atomic optimizer.
enum class ScanOptions { DPP, Iterative, None };
FunctionPass *createAMDGPUAtomicOptimizerPass(ScanOptions ScanStrategy);
void initializeAMDGPUAtomicOptimizerPass(PassRegistry &);
extern char &AMDGPUAtomicOptimizerID;

ModulePass *createAMDGPUCtorDtorLoweringLegacyPass();
void initializeAMDGPUCtorDtorLoweringLegacyPass(PassRegistry &);
extern char &AMDGPUCtorDtorLoweringLegacyPassID;

FunctionPass *createAMDGPULowerKernelArgumentsPass();
void initializeAMDGPULowerKernelArgumentsPass(PassRegistry &);
extern char &AMDGPULowerKernelArgumentsID;

FunctionPass *createAMDGPUPromoteKernelArgumentsPass();
void initializeAMDGPUPromoteKernelArgumentsPass(PassRegistry &);
extern char &AMDGPUPromoteKernelArgumentsID;

struct AMDGPUPromoteKernelArgumentsPass
    : PassInfoMixin<AMDGPUPromoteKernelArgumentsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

ModulePass *createAMDGPULowerKernelAttributesPass();
void initializeAMDGPULowerKernelAttributesPass(PassRegistry &);
extern char &AMDGPULowerKernelAttributesID;

struct AMDGPULowerKernelAttributesPass
    : PassInfoMixin<AMDGPULowerKernelAttributesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeAMDGPULowerModuleLDSLegacyPass(PassRegistry &);
extern char &AMDGPULowerModuleLDSLegacyPassID;

struct AMDGPULowerModuleLDSPass : PassInfoMixin<AMDGPULowerModuleLDSPass> {
  const AMDGPUTargetMachine &TM;
  AMDGPULowerModuleLDSPass(const AMDGPUTargetMachine &TM_) : TM(TM_) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeAMDGPUMarkPromotableLaneSharedLegacyPass(PassRegistry &);
extern char &AMDGPUMarkPromotableLaneSharedLegacyPassID;

struct AMDGPUMarkPromotableLaneSharedPass
    : PassInfoMixin<AMDGPUMarkPromotableLaneSharedPass> {
  AMDGPUMarkPromotableLaneSharedPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeAMDGPUMarkPromotablePrivateLegacyPass(PassRegistry &);
extern char &AMDGPUMarkPromotablePrivateLegacyPassID;

struct AMDGPUMarkPromotablePrivatePass
    : PassInfoMixin<AMDGPUMarkPromotablePrivatePass> {
  AMDGPUMarkPromotablePrivatePass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeAMDGPULowerBufferFatPointersPass(PassRegistry &);
extern char &AMDGPULowerBufferFatPointersID;

struct AMDGPULowerBufferFatPointersPass
    : PassInfoMixin<AMDGPULowerBufferFatPointersPass> {
  AMDGPULowerBufferFatPointersPass(const TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  const TargetMachine &TM;
};

void initializeAMDGPUReserveWWMRegsLegacyPass(PassRegistry &);
extern char &AMDGPUReserveWWMRegsLegacyID;

void initializeAMDGPURewriteOutArgumentsPass(PassRegistry &);
extern char &AMDGPURewriteOutArgumentsID;

void initializeGCNDPPCombineLegacyPass(PassRegistry &);
extern char &GCNDPPCombineLegacyID;

void initializeSIFoldOperandsLegacyPass(PassRegistry &);
extern char &SIFoldOperandsLegacyID;

void initializeSIPeepholeSDWALegacyPass(PassRegistry &);
extern char &SIPeepholeSDWALegacyID;

void initializeSIShrinkInstructionsLegacyPass(PassRegistry &);
extern char &SIShrinkInstructionsLegacyID;

void initializeSIFixSGPRCopiesLegacyPass(PassRegistry &);
extern char &SIFixSGPRCopiesLegacyID;

void initializeSIFixVGPRCopiesLegacyPass(PassRegistry &);
extern char &SIFixVGPRCopiesID;

void initializeSILowerWWMCopiesLegacyPass(PassRegistry &);
extern char &SILowerWWMCopiesLegacyID;

void initializeSILowerI1CopiesLegacyPass(PassRegistry &);
extern char &SILowerI1CopiesLegacyID;

void initializeAMDGPUGlobalISelDivergenceLoweringPass(PassRegistry &);
extern char &AMDGPUGlobalISelDivergenceLoweringID;

void initializeAMDGPURegBankSelectPass(PassRegistry &);
extern char &AMDGPURegBankSelectID;

void initializeAMDGPURegBankLegalizePass(PassRegistry &);
extern char &AMDGPURegBankLegalizeID;

void initializeAMDGPUMarkLastScratchLoadLegacyPass(PassRegistry &);
extern char &AMDGPUMarkLastScratchLoadID;

void initializeSILowerSGPRSpillsLegacyPass(PassRegistry &);
extern char &SILowerSGPRSpillsLegacyID;

void initializeSILoadStoreOptimizerLegacyPass(PassRegistry &);
extern char &SILoadStoreOptimizerLegacyID;

void initializeSIWholeQuadModeLegacyPass(PassRegistry &);
extern char &SIWholeQuadModeID;

void initializeAMDGPUBundleIdxLdStPass(PassRegistry &);
extern char &AMDGPUBundleIdxLdStID;

void initializeAMDGPUIdxRegAllocPass(PassRegistry &);
extern char &AMDGPUIdxRegAllocID;

void initializeSILowerControlFlowLegacyPass(PassRegistry &);
extern char &SILowerControlFlowLegacyID;

void initializeSIPreEmitPeepholeLegacyPass(PassRegistry &);
extern char &SIPreEmitPeepholeID;

void initializeSILateBranchLoweringLegacyPass(PassRegistry &);
extern char &SILateBranchLoweringPassID;

void initializeSIOptimizeExecMaskingLegacyPass(PassRegistry &);
extern char &SIOptimizeExecMaskingLegacyID;

void initializeSIPreAllocateWWMRegsLegacyPass(PassRegistry &);
extern char &SIPreAllocateWWMRegsLegacyID;

void initializeAMDGPUImageIntrinsicOptimizerPass(PassRegistry &);
extern char &AMDGPUImageIntrinsicOptimizerID;

void initializeAMDGPUPerfHintAnalysisLegacyPass(PassRegistry &);
extern char &AMDGPUPerfHintAnalysisLegacyID;

void initializeGCNRegPressurePrinterPass(PassRegistry &);
extern char &GCNRegPressurePrinterID;

void initializeAMDGPUPreloadKernArgPrologLegacyPass(PassRegistry &);
extern char &AMDGPUPreloadKernArgPrologLegacyID;

void initializeAMDGPUPrivateObjectVGPRsPass(PassRegistry &);
extern char &AMDGPUPrivateObjectVGPRsID;

void initializeAMDGPUIndexingInfoWrapperPass(PassRegistry &);
extern char &AMDGPUIndexingInfoWrapperID;

// Passes common to R600 and SI
FunctionPass *createAMDGPUPromoteAlloca();
void initializeAMDGPUPromoteAllocaPass(PassRegistry&);
extern char &AMDGPUPromoteAllocaID;

FunctionPass *createAMDGPUPromoteAllocaToVector();
void initializeAMDGPUPromoteAllocaToVectorPass(PassRegistry&);
extern char &AMDGPUPromoteAllocaToVectorID;

struct AMDGPUPromoteAllocaPass : PassInfoMixin<AMDGPUPromoteAllocaPass> {
  AMDGPUPromoteAllocaPass(TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
};

struct AMDGPUPromoteAllocaToVectorPass
    : PassInfoMixin<AMDGPUPromoteAllocaToVectorPass> {
  AMDGPUPromoteAllocaToVectorPass(TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
};

struct AMDGPUAtomicOptimizerPass : PassInfoMixin<AMDGPUAtomicOptimizerPass> {
  AMDGPUAtomicOptimizerPass(TargetMachine &TM, ScanOptions ScanImpl)
      : TM(TM), ScanImpl(ScanImpl) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  TargetMachine &TM;
  ScanOptions ScanImpl;
};

struct AMDGPUInsertDelayAluPass
    : public PassInfoMixin<AMDGPUInsertDelayAluPass> {
  PreservedAnalyses run(MachineFunction &F,
                        MachineFunctionAnalysisManager &MFAM);
};

Pass *createAMDGPUStructurizeCFGPass();
FunctionPass *createAMDGPUISelDag(TargetMachine &TM, CodeGenOptLevel OptLevel);
ModulePass *createAMDGPUAlwaysInlinePass(bool GlobalOpt = true);

struct AMDGPUAlwaysInlinePass : PassInfoMixin<AMDGPUAlwaysInlinePass> {
  AMDGPUAlwaysInlinePass(bool GlobalOpt = true) : GlobalOpt(GlobalOpt) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  bool GlobalOpt;
};

void initializeAMDGPUSwLowerLDSLegacyPass(PassRegistry &);
extern char &AMDGPUSwLowerLDSLegacyPassID;
ModulePass *
createAMDGPUSwLowerLDSLegacyPass(const AMDGPUTargetMachine *TM = nullptr);

struct AMDGPUSwLowerLDSPass : PassInfoMixin<AMDGPUSwLowerLDSPass> {
  const AMDGPUTargetMachine &TM;
  AMDGPUSwLowerLDSPass(const AMDGPUTargetMachine &TM_) : TM(TM_) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AMDGPUCodeGenPreparePass
    : public PassInfoMixin<AMDGPUCodeGenPreparePass> {
private:
  TargetMachine &TM;

public:
  AMDGPUCodeGenPreparePass(TargetMachine &TM) : TM(TM){};
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};

class AMDGPULateCodeGenPreparePass
    : public PassInfoMixin<AMDGPULateCodeGenPreparePass> {
private:
  const GCNTargetMachine &TM;

public:
  AMDGPULateCodeGenPreparePass(const GCNTargetMachine &TM) : TM(TM) {};
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};

class AMDGPULowerKernelArgumentsPass
    : public PassInfoMixin<AMDGPULowerKernelArgumentsPass> {
private:
  TargetMachine &TM;

public:
  AMDGPULowerKernelArgumentsPass(TargetMachine &TM) : TM(TM){};
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};

struct AMDGPUAttributorOptions {
  bool IsClosedWorld = false;
};

class AMDGPUAttributorPass : public PassInfoMixin<AMDGPUAttributorPass> {
private:
  TargetMachine &TM;

  AMDGPUAttributorOptions Options;

  const ThinOrFullLTOPhase LTOPhase;

public:
  AMDGPUAttributorPass(TargetMachine &TM, AMDGPUAttributorOptions Options,
                       ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None)
      : TM(TM), Options(Options), LTOPhase(LTOPhase) {};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AMDGPUAnnotateUniformValuesPass
    : public PassInfoMixin<AMDGPUAnnotateUniformValuesPass> {
public:
  AMDGPUAnnotateUniformValuesPass() {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class SIModeRegisterPass : public PassInfoMixin<SIModeRegisterPass> {
public:
  SIModeRegisterPass() {}
  PreservedAnalyses run(MachineFunction &F, MachineFunctionAnalysisManager &AM);
};

class SIMemoryLegalizerPass : public PassInfoMixin<SIMemoryLegalizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

class GCNCreateVOPDPass : public PassInfoMixin<GCNCreateVOPDPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &AM);
};

class AMDGPUMarkLastScratchLoadPass
    : public PassInfoMixin<AMDGPUMarkLastScratchLoadPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &AM);
};

class SIInsertWaitcntsPass : public PassInfoMixin<SIInsertWaitcntsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

class SIInsertHardClausesPass : public PassInfoMixin<SIInsertHardClausesPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class SILateBranchLoweringPass
    : public PassInfoMixin<SILateBranchLoweringPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

class SIPreEmitPeepholePass : public PassInfoMixin<SIPreEmitPeepholePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

class AMDGPUSetWavePriorityPass
    : public PassInfoMixin<AMDGPUSetWavePriorityPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createAMDGPUAnnotateUniformValuesLegacy();

ModulePass *createAMDGPUPrintfRuntimeBinding();
void initializeAMDGPUPrintfRuntimeBindingPass(PassRegistry&);
extern char &AMDGPUPrintfRuntimeBindingID;

void initializeAMDGPUResourceUsageAnalysisPass(PassRegistry &);
extern char &AMDGPUResourceUsageAnalysisID;

struct AMDGPUPrintfRuntimeBindingPass
    : PassInfoMixin<AMDGPUPrintfRuntimeBindingPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass* createAMDGPUUnifyMetadataPass();
void initializeAMDGPUUnifyMetadataPass(PassRegistry&);
extern char &AMDGPUUnifyMetadataID;

struct AMDGPUUnifyMetadataPass : PassInfoMixin<AMDGPUUnifyMetadataPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeSIOptimizeExecMaskingPreRALegacyPass(PassRegistry &);
extern char &SIOptimizeExecMaskingPreRAID;

void initializeSIOptimizeVGPRLiveRangeLegacyPass(PassRegistry &);
extern char &SIOptimizeVGPRLiveRangeLegacyID;

void initializeAMDGPUAnnotateUniformValuesLegacyPass(PassRegistry &);
extern char &AMDGPUAnnotateUniformValuesLegacyPassID;

void initializeAMDGPUCodeGenPreparePass(PassRegistry&);
extern char &AMDGPUCodeGenPrepareID;

void initializeAMDGPURemoveIncompatibleFunctionsLegacyPass(PassRegistry &);
extern char &AMDGPURemoveIncompatibleFunctionsID;

void initializeAMDGPULateCodeGenPrepareLegacyPass(PassRegistry &);
extern char &AMDGPULateCodeGenPrepareLegacyID;

FunctionPass *createAMDGPURewriteUndefForPHILegacyPass();
void initializeAMDGPURewriteUndefForPHILegacyPass(PassRegistry &);
extern char &AMDGPURewriteUndefForPHILegacyPassID;

class AMDGPURewriteUndefForPHIPass
    : public PassInfoMixin<AMDGPURewriteUndefForPHIPass> {
public:
  AMDGPURewriteUndefForPHIPass() = default;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class SIAnnotateControlFlowPass
    : public PassInfoMixin<SIAnnotateControlFlowPass> {
private:
  const AMDGPUTargetMachine &TM;

public:
  SIAnnotateControlFlowPass(const AMDGPUTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeSIAnnotateControlFlowLegacyPass(PassRegistry &);
extern char &SIAnnotateControlFlowLegacyPassID;

void initializeSIMemoryLegalizerLegacyPass(PassRegistry &);
extern char &SIMemoryLegalizerID;

void initializeSIModeRegisterLegacyPass(PassRegistry &);
extern char &SIModeRegisterID;

void initializeAMDGPUInsertDelayAluLegacyPass(PassRegistry &);
extern char &AMDGPUInsertDelayAluID;

void initializeAMDGPUInsertSingleUseVDSTPass(PassRegistry &);
extern char &AMDGPUInsertSingleUseVDSTID;

void initializeAMDGPULowerVGPREncodingPass(PassRegistry &);
extern char &AMDGPULowerVGPREncodingID;

void initializeSIInsertHardClausesLegacyPass(PassRegistry &);
extern char &SIInsertHardClausesID;

void initializeSIInsertWaitcntsLegacyPass(PassRegistry &);
extern char &SIInsertWaitcntsID;

void initializeSIFormMemoryClausesLegacyPass(PassRegistry &);
extern char &SIFormMemoryClausesID;

void initializeSIPostRABundlerLegacyPass(PassRegistry &);
extern char &SIPostRABundlerLegacyID;

void initializeGCNCreateVOPDLegacyPass(PassRegistry &);
extern char &GCNCreateVOPDID;

void initializeAMDGPUUnifyDivergentExitNodesPass(PassRegistry&);
extern char &AMDGPUUnifyDivergentExitNodesID;

ImmutablePass *createAMDGPUAAWrapperPass();
void initializeAMDGPUAAWrapperPassPass(PassRegistry&);
ImmutablePass *createAMDGPUExternalAAWrapperPass();
void initializeAMDGPUExternalAAWrapperPass(PassRegistry&);

void initializeAMDGPUArgumentUsageInfoPass(PassRegistry &);

ModulePass *createAMDGPUExportKernelRuntimeHandlesLegacyPass();
void initializeAMDGPUExportKernelRuntimeHandlesLegacyPass(PassRegistry &);
extern char &AMDGPUExportKernelRuntimeHandlesLegacyID;

void initializeGCNNSAReassignLegacyPass(PassRegistry &);
extern char &GCNNSAReassignID;

void initializeGCNPreRALongBranchRegLegacyPass(PassRegistry &);
extern char &GCNPreRALongBranchRegID;

void initializeGCNPreRAOptimizationsLegacyPass(PassRegistry &);
extern char &GCNPreRAOptimizationsID;

FunctionPass *createAMDGPUSetWavePriorityPass();
void initializeAMDGPUSetWavePriorityLegacyPass(PassRegistry &);

void initializeGCNRewritePartialRegUsesLegacyPass(llvm::PassRegistry &);
extern char &GCNRewritePartialRegUsesID;

void initializeAMDGPUWaitSGPRHazardsLegacyPass(PassRegistry &);
extern char &AMDGPUWaitSGPRHazardsLegacyID;

namespace AMDGPU {
enum TargetIndex {
  TI_CONSTDATA_START,
  TI_SCRATCH_RSRC_DWORD0,
  TI_SCRATCH_RSRC_DWORD1,
  TI_SCRATCH_RSRC_DWORD2,
  TI_SCRATCH_RSRC_DWORD3,
  TI_NUM_VGPRS,
  TI_NUM_VGPRS_RANK0,
  TI_NUM_VGPRS_RANK1,
  TI_NUM_VGPRS_RANK2,
  TI_NUM_VGPRS_RANK3,
  TI_NUM_VGPRS_RANK4,
  TI_NUM_VGPRS_RANK5,
  TI_NUM_VGPRS_RANK6,
  TI_NUM_VGPRS_RANK7,
};

static inline bool addrspacesMayAlias(unsigned AS1, unsigned AS2) {

  if (AS1 > AMDGPUAS::MAX_AMDGPU_ADDRESS || AS2 > AMDGPUAS::MAX_AMDGPU_ADDRESS)
    return true;

  // clang-format off
  static const bool ASAliasRules[][AMDGPUAS::MAX_AMDGPU_ADDRESS + 1] = {
    /*                       Flat   Global Region  Local Constant Private Const32 BufFatPtr BufRsrc BufStrdPtr LaneShared Distributed*/
    /* Flat     */            {true,  true,  false, true,  true,  true,  true,  true,  true,  true,  true,  true},
    /* Global   */            {true,  true,  false, false, true,  false, true,  true,  true,  true,  false, false},
    /* Region   */            {false, false, true,  false, false, false, false, false, false, false, false, false},
    /* Local    */            {true,  false, false, true,  false, false, false, false, false, false, false, true},
    /* Constant */            {true,  true,  false, false, false, false, true,  true,  true,  true,  false, false},
    /* Private  */            {true,  false, false, false, false, true,  false, false, false, false, false, false},
    /* Constant 32-bit */     {true,  true,  false, false, true,  false, false, true,  true,  true,  false, false},
    /* Buffer Fat Ptr  */     {true,  true,  false, false, true,  false, true,  true,  true,  true,  false, false},
    /* Buffer Resource */     {true,  true,  false, false, true,  false, true,  true,  true,  true,  false, false},
    /* Buffer Strided Ptr  */ {true,  true,  false, false, true,  false, true,  true,  true,  true,  false, false},
    /* Lane Shared */         {true,  false, false, false, false, false, false, false, false, false, true,  false},
    /* Distributed */         {true,  false, false, true,  false, false, false, false, false, false, false, true},
  };
  // clang-format on
  static_assert(std::size(ASAliasRules) == AMDGPUAS::MAX_AMDGPU_ADDRESS + 1);

  return ASAliasRules[AS1][AS2];
}

}

} // End namespace llvm

#endif
