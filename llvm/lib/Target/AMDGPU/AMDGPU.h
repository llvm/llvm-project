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
ModulePass *createAMDGPULowerBufferFatPointersPass();
ModulePass *createAMDGPULowerIntrinsicsLegacyPass();
FunctionPass *createSIModeRegisterPass();
FunctionPass *createGCNPreRAOptimizationsLegacyPass();
FunctionPass *createAMDGPUPreloadKernArgPrologLegacyPass();
ModulePass *createAMDGPUPreloadKernelArgumentsLegacyPass(const TargetMachine *);

struct AMDGPUSimplifyLibCallsPass : PassInfoMixin<AMDGPUSimplifyLibCallsPass> {
  AMDGPUSimplifyLibCallsPass() = default;
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

// DPP/Iterative option enables the atomic optimizer with given strategy
// whereas None disables the atomic optimizer.
enum class ScanOptions { DPP, Iterative, None };
FunctionPass *createAMDGPUAtomicOptimizerPass(ScanOptions ScanStrategy);
void initializeAMDGPUAtomicOptimizerPass(PassRegistry &);
extern const char &AMDGPUAtomicOptimizerID;

ModulePass *createAMDGPUCtorDtorLoweringLegacyPass();
void initializeAMDGPUCtorDtorLoweringLegacyPass(PassRegistry &);
extern const char &AMDGPUCtorDtorLoweringLegacyPassID;

FunctionPass *createAMDGPULowerKernelArgumentsPass();
void initializeAMDGPULowerKernelArgumentsPass(PassRegistry &);
extern const char &AMDGPULowerKernelArgumentsID;

FunctionPass *createAMDGPUPromoteKernelArgumentsPass();
void initializeAMDGPUPromoteKernelArgumentsPass(PassRegistry &);
extern const char &AMDGPUPromoteKernelArgumentsID;

struct AMDGPUPromoteKernelArgumentsPass
    : PassInfoMixin<AMDGPUPromoteKernelArgumentsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

ModulePass *createAMDGPULowerKernelAttributesPass();
void initializeAMDGPULowerKernelAttributesPass(PassRegistry &);
extern const char &AMDGPULowerKernelAttributesID;

struct AMDGPULowerKernelAttributesPass
    : PassInfoMixin<AMDGPULowerKernelAttributesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

void initializeAMDGPULowerModuleLDSLegacyPass(PassRegistry &);
extern const char &AMDGPULowerModuleLDSLegacyPassID;

struct AMDGPULowerModuleLDSPass : PassInfoMixin<AMDGPULowerModuleLDSPass> {
  const AMDGPUTargetMachine &TM;
  AMDGPULowerModuleLDSPass(const AMDGPUTargetMachine &TM_) : TM(TM_) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeAMDGPULowerBufferFatPointersPass(PassRegistry &);
extern const char &AMDGPULowerBufferFatPointersID;

struct AMDGPULowerBufferFatPointersPass
    : PassInfoMixin<AMDGPULowerBufferFatPointersPass> {
  AMDGPULowerBufferFatPointersPass(const TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  const TargetMachine &TM;
};

void initializeAMDGPULowerIntrinsicsLegacyPass(PassRegistry &);

struct AMDGPULowerIntrinsicsPass : PassInfoMixin<AMDGPULowerIntrinsicsPass> {
  AMDGPULowerIntrinsicsPass(const AMDGPUTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

private:
  const AMDGPUTargetMachine &TM;
};

void initializeAMDGPUPrepareAGPRAllocLegacyPass(PassRegistry &);
extern const char &AMDGPUPrepareAGPRAllocLegacyID;

void initializeAMDGPUReserveWWMRegsLegacyPass(PassRegistry &);
extern const char &AMDGPUReserveWWMRegsLegacyID;

void initializeAMDGPURewriteOutArgumentsPass(PassRegistry &);
extern const char &AMDGPURewriteOutArgumentsID;

void initializeGCNDPPCombineLegacyPass(PassRegistry &);
extern const char &GCNDPPCombineLegacyID;

void initializeSIFoldOperandsLegacyPass(PassRegistry &);
extern const char &SIFoldOperandsLegacyID;

void initializeSIPeepholeSDWALegacyPass(PassRegistry &);
extern const char &SIPeepholeSDWALegacyID;

void initializeSIShrinkInstructionsLegacyPass(PassRegistry &);
extern const char &SIShrinkInstructionsLegacyID;

void initializeSIFixSGPRCopiesLegacyPass(PassRegistry &);
extern const char &SIFixSGPRCopiesLegacyID;

void initializeSIFixVGPRCopiesLegacyPass(PassRegistry &);
extern const char &SIFixVGPRCopiesID;

void initializeSILowerWWMCopiesLegacyPass(PassRegistry &);
extern const char &SILowerWWMCopiesLegacyID;

void initializeSILowerI1CopiesLegacyPass(PassRegistry &);
extern const char &SILowerI1CopiesLegacyID;

void initializeAMDGPUGlobalISelDivergenceLoweringPass(PassRegistry &);
extern const char &AMDGPUGlobalISelDivergenceLoweringID;

void initializeAMDGPURegBankSelectPass(PassRegistry &);
extern const char &AMDGPURegBankSelectID;

void initializeAMDGPURegBankLegalizePass(PassRegistry &);
extern const char &AMDGPURegBankLegalizeID;

void initializeAMDGPUMarkLastScratchLoadLegacyPass(PassRegistry &);
extern const char &AMDGPUMarkLastScratchLoadID;

void initializeSILowerSGPRSpillsLegacyPass(PassRegistry &);
extern const char &SILowerSGPRSpillsLegacyID;

void initializeSILoadStoreOptimizerLegacyPass(PassRegistry &);
extern const char &SILoadStoreOptimizerLegacyID;

void initializeSIWholeQuadModeLegacyPass(PassRegistry &);
extern const char &SIWholeQuadModeID;

void initializeSILowerControlFlowLegacyPass(PassRegistry &);
extern const char &SILowerControlFlowLegacyID;

void initializeSIPreEmitPeepholeLegacyPass(PassRegistry &);
extern const char &SIPreEmitPeepholeID;

void initializeSILateBranchLoweringLegacyPass(PassRegistry &);
extern const char &SILateBranchLoweringPassID;

void initializeSIOptimizeExecMaskingLegacyPass(PassRegistry &);
extern const char &SIOptimizeExecMaskingLegacyID;

void initializeSIPreAllocateWWMRegsLegacyPass(PassRegistry &);
extern const char &SIPreAllocateWWMRegsLegacyID;

void initializeAMDGPUImageIntrinsicOptimizerPass(PassRegistry &);
extern const char &AMDGPUImageIntrinsicOptimizerID;

void initializeAMDGPUPerfHintAnalysisLegacyPass(PassRegistry &);
extern const char &AMDGPUPerfHintAnalysisLegacyID;

void initializeGCNRegPressurePrinterPass(PassRegistry &);
extern const char &GCNRegPressurePrinterID;

void initializeAMDGPUPreloadKernArgPrologLegacyPass(PassRegistry &);
extern const char &AMDGPUPreloadKernArgPrologLegacyID;

void initializeAMDGPUPreloadKernelArgumentsLegacyPass(PassRegistry &);
extern const char &AMDGPUPreloadKernelArgumentsLegacyID;

// Passes common to R600 and SI
FunctionPass *createAMDGPUPromoteAlloca();
void initializeAMDGPUPromoteAllocaPass(PassRegistry&);
extern const char &AMDGPUPromoteAllocaID;

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

void initializeAMDGPULowerExecSyncLegacyPass(PassRegistry &);
extern const char &AMDGPULowerExecSyncLegacyPassID;
ModulePass *createAMDGPULowerExecSyncLegacyPass();

struct AMDGPULowerExecSyncPass : PassInfoMixin<AMDGPULowerExecSyncPass> {
  AMDGPULowerExecSyncPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeAMDGPUSwLowerLDSLegacyPass(PassRegistry &);
extern const char &AMDGPUSwLowerLDSLegacyPassID;
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

class AMDGPUPreloadKernelArgumentsPass
    : public PassInfoMixin<AMDGPUPreloadKernelArgumentsPass> {
  const TargetMachine &TM;

public:
  explicit AMDGPUPreloadKernelArgumentsPass(const TargetMachine &TM) : TM(TM) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AMDGPUAnnotateUniformValuesPass
    : public PassInfoMixin<AMDGPUAnnotateUniformValuesPass> {
public:
  AMDGPUAnnotateUniformValuesPass() = default;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class SIModeRegisterPass : public PassInfoMixin<SIModeRegisterPass> {
public:
  SIModeRegisterPass() = default;
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
extern const char &AMDGPUPrintfRuntimeBindingID;

void initializeAMDGPUResourceUsageAnalysisWrapperPassPass(PassRegistry &);
extern const char &AMDGPUResourceUsageAnalysisID;

struct AMDGPUPrintfRuntimeBindingPass
    : PassInfoMixin<AMDGPUPrintfRuntimeBindingPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

void initializeSIOptimizeExecMaskingPreRALegacyPass(PassRegistry &);
extern const char &SIOptimizeExecMaskingPreRAID;

void initializeSIOptimizeVGPRLiveRangeLegacyPass(PassRegistry &);
extern const char &SIOptimizeVGPRLiveRangeLegacyID;

void initializeAMDGPUAnnotateUniformValuesLegacyPass(PassRegistry &);
extern const char &AMDGPUAnnotateUniformValuesLegacyPassID;

void initializeAMDGPUCodeGenPreparePass(PassRegistry&);
extern const char &AMDGPUCodeGenPrepareID;

void initializeAMDGPURemoveIncompatibleFunctionsLegacyPass(PassRegistry &);
extern const char &AMDGPURemoveIncompatibleFunctionsID;

void initializeAMDGPULateCodeGenPrepareLegacyPass(PassRegistry &);
extern const char &AMDGPULateCodeGenPrepareLegacyID;

FunctionPass *createAMDGPURewriteUndefForPHILegacyPass();
void initializeAMDGPURewriteUndefForPHILegacyPass(PassRegistry &);
extern const char &AMDGPURewriteUndefForPHILegacyPassID;

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
extern const char &SIAnnotateControlFlowLegacyPassID;

void initializeSIMemoryLegalizerLegacyPass(PassRegistry &);
extern const char &SIMemoryLegalizerID;

void initializeSIModeRegisterLegacyPass(PassRegistry &);
extern const char &SIModeRegisterID;

void initializeAMDGPUInsertDelayAluLegacyPass(PassRegistry &);
extern const char &AMDGPUInsertDelayAluID;

void initializeAMDGPULowerVGPREncodingLegacyPass(PassRegistry &);
extern const char &AMDGPULowerVGPREncodingLegacyID;

void initializeSIInsertHardClausesLegacyPass(PassRegistry &);
extern const char &SIInsertHardClausesID;

void initializeSIInsertWaitcntsLegacyPass(PassRegistry &);
extern const char &SIInsertWaitcntsID;

void initializeSIFormMemoryClausesLegacyPass(PassRegistry &);
extern const char &SIFormMemoryClausesID;

void initializeSIPostRABundlerLegacyPass(PassRegistry &);
extern const char &SIPostRABundlerLegacyID;

void initializeGCNCreateVOPDLegacyPass(PassRegistry &);
extern const char &GCNCreateVOPDID;

void initializeAMDGPUUnifyDivergentExitNodesPass(PassRegistry&);
extern const char &AMDGPUUnifyDivergentExitNodesID;

ImmutablePass *createAMDGPUAAWrapperPass();
void initializeAMDGPUAAWrapperPassPass(PassRegistry&);
ImmutablePass *createAMDGPUExternalAAWrapperPass();
void initializeAMDGPUExternalAAWrapperPass(PassRegistry&);

void initializeAMDGPUArgumentUsageInfoWrapperLegacyPass(PassRegistry &);

ModulePass *createAMDGPUExportKernelRuntimeHandlesLegacyPass();
void initializeAMDGPUExportKernelRuntimeHandlesLegacyPass(PassRegistry &);
extern const char &AMDGPUExportKernelRuntimeHandlesLegacyID;

void initializeGCNNSAReassignLegacyPass(PassRegistry &);
extern const char &GCNNSAReassignID;

void initializeGCNPreRALongBranchRegLegacyPass(PassRegistry &);
extern const char &GCNPreRALongBranchRegID;

void initializeGCNPreRAOptimizationsLegacyPass(PassRegistry &);
extern const char &GCNPreRAOptimizationsID;

FunctionPass *createAMDGPUSetWavePriorityPass();
void initializeAMDGPUSetWavePriorityLegacyPass(PassRegistry &);

void initializeGCNRewritePartialRegUsesLegacyPass(llvm::PassRegistry &);
extern const char &GCNRewritePartialRegUsesID;

void initializeAMDGPUWaitSGPRHazardsLegacyPass(PassRegistry &);
extern const char &AMDGPUWaitSGPRHazardsLegacyID;

class AMDGPURewriteAGPRCopyMFMAPass
    : public PassInfoMixin<AMDGPURewriteAGPRCopyMFMAPass> {
public:
  AMDGPURewriteAGPRCopyMFMAPass() = default;
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

void initializeAMDGPURewriteAGPRCopyMFMALegacyPass(PassRegistry &);
extern const char &AMDGPURewriteAGPRCopyMFMALegacyID;

void initializeAMDGPUUniformIntrinsicCombineLegacyPass(PassRegistry &);
extern const char &AMDGPUUniformIntrinsicCombineLegacyPassID;
FunctionPass *createAMDGPUUniformIntrinsicCombineLegacyPass();

struct AMDGPUUniformIntrinsicCombinePass
    : public PassInfoMixin<AMDGPUUniformIntrinsicCombinePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

namespace AMDGPU {
enum TargetIndex {
  TI_CONSTDATA_START,
  TI_SCRATCH_RSRC_DWORD0,
  TI_SCRATCH_RSRC_DWORD1,
  TI_SCRATCH_RSRC_DWORD2,
  TI_SCRATCH_RSRC_DWORD3
};

static inline bool addrspacesMayAlias(unsigned AS1, unsigned AS2) {
  if (AS1 > AMDGPUAS::MAX_AMDGPU_ADDRESS || AS2 > AMDGPUAS::MAX_AMDGPU_ADDRESS)
    return true;

  // clang-format off
  static const bool ASAliasRules[][AMDGPUAS::MAX_AMDGPU_ADDRESS + 1] = {
    /*                       Flat   Global Region  Local Constant Private Const32 BufFatPtr BufRsrc BufStrdPtr */
    /* Flat     */            {true,  true,  false, true,  true,  true,  true,  true,  true,  true},
    /* Global   */            {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Region   */            {false, false, true,  false, false, false, false, false, false, false},
    /* Local    */            {true,  false, false, true,  false, false, false, false, false, false},
    /* Constant */            {true,  true,  false, false, false, false, true,  true,  true,  true},
    /* Private  */            {true,  false, false, false, false, true,  false, false, false, false},
    /* Constant 32-bit */     {true,  true,  false, false, true,  false, false, true,  true,  true},
    /* Buffer Fat Ptr  */     {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Buffer Resource */     {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Buffer Strided Ptr  */ {true,  true,  false, false, true,  false, true,  true,  true,  true},
  };
  // clang-format on
  static_assert(std::size(ASAliasRules) == AMDGPUAS::MAX_AMDGPU_ADDRESS + 1);

  return ASAliasRules[AS1][AS2];
}

}

} // End namespace llvm

#endif
