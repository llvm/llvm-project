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
FunctionPass *createSIModeRegisterPass();
FunctionPass *createGCNPreRAOptimizationsLegacyPass();
FunctionPass *createAMDGPUPreloadKernArgPrologLegacyPass();

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

Pass *createAMDGPUAnnotateKernelFeaturesPass();
Pass *createAMDGPUAttributorLegacyPass();
void initializeAMDGPUAttributorLegacyPass(PassRegistry &);
void initializeAMDGPUAnnotateKernelFeaturesPass(PassRegistry &);
extern char &AMDGPUAnnotateKernelFeaturesID;

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

void initializeAMDGPULowerBufferFatPointersPass(PassRegistry &);
extern char &AMDGPULowerBufferFatPointersID;

struct AMDGPULowerBufferFatPointersPass
    : PassInfoMixin<AMDGPULowerBufferFatPointersPass> {
  AMDGPULowerBufferFatPointersPass(const TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  const TargetMachine &TM;
};

void initializeAMDGPUReserveWWMRegsPass(PassRegistry &);
extern char &AMDGPUReserveWWMRegsID;

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

void initializeAMDGPUMarkLastScratchLoadPass(PassRegistry &);
extern char &AMDGPUMarkLastScratchLoadID;

void initializeSILowerSGPRSpillsLegacyPass(PassRegistry &);
extern char &SILowerSGPRSpillsLegacyID;

void initializeSILoadStoreOptimizerLegacyPass(PassRegistry &);
extern char &SILoadStoreOptimizerLegacyID;

void initializeSIWholeQuadModeLegacyPass(PassRegistry &);
extern char &SIWholeQuadModeID;

void initializeSILowerControlFlowLegacyPass(PassRegistry &);
extern char &SILowerControlFlowLegacyID;

void initializeSIPreEmitPeepholePass(PassRegistry &);
extern char &SIPreEmitPeepholeID;

void initializeSILateBranchLoweringPass(PassRegistry &);
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

public:
  AMDGPUAttributorPass(TargetMachine &TM, AMDGPUAttributorOptions Options = {})
      : TM(TM), Options(Options) {};
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

class GCNCreateVOPDPass : public PassInfoMixin<GCNCreateVOPDPass> {
public:
  PreservedAnalyses run(MachineFunction &MF, MachineFunctionAnalysisManager &AM);
};

class SIMemoryLegalizerPass : public PassInfoMixin<SIMemoryLegalizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
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

void initializeSIInsertHardClausesPass(PassRegistry &);
extern char &SIInsertHardClausesID;

void initializeSIInsertWaitcntsPass(PassRegistry&);
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

ModulePass *createAMDGPUOpenCLEnqueuedBlockLoweringLegacyPass();
void initializeAMDGPUOpenCLEnqueuedBlockLoweringLegacyPass(PassRegistry &);
extern char &AMDGPUOpenCLEnqueuedBlockLoweringLegacyID;

void initializeGCNNSAReassignLegacyPass(PassRegistry &);
extern char &GCNNSAReassignID;

void initializeGCNPreRALongBranchRegLegacyPass(PassRegistry &);
extern char &GCNPreRALongBranchRegID;

void initializeGCNPreRAOptimizationsLegacyPass(PassRegistry &);
extern char &GCNPreRAOptimizationsID;

FunctionPass *createAMDGPUSetWavePriorityPass();
void initializeAMDGPUSetWavePriorityPass(PassRegistry &);

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
  TI_SCRATCH_RSRC_DWORD3
};

static inline bool addrspacesMayAlias(unsigned AS1, unsigned AS2) {
  static_assert(AMDGPUAS::MAX_AMDGPU_ADDRESS <= 9, "Addr space out of range");

  if (AS1 > AMDGPUAS::MAX_AMDGPU_ADDRESS || AS2 > AMDGPUAS::MAX_AMDGPU_ADDRESS)
    return true;

  // This array is indexed by address space value enum elements 0 ... to 9
  // clang-format off
  static const bool ASAliasRules[10][10] = {
    /*                       Flat   Global Region  Group Constant Private Const32 BufFatPtr BufRsrc BufStrdPtr */
    /* Flat     */            {true,  true,  false, true,  true,  true,  true,  true,  true,  true},
    /* Global   */            {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Region   */            {false, false, true,  false, false, false, false, false, false, false},
    /* Group    */            {true,  false, false, true,  false, false, false, false, false, false},
    /* Constant */            {true,  true,  false, false, false, false, true,  true,  true,  true},
    /* Private  */            {true,  false, false, false, false, true,  false, false, false, false},
    /* Constant 32-bit */     {true,  true,  false, false, true,  false, false, true,  true,  true},
    /* Buffer Fat Ptr  */     {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Buffer Resource */     {true,  true,  false, false, true,  false, true,  true,  true,  true},
    /* Buffer Strided Ptr  */ {true,  true,  false, false, true,  false, true,  true,  true,  true},
  };
  // clang-format on

  return ASAliasRules[AS1][AS2];
}

}

} // End namespace llvm

#endif
