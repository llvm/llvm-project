//===-- AMDGPU.h - MachineFunction passes hw codegen --------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPU_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPU_H

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AMDGPUTargetMachine;
class FunctionPass;
class GCNTargetMachine;
class ModulePass;
class Pass;
class Target;
class TargetMachine;
class PassRegistry;
class Module;

// R600 Passes
FunctionPass *createR600VectorRegMerger();
FunctionPass *createR600ExpandSpecialInstrsPass();
FunctionPass *createR600EmitClauseMarkers();
FunctionPass *createR600ClauseMergePass();
FunctionPass *createR600Packetizer();
FunctionPass *createR600ControlFlowFinalizer();
FunctionPass *createAMDGPUCFGStructurizerPass();

// SI Passes
FunctionPass *createSITypeRewriter();
FunctionPass *createSIAnnotateControlFlowPass();
FunctionPass *createSIFoldOperandsPass();
FunctionPass *createSIPeepholeSDWAPass();
FunctionPass *createSILowerI1CopiesPass();
FunctionPass *createSIShrinkInstructionsPass();
FunctionPass *createSILoadStoreOptimizerPass();
FunctionPass *createSIWholeQuadModePass();
FunctionPass *createSIFixControlFlowLiveIntervalsPass();
FunctionPass *createSIFixSGPRCopiesPass();
FunctionPass *createSIDebuggerInsertNopsPass();
FunctionPass *createSIInsertWaitsPass();
FunctionPass *createSIInsertWaitcntsPass();
FunctionPass *createAMDGPUCodeGenPreparePass();
FunctionPass *createAMDGPUMachineCFGStructurizerPass();

void initializeAMDGPUMachineCFGStructurizerPass(PassRegistry&);
extern char &AMDGPUMachineCFGStructurizerID;

void initializeAMDGPUAlwaysInlinePass(PassRegistry&);

ModulePass *createAMDGPUAnnotateKernelFeaturesPass();
void initializeAMDGPUAnnotateKernelFeaturesPass(PassRegistry &);
extern char &AMDGPUAnnotateKernelFeaturesID;

ModulePass *createAMDGPULowerIntrinsicsPass();
void initializeAMDGPULowerIntrinsicsPass(PassRegistry &);
extern char &AMDGPULowerIntrinsicsID;

void initializeSIFoldOperandsPass(PassRegistry &);
extern char &SIFoldOperandsID;

void initializeSIPeepholeSDWAPass(PassRegistry &);
extern char &SIPeepholeSDWAID;

void initializeSIShrinkInstructionsPass(PassRegistry&);
extern char &SIShrinkInstructionsID;

void initializeSIFixSGPRCopiesPass(PassRegistry &);
extern char &SIFixSGPRCopiesID;

void initializeSIFixVGPRCopiesPass(PassRegistry &);
extern char &SIFixVGPRCopiesID;

void initializeSILowerI1CopiesPass(PassRegistry &);
extern char &SILowerI1CopiesID;

void initializeSILoadStoreOptimizerPass(PassRegistry &);
extern char &SILoadStoreOptimizerID;

void initializeSIWholeQuadModePass(PassRegistry &);
extern char &SIWholeQuadModeID;

void initializeSILowerControlFlowPass(PassRegistry &);
extern char &SILowerControlFlowID;

void initializeSIInsertSkipsPass(PassRegistry &);
extern char &SIInsertSkipsPassID;

void initializeSIOptimizeExecMaskingPass(PassRegistry &);
extern char &SIOptimizeExecMaskingID;

// Passes common to R600 and SI
FunctionPass *createAMDGPUPromoteAlloca();
void initializeAMDGPUPromoteAllocaPass(PassRegistry&);
extern char &AMDGPUPromoteAllocaID;

Pass *createAMDGPUStructurizeCFGPass();
FunctionPass *createAMDGPUISelDag(TargetMachine &TM,
                                  CodeGenOpt::Level OptLevel);
ModulePass *createAMDGPUAlwaysInlinePass(bool GlobalOpt = true);
ModulePass *createAMDGPUOpenCLImageTypeLoweringPass();
FunctionPass *createAMDGPUAnnotateUniformValues();

ModulePass* createAMDGPUUnifyMetadataPass();
void initializeAMDGPUUnifyMetadataPass(PassRegistry&);
extern char &AMDGPUUnifyMetadataID;

void initializeSIFixControlFlowLiveIntervalsPass(PassRegistry&);
extern char &SIFixControlFlowLiveIntervalsID;

void initializeAMDGPUAnnotateUniformValuesPass(PassRegistry&);
extern char &AMDGPUAnnotateUniformValuesPassID;

void initializeAMDGPUCodeGenPreparePass(PassRegistry&);
extern char &AMDGPUCodeGenPrepareID;

void initializeSIAnnotateControlFlowPass(PassRegistry&);
extern char &SIAnnotateControlFlowPassID;

void initializeSIDebuggerInsertNopsPass(PassRegistry&);
extern char &SIDebuggerInsertNopsID;

void initializeSIInsertWaitsPass(PassRegistry&);
extern char &SIInsertWaitsID;

void initializeSIInsertWaitcntsPass(PassRegistry&);
extern char &SIInsertWaitcntsID;

void initializeAMDGPUUnifyDivergentExitNodesPass(PassRegistry&);
extern char &AMDGPUUnifyDivergentExitNodesID;

ImmutablePass *createAMDGPUAAWrapperPass();
void initializeAMDGPUAAWrapperPassPass(PassRegistry&);

Target &getTheAMDGPUTarget();
Target &getTheGCNTarget();

namespace AMDGPU {
enum TargetIndex {
  TI_CONSTDATA_START,
  TI_SCRATCH_RSRC_DWORD0,
  TI_SCRATCH_RSRC_DWORD1,
  TI_SCRATCH_RSRC_DWORD2,
  TI_SCRATCH_RSRC_DWORD3
};
}

} // End namespace llvm

/// OpenCL uses address spaces to differentiate between
/// various memory regions on the hardware. On the CPU
/// all of the address spaces point to the same memory,
/// however on the GPU, each address space points to
/// a separate piece of memory that is unique from other
/// memory locations.
struct AMDGPUAS {
  // The following address space values depend on the triple environment.
  unsigned PRIVATE_ADDRESS;  ///< Address space for private memory.
  unsigned FLAT_ADDRESS;     ///< Address space for flat memory.
  unsigned REGION_ADDRESS;   ///< Address space for region memory.

  // The maximum value for flat, generic, local, private, constant and region.
  const static unsigned MAX_COMMON_ADDRESS = 5;

  const static unsigned GLOBAL_ADDRESS   = 1;  ///< Address space for global memory (RAT0, VTX0).
  const static unsigned CONSTANT_ADDRESS = 2;  ///< Address space for constant memory (VTX2)
  const static unsigned LOCAL_ADDRESS    = 3;  ///< Address space for local memory.
  const static unsigned PARAM_D_ADDRESS  = 6;  ///< Address space for direct addressible parameter memory (CONST0)
  const static unsigned PARAM_I_ADDRESS  = 7;  ///< Address space for indirect addressible parameter memory (VTX1)

  // Do not re-order the CONSTANT_BUFFER_* enums.  Several places depend on this
  // order to be able to dynamically index a constant buffer, for example:
  //
  // ConstantBufferAS = CONSTANT_BUFFER_0 + CBIdx

  const static unsigned CONSTANT_BUFFER_0 = 8;
  const static unsigned CONSTANT_BUFFER_1 = 9;
  const static unsigned CONSTANT_BUFFER_2 = 10;
  const static unsigned CONSTANT_BUFFER_3 = 11;
  const static unsigned CONSTANT_BUFFER_4 = 12;
  const static unsigned CONSTANT_BUFFER_5 = 13;
  const static unsigned CONSTANT_BUFFER_6 = 14;
  const static unsigned CONSTANT_BUFFER_7 = 15;
  const static unsigned CONSTANT_BUFFER_8 = 16;
  const static unsigned CONSTANT_BUFFER_9 = 17;
  const static unsigned CONSTANT_BUFFER_10 = 18;
  const static unsigned CONSTANT_BUFFER_11 = 19;
  const static unsigned CONSTANT_BUFFER_12 = 20;
  const static unsigned CONSTANT_BUFFER_13 = 21;
  const static unsigned CONSTANT_BUFFER_14 = 22;
  const static unsigned CONSTANT_BUFFER_15 = 23;

  // Some places use this if the address space can't be determined.
  const static unsigned UNKNOWN_ADDRESS_SPACE = ~0u;
};

namespace llvm {
namespace AMDGPU {
AMDGPUAS getAMDGPUAS(const Module &M);
AMDGPUAS getAMDGPUAS(const TargetMachine &TM);
AMDGPUAS getAMDGPUAS(Triple T);
} // namespace AMDGPU
} // namespace llvm

#endif
