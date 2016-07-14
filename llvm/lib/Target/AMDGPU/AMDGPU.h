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

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AMDGPUInstrPrinter;
class AMDGPUSubtarget;
class AMDGPUTargetMachine;
class FunctionPass;
struct MachineSchedContext;
class MCAsmInfo;
class raw_ostream;
class ScheduleDAGInstrs;
class Target;
class TargetMachine;

// R600 Passes
FunctionPass *createR600VectorRegMerger(TargetMachine &tm);
FunctionPass *createR600ExpandSpecialInstrsPass(TargetMachine &tm);
FunctionPass *createR600EmitClauseMarkers();
FunctionPass *createR600ClauseMergePass(TargetMachine &tm);
FunctionPass *createR600Packetizer(TargetMachine &tm);
FunctionPass *createR600ControlFlowFinalizer(TargetMachine &tm);
FunctionPass *createAMDGPUCFGStructurizerPass();

// SI Passes
FunctionPass *createSITypeRewriter();
FunctionPass *createSIAnnotateControlFlowPass();
FunctionPass *createSIFoldOperandsPass();
FunctionPass *createSILowerI1CopiesPass();
FunctionPass *createSIShrinkInstructionsPass();
FunctionPass *createSILoadStoreOptimizerPass(TargetMachine &tm);
FunctionPass *createSIWholeQuadModePass();
FunctionPass *createSILowerControlFlowPass();
FunctionPass *createSIFixControlFlowLiveIntervalsPass();
FunctionPass *createSIFixSGPRCopiesPass();
FunctionPass *createSICodeEmitterPass(formatted_raw_ostream &OS);
FunctionPass *createSIDebuggerInsertNopsPass();
FunctionPass *createSIInsertWaitsPass();
FunctionPass *createAMDGPUCodeGenPreparePass(const TargetMachine *TM = nullptr);

ScheduleDAGInstrs *createSIMachineScheduler(MachineSchedContext *C);

ModulePass *createAMDGPUAnnotateKernelFeaturesPass();
void initializeAMDGPUAnnotateKernelFeaturesPass(PassRegistry &);
extern char &AMDGPUAnnotateKernelFeaturesID;

void initializeSIFoldOperandsPass(PassRegistry &);
extern char &SIFoldOperandsID;

void initializeSIShrinkInstructionsPass(PassRegistry&);
extern char &SIShrinkInstructionsID;

void initializeSIFixSGPRCopiesPass(PassRegistry &);
extern char &SIFixSGPRCopiesID;

void initializeSILowerI1CopiesPass(PassRegistry &);
extern char &SILowerI1CopiesID;

void initializeSILoadStoreOptimizerPass(PassRegistry &);
extern char &SILoadStoreOptimizerID;

void initializeSIWholeQuadModePass(PassRegistry &);
extern char &SIWholeQuadModeID;

void initializeSILowerControlFlowPass(PassRegistry &);
extern char &SILowerControlFlowPassID;


// Passes common to R600 and SI
FunctionPass *createAMDGPUPromoteAlloca(const TargetMachine *TM = nullptr);
void initializeAMDGPUPromoteAllocaPass(PassRegistry&);
extern char &AMDGPUPromoteAllocaID;

FunctionPass *createAMDGPUAddDivergenceMetadata(const AMDGPUSubtarget &ST);
Pass *createAMDGPUStructurizeCFGPass();
FunctionPass *createAMDGPUISelDag(TargetMachine &tm);
ModulePass *createAMDGPUAlwaysInlinePass();
ModulePass *createAMDGPUOpenCLImageTypeLoweringPass();
FunctionPass *createAMDGPUAnnotateUniformValues();

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

extern Target TheAMDGPUTarget;
extern Target TheGCNTarget;

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
namespace AMDGPUAS {
enum AddressSpaces : unsigned {
  PRIVATE_ADDRESS  = 0, ///< Address space for private memory.
  GLOBAL_ADDRESS   = 1, ///< Address space for global memory (RAT0, VTX0).
  CONSTANT_ADDRESS = 2, ///< Address space for constant memory (VTX2)
  LOCAL_ADDRESS    = 3, ///< Address space for local memory.
  FLAT_ADDRESS     = 4, ///< Address space for flat memory.
  REGION_ADDRESS   = 5, ///< Address space for region memory.
  PARAM_D_ADDRESS  = 6, ///< Address space for direct addressible parameter memory (CONST0)
  PARAM_I_ADDRESS  = 7, ///< Address space for indirect addressible parameter memory (VTX1)

  // Do not re-order the CONSTANT_BUFFER_* enums.  Several places depend on this
  // order to be able to dynamically index a constant buffer, for example:
  //
  // ConstantBufferAS = CONSTANT_BUFFER_0 + CBIdx

  CONSTANT_BUFFER_0 = 8,
  CONSTANT_BUFFER_1 = 9,
  CONSTANT_BUFFER_2 = 10,
  CONSTANT_BUFFER_3 = 11,
  CONSTANT_BUFFER_4 = 12,
  CONSTANT_BUFFER_5 = 13,
  CONSTANT_BUFFER_6 = 14,
  CONSTANT_BUFFER_7 = 15,
  CONSTANT_BUFFER_8 = 16,
  CONSTANT_BUFFER_9 = 17,
  CONSTANT_BUFFER_10 = 18,
  CONSTANT_BUFFER_11 = 19,
  CONSTANT_BUFFER_12 = 20,
  CONSTANT_BUFFER_13 = 21,
  CONSTANT_BUFFER_14 = 22,
  CONSTANT_BUFFER_15 = 23,

  // Some places use this if the address space can't be determined.
  UNKNOWN_ADDRESS_SPACE = ~0u
};

} // namespace AMDGPUAS

#endif
