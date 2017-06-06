//===-- CodeGen.cpp -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common initialization routines for the
// CodeGen library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Initialization.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void llvm::initializeCodeGen(PassRegistry &Registry) {
  initializeAtomicExpandPass(Registry);
  initializeBranchCoalescingPass(Registry);
  initializeBranchFolderPassPass(Registry);
  initializeBranchRelaxationPass(Registry);
  initializeCodeGenPreparePass(Registry);
  initializeCountingFunctionInserterPass(Registry);
  initializeDeadMachineInstructionElimPass(Registry);
  initializeDetectDeadLanesPass(Registry);
  initializeDwarfEHPreparePass(Registry);
  initializeEarlyIfConverterPass(Registry);
  initializeExpandISelPseudosPass(Registry);
  initializeExpandPostRAPass(Registry);
  initializeFEntryInserterPass(Registry);
  initializeFinalizeMachineBundlesPass(Registry);
  initializeFuncletLayoutPass(Registry);
  initializeGCMachineCodeAnalysisPass(Registry);
  initializeGCModuleInfoPass(Registry);
  initializeIfConverterPass(Registry);
  initializeImplicitNullChecksPass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeLiveDebugValuesPass(Registry);
  initializeLiveDebugVariablesPass(Registry);
  initializeLiveIntervalsPass(Registry);
  initializeLiveRangeShrinkPass(Registry);
  initializeLiveStacksPass(Registry);
  initializeLiveVariablesPass(Registry);
  initializeLocalStackSlotPassPass(Registry);
  initializeLowerIntrinsicsPass(Registry);
  initializeMachineBlockFrequencyInfoPass(Registry);
  initializeMachineBlockPlacementPass(Registry);
  initializeMachineBlockPlacementStatsPass(Registry);
  initializeMachineCSEPass(Registry);
  initializeMachineCombinerPass(Registry);
  initializeMachineCopyPropagationPass(Registry);
  initializeMachineDominatorTreePass(Registry);
  initializeMachineFunctionPrinterPassPass(Registry);
  initializeMachineLICMPass(Registry);
  initializeMachineLoopInfoPass(Registry);
  initializeMachineModuleInfoPass(Registry);
  initializeMachineOptimizationRemarkEmitterPassPass(Registry);
  initializeMachineOutlinerPass(Registry);
  initializeMachinePipelinerPass(Registry);
  initializeMachinePostDominatorTreePass(Registry);
  initializeMachineRegionInfoPassPass(Registry);
  initializeMachineSchedulerPass(Registry);
  initializeMachineSinkingPass(Registry);
  initializeMachineVerifierPassPass(Registry);
  initializeOptimizePHIsPass(Registry);
  initializePEIPass(Registry);
  initializePHIEliminationPass(Registry);
  initializePatchableFunctionPass(Registry);
  initializePeepholeOptimizerPass(Registry);
  initializePostMachineSchedulerPass(Registry);
  initializePostRAHazardRecognizerPass(Registry);
  initializePostRASchedulerPass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  initializeProcessImplicitDefsPass(Registry);
  initializeRABasicPass(Registry);
  initializeRAGreedyPass(Registry);
  initializeRegisterCoalescerPass(Registry);
  initializeRenameIndependentSubregsPass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeScalarizeMaskedMemIntrinPass(Registry);
  initializeShrinkWrapPass(Registry);
  initializeSlotIndexesPass(Registry);
  initializeStackColoringPass(Registry);
  initializeStackMapLivenessPass(Registry);
  initializeStackProtectorPass(Registry);
  initializeStackSlotColoringPass(Registry);
  initializeTailDuplicatePassPass(Registry);
  initializeTargetPassConfigPass(Registry);
  initializeTwoAddressInstructionPassPass(Registry);
  initializeUnpackMachineBundlesPass(Registry);
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeUnreachableMachineBlockElimPass(Registry);
  initializeVirtRegMapPass(Registry);
  initializeVirtRegRewriterPass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeXRayInstrumentationPass(Registry);
}

void LLVMInitializeCodeGen(LLVMPassRegistryRef R) {
  initializeCodeGen(*unwrap(R));
}
