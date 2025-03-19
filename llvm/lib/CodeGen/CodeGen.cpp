//===-- CodeGen.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the common initialization routines for the
// CodeGen library.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void llvm::initializeCodeGen(PassRegistry &Registry) {
  initializeAssignmentTrackingAnalysisPass(Registry);
  initializeAtomicExpandLegacyPass(Registry);
  initializeBasicBlockPathCloningPass(Registry);
  initializeBasicBlockSectionsPass(Registry);
  initializeBranchFolderLegacyPass(Registry);
  initializeBranchRelaxationPass(Registry);
  initializeBreakFalseDepsPass(Registry);
  initializeCallBrPreparePass(Registry);
  initializeCFGuardLongjmpPass(Registry);
  initializeCFIFixupPass(Registry);
  initializeCFIInstrInserterPass(Registry);
  initializeCheckDebugMachineModulePass(Registry);
  initializeCodeGenPrepareLegacyPassPass(Registry);
  initializeDeadMachineInstructionElimPass(Registry);
  initializeDebugifyMachineModulePass(Registry);
  initializeDetectDeadLanesLegacyPass(Registry);
  initializeDwarfEHPrepareLegacyPassPass(Registry);
  initializeEarlyIfConverterLegacyPass(Registry);
  initializeEarlyIfPredicatorPass(Registry);
  initializeEarlyMachineLICMPass(Registry);
  initializeEarlyTailDuplicateLegacyPass(Registry);
  initializeExpandLargeDivRemLegacyPassPass(Registry);
  initializeExpandFpLegacyPassPass(Registry);
  initializeExpandMemCmpLegacyPassPass(Registry);
  initializeExpandPostRALegacyPass(Registry);
  initializeFEntryInserterLegacyPass(Registry);
  initializeFinalizeISelPass(Registry);
  initializeFinalizeMachineBundlesPass(Registry);
  initializeFixupStatepointCallerSavedLegacyPass(Registry);
  initializeFuncletLayoutPass(Registry);
  initializeGCMachineCodeAnalysisPass(Registry);
  initializeGCModuleInfoPass(Registry);
  initializeHardwareLoopsLegacyPass(Registry);
  initializeIfConverterPass(Registry);
  initializeImplicitNullChecksPass(Registry);
  initializeIndirectBrExpandLegacyPassPass(Registry);
  initializeInitUndefPass(Registry);
  initializeInterleavedLoadCombinePass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeJMCInstrumenterPass(Registry);
  initializeLiveDebugValuesPass(Registry);
  initializeLiveDebugVariablesWrapperLegacyPass(Registry);
  initializeLiveIntervalsWrapperPassPass(Registry);
  initializeLiveRangeShrinkPass(Registry);
  initializeLiveStacksWrapperLegacyPass(Registry);
  initializeLiveVariablesWrapperPassPass(Registry);
  initializeLocalStackSlotPassPass(Registry);
  initializeLowerGlobalDtorsLegacyPassPass(Registry);
  initializeLowerIntrinsicsPass(Registry);
  initializeMIRAddFSDiscriminatorsPass(Registry);
  initializeMIRCanonicalizerPass(Registry);
  initializeMIRNamerPass(Registry);
  initializeMIRProfileLoaderPassPass(Registry);
  initializeMachineBlockFrequencyInfoWrapperPassPass(Registry);
  initializeMachineBlockPlacementLegacyPass(Registry);
  initializeMachineBlockPlacementStatsPass(Registry);
  initializeMachineCFGPrinterPass(Registry);
  initializeMachineCSELegacyPass(Registry);
  initializeMachineCombinerPass(Registry);
  initializeMachineCopyPropagationLegacyPass(Registry);
  initializeMachineCycleInfoPrinterLegacyPass(Registry);
  initializeMachineCycleInfoWrapperPassPass(Registry);
  initializeMachineDominatorTreeWrapperPassPass(Registry);
  initializeMachineFunctionPrinterPassPass(Registry);
  initializeMachineFunctionSplitterPass(Registry);
  initializeMachineLateInstrsCleanupLegacyPass(Registry);
  initializeMachineLICMPass(Registry);
  initializeMachineLoopInfoWrapperPassPass(Registry);
  initializeMachineModuleInfoWrapperPassPass(Registry);
  initializeMachineOptimizationRemarkEmitterPassPass(Registry);
  initializeMachineOutlinerPass(Registry);
  initializeMachinePipelinerPass(Registry);
  initializeMachineSanitizerBinaryMetadataPass(Registry);
  initializeModuloScheduleTestPass(Registry);
  initializeMachinePostDominatorTreeWrapperPassPass(Registry);
  initializeMachineRegionInfoPassPass(Registry);
  initializeMachineSchedulerLegacyPass(Registry);
  initializeMachineSinkingLegacyPass(Registry);
  initializeMachineUniformityAnalysisPassPass(Registry);
  initializeMachineUniformityInfoPrinterPassPass(Registry);
  initializeMachineVerifierLegacyPassPass(Registry);
  initializeObjCARCContractLegacyPassPass(Registry);
  initializeOptimizePHIsLegacyPass(Registry);
  initializePEIPass(Registry);
  initializePHIEliminationPass(Registry);
  initializePatchableFunctionLegacyPass(Registry);
  initializePeepholeOptimizerLegacyPass(Registry);
  initializePostMachineSchedulerLegacyPass(Registry);
  initializePostRAHazardRecognizerPass(Registry);
  initializePostRAMachineSinkingPass(Registry);
  initializePostRASchedulerLegacyPass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  initializeProcessImplicitDefsPass(Registry);
  initializeRABasicPass(Registry);
  initializeRAGreedyLegacyPass(Registry);
  initializeRegAllocFastPass(Registry);
  initializeRegUsageInfoCollectorLegacyPass(Registry);
  initializeRegUsageInfoPropagationLegacyPass(Registry);
  initializeRegisterCoalescerLegacyPass(Registry);
  initializeRemoveLoadsIntoFakeUsesPass(Registry);
  initializeRemoveRedundantDebugValuesLegacyPass(Registry);
  initializeRenameIndependentSubregsLegacyPass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeSelectOptimizePass(Registry);
  initializeShadowStackGCLoweringPass(Registry);
  initializeShrinkWrapPass(Registry);
  initializeSjLjEHPreparePass(Registry);
  initializeSlotIndexesWrapperPassPass(Registry);
  initializeStackColoringLegacyPass(Registry);
  initializeStackFrameLayoutAnalysisPassPass(Registry);
  initializeStackMapLivenessPass(Registry);
  initializeStackProtectorPass(Registry);
  initializeStackSlotColoringLegacyPass(Registry);
  initializeStaticDataSplitterPass(Registry);
  initializeStripDebugMachineModulePass(Registry);
  initializeTailDuplicateLegacyPass(Registry);
  initializeTargetPassConfigPass(Registry);
  initializeTwoAddressInstructionLegacyPassPass(Registry);
  initializeTypePromotionLegacyPass(Registry);
  initializeUnpackMachineBundlesPass(Registry);
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeUnreachableMachineBlockElimPass(Registry);
  initializeVirtRegMapWrapperLegacyPass(Registry);
  initializeVirtRegRewriterPass(Registry);
  initializeWasmEHPreparePass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeXRayInstrumentationPass(Registry);
}
