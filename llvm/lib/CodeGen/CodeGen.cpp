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
#ifndef EJIT_BARE_METAL
  initializeBasicBlockPathCloningPass(Registry);
  initializeBasicBlockSectionsPass(Registry);
  initializeBranchFolderLegacyPass(Registry);
  initializeBranchRelaxationLegacyPass(Registry);
  initializeBreakFalseDepsPass(Registry);
#endif
  initializeCallBrPreparePass(Registry);
#ifndef EJIT_BARE_METAL
  initializeCFGuardLongjmpPass(Registry);
#endif
  initializeCFIFixupPass(Registry);
  initializeCFIInstrInserterPass(Registry);
  initializeCheckDebugMachineModulePass(Registry);
  initializeCodeGenPrepareLegacyPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeDeadMachineInstructionElimPass(Registry);
  initializeDebugifyMachineModulePass(Registry);
  initializeDetectDeadLanesLegacyPass(Registry);
  initializeDwarfEHPrepareLegacyPassPass(Registry);
  initializeEarlyIfConverterLegacyPass(Registry);
  initializeEarlyIfPredicatorPass(Registry);
  initializeEarlyMachineLICMPass(Registry);
  initializeEarlyTailDuplicateLegacyPass(Registry);
  initializeExpandLargeDivRemLegacyPassPass(Registry);
#endif
  initializeExpandFpLegacyPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeExpandMemCmpLegacyPassPass(Registry);
#endif
  initializeExpandPostRALegacyPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeFEntryInserterLegacyPass(Registry);
#endif
  initializeFinalizeISelPass(Registry);
  initializeFinalizeMachineBundlesPass(Registry);
  initializeFixupStatepointCallerSavedLegacyPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeFuncletLayoutPass(Registry);
  initializeGCMachineCodeAnalysisPass(Registry);
  initializeGCModuleInfoPass(Registry);
  initializeHardwareLoopsLegacyPass(Registry);
  initializeIfConverterPass(Registry);
  initializeImplicitNullChecksPass(Registry);
  initializeIndirectBrExpandLegacyPassPass(Registry);
#endif
  initializeInitUndefLegacyPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeInterleavedLoadCombinePass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeJMCInstrumenterPass(Registry);
  initializeLiveDebugValuesLegacyPass(Registry);
  initializeLiveDebugVariablesWrapperLegacyPass(Registry);
#endif
  initializeLiveIntervalsWrapperPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeLiveRangeShrinkPass(Registry);
  initializeLiveStacksWrapperLegacyPass(Registry);
#endif
  initializeLiveVariablesWrapperPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeLocalStackSlotPassPass(Registry);
  initializeLowerGlobalDtorsLegacyPassPass(Registry);
#endif
  initializeLowerIntrinsicsPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeMIRAddFSDiscriminatorsPass(Registry);
  initializeMIRCanonicalizerPass(Registry);
  initializeMIRNamerPass(Registry);
  initializeMIRProfileLoaderPassPass(Registry);
  initializeMachineBlockFrequencyInfoWrapperPassPass(Registry);
  initializeMachineBlockPlacementLegacyPass(Registry);
  initializeMachineBlockPlacementStatsLegacyPass(Registry);
  initializeMachineCFGPrinterPass(Registry);
  initializeMachineCSELegacyPass(Registry);
  initializeMachineCombinerPass(Registry);
  initializeMachineCopyPropagationLegacyPass(Registry);
  initializeMachineCycleInfoPrinterLegacyPass(Registry);
#endif
  initializeMachineCycleInfoWrapperPassPass(Registry);
  initializeMachineDominatorTreeWrapperPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeMachineFunctionPrinterPassPass(Registry);
  initializeMachineFunctionSplitterPass(Registry);
  initializeMachineLateInstrsCleanupLegacyPass(Registry);
#endif
  initializeMachineLICMPass(Registry);
  initializeMachineLoopInfoWrapperPassPass(Registry);
  initializeMachineModuleInfoWrapperPassPass(Registry);
  initializeMachineOptimizationRemarkEmitterPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeMachineOutlinerPass(Registry);
  initializeMachinePipelinerPass(Registry);
  initializeMachineSanitizerBinaryMetadataLegacyPass(Registry);
  initializeModuloScheduleTestPass(Registry);
  initializeMachinePostDominatorTreeWrapperPassPass(Registry);
  initializeMachineRegionInfoPassPass(Registry);
#endif
  initializeMachineSchedulerLegacyPass(Registry);
  initializeMachineSinkingLegacyPass(Registry);
  initializeMachineUniformityAnalysisPassPass(Registry);
  initializeMachineUniformityInfoPrinterPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeMachineVerifierLegacyPassPass(Registry);
#endif
#ifndef EJIT_BARE_METAL
  initializeObjCARCContractLegacyPassPass(Registry);
#endif
  initializeOptimizePHIsLegacyPass(Registry);
  initializePEILegacyPass(Registry);
  initializePHIEliminationPass(Registry);
#ifndef EJIT_BARE_METAL
  initializePatchableFunctionLegacyPass(Registry);
#endif
  initializePeepholeOptimizerLegacyPass(Registry);
  initializePostMachineSchedulerLegacyPass(Registry);
#ifndef EJIT_BARE_METAL
  initializePostRAMachineSinkingLegacyPass(Registry);
  initializePostRAHazardRecognizerLegacyPass(Registry);
  initializePostRASchedulerLegacyPass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
#endif
  initializeProcessImplicitDefsPass(Registry);
  initializeRABasicPass(Registry);
  initializeRAGreedyLegacyPass(Registry);
  initializeRegAllocFastPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeRegUsageInfoCollectorLegacyPass(Registry);
  initializeRegUsageInfoPropagationLegacyPass(Registry);
#endif
  initializeRegisterCoalescerLegacyPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeRemoveLoadsIntoFakeUsesLegacyPass(Registry);
  initializeRemoveRedundantDebugValuesLegacyPass(Registry);
  initializeRenameIndependentSubregsLegacyPass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeSelectOptimizePass(Registry);
  initializeShadowStackGCLoweringPass(Registry);
  initializeShrinkWrapLegacyPass(Registry);
  initializeSjLjEHPreparePass(Registry);
#endif
  initializeSlotIndexesWrapperPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeStackColoringLegacyPass(Registry);
  initializeStackFrameLayoutAnalysisLegacyPass(Registry);
  initializeStackMapLivenessPass(Registry);
  initializeStackProtectorPass(Registry);
  initializeStackSlotColoringLegacyPass(Registry);
  initializeStaticDataSplitterPass(Registry);
  initializeStaticDataAnnotatorPass(Registry);
  initializeStripDebugMachineModulePass(Registry);
  initializeTailDuplicateLegacyPass(Registry);
#endif
  initializeTargetPassConfigPass(Registry);
  initializeTwoAddressInstructionLegacyPassPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeTypePromotionLegacyPass(Registry);
#endif
  initializeUnpackMachineBundlesPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeUnreachableMachineBlockElimLegacyPass(Registry);
#endif
  initializeVirtRegMapWrapperLegacyPass(Registry);
  initializeVirtRegRewriterLegacyPass(Registry);
#ifndef EJIT_BARE_METAL
  initializeWasmEHPreparePass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeXRayInstrumentationLegacyPass(Registry);
#endif
}
