//===- llvm/InitializePasses.h - Initialize All Passes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for the pass initialization routines
// for the entire LLVM project.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INITIALIZEPASSES_H
#define LLVM_INITIALIZEPASSES_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class PassRegistry;

/// Initialize all passes linked into the Core library.
LLVM_FUNC_ABI void initializeCore(PassRegistry&);

/// Initialize all passes linked into the TransformUtils library.
LLVM_FUNC_ABI void initializeTransformUtils(PassRegistry&);

/// Initialize all passes linked into the ScalarOpts library.
LLVM_FUNC_ABI void initializeScalarOpts(PassRegistry&);

/// Initialize all passes linked into the Vectorize library.
LLVM_FUNC_ABI void initializeVectorization(PassRegistry&);

/// Initialize all passes linked into the InstCombine library.
LLVM_FUNC_ABI void initializeInstCombine(PassRegistry&);

/// Initialize all passes linked into the IPO library.
LLVM_FUNC_ABI void initializeIPO(PassRegistry&);

/// Initialize all passes linked into the Analysis library.
LLVM_FUNC_ABI void initializeAnalysis(PassRegistry&);

/// Initialize all passes linked into the CodeGen library.
LLVM_FUNC_ABI void initializeCodeGen(PassRegistry&);

/// Initialize all passes linked into the GlobalISel library.
LLVM_FUNC_ABI void initializeGlobalISel(PassRegistry&);

/// Initialize all passes linked into the CodeGen library.
LLVM_FUNC_ABI void initializeTarget(PassRegistry&);

LLVM_FUNC_ABI void initializeAAResultsWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeAlwaysInlinerLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeAssignmentTrackingAnalysisPass(PassRegistry &);
LLVM_FUNC_ABI void initializeAssumeBuilderPassLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeAssumptionCacheTrackerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeAtomicExpandPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBasicBlockPathCloningPass(PassRegistry &);
LLVM_FUNC_ABI void initializeBasicBlockSectionsProfileReaderPass(PassRegistry &);
LLVM_FUNC_ABI void initializeBasicBlockSectionsPass(PassRegistry &);
LLVM_FUNC_ABI void initializeBarrierNoopPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBasicAAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBlockFrequencyInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBranchFolderPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBranchProbabilityInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBranchRelaxationPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBreakCriticalEdgesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeBreakFalseDepsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCanonicalizeFreezeInLoopsPass(PassRegistry &);
LLVM_FUNC_ABI void initializeCFGOnlyPrinterLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFGOnlyViewerLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFGPrinterLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFGSimplifyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFGuardPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFGuardLongjmpPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFGViewerLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFIFixupPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCFIInstrInserterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCallBrPreparePass(PassRegistry &);
LLVM_FUNC_ABI void initializeCallGraphDOTPrinterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCallGraphPrinterLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCallGraphViewerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCallGraphWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCheckDebugMachineModulePass(PassRegistry &);
LLVM_FUNC_ABI void initializeCodeGenPreparePass(PassRegistry&);
LLVM_FUNC_ABI void initializeComplexDeinterleavingLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeConstantHoistingLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCostModelAnalysisPass(PassRegistry&);
LLVM_FUNC_ABI void initializeCycleInfoWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeDAEPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDAHPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDCELegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDeadMachineInstructionElimPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDebugifyMachineModulePass(PassRegistry &);
LLVM_FUNC_ABI void initializeDelinearizationPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDependenceAnalysisWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDetectDeadLanesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDomOnlyPrinterWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeDomOnlyViewerWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeDomPrinterWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeDomViewerWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeDominanceFrontierWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDominatorTreeWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeDwarfEHPrepareLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeEarlyCSELegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeEarlyCSEMemSSALegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeEarlyIfConverterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeEarlyIfPredicatorPass(PassRegistry &);
LLVM_FUNC_ABI void initializeEarlyMachineLICMPass(PassRegistry&);
LLVM_FUNC_ABI void initializeEarlyTailDuplicatePass(PassRegistry&);
LLVM_FUNC_ABI void initializeEdgeBundlesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeEHContGuardCatchretPass(PassRegistry &);
LLVM_FUNC_ABI void initializeExpandLargeFpConvertLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeExpandLargeDivRemLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeExpandMemCmpPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeExpandPostRAPass(PassRegistry&);
LLVM_FUNC_ABI void initializeExpandReductionsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeExpandVectorPredicationPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMakeGuardsExplicitLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeExternalAAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeFEntryInserterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeFinalizeISelPass(PassRegistry&);
LLVM_FUNC_ABI void initializeFinalizeMachineBundlesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeFixIrreduciblePass(PassRegistry &);
LLVM_FUNC_ABI void initializeFixupStatepointCallerSavedPass(PassRegistry&);
LLVM_FUNC_ABI void initializeFlattenCFGLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeFuncletLayoutPass(PassRegistry&);
LLVM_FUNC_ABI void initializeGCEmptyBasicBlocksPass(PassRegistry &);
LLVM_FUNC_ABI void initializeGCMachineCodeAnalysisPass(PassRegistry&);
LLVM_FUNC_ABI void initializeGCModuleInfoPass(PassRegistry&);
LLVM_FUNC_ABI void initializeGVNLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeGlobalMergePass(PassRegistry&);
LLVM_FUNC_ABI void initializeGlobalsAAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeGuardWideningLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeHardwareLoopsLegacyPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMIRProfileLoaderPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeIRSimilarityIdentifierWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeIRTranslatorPass(PassRegistry&);
LLVM_FUNC_ABI void initializeIVUsersWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeIfConverterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeImmutableModuleSummaryIndexWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeImplicitNullChecksPass(PassRegistry&);
LLVM_FUNC_ABI void initializeIndirectBrExpandPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeInferAddressSpacesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeInstCountLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeInstSimplifyLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeInstructionCombiningPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeInstructionSelectPass(PassRegistry&);
LLVM_FUNC_ABI void initializeInterleavedAccessPass(PassRegistry&);
LLVM_FUNC_ABI void initializeInterleavedLoadCombinePass(PassRegistry &);
LLVM_FUNC_ABI void initializeIntervalPartitionPass(PassRegistry&);
LLVM_FUNC_ABI void initializeJMCInstrumenterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeKCFIPass(PassRegistry &);
LLVM_FUNC_ABI void initializeLCSSAVerificationPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLCSSAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLazyBlockFrequencyInfoPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLazyBranchProbabilityInfoPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLazyMachineBlockFrequencyInfoPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLazyValueInfoPrinterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLazyValueInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLegacyLICMPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLegacyLoopSinkPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLegalizerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeGISelCSEAnalysisWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeGISelKnownBitsAnalysisPass(PassRegistry &);
LLVM_FUNC_ABI void initializeLiveDebugValuesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLiveDebugVariablesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLiveIntervalsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLiveRangeShrinkPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLiveRegMatrixPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLiveStacksPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLiveVariablesPass(PassRegistry &);
LLVM_FUNC_ABI void initializeLoadStoreOptPass(PassRegistry &);
LLVM_FUNC_ABI void initializeLoadStoreVectorizerLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLocalStackSlotPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLocalizerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopDataPrefetchLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopExtractorLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeLoopGuardWideningLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopInstSimplifyLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopPredicationLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopRotateLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopSimplifyCFGLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopSimplifyPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopStrengthReducePass(PassRegistry&);
LLVM_FUNC_ABI void initializeLoopUnrollPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerAtomicLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerConstantIntrinsicsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerEmuTLSPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerGlobalDtorsLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeLowerWidenableConditionLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerIntrinsicsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerInvokeLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeLowerSwitchLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeKCFIPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMIRAddFSDiscriminatorsPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMIRCanonicalizerPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMIRNamerPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMIRPrintingPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineBlockFrequencyInfoPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineBlockPlacementPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineBlockPlacementStatsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineBranchProbabilityInfoPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineCFGPrinterPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineCSEPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineCombinerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineCopyPropagationPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineCycleInfoPrinterPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineCycleInfoWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineDominanceFrontierPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineDominatorTreePass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineFunctionPrinterPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineFunctionSplitterPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineLateInstrsCleanupPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineLICMPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineLoopInfoPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineModuleInfoWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineOptimizationRemarkEmitterPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineOutlinerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachinePipelinerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachinePostDominatorTreePass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineRegionInfoPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineSanitizerBinaryMetadataPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineSchedulerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineSinkingPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineTraceMetricsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMachineUniformityInfoPrinterPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineUniformityAnalysisPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMachineVerifierPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMemoryDependenceWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMemorySSAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeMergeICmpsLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeMergedLoadStoreMotionLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeModuleSummaryIndexWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeModuloScheduleTestPass(PassRegistry&);
LLVM_FUNC_ABI void initializeNaryReassociateLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeObjCARCContractLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeOptimizationRemarkEmitterWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeOptimizePHIsPass(PassRegistry&);
LLVM_FUNC_ABI void initializePEIPass(PassRegistry&);
LLVM_FUNC_ABI void initializePHIEliminationPass(PassRegistry&);
LLVM_FUNC_ABI void initializePartiallyInlineLibCallsLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializePatchableFunctionPass(PassRegistry&);
LLVM_FUNC_ABI void initializePeepholeOptimizerPass(PassRegistry&);
LLVM_FUNC_ABI void initializePhiValuesWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializePhysicalRegisterUsageInfoPass(PassRegistry&);
LLVM_FUNC_ABI void initializePlaceBackedgeSafepointsLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializePostDomOnlyPrinterWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializePostDomOnlyViewerWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializePostDomPrinterWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializePostDomViewerWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializePostDominatorTreeWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializePostMachineSchedulerPass(PassRegistry&);
LLVM_FUNC_ABI void initializePostRAHazardRecognizerPass(PassRegistry&);
LLVM_FUNC_ABI void initializePostRAMachineSinkingPass(PassRegistry&);
LLVM_FUNC_ABI void initializePostRASchedulerPass(PassRegistry&);
LLVM_FUNC_ABI void initializePreISelIntrinsicLoweringLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializePredicateInfoPrinterLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializePrintFunctionPassWrapperPass(PassRegistry&);
LLVM_FUNC_ABI void initializePrintModulePassWrapperPass(PassRegistry&);
LLVM_FUNC_ABI void initializeProcessImplicitDefsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeProfileSummaryInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializePromoteLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRABasicPass(PassRegistry&);
LLVM_FUNC_ABI void initializePseudoProbeInserterPass(PassRegistry &);
LLVM_FUNC_ABI void initializeRAGreedyPass(PassRegistry&);
LLVM_FUNC_ABI void initializeReachingDefAnalysisPass(PassRegistry&);
LLVM_FUNC_ABI void initializeReassociateLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRedundantDbgInstEliminationPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegAllocEvictionAdvisorAnalysisPass(PassRegistry &);
LLVM_FUNC_ABI void initializeRegAllocFastPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegAllocPriorityAdvisorAnalysisPass(PassRegistry &);
LLVM_FUNC_ABI void initializeRegAllocScoringPass(PassRegistry &);
LLVM_FUNC_ABI void initializeRegBankSelectPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegToMemLegacyPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegUsageInfoCollectorPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegUsageInfoPropagationPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegionInfoPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegionOnlyPrinterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegionOnlyViewerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegionPrinterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegionViewerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRegisterCoalescerPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRemoveRedundantDebugValuesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeRenameIndependentSubregsPass(PassRegistry&);
LLVM_FUNC_ABI void initializeReplaceWithVeclibLegacyPass(PassRegistry &);
LLVM_FUNC_ABI void initializeResetMachineFunctionPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSCEVAAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSROALegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSafeStackLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSafepointIRVerifierPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSelectOptimizePass(PassRegistry &);
LLVM_FUNC_ABI void initializeScalarEvolutionWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeScalarizeMaskedMemIntrinLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeScalarizerLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeScavengerTestPass(PassRegistry&);
LLVM_FUNC_ABI void initializeScopedNoAliasAAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSeparateConstOffsetFromGEPLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeShadowStackGCLoweringPass(PassRegistry&);
LLVM_FUNC_ABI void initializeShrinkWrapPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSimpleLoopUnswitchLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSingleLoopExtractorPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSinkingLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSjLjEHPreparePass(PassRegistry&);
LLVM_FUNC_ABI void initializeSlotIndexesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSpeculativeExecutionLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeSpillPlacementPass(PassRegistry&);
LLVM_FUNC_ABI void initializeStackColoringPass(PassRegistry&);
LLVM_FUNC_ABI void initializeStackFrameLayoutAnalysisPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeStackMapLivenessPass(PassRegistry&);
LLVM_FUNC_ABI void initializeStackProtectorPass(PassRegistry&);
LLVM_FUNC_ABI void initializeStackSafetyGlobalInfoWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeStackSafetyInfoWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeStackSlotColoringPass(PassRegistry&);
LLVM_FUNC_ABI void initializeStraightLineStrengthReduceLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeStripDebugMachineModulePass(PassRegistry &);
LLVM_FUNC_ABI void initializeStripGCRelocatesLegacyPass(PassRegistry &);
LLVM_FUNC_ABI void initializeStructurizeCFGLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeTailCallElimPass(PassRegistry&);
LLVM_FUNC_ABI void initializeTailDuplicatePass(PassRegistry&);
LLVM_FUNC_ABI void initializeTargetLibraryInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeTargetPassConfigPass(PassRegistry&);
LLVM_FUNC_ABI void initializeTargetTransformInfoWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeTLSVariableHoistLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeTwoAddressInstructionPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeTypeBasedAAWrapperPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeTypePromotionLegacyPass(PassRegistry&);
LLVM_FUNC_ABI void initializeUniformityInfoWrapperPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeUnifyFunctionExitNodesLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeUnifyLoopExitsLegacyPassPass(PassRegistry &);
LLVM_FUNC_ABI void initializeUnpackMachineBundlesPass(PassRegistry&);
LLVM_FUNC_ABI void initializeUnreachableBlockElimLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeUnreachableMachineBlockElimPass(PassRegistry&);
LLVM_FUNC_ABI void initializeVerifierLegacyPassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeVirtRegMapPass(PassRegistry&);
LLVM_FUNC_ABI void initializeVirtRegRewriterPass(PassRegistry&);
LLVM_FUNC_ABI void initializeWasmEHPreparePass(PassRegistry&);
LLVM_FUNC_ABI void initializeWinEHPreparePass(PassRegistry&);
LLVM_FUNC_ABI void initializeWriteBitcodePassPass(PassRegistry&);
LLVM_FUNC_ABI void initializeXRayInstrumentationPass(PassRegistry&);

} // end namespace llvm

#endif // LLVM_INITIALIZEPASSES_H
