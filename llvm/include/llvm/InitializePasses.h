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

namespace llvm {

class PassRegistry;

/// Initialize all passes linked into the TransformUtils library.
void initializeCore(PassRegistry&);

/// Initialize all passes linked into the TransformUtils library.
void initializeTransformUtils(PassRegistry&);

/// Initialize all passes linked into the ScalarOpts library.
void initializeScalarOpts(PassRegistry&);

/// Initialize all passes linked into the ObjCARCOpts library.
void initializeObjCARCOpts(PassRegistry&);

/// Initialize all passes linked into the Vectorize library.
void initializeVectorization(PassRegistry&);

/// Initialize all passes linked into the InstCombine library.
void initializeInstCombine(PassRegistry&);

/// Initialize all passes linked into the AggressiveInstCombine library.
void initializeAggressiveInstCombine(PassRegistry&);

/// Initialize all passes linked into the IPO library.
void initializeIPO(PassRegistry&);

/// Initialize all passes linked into the Instrumentation library.
void initializeInstrumentation(PassRegistry&);

/// Initialize all passes linked into the Analysis library.
void initializeAnalysis(PassRegistry&);

/// Initialize all passes linked into the CodeGen library.
void initializeCodeGen(PassRegistry&);

/// Initialize all passes linked into the GlobalISel library.
void initializeGlobalISel(PassRegistry&);

/// Initialize all passes linked into the CodeGen library.
void initializeTarget(PassRegistry&);

void initializeAAEvalLegacyPassPass(PassRegistry&);
void initializeAAResultsWrapperPassPass(PassRegistry&);
void initializeADCELegacyPassPass(PassRegistry&);
void initializeAddDiscriminatorsLegacyPassPass(PassRegistry&);
void initializeAddFSDiscriminatorsPass(PassRegistry &);
void initializeAggressiveInstCombinerLegacyPassPass(PassRegistry&);
void initializeAliasSetPrinterPass(PassRegistry&);
void initializeAlignmentFromAssumptionsPass(PassRegistry&);
void initializeAlwaysInlinerLegacyPassPass(PassRegistry&);
void initializeAssumeSimplifyPassLegacyPassPass(PassRegistry &);
void initializeAssumeBuilderPassLegacyPassPass(PassRegistry &);
void initializeAnnotation2MetadataLegacyPass(PassRegistry &);
void initializeAnnotationRemarksLegacyPass(PassRegistry &);
void initializeOpenMPOptCGSCCLegacyPassPass(PassRegistry &);
void initializeAssumptionCacheTrackerPass(PassRegistry&);
void initializeAtomicExpandPass(PassRegistry&);
void initializeAttributorLegacyPassPass(PassRegistry&);
void initializeAttributorCGSCCLegacyPassPass(PassRegistry &);
void initializeBasicBlockSectionsProfileReaderPass(PassRegistry &);
void initializeBasicBlockSectionsPass(PassRegistry &);
void initializeBDCELegacyPassPass(PassRegistry&);
void initializeBarrierNoopPass(PassRegistry&);
void initializeBasicAAWrapperPassPass(PassRegistry&);
void initializeBlockExtractorLegacyPassPass(PassRegistry &);
void initializeBlockFrequencyInfoWrapperPassPass(PassRegistry&);
void initializeBoundsCheckingLegacyPassPass(PassRegistry&);
void initializeBranchFolderPassPass(PassRegistry&);
void initializeBranchProbabilityInfoWrapperPassPass(PassRegistry&);
void initializeBranchRelaxationPass(PassRegistry&);
void initializeBreakCriticalEdgesPass(PassRegistry&);
void initializeBreakFalseDepsPass(PassRegistry&);
void initializeCanonicalizeAliasesLegacyPassPass(PassRegistry &);
void initializeCanonicalizeFreezeInLoopsPass(PassRegistry &);
void initializeCFGOnlyPrinterLegacyPassPass(PassRegistry&);
void initializeCFGOnlyViewerLegacyPassPass(PassRegistry&);
void initializeCFGPrinterLegacyPassPass(PassRegistry&);
void initializeCFGSimplifyPassPass(PassRegistry&);
void initializeCFGuardPass(PassRegistry&);
void initializeCFGuardLongjmpPass(PassRegistry&);
void initializeCFGViewerLegacyPassPass(PassRegistry&);
void initializeCFIFixupPass(PassRegistry&);
void initializeCFIInstrInserterPass(PassRegistry&);
void initializeCFLAndersAAWrapperPassPass(PassRegistry&);
void initializeCFLSteensAAWrapperPassPass(PassRegistry&);
void initializeCGProfileLegacyPassPass(PassRegistry &);
void initializeCallGraphDOTPrinterPass(PassRegistry&);
void initializeCallGraphPrinterLegacyPassPass(PassRegistry&);
void initializeCallGraphViewerPass(PassRegistry&);
void initializeCallGraphWrapperPassPass(PassRegistry&);
void initializeCallSiteSplittingLegacyPassPass(PassRegistry&);
void initializeCalledValuePropagationLegacyPassPass(PassRegistry &);
void initializeCheckDebugMachineModulePass(PassRegistry &);
void initializeCodeGenPreparePass(PassRegistry&);
void initializeConstantHoistingLegacyPassPass(PassRegistry&);
void initializeConstantMergeLegacyPassPass(PassRegistry&);
void initializeConstraintEliminationPass(PassRegistry &);
void initializeControlHeightReductionLegacyPassPass(PassRegistry&);
void initializeCorrelatedValuePropagationPass(PassRegistry&);
void initializeCostModelAnalysisPass(PassRegistry&);
void initializeCrossDSOCFIPass(PassRegistry&);
void initializeCycleInfoWrapperPassPass(PassRegistry &);
void initializeDAEPass(PassRegistry&);
void initializeDAHPass(PassRegistry&);
void initializeDCELegacyPassPass(PassRegistry&);
void initializeDFAJumpThreadingLegacyPassPass(PassRegistry &);
void initializeDSELegacyPassPass(PassRegistry&);
void initializeDataFlowSanitizerLegacyPassPass(PassRegistry &);
void initializeDeadMachineInstructionElimPass(PassRegistry&);
void initializeDebugifyMachineModulePass(PassRegistry &);
void initializeDelinearizationPass(PassRegistry&);
void initializeDemandedBitsWrapperPassPass(PassRegistry&);
void initializeDependenceAnalysisPass(PassRegistry&);
void initializeDependenceAnalysisWrapperPassPass(PassRegistry&);
void initializeDetectDeadLanesPass(PassRegistry&);
void initializeDivRemPairsLegacyPassPass(PassRegistry&);
void initializeDomOnlyPrinterWrapperPassPass(PassRegistry &);
void initializeDomOnlyViewerWrapperPassPass(PassRegistry &);
void initializeDomPrinterWrapperPassPass(PassRegistry &);
void initializeDomViewerWrapperPassPass(PassRegistry &);
void initializeDominanceFrontierWrapperPassPass(PassRegistry&);
void initializeDominatorTreeWrapperPassPass(PassRegistry&);
void initializeDwarfEHPrepareLegacyPassPass(PassRegistry &);
void initializeEarlyCSELegacyPassPass(PassRegistry&);
void initializeEarlyCSEMemSSALegacyPassPass(PassRegistry&);
void initializeEarlyIfConverterPass(PassRegistry&);
void initializeEarlyIfPredicatorPass(PassRegistry &);
void initializeEarlyMachineLICMPass(PassRegistry&);
void initializeEarlyTailDuplicatePass(PassRegistry&);
void initializeEdgeBundlesPass(PassRegistry&);
void initializeEHContGuardCatchretPass(PassRegistry &);
void initializeEliminateAvailableExternallyLegacyPassPass(PassRegistry&);
void initializeEntryExitInstrumenterPass(PassRegistry&);
void initializeExpandMemCmpPassPass(PassRegistry&);
void initializeExpandPostRAPass(PassRegistry&);
void initializeExpandReductionsPass(PassRegistry&);
void initializeExpandVectorPredicationPass(PassRegistry &);
void initializeMakeGuardsExplicitLegacyPassPass(PassRegistry&);
void initializeExternalAAWrapperPassPass(PassRegistry&);
void initializeFEntryInserterPass(PassRegistry&);
void initializeFinalizeISelPass(PassRegistry&);
void initializeFinalizeMachineBundlesPass(PassRegistry&);
void initializeFixIrreduciblePass(PassRegistry &);
void initializeFixupStatepointCallerSavedPass(PassRegistry&);
void initializeFlattenCFGLegacyPassPass(PassRegistry &);
void initializeFloat2IntLegacyPassPass(PassRegistry&);
void initializeForceFunctionAttrsLegacyPassPass(PassRegistry&);
void initializeForwardControlFlowIntegrityPass(PassRegistry&);
void initializeFuncletLayoutPass(PassRegistry&);
void initializeFunctionImportLegacyPassPass(PassRegistry&);
void initializeFunctionSpecializationLegacyPassPass(PassRegistry &);
void initializeGCMachineCodeAnalysisPass(PassRegistry&);
void initializeGCModuleInfoPass(PassRegistry&);
void initializeGVNHoistLegacyPassPass(PassRegistry&);
void initializeGVNLegacyPassPass(PassRegistry&);
void initializeGVNSinkLegacyPassPass(PassRegistry&);
void initializeGlobalDCELegacyPassPass(PassRegistry&);
void initializeGlobalMergePass(PassRegistry&);
void initializeGlobalOptLegacyPassPass(PassRegistry&);
void initializeGlobalSplitPass(PassRegistry&);
void initializeGlobalsAAWrapperPassPass(PassRegistry&);
void initializeGuardWideningLegacyPassPass(PassRegistry&);
void initializeHardwareLoopsPass(PassRegistry&);
void initializeMIRProfileLoaderPassPass(PassRegistry &);
void initializeMemProfilerLegacyPassPass(PassRegistry &);
void initializeHotColdSplittingLegacyPassPass(PassRegistry&);
void initializeIPSCCPLegacyPassPass(PassRegistry&);
void initializeIRCELegacyPassPass(PassRegistry&);
void initializeIROutlinerLegacyPassPass(PassRegistry&);
void initializeIRSimilarityIdentifierWrapperPassPass(PassRegistry&);
void initializeIRTranslatorPass(PassRegistry&);
void initializeIVUsersWrapperPassPass(PassRegistry&);
void initializeIfConverterPass(PassRegistry&);
void initializeImmutableModuleSummaryIndexWrapperPassPass(PassRegistry&);
void initializeImplicitNullChecksPass(PassRegistry&);
void initializeIndVarSimplifyLegacyPassPass(PassRegistry&);
void initializeIndirectBrExpandPassPass(PassRegistry&);
void initializeInferAddressSpacesPass(PassRegistry&);
void initializeInferFunctionAttrsLegacyPassPass(PassRegistry&);
void initializeInjectTLIMappingsLegacyPass(PassRegistry &);
void initializeInlineCostAnalysisPass(PassRegistry&);
void initializeInstCountLegacyPassPass(PassRegistry &);
void initializeInstNamerPass(PassRegistry&);
void initializeInstSimplifyLegacyPassPass(PassRegistry &);
void initializeInstrProfilingLegacyPassPass(PassRegistry&);
void initializeInstrOrderFileLegacyPassPass(PassRegistry&);
void initializeInstructionCombiningPassPass(PassRegistry&);
void initializeInstructionSelectPass(PassRegistry&);
void initializeInterleavedAccessPass(PassRegistry&);
void initializeInterleavedLoadCombinePass(PassRegistry &);
void initializeInternalizeLegacyPassPass(PassRegistry&);
void initializeIntervalPartitionPass(PassRegistry&);
void initializeJMCInstrumenterPass(PassRegistry&);
void initializeJumpThreadingPass(PassRegistry&);
void initializeLCSSAVerificationPassPass(PassRegistry&);
void initializeLCSSAWrapperPassPass(PassRegistry&);
void initializeLazyBlockFrequencyInfoPassPass(PassRegistry&);
void initializeLazyBranchProbabilityInfoPassPass(PassRegistry&);
void initializeLazyMachineBlockFrequencyInfoPassPass(PassRegistry&);
void initializeLazyValueInfoPrinterPass(PassRegistry&);
void initializeLazyValueInfoWrapperPassPass(PassRegistry&);
void initializeLegacyDivergenceAnalysisPass(PassRegistry&);
void initializeLegacyLICMPassPass(PassRegistry&);
void initializeLegacyLoopSinkPassPass(PassRegistry&);
void initializeLegalizerPass(PassRegistry&);
void initializeGISelCSEAnalysisWrapperPassPass(PassRegistry &);
void initializeGISelKnownBitsAnalysisPass(PassRegistry &);
void initializeLibCallsShrinkWrapLegacyPassPass(PassRegistry&);
void initializeLintLegacyPassPass(PassRegistry &);
void initializeLiveDebugValuesPass(PassRegistry&);
void initializeLiveDebugVariablesPass(PassRegistry&);
void initializeLiveIntervalsPass(PassRegistry&);
void initializeLiveRangeShrinkPass(PassRegistry&);
void initializeLiveRegMatrixPass(PassRegistry&);
void initializeLiveStacksPass(PassRegistry&);
void initializeLiveVariablesPass(PassRegistry &);
void initializeLoadStoreOptPass(PassRegistry &);
void initializeLoadStoreVectorizerLegacyPassPass(PassRegistry&);
void initializeLoaderPassPass(PassRegistry&);
void initializeLocalStackSlotPassPass(PassRegistry&);
void initializeLocalizerPass(PassRegistry&);
void initializeLoopAccessLegacyAnalysisPass(PassRegistry&);
void initializeLoopDataPrefetchLegacyPassPass(PassRegistry&);
void initializeLoopDeletionLegacyPassPass(PassRegistry&);
void initializeLoopDistributeLegacyPass(PassRegistry&);
void initializeLoopExtractorLegacyPassPass(PassRegistry &);
void initializeLoopGuardWideningLegacyPassPass(PassRegistry&);
void initializeLoopFuseLegacyPass(PassRegistry&);
void initializeLoopIdiomRecognizeLegacyPassPass(PassRegistry&);
void initializeLoopInfoWrapperPassPass(PassRegistry&);
void initializeLoopInstSimplifyLegacyPassPass(PassRegistry&);
void initializeLoopInterchangeLegacyPassPass(PassRegistry &);
void initializeLoopFlattenLegacyPassPass(PassRegistry&);
void initializeLoopLoadEliminationPass(PassRegistry&);
void initializeLoopPassPass(PassRegistry&);
void initializeLoopPredicationLegacyPassPass(PassRegistry&);
void initializeLoopRerollLegacyPassPass(PassRegistry &);
void initializeLoopRotateLegacyPassPass(PassRegistry&);
void initializeLoopSimplifyCFGLegacyPassPass(PassRegistry&);
void initializeLoopSimplifyPass(PassRegistry&);
void initializeLoopStrengthReducePass(PassRegistry&);
void initializeLoopUnrollAndJamPass(PassRegistry&);
void initializeLoopUnrollPass(PassRegistry&);
void initializeLoopUnswitchPass(PassRegistry&);
void initializeLoopVectorizePass(PassRegistry&);
void initializeLoopVersioningLICMLegacyPassPass(PassRegistry &);
void initializeLoopVersioningLegacyPassPass(PassRegistry &);
void initializeLowerAtomicLegacyPassPass(PassRegistry&);
void initializeLowerConstantIntrinsicsPass(PassRegistry&);
void initializeLowerEmuTLSPass(PassRegistry&);
void initializeLowerExpectIntrinsicPass(PassRegistry&);
void initializeLowerGlobalDtorsLegacyPassPass(PassRegistry &);
void initializeLowerGuardIntrinsicLegacyPassPass(PassRegistry&);
void initializeLowerWidenableConditionLegacyPassPass(PassRegistry&);
void initializeLowerIntrinsicsPass(PassRegistry&);
void initializeLowerInvokeLegacyPassPass(PassRegistry&);
void initializeLowerSwitchLegacyPassPass(PassRegistry &);
void initializeLowerTypeTestsPass(PassRegistry&);
void initializeLowerMatrixIntrinsicsLegacyPassPass(PassRegistry &);
void initializeLowerMatrixIntrinsicsMinimalLegacyPassPass(PassRegistry &);
void initializeMIRAddFSDiscriminatorsPass(PassRegistry &);
void initializeMIRCanonicalizerPass(PassRegistry &);
void initializeMIRNamerPass(PassRegistry &);
void initializeMIRPrintingPassPass(PassRegistry&);
void initializeMachineBlockFrequencyInfoPass(PassRegistry&);
void initializeMachineBlockPlacementPass(PassRegistry&);
void initializeMachineBlockPlacementStatsPass(PassRegistry&);
void initializeMachineBranchProbabilityInfoPass(PassRegistry&);
void initializeMachineCSEPass(PassRegistry&);
void initializeMachineCombinerPass(PassRegistry&);
void initializeMachineCopyPropagationPass(PassRegistry&);
void initializeMachineCycleInfoPrinterPassPass(PassRegistry &);
void initializeMachineCycleInfoWrapperPassPass(PassRegistry &);
void initializeMachineDominanceFrontierPass(PassRegistry&);
void initializeMachineDominatorTreePass(PassRegistry&);
void initializeMachineFunctionPrinterPassPass(PassRegistry&);
void initializeMachineFunctionSplitterPass(PassRegistry &);
void initializeMachineLICMPass(PassRegistry&);
void initializeMachineLoopInfoPass(PassRegistry&);
void initializeMachineModuleInfoWrapperPassPass(PassRegistry &);
void initializeMachineOptimizationRemarkEmitterPassPass(PassRegistry&);
void initializeMachineOutlinerPass(PassRegistry&);
void initializeMachinePipelinerPass(PassRegistry&);
void initializeMachinePostDominatorTreePass(PassRegistry&);
void initializeMachineRegionInfoPassPass(PassRegistry&);
void initializeMachineSchedulerPass(PassRegistry&);
void initializeMachineSinkingPass(PassRegistry&);
void initializeMachineTraceMetricsPass(PassRegistry&);
void initializeMachineVerifierPassPass(PassRegistry&);
void initializeMemCpyOptLegacyPassPass(PassRegistry&);
void initializeMemDepPrinterPass(PassRegistry&);
void initializeMemDerefPrinterPass(PassRegistry&);
void initializeMemoryDependenceWrapperPassPass(PassRegistry&);
void initializeMemorySSAPrinterLegacyPassPass(PassRegistry&);
void initializeMemorySSAWrapperPassPass(PassRegistry&);
void initializeMergeFunctionsLegacyPassPass(PassRegistry&);
void initializeMergeICmpsLegacyPassPass(PassRegistry &);
void initializeMergedLoadStoreMotionLegacyPassPass(PassRegistry&);
void initializeMetaRenamerPass(PassRegistry&);
void initializeModuleDebugInfoLegacyPrinterPass(PassRegistry &);
void initializeModuleMemProfilerLegacyPassPass(PassRegistry &);
void initializeModuleSummaryIndexWrapperPassPass(PassRegistry&);
void initializeModuloScheduleTestPass(PassRegistry&);
void initializeMustExecutePrinterPass(PassRegistry&);
void initializeMustBeExecutedContextPrinterPass(PassRegistry&);
void initializeNameAnonGlobalLegacyPassPass(PassRegistry&);
void initializeNaryReassociateLegacyPassPass(PassRegistry&);
void initializeNewGVNLegacyPassPass(PassRegistry&);
void initializeObjCARCAAWrapperPassPass(PassRegistry&);
void initializeObjCARCAPElimPass(PassRegistry&);
void initializeObjCARCContractLegacyPassPass(PassRegistry &);
void initializeObjCARCExpandPass(PassRegistry&);
void initializeObjCARCOptLegacyPassPass(PassRegistry &);
void initializeOptimizationRemarkEmitterWrapperPassPass(PassRegistry&);
void initializeOptimizePHIsPass(PassRegistry&);
void initializePAEvalPass(PassRegistry&);
void initializePEIPass(PassRegistry&);
void initializePHIEliminationPass(PassRegistry&);
void initializePartialInlinerLegacyPassPass(PassRegistry&);
void initializePartiallyInlineLibCallsLegacyPassPass(PassRegistry&);
void initializePatchableFunctionPass(PassRegistry&);
void initializePeepholeOptimizerPass(PassRegistry&);
void initializePhiValuesWrapperPassPass(PassRegistry&);
void initializePhysicalRegisterUsageInfoPass(PassRegistry&);
void initializePlaceBackedgeSafepointsImplPass(PassRegistry&);
void initializePlaceSafepointsPass(PassRegistry&);
void initializePostDomOnlyPrinterWrapperPassPass(PassRegistry &);
void initializePostDomOnlyViewerWrapperPassPass(PassRegistry &);
void initializePostDomPrinterWrapperPassPass(PassRegistry &);
void initializePostDomViewerWrapperPassPass(PassRegistry &);
void initializePostDominatorTreeWrapperPassPass(PassRegistry&);
void initializePostInlineEntryExitInstrumenterPass(PassRegistry&);
void initializePostMachineSchedulerPass(PassRegistry&);
void initializePostOrderFunctionAttrsLegacyPassPass(PassRegistry&);
void initializePostRAHazardRecognizerPass(PassRegistry&);
void initializePostRAMachineSinkingPass(PassRegistry&);
void initializePostRASchedulerPass(PassRegistry&);
void initializePreISelIntrinsicLoweringLegacyPassPass(PassRegistry&);
void initializePredicateInfoPrinterLegacyPassPass(PassRegistry&);
void initializePrintFunctionPassWrapperPass(PassRegistry&);
void initializePrintModulePassWrapperPass(PassRegistry&);
void initializeProcessImplicitDefsPass(PassRegistry&);
void initializeProfileSummaryInfoWrapperPassPass(PassRegistry&);
void initializePromoteLegacyPassPass(PassRegistry&);
void initializePruneEHPass(PassRegistry&);
void initializeRABasicPass(PassRegistry&);
void initializePseudoProbeInserterPass(PassRegistry &);
void initializeRAGreedyPass(PassRegistry&);
void initializeReachingDefAnalysisPass(PassRegistry&);
void initializeReassociateLegacyPassPass(PassRegistry&);
void initializeRedundantDbgInstEliminationPass(PassRegistry&);
void initializeRegAllocEvictionAdvisorAnalysisPass(PassRegistry &);
void initializeRegAllocFastPass(PassRegistry&);
void initializeRegAllocScoringPass(PassRegistry &);
void initializeRegBankSelectPass(PassRegistry&);
void initializeRegToMemLegacyPass(PassRegistry&);
void initializeRegUsageInfoCollectorPass(PassRegistry&);
void initializeRegUsageInfoPropagationPass(PassRegistry&);
void initializeRegionInfoPassPass(PassRegistry&);
void initializeRegionOnlyPrinterPass(PassRegistry&);
void initializeRegionOnlyViewerPass(PassRegistry&);
void initializeRegionPrinterPass(PassRegistry&);
void initializeRegionViewerPass(PassRegistry&);
void initializeRegisterCoalescerPass(PassRegistry&);
void initializeRemoveRedundantDebugValuesPass(PassRegistry&);
void initializeRenameIndependentSubregsPass(PassRegistry&);
void initializeReplaceWithVeclibLegacyPass(PassRegistry &);
void initializeResetMachineFunctionPass(PassRegistry&);
void initializeReversePostOrderFunctionAttrsLegacyPassPass(PassRegistry&);
void initializeRewriteStatepointsForGCLegacyPassPass(PassRegistry &);
void initializeRewriteSymbolsLegacyPassPass(PassRegistry&);
void initializeSCCPLegacyPassPass(PassRegistry&);
void initializeSCEVAAWrapperPassPass(PassRegistry&);
void initializeSLPVectorizerPass(PassRegistry&);
void initializeSROALegacyPassPass(PassRegistry&);
void initializeSafeStackLegacyPassPass(PassRegistry&);
void initializeSafepointIRVerifierPass(PassRegistry&);
void initializeSampleProfileLoaderLegacyPassPass(PassRegistry&);
void initializeSelectOptimizePass(PassRegistry &);
void initializeModuleSanitizerCoverageLegacyPassPass(PassRegistry &);
void initializeScalarEvolutionWrapperPassPass(PassRegistry&);
void initializeScalarizeMaskedMemIntrinLegacyPassPass(PassRegistry &);
void initializeScalarizerLegacyPassPass(PassRegistry&);
void initializeScavengerTestPass(PassRegistry&);
void initializeScopedNoAliasAAWrapperPassPass(PassRegistry&);
void initializeSeparateConstOffsetFromGEPLegacyPassPass(PassRegistry &);
void initializeShadowStackGCLoweringPass(PassRegistry&);
void initializeShrinkWrapPass(PassRegistry&);
void initializeSimpleInlinerPass(PassRegistry&);
void initializeSimpleLoopUnswitchLegacyPassPass(PassRegistry&);
void initializeSingleLoopExtractorPass(PassRegistry&);
void initializeSinkingLegacyPassPass(PassRegistry&);
void initializeSjLjEHPreparePass(PassRegistry&);
void initializeSlotIndexesPass(PassRegistry&);
void initializeSpeculativeExecutionLegacyPassPass(PassRegistry&);
void initializeSpillPlacementPass(PassRegistry&);
void initializeStackColoringPass(PassRegistry&);
void initializeStackMapLivenessPass(PassRegistry&);
void initializeStackProtectorPass(PassRegistry&);
void initializeStackSafetyGlobalInfoWrapperPassPass(PassRegistry &);
void initializeStackSafetyInfoWrapperPassPass(PassRegistry &);
void initializeStackSlotColoringPass(PassRegistry&);
void initializeStraightLineStrengthReduceLegacyPassPass(PassRegistry &);
void initializeStripDeadDebugInfoPass(PassRegistry&);
void initializeStripDeadPrototypesLegacyPassPass(PassRegistry&);
void initializeStripDebugDeclarePass(PassRegistry&);
void initializeStripDebugMachineModulePass(PassRegistry &);
void initializeStripGCRelocatesLegacyPass(PassRegistry &);
void initializeStripNonDebugSymbolsPass(PassRegistry&);
void initializeStripNonLineTableDebugLegacyPassPass(PassRegistry &);
void initializeStripSymbolsPass(PassRegistry&);
void initializeStructurizeCFGLegacyPassPass(PassRegistry &);
void initializeTailCallElimPass(PassRegistry&);
void initializeTailDuplicatePass(PassRegistry&);
void initializeTargetLibraryInfoWrapperPassPass(PassRegistry&);
void initializeTargetPassConfigPass(PassRegistry&);
void initializeTargetTransformInfoWrapperPassPass(PassRegistry&);
void initializeTLSVariableHoistLegacyPassPass(PassRegistry &);
void initializeTwoAddressInstructionPassPass(PassRegistry&);
void initializeTypeBasedAAWrapperPassPass(PassRegistry&);
void initializeTypePromotionPass(PassRegistry&);
void initializeUnifyFunctionExitNodesLegacyPassPass(PassRegistry &);
void initializeUnifyLoopExitsLegacyPassPass(PassRegistry &);
void initializeUnpackMachineBundlesPass(PassRegistry&);
void initializeUnreachableBlockElimLegacyPassPass(PassRegistry&);
void initializeUnreachableMachineBlockElimPass(PassRegistry&);
void initializeVectorCombineLegacyPassPass(PassRegistry&);
void initializeVerifierLegacyPassPass(PassRegistry&);
void initializeVirtRegMapPass(PassRegistry&);
void initializeVirtRegRewriterPass(PassRegistry&);
void initializeWarnMissedTransformationsLegacyPass(PassRegistry &);
void initializeWasmEHPreparePass(PassRegistry&);
void initializeWholeProgramDevirtPass(PassRegistry&);
void initializeWinEHPreparePass(PassRegistry&);
void initializeWriteBitcodePassPass(PassRegistry&);
void initializeWriteThinLTOBitcodePass(PassRegistry&);
void initializeXRayInstrumentationPass(PassRegistry&);

} // end namespace llvm

#endif // LLVM_INITIALIZEPASSES_H
