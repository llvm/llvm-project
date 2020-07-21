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

/// Initialize all passes linked into the Coroutines library.
void initializeCoroutines(PassRegistry&);

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
void initializeModuleAddressSanitizerLegacyPassPass(PassRegistry &);
void initializeASanGlobalsMetadataWrapperPassPass(PassRegistry &);
void initializeAddressSanitizerLegacyPassPass(PassRegistry &);
void initializeAggressiveInstCombinerLegacyPassPass(PassRegistry&);
void initializeAliasSetPrinterPass(PassRegistry&);
void initializeAlignmentFromAssumptionsPass(PassRegistry&);
void initializeAlwaysInlinerLegacyPassPass(PassRegistry&);
void initializeAssumeSimplifyPassLegacyPassPass(PassRegistry &);
void initializeAssumeBuilderPassLegacyPassPass(PassRegistry &);
void initializeOpenMPOptLegacyPassPass(PassRegistry &);
void initializeArgPromotionPass(PassRegistry&);
void initializeAssumptionCacheTrackerPass(PassRegistry&);
void initializeAtomicExpandPass(PassRegistry&);
void initializeAttributorLegacyPassPass(PassRegistry&);
void initializeAttributorCGSCCLegacyPassPass(PassRegistry &);
void initializeBBSectionsPreparePass(PassRegistry &);
void initializeBDCELegacyPassPass(PassRegistry&);
void initializeBarrierNoopPass(PassRegistry&);
void initializeBasicAAWrapperPassPass(PassRegistry&);
void initializeBlockExtractorPass(PassRegistry &);
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
void initializeCodeGenPreparePass(PassRegistry&);
void initializeConstantHoistingLegacyPassPass(PassRegistry&);
void initializeConstantMergeLegacyPassPass(PassRegistry&);
void initializeConstantPropagationPass(PassRegistry&);
void initializeControlHeightReductionLegacyPassPass(PassRegistry&);
void initializeCorrelatedValuePropagationPass(PassRegistry&);
void initializeCostModelAnalysisPass(PassRegistry&);
void initializeCrossDSOCFIPass(PassRegistry&);
void initializeDAEPass(PassRegistry&);
void initializeDAHPass(PassRegistry&);
void initializeDCELegacyPassPass(PassRegistry&);
void initializeDSELegacyPassPass(PassRegistry&);
void initializeDataFlowSanitizerPass(PassRegistry&);
void initializeDeadInstEliminationPass(PassRegistry&);
void initializeDeadMachineInstructionElimPass(PassRegistry&);
void initializeDebugifyMachineModulePass(PassRegistry &);
void initializeDelinearizationPass(PassRegistry&);
void initializeDemandedBitsWrapperPassPass(PassRegistry&);
void initializeDependenceAnalysisPass(PassRegistry&);
void initializeDependenceAnalysisWrapperPassPass(PassRegistry&);
void initializeDetectDeadLanesPass(PassRegistry&);
void initializeDivRemPairsLegacyPassPass(PassRegistry&);
void initializeDomOnlyPrinterPass(PassRegistry&);
void initializeDomOnlyViewerPass(PassRegistry&);
void initializeDomPrinterPass(PassRegistry&);
void initializeDomViewerPass(PassRegistry&);
void initializeDominanceFrontierWrapperPassPass(PassRegistry&);
void initializeDominatorTreeWrapperPassPass(PassRegistry&);
void initializeDwarfEHPreparePass(PassRegistry&);
void initializeEarlyCSELegacyPassPass(PassRegistry&);
void initializeEarlyCSEMemSSALegacyPassPass(PassRegistry&);
void initializeEarlyIfConverterPass(PassRegistry&);
void initializeEarlyIfPredicatorPass(PassRegistry &);
void initializeEarlyMachineLICMPass(PassRegistry&);
void initializeEarlyTailDuplicatePass(PassRegistry&);
void initializeEdgeBundlesPass(PassRegistry&);
void initializeEliminateAvailableExternallyLegacyPassPass(PassRegistry&);
void initializeEntryExitInstrumenterPass(PassRegistry&);
void initializeExpandMemCmpPassPass(PassRegistry&);
void initializeExpandPostRAPass(PassRegistry&);
void initializeExpandReductionsPass(PassRegistry&);
void initializeMakeGuardsExplicitLegacyPassPass(PassRegistry&);
void initializeExternalAAWrapperPassPass(PassRegistry&);
void initializeFEntryInserterPass(PassRegistry&);
void initializeFinalizeISelPass(PassRegistry&);
void initializeFinalizeMachineBundlesPass(PassRegistry&);
void initializeFixIrreduciblePass(PassRegistry &);
void initializeFixupStatepointCallerSavedPass(PassRegistry&);
void initializeFlattenCFGPassPass(PassRegistry&);
void initializeFloat2IntLegacyPassPass(PassRegistry&);
void initializeForceFunctionAttrsLegacyPassPass(PassRegistry&);
void initializeForwardControlFlowIntegrityPass(PassRegistry&);
void initializeFuncletLayoutPass(PassRegistry&);
void initializeFunctionImportLegacyPassPass(PassRegistry&);
void initializeGCMachineCodeAnalysisPass(PassRegistry&);
void initializeGCModuleInfoPass(PassRegistry&);
void initializeGCOVProfilerLegacyPassPass(PassRegistry&);
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
void initializeHotColdSplittingLegacyPassPass(PassRegistry&);
void initializeHWAddressSanitizerLegacyPassPass(PassRegistry &);
void initializeIPCPPass(PassRegistry&);
void initializeIPSCCPLegacyPassPass(PassRegistry&);
void initializeIRCELegacyPassPass(PassRegistry&);
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
void initializeInstCountPass(PassRegistry&);
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
void initializeLintPass(PassRegistry&);
void initializeLiveDebugValuesPass(PassRegistry&);
void initializeLiveDebugVariablesPass(PassRegistry&);
void initializeLiveIntervalsPass(PassRegistry&);
void initializeLiveRangeShrinkPass(PassRegistry&);
void initializeLiveRegMatrixPass(PassRegistry&);
void initializeLiveStacksPass(PassRegistry&);
void initializeLiveVariablesPass(PassRegistry&);
void initializeLoadStoreVectorizerLegacyPassPass(PassRegistry&);
void initializeLoaderPassPass(PassRegistry&);
void initializeLocalStackSlotPassPass(PassRegistry&);
void initializeLocalizerPass(PassRegistry&);
void initializeLoopAccessLegacyAnalysisPass(PassRegistry&);
void initializeLoopDataPrefetchLegacyPassPass(PassRegistry&);
void initializeLoopDeletionLegacyPassPass(PassRegistry&);
void initializeLoopDistributeLegacyPass(PassRegistry&);
void initializeLoopExtractorPass(PassRegistry&);
void initializeLoopGuardWideningLegacyPassPass(PassRegistry&);
void initializeLoopFuseLegacyPass(PassRegistry&);
void initializeLoopIdiomRecognizeLegacyPassPass(PassRegistry&);
void initializeLoopInfoWrapperPassPass(PassRegistry&);
void initializeLoopInstSimplifyLegacyPassPass(PassRegistry&);
void initializeLoopInterchangePass(PassRegistry&);
void initializeLoopLoadEliminationPass(PassRegistry&);
void initializeLoopPassPass(PassRegistry&);
void initializeLoopPredicationLegacyPassPass(PassRegistry&);
void initializeLoopRerollPass(PassRegistry&);
void initializeLoopRotateLegacyPassPass(PassRegistry&);
void initializeLoopSimplifyCFGLegacyPassPass(PassRegistry&);
void initializeLoopSimplifyPass(PassRegistry&);
void initializeLoopStrengthReducePass(PassRegistry&);
void initializeLoopUnrollAndJamPass(PassRegistry&);
void initializeLoopUnrollPass(PassRegistry&);
void initializeLoopUnswitchPass(PassRegistry&);
void initializeLoopVectorizePass(PassRegistry&);
void initializeLoopVersioningLICMPass(PassRegistry&);
void initializeLoopVersioningPassPass(PassRegistry&);
void initializeLowerAtomicLegacyPassPass(PassRegistry&);
void initializeLowerConstantIntrinsicsPass(PassRegistry&);
void initializeLowerEmuTLSPass(PassRegistry&);
void initializeLowerExpectIntrinsicPass(PassRegistry&);
void initializeLowerGuardIntrinsicLegacyPassPass(PassRegistry&);
void initializeLowerWidenableConditionLegacyPassPass(PassRegistry&);
void initializeLowerIntrinsicsPass(PassRegistry&);
void initializeLowerInvokeLegacyPassPass(PassRegistry&);
void initializeLowerSwitchPass(PassRegistry&);
void initializeLowerTypeTestsPass(PassRegistry&);
void initializeLowerMatrixIntrinsicsLegacyPassPass(PassRegistry &);
void initializeLowerMatrixIntrinsicsMinimalLegacyPassPass(PassRegistry &);
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
void initializeMachineDominanceFrontierPass(PassRegistry&);
void initializeMachineDominatorTreePass(PassRegistry&);
void initializeMachineFunctionPrinterPassPass(PassRegistry&);
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
void initializeMemorySanitizerLegacyPassPass(PassRegistry&);
void initializeMergeFunctionsLegacyPassPass(PassRegistry&);
void initializeMergeICmpsLegacyPassPass(PassRegistry &);
void initializeMergedLoadStoreMotionLegacyPassPass(PassRegistry&);
void initializeMetaRenamerPass(PassRegistry&);
void initializeModuleDebugInfoPrinterPass(PassRegistry&);
void initializeModuleSummaryIndexWrapperPassPass(PassRegistry&);
void initializeModuloScheduleTestPass(PassRegistry&);
void initializeMustExecutePrinterPass(PassRegistry&);
void initializeMustBeExecutedContextPrinterPass(PassRegistry&);
void initializeNameAnonGlobalLegacyPassPass(PassRegistry&);
void initializeUniqueInternalLinkageNamesLegacyPassPass(PassRegistry &);
void initializeNaryReassociateLegacyPassPass(PassRegistry&);
void initializeNewGVNLegacyPassPass(PassRegistry&);
void initializeObjCARCAAWrapperPassPass(PassRegistry&);
void initializeObjCARCAPElimPass(PassRegistry&);
void initializeObjCARCContractPass(PassRegistry&);
void initializeObjCARCExpandPass(PassRegistry&);
void initializeObjCARCOptPass(PassRegistry&);
void initializeOptimizationRemarkEmitterWrapperPassPass(PassRegistry&);
void initializeOptimizePHIsPass(PassRegistry&);
void initializePAEvalPass(PassRegistry&);
void initializePEIPass(PassRegistry&);
void initializePGOIndirectCallPromotionLegacyPassPass(PassRegistry&);
void initializePGOInstrumentationGenLegacyPassPass(PassRegistry&);
void initializePGOInstrumentationUseLegacyPassPass(PassRegistry&);
void initializePGOInstrumentationGenCreateVarLegacyPassPass(PassRegistry&);
void initializePGOMemOPSizeOptLegacyPassPass(PassRegistry&);
void initializePHIEliminationPass(PassRegistry&);
void initializePartialInlinerLegacyPassPass(PassRegistry&);
void initializePartiallyInlineLibCallsLegacyPassPass(PassRegistry&);
void initializePatchableFunctionPass(PassRegistry&);
void initializePeepholeOptimizerPass(PassRegistry&);
void initializePhiValuesWrapperPassPass(PassRegistry&);
void initializePhysicalRegisterUsageInfoPass(PassRegistry&);
void initializePlaceBackedgeSafepointsImplPass(PassRegistry&);
void initializePlaceSafepointsPass(PassRegistry&);
void initializePostDomOnlyPrinterPass(PassRegistry&);
void initializePostDomOnlyViewerPass(PassRegistry&);
void initializePostDomPrinterPass(PassRegistry&);
void initializePostDomViewerPass(PassRegistry&);
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
void initializeRAGreedyPass(PassRegistry&);
void initializeReachingDefAnalysisPass(PassRegistry&);
void initializeReassociateLegacyPassPass(PassRegistry&);
void initializeRedundantDbgInstEliminationPass(PassRegistry&);
void initializeRegAllocFastPass(PassRegistry&);
void initializeRegBankSelectPass(PassRegistry&);
void initializeRegToMemPass(PassRegistry&);
void initializeRegUsageInfoCollectorPass(PassRegistry&);
void initializeRegUsageInfoPropagationPass(PassRegistry&);
void initializeRegionInfoPassPass(PassRegistry&);
void initializeRegionOnlyPrinterPass(PassRegistry&);
void initializeRegionOnlyViewerPass(PassRegistry&);
void initializeRegionPrinterPass(PassRegistry&);
void initializeRegionViewerPass(PassRegistry&);
void initializeRegisterCoalescerPass(PassRegistry&);
void initializeRenameIndependentSubregsPass(PassRegistry&);
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
void initializeModuleSanitizerCoverageLegacyPassPass(PassRegistry &);
void initializeScalarEvolutionWrapperPassPass(PassRegistry&);
void initializeScalarizeMaskedMemIntrinPass(PassRegistry&);
void initializeScalarizerLegacyPassPass(PassRegistry&);
void initializeScavengerTestPass(PassRegistry&);
void initializeScopedNoAliasAAWrapperPassPass(PassRegistry&);
void initializeSeparateConstOffsetFromGEPPass(PassRegistry&);
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
void initializeStraightLineStrengthReducePass(PassRegistry&);
void initializeStripDeadDebugInfoPass(PassRegistry&);
void initializeStripDeadPrototypesLegacyPassPass(PassRegistry&);
void initializeStripDebugDeclarePass(PassRegistry&);
void initializeStripDebugMachineModulePass(PassRegistry &);
void initializeStripGCRelocatesPass(PassRegistry&);
void initializeStripNonDebugSymbolsPass(PassRegistry&);
void initializeStripNonLineTableDebugInfoPass(PassRegistry&);
void initializeStripSymbolsPass(PassRegistry&);
void initializeStructurizeCFGPass(PassRegistry&);
void initializeTailCallElimPass(PassRegistry&);
void initializeTailDuplicatePass(PassRegistry&);
void initializeTargetLibraryInfoWrapperPassPass(PassRegistry&);
void initializeTargetPassConfigPass(PassRegistry&);
void initializeTargetTransformInfoWrapperPassPass(PassRegistry&);
void initializeThreadSanitizerLegacyPassPass(PassRegistry&);
void initializeTwoAddressInstructionPassPass(PassRegistry&);
void initializeTypeBasedAAWrapperPassPass(PassRegistry&);
void initializeTypePromotionPass(PassRegistry&);
void initializeUnifyFunctionExitNodesPass(PassRegistry&);
void initializeUnifyLoopExitsPass(PassRegistry &);
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
