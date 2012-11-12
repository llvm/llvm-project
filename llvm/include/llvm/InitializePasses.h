//===- llvm/InitializePasses.h -------- Initialize All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

/// initializeCore - Initialize all passes linked into the
/// TransformUtils library.
void initializeCore(PassRegistry&);

/// initializeTransformUtils - Initialize all passes linked into the
/// TransformUtils library.
void initializeTransformUtils(PassRegistry&);

/// initializeScalarOpts - Initialize all passes linked into the
/// ScalarOpts library.
void initializeScalarOpts(PassRegistry&);

/// initializeVectorization - Initialize all passes linked into the
/// Vectorize library.
void initializeVectorization(PassRegistry&);

/// initializeInstCombine - Initialize all passes linked into the
/// ScalarOpts library.
void initializeInstCombine(PassRegistry&);

/// initializeIPO - Initialize all passes linked into the IPO library.
void initializeIPO(PassRegistry&);

/// initializeInstrumentation - Initialize all passes linked into the
/// Instrumentation library.
void initializeInstrumentation(PassRegistry&);

/// initializeAnalysis - Initialize all passes linked into the Analysis library.
void initializeAnalysis(PassRegistry&);

/// initializeIPA - Initialize all passes linked into the IPA library.
void initializeIPA(PassRegistry&);

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void initializeCodeGen(PassRegistry&);

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void initializeTarget(PassRegistry&);

void initializeAAEvalPass(PassRegistry&);
void initializeADCEPass(PassRegistry&);
void initializeAliasAnalysisAnalysisGroup(PassRegistry&);
void initializeAliasAnalysisCounterPass(PassRegistry&);
void initializeAliasDebuggerPass(PassRegistry&);
void initializeAliasSetPrinterPass(PassRegistry&);
void initializeAlwaysInlinerPass(PassRegistry&);
void initializeArgPromotionPass(PassRegistry&);
void initializeBarrierNoopPass(PassRegistry&);
void initializeBasicAliasAnalysisPass(PassRegistry&);
void initializeBasicCallGraphPass(PassRegistry&);
void initializeBlockExtractorPassPass(PassRegistry&);
void initializeBlockFrequencyInfoPass(PassRegistry&);
void initializeBlockPlacementPass(PassRegistry&);
void initializeBoundsCheckingPass(PassRegistry&);
void initializeBranchFolderPassPass(PassRegistry&);
void initializeBranchProbabilityInfoPass(PassRegistry&);
void initializeBreakCriticalEdgesPass(PassRegistry&);
void initializeCFGOnlyPrinterPass(PassRegistry&);
void initializeCFGOnlyViewerPass(PassRegistry&);
void initializeCFGPrinterPass(PassRegistry&);
void initializeCFGSimplifyPassPass(PassRegistry&);
void initializeCFGViewerPass(PassRegistry&);
void initializeCalculateSpillWeightsPass(PassRegistry&);
void initializeCallGraphAnalysisGroup(PassRegistry&);
void initializeCodeGenPreparePass(PassRegistry&);
void initializeCodePlacementOptPass(PassRegistry&);
void initializeConstantMergePass(PassRegistry&);
void initializeConstantPropagationPass(PassRegistry&);
void initializeMachineCopyPropagationPass(PassRegistry&);
void initializeCostModelAnalysisPass(PassRegistry&);
void initializeCorrelatedValuePropagationPass(PassRegistry&);
void initializeDAEPass(PassRegistry&);
void initializeDAHPass(PassRegistry&);
void initializeDCEPass(PassRegistry&);
void initializeDSEPass(PassRegistry&);
void initializeDeadInstEliminationPass(PassRegistry&);
void initializeDeadMachineInstructionElimPass(PassRegistry&);
void initializeDependenceAnalysisPass(PassRegistry&);
void initializeDomOnlyPrinterPass(PassRegistry&);
void initializeDomOnlyViewerPass(PassRegistry&);
void initializeDomPrinterPass(PassRegistry&);
void initializeDomViewerPass(PassRegistry&);
void initializeDominanceFrontierPass(PassRegistry&);
void initializeDominatorTreePass(PassRegistry&);
void initializeEarlyIfConverterPass(PassRegistry&);
void initializeEdgeBundlesPass(PassRegistry&);
void initializeEdgeProfilerPass(PassRegistry&);
void initializeExpandPostRAPass(PassRegistry&);
void initializePathProfilerPass(PassRegistry&);
void initializeGCOVProfilerPass(PassRegistry&);
void initializeAddressSanitizerPass(PassRegistry&);
void initializeThreadSanitizerPass(PassRegistry&);
void initializeEarlyCSEPass(PassRegistry&);
void initializeExpandISelPseudosPass(PassRegistry&);
void initializeFindUsedTypesPass(PassRegistry&);
void initializeFunctionAttrsPass(PassRegistry&);
void initializeGCInfoDeleterPass(PassRegistry&);
void initializeGCMachineCodeAnalysisPass(PassRegistry&);
void initializeGCModuleInfoPass(PassRegistry&);
void initializeGVNPass(PassRegistry&);
void initializeGlobalDCEPass(PassRegistry&);
void initializeGlobalOptPass(PassRegistry&);
void initializeGlobalsModRefPass(PassRegistry&);
void initializeIPCPPass(PassRegistry&);
void initializeIPSCCPPass(PassRegistry&);
void initializeIVUsersPass(PassRegistry&);
void initializeIfConverterPass(PassRegistry&);
void initializeIndVarSimplifyPass(PassRegistry&);
void initializeInstCombinerPass(PassRegistry&);
void initializeInstCountPass(PassRegistry&);
void initializeInstNamerPass(PassRegistry&);
void initializeInternalizePassPass(PassRegistry&);
void initializeIntervalPartitionPass(PassRegistry&);
void initializeJumpThreadingPass(PassRegistry&);
void initializeLCSSAPass(PassRegistry&);
void initializeLICMPass(PassRegistry&);
void initializeLazyValueInfoPass(PassRegistry&);
void initializeLibCallAliasAnalysisPass(PassRegistry&);
void initializeLintPass(PassRegistry&);
void initializeLiveDebugVariablesPass(PassRegistry&);
void initializeLiveIntervalsPass(PassRegistry&);
void initializeLiveRegMatrixPass(PassRegistry&);
void initializeLiveStacksPass(PassRegistry&);
void initializeLiveVariablesPass(PassRegistry&);
void initializeLoaderPassPass(PassRegistry&);
void initializeProfileMetadataLoaderPassPass(PassRegistry&);
void initializePathProfileLoaderPassPass(PassRegistry&);
void initializeLocalStackSlotPassPass(PassRegistry&);
void initializeLoopDeletionPass(PassRegistry&);
void initializeLoopExtractorPass(PassRegistry&);
void initializeLoopInfoPass(PassRegistry&);
void initializeLoopInstSimplifyPass(PassRegistry&);
void initializeLoopRotatePass(PassRegistry&);
void initializeLoopSimplifyPass(PassRegistry&);
void initializeLoopStrengthReducePass(PassRegistry&);
void initializeGlobalMergePass(PassRegistry&);
void initializeLoopUnrollPass(PassRegistry&);
void initializeLoopUnswitchPass(PassRegistry&);
void initializeLoopIdiomRecognizePass(PassRegistry&);
void initializeLowerAtomicPass(PassRegistry&);
void initializeLowerExpectIntrinsicPass(PassRegistry&);
void initializeLowerIntrinsicsPass(PassRegistry&);
void initializeLowerInvokePass(PassRegistry&);
void initializeLowerSwitchPass(PassRegistry&);
void initializeMachineBlockFrequencyInfoPass(PassRegistry&);
void initializeMachineBlockPlacementPass(PassRegistry&);
void initializeMachineBlockPlacementStatsPass(PassRegistry&);
void initializeMachineBranchProbabilityInfoPass(PassRegistry&);
void initializeMachineCSEPass(PassRegistry&);
void initializeMachineDominatorTreePass(PassRegistry&);
void initializeMachinePostDominatorTreePass(PassRegistry&);
void initializeMachineLICMPass(PassRegistry&);
void initializeMachineLoopInfoPass(PassRegistry&);
void initializeMachineLoopRangesPass(PassRegistry&);
void initializeMachineModuleInfoPass(PassRegistry&);
void initializeMachineSchedulerPass(PassRegistry&);
void initializeMachineSinkingPass(PassRegistry&);
void initializeMachineTraceMetricsPass(PassRegistry&);
void initializeMachineVerifierPassPass(PassRegistry&);
void initializeMemCpyOptPass(PassRegistry&);
void initializeMemDepPrinterPass(PassRegistry&);
void initializeMemoryDependenceAnalysisPass(PassRegistry&);
void initializeMetaRenamerPass(PassRegistry&);
void initializeMergeFunctionsPass(PassRegistry&);
void initializeModuleDebugInfoPrinterPass(PassRegistry&);
void initializeNoAAPass(PassRegistry&);
void initializeNoProfileInfoPass(PassRegistry&);
void initializeNoPathProfileInfoPass(PassRegistry&);
void initializeObjCARCAliasAnalysisPass(PassRegistry&);
void initializeObjCARCAPElimPass(PassRegistry&);
void initializeObjCARCExpandPass(PassRegistry&);
void initializeObjCARCContractPass(PassRegistry&);
void initializeObjCARCOptPass(PassRegistry&);
void initializeOptimalEdgeProfilerPass(PassRegistry&);
void initializeOptimizePHIsPass(PassRegistry&);
void initializePEIPass(PassRegistry&);
void initializePHIEliminationPass(PassRegistry&);
void initializePartialInlinerPass(PassRegistry&);
void initializePeepholeOptimizerPass(PassRegistry&);
void initializePostDomOnlyPrinterPass(PassRegistry&);
void initializePostDomOnlyViewerPass(PassRegistry&);
void initializePostDomPrinterPass(PassRegistry&);
void initializePostDomViewerPass(PassRegistry&);
void initializePostDominatorTreePass(PassRegistry&);
void initializePostRASchedulerPass(PassRegistry&);
void initializePreVerifierPass(PassRegistry&);
void initializePrintDbgInfoPass(PassRegistry&);
void initializePrintFunctionPassPass(PassRegistry&);
void initializePrintModulePassPass(PassRegistry&);
void initializeProcessImplicitDefsPass(PassRegistry&);
void initializeProfileEstimatorPassPass(PassRegistry&);
void initializeProfileInfoAnalysisGroup(PassRegistry&);
void initializePathProfileInfoAnalysisGroup(PassRegistry&);
void initializePathProfileVerifierPass(PassRegistry&);
void initializeProfileVerifierPassPass(PassRegistry&);
void initializePromotePassPass(PassRegistry&);
void initializePruneEHPass(PassRegistry&);
void initializeReassociatePass(PassRegistry&);
void initializeRegToMemPass(PassRegistry&);
void initializeRegionInfoPass(PassRegistry&);
void initializeRegionOnlyPrinterPass(PassRegistry&);
void initializeRegionOnlyViewerPass(PassRegistry&);
void initializeRegionPrinterPass(PassRegistry&);
void initializeRegionViewerPass(PassRegistry&);
void initializeSCCPPass(PassRegistry&);
void initializeSROAPass(PassRegistry&);
void initializeSROA_DTPass(PassRegistry&);
void initializeSROA_SSAUpPass(PassRegistry&);
void initializeScalarEvolutionAliasAnalysisPass(PassRegistry&);
void initializeScalarEvolutionPass(PassRegistry&);
void initializeSimpleInlinerPass(PassRegistry&);
void initializeRegisterCoalescerPass(PassRegistry&);
void initializeSimplifyLibCallsPass(PassRegistry&);
void initializeSingleLoopExtractorPass(PassRegistry&);
void initializeSinkingPass(PassRegistry&);
void initializeSlotIndexesPass(PassRegistry&);
void initializeSpillPlacementPass(PassRegistry&);
void initializeStackProtectorPass(PassRegistry&);
void initializeStackColoringPass(PassRegistry&);
void initializeStackSlotColoringPass(PassRegistry&);
void initializeStripDeadDebugInfoPass(PassRegistry&);
void initializeStripDeadPrototypesPassPass(PassRegistry&);
void initializeStripDebugDeclarePass(PassRegistry&);
void initializeStripNonDebugSymbolsPass(PassRegistry&);
void initializeStripSymbolsPass(PassRegistry&);
void initializeStrongPHIEliminationPass(PassRegistry&);
void initializeTailCallElimPass(PassRegistry&);
void initializeTailDuplicatePassPass(PassRegistry&);
void initializeTargetPassConfigPass(PassRegistry&);
void initializeDataLayoutPass(PassRegistry&);
void initializeTargetTransformInfoPass(PassRegistry&);
void initializeTargetLibraryInfoPass(PassRegistry&);
void initializeTwoAddressInstructionPassPass(PassRegistry&);
void initializeTypeBasedAliasAnalysisPass(PassRegistry&);
void initializeUnifyFunctionExitNodesPass(PassRegistry&);
void initializeUnreachableBlockElimPass(PassRegistry&);
void initializeUnreachableMachineBlockElimPass(PassRegistry&);
void initializeVerifierPass(PassRegistry&);
void initializeVirtRegMapPass(PassRegistry&);
void initializeVirtRegRewriterPass(PassRegistry&);
void initializeInstSimplifierPass(PassRegistry&);
void initializeUnpackMachineBundlesPass(PassRegistry&);
void initializeFinalizeMachineBundlesPass(PassRegistry&);
void initializeLoopVectorizePass(PassRegistry&);
void initializeBBVectorizePass(PassRegistry&);
void initializeMachineFunctionPrinterPassPass(PassRegistry&);
}

#endif
