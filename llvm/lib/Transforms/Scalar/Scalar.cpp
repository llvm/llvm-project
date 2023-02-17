//===-- Scalar.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMScalarOpts.a, which
// implements several scalar transformations over the LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"

using namespace llvm;

/// initializeScalarOptsPasses - Initialize all passes linked into the
/// ScalarOpts library.
void llvm::initializeScalarOpts(PassRegistry &Registry) {
  initializeADCELegacyPassPass(Registry);
  initializeBDCELegacyPassPass(Registry);
  initializeAlignmentFromAssumptionsPass(Registry);
  initializeCallSiteSplittingLegacyPassPass(Registry);
  initializeConstantHoistingLegacyPassPass(Registry);
  initializeCorrelatedValuePropagationPass(Registry);
  initializeDCELegacyPassPass(Registry);
  initializeDivRemPairsLegacyPassPass(Registry);
  initializeScalarizerLegacyPassPass(Registry);
  initializeDSELegacyPassPass(Registry);
  initializeGuardWideningLegacyPassPass(Registry);
  initializeLoopGuardWideningLegacyPassPass(Registry);
  initializeGVNLegacyPassPass(Registry);
  initializeNewGVNLegacyPassPass(Registry);
  initializeEarlyCSELegacyPassPass(Registry);
  initializeEarlyCSEMemSSALegacyPassPass(Registry);
  initializeMakeGuardsExplicitLegacyPassPass(Registry);
  initializeGVNHoistLegacyPassPass(Registry);
  initializeGVNSinkLegacyPassPass(Registry);
  initializeFlattenCFGLegacyPassPass(Registry);
  initializeIRCELegacyPassPass(Registry);
  initializeIndVarSimplifyLegacyPassPass(Registry);
  initializeInferAddressSpacesPass(Registry);
  initializeInstSimplifyLegacyPassPass(Registry);
  initializeJumpThreadingPass(Registry);
  initializeDFAJumpThreadingLegacyPassPass(Registry);
  initializeLegacyLICMPassPass(Registry);
  initializeLegacyLoopSinkPassPass(Registry);
  initializeLoopFuseLegacyPass(Registry);
  initializeLoopDataPrefetchLegacyPassPass(Registry);
  initializeLoopDeletionLegacyPassPass(Registry);
  initializeLoopAccessLegacyAnalysisPass(Registry);
  initializeLoopInstSimplifyLegacyPassPass(Registry);
  initializeLoopInterchangeLegacyPassPass(Registry);
  initializeLoopFlattenLegacyPassPass(Registry);
  initializeLoopPredicationLegacyPassPass(Registry);
  initializeLoopRotateLegacyPassPass(Registry);
  initializeLoopStrengthReducePass(Registry);
  initializeLoopRerollLegacyPassPass(Registry);
  initializeLoopUnrollPass(Registry);
  initializeLoopVersioningLICMLegacyPassPass(Registry);
  initializeLoopIdiomRecognizeLegacyPassPass(Registry);
  initializeLowerAtomicLegacyPassPass(Registry);
  initializeLowerConstantIntrinsicsPass(Registry);
  initializeLowerExpectIntrinsicPass(Registry);
  initializeLowerGuardIntrinsicLegacyPassPass(Registry);
  initializeLowerMatrixIntrinsicsLegacyPassPass(Registry);
  initializeLowerMatrixIntrinsicsMinimalLegacyPassPass(Registry);
  initializeLowerWidenableConditionLegacyPassPass(Registry);
  initializeMemCpyOptLegacyPassPass(Registry);
  initializeMergeICmpsLegacyPassPass(Registry);
  initializeMergedLoadStoreMotionLegacyPassPass(Registry);
  initializeNaryReassociateLegacyPassPass(Registry);
  initializePartiallyInlineLibCallsLegacyPassPass(Registry);
  initializeReassociateLegacyPassPass(Registry);
  initializeRedundantDbgInstEliminationPass(Registry);
  initializeRegToMemLegacyPass(Registry);
  initializeRewriteStatepointsForGCLegacyPassPass(Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(Registry);
  initializeSCCPLegacyPassPass(Registry);
  initializeSROALegacyPassPass(Registry);
  initializeCFGSimplifyPassPass(Registry);
  initializeStructurizeCFGLegacyPassPass(Registry);
  initializeSimpleLoopUnswitchLegacyPassPass(Registry);
  initializeSinkingLegacyPassPass(Registry);
  initializeTailCallElimPass(Registry);
  initializeTLSVariableHoistLegacyPassPass(Registry);
  initializeSeparateConstOffsetFromGEPLegacyPassPass(Registry);
  initializeSpeculativeExecutionLegacyPassPass(Registry);
  initializeStraightLineStrengthReduceLegacyPassPass(Registry);
  initializePlaceBackedgeSafepointsImplPass(Registry);
  initializePlaceSafepointsPass(Registry);
  initializeFloat2IntLegacyPassPass(Registry);
  initializeLoopSimplifyCFGLegacyPassPass(Registry);
}

void LLVMAddLoopSimplifyCFGPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopSimplifyCFGPass());
}

void LLVMInitializeScalarOpts(LLVMPassRegistryRef R) {
  initializeScalarOpts(*unwrap(R));
}

void LLVMAddAggressiveDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAggressiveDCEPass());
}

void LLVMAddDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadCodeEliminationPass());
}

void LLVMAddBitTrackingDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBitTrackingDCEPass());
}

void LLVMAddAlignmentFromAssumptionsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAlignmentFromAssumptionsPass());
}

void LLVMAddCFGSimplificationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createCFGSimplificationPass());
}

void LLVMAddDeadStoreEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadStoreEliminationPass());
}

void LLVMAddScalarizerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createScalarizerPass());
}

void LLVMAddGVNPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGVNPass());
}

void LLVMAddNewGVNPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createNewGVNPass());
}

void LLVMAddMergedLoadStoreMotionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMergedLoadStoreMotionPass());
}

void LLVMAddIndVarSimplifyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIndVarSimplifyPass());
}

void LLVMAddInstructionSimplifyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createInstSimplifyLegacyPass());
}

void LLVMAddJumpThreadingPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createJumpThreadingPass());
}

void LLVMAddLoopSinkPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopSinkPass());
}

void LLVMAddLICMPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLICMPass());
}

void LLVMAddLoopDeletionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopDeletionPass());
}

void LLVMAddLoopFlattenPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopFlattenPass());
}

void LLVMAddLoopIdiomPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopIdiomPass());
}

void LLVMAddLoopRotatePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopRotatePass());
}

void LLVMAddLoopRerollPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopRerollPass());
}

void LLVMAddLoopUnrollPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopUnrollPass());
}

void LLVMAddLowerAtomicPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerAtomicPass());
}

void LLVMAddMemCpyOptPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMemCpyOptPass());
}

void LLVMAddPartiallyInlineLibCallsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPartiallyInlineLibCallsPass());
}

void LLVMAddReassociatePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createReassociatePass());
}

void LLVMAddSCCPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSCCPPass());
}

void LLVMAddScalarReplAggregatesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSROAPass());
}

void LLVMAddScalarReplAggregatesPassSSA(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSROAPass());
}

void LLVMAddScalarReplAggregatesPassWithThreshold(LLVMPassManagerRef PM,
                                                  int Threshold) {
  unwrap(PM)->add(createSROAPass());
}

void LLVMAddSimplifyLibCallsPass(LLVMPassManagerRef PM) {
  // NOTE: The simplify-libcalls pass has been removed.
}

void LLVMAddTailCallEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createTailCallEliminationPass());
}

void LLVMAddDemoteMemoryToRegisterPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDemoteRegisterToMemoryPass());
}

void LLVMAddVerifierPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createVerifierPass());
}

void LLVMAddCorrelatedValuePropagationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createCorrelatedValuePropagationPass());
}

void LLVMAddEarlyCSEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createEarlyCSEPass(false/*=UseMemorySSA*/));
}

void LLVMAddEarlyCSEMemSSAPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createEarlyCSEPass(true/*=UseMemorySSA*/));
}

void LLVMAddGVNHoistLegacyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGVNHoistPass());
}

void LLVMAddTypeBasedAliasAnalysisPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createTypeBasedAAWrapperPass());
}

void LLVMAddScopedNoAliasAAPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createScopedNoAliasAAWrapperPass());
}

void LLVMAddBasicAliasAnalysisPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBasicAAWrapperPass());
}

void LLVMAddLowerConstantIntrinsicsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerConstantIntrinsicsPass());
}

void LLVMAddLowerExpectIntrinsicPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerExpectIntrinsicPass());
}

void LLVMAddUnifyFunctionExitNodesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createUnifyFunctionExitNodesPass());
}
