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

#include "llvm/InitializePasses.h"

using namespace llvm;

/// initializeScalarOptsPasses - Initialize all passes linked into the
/// ScalarOpts library.
void llvm::initializeScalarOpts(PassRegistry &Registry) {
  initializeConstantHoistingLegacyPassPass(Registry);
  initializeDCELegacyPassPass(Registry);
  initializeScalarizerLegacyPassPass(Registry);
  initializeGuardWideningLegacyPassPass(Registry);
  initializeLoopGuardWideningLegacyPassPass(Registry);
  initializeGVNLegacyPassPass(Registry);
  initializeEarlyCSELegacyPassPass(Registry);
  initializeEarlyCSEMemSSALegacyPassPass(Registry);
  initializeMakeGuardsExplicitLegacyPassPass(Registry);
  initializeFlattenCFGLegacyPassPass(Registry);
  initializeInferAddressSpacesPass(Registry);
  initializeInstSimplifyLegacyPassPass(Registry);
  initializeLegacyLICMPassPass(Registry);
  initializeLegacyLoopSinkPassPass(Registry);
  initializeLoopDataPrefetchLegacyPassPass(Registry);
  initializeLoopInstSimplifyLegacyPassPass(Registry);
  initializeLoopPredicationLegacyPassPass(Registry);
  initializeLoopRotateLegacyPassPass(Registry);
  initializeLoopStrengthReducePass(Registry);
  initializeLoopUnrollPass(Registry);
  initializeLowerAtomicLegacyPassPass(Registry);
  initializeLowerConstantIntrinsicsPass(Registry);
  initializeLowerExpectIntrinsicPass(Registry);
  initializeLowerGuardIntrinsicLegacyPassPass(Registry);
  initializeLowerWidenableConditionLegacyPassPass(Registry);
  initializeMergeICmpsLegacyPassPass(Registry);
  initializeMergedLoadStoreMotionLegacyPassPass(Registry);
  initializeNaryReassociateLegacyPassPass(Registry);
  initializePartiallyInlineLibCallsLegacyPassPass(Registry);
  initializeReassociateLegacyPassPass(Registry);
  initializeRedundantDbgInstEliminationPass(Registry);
  initializeRegToMemLegacyPass(Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(Registry);
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
  initializePlaceBackedgeSafepointsLegacyPassPass(Registry);
  initializeLoopSimplifyCFGLegacyPassPass(Registry);
}
