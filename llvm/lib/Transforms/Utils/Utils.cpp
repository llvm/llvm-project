//===-- Utils.cpp - TransformUtils Infrastructure -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common initialization infrastructure for the
// TransformUtils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/Utils.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

/// initializeTransformUtils - Initialize all passes in the TransformUtils
/// library.
void llvm::initializeTransformUtils(PassRegistry &Registry) {
  initializeAddDiscriminatorsLegacyPassPass(Registry);
  initializeAssumeSimplifyPassLegacyPassPass(Registry);
  initializeAssumeBuilderPassLegacyPassPass(Registry);
  initializeBreakCriticalEdgesPass(Registry);
  initializeCanonicalizeAliasesLegacyPassPass(Registry);
  initializeCanonicalizeFreezeInLoopsPass(Registry);
  initializeInstNamerPass(Registry);
  initializeLCSSAWrapperPassPass(Registry);
  initializeLibCallsShrinkWrapLegacyPassPass(Registry);
  initializeLoopSimplifyPass(Registry);
  initializeLowerInvokeLegacyPassPass(Registry);
  initializeLowerSwitchLegacyPassPass(Registry);
  initializeNameAnonGlobalLegacyPassPass(Registry);
  initializePromoteLegacyPassPass(Registry);
  initializeStripNonLineTableDebugInfoPass(Registry);
  initializeUnifyFunctionExitNodesPass(Registry);
  initializeMetaRenamerPass(Registry);
  initializeStripGCRelocatesPass(Registry);
  initializePredicateInfoPrinterLegacyPassPass(Registry);
  initializeInjectTLIMappingsLegacyPass(Registry);
  initializeFixIrreduciblePass(Registry);
  initializeUnifyLoopExitsPass(Registry);
  initializeUniqueInternalLinkageNamesLegacyPassPass(Registry);
}

/// LLVMInitializeTransformUtils - C binding for initializeTransformUtilsPasses.
void LLVMInitializeTransformUtils(LLVMPassRegistryRef R) {
  initializeTransformUtils(*unwrap(R));
}

void LLVMAddLowerSwitchPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerSwitchPass());
}

void LLVMAddPromoteMemoryToRegisterPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPromoteMemoryToRegisterPass());
}

void LLVMAddAddDiscriminatorsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAddDiscriminatorsPass());
}
