//===-- IPO.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the common infrastructure (including C bindings) for
// libLLVMIPO.a, which implements several transformations over the LLVM
// intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/IPO.h"
#include "llvm-c/Initialization.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"

using namespace llvm;

void llvm::initializeIPO(PassRegistry &Registry) {
  initializeAnnotation2MetadataLegacyPass(Registry);
  initializeConstantMergeLegacyPassPass(Registry);
  initializeCrossDSOCFIPass(Registry);
  initializeDAEPass(Registry);
  initializeDAHPass(Registry);
  initializeForceFunctionAttrsLegacyPassPass(Registry);
  initializeGlobalDCELegacyPassPass(Registry);
  initializeGlobalOptLegacyPassPass(Registry);
  initializeGlobalSplitPass(Registry);
  initializeAlwaysInlinerLegacyPassPass(Registry);
  initializeSimpleInlinerPass(Registry);
  initializeInferFunctionAttrsLegacyPassPass(Registry);
  initializeLoopExtractorLegacyPassPass(Registry);
  initializeSingleLoopExtractorPass(Registry);
  initializeAttributorLegacyPassPass(Registry);
  initializeAttributorCGSCCLegacyPassPass(Registry);
  initializePostOrderFunctionAttrsLegacyPassPass(Registry);
  initializeReversePostOrderFunctionAttrsLegacyPassPass(Registry);
  initializeIPSCCPLegacyPassPass(Registry);
  initializeBarrierNoopPass(Registry);
  initializeEliminateAvailableExternallyLegacyPassPass(Registry);
}

void LLVMInitializeIPO(LLVMPassRegistryRef R) {
  initializeIPO(*unwrap(R));
}

void LLVMAddConstantMergePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createConstantMergePass());
}

void LLVMAddDeadArgEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadArgEliminationPass());
}

void LLVMAddFunctionAttrsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPostOrderFunctionAttrsLegacyPass());
}

void LLVMAddFunctionInliningPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFunctionInliningPass());
}

void LLVMAddAlwaysInlinerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(llvm::createAlwaysInlinerLegacyPass());
}

void LLVMAddGlobalDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGlobalDCEPass());
}

void LLVMAddGlobalOptimizerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGlobalOptimizerPass());
}

void LLVMAddIPSCCPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIPSCCPPass());
}
