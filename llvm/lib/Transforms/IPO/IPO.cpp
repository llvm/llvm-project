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
  initializeCalledValuePropagationLegacyPassPass(Registry);
  initializeConstantMergeLegacyPassPass(Registry);
  initializeCrossDSOCFIPass(Registry);
  initializeDAEPass(Registry);
  initializeDAHPass(Registry);
  initializeForceFunctionAttrsLegacyPassPass(Registry);
  initializeFunctionSpecializationLegacyPassPass(Registry);
  initializeGlobalDCELegacyPassPass(Registry);
  initializeGlobalOptLegacyPassPass(Registry);
  initializeGlobalSplitPass(Registry);
  initializeHotColdSplittingLegacyPassPass(Registry);
  initializeIROutlinerLegacyPassPass(Registry);
  initializeAlwaysInlinerLegacyPassPass(Registry);
  initializeSimpleInlinerPass(Registry);
  initializeInferFunctionAttrsLegacyPassPass(Registry);
  initializeInternalizeLegacyPassPass(Registry);
  initializeLoopExtractorLegacyPassPass(Registry);
  initializeBlockExtractorLegacyPassPass(Registry);
  initializeSingleLoopExtractorPass(Registry);
  initializeMergeFunctionsLegacyPassPass(Registry);
  initializePartialInlinerLegacyPassPass(Registry);
  initializeAttributorLegacyPassPass(Registry);
  initializeAttributorCGSCCLegacyPassPass(Registry);
  initializePostOrderFunctionAttrsLegacyPassPass(Registry);
  initializeReversePostOrderFunctionAttrsLegacyPassPass(Registry);
  initializeIPSCCPLegacyPassPass(Registry);
  initializeStripDeadPrototypesLegacyPassPass(Registry);
  initializeStripSymbolsPass(Registry);
  initializeStripDebugDeclarePass(Registry);
  initializeStripDeadDebugInfoPass(Registry);
  initializeStripNonDebugSymbolsPass(Registry);
  initializeBarrierNoopPass(Registry);
  initializeEliminateAvailableExternallyLegacyPassPass(Registry);
}

void LLVMInitializeIPO(LLVMPassRegistryRef R) {
  initializeIPO(*unwrap(R));
}

void LLVMAddCalledValuePropagationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createCalledValuePropagationPass());
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

void LLVMAddMergeFunctionsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMergeFunctionsPass());
}

void LLVMAddInternalizePass(LLVMPassManagerRef PM, unsigned AllButMain) {
  auto PreserveMain = [=](const GlobalValue &GV) {
    return AllButMain && GV.getName() == "main";
  };
  unwrap(PM)->add(createInternalizePass(PreserveMain));
}

void LLVMAddInternalizePassWithMustPreservePredicate(
    LLVMPassManagerRef PM,
    void *Context,
    LLVMBool (*Pred)(LLVMValueRef, void *)) {
  unwrap(PM)->add(createInternalizePass([=](const GlobalValue &GV) {
    return Pred(wrap(&GV), Context) == 0 ? false : true;
  }));
}

void LLVMAddStripDeadPrototypesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStripDeadPrototypesPass());
}

void LLVMAddStripSymbolsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStripSymbolsPass());
}
