//===-- Annotation2Metadata.cpp - Add !annotation metadata. ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Add !annotation metadata for entries in @llvm.global.anotations, generated
// using __attribute__((annotate("_name"))) on functions in Clang.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Annotation2Metadata.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "annotation2metadata"

static bool convertAnnotation2Metadata(Module &M) {
  // Only add !annotation metadata if the corresponding remarks pass is also
  // enabled.
  if (!OptimizationRemarkEmitter::allowExtraAnalysis(M.getContext(),
                                                     "annotation-remarks"))
    return false;

  auto *Annotations = M.getGlobalVariable("llvm.global.annotations");
  auto *C = dyn_cast_or_null<Constant>(Annotations);
  if (!C || C->getNumOperands() != 1)
    return false;

  C = cast<Constant>(C->getOperand(0));

  // Iterate over all entries in C and attach !annotation metadata to suitable
  // entries.
  for (auto &Op : C->operands()) {
    // Look at the operands to check if we can use the entry to generate
    // !annotation metadata.
    auto *OpC = dyn_cast<ConstantStruct>(&Op);
    if (!OpC || OpC->getNumOperands() != 4)
      continue;
    auto *StrC = dyn_cast<GlobalValue>(OpC->getOperand(1)->stripPointerCasts());
    if (!StrC)
      continue;
    auto *StrData = dyn_cast<ConstantDataSequential>(StrC->getOperand(0));
    if (!StrData)
      continue;
    auto *Fn = dyn_cast<Function>(OpC->getOperand(0)->stripPointerCasts());
    if (!Fn)
      continue;

    // Add annotation to all instructions in the function.
    for (auto &I : instructions(Fn))
      I.addAnnotationMetadata(StrData->getAsCString());
  }
  return true;
}

namespace {
struct Annotation2MetadataLegacy : public ModulePass {
  static char ID;

  Annotation2MetadataLegacy() : ModulePass(ID) {
    initializeAnnotation2MetadataLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override { return convertAnnotation2Metadata(M); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // end anonymous namespace

char Annotation2MetadataLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(Annotation2MetadataLegacy, DEBUG_TYPE,
                      "Annotation2Metadata", false, false)
INITIALIZE_PASS_END(Annotation2MetadataLegacy, DEBUG_TYPE,
                    "Annotation2Metadata", false, false)

ModulePass *llvm::createAnnotation2MetadataLegacyPass() {
  return new Annotation2MetadataLegacy();
}

PreservedAnalyses Annotation2MetadataPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  convertAnnotation2Metadata(M);
  return PreservedAnalyses::all();
}
