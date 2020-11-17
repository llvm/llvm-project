//===-- AnnotationRemarks.cpp - Generate remarks for annotated instrs. ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate remarks for instructions marked with !annotation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/AnnotationRemarks.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
using namespace llvm::ore;

#define DEBUG_TYPE "annotation-remarks"
#define REMARK_PASS DEBUG_TYPE

static void runImpl(Function &F) {
  if (!OptimizationRemarkEmitter::allowExtraAnalysis(F, REMARK_PASS))
    return;

  OptimizationRemarkEmitter ORE(&F);
  // For now, just generate a summary of the annotated instructions.
  MapVector<StringRef, unsigned> Mapping;
  for (Instruction &I : instructions(F)) {
    if (!I.hasMetadata(LLVMContext::MD_annotation))
      continue;
    for (const MDOperand &Op :
         I.getMetadata(LLVMContext::MD_annotation)->operands()) {
      auto Iter = Mapping.insert({cast<MDString>(Op.get())->getString(), 0});
      Iter.first->second++;
    }
  }

  Instruction *IP = &*F.begin()->begin();
  for (const auto &KV : Mapping)
    ORE.emit(OptimizationRemarkAnalysis(REMARK_PASS, "AnnotationSummary", IP)
             << "Annotated " << NV("count", KV.second) << " instructions with "
             << NV("type", KV.first));
}

namespace {

struct AnnotationRemarksLegacy : public FunctionPass {
  static char ID;

  AnnotationRemarksLegacy() : FunctionPass(ID) {
    initializeAnnotationRemarksLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    runImpl(F);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // end anonymous namespace

char AnnotationRemarksLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AnnotationRemarksLegacy, "annotation-remarks",
                      "Annotation Remarks", false, false)
INITIALIZE_PASS_END(AnnotationRemarksLegacy, "annotation-remarks",
                    "Annotation Remarks", false, false)

FunctionPass *llvm::createAnnotationRemarksLegacyPass() {
  return new AnnotationRemarksLegacy();
}

PreservedAnalyses AnnotationRemarksPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  runImpl(F);
  return PreservedAnalyses::all();
}
