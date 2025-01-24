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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/MemoryOpRemark.h"

using namespace llvm;
using namespace llvm::ore;

#define DEBUG_TYPE "annotation-remarks"
#define REMARK_PASS DEBUG_TYPE

/* TO_UPSTREAM(BoundsSafety) ON */
static void tryEmitBoundsSafetyRemark(ArrayRef<Instruction *> Instructions,
                                   OptimizationRemarkEmitter &ORE) {
  SmallSetVector<StringRef, 4> Annotations;
  SmallSetVector<StringRef, 4> MissedOptimizationInfo;
  SmallVector<std::string> Pretty;
  for (auto *I : Instructions) {
    if (!I->hasMetadata(LLVMContext::MD_annotation))
      continue;
    bool HasBoundsSafety = false;
    for (const MDOperand &Op :
         I->getMetadata(LLVMContext::MD_annotation)->operands()) {
      StringRef S;
      StringRef ExtraInfo;
      if (isa<MDString>(Op.get()))
        S = cast<MDString>(Op.get())->getString();
      else {
        auto *AnnotationTuple = cast<MDTuple>(Op.get());
        S = cast<MDString>(AnnotationTuple->getOperand(0).get())->getString();
        ExtraInfo =
            cast<MDString>(AnnotationTuple->getOperand(1).get())->getString();
      }
      if (S.starts_with("bounds-safety")) {
        Annotations.insert(S);
        HasBoundsSafety = true;
      }
      if (!ExtraInfo.empty())
        MissedOptimizationInfo.insert(ExtraInfo);
    }

    if (!HasBoundsSafety)
      continue;
    switch (I->getOpcode()) {
    case Instruction::ICmp: {
      std::string R =
          std::string("cmp ") +
          CmpInst::getPredicateName(cast<CmpInst>(I)->getPredicate()).str();
      if (I->getOperand(1)->getName().starts_with("upper"))
        R += " UPPER_BOUND";

      Pretty.push_back(R);
      break;
    }
    case Instruction::And:
      Pretty.push_back("and");
      break;
    case Instruction::Br:
      if (cast<BranchInst>(I)->isConditional())
        Pretty.push_back("cond branch");
      else
        Pretty.push_back("uncond branch");
      break;
    case Instruction::Call: {
      auto II = dyn_cast<IntrinsicInst>(I);
      if (II && II->getIntrinsicID() == Intrinsic::ubsantrap)
        Pretty.push_back("trap");
      else
        Pretty.push_back("call");
      break;
    }
    default:
      Pretty.push_back("other");
      break;
    }
    Pretty.back() = Pretty.back() + " (LLVM IR '" + I->getOpcodeName() + "')";
  }

  if (!Pretty.empty()) {
    auto ORA = OptimizationRemarkAnalysis(REMARK_PASS, "BoundsSafetyCheck",
                                          Instructions[0]);
    ORA << "Inserted " << NV("count", Pretty.size()) << " LLVM IR instruction"
        << (Pretty.size() > 1 ? "s" : "") << "\n"
        << "used for:\n"
        << join(Annotations, ", ") << "\n\ninstructions:\n"
        << join(Pretty, "\n");
    if (!MissedOptimizationInfo.empty())
      ORA << "Missed Optimization Info\n" << join(MissedOptimizationInfo, "\n");
    ORE.emit(ORA);
  }
}
/* TO_UPSTREAM(BoundsSafety) OFF */

static void tryEmitAutoInitRemark(ArrayRef<Instruction *> Instructions,
                                  OptimizationRemarkEmitter &ORE,
                                  const TargetLibraryInfo &TLI) {
  // For every auto-init annotation generate a separate remark.
  for (Instruction *I : Instructions) {
    if (!AutoInitRemark::canHandle(I))
      continue;

    Function &F = *I->getParent()->getParent();
    const DataLayout &DL = F.getDataLayout();
    AutoInitRemark Remark(ORE, REMARK_PASS, DL, TLI);
    Remark.visit(I);
  }
}

static void runImpl(Function &F, const TargetLibraryInfo &TLI) {
  if (!OptimizationRemarkEmitter::allowExtraAnalysis(F, REMARK_PASS))
    return;

  // Track all annotated instructions aggregated based on their debug location.
  MapVector<MDNode *, SmallVector<Instruction *, 4>> DebugLoc2Annotated;

  OptimizationRemarkEmitter ORE(&F);
  // First, generate a summary of the annotated instructions.
  MapVector<StringRef, unsigned> Mapping;
/* TO_UPSTREAM(BoundsSafety) ON */
  unsigned BoundsSafetySummaryCount = 0;
/* TO_UPSTREAM(BoundsSafety) OFF */
  for (Instruction &I : instructions(F)) {
    if (!I.hasMetadata(LLVMContext::MD_annotation))
      continue;
    DebugLoc2Annotated[I.getDebugLoc().getAsMDNode()].push_back(&I);

    for (const MDOperand &Op :
         I.getMetadata(LLVMContext::MD_annotation)->operands()) {
      StringRef AnnotationStr =
          isa<MDString>(Op.get())
              ? cast<MDString>(Op.get())->getString()
              : cast<MDString>(cast<MDTuple>(Op.get())->getOperand(0).get())
                    ->getString();
      Mapping[AnnotationStr]++;
    }
    /* TO_UPSTREAM(BoundsSafety) ON */
    // Compute the number of instructions with -fbounds-safety annotation.
    if (any_of(I.getMetadata(LLVMContext::MD_annotation)->operands(),
               [](const MDOperand &Op) {
                 // skip bounds-safety-missed remarks.
                 if (!isa<MDString>(Op.get())) {
                   auto *AnnotationTuple = cast<MDTuple>(Op.get());
                   auto Str =
                       cast<MDString>(AnnotationTuple->getOperand(0).get())
                           ->getString();
                   if (Str.starts_with("bounds-safety-missed"))
                     return false;
                 }
                 auto AnnotationStr = cast<MDString>(Op.get())->getString();
                 return AnnotationStr.starts_with("bounds-safety");
               }))
      BoundsSafetySummaryCount++;
    /* TO_UPSTREAM(BoundsSafety) OFF */
  }

  for (const auto &KV : Mapping)
    ORE.emit(OptimizationRemarkAnalysis(REMARK_PASS, "AnnotationSummary",
                                        F.getSubprogram(), &F.front())
             << "Annotated " << NV("count", KV.second) << " instructions with "
             << NV("type", KV.first));

  /* TO_UPSTREAM(BoundsSafety) ON */
  if (BoundsSafetySummaryCount > 0)
    ORE.emit(OptimizationRemarkAnalysis(REMARK_PASS, "AnnotationSummary",
                                        F.getSubprogram(), &F.front())
             << "Annotated " << NV("count", BoundsSafetySummaryCount)
             << " instructions with " << NV("type", "bounds-safety-total-summary"));
  /* TO_UPSTREAM(BoundsSafety) OFF */

  // For each debug location, look for all the instructions with
  // annotations and generate more detailed remarks to be displayed at
  // that location.
  for (auto &KV : DebugLoc2Annotated) {
    // Don't generate remarks with no debug location.
    if (!KV.first)
      continue;

    /* TO_UPSTREAM(BoundsSafety) ON */
    tryEmitBoundsSafetyRemark(KV.second, ORE);
    /* TO_UPSTREAM(BoundsSafety) OFF */
    tryEmitAutoInitRemark(KV.second, ORE, TLI);
  }
}

PreservedAnalyses AnnotationRemarksPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  runImpl(F, TLI);
  return PreservedAnalyses::all();
}
