//===--- InliningTreeAnalyzer.cpp - LLVM Advisor -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/InliningTreeAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
InliningTreeAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    json::Array Edges;
    int64_t CandidateCount = 0;
    int64_t ViableCount = 0;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;
          Function *Callee = CB->getCalledFunction();
          if (!Callee || Callee->isDeclaration())
            continue;

          bool IsCandidate =
              Callee->hasFnAttribute(Attribute::AlwaysInline) ||
              Callee->hasLocalLinkage();
          bool IsViable = false;
          if (IsCandidate) {
            InlineFunctionInfo IFI;
            InlineResult R = CanInlineCallSite(*CB, IFI);
            IsViable = R.isSuccess();
          }
          if (IsCandidate)
            ++CandidateCount;
          if (IsViable)
            ++ViableCount;

          Edges.push_back(json::Object{
              {"caller", F.getName().str()},
              {"callee", Callee->getName().str()},
              {"inline_candidate", IsCandidate},
              {"inline_viable", IsViable},
          });
        }
      }
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"candidate_count", CandidateCount},
        {"viable_count", ViableCount},
        {"edges", std::move(Edges)}});
  });
}
