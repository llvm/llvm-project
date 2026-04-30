//===------------------- InliningTreeAnalyzer.cpp - LLVM Advisor ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/InliningTreeAnalyzer.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
InliningTreeAnalyzer::run(const CapabilityContext &Context) {
  if (Context.IRPath.empty()) {
    return std::make_unique<JSONCapabilityResult>(json::Object{
        {"capability", getCapabilityID()},
        {"unit_id", Context.Unit.ID},
        {"available", false},
        {"reason", "missing IR artifact"},
    });
  }

  LLVMContext LLVMCtx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(Context.IRPath, Err, LLVMCtx);
  if (!M)
    return createStringError(inconvertibleErrorCode(), "cannot parse IR: %s",
                             Context.IRPath.c_str());

  json::Array Edges;
  int64_t Candidates = 0;
  int64_t Inlined = 0;

  for (Function &F : *M) {
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
        ++Candidates;
        bool ShouldTryInline =
            Callee->hasFnAttribute(Attribute::AlwaysInline) ||
            Callee->hasLocalLinkage();
        bool Success = false;
        if (ShouldTryInline) {
          InlineFunctionInfo IFI;
          InlineResult R = InlineFunction(*CB, IFI);
          Success = R.isSuccess();
        }
        if (Success)
          ++Inlined;

        Edges.push_back(json::Object{
            {"caller", F.getName().str()},
            {"callee", Callee->getName().str()},
            {"inline_candidate", ShouldTryInline},
            {"inline_succeeded", Success},
        });
      }
    }
  }

  return std::make_unique<JSONCapabilityResult>(json::Object{
      {"capability", getCapabilityID()},
      {"unit_id", Context.Unit.ID},
      {"candidate_calls", Candidates},
      {"inlined_calls", Inlined},
      {"edges", std::move(Edges)},
  });
}
