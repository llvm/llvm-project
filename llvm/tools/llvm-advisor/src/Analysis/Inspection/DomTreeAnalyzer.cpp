//===------------------- DomTreeAnalyzer.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/DomTreeAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
DomTreeAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    json::Array Functions;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      DominatorTree DT(F);
      int64_t NodeCount = 0;
      for (BasicBlock &BB : F) {
        if (DT.getNode(&BB))
          ++NodeCount;
      }
      Functions.push_back(json::Object{{"name", F.getName()},
                                       {"dom_tree_nodes", NodeCount}});
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"functions", std::move(Functions)}});
  });
}
