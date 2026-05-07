//===------------------- LoopInfoAnalyzer.cpp - LLVM Advisor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/LoopInfoAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
LoopInfoAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    json::Array Functions;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      DominatorTree DT(F);
      LoopInfo LI(DT);
      int64_t Count = 0;
      int64_t MaxDepth = 0;
      for (Loop *Top : LI) {
        SmallVector<Loop *, 16> Stack{Top};
        while (!Stack.empty()) {
          Loop *L = Stack.pop_back_val();
          ++Count;
          MaxDepth = std::max<int64_t>(MaxDepth, L->getLoopDepth());
          for (Loop *Sub : *L)
            Stack.push_back(Sub);
        }
      }
      Functions.push_back(json::Object{{"name", F.getName()},
                                       {"loops", Count},
                                       {"max_depth", MaxDepth}});
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"functions", std::move(Functions)}});
  });
}
