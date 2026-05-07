//===------------------- CFGAnalyzer.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/CFGAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Instruction.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
CFGAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    json::Array Functions;
    int64_t TotalEdges = 0;
    int64_t TotalBlocks = 0;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      int64_t Blocks = 0;
      int64_t Edges = 0;
      for (BasicBlock &B : F) {
        ++Blocks;
        if (const Instruction *Term = B.getTerminator())
          Edges += static_cast<int64_t>(Term->getNumSuccessors());
      }
      TotalBlocks += Blocks;
      TotalEdges += Edges;
      Functions.push_back(json::Object{{"name", F.getName()},
                                       {"basic_blocks", Blocks},
                                       {"edges", Edges}});
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"functions", std::move(Functions)},
        {"total_basic_blocks", TotalBlocks},
        {"total_edges", TotalEdges}});
  });
}
