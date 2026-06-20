//===------------------- CallGraphAnalyzer.cpp - LLVM Advisor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/CallGraphAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/CallGraph.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
CallGraphAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    CallGraph CG(M);
    int64_t EdgeCount = 0;

    DenseMap<const Function *, int64_t> OutMap, InMap;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      const CallGraphNode *N = CG[&F];
      int64_t Out = 0;
      if (N) {
        for (const auto &Entry : *N) {
          if (Entry.second && Entry.second->getFunction()) {
            ++Out;
            ++EdgeCount;
            InMap[Entry.second->getFunction()]++;
          }
        }
      }
      OutMap[&F] = Out;
    }

    json::Array Functions;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      Functions.push_back(json::Object{
          {"name", F.getName()},
          {"outgoing_calls", OutMap.lookup(&F)},
          {"incoming_calls", InMap.lookup(&F)}});
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"edge_count", EdgeCount},
        {"functions", std::move(Functions)}});
  });
}
