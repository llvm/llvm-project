//===--- LoopNesting.cpp - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/LoopNesting.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

constexpr int DeepNestingThreshold = 3;

} // namespace

Expected<InsightOutput>
LoopNestingInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  const json::Array *Fns = D.getArray("functions");
  if (!Fns || Fns->empty())
    return noDataError();

  struct FnEntry {
    std::string Name;
    int64_t Loops;
    int64_t MaxDepth;
  };
  SmallVector<FnEntry, 64> Entries;
  int64_t TotalLoops = 0;
  int64_t GlobalMaxDepth = 0;
  int64_t DeeplyNestedFunctions = 0;

  for (const json::Value &V : *Fns) {
    const json::Object *F = V.getAsObject();
    if (!F)
      continue;
    FnEntry E;
    E.Name = F->getString("name").value_or("?").str();
    E.Loops = getInt(*F, "loops");
    E.MaxDepth = getInt(*F, "max_depth");
    TotalLoops += E.Loops;
    if (E.MaxDepth > GlobalMaxDepth)
      GlobalMaxDepth = E.MaxDepth;
    if (E.MaxDepth >= DeepNestingThreshold)
      DeeplyNestedFunctions++;
    Entries.push_back(std::move(E));
  }

  // Sort by max_depth desc, then loops desc.
  llvm::sort(Entries, [](const FnEntry &A, const FnEntry &B) {
    if (A.MaxDepth != B.MaxDepth)
      return A.MaxDepth > B.MaxDepth;
    return A.Loops > B.Loops;
  });

  json::Array TopFunctions;
  int Count = 0;
  for (const FnEntry &E : Entries) {
    if (E.Loops == 0)
      continue;
    if (Count++ >= DefaultTopN)
      break;
    TopFunctions.push_back(json::Object{
        {"name", E.Name},
        {"loops", E.Loops},
        {"max_depth", E.MaxDepth},
        {"deeply_nested", E.MaxDepth >= DeepNestingThreshold},
    });
  }

  SmallVector<std::string, 4> Warnings;
  if (GlobalMaxDepth >= 5)
    Warnings.push_back("Nesting depth >= 5 detected — consider loop tiling or "
                       "restructuring for cache locality");
  if (DeeplyNestedFunctions > 10)
    Warnings.push_back(
        "Many functions with deep loop nesting — review vectorization reports");

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Warnings = std::move(Warnings);
  Out.Data = json::Object{
      {"total_loops", TotalLoops},
      {"global_max_depth", GlobalMaxDepth},
      {"deeply_nested_functions", DeeplyNestedFunctions},
      {"deep_nesting_threshold", static_cast<int64_t>(DeepNestingThreshold)},
      {"top_by_nesting", std::move(TopFunctions)},
  };
  return Out;
}
