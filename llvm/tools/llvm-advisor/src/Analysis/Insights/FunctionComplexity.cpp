//===--- FunctionComplexity.cpp - LLVM Advisor ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/FunctionComplexity.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<InsightOutput>
FunctionComplexityInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  const json::Array *Fns = D.getArray("functions");
  if (!Fns || Fns->empty())
    return noDataError();

  // Collect (instructions, name) pairs.
  struct FnEntry {
    int64_t Instructions;
    int64_t BasicBlocks;
    int64_t ArgCount;
    std::string Name;
  };
  SmallVector<FnEntry, 64> Entries;
  Entries.reserve(Fns->size());

  int64_t TotalInstructions = 0;
  for (const json::Value &V : *Fns) {
    const json::Object *F = V.getAsObject();
    if (!F)
      continue;
    FnEntry E;
    E.Name = F->getString("name").value_or("?").str();
    E.Instructions = getInt(*F, "instructions");
    E.BasicBlocks = getInt(*F, "basic_blocks");
    E.ArgCount = getInt(*F, "arg_count");
    TotalInstructions += E.Instructions;
    Entries.push_back(std::move(E));
  }

  // Sort descending by instruction count.
  llvm::sort(Entries, [](const FnEntry &A, const FnEntry &B) {
    return A.Instructions > B.Instructions;
  });

  // Compute 90th-percentile threshold. Entries are sorted descending, so the
  // 90th percentile is at the 10% mark from the top.
  double P90Threshold = 0.0;
  if (!Entries.empty()) {
    size_t Idx = static_cast<size_t>((Entries.size() - 1) * 0.1);
    P90Threshold = static_cast<double>(Entries[Idx].Instructions);
  }

  // Build top-N output.
  json::Array TopFunctions;
  int Count = 0;
  for (const FnEntry &E : Entries) {
    if (Count++ >= DefaultTopN)
      break;
    double Pct = TotalInstructions > 0
                     ? 100.0 * E.Instructions / TotalInstructions
                     : 0.0;
    TopFunctions.push_back(json::Object{
        {"name", E.Name},
        {"instructions", E.Instructions},
        {"basic_blocks", E.BasicBlocks},
        {"arg_count", E.ArgCount},
        {"pct_of_total", roundToOneDecimal(Pct)},
    });
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"total_functions", static_cast<int64_t>(Entries.size())},
      {"total_instructions", TotalInstructions},
      {"p90_instruction_threshold", static_cast<int64_t>(P90Threshold)},
      {"top_by_instructions", std::move(TopFunctions)},
  };
  return Out;
}
