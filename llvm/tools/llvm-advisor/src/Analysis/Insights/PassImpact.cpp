//===--- PassImpact.cpp - LLVM Advisor -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/PassImpact.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<InsightOutput>
PassImpactInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  if (getInt(D, "count") == 0 && !D.getObject("by_pass"))
    return noDataError();

  int64_t TotalRemarks = getInt(D, "count");

  // Collect pass → count from by_pass.
  struct PassEntry {
    std::string Pass;
    int64_t Count;
  };
  SmallVector<PassEntry, 32> Passes;

  if (const json::Object *ByPass = D.getObject("by_pass")) {
    for (auto &KV : *ByPass) {
      int64_t Count = 0;
      if (auto N = KV.getSecond().getAsInteger())
        Count = *N;
      Passes.push_back(PassEntry{KV.getFirst().str(), Count});
    }
  }

  llvm::sort(Passes, [](const PassEntry &A, const PassEntry &B) {
    return A.Count > B.Count;
  });

  // Collect type breakdown.
  json::Object TypeBreakdown;
  if (const json::Object *ByType = D.getObject("by_type")) {
    for (auto &KV : *ByType) {
      int64_t Count = 0;
      if (auto N = KV.getSecond().getAsInteger())
        Count = *N;
      TypeBreakdown[KV.getFirst()] = Count;
    }
  }

  int64_t Passed = 0, Missed = 0;
  if (auto V = TypeBreakdown.getInteger("Passed"))
    Passed = *V;
  if (auto V = TypeBreakdown.getInteger("Missed"))
    Missed = *V;
  double HitRate =
      (Passed + Missed) > 0 ? 100.0 * Passed / (Passed + Missed) : 0.0;

  json::Array TopPasses;
  int Count = 0;
  for (const PassEntry &E : Passes) {
    if (Count++ >= DefaultTopN)
      break;
    double Pct = TotalRemarks > 0 ? 100.0 * E.Count / TotalRemarks : 0.0;
    TopPasses.push_back(json::Object{
        {"pass", E.Pass},
        {"count", E.Count},
        {"pct_of_total", roundToOneDecimal(Pct)},
    });
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"total_remarks", TotalRemarks},
      {"optimization_hit_rate_pct", roundToOneDecimal(HitRate)},
      {"by_type", std::move(TypeBreakdown)},
      {"top_passes_by_remarks", std::move(TopPasses)},
  };
  return Out;
}
