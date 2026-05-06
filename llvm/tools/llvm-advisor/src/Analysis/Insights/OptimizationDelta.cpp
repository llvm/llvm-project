//===--- OptimizationDelta.cpp - LLVM Advisor ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/OptimizationDelta.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<InsightOutput>
OptimizationDeltaInsight::analyze(const InsightInput &Input) const {
  const json::Object &Primary = *Input.PrimaryData;
  const json::Object &Baseline = *Input.BaselineData;

  int64_t PrimaryTotal = getInt(Primary, "count");
  int64_t BaselineTotal = getInt(Baseline, "count");
  int64_t TotalDelta = PrimaryTotal - BaselineTotal;

  // Diff by_pass: collect all pass names from both sides.
  StringMap<int64_t> PrimaryPasses, BaselinePasses;
  if (const json::Object *BP = Primary.getObject("by_pass"))
    for (auto &KV : *BP)
      if (auto N = KV.getSecond().getAsInteger())
        PrimaryPasses[KV.getFirst()] = *N;
  if (const json::Object *BP = Baseline.getObject("by_pass"))
    for (auto &KV : *BP)
      if (auto N = KV.getSecond().getAsInteger())
        BaselinePasses[KV.getFirst()] = *N;

  struct DeltaEntry {
    std::string Pass;
    int64_t Primary;
    int64_t Baseline;
    int64_t Delta;
  };
  SmallVector<DeltaEntry, 32> Deltas;

  // Union of all pass names.
  StringSet<> Seen;
  for (auto &KV : PrimaryPasses)
    Seen.insert(KV.getKey());
  for (auto &KV : BaselinePasses)
    Seen.insert(KV.getKey());

  for (auto &S : Seen) {
    int64_t P = PrimaryPasses.lookup(S.getKey());
    int64_t B = BaselinePasses.lookup(S.getKey());
    Deltas.push_back({S.getKey().str(), P, B, P - B});
  }

  // Sort by absolute delta descending.
  llvm::sort(Deltas, [](const DeltaEntry &A, const DeltaEntry &B) {
    return std::abs(A.Delta) > std::abs(B.Delta);
  });

  json::Array PassDeltas;
  for (const DeltaEntry &E : Deltas) {
    if (E.Delta == 0)
      continue;
    PassDeltas.push_back(json::Object{
        {"pass", E.Pass},
        {"primary", E.Primary},
        {"baseline", E.Baseline},
        {"delta", E.Delta},
    });
  }

  // Diff by_type (Passed/Missed/Analysis).
  json::Object TypeDeltas;
  StringSet<> TypeSeen;
  StringMap<int64_t> PrimaryTypes, BaselineTypes;
  if (const json::Object *BT = Primary.getObject("by_type"))
    for (auto &KV : *BT)
      if (auto N = KV.getSecond().getAsInteger())
        PrimaryTypes[KV.getFirst()] = *N;
  if (const json::Object *BT = Baseline.getObject("by_type"))
    for (auto &KV : *BT)
      if (auto N = KV.getSecond().getAsInteger())
        BaselineTypes[KV.getFirst()] = *N;
  for (auto &KV : PrimaryTypes)
    TypeSeen.insert(KV.getKey());
  for (auto &KV : BaselineTypes)
    TypeSeen.insert(KV.getKey());
  for (auto &S : TypeSeen) {
    int64_t P = PrimaryTypes.lookup(S.getKey());
    int64_t B = BaselineTypes.lookup(S.getKey());
    TypeDeltas[S.getKey()] =
        json::Object{{"primary", P}, {"baseline", B}, {"delta", P - B}};
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"total_delta", TotalDelta},
      {"primary_total", PrimaryTotal},
      {"baseline_total", BaselineTotal},
      {"by_type_delta", std::move(TypeDeltas)},
      {"top_changed_passes", std::move(PassDeltas)},
  };
  return Out;
}
