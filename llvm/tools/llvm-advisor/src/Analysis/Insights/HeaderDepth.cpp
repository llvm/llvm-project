//===--- HeaderDepth.cpp - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/HeaderDepth.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

constexpr int DeepChainThreshold = 10;

struct HeaderEntry {
  std::string Path;
  int64_t IncludeCount;
  int64_t Depth;
};

json::Array BuildArray(ArrayRef<HeaderEntry> Sorted) {
  json::Array A;
  int Count = 0;
  for (const HeaderEntry &E : Sorted) {
    if (Count++ >= DefaultTopN)
      break;
    A.push_back(json::Object{
        {"path", E.Path},
        {"include_count", E.IncludeCount},
        {"depth", E.Depth},
    });
  }
  return A;
}

} // namespace

Expected<InsightOutput>
HeaderDepthInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  int64_t TotalHeaders = getInt(D, "total_headers");
  int64_t MaxDepth = getInt(D, "max_depth");

  const json::Array *Headers = D.getArray("headers");

  SmallVector<HeaderEntry, 64> Entries;

  if (Headers) {
    for (const json::Value &V : *Headers) {
      const json::Object *H = V.getAsObject();
      if (!H)
        continue;
      HeaderEntry E;
      E.Path = H->getString("path").value_or("?").str();
      E.IncludeCount = getInt(*H, "include_count");
      E.Depth = getInt(*H, "depth");
      Entries.push_back(std::move(E));
    }
  }

  SmallVector<HeaderEntry, 64> ByCount = Entries;
  llvm::sort(ByCount, [](const HeaderEntry &A, const HeaderEntry &B) {
    return A.IncludeCount > B.IncludeCount;
  });

  SmallVector<HeaderEntry, 64> ByDepth = Entries;
  llvm::sort(ByDepth, [](const HeaderEntry &A, const HeaderEntry &B) {
    return A.Depth > B.Depth;
  });

  SmallVector<std::string, 4> Warnings;
  if (MaxDepth >= DeepChainThreshold)
    Warnings.push_back(
        "Include chain depth >= 10 — deep nesting slows parsing and increases "
        "transitive dependency exposure");
  if (TotalHeaders > 500)
    Warnings.push_back(
        "Very high header count — consider precompiled headers or modules to "
        "reduce preprocessing overhead");
  if (!ByCount.empty() && ByCount[0].IncludeCount > 50)
    Warnings.push_back("One header included more than 50 times — ensure it has "
                       "include guards or #pragma once");

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Warnings = std::move(Warnings);
  Out.Data = json::Object{
      {"total_headers", TotalHeaders},
      {"max_depth", MaxDepth},
      {"most_included", BuildArray(ByCount)},
      {"deepest_chains", BuildArray(ByDepth)},
  };
  return Out;
}
