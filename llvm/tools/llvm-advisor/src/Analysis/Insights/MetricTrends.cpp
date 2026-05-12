//===--- MetricTrends.cpp - LLVM Advisor ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/MetricTrends.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

// Size-class thresholds based on instruction count for the whole module.
StringRef classifyModuleSize(int64_t Instructions) {
  if (Instructions < 1000)
    return "small";
  if (Instructions < 10000)
    return "medium";
  if (Instructions < 100000)
    return "large";
  return "very_large";
}

// Density heuristic: instructions per function.
StringRef classifyDensity(double InstrPerFn) {
  if (InstrPerFn < 5.0)
    return "trivial";
  if (InstrPerFn < 20.0)
    return "light";
  if (InstrPerFn < 60.0)
    return "moderate";
  if (InstrPerFn < 150.0)
    return "heavy";
  return "very_heavy";
}

} // namespace

Expected<InsightOutput>
MetricTrendsInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  int64_t Functions = getInt(D, "functions");
  if (!Functions) Functions = getInt(D, "function_count");
  int64_t Instructions = getInt(D, "instructions");
  if (!Instructions) Instructions = getInt(D, "instruction_count");
  int64_t Globals = getInt(D, "globals");
  if (!Globals) Globals = getInt(D, "global_count");

  if (Functions == 0 && Instructions == 0)
    return noDataError();

  double InstrPerFn =
      Functions > 0 ? static_cast<double>(Instructions) / Functions : 0.0;

  StringRef SizeClass = classifyModuleSize(Instructions);
  StringRef DensityClass = classifyDensity(InstrPerFn);

  // Derive interpretation strings.
  SmallVector<std::string, 4> Interpretations;
  if (Instructions > 100000)
    Interpretations.push_back(
        "Very large module — consider splitting into sub-libraries for faster "
        "incremental builds");
  if (InstrPerFn > 150.0)
    Interpretations.push_back(
        "High instruction density per function — potential candidates for "
        "inlining or splitting");
  if (Globals > Functions * 2)
    Interpretations.push_back(
        "High global-to-function ratio — check for excessive static data or "
        "constant arrays");
  if (Interpretations.empty())
    Interpretations.push_back("Module metrics within typical range");

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"functions", Functions},
      {"instructions", Instructions},
      {"globals", Globals},
      {"instructions_per_function", roundToOneDecimal(InstrPerFn)},
      {"size_class", SizeClass},
      {"density_class", DensityClass},
      {"interpretations", toJSONArray(Interpretations)},
  };
  return Out;
}
