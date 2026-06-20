//===--- DebugInfo.cpp - LLVM Advisor ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/DebugInfo.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

StringRef assessCoverage(bool HasDebugInfo, int64_t CompileUnits) {
  if (!HasDebugInfo || CompileUnits == 0)
    return "none";
  if (CompileUnits < 5)
    return "minimal";
  if (CompileUnits < 20)
    return "partial";
  return "full";
}

} // namespace

Expected<InsightOutput>
DebugInfoInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  bool HasDebugInfo = D.getBoolean("has_debug_info").value_or(false);
  int64_t CompileUnits = getInt(D, "compile_units");
  int64_t MaxDwoVersion = getInt(D, "max_dwo_version");

  StringRef Coverage = assessCoverage(HasDebugInfo, CompileUnits);

  SmallVector<std::string, 4> Interpretations;
  if (!HasDebugInfo) {
    Interpretations.push_back(
        "No DWARF debug info present — symbolication and profiling tools will "
        "not be able to resolve function names");
  } else {
    if (MaxDwoVersion > 0)
      Interpretations.push_back(
          "Split DWARF (DWO) detected — ensure .dwo files are distributed "
          "alongside the binary for debuggers");
    if (CompileUnits > 100)
      Interpretations.push_back(
          "Large number of compile units — consider using -gsplit-dwarf to "
          "reduce linker memory pressure");
    if (Coverage == "minimal")
      Interpretations.push_back(
          "Very few compile units — debug info may be incomplete or stripped");
  }

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Data = json::Object{
      {"has_debug_info", HasDebugInfo},
      {"compile_units", CompileUnits},
      {"max_dwo_version", MaxDwoVersion},
      {"coverage", Coverage},
      {"interpretations", toJSONArray(Interpretations)},
  };
  return Out;
}
