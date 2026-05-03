//===--- DebugSummaryAnalyzer.cpp - LLVM Advisor -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/DebugSummaryAnalyzer.h"
#include "Analysis/Binary/BinaryAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
DebugSummaryAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withDWARFContext(Context, CapID, UnitID, [&](DWARFContext &DW) {
    int64_t NumCUs = static_cast<int64_t>(DW.getNumCompileUnits());
    return makeJSONResult(CapID, UnitID, json::Object{
        {"compile_units", NumCUs},
        {"max_dwo_version", static_cast<int64_t>(DW.getMaxDWOVersion())},
        {"has_debug_info", NumCUs > 0}});
  });
}
