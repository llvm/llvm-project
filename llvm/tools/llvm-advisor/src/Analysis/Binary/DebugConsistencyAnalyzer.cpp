//===--- DebugConsistencyAnalyzer.cpp - LLVM Advisor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/DebugConsistencyAnalyzer.h"
#include "Analysis/Binary/BinaryAnalysisUtils.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
DebugConsistencyAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withDWARFContext(Context, CapID, UnitID, [&](DWARFContext &DW) {
    int64_t UnitsMissingRoot = 0;
    int64_t TotalUnits = 0;
    for (const std::unique_ptr<DWARFUnit> &CU : DW.compile_units()) {
      ++TotalUnits;
      if (!CU || !CU->getUnitDIE().isValid())
        ++UnitsMissingRoot;
    }
    return makeJSONResult(CapID, UnitID, json::Object{
        {"total_units", TotalUnits},
        {"units_missing_root_die", UnitsMissingRoot},
        {"consistent", UnitsMissingRoot == 0}});
  });
}
