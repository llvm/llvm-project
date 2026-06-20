//===--- DebugDetailAnalyzer.cpp - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/DebugDetailAnalyzer.h"
#include "Analysis/Binary/BinaryAnalysisUtils.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
DebugDetailAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withDWARFContext(Context, CapID, UnitID, [&](DWARFContext &DW) {
    int64_t CUCount = 0;
    for (const std::unique_ptr<DWARFUnit> &CU : DW.compile_units()) {
      (void)CU;
      ++CUCount;
    }
    return makeJSONResult(CapID, UnitID, json::Object{
        {"compile_units", CUCount},
        {"max_dwo_version", static_cast<int64_t>(DW.getMaxDWOVersion())}});
  });
}
