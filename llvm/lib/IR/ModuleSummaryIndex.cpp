//===-- ModuleSummaryIndex.cpp - Module Summary Index ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the module index and summary classes for the
// IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/ADT/StringMap.h"
using namespace llvm;

// Collect for the given module the list of function it defines
// (GUID -> Summary).
void ModuleSummaryIndex::collectDefinedFunctionsForModule(
    StringRef ModulePath, GVSummaryMapTy &GVSummaryMap) const {
  for (auto &GlobalList : *this) {
    auto GUID = GlobalList.first;
    for (auto &GlobSummary : GlobalList.second) {
      auto *Summary = dyn_cast_or_null<FunctionSummary>(GlobSummary.get());
      if (!Summary)
        // Ignore global variable, focus on functions
        continue;
      // Ignore summaries from other modules.
      if (Summary->modulePath() != ModulePath)
        continue;
      GVSummaryMap[GUID] = Summary;
    }
  }
}

// Collect for each module the list of function it defines (GUID -> Summary).
void ModuleSummaryIndex::collectDefinedGVSummariesPerModule(
    StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries) const {
  for (auto &GlobalList : *this) {
    auto GUID = GlobalList.first;
    for (auto &Summary : GlobalList.second) {
      ModuleToDefinedGVSummaries[Summary->modulePath()][GUID] = Summary.get();
    }
  }
}

GlobalValueSummary *
ModuleSummaryIndex::getGlobalValueSummary(uint64_t ValueGUID,
                                          bool PerModuleIndex) const {
  auto SummaryList = findGlobalValueSummaryList(ValueGUID);
  assert(SummaryList != end() && "GlobalValue not found in index");
  assert((!PerModuleIndex || SummaryList->second.size() == 1) &&
         "Expected a single entry per global value in per-module index");
  auto &Summary = SummaryList->second[0];
  return Summary.get();
}
