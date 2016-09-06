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

// Create the combined module index/summary from multiple
// per-module instances.
void ModuleSummaryIndex::mergeFrom(std::unique_ptr<ModuleSummaryIndex> Other,
                                   uint64_t NextModuleId) {

  StringRef ModPath;
  for (auto &OtherGlobalValSummaryLists : *Other) {
    GlobalValue::GUID ValueGUID = OtherGlobalValSummaryLists.first;
    GlobalValueSummaryList &List = OtherGlobalValSummaryLists.second;

    // Assert that the value summary list only has one entry, since we shouldn't
    // have duplicate names within a single per-module index.
    assert(List.size() == 1);
    std::unique_ptr<GlobalValueSummary> Summary = std::move(List.front());

    // Add the module path string ref for this module if we haven't already
    // saved a reference to it.
    if (ModPath.empty()) {
      auto Path = Summary->modulePath();
      ModPath = addModulePath(Path, NextModuleId, Other->getModuleHash(Path))
                    ->first();
    } else
      assert(ModPath == Summary->modulePath() &&
             "Each module in the combined map should have a unique ID");

    // Note the module path string ref was copied above and is still owned by
    // the original per-module index. Reset it to the new module path
    // string reference owned by the combined index.
    Summary->setModulePath(ModPath);

    // Add new value summary to existing list. There may be duplicates when
    // combining GlobalValueMap entries, due to COMDAT values. Any local
    // values were given unique global IDs.
    addGlobalValueSummary(ValueGUID, std::move(Summary));
  }
}

void ModuleSummaryIndex::removeEmptySummaryEntries() {
  for (auto MI = begin(), MIE = end(); MI != MIE;) {
    // Only expect this to be called on a per-module index, which has a single
    // entry per value entry list.
    assert(MI->second.size() == 1);
    if (!MI->second[0])
      MI = GlobalValueMap.erase(MI);
    else
      ++MI;
  }
}

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
