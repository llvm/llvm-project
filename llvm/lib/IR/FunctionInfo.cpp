//===-- FunctionInfo.cpp - Function Info Index ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the function info index and summary classes for the
// IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/FunctionInfo.h"
#include "llvm/ADT/StringMap.h"
using namespace llvm;

// Create the combined function index/summary from multiple
// per-module instances.
void FunctionInfoIndex::mergeFrom(std::unique_ptr<FunctionInfoIndex> Other,
                                  uint64_t NextModuleId) {

  StringRef ModPath;
  for (auto &OtherFuncInfoLists : *Other) {
    std::string FuncName = OtherFuncInfoLists.getKey();
    FunctionInfoList &List = OtherFuncInfoLists.second;

    // Assert that the func info list only has one entry, since we shouldn't
    // have duplicate names within a single per-module index.
    assert(List.size() == 1);
    std::unique_ptr<FunctionInfo> Info = std::move(List.front());

    // Skip if there was no function summary section.
    if (!Info->functionSummary())
      continue;

    // Add the module path string ref for this module if we haven't already
    // saved a reference to it.
    if (ModPath.empty())
      ModPath =
          addModulePath(Info->functionSummary()->modulePath(), NextModuleId);
    else
      assert(ModPath == Info->functionSummary()->modulePath() &&
             "Each module in the combined map should have a unique ID");

    // Note the module path string ref was copied above and is still owned by
    // the original per-module index. Reset it to the new module path
    // string reference owned by the combined index.
    Info->functionSummary()->setModulePath(ModPath);

    // If it is a local function, rename it.
    if (GlobalValue::isLocalLinkage(
            Info->functionSummary()->getFunctionLinkage())) {
      // Any local functions are virtually renamed when being added to the
      // combined index map, to disambiguate from other functions with
      // the same name. The symbol table created for the combined index
      // file should contain the renamed symbols.
      FuncName =
          FunctionInfoIndex::getGlobalNameForLocal(FuncName, NextModuleId);
    }

    // Add new function info to existing list. There may be duplicates when
    // combining FunctionMap entries, due to COMDAT functions. Any local
    // functions were virtually renamed above.
    addFunctionInfo(FuncName, std::move(Info));
  }
}
