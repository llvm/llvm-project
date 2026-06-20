//===--- LTOAnalyzer.cpp - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/LTO/LTOAnalyzer.h"
#include "Analysis/LTO/LTOUtils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

Expected<std::unique_ptr<CapabilityResult>>
LTOAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string InputPath = resolveLTOInputPath(Context);
  if (InputPath.empty())
    return makeUnavailableResult(CapID, UnitID,
                                 "no bitcode file found for LTO summary analysis");

  Expected<std::unique_ptr<ModuleSummaryIndex>> IndexOrErr =
      getModuleSummaryIndexForFile(InputPath);
  if (!IndexOrErr)
    return IndexOrErr.takeError();
  const ModuleSummaryIndex &Index = **IndexOrErr;

  // Count modules, functions, globals, aliases.
  int64_t FunctionCount = 0, GlobalCount = 0, AliasCount = 0;
  int64_t EligibleToImport = 0, TotalInstructions = 0;

  for (const auto &Entry : Index) {
    for (const auto &Summary : Entry.second.getSummaryList()) {
      if (isa<FunctionSummary>(Summary.get())) {
        FunctionCount++;
        const auto *FS = cast<FunctionSummary>(Summary.get());
        TotalInstructions += FS->instCount();
        if (!FS->notEligibleToImport())
          EligibleToImport++;
      } else if (isa<GlobalVarSummary>(Summary.get())) {
        GlobalCount++;
      } else if (isa<AliasSummary>(Summary.get())) {
        AliasCount++;
      }
    }
  }

  int64_t ModuleCount = static_cast<int64_t>(Index.modulePaths().size());

  // Collect module names.
  json::Array Modules;
  for (const auto &M : Index.modulePaths())
    Modules.push_back(M.getKey());

  return makeJSONResult(CapID, UnitID, json::Object{
      {"bitcode_path", InputPath},
      {"module_count", ModuleCount},
      {"function_count", FunctionCount},
      {"global_count", GlobalCount},
      {"alias_count", AliasCount},
      {"functions_eligible_to_import", EligibleToImport},
      {"total_instructions", TotalInstructions},
      {"modules", std::move(Modules)},
  });
}

} // namespace llvm::advisor
