//===--- LTOFunctionStatsAnalyzer.cpp - LLVM Advisor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/LTO/LTOFunctionStatsAnalyzer.h"
#include "Analysis/LTO/LTOUtils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

static constexpr unsigned MaxFunctionsInOutput = 500;

static StringRef linkageName(GlobalValue::LinkageTypes L) {
  switch (L) {
  case GlobalValue::ExternalLinkage:
    return "external";
  case GlobalValue::InternalLinkage:
    return "internal";
  case GlobalValue::PrivateLinkage:
    return "private";
  case GlobalValue::WeakAnyLinkage:
    return "weak";
  case GlobalValue::WeakODRLinkage:
    return "weak_odr";
  case GlobalValue::LinkOnceAnyLinkage:
    return "linkonce";
  case GlobalValue::LinkOnceODRLinkage:
    return "linkonce_odr";
  case GlobalValue::AvailableExternallyLinkage:
    return "available_externally";
  case GlobalValue::CommonLinkage:
    return "common";
  case GlobalValue::AppendingLinkage:
    return "appending";
  case GlobalValue::ExternalWeakLinkage:
    return "extern_weak";
  default:
    return "unknown";
  }
}

Expected<std::unique_ptr<CapabilityResult>>
LTOFunctionStatsAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string InputPath = resolveLTOInputPath(Context);
  if (InputPath.empty())
    return makeUnavailableResult(CapID, UnitID,
                                 "no bitcode file found for LTO function stats");

  Expected<std::unique_ptr<ModuleSummaryIndex>> IndexOrErr =
      getModuleSummaryIndexForFile(InputPath);
  if (!IndexOrErr)
    return IndexOrErr.takeError();
  const ModuleSummaryIndex &Index = **IndexOrErr;

  struct FnEntry {
    std::string Name;
    unsigned InstCount;
    bool EligibleToImport;
    StringRef Linkage;
  };
  SmallVector<FnEntry, 256> Functions;
  int64_t TotalInstructions = 0;

  for (const auto &Entry : Index) {
    for (const auto &SPtr : Entry.second.getSummaryList()) {
      const auto *FS = dyn_cast<FunctionSummary>(SPtr.get());
      if (!FS)
        continue;
      FnEntry E;
      // ValueInfo name is only available if we have GVs; use GUID as fallback.
      ValueInfo VI = Index.getValueInfo(Entry.first);
      if (VI && VI.name().size())
        E.Name = VI.name().str();
      else {
        raw_string_ostream OS(E.Name);
        OS << "guid:" << Entry.first;
      }
      E.InstCount = FS->instCount();
      E.EligibleToImport = !FS->notEligibleToImport();
      E.Linkage = linkageName(
          static_cast<GlobalValue::LinkageTypes>(FS->flags().Linkage));
      TotalInstructions += E.InstCount;
      Functions.push_back(std::move(E));
    }
  }

  // Sort by instruction count descending.
  llvm::sort(Functions, [](const FnEntry &A, const FnEntry &B) {
    return A.InstCount > B.InstCount;
  });

  json::Array FnArray;
  unsigned Count = 0;
  for (const FnEntry &F : Functions) {
    if (Count++ >= MaxFunctionsInOutput)
      break;
    FnArray.push_back(json::Object{
        {"name", F.Name},
        {"instructions", static_cast<int64_t>(F.InstCount)},
        {"eligible_to_import", F.EligibleToImport},
        {"linkage", F.Linkage},
    });
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"bitcode_path", InputPath},
      {"total_functions", static_cast<int64_t>(Functions.size())},
      {"total_instructions", TotalInstructions},
      {"functions_shown", static_cast<int64_t>(FnArray.size())},
      {"functions", std::move(FnArray)},
  });
}

} // namespace llvm::advisor
