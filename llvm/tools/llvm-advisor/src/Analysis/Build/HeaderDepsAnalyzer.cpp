//===--- HeaderDepsAnalyzer.cpp - LLVM Advisor --------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Build/HeaderDepsAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

// Walk the SourceManager to collect all included files with their include
// depth relative to the main file.
static void collectHeaders(const clang::SourceManager &SM,
                           StringMap<int64_t> &IncludeCounts,
                           StringMap<int64_t> &MaxDepths) {
  struct DepthCalculator {
    const clang::SourceManager &SM;
    DenseMap<clang::FileID, int> Cache;

    explicit DepthCalculator(const clang::SourceManager &SM) : SM(SM) {}

    int getDepth(clang::FileID FID) {
      auto It = Cache.find(FID);
      if (It != Cache.end())
        return It->second;

      clang::SourceLocation IncludeLoc = SM.getIncludeLoc(FID);
      int Depth = 0;
      if (IncludeLoc.isValid()) {
        clang::FileID ParentFID = SM.getFileID(IncludeLoc);
        if (ParentFID.isValid() && ParentFID != FID)
          Depth = getDepth(ParentFID) + 1;
      }
      Cache[FID] = Depth;
      return Depth;
    }
  };

  DepthCalculator Calc(SM);

  forEachIncludedFile(SM, [&](const clang::FileEntry &FE, clang::FileID FID) {
    StringRef Name = FE.tryGetRealPathName();
    if (Name.empty())
      return;
    int Depth = Calc.getDepth(FID);
    IncludeCounts[Name]++;
    int64_t &MaxD = MaxDepths[Name];
    if (Depth > MaxD)
      MaxD = Depth;
  });
}

Expected<std::unique_ptr<CapabilityResult>>
HeaderDepsAnalyzer::run(const CapabilityContext &Context) {
  Expected<std::unique_ptr<clang::ASTUnit>> ASTOrErr = buildASTUnit(Context);
  if (!ASTOrErr)
    return ASTOrErr.takeError();

  clang::ASTUnit &AST = **ASTOrErr;
  const clang::SourceManager &SM = AST.getSourceManager();

  StringMap<int64_t> IncludeCounts;
  StringMap<int64_t> MaxDepths;
  collectHeaders(SM, IncludeCounts, MaxDepths);

  int64_t MaxDepth = 0;
  for (auto &KV : MaxDepths)
    if (KV.second > MaxDepth)
      MaxDepth = KV.second;

  // Build headers array.
  struct HeaderEntry {
    std::string Path;
    int64_t IncludeCount;
    int64_t Depth;
  };
  SmallVector<HeaderEntry, 128> Entries;
  for (auto &KV : IncludeCounts)
    Entries.push_back(
        {KV.getKey().str(), KV.second, MaxDepths.lookup(KV.getKey())});

  // Sort by include count descending for stable output.
  llvm::sort(Entries, [](const HeaderEntry &A, const HeaderEntry &B) {
    return A.IncludeCount > B.IncludeCount;
  });

  json::Array Headers;
  for (const HeaderEntry &E : Entries)
    Headers.push_back(json::Object{
        {"path", E.Path},
        {"include_count", E.IncludeCount},
        {"depth", E.Depth},
    });

  json::Value Result = json::Object{
      {"total_headers", static_cast<int64_t>(Entries.size())},
      {"max_depth", MaxDepth},
      {"headers", std::move(Headers)},
  };
  return std::make_unique<JSONCapabilityResult>(std::move(Result));
}

} // namespace llvm::advisor
