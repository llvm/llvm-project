//===--- SectionSizes.cpp - LLVM Advisor ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/SectionSizes.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

StringRef classifySection(StringRef Name) {
  if (Name == ".text" || Name.starts_with(".text."))
    return "text";
  if (Name == ".rodata" || Name.starts_with(".rodata.") ||
      Name == "__TEXT,__text")
    return "text";
  if (Name == ".data" || Name.starts_with(".data.") || Name == ".bss" ||
      Name.starts_with(".bss."))
    return "data";
  if (Name.starts_with(".debug_") || Name.starts_with("__debug_") ||
      Name == ".dwarf")
    return "debug";
  if (Name == ".eh_frame" || Name == ".gcc_except_table" ||
      Name.starts_with(".rela") || Name.starts_with(".rel."))
    return "metadata";
  return "other";
}

} // namespace

Expected<InsightOutput>
SectionSizesInsight::analyze(const InsightInput &Input) const {
  const json::Object &D = *Input.PrimaryData;

  const json::Array *Sections = D.getArray("sections");
  if (!Sections || Sections->empty())
    return noDataError();

  struct SectionEntry {
    std::string Name;
    int64_t Size;
    std::string Category;
  };
  SmallVector<SectionEntry, 32> Entries;
  StringMap<int64_t> CategoryTotals;
  int64_t TotalSize = 0;

  for (const json::Value &V : *Sections) {
    const json::Object *S = V.getAsObject();
    if (!S)
      continue;
    SectionEntry E;
    E.Name = S->getString("name").value_or("?").str();
    E.Size = getInt(*S, "size");
    E.Category = classifySection(E.Name).str();
    CategoryTotals[E.Category] += E.Size;
    TotalSize += E.Size;
    Entries.push_back(std::move(E));
  }

  llvm::sort(Entries, [](const SectionEntry &A, const SectionEntry &B) {
    return A.Size > B.Size;
  });

  json::Array SectionArray;
  for (const SectionEntry &E : Entries) {
    double Pct = TotalSize > 0 ? 100.0 * E.Size / TotalSize : 0.0;
    SectionArray.push_back(json::Object{
        {"name", E.Name},
        {"size", E.Size},
        {"category", E.Category},
        {"pct_of_total", roundToOneDecimal(Pct)},
    });
  }

  json::Object CategoryBreakdown;
  for (auto &KV : CategoryTotals) {
    double Pct = TotalSize > 0 ? 100.0 * KV.second / TotalSize : 0.0;
    CategoryBreakdown[KV.getKey()] = json::Object{
        {"size", KV.second},
        {"pct_of_total", roundToOneDecimal(Pct)},
    };
  }

  SmallVector<std::string, 4> Warnings;
  int64_t DebugSize = CategoryTotals.lookup("debug");
  int64_t TextSize = CategoryTotals.lookup("text");
  if (DebugSize > TextSize)
    Warnings.push_back(
        "Debug sections larger than text — consider stripping debug info for "
        "release builds");
  if (TextSize == 0)
    Warnings.push_back(
        "No text section found — binary may lack executable code");

  InsightOutput Out;
  Out.Kind = getKind();
  Out.Name = getName().str();
  Out.Warnings = std::move(Warnings);
  Out.Data = json::Object{
      {"total_size", TotalSize},
      {"format", D.getString("format").value_or("unknown")},
      {"category_breakdown", std::move(CategoryBreakdown)},
      {"sections", std::move(SectionArray)},
  };
  return Out;
}
