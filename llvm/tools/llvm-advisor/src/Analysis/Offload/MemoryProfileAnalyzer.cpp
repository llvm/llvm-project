//===--- MemoryProfileAnalyzer.cpp - LLVM Advisor ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/MemoryProfileAnalyzer.h"
#include "Analysis/Utils/CSVUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

static std::string findMemProfileCSV(const CapabilityContext &Context) {
  if (Context.WorkingDirectory.empty())
    return {};
  std::error_code EC;
  for (sys::fs::directory_iterator It(Context.WorkingDirectory, EC), End;
       !EC && It != End; It.increment(EC)) {
    StringRef P = It->path();
    StringRef Stem = sys::path::stem(P);
    if (P.ends_with(".csv") && (Stem.contains_insensitive("mem") ||
                                Stem.contains_insensitive("profile") ||
                                Stem.contains_insensitive("transfer")))
      return P.str();
  }
  return {};
}

Expected<std::unique_ptr<CapabilityResult>>
MemoryProfileAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string Path = findMemProfileCSV(Context);
  if (Path.empty())
    return makeUnavailableResult(CapID, UnitID, "no memory profile CSV found");

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(), "cannot read memory profile: %s",
                             Path.c_str());

  SmallVector<StringRef, 256> Lines;
  (*MB)->getBuffer().split(Lines, '\n');
  if (Lines.empty())
    return makeUnavailableResult(CapID, UnitID, "empty CSV file");

  SmallVector<StringRef, 16> Headers = splitCSVRow(Lines[0]);
  int SizeCol = findCol(Headers, "SizeBytes");
  if (SizeCol < 0)
    SizeCol = findCol(Headers, "Size");
  if (SizeCol < 0)
    return makeUnavailableResult(CapID, UnitID,
                                 "CSV missing SizeBytes/Size column");

  // Histogram buckets: 0-1K, 1K-64K, 64K-1M, 1M-64M, 64M+
  int64_t Buckets[5] = {0, 0, 0, 0, 0};
  int64_t TotalBytes = 0, TotalRows = 0;

  for (size_t I = 1; I < Lines.size(); ++I) {
    StringRef Row = Lines[I].trim();
    if (Row.empty())
      continue;
    SmallVector<StringRef, 16> F = splitCSVRow(Row);
    if (SizeCol >= static_cast<int>(F.size()))
      continue;
    int64_t Sz = 0;
    if (F[SizeCol].trim('"').getAsInteger(10, Sz))
      continue;
    if (Sz <= 1024)
      ++Buckets[0];
    else if (Sz <= 64 * 1024)
      ++Buckets[1];
    else if (Sz <= 1024 * 1024)
      ++Buckets[2];
    else if (Sz <= 64 * 1024 * 1024)
      ++Buckets[3];
    else
      ++Buckets[4];
    TotalBytes += Sz;
    ++TotalRows;
  }

  json::Array Histogram;
  Histogram.push_back(json::Object{{"range", "0-1K"}, {"count", Buckets[0]}});
  Histogram.push_back(json::Object{{"range", "1K-64K"}, {"count", Buckets[1]}});
  Histogram.push_back(json::Object{{"range", "64K-1M"}, {"count", Buckets[2]}});
  Histogram.push_back(json::Object{{"range", "1M-64M"}, {"count", Buckets[3]}});
  Histogram.push_back(json::Object{{"range", "64M+"}, {"count", Buckets[4]}});

  return makeJSONResult(CapID, UnitID, json::Object{
      {"profile_path", Path},
      {"transfer_count", TotalRows},
      {"total_bytes", TotalBytes},
      {"byte_histogram", std::move(Histogram)},
  });
}
