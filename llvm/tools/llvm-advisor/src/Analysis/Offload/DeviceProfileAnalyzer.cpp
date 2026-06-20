//===--- DeviceProfileAnalyzer.cpp - LLVM Advisor ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/DeviceProfileAnalyzer.h"
#include "Analysis/Utils/CSVUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

// rocprof by default writes "results.csv" / "pmc_1.csv" / "*.csv" in CWD.
static SmallVector<std::string, 4>
findRocprofCSVFiles(const CapabilityContext &Context) {
  SmallVector<std::string, 4> Found;
  if (Context.WorkingDirectory.empty())
    return Found;

  std::error_code EC;
  for (sys::fs::directory_iterator It(Context.WorkingDirectory, EC), End;
       !EC && It != End; It.increment(EC)) {
    StringRef P = It->path();
    if (!P.ends_with(".csv"))
      continue;
    // Skip memory-copy CSVs (handled by MemoryTransferAnalyzer).
    StringRef Stem = sys::path::stem(P);
    if (Stem.contains("mem") || Stem.contains("transfer"))
      continue;
    Found.push_back(P.str());
  }
  return Found;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct KernelStat {
  std::string Name;
  std::string Agent;
  int64_t TotalNs = 0;
  int64_t MinNs = INT64_MAX;
  int64_t MaxNs = 0;
  int64_t Count = 0;
};

static void accumulateRow(StringMap<KernelStat> &Stats,
                          const SmallVector<StringRef, 16> &Headers,
                          const SmallVector<StringRef, 16> &Row) {
  int KernelCol = findCol(Headers, "KernelName");
  if (KernelCol < 0)
    KernelCol = findCol(Headers, "Name");
  int BeginCol = findCol(Headers, "BeginNs");
  int EndCol = findCol(Headers, "EndNs");
  int AgentCol = findCol(Headers, "gpu-agent");

  if (KernelCol < 0 || BeginCol < 0 || EndCol < 0)
    return;
  if (KernelCol >= (int)Row.size() || BeginCol >= (int)Row.size() ||
      EndCol >= (int)Row.size())
    return;

  StringRef KernelName = Row[KernelCol].trim('"');
  int64_t Begin, End;
  if (Row[BeginCol].getAsInteger(10, Begin) || Row[EndCol].getAsInteger(10, End))
    return;

  int64_t DurNs = End - Begin;
  if (DurNs < 0)
    return;

  KernelStat &S = Stats[KernelName];
  if (S.Name.empty()) {
    S.Name = KernelName.str();
    if (AgentCol >= 0 && AgentCol < (int)Row.size())
      S.Agent = Row[AgentCol].trim('"').str();
  }
  S.TotalNs += DurNs;
  S.MinNs = std::min(S.MinNs, DurNs);
  S.MaxNs = std::max(S.MaxNs, DurNs);
  S.Count++;
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

Expected<std::unique_ptr<CapabilityResult>>
DeviceProfileAnalyzer::run(const CapabilityContext &Context) {
  SmallVector<std::string, 4> CSVFiles = findRocprofCSVFiles(Context);
  if (CSVFiles.empty())
    return makeUnavailableResult(
        getCapabilityID(), Context.Unit.ID,
        "no rocprof CSV output found in working directory");

  StringMap<KernelStat> Stats;
  SmallVector<std::string, 4> ParsedFiles;

  for (const std::string &FilePath : CSVFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(FilePath);
    if (!BufOrErr)
      continue;

    SmallVector<StringRef, 256> Lines;
    (*BufOrErr)->getBuffer().split(Lines, '\n');
    if (Lines.empty())
      continue;

    SmallVector<StringRef, 16> Headers = splitCSVRow(Lines[0]);
    if (findCol(Headers, "BeginNs") < 0 && findCol(Headers, "EndNs") < 0)
      continue; // not a kernel timing CSV

    for (size_t I = 1; I < Lines.size(); ++I) {
      StringRef Row = Lines[I].trim();
      if (Row.empty())
        continue;
      SmallVector<StringRef, 16> Fields = splitCSVRow(Row);
      accumulateRow(Stats, Headers, Fields);
    }
    ParsedFiles.push_back(FilePath);
  }

  if (Stats.empty())
    return makeUnavailableResult(
        getCapabilityID(), Context.Unit.ID,
        "no kernel dispatch records found in CSV files");

  // Collect and sort kernels by total time descending.
  SmallVector<const KernelStat *, 32> Sorted;
  for (const auto &KV : Stats)
    Sorted.push_back(&KV.second);
  llvm::sort(Sorted, [](const KernelStat *A, const KernelStat *B) {
    return A->TotalNs > B->TotalNs;
  });

  int64_t GrandTotalNs = 0;
  for (const KernelStat *S : Sorted)
    GrandTotalNs += S->TotalNs;

  json::Array Kernels;
  for (const KernelStat *S : Sorted) {
    int64_t MeanNs = S->Count > 0 ? S->TotalNs / S->Count : 0;
    double PctTime = GrandTotalNs > 0
                         ? 100.0 * static_cast<double>(S->TotalNs) /
                               static_cast<double>(GrandTotalNs)
                         : 0.0;
    json::Object K;
    K["name"] = S->Name;
    K["dispatches"] = S->Count;
    K["total_ns"] = S->TotalNs;
    K["mean_ns"] = MeanNs;
    K["min_ns"] = S->MinNs == INT64_MAX ? 0 : S->MinNs;
    K["max_ns"] = S->MaxNs;
    K["pct_of_total"] = PctTime;
    if (!S->Agent.empty())
      K["gpu_agent"] = S->Agent;
    Kernels.push_back(std::move(K));
  }

  json::Array FilesArr;
  for (const std::string &F : ParsedFiles)
    FilesArr.push_back(F);

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"kernel_count", static_cast<int64_t>(Stats.size())},
      {"total_gpu_time_ns", GrandTotalNs},
      {"sources", std::move(FilesArr)},
      {"kernels", std::move(Kernels)},
  });
}

} // namespace llvm::advisor
