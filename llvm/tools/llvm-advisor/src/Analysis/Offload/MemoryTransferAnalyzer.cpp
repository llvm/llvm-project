//===--- MemoryTransferAnalyzer.cpp - LLVM Advisor -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/MemoryTransferAnalyzer.h"
#include "Analysis/Utils/CSVUtils.h"
#include "Analysis/Utils/TraceDiscovery.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

// rocprof --hip-trace and --sys-trace write a JSON trace (Chrome format).
// Memory-copy events appear as "hsa_amd_memory_async_copy" or "hipMemcpy*".
// Some rocprof versions also write a separate "mem_copy.csv".
static std::string findMemCSV(const CapabilityContext &Context) {
  if (Context.WorkingDirectory.empty())
    return {};
  std::error_code EC;
  for (sys::fs::directory_iterator It(Context.WorkingDirectory, EC), End;
       !EC && It != End; It.increment(EC)) {
    StringRef P = It->path();
    StringRef Stem = sys::path::stem(P);
    if (P.ends_with(".csv") && (Stem.contains_insensitive("mem") ||
                                Stem.contains_insensitive("copy") ||
                                Stem.contains_insensitive("transfer")))
      return P.str();
  }
  return {};
}

// ---------------------------------------------------------------------------
// Transfer record
// ---------------------------------------------------------------------------

struct TransferRecord {
  std::string Name;      // API name or "hipMemcpyAsync" etc.
  std::string Direction; // "H2D", "D2H", "D2D", "H2H", "unknown"
  int64_t Bytes = 0;
  int64_t DurationNs = 0;
};

static StringRef classifyDirection(StringRef Name) {
  if (Name.contains_insensitive("H2D") ||
      Name.contains_insensitive("HostToDevice"))
    return "H2D";
  if (Name.contains_insensitive("D2H") ||
      Name.contains_insensitive("DeviceToHost"))
    return "D2H";
  if (Name.contains_insensitive("D2D") ||
      Name.contains_insensitive("DeviceToDevice"))
    return "D2D";
  if (Name.contains_insensitive("H2H") ||
      Name.contains_insensitive("HostToHost"))
    return "H2H";
  return "unknown";
}

// Parse Chrome-format JSON trace for memory copy events.
static void parseTraceJSON(StringRef Content,
                           SmallVectorImpl<TransferRecord> &Out) {
  Expected<json::Value> Parsed = json::parse(Content);
  if (!Parsed)
    return;

  const json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return;

  const json::Array *Events = Root->getArray("traceEvents");
  if (!Events)
    return;

  for (const json::Value &Ev : *Events) {
    const json::Object *E = Ev.getAsObject();
    if (!E)
      continue;

    std::optional<StringRef> Name = E->getString("name");
    if (!Name)
      continue;

    // Only care about memory copy / DMA events.
    bool IsCopy = Name->contains_insensitive("Memcpy") ||
                  Name->contains_insensitive("memory_copy") ||
                  Name->contains_insensitive("MemTransfer") ||
                  Name->contains_insensitive("async_copy");
    if (!IsCopy)
      continue;

    TransferRecord T;
    T.Name = Name->str();
    T.Direction = classifyDirection(*Name).str();

    if (std::optional<int64_t> Dur = E->getInteger("dur"))
      T.DurationNs = *Dur * 1000; // Chrome traces use microseconds

    // args.size (bytes) if present.
    if (const json::Object *Args = E->getObject("args"))
      if (std::optional<int64_t> Sz = Args->getInteger("size"))
        T.Bytes = *Sz;

    Out.push_back(std::move(T));
  }
}

// Parse a memory-copy CSV (columns:
// Name/Direction/SizeBytes/BandwidthMBs/TimeNs).
static void parseMemCSV(StringRef Content,
                        SmallVectorImpl<TransferRecord> &Out) {
  SmallVector<StringRef, 256> Lines;
  Content.split(Lines, '\n');
  if (Lines.empty())
    return;

  SmallVector<StringRef, 16> Headers = splitCSVRow(Lines[0]);
  int NameCol = findCol(Headers, "Name");
  int DirCol = findCol(Headers, "Direction");
  int SizeCol = findCol(Headers, "SizeBytes");
  int TimeCol = findCol(Headers, "TimeNs");

  if (NameCol < 0 && SizeCol < 0)
    return;

  for (size_t I = 1; I < Lines.size(); ++I) {
    StringRef Row = Lines[I].trim();
    if (Row.empty())
      continue;
    SmallVector<StringRef, 16> F = splitCSVRow(Row);

    TransferRecord T;
    if (NameCol >= 0 && NameCol < (int)F.size())
      T.Name = F[NameCol].trim('"').str();
    if (DirCol >= 0 && DirCol < (int)F.size())
      T.Direction = F[DirCol].trim('"').str();
    else
      T.Direction = classifyDirection(T.Name).str();
    if (SizeCol >= 0 && SizeCol < (int)F.size())
      F[SizeCol].getAsInteger(10, T.Bytes);
    if (TimeCol >= 0 && TimeCol < (int)F.size())
      F[TimeCol].getAsInteger(10, T.DurationNs);

    Out.push_back(std::move(T));
  }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

Expected<std::unique_ptr<CapabilityResult>>
MemoryTransferAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  SmallVector<TransferRecord, 64> Transfers;
  json::Array Sources;

  std::string TraceJSON = findTraceJSON(Context.WorkingDirectory);
  if (!TraceJSON.empty()) {
    if (auto MB = MemoryBuffer::getFile(TraceJSON)) {
      parseTraceJSON((*MB)->getBuffer(), Transfers);
      Sources.push_back(TraceJSON);
    }
  }

  std::string MemCSV = findMemCSV(Context);
  if (!MemCSV.empty()) {
    if (auto MB = MemoryBuffer::getFile(MemCSV)) {
      parseMemCSV((*MB)->getBuffer(), Transfers);
      Sources.push_back(MemCSV);
    }
  }

  if (Transfers.empty())
    return makeUnavailableResult(
        CapID, UnitID,
        "no memory transfer data found (run with rocprof --hip-trace or "
        "--sys-trace)");

  // Aggregate by direction.
  StringMap<int64_t> BytesByDir, TimeByDir, CountByDir;
  int64_t TotalBytes = 0, TotalNs = 0;
  for (const TransferRecord &T : Transfers) {
    BytesByDir[T.Direction] += T.Bytes;
    TimeByDir[T.Direction] += T.DurationNs;
    CountByDir[T.Direction]++;
    TotalBytes += T.Bytes;
    TotalNs += T.DurationNs;
  }

  json::Array ByDir;
  for (const auto &KV : BytesByDir) {
    double BW = TimeByDir[KV.getKey()] > 0
                    ? static_cast<double>(KV.second) /
                          (static_cast<double>(TimeByDir[KV.getKey()]) / 1e9) /
                          (1024.0 * 1024.0)
                    : 0.0;
    ByDir.push_back(json::Object{
        {"direction", KV.getKey().str()},
        {"total_bytes", KV.second},
        {"total_ns", TimeByDir[KV.getKey()]},
        {"count", CountByDir[KV.getKey()]},
        {"bandwidth_mbs", BW},
    });
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"transfer_count", static_cast<int64_t>(Transfers.size())},
      {"total_bytes", TotalBytes},
      {"total_ns", TotalNs},
      {"by_direction", std::move(ByDir)},
      {"sources", std::move(Sources)},
  });
}

} // namespace llvm::advisor
