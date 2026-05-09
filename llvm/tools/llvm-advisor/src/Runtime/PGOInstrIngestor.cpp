//===------------------- PGOInstrIngestor.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of PGOInstrIngestor in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/PGOInstrIngestor.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> PGOInstrIngestor::load(StringRef Path) {
  IntrusiveRefCntPtr<vfs::FileSystem> FS = vfs::getRealFileSystem();
  Expected<std::unique_ptr<InstrProfReader>> Reader =
      InstrProfReader::create(Path, *FS);
  if (!Reader)
    return Reader.takeError();

  uint64_t FunctionCount = 0;
  uint64_t CounterCount = 0;
  uint64_t TotalCount = 0;
  uint64_t MaxFunctionCount = 0;
  uint64_t ValueSites = 0;
  json::Array Functions;

  for (const NamedInstrProfRecord &Record : **Reader) {
    uint64_t FunctionTotal = 0;
    for (uint64_t Count : Record.Counts) {
      FunctionTotal += Count;
      ++CounterCount;
    }
    for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
      ValueSites += Record.getNumValueSites(Kind);

    TotalCount += FunctionTotal;
    MaxFunctionCount = std::max(MaxFunctionCount, FunctionTotal);
    ++FunctionCount;

    if (Functions.size() < 256)
      Functions.push_back(json::Object{
          {"function", Record.Name},
          {"hash", static_cast<int64_t>(Record.Hash)},
          {"counters", static_cast<int64_t>(Record.Counts.size())},
          {"entry_count",
           static_cast<int64_t>(Record.Counts.empty() ? 0 : Record.Counts[0])},
          {"total_count", static_cast<int64_t>(FunctionTotal)}});
  }

  if (Error Err = (*Reader)->getError())
    return std::move(Err);

  return json::Object{
      {"kind", "pgo-profile"},
      {"format", "instrprof"},
      {"path", Path},
      {"version", static_cast<int64_t>((*Reader)->getVersion())},
      {"ir_level", (*Reader)->isIRLevelProfile()},
      {"context_sensitive", (*Reader)->hasCSIRLevelProfile()},
      {"memory_profile", (*Reader)->hasMemoryProfile()},
      {"function_count", static_cast<int64_t>(FunctionCount)},
      {"counter_count", static_cast<int64_t>(CounterCount)},
      {"value_site_count", static_cast<int64_t>(ValueSites)},
      {"total_count", static_cast<int64_t>(TotalCount)},
      {"max_function_count", static_cast<int64_t>(MaxFunctionCount)},
      {"functions", std::move(Functions)}};
}
