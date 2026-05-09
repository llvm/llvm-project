//===------------------- MemProfIngestor.cpp - LLVM Advisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of MemProfIngestor in Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/MemProfIngestor.h"
#include "llvm/ProfileData/MemProfReader.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> MemProfIngestor::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(), "empty memprof path");
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(), "cannot read memprof '%s'",
                             Path.data());

  if (!memprof::YAMLMemProfReader::hasFormat(**Buffer))
    return createStringError(inconvertibleErrorCode(),
                             "unsupported memprof format for '%s'",
                             Path.data());

  Expected<std::unique_ptr<memprof::YAMLMemProfReader>> Reader =
      memprof::YAMLMemProfReader::create(std::move(*Buffer));
  if (!Reader)
    return Reader.takeError();

  uint64_t FunctionCount = 0;
  uint64_t AllocSites = 0;
  uint64_t CallSites = 0;
  json::Array Functions;
  for (const memprof::MemProfReader::GuidMemProfRecordPair &Entry : **Reader) {
    AllocSites += Entry.second.AllocSites.size();
    CallSites += Entry.second.CallSites.size();
    ++FunctionCount;
    if (Functions.size() < 256)
      Functions.push_back(json::Object{
          {"guid", static_cast<int64_t>(Entry.first)},
          {"alloc_sites", static_cast<int64_t>(Entry.second.AllocSites.size())},
          {"call_sites", static_cast<int64_t>(Entry.second.CallSites.size())}});
  }

  return json::Object{{"kind", "memprof-profile"},
                      {"format", "memprof-yaml"},
                      {"path", Path},
                      {"function_count", static_cast<int64_t>(FunctionCount)},
                      {"alloc_sites", static_cast<int64_t>(AllocSites)},
                      {"call_sites", static_cast<int64_t>(CallSites)},
                      {"functions", std::move(Functions)}};
}
