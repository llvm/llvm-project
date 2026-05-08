//===--- DeviceTraceAnalyzer.cpp - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/DeviceTraceAnalyzer.h"
#include "Analysis/Utils/TraceDiscovery.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

static bool isKernelEvent(StringRef Name) {
  return Name.contains("Kernel") || Name.contains("kernel") ||
         Name.contains("Launch") || Name.contains("Dispatch");
}

Expected<std::unique_ptr<CapabilityResult>>
DeviceTraceAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  std::string Path = findTraceJSON(Context.WorkingDirectory);
  if (Path.empty())
    return makeUnavailableResult(CapID, UnitID,
                                 "no device trace JSON found");

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(), "cannot read trace: %s",
                             Path.c_str());

  Expected<json::Value> Parsed = json::parse((*MB)->getBuffer());
  if (!Parsed)
    return Parsed.takeError();

  const json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "trace root is not an object");

  const json::Array *Events = Root->getArray("traceEvents");
  if (!Events)
    return makeUnavailableResult(CapID, UnitID,
                                 "traceEvents array missing");

  json::Array KernelEvents;
  int64_t TotalKernelNs = 0;
  int64_t EventCount = 0;
  for (const json::Value &Ev : *Events) {
    const json::Object *E = Ev.getAsObject();
    if (!E)
      continue;
    std::optional<StringRef> Name = E->getString("name");
    if (!Name || !isKernelEvent(*Name))
      continue;
    json::Object K;
    K["name"] = *Name;
    if (std::optional<int64_t> Ts = E->getInteger("ts"))
      K["timestamp_us"] = *Ts;
    if (std::optional<int64_t> Dur = E->getInteger("dur")) {
      K["duration_us"] = *Dur;
      TotalKernelNs += *Dur * 1000;
    }
    if (std::optional<int64_t> TID = E->getInteger("tid"))
      K["tid"] = *TID;
    KernelEvents.push_back(std::move(K));
    ++EventCount;
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"trace_path", Path},
      {"kernel_event_count", EventCount},
      {"total_kernel_time_ns", TotalKernelNs},
      {"kernel_events", std::move(KernelEvents)},
  });
}
