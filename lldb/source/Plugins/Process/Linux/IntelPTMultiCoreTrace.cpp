//===-- IntelPTMultiCoreTrace.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTMultiCoreTrace.h"

#include "Procfs.h"

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

static bool IsTotalBufferLimitReached(ArrayRef<core_id_t> cores,
                                      const TraceIntelPTStartRequest &request) {
  uint64_t required = cores.size() * request.trace_buffer_size;
  uint64_t limit = request.process_buffer_size_limit.getValueOr(
      std::numeric_limits<uint64_t>::max());
  return required > limit;
}

static Error IncludePerfEventParanoidMessageInError(Error &&error) {
  return createStringError(
      inconvertibleErrorCode(),
      "%s\nYou might need to rerun as sudo or to set "
      "/proc/sys/kernel/perf_event_paranoid to a value of 0 or -1.",
      toString(std::move(error)).c_str());
}

Expected<IntelPTMultiCoreTraceUP> IntelPTMultiCoreTrace::StartOnAllCores(
    const TraceIntelPTStartRequest &request) {
  Expected<ArrayRef<core_id_t>> core_ids = GetAvailableLogicalCoreIDs();
  if (!core_ids)
    return core_ids.takeError();

  if (IsTotalBufferLimitReached(*core_ids, request))
    return createStringError(
        inconvertibleErrorCode(),
        "The process can't be traced because the process trace size limit "
        "has been reached. Consider retracing with a higher limit.");

  llvm::DenseMap<core_id_t, IntelPTSingleBufferTraceUP> buffers;
  for (core_id_t core_id : *core_ids) {
    if (Expected<IntelPTSingleBufferTraceUP> core_trace =
            IntelPTSingleBufferTrace::Start(request, /*tid=*/None, core_id))
      buffers.try_emplace(core_id, std::move(*core_trace));
    else
      return IncludePerfEventParanoidMessageInError(core_trace.takeError());
  }

  return IntelPTMultiCoreTraceUP(new IntelPTMultiCoreTrace(std::move(buffers)));
}

void IntelPTMultiCoreTrace::ForEachCore(
    std::function<void(core_id_t core_id,
                       const IntelPTSingleBufferTrace &core_trace)>
        callback) {
  for (auto &it : m_traces_per_core)
    callback(it.first, *it.second);
}
