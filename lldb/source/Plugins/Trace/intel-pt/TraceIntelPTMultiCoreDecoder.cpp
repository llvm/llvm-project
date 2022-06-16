//===-- TraceIntelPTMultiCoreDecoder.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTMultiCoreDecoder.h"

#include "TraceIntelPT.h"

#include "llvm/Support/Error.h"

#include <linux/perf_event.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

struct PerfContextSwitchRecord {
  struct perf_event_header header;
  uint32_t next_prev_pid;
  uint32_t next_prev_tid;
  uint32_t pid, tid;
  uint64_t time_in_nanos;

  bool IsOut() const { return header.misc & PERF_RECORD_MISC_SWITCH_OUT; }
};

struct ContextSwitchRecord {
  uint64_t tsc;
  bool is_out;
  /// A pid of 0 indicates an execution in the kernel
  lldb::pid_t pid;
  lldb::tid_t tid;

  bool IsOut() const { return is_out; }

  bool IsIn() const { return !is_out; }
};

uint64_t ThreadContinuousExecution::GetErrorFreeTSC() const {
  switch (variant) {
  case Variant::Complete:
    return tscs.complete.start; // end would also work
  case Variant::HintedStart:
    return tscs.hinted_start.end;
  case Variant::HintedEnd:
    return tscs.hinted_end.start;
  case Variant::OnlyEnd:
    return tscs.only_end.end;
  case Variant::OnlyStart:
    return tscs.only_start.start;
  }
}

ThreadContinuousExecution ThreadContinuousExecution::CreateCompleteExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, uint64_t start, uint64_t end) {
  ThreadContinuousExecution o(core_id, tid);
  o.variant = Variant::Complete;
  o.tscs.complete.start = start;
  o.tscs.complete.end = end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateHintedStartExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, uint64_t hinted_start,
    uint64_t end) {
  ThreadContinuousExecution o(core_id, tid);
  o.variant = Variant::HintedStart;
  o.tscs.hinted_start.hinted_start = hinted_start;
  o.tscs.hinted_start.end = end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateHintedEndExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, uint64_t start,
    uint64_t hinted_end) {
  ThreadContinuousExecution o(core_id, tid);
  o.variant = Variant::HintedEnd;
  o.tscs.hinted_end.start = start;
  o.tscs.hinted_end.hinted_end = hinted_end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateOnlyEndExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, uint64_t end) {
  ThreadContinuousExecution o(core_id, tid);
  o.variant = Variant::OnlyEnd;
  o.tscs.only_end.end = end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateOnlyStartExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, uint64_t start) {
  ThreadContinuousExecution o(core_id, tid);
  o.variant = Variant::OnlyStart;
  o.tscs.only_start.start = start;
  return o;
}

bool ThreadContinuousExecution::operator<(
    const ThreadContinuousExecution &o) const {
  // We can compare by GetErrorFreeTSC because context switches across CPUs can
  // be sorted by any of its TSC.
  return GetErrorFreeTSC() < o.GetErrorFreeTSC();
}

/// Tries to recover a continuous execution by analyzing two consecutive context
/// switch records.
static Error
HandleContextSwitch(core_id_t core_id,
                    const LinuxPerfZeroTscConversion &tsc_conversion,
                    const ContextSwitchRecord &record,
                    const Optional<ContextSwitchRecord> &prev_record,
                    std::function<void(ThreadContinuousExecution &&execution)>
                        on_new_thread_execution) {
  if (!prev_record) {
    if (record.IsOut())
      on_new_thread_execution(ThreadContinuousExecution::CreateOnlyEndExecution(
          core_id, record.tid, record.tsc));
    // The 'in' case will be handled later when we try to look for its end
    return Error::success();
  }

  const ContextSwitchRecord &prev = *prev_record;
  if (prev.tsc > record.tsc)
    return createStringError(
        inconvertibleErrorCode(),
        formatv("A context switch record out doesn't happen after the previous "
                "record. Previous TSC= {0}, current TSC = {1}.",
                prev.tsc, record.tsc));

  if (record.IsIn() && prev.IsIn()) {
    // We found two consecutive ins, which means that we didn't capture
    // the end of the previous execution.
    on_new_thread_execution(ThreadContinuousExecution::CreateHintedEndExecution(
        core_id, prev.tid, prev.tsc, record.tsc - 1));
  } else if (record.IsOut() && prev.IsOut()) {
    // We found two consecutive outs, that means that we didn't capture
    // the beginning of the current execution.
    on_new_thread_execution(
        ThreadContinuousExecution::CreateHintedStartExecution(
            core_id, record.tid, prev.tsc + 1, record.tsc));
  } else if (record.IsOut() && prev.IsIn()) {
    if (record.pid == prev.pid && record.tid == prev.tid) {
      /// A complete execution
      on_new_thread_execution(
          ThreadContinuousExecution::CreateCompleteExecution(
              core_id, record.tid, prev.tsc, record.tsc));
    } else {
      // An out after the in of a different thread. The first one doesn't
      // have an end, and the second one doesn't have a start.
      on_new_thread_execution(
          ThreadContinuousExecution::CreateHintedEndExecution(
              core_id, prev.tid, prev.tsc, record.tsc - 1));
      on_new_thread_execution(
          ThreadContinuousExecution::CreateHintedStartExecution(
              core_id, record.tid, prev.tsc + 1, record.tsc));
    }
  }
  return Error::success();
}

/// Decodes a context switch trace gotten with perf_event_open.
///
/// \param[in] data
///   The context switch trace in binary format.
///
/// \param[i] core_id
///   The core_id where the trace were gotten from.
///
/// \param[in] tsc_conversion
///   The conversion values used to confert nanoseconds to TSC.
///
/// \param[in] on_new_thread_execution
///   Callback to be invoked whenever a continuous execution is recovered from
///   the trace.
static Error DecodePerfContextSwitchTrace(
    ArrayRef<uint8_t> data, core_id_t core_id,
    const LinuxPerfZeroTscConversion &tsc_conversion,
    std::function<void(ThreadContinuousExecution &&execution)>
        on_new_thread_execution) {
  auto CreateError = [&](size_t offset, auto error) -> Error {
    return createStringError(inconvertibleErrorCode(),
                             formatv("Malformed perf context switch trace for "
                                     "cpu {0} at offset {1}. {2}",
                                     core_id, offset, error));
  };

  Optional<ContextSwitchRecord> prev_record;
  for (size_t offset = 0; offset < data.size();) {
    const PerfContextSwitchRecord &perf_record =
        *reinterpret_cast<const PerfContextSwitchRecord *>(data.data() +
                                                           offset);
    // A record of 1000 uint64_t's or more should mean that the data is wrong
    if (perf_record.header.size == 0 ||
        perf_record.header.size > sizeof(uint64_t) * 1000)
      return CreateError(offset, formatv("A record of {0} bytes was found.",
                                         perf_record.header.size));

    // We add + 100 to this record because some systems might have custom
    // records. In any case, we are looking only for abnormal data.
    if (perf_record.header.type >= PERF_RECORD_MAX + 100)
      return CreateError(offset, formatv("Invalid record type {0} was found.",
                                         perf_record.header.type));

    if (perf_record.header.type == PERF_RECORD_SWITCH_CPU_WIDE) {
      ContextSwitchRecord record{tsc_conversion.ToTSC(std::chrono::nanoseconds(
                                     perf_record.time_in_nanos)),
                                 perf_record.IsOut(),
                                 static_cast<lldb::pid_t>(perf_record.pid),
                                 static_cast<lldb::tid_t>(perf_record.tid)};

      if (Error err = HandleContextSwitch(core_id, tsc_conversion, record,
                                          prev_record, on_new_thread_execution))
        return CreateError(offset, toString(std::move(err)));

      prev_record = record;
    }
    offset += perf_record.header.size;
  }

  // We might have an incomplete last record
  if (prev_record && prev_record->IsIn())
    on_new_thread_execution(ThreadContinuousExecution::CreateOnlyStartExecution(
        core_id, prev_record->tid, prev_record->tsc));

  return Error::success();
}

TraceIntelPTMultiCoreDecoder::TraceIntelPTMultiCoreDecoder(
    TraceIntelPT &trace, ArrayRef<core_id_t> core_ids, ArrayRef<tid_t> tids,
    const LinuxPerfZeroTscConversion &tsc_conversion)
    : m_trace(trace), m_cores(core_ids.begin(), core_ids.end()),
      m_tids(tids.begin(), tids.end()), m_tsc_conversion(tsc_conversion) {}

bool TraceIntelPTMultiCoreDecoder::TracesThread(lldb::tid_t tid) const {
  return m_tids.count(tid);
}

DecodedThreadSP TraceIntelPTMultiCoreDecoder::Decode(Thread &thread) {
  if (Error err = DecodeContextSwitchTraces())
    return std::make_shared<DecodedThread>(thread.shared_from_this(),
                                           std::move(err));

  return std::make_shared<DecodedThread>(
      thread.shared_from_this(),
      createStringError(inconvertibleErrorCode(), "unimplemented"));
}

Error TraceIntelPTMultiCoreDecoder::DecodeContextSwitchTraces() {
  if (m_setup_error)
    return createStringError(inconvertibleErrorCode(), m_setup_error->c_str());

  if (m_continuous_executions_per_thread)
    return Error::success();

  m_continuous_executions_per_thread.emplace();

  auto do_decode = [&]() -> Error {
    // We'll decode all context switch traces, identify continuous executions
    // and group them by thread.
    for (core_id_t core_id : m_cores) {
      Error err = m_trace.OnCoreBinaryDataRead(
          core_id, IntelPTDataKinds::kPerfContextSwitchTrace,
          [&](ArrayRef<uint8_t> data) -> Error {
            return DecodePerfContextSwitchTrace(
                data, core_id, m_tsc_conversion,
                [&](const ThreadContinuousExecution &execution) {
                  (*m_continuous_executions_per_thread)[execution.tid]
                      .push_back(execution);
                });
          });
      if (err) {
        m_setup_error = toString(std::move(err));
        return createStringError(inconvertibleErrorCode(),
                                 m_setup_error->c_str());
      }
    }
    // We now sort the executions of each to have them ready for instruction
    // decoding
    for (auto &tid_executions : *m_continuous_executions_per_thread)
      std::sort(tid_executions.second.begin(), tid_executions.second.end());

    return Error::success();
  };

  return m_trace.GetTimer().ForGlobal().TimeTask<Error>(
      "Context switch trace decoding", do_decode);
}

size_t TraceIntelPTMultiCoreDecoder::GetNumContinuousExecutionsForThread(
    lldb::tid_t tid) const {
  if (!m_continuous_executions_per_thread)
    return 0;
  auto it = m_continuous_executions_per_thread->find(tid);
  if (it == m_continuous_executions_per_thread->end())
    return 0;
  return it->second.size();
}

size_t TraceIntelPTMultiCoreDecoder::GetTotalContinuousExecutionsCount() const {
  if (!m_continuous_executions_per_thread)
    return 0;
  size_t count = 0;
  for (const auto &kv : *m_continuous_executions_per_thread)
    count += kv.second.size();
  return count;
}
