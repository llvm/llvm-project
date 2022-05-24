//===-- PerfContextSwitchDecoder.cpp --======------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfContextSwitchDecoder.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

/// Copied from <linux/perf_event.h> to avoid depending on perf_event.h on
/// non-linux platforms.
/// \{
struct perf_event_header {
  uint32_t type;
  uint16_t misc;
  uint16_t size;
};

#define PERF_RECORD_MISC_SWITCH_OUT (1 << 13)
#define PERF_RECORD_MAX 19
#define PERF_RECORD_SWITCH_CPU_WIDE 15
/// \}

/// Record found in the perf_event context switch traces. It might contain
/// additional fields in memory, but header.size should have the actual size
/// of the record.
struct PerfContextSwitchRecord {
  struct perf_event_header header;
  uint32_t next_prev_pid;
  uint32_t next_prev_tid;
  uint32_t pid, tid;
  uint64_t time_in_nanos;

  bool IsOut() const { return header.misc & PERF_RECORD_MISC_SWITCH_OUT; }

  bool IsContextSwitchRecord() const {
    return header.type == PERF_RECORD_SWITCH_CPU_WIDE;
  }

  /// \return
  ///   An \a llvm::Error if the record looks obviously wrong, or \a
  ///   llvm::Error::success() otherwise.
  Error SanityCheck() const {
    // A record of too many uint64_t's or more should mean that the data is
    // wrong
    if (header.size == 0 || header.size > sizeof(uint64_t) * 1000)
      return createStringError(
          inconvertibleErrorCode(),
          formatv("A record of {0} bytes was found.", header.size));

    // We add some numbers to PERF_RECORD_MAX because some systems might have
    // custom records. In any case, we are looking only for abnormal data.
    if (header.type >= PERF_RECORD_MAX + 100)
      return createStringError(
          inconvertibleErrorCode(),
          formatv("Invalid record type {0} was found.", header.type));
    return Error::success();
  }
};

/// Record produced after parsing the raw context switch trace produce by
/// perf_event. A major difference between this struct and
/// PerfContextSwitchRecord is that this one uses tsc instead of nanos.
struct ContextSwitchRecord {
  uint64_t tsc;
  /// Whether the switch is in or out
  bool is_out;
  /// pid = 0 and tid = 0 indicate the swapper or idle process, which normally
  /// runs after a context switch out of a normal user thread.
  lldb::pid_t pid;
  lldb::tid_t tid;

  bool IsOut() const { return is_out; }

  bool IsIn() const { return !is_out; }
};

uint64_t ThreadContinuousExecution::GetLowestKnownTSC() const {
  switch (variant) {
  case Variant::Complete:
    return tscs.complete.start;
  case Variant::OnlyStart:
    return tscs.only_start.start;
  case Variant::OnlyEnd:
    return tscs.only_end.end;
  case Variant::HintedEnd:
    return tscs.hinted_end.start;
  case Variant::HintedStart:
    return tscs.hinted_start.end;
  }
}

uint64_t ThreadContinuousExecution::GetStartTSC() const {
  switch (variant) {
  case Variant::Complete:
    return tscs.complete.start;
  case Variant::OnlyStart:
    return tscs.only_start.start;
  case Variant::OnlyEnd:
    return 0;
  case Variant::HintedEnd:
    return tscs.hinted_end.start;
  case Variant::HintedStart:
    return tscs.hinted_start.hinted_start;
  }
}

uint64_t ThreadContinuousExecution::GetEndTSC() const {
  switch (variant) {
  case Variant::Complete:
    return tscs.complete.end;
  case Variant::OnlyStart:
    return std::numeric_limits<uint64_t>::max();
  case Variant::OnlyEnd:
    return tscs.only_end.end;
  case Variant::HintedEnd:
    return tscs.hinted_end.hinted_end;
  case Variant::HintedStart:
    return tscs.hinted_start.end;
  }
}

ThreadContinuousExecution ThreadContinuousExecution::CreateCompleteExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, lldb::pid_t pid, uint64_t start,
    uint64_t end) {
  ThreadContinuousExecution o(core_id, tid, pid);
  o.variant = Variant::Complete;
  o.tscs.complete.start = start;
  o.tscs.complete.end = end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateHintedStartExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, lldb::pid_t pid,
    uint64_t hinted_start, uint64_t end) {
  ThreadContinuousExecution o(core_id, tid, pid);
  o.variant = Variant::HintedStart;
  o.tscs.hinted_start.hinted_start = hinted_start;
  o.tscs.hinted_start.end = end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateHintedEndExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, lldb::pid_t pid, uint64_t start,
    uint64_t hinted_end) {
  ThreadContinuousExecution o(core_id, tid, pid);
  o.variant = Variant::HintedEnd;
  o.tscs.hinted_end.start = start;
  o.tscs.hinted_end.hinted_end = hinted_end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateOnlyEndExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, lldb::pid_t pid, uint64_t end) {
  ThreadContinuousExecution o(core_id, tid, pid);
  o.variant = Variant::OnlyEnd;
  o.tscs.only_end.end = end;
  return o;
}

ThreadContinuousExecution ThreadContinuousExecution::CreateOnlyStartExecution(
    lldb::core_id_t core_id, lldb::tid_t tid, lldb::pid_t pid, uint64_t start) {
  ThreadContinuousExecution o(core_id, tid, pid);
  o.variant = Variant::OnlyStart;
  o.tscs.only_start.start = start;
  return o;
}

static Error RecoverExecutionsFromConsecutiveRecords(
    core_id_t core_id, const LinuxPerfZeroTscConversion &tsc_conversion,
    const ContextSwitchRecord &current_record,
    const Optional<ContextSwitchRecord> &prev_record,
    std::function<void(const ThreadContinuousExecution &execution)>
        on_new_execution) {
  if (!prev_record) {
    if (current_record.IsOut()) {
      on_new_execution(ThreadContinuousExecution::CreateOnlyEndExecution(
          core_id, current_record.tid, current_record.pid, current_record.tsc));
    }
    // The 'in' case will be handled later when we try to look for its end
    return Error::success();
  }

  const ContextSwitchRecord &prev = *prev_record;
  if (prev.tsc >= current_record.tsc)
    return createStringError(
        inconvertibleErrorCode(),
        formatv("A context switch record doesn't happen after the previous "
                "record. Previous TSC= {0}, current TSC = {1}.",
                prev.tsc, current_record.tsc));

  if (current_record.IsIn() && prev.IsIn()) {
    // We found two consecutive ins, which means that we didn't capture
    // the end of the previous execution.
    on_new_execution(ThreadContinuousExecution::CreateHintedEndExecution(
        core_id, prev.tid, prev.pid, prev.tsc, current_record.tsc - 1));
  } else if (current_record.IsOut() && prev.IsOut()) {
    // We found two consecutive outs, that means that we didn't capture
    // the beginning of the current execution.
    on_new_execution(ThreadContinuousExecution::CreateHintedStartExecution(
        core_id, current_record.tid, current_record.pid, prev.tsc + 1,
        current_record.tsc));
  } else if (current_record.IsOut() && prev.IsIn()) {
    if (current_record.pid == prev.pid && current_record.tid == prev.tid) {
      /// A complete execution
      on_new_execution(ThreadContinuousExecution::CreateCompleteExecution(
          core_id, current_record.tid, current_record.pid, prev.tsc,
          current_record.tsc));
    } else {
      // An out after the in of a different thread. The first one doesn't
      // have an end, and the second one doesn't have a start.
      on_new_execution(ThreadContinuousExecution::CreateHintedEndExecution(
          core_id, prev.tid, prev.pid, prev.tsc, current_record.tsc - 1));
      on_new_execution(ThreadContinuousExecution::CreateHintedStartExecution(
          core_id, current_record.tid, current_record.pid, prev.tsc + 1,
          current_record.tsc));
    }
  }
  return Error::success();
}

#include <fstream>

Expected<std::vector<ThreadContinuousExecution>>
lldb_private::trace_intel_pt::DecodePerfContextSwitchTrace(
    ArrayRef<uint8_t> data, core_id_t core_id,
    const LinuxPerfZeroTscConversion &tsc_conversion) {

  std::vector<ThreadContinuousExecution> executions;

  // This offset is used to create the error message in case of failures.
  size_t offset = 0;

  auto do_decode = [&]() -> Error {
    Optional<ContextSwitchRecord> prev_record;
    while (offset < data.size()) {
      const PerfContextSwitchRecord &perf_record =
          *reinterpret_cast<const PerfContextSwitchRecord *>(data.data() +
                                                             offset);
      if (Error err = perf_record.SanityCheck())
        return err;

      if (perf_record.IsContextSwitchRecord()) {
        ContextSwitchRecord record{
            tsc_conversion.ToTSC(perf_record.time_in_nanos),
            perf_record.IsOut(), static_cast<lldb::pid_t>(perf_record.pid),
            static_cast<lldb::tid_t>(perf_record.tid)};

        if (Error err = RecoverExecutionsFromConsecutiveRecords(
                core_id, tsc_conversion, record, prev_record,
                [&](const ThreadContinuousExecution &execution) {
                  executions.push_back(execution);
                }))
          return err;

        prev_record = record;
      }
      offset += perf_record.header.size;
    }

    // We might have an incomplete last record
    if (prev_record && prev_record->IsIn())
      executions.push_back(ThreadContinuousExecution::CreateOnlyStartExecution(
          core_id, prev_record->tid, prev_record->pid, prev_record->tsc));
    return Error::success();
  };

  if (Error err = do_decode())
    return createStringError(inconvertibleErrorCode(),
                             formatv("Malformed perf context switch trace for "
                                     "cpu {0} at offset {1}. {2}",
                                     core_id, offset,
                                     toString(std::move(err))));

  return executions;
}
