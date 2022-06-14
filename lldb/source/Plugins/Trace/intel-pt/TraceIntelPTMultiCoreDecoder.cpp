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

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

TraceIntelPTMultiCoreDecoder::TraceIntelPTMultiCoreDecoder(TraceIntelPT &trace)
    : m_trace(&trace) {
  for (Process *proc : trace.GetAllProcesses()) {
    for (ThreadSP thread_sp : proc->GetThreadList().Threads()) {
      m_tids.insert(thread_sp->GetID());
    }
  }
}

bool TraceIntelPTMultiCoreDecoder::TracesThread(lldb::tid_t tid) const {
  return m_tids.count(tid);
}

DecodedThreadSP TraceIntelPTMultiCoreDecoder::Decode(Thread &thread) {
  if (Error err = CorrelateContextSwitchesAndIntelPtTraces())
    return std::make_shared<DecodedThread>(thread.shared_from_this(),
                                           std::move(err));
  auto it = m_decoded_threads.find(thread.GetID());
  if (it != m_decoded_threads.end())
    return it->second;

  DecodedThreadSP decoded_thread_sp =
      std::make_shared<DecodedThread>(thread.shared_from_this());

  Error err = m_trace->OnAllCoresBinaryDataRead(
      IntelPTDataKinds::kTraceBuffer,
      [&](const DenseMap<core_id_t, ArrayRef<uint8_t>>& buffers) -> Error {
        auto it = m_continuous_executions_per_thread->find(thread.GetID());
        if (it != m_continuous_executions_per_thread->end())
          DecodeSystemWideTraceForThread(*decoded_thread_sp, *m_trace, buffers, it->second);

        return Error::success();
      });
  if (err)
    decoded_thread_sp->SetAsFailed(std::move(err));

  m_decoded_threads.try_emplace(thread.GetID(), decoded_thread_sp);
  return decoded_thread_sp;
}

static Expected<std::vector<IntelPTThreadSubtrace>>
GetIntelPTSubtracesForCore(TraceIntelPT &trace, core_id_t core_id) {
  std::vector<IntelPTThreadSubtrace> intel_pt_subtraces;
  Error err = trace.OnCoreBinaryDataRead(
      core_id, IntelPTDataKinds::kTraceBuffer,
      [&](ArrayRef<uint8_t> data) -> Error {
        Expected<std::vector<IntelPTThreadSubtrace>> split_trace =
            SplitTraceInContinuousExecutions(trace, data);
        if (!split_trace)
          return split_trace.takeError();

        intel_pt_subtraces = std::move(*split_trace);
        return Error::success();
      });
  if (err)
    return std::move(err);
  return intel_pt_subtraces;
}

Expected<
    DenseMap<lldb::tid_t, std::vector<IntelPTThreadContinousExecution>>>
TraceIntelPTMultiCoreDecoder::DoCorrelateContextSwitchesAndIntelPtTraces() {
  DenseMap<lldb::tid_t, std::vector<IntelPTThreadContinousExecution>>
      continuous_executions_per_thread;

  Optional<LinuxPerfZeroTscConversion> conv_opt =
      m_trace->GetPerfZeroTscConversion();
  if (!conv_opt)
    return createStringError(
        inconvertibleErrorCode(),
        "TSC to nanoseconds conversion values were not found");

  LinuxPerfZeroTscConversion tsc_conversion = *conv_opt;

  for (core_id_t core_id : m_trace->GetTracedCores()) {
    Expected<std::vector<IntelPTThreadSubtrace>> intel_pt_subtraces =
        GetIntelPTSubtracesForCore(*m_trace, core_id);
    if (!intel_pt_subtraces)
      return intel_pt_subtraces.takeError();

    // We'll be iterating through the thread continuous executions and the intel
    // pt subtraces sorted by time.
    auto it = intel_pt_subtraces->begin();
    auto on_new_thread_execution =
        [&](const ThreadContinuousExecution& thread_execution) {
          IntelPTThreadContinousExecution execution(thread_execution);

          for (; it != intel_pt_subtraces->end() &&
                 it->tsc < thread_execution.GetEndTSC();
               it++) {
            if (it->tsc > thread_execution.GetStartTSC()) {
              execution.intelpt_subtraces.push_back(*it);
            } else {
              m_unattributed_intelpt_subtraces++;
            }
          }
          continuous_executions_per_thread[thread_execution.tid].push_back(
              execution);
        };
    Error err = m_trace->OnCoreBinaryDataRead(
        core_id, IntelPTDataKinds::kPerfContextSwitchTrace,
        [&](ArrayRef<uint8_t> data) -> Error {
          Expected<std::vector<ThreadContinuousExecution>> executions =
              DecodePerfContextSwitchTrace(data, core_id, tsc_conversion);
          if (!executions)
            return executions.takeError();
          for (const ThreadContinuousExecution &exec : *executions)
            on_new_thread_execution(exec);
          return Error::success();
        });
    if (err)
      return std::move(err);
  }
  // We now sort the executions of each thread to have them ready for
  // instruction decoding
  for (auto &tid_executions : continuous_executions_per_thread)
    std::sort(tid_executions.second.begin(), tid_executions.second.end());

  return continuous_executions_per_thread;
}

Error TraceIntelPTMultiCoreDecoder::CorrelateContextSwitchesAndIntelPtTraces() {
  if (m_setup_error)
    return createStringError(inconvertibleErrorCode(), m_setup_error->c_str());

  if (m_continuous_executions_per_thread)
    return Error::success();

  Error err = m_trace->GetTimer().ForGlobal().TimeTask<Error>(
      "Context switch and Intel PT traces correlation", [&]() -> Error {
        if (auto correlation = DoCorrelateContextSwitchesAndIntelPtTraces()) {
          m_continuous_executions_per_thread.emplace(std::move(*correlation));
          return Error::success();
        } else {
          return correlation.takeError();
        }
      });
  if (err) {
    m_setup_error = toString(std::move(err));
    return createStringError(inconvertibleErrorCode(), m_setup_error->c_str());
  }
  return Error::success();
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
