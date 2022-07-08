//===-- DecodedThread.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DecodedThread.h"

#include <intel-pt.h>

#include "TraceCursorIntelPT.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

bool lldb_private::trace_intel_pt::IsLibiptError(int libipt_status) {
  return libipt_status < 0;
}

bool lldb_private::trace_intel_pt::IsEndOfStream(int libipt_status) {
  return libipt_status == -pte_eos;
}

bool lldb_private::trace_intel_pt::IsTscUnavailable(int libipt_status) {
  return libipt_status == -pte_no_time;
}

char IntelPTError::ID;

IntelPTError::IntelPTError(int libipt_error_code, lldb::addr_t address)
    : m_libipt_error_code(libipt_error_code), m_address(address) {
  assert(libipt_error_code < 0);
}

void IntelPTError::log(llvm::raw_ostream &OS) const {
  OS << pt_errstr(pt_errcode(m_libipt_error_code));
  if (m_address != LLDB_INVALID_ADDRESS && m_address > 0)
    OS << formatv(": {0:x+16}", m_address);
}

int64_t DecodedThread::GetItemsCount() const {
  return static_cast<int64_t>(m_item_kinds.size());
}

lldb::addr_t DecodedThread::GetInstructionLoadAddress(size_t item_index) const {
  return m_item_data[item_index].load_address;
}

ThreadSP DecodedThread::GetThread() { return m_thread_sp; }

DecodedThread::TraceItemStorage &
DecodedThread::CreateNewTraceItem(lldb::TraceItemKind kind) {
  m_item_kinds.push_back(kind);
  m_item_data.emplace_back();
  return m_item_data.back();
}

void DecodedThread::NotifyTsc(uint64_t tsc) {
  if (!m_last_tsc || *m_last_tsc != tsc) {
    m_timestamps.emplace(m_item_kinds.size(), tsc);
    m_last_tsc = tsc;
  }
}

void DecodedThread::NotifyCPU(lldb::cpu_id_t cpu_id) {
  if (!m_last_cpu || *m_last_cpu != cpu_id) {
    m_cpus.emplace(m_item_kinds.size(), cpu_id);
    m_last_cpu = cpu_id;
    AppendEvent(lldb::eTraceEventCPUChanged);
  }
}

Optional<lldb::cpu_id_t>
DecodedThread::GetCPUByIndex(uint64_t insn_index) const {
  // Could possibly optimize the search
  auto it = m_cpus.upper_bound(insn_index);
  if (it == m_cpus.begin())
    return None;
  return prev(it)->second;
}

void DecodedThread::AppendEvent(lldb::TraceEvent event) {
  CreateNewTraceItem(lldb::eTraceItemKindEvent).event = event;
  m_events_stats.RecordEvent(event);
}

void DecodedThread::AppendInstruction(const pt_insn &insn) {
  CreateNewTraceItem(lldb::eTraceItemKindInstruction).load_address = insn.ip;
}

void DecodedThread::AppendError(const IntelPTError &error) {
  // End of stream shouldn't be a public error
  if (IsEndOfStream(error.GetLibiptErrorCode()))
    return;
  CreateNewTraceItem(lldb::eTraceItemKindError).error =
      ConstString(error.message()).AsCString();
}

void DecodedThread::AppendCustomError(StringRef err) {
  CreateNewTraceItem(lldb::eTraceItemKindError).error =
      ConstString(err).AsCString();
}

lldb::TraceEvent DecodedThread::GetEventByIndex(int item_index) const {
  return m_item_data[item_index].event;
}

void DecodedThread::LibiptErrorsStats::RecordError(int libipt_error_code) {
  libipt_errors_counts[pt_errstr(pt_errcode(libipt_error_code))]++;
  total_count++;
}

void DecodedThread::RecordTscError(int libipt_error_code) {
  m_tsc_errors_stats.RecordError(libipt_error_code);
}

const DecodedThread::LibiptErrorsStats &
DecodedThread::GetTscErrorsStats() const {
  return m_tsc_errors_stats;
}

const DecodedThread::EventsStats &DecodedThread::GetEventsStats() const {
  return m_events_stats;
}

void DecodedThread::EventsStats::RecordEvent(lldb::TraceEvent event) {
  events_counts[event]++;
  total_count++;
}

Optional<DecodedThread::TscRange> DecodedThread::CalculateTscRange(
    size_t insn_index,
    const Optional<DecodedThread::TscRange> &hint_range) const {
  // We first try to check the given hint range in case we are traversing the
  // trace in short jumps. If that fails, then we do the more expensive
  // arbitrary lookup.
  if (hint_range) {
    Optional<TscRange> candidate_range;
    if (insn_index < hint_range->GetStartInstructionIndex())
      candidate_range = hint_range->Prev();
    else if (insn_index > hint_range->GetEndInstructionIndex())
      candidate_range = hint_range->Next();
    else
      candidate_range = hint_range;

    if (candidate_range && candidate_range->InRange(insn_index))
      return candidate_range;
  }
  // Now we do a more expensive lookup
  auto it = m_timestamps.upper_bound(insn_index);
  if (it == m_timestamps.begin())
    return None;

  return TscRange(--it, *this);
}

lldb::TraceItemKind DecodedThread::GetItemKindByIndex(size_t item_index) const {
  return static_cast<lldb::TraceItemKind>(m_item_kinds[item_index]);
}

const char *DecodedThread::GetErrorByIndex(size_t item_index) const {
  return m_item_data[item_index].error;
}

DecodedThread::DecodedThread(ThreadSP thread_sp) : m_thread_sp(thread_sp) {}

lldb::TraceCursorUP DecodedThread::CreateNewCursor() {
  return std::make_unique<TraceCursorIntelPT>(m_thread_sp, shared_from_this());
}

size_t DecodedThread::CalculateApproximateMemoryUsage() const {
  return sizeof(TraceItemStorage) * m_item_data.size() +
         sizeof(uint8_t) * m_item_kinds.size() +
         (sizeof(size_t) + sizeof(uint64_t)) * m_timestamps.size() +
         (sizeof(size_t) + sizeof(lldb::cpu_id_t)) * m_cpus.size();
}

DecodedThread::TscRange::TscRange(std::map<size_t, uint64_t>::const_iterator it,
                                  const DecodedThread &decoded_thread)
    : m_it(it), m_decoded_thread(&decoded_thread) {
  auto next_it = m_it;
  ++next_it;
  m_end_index = (next_it == m_decoded_thread->m_timestamps.end())
                    ? std::numeric_limits<uint64_t>::max()
                    : next_it->first - 1;
}

size_t DecodedThread::TscRange::GetTsc() const { return m_it->second; }

size_t DecodedThread::TscRange::GetStartInstructionIndex() const {
  return m_it->first;
}

size_t DecodedThread::TscRange::GetEndInstructionIndex() const {
  return m_end_index;
}

bool DecodedThread::TscRange::InRange(size_t insn_index) const {
  return GetStartInstructionIndex() <= insn_index &&
         insn_index <= GetEndInstructionIndex();
}

Optional<DecodedThread::TscRange> DecodedThread::TscRange::Next() const {
  auto next_it = m_it;
  ++next_it;
  if (next_it == m_decoded_thread->m_timestamps.end())
    return None;
  return TscRange(next_it, *m_decoded_thread);
}

Optional<DecodedThread::TscRange> DecodedThread::TscRange::Prev() const {
  if (m_it == m_decoded_thread->m_timestamps.begin())
    return None;
  auto prev_it = m_it;
  --prev_it;
  return TscRange(prev_it, *m_decoded_thread);
}
