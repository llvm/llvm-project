//===-- DecodedThread.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DecodedThread.h"

#include <intel-pt.h>
#include <memory>

#include "TraceCursorIntelPT.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

char IntelPTError::ID;

IntelPTError::IntelPTError(int libipt_error_code, lldb::addr_t address)
    : m_libipt_error_code(libipt_error_code), m_address(address) {
  assert(libipt_error_code < 0);
}

void IntelPTError::log(llvm::raw_ostream &OS) const {
  const char *libipt_error_message = pt_errstr(pt_errcode(m_libipt_error_code));
  if (m_address != LLDB_INVALID_ADDRESS && m_address > 0) {
    write_hex(OS, m_address, HexPrintStyle::PrefixLower, 18);
    OS << "    ";
  }
  OS << "error: " << libipt_error_message;
}

Optional<size_t> DecodedThread::GetRawTraceSize() const {
  return m_raw_trace_size;
}

size_t DecodedThread::GetInstructionsCount() const {
  return m_instruction_ips.size();
}

lldb::addr_t DecodedThread::GetInstructionLoadAddress(size_t insn_index) const {
  return m_instruction_ips[insn_index];
}

TraceInstructionControlFlowType
DecodedThread::GetInstructionControlFlowType(size_t insn_index) const {
  if (IsInstructionAnError(insn_index))
    return (TraceInstructionControlFlowType)0;

  TraceInstructionControlFlowType mask =
      eTraceInstructionControlFlowTypeInstruction;

  lldb::addr_t load_address = m_instruction_ips[insn_index];
  uint8_t insn_byte_size = m_instruction_sizes[insn_index];
  pt_insn_class iclass = m_instruction_classes[insn_index];

  switch (iclass) {
  case ptic_cond_jump:
  case ptic_jump:
  case ptic_far_jump:
    mask |= eTraceInstructionControlFlowTypeBranch;
    if (insn_index + 1 < m_instruction_ips.size() &&
        load_address + insn_byte_size != m_instruction_ips[insn_index + 1])
      mask |= eTraceInstructionControlFlowTypeTakenBranch;
    break;
  case ptic_return:
  case ptic_far_return:
    mask |= eTraceInstructionControlFlowTypeReturn;
    break;
  case ptic_call:
  case ptic_far_call:
    mask |= eTraceInstructionControlFlowTypeCall;
    break;
  default:
    break;
  }

  return mask;
}

ThreadSP DecodedThread::GetThread() { return m_thread_sp; }

void DecodedThread::RecordTscForLastInstruction(uint64_t tsc) {
  if (!m_last_tsc || *m_last_tsc != tsc) {
    // In case the first instructions are errors or did not have a TSC, we'll
    // get a first valid TSC not in position 0. We can safely force these error
    // instructions to use the first valid TSC, so that all the trace has TSCs.
    size_t start_index =
        m_instruction_timestamps.empty() ? 0 : m_instruction_ips.size() - 1;
    m_instruction_timestamps.emplace(start_index, tsc);
    m_last_tsc = tsc;
  }
}

void DecodedThread::AppendInstruction(const pt_insn &insn) {
  m_instruction_ips.emplace_back(insn.ip);
  m_instruction_sizes.emplace_back(insn.size);
  m_instruction_classes.emplace_back(insn.iclass);
}

void DecodedThread::AppendInstruction(const pt_insn &insn, uint64_t tsc) {
  AppendInstruction(insn);
  RecordTscForLastInstruction(tsc);
}

void DecodedThread::AppendError(llvm::Error &&error) {
  m_errors.try_emplace(m_instruction_ips.size(), toString(std::move(error)));
  m_instruction_ips.emplace_back(LLDB_INVALID_ADDRESS);
  m_instruction_sizes.emplace_back(0);
  m_instruction_classes.emplace_back(pt_insn_class::ptic_error);
}

void DecodedThread::AppendError(llvm::Error &&error, uint64_t tsc) {
  AppendError(std::move(error));
  RecordTscForLastInstruction(tsc);
}

void DecodedThread::LibiptErrors::RecordError(int libipt_error_code) {
  libipt_errors[pt_errstr(pt_errcode(libipt_error_code))]++;
  total_count++;
}

void DecodedThread::RecordTscError(int libipt_error_code) {
  m_tsc_errors.RecordError(libipt_error_code);
}

const DecodedThread::LibiptErrors &DecodedThread::GetTscErrors() const {
  return m_tsc_errors;
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
  auto it = m_instruction_timestamps.upper_bound(insn_index);
  if (it == m_instruction_timestamps.begin())
    return None;

  return TscRange(--it, *this);
}

bool DecodedThread::IsInstructionAnError(size_t insn_idx) const {
  return m_instruction_ips[insn_idx] == LLDB_INVALID_ADDRESS;
}

const char *DecodedThread::GetErrorByInstructionIndex(size_t insn_idx) {
  auto it = m_errors.find(insn_idx);
  if (it == m_errors.end())
    return nullptr;

  return it->second.c_str();
}

DecodedThread::DecodedThread(ThreadSP thread_sp) : m_thread_sp(thread_sp) {}

DecodedThread::DecodedThread(ThreadSP thread_sp, Error &&error)
    : m_thread_sp(thread_sp) {
  AppendError(std::move(error));
}

void DecodedThread::SetRawTraceSize(size_t size) { m_raw_trace_size = size; }

lldb::TraceCursorUP DecodedThread::GetCursor() {
  // We insert a fake error signaling an empty trace if needed becasue the
  // TraceCursor requires non-empty traces.
  if (m_instruction_ips.empty())
    AppendError(createStringError(inconvertibleErrorCode(), "empty trace"));
  return std::make_unique<TraceCursorIntelPT>(m_thread_sp, shared_from_this());
}

size_t DecodedThread::CalculateApproximateMemoryUsage() const {
  return sizeof(pt_insn::ip) * m_instruction_ips.size() +
         sizeof(pt_insn::size) * m_instruction_sizes.size() +
         sizeof(pt_insn::iclass) * m_instruction_classes.size() +
         (sizeof(size_t) + sizeof(uint64_t)) * m_instruction_timestamps.size() +
         m_errors.getMemorySize();
}

DecodedThread::TscRange::TscRange(std::map<size_t, uint64_t>::const_iterator it,
                                  const DecodedThread &decoded_thread)
    : m_it(it), m_decoded_thread(&decoded_thread) {
  auto next_it = m_it;
  ++next_it;
  m_end_index = (next_it == m_decoded_thread->m_instruction_timestamps.end())
                    ? m_decoded_thread->GetInstructionsCount() - 1
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
  if (next_it == m_decoded_thread->m_instruction_timestamps.end())
    return None;
  return TscRange(next_it, *m_decoded_thread);
}

Optional<DecodedThread::TscRange> DecodedThread::TscRange::Prev() const {
  if (m_it == m_decoded_thread->m_instruction_timestamps.begin())
    return None;
  auto prev_it = m_it;
  --prev_it;
  return TscRange(prev_it, *m_decoded_thread);
}
