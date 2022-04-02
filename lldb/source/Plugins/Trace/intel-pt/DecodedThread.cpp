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

IntelPTInstruction::IntelPTInstruction() {
  m_pt_insn.ip = LLDB_INVALID_ADDRESS;
  m_pt_insn.iclass = ptic_error;
  m_is_error = true;
}

bool IntelPTInstruction::IsError() const { return m_is_error; }

lldb::addr_t IntelPTInstruction::GetLoadAddress() const { return m_pt_insn.ip; }

size_t IntelPTInstruction::GetMemoryUsage() {
  return sizeof(IntelPTInstruction);
}

Optional<size_t> DecodedThread::GetRawTraceSize() const {
  return m_raw_trace_size;
}

TraceInstructionControlFlowType
IntelPTInstruction::GetControlFlowType(lldb::addr_t next_load_address) const {
  if (IsError())
    return (TraceInstructionControlFlowType)0;

  TraceInstructionControlFlowType mask =
      eTraceInstructionControlFlowTypeInstruction;

  switch (m_pt_insn.iclass) {
  case ptic_cond_jump:
  case ptic_jump:
  case ptic_far_jump:
    mask |= eTraceInstructionControlFlowTypeBranch;
    if (m_pt_insn.ip + m_pt_insn.size != next_load_address)
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

void DecodedThread::AppendInstruction(const pt_insn &insn) {
  m_instructions.emplace_back(insn);
}

void DecodedThread::AppendInstruction(const pt_insn &insn, uint64_t tsc) {
  m_instructions.emplace_back(insn);
  if (!m_last_tsc || *m_last_tsc != tsc) {
    // In case the first instructions are errors or did not have a TSC, we'll
    // get a first valid TSC not in position 0. We can safely force these error
    // instructions to use the first valid TSC, so that all the trace has TSCs.
    size_t start_index =
        m_instruction_timestamps.empty() ? 0 : m_instructions.size() - 1;
    m_instruction_timestamps.emplace(start_index, tsc);
    m_last_tsc = tsc;
  }
}

void DecodedThread::AppendError(llvm::Error &&error) {
  m_errors.try_emplace(m_instructions.size(), toString(std::move(error)));
  m_instructions.emplace_back();
}

ArrayRef<IntelPTInstruction> DecodedThread::GetInstructions() const {
  return makeArrayRef(m_instructions);
}

Optional<DecodedThread::TscRange>
DecodedThread::CalculateTscRange(size_t insn_index) const {
  auto it = m_instruction_timestamps.upper_bound(insn_index);
  if (it == m_instruction_timestamps.begin())
    return None;

  return TscRange(--it, *this);
}

bool DecodedThread::IsInstructionAnError(size_t insn_idx) const {
  return m_instructions[insn_idx].IsError();
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
  if (m_instructions.empty())
    AppendError(createStringError(inconvertibleErrorCode(), "empty trace"));
  return std::make_unique<TraceCursorIntelPT>(m_thread_sp, shared_from_this());
}

size_t DecodedThread::CalculateApproximateMemoryUsage() const {
  return IntelPTInstruction::GetMemoryUsage() * m_instructions.size() +
         m_errors.getMemorySize();
}

DecodedThread::TscRange::TscRange(std::map<size_t, uint64_t>::const_iterator it,
                                  const DecodedThread &decoded_thread)
    : m_it(it), m_decoded_thread(&decoded_thread) {
  auto next_it = m_it;
  ++next_it;
  m_end_index = (next_it == m_decoded_thread->m_instruction_timestamps.end())
                    ? m_decoded_thread->GetInstructions().size() - 1
                    : next_it->first - 1;
}

size_t DecodedThread::TscRange::GetTsc() const { return m_it->second; }

size_t DecodedThread::TscRange::GetStartInstructionIndex() const {
  return m_it->first;
}

size_t DecodedThread::TscRange::GetEndInstructionIndex() const {
  return m_end_index;
}

bool DecodedThread::TscRange::InRange(size_t insn_index) {
  return GetStartInstructionIndex() <= insn_index &&
         insn_index <= GetEndInstructionIndex();
}

Optional<DecodedThread::TscRange> DecodedThread::TscRange::Next() {
  auto next_it = m_it;
  ++next_it;
  if (next_it == m_decoded_thread->m_instruction_timestamps.end())
    return None;
  return TscRange(next_it, *m_decoded_thread);
}

Optional<DecodedThread::TscRange> DecodedThread::TscRange::Prev() {
  if (m_it == m_decoded_thread->m_instruction_timestamps.begin())
    return None;
  auto prev_it = m_it;
  --prev_it;
  return TscRange(prev_it, *m_decoded_thread);
}