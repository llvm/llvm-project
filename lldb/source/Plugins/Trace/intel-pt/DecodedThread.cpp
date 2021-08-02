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

IntelPTInstruction::IntelPTInstruction(llvm::Error err) {
  llvm::handleAllErrors(std::move(err),
                        [&](std::unique_ptr<llvm::ErrorInfoBase> info) {
                          m_error = std::move(info);
                        });
  m_pt_insn.ip = LLDB_INVALID_ADDRESS;
  m_pt_insn.iclass = ptic_error;
}

bool IntelPTInstruction::IsError() const { return (bool)m_error; }

lldb::addr_t IntelPTInstruction::GetLoadAddress() const { return m_pt_insn.ip; }

Optional<uint64_t> IntelPTInstruction::GetTimestampCounter() const {
  return m_timestamp;
}

Error IntelPTInstruction::ToError() const {
  if (!IsError())
    return Error::success();

  if (m_error->isA<IntelPTError>())
    return make_error<IntelPTError>(static_cast<IntelPTError &>(*m_error));
  return make_error<StringError>(m_error->message(),
                                 m_error->convertToErrorCode());
}
size_t DecodedThread::GetRawTraceSize() const { return m_raw_trace_size; }

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

ArrayRef<IntelPTInstruction> DecodedThread::GetInstructions() const {
  return makeArrayRef(m_instructions);
}

DecodedThread::DecodedThread(ThreadSP thread_sp, Error error)
    : m_thread_sp(thread_sp) {
  m_instructions.emplace_back(std::move(error));
}

DecodedThread::DecodedThread(ThreadSP thread_sp,
                             std::vector<IntelPTInstruction> &&instructions,
                             size_t raw_trace_size)
    : m_thread_sp(thread_sp), m_instructions(std::move(instructions)),
      m_raw_trace_size(raw_trace_size) {
  if (m_instructions.empty())
    m_instructions.emplace_back(
        createStringError(inconvertibleErrorCode(), "empty trace"));
}

lldb::TraceCursorUP DecodedThread::GetCursor() {
  return std::make_unique<TraceCursorIntelPT>(m_thread_sp, shared_from_this());
}
