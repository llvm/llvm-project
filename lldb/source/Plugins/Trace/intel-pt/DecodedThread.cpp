//===-- DecodedThread.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DecodedThread.h"

#include "lldb/Utility/StreamString.h"

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

bool IntelPTInstruction::IsError() const { return (bool)m_error; }

Expected<lldb::addr_t> IntelPTInstruction::GetLoadAddress() const {
  if (IsError())
    return ToError();
  return m_pt_insn.ip;
}

Error IntelPTInstruction::ToError() const {
  if (!IsError())
    return Error::success();

  if (m_error->isA<IntelPTError>())
    return make_error<IntelPTError>(static_cast<IntelPTError &>(*m_error));
  return make_error<StringError>(m_error->message(),
                                 m_error->convertToErrorCode());
}

size_t DecodedThread::GetLastPosition() const {
  return m_instructions.empty() ? 0 : m_instructions.size() - 1;
}

ArrayRef<IntelPTInstruction> DecodedThread::GetInstructions() const {
  return makeArrayRef(m_instructions);
}

size_t DecodedThread::GetCursorPosition() const { return m_position; }

size_t DecodedThread::SetCursorPosition(size_t new_position) {
  m_position = std::min(new_position, GetLastPosition());
  return m_position;
}

DecodedThread::DecodedThread(Error error) {
  m_instructions.emplace_back(std::move(error));
  m_position = GetLastPosition();
}
