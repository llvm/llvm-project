//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ConnectionConPTYWindows.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timeout.h"

using namespace lldb;
using namespace lldb_private;

ConnectionConPTY::ConnectionConPTY(std::shared_ptr<PseudoConsole> pty)
    : ConnectionGenericFile(pty->GetSTDOUTHandle(), false), m_pty(pty) {}

ConnectionConPTY::~ConnectionConPTY() = default;

lldb::ConnectionStatus ConnectionConPTY::Connect(llvm::StringRef s,
                                                 Status *error_ptr) {
  if (m_pty->IsConnected())
    return eConnectionStatusSuccess;
  return eConnectionStatusNoConnection;
}

lldb::ConnectionStatus ConnectionConPTY::Disconnect(Status *error_ptr) {
  m_pty->Close();
  return eConnectionStatusSuccess;
}

size_t ConnectionConPTY::Read(void *dst, size_t dst_len,
                              const Timeout<std::micro> &timeout,
                              lldb::ConnectionStatus &status,
                              Status *error_ptr) {
  {
    std::unique_lock<std::mutex> guard(m_pty->GetMutex());
    if (m_pty->IsStopping())
      m_pty->GetCV().wait(guard, [this] { return !m_pty->IsStopping(); });
    if (!m_pty->IsConnected()) {
      status = eConnectionStatusEndOfFile;
      return 0;
    }
  }

  char *out = static_cast<char *>(dst);
  size_t bytes_read =
      ConnectionGenericFile::Read(out, dst_len, timeout, status, error_ptr);

  if (bytes_read > 0) {
    StripConPTYSequences(out, bytes_read, !m_conpty_sequences_stripped);
    m_conpty_sequences_stripped = true;
  }

  return bytes_read;
}

size_t ConnectionConPTY::Write(const void *src, size_t src_len,
                               lldb::ConnectionStatus &status,
                               Status *error_ptr) {
  if (!m_pty || !m_pty->IsConnected()) {
    status = eConnectionStatusNoConnection;
    if (error_ptr)
      *error_ptr = Status::FromErrorString("ConPTY not connected");
    return 0;
  }
  HANDLE stdin_handle = m_pty->GetSTDINHandle();
  if (stdin_handle == INVALID_HANDLE_VALUE || stdin_handle == nullptr) {
    status = eConnectionStatusNoConnection;
    if (error_ptr)
      *error_ptr = Status::FromErrorString("ConPTY STDIN handle is invalid");
    return 0;
  }
  DWORD written = 0;
  if (!::WriteFile(stdin_handle, src, static_cast<DWORD>(src_len), &written,
                   nullptr)) {
    DWORD err = ::GetLastError();
    status = (err == ERROR_BROKEN_PIPE || err == ERROR_NO_DATA)
                 ? eConnectionStatusEndOfFile
                 : eConnectionStatusError;
    if (error_ptr)
      *error_ptr = Status(err, lldb::eErrorTypeWin32);
    return written;
  }
  status = eConnectionStatusSuccess;
  return written;
}
