//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ConnectionConPTYWindows.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

/// Strips the ConPTY initialization sequences that Windows unconditionally
/// emits when a process is first attached to a pseudo console.
///
/// These are emitted by ConPTY's host process (conhost.exe) at process attach
/// time, not by the debuggee. They are always the first bytes on the output
/// pipe and are always present as a contiguous prefix.
///
/// \param dst  Buffer containing the data read from the ConPTY output pipe.
///             Modified in place: if the initialization sequences are present
///             as a prefix, they are removed by shifting the remaining bytes
///             to the front of the buffer.
/// \param len  On input, the number of valid bytes in \p dst. On output,
///             reduced by the number of bytes stripped.
/// \return
///     \p true if the sequence was found and stripped.
static bool StripConPTYInitSequences(void *dst, size_t &len) {
  static const char sequences[] = "\x1b[?9001l\x1b[?1004l";
  static const size_t sequences_len = sizeof(sequences) - 1;

  char *buf = static_cast<char *>(dst);
  if (len >= sequences_len && memcmp(buf, sequences, sequences_len) == 0) {
    memmove(buf, buf + sequences_len, len - sequences_len);
    len -= sequences_len;
    return true;
  }
  return false;
}

ConnectionConPTY::ConnectionConPTY(std::shared_ptr<PseudoConsole> pty)
    : m_pty(pty), ConnectionGenericFile(pty->GetSTDOUTHandle(), false) {};

ConnectionConPTY::~ConnectionConPTY() {}

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
  std::unique_lock<std::mutex> guard(m_pty->GetMutex());
  if (m_pty->IsStopping().load()) {
    m_pty->GetCV().wait(guard, [this] { return !m_pty->IsStopping().load(); });
  }

  size_t bytes_read =
      ConnectionGenericFile::Read(dst, dst_len, timeout, status, error_ptr);

  if (bytes_read > 0 && !m_pty_vt_sequence_was_stripped) {
    if (StripConPTYInitSequences(dst, bytes_read))
      m_pty_vt_sequence_was_stripped = true;
  }

  return bytes_read;
}

size_t ConnectionConPTY::Write(const void *src, size_t src_len,
                               lldb::ConnectionStatus &status,
                               Status *error_ptr) {
  llvm_unreachable("not implemented");
}
