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

#include <cstring>

using namespace lldb;
using namespace lldb_private;

/// Remove ConPTY management sequences from a buffer in-place.
///
/// ConPTY injects several VT sequences into its output pipe that are not part
/// of the inferior's output: a cursor-position query (\x1b[6n) emitted during
/// PSEUDOCONSOLE_INHERIT_CURSOR initialisation, Win32 Input Mode toggles
/// (\x1b[?9001h/l), focus-event toggles (\x1b[?1004h/l), and a window-title
/// OSC sequence (\x1b]0;...\x07). These sequences must not reach the outer
/// terminal.
///
/// \param[in,out] data  Buffer containing raw ConPTY output.
/// \param[in,out] len   On entry, the number of valid bytes in \p data.
///                      Updated to the number of bytes after stripping.
/// \param[in] strip_init  If true, also strip init-only sequences (\x1b[m,
///                        \x1b[?25h) that ConPTY emits at startup.
static void StripConPTYSequences(void *data, size_t &len, bool strip_init) {
  auto *buf = static_cast<char *>(data);
  char *out = buf;
  const char *in = buf;
  const char *end = buf + len;

  while (in < end) {
    if (*in != '\x1b') {
      *out++ = *in++;
      continue;
    }

    size_t remaining = end - in;

    // \x1b[6n - cursor-position query (PSEUDOCONSOLE_INHERIT_CURSOR init)
    // This query is always replied to in OpenPseudoConsole.
    if (remaining >= 4 && memcmp(in, "\x1b[6n", 4) == 0) {
      in += 4;
      continue;
    }

    if (strip_init) {
      // \x1b[m - SGR reset (ConPTY init)
      if (remaining >= 3 && memcmp(in, "\x1b[m", 3) == 0) {
        in += 3;
        continue;
      }

      // \x1b[?25h - show cursor (ConPTY init)
      if (remaining >= 6 && memcmp(in, "\x1b[?25h", 6) == 0) {
        in += 6;
        continue;
      }
    }

    // \x1b[?9001h / \x1b[?9001l - Win32 Input Mode enable/disable
    if (remaining >= 8 && memcmp(in, "\x1b[?9001", 7) == 0 &&
        (in[7] == 'h' || in[7] == 'l')) {
      in += 8;
      continue;
    }

    // \x1b[?1004h / \x1b[?1004l - focus-event reporting enable/disable
    if (remaining >= 8 && memcmp(in, "\x1b[?1004", 7) == 0 &&
        (in[7] == 'h' || in[7] == 'l')) {
      in += 8;
      continue;
    }

    // \x1b]0;...\x07 - ConPTY window-title OSC sequence
    if (remaining >= 4 && in[1] == ']' && in[2] == '0' && in[3] == ';') {
      const char *bel =
          static_cast<const char *>(memchr(in + 4, '\x07', end - in - 4));
      // We assume a sequence is not split accross multiple chunks.
      if (bel)
        in = bel + 1;
      else
        in = end;
      continue;
    }

    *out++ = *in++;
  }

  len = static_cast<size_t>(out - buf);
}

ConnectionConPTY::ConnectionConPTY(std::shared_ptr<PseudoConsole> pty)
    : ConnectionGenericFile(pty->GetSTDOUTHandle(), false), m_pty(pty) {}

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
  llvm_unreachable("not implemented");
}
