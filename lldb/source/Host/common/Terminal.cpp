//===-- Terminal.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Terminal.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/PosixApi.h"
#include "llvm/ADT/STLExtras.h"

#include <csignal>
#include <fcntl.h>

#if LLDB_ENABLE_TERMIOS
#include <termios.h>
#endif

using namespace lldb_private;

struct Terminal::Data {
#if LLDB_ENABLE_TERMIOS
  struct termios m_termios; ///< Cached terminal state information.
#endif
};

bool Terminal::IsATerminal() const { return m_fd >= 0 && ::isatty(m_fd); }

llvm::Expected<Terminal::Data> Terminal::GetData() {
  if (!FileDescriptorIsValid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid fd");

#if LLDB_ENABLE_TERMIOS
  if (!IsATerminal())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "fd not a terminal");

  Data data;
  if (::tcgetattr(m_fd, &data.m_termios) != 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "unable to get teletype attributes");
  return data;
#else // !LLDB_ENABLE_TERMIOS
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "termios support missing in LLDB");
#endif // LLDB_ENABLE_TERMIOS
}

llvm::Error Terminal::SetData(const Terminal::Data &data) {
#if LLDB_ENABLE_TERMIOS
  assert(FileDescriptorIsValid());
  assert(IsATerminal());

  if (::tcsetattr(m_fd, TCSANOW, &data.m_termios) != 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "unable to set teletype attributes");
  return llvm::Error::success();
#else // !LLDB_ENABLE_TERMIOS
  llvm_unreachable("SetData() should not be called if !LLDB_ENABLE_TERMIOS");
#endif // LLDB_ENABLE_TERMIOS
}

llvm::Error Terminal::SetEcho(bool enabled) {
  llvm::Expected<Data> data = GetData();
  if (!data)
    return data.takeError();

#if LLDB_ENABLE_TERMIOS
  struct termios &fd_termios = data->m_termios;
  fd_termios.c_lflag &= ~ECHO;
  if (enabled)
    fd_termios.c_lflag |= ECHO;
  return SetData(data.get());
#endif // LLDB_ENABLE_TERMIOS
}

llvm::Error Terminal::SetCanonical(bool enabled) {
  llvm::Expected<Data> data = GetData();
  if (!data)
    return data.takeError();

#if LLDB_ENABLE_TERMIOS
  struct termios &fd_termios = data->m_termios;
  fd_termios.c_lflag &= ~ICANON;
  if (enabled)
    fd_termios.c_lflag |= ICANON;
  return SetData(data.get());
#endif // LLDB_ENABLE_TERMIOS
}

TerminalState::TerminalState(Terminal term, bool save_process_group)
    : m_tty(term) {
  Save(term, save_process_group);
}

TerminalState::~TerminalState() { Restore(); }

void TerminalState::Clear() {
  m_tty.Clear();
  m_tflags = -1;
  m_data.reset();
  m_process_group = -1;
}

bool TerminalState::Save(Terminal term, bool save_process_group) {
  Clear();
  m_tty = term;
  if (m_tty.IsATerminal()) {
    int fd = m_tty.GetFileDescriptor();
#if LLDB_ENABLE_POSIX
    m_tflags = ::fcntl(fd, F_GETFL, 0);
#if LLDB_ENABLE_TERMIOS
    std::unique_ptr<Terminal::Data> new_data{new Terminal::Data()};
    if (::tcgetattr(fd, &new_data->m_termios) == 0)
      m_data = std::move(new_data);
#endif // LLDB_ENABLE_TERMIOS
    if (save_process_group)
      m_process_group = ::tcgetpgrp(fd);
#endif // LLDB_ENABLE_POSIX
  }
  return IsValid();
}

bool TerminalState::Restore() const {
#if LLDB_ENABLE_POSIX
  if (IsValid()) {
    const int fd = m_tty.GetFileDescriptor();
    if (TFlagsIsValid())
      fcntl(fd, F_SETFL, m_tflags);

#if LLDB_ENABLE_TERMIOS
    if (TTYStateIsValid())
      tcsetattr(fd, TCSANOW, &m_data->m_termios);
#endif // LLDB_ENABLE_TERMIOS

    if (ProcessGroupIsValid()) {
      // Save the original signal handler.
      void (*saved_sigttou_callback)(int) = nullptr;
      saved_sigttou_callback = (void (*)(int))signal(SIGTTOU, SIG_IGN);
      // Set the process group
      tcsetpgrp(fd, m_process_group);
      // Restore the original signal handler.
      signal(SIGTTOU, saved_sigttou_callback);
    }
    return true;
  }
#endif // LLDB_ENABLE_POSIX
  return false;
}

bool TerminalState::IsValid() const {
  return m_tty.FileDescriptorIsValid() &&
         (TFlagsIsValid() || TTYStateIsValid() || ProcessGroupIsValid());
}

bool TerminalState::TFlagsIsValid() const { return m_tflags != -1; }

bool TerminalState::TTYStateIsValid() const { return bool(m_data); }

bool TerminalState::ProcessGroupIsValid() const {
  return static_cast<int32_t>(m_process_group) != -1;
}
