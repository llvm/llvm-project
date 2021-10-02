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

bool Terminal::IsATerminal() const { return m_fd >= 0 && ::isatty(m_fd); }

bool Terminal::SetEcho(bool enabled) {
  if (FileDescriptorIsValid()) {
#if LLDB_ENABLE_TERMIOS
    if (IsATerminal()) {
      struct termios fd_termios;
      if (::tcgetattr(m_fd, &fd_termios) == 0) {
        bool set_corectly = false;
        if (enabled) {
          if (fd_termios.c_lflag & ECHO)
            set_corectly = true;
          else
            fd_termios.c_lflag |= ECHO;
        } else {
          if (fd_termios.c_lflag & ECHO)
            fd_termios.c_lflag &= ~ECHO;
          else
            set_corectly = true;
        }

        if (set_corectly)
          return true;
        return ::tcsetattr(m_fd, TCSANOW, &fd_termios) == 0;
      }
    }
#endif // #if LLDB_ENABLE_TERMIOS
  }
  return false;
}

bool Terminal::SetCanonical(bool enabled) {
  if (FileDescriptorIsValid()) {
#if LLDB_ENABLE_TERMIOS
    if (IsATerminal()) {
      struct termios fd_termios;
      if (::tcgetattr(m_fd, &fd_termios) == 0) {
        bool set_corectly = false;
        if (enabled) {
          if (fd_termios.c_lflag & ICANON)
            set_corectly = true;
          else
            fd_termios.c_lflag |= ICANON;
        } else {
          if (fd_termios.c_lflag & ICANON)
            fd_termios.c_lflag &= ~ICANON;
          else
            set_corectly = true;
        }

        if (set_corectly)
          return true;
        return ::tcsetattr(m_fd, TCSANOW, &fd_termios) == 0;
      }
    }
#endif // #if LLDB_ENABLE_TERMIOS
  }
  return false;
}

struct TerminalState::Data {
#if LLDB_ENABLE_TERMIOS
  struct termios m_termios; ///< Cached terminal state information.
#endif
};

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
    std::unique_ptr<Data> new_data{new Data()};
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
