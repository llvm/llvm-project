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

// Default constructor
TerminalState::TerminalState()
    : m_tty()
#if LLDB_ENABLE_TERMIOS
      ,
      m_termios_up()
#endif
{
}

// Destructor
TerminalState::~TerminalState() = default;

void TerminalState::Clear() {
  m_tty.Clear();
  m_tflags = -1;
#if LLDB_ENABLE_TERMIOS
  m_termios_up.reset();
#endif
  m_process_group = -1;
}

// Save the current state of the TTY for the file descriptor "fd" and if
// "save_process_group" is true, attempt to save the process group info for the
// TTY.
bool TerminalState::Save(int fd, bool save_process_group) {
  m_tty.SetFileDescriptor(fd);
  if (m_tty.IsATerminal()) {
#if LLDB_ENABLE_POSIX
    m_tflags = ::fcntl(fd, F_GETFL, 0);
#endif
#if LLDB_ENABLE_TERMIOS
    if (m_termios_up == nullptr)
      m_termios_up.reset(new struct termios);
    int err = ::tcgetattr(fd, m_termios_up.get());
    if (err != 0)
      m_termios_up.reset();
#endif // #if LLDB_ENABLE_TERMIOS
#if LLDB_ENABLE_POSIX
    if (save_process_group)
      m_process_group = ::tcgetpgrp(0);
    else
      m_process_group = -1;
#endif
  } else {
    m_tty.Clear();
    m_tflags = -1;
#if LLDB_ENABLE_TERMIOS
    m_termios_up.reset();
#endif
    m_process_group = -1;
  }
  return IsValid();
}

// Restore the state of the TTY using the cached values from a previous call to
// Save().
bool TerminalState::Restore() const {
#if LLDB_ENABLE_POSIX
  if (IsValid()) {
    const int fd = m_tty.GetFileDescriptor();
    if (TFlagsIsValid())
      fcntl(fd, F_SETFL, m_tflags);

#if LLDB_ENABLE_TERMIOS
    if (TTYStateIsValid())
      tcsetattr(fd, TCSANOW, m_termios_up.get());
#endif // #if LLDB_ENABLE_TERMIOS

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
#endif
  return false;
}

// Returns true if this object has valid saved TTY state settings that can be
// used to restore a previous state.
bool TerminalState::IsValid() const {
  return m_tty.FileDescriptorIsValid() &&
         (TFlagsIsValid() || TTYStateIsValid());
}

// Returns true if m_tflags is valid
bool TerminalState::TFlagsIsValid() const { return m_tflags != -1; }

// Returns true if m_ttystate is valid
bool TerminalState::TTYStateIsValid() const {
#if LLDB_ENABLE_TERMIOS
  return m_termios_up != nullptr;
#else
  return false;
#endif
}

// Returns true if m_process_group is valid
bool TerminalState::ProcessGroupIsValid() const {
  return static_cast<int32_t>(m_process_group) != -1;
}

// Constructor
TerminalStateSwitcher::TerminalStateSwitcher() = default;

// Destructor
TerminalStateSwitcher::~TerminalStateSwitcher() = default;

// Returns the number of states that this switcher contains
uint32_t TerminalStateSwitcher::GetNumberOfStates() const {
  return llvm::array_lengthof(m_ttystates);
}

// Restore the state at index "idx".
//
// Returns true if the restore was successful, false otherwise.
bool TerminalStateSwitcher::Restore(uint32_t idx) const {
  const uint32_t num_states = GetNumberOfStates();
  if (idx >= num_states)
    return false;

  // See if we already are in this state?
  if (m_currentState < num_states && (idx == m_currentState) &&
      m_ttystates[idx].IsValid())
    return true;

  // Set the state to match the index passed in and only update the current
  // state if there are no errors.
  if (m_ttystates[idx].Restore()) {
    m_currentState = idx;
    return true;
  }

  // We failed to set the state. The tty state was invalid or not initialized.
  return false;
}

// Save the state at index "idx" for file descriptor "fd" and save the process
// group if requested.
//
// Returns true if the restore was successful, false otherwise.
bool TerminalStateSwitcher::Save(uint32_t idx, int fd,
                                 bool save_process_group) {
  const uint32_t num_states = GetNumberOfStates();
  if (idx < num_states)
    return m_ttystates[idx].Save(fd, save_process_group);
  return false;
}
