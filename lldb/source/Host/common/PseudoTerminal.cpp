//===-- PseudoTerminal.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/Config.h"

#include "llvm/Support/Errno.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(TIOCSCTTY)
#include <sys/ioctl.h>
#endif

#include "lldb/Host/PosixApi.h"

#if defined(__ANDROID__)
int posix_openpt(int flags);
#endif

using namespace lldb_private;

// Write string describing error number
static void ErrnoToStr(char *error_str, size_t error_len) {
  std::string strerror = llvm::sys::StrError();
  ::snprintf(error_str, error_len, "%s", strerror.c_str());
}

// PseudoTerminal constructor
PseudoTerminal::PseudoTerminal()
    : m_primary_fd(invalid_fd), m_secondary_fd(invalid_fd) {}

// Destructor
//
// The destructor will close the primary and secondary file descriptors if they
// are valid and ownership has not been released using the
// ReleasePrimaryFileDescriptor() or the ReleaseSaveFileDescriptor() member
// functions.
PseudoTerminal::~PseudoTerminal() {
  ClosePrimaryFileDescriptor();
  CloseSecondaryFileDescriptor();
}

// Close the primary file descriptor if it is valid.
void PseudoTerminal::ClosePrimaryFileDescriptor() {
  if (m_primary_fd >= 0) {
    ::close(m_primary_fd);
    m_primary_fd = invalid_fd;
  }
}

// Close the secondary file descriptor if it is valid.
void PseudoTerminal::CloseSecondaryFileDescriptor() {
  if (m_secondary_fd >= 0) {
    ::close(m_secondary_fd);
    m_secondary_fd = invalid_fd;
  }
}

// Open the first available pseudo terminal with OFLAG as the permissions. The
// file descriptor is stored in this object and can be accessed with the
// PrimaryFileDescriptor() accessor. The ownership of the primary file
// descriptor can be released using the ReleasePrimaryFileDescriptor() accessor.
// If this object has a valid primary files descriptor when its destructor is
// called, it will close the primary file descriptor, therefore clients must
// call ReleasePrimaryFileDescriptor() if they wish to use the primary file
// descriptor after this object is out of scope or destroyed.
//
// RETURNS:
//  True when successful, false indicating an error occurred.
bool PseudoTerminal::OpenFirstAvailablePrimary(int oflag, char *error_str,
                                               size_t error_len) {
  if (error_str)
    error_str[0] = '\0';

#if LLDB_ENABLE_POSIX
  // Open the primary side of a pseudo terminal
  m_primary_fd = ::posix_openpt(oflag);
  if (m_primary_fd < 0) {
    if (error_str)
      ErrnoToStr(error_str, error_len);
    return false;
  }

  // Grant access to the secondary pseudo terminal
  if (::grantpt(m_primary_fd) < 0) {
    if (error_str)
      ErrnoToStr(error_str, error_len);
    ClosePrimaryFileDescriptor();
    return false;
  }

  // Clear the lock flag on the secondary pseudo terminal
  if (::unlockpt(m_primary_fd) < 0) {
    if (error_str)
      ErrnoToStr(error_str, error_len);
    ClosePrimaryFileDescriptor();
    return false;
  }

  return true;
#else
  if (error_str)
    ::snprintf(error_str, error_len, "%s", "pseudo terminal not supported");
  return false;
#endif
}

// Open the secondary pseudo terminal for the current primary pseudo terminal. A
// primary pseudo terminal should already be valid prior to calling this
// function (see OpenFirstAvailablePrimary()). The file descriptor is stored
// this object's member variables and can be accessed via the
// GetSecondaryFileDescriptor(), or released using the
// ReleaseSecondaryFileDescriptor() member function.
//
// RETURNS:
//  True when successful, false indicating an error occurred.
bool PseudoTerminal::OpenSecondary(int oflag, char *error_str,
                                   size_t error_len) {
  if (error_str)
    error_str[0] = '\0';

  CloseSecondaryFileDescriptor();

  // Open the primary side of a pseudo terminal
  const char *secondary_name = GetSecondaryName(error_str, error_len);

  if (secondary_name == nullptr)
    return false;

  m_secondary_fd =
      llvm::sys::RetryAfterSignal(-1, ::open, secondary_name, oflag);

  if (m_secondary_fd < 0) {
    if (error_str)
      ErrnoToStr(error_str, error_len);
    return false;
  }

  return true;
}

// Get the name of the secondary pseudo terminal. A primary pseudo terminal
// should already be valid prior to calling this function (see
// OpenFirstAvailablePrimary()).
//
// RETURNS:
//  NULL if no valid primary pseudo terminal or if ptsname() fails.
//  The name of the secondary pseudo terminal as a NULL terminated C string
//  that comes from static memory, so a copy of the string should be
//  made as subsequent calls can change this value.
const char *PseudoTerminal::GetSecondaryName(char *error_str,
                                             size_t error_len) const {
  if (error_str)
    error_str[0] = '\0';

  if (m_primary_fd < 0) {
    if (error_str)
      ::snprintf(error_str, error_len, "%s",
                 "primary file descriptor is invalid");
    return nullptr;
  }
  const char *secondary_name = ::ptsname(m_primary_fd);

  if (error_str && secondary_name == nullptr)
    ErrnoToStr(error_str, error_len);

  return secondary_name;
}

// Fork a child process and have its stdio routed to a pseudo terminal.
//
// In the parent process when a valid pid is returned, the primary file
// descriptor can be used as a read/write access to stdio of the child process.
//
// In the child process the stdin/stdout/stderr will already be routed to the
// secondary pseudo terminal and the primary file descriptor will be closed as
// it is no longer needed by the child process.
//
// This class will close the file descriptors for the primary/secondary when the
// destructor is called, so be sure to call ReleasePrimaryFileDescriptor() or
// ReleaseSecondaryFileDescriptor() if any file descriptors are going to be used
// past the lifespan of this object.
//
// RETURNS:
//  in the parent process: the pid of the child, or -1 if fork fails
//  in the child process: zero
lldb::pid_t PseudoTerminal::Fork(char *error_str, size_t error_len) {
  if (error_str)
    error_str[0] = '\0';
  pid_t pid = LLDB_INVALID_PROCESS_ID;
#if LLDB_ENABLE_POSIX
  int flags = O_RDWR;
  flags |= O_CLOEXEC;
  if (OpenFirstAvailablePrimary(flags, error_str, error_len)) {
    // Successfully opened our primary pseudo terminal

    pid = ::fork();
    if (pid < 0) {
      // Fork failed
      if (error_str)
        ErrnoToStr(error_str, error_len);
    } else if (pid == 0) {
      // Child Process
      ::setsid();

      if (OpenSecondary(O_RDWR, error_str, error_len)) {
        // Successfully opened secondary

        // Primary FD should have O_CLOEXEC set, but let's close it just in
        // case...
        ClosePrimaryFileDescriptor();

#if defined(TIOCSCTTY)
        // Acquire the controlling terminal
        if (::ioctl(m_secondary_fd, TIOCSCTTY, (char *)0) < 0) {
          if (error_str)
            ErrnoToStr(error_str, error_len);
        }
#endif
        // Duplicate all stdio file descriptors to the secondary pseudo terminal
        if (::dup2(m_secondary_fd, STDIN_FILENO) != STDIN_FILENO) {
          if (error_str && !error_str[0])
            ErrnoToStr(error_str, error_len);
        }

        if (::dup2(m_secondary_fd, STDOUT_FILENO) != STDOUT_FILENO) {
          if (error_str && !error_str[0])
            ErrnoToStr(error_str, error_len);
        }

        if (::dup2(m_secondary_fd, STDERR_FILENO) != STDERR_FILENO) {
          if (error_str && !error_str[0])
            ErrnoToStr(error_str, error_len);
        }
      }
    } else {
      // Parent Process
      // Do nothing and let the pid get returned!
    }
  }
#endif
  return pid;
}

// The primary file descriptor accessor. This object retains ownership of the
// primary file descriptor when this accessor is used. Use
// ReleasePrimaryFileDescriptor() if you wish this object to release ownership
// of the primary file descriptor.
//
// Returns the primary file descriptor, or -1 if the primary file descriptor is
// not currently valid.
int PseudoTerminal::GetPrimaryFileDescriptor() const { return m_primary_fd; }

// The secondary file descriptor accessor.
//
// Returns the secondary file descriptor, or -1 if the secondary file descriptor
// is not currently valid.
int PseudoTerminal::GetSecondaryFileDescriptor() const {
  return m_secondary_fd;
}

// Release ownership of the primary pseudo terminal file descriptor without
// closing it. The destructor for this class will close the primary file
// descriptor if the ownership isn't released using this call and the primary
// file descriptor has been opened.
int PseudoTerminal::ReleasePrimaryFileDescriptor() {
  // Release ownership of the primary pseudo terminal file descriptor without
  // closing it. (the destructor for this class will close it otherwise!)
  int fd = m_primary_fd;
  m_primary_fd = invalid_fd;
  return fd;
}

// Release ownership of the secondary pseudo terminal file descriptor without
// closing it. The destructor for this class will close the secondary file
// descriptor if the ownership isn't released using this call and the secondary
// file descriptor has been opened.
int PseudoTerminal::ReleaseSecondaryFileDescriptor() {
  // Release ownership of the secondary pseudo terminal file descriptor without
  // closing it (the destructor for this class will close it otherwise!)
  int fd = m_secondary_fd;
  m_secondary_fd = invalid_fd;
  return fd;
}
