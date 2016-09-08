//===-- PseudoTerminal.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 1/8/08.
//
//===----------------------------------------------------------------------===//

#include "PseudoTerminal.h"
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>

//----------------------------------------------------------------------
// PseudoTerminal constructor
//----------------------------------------------------------------------
PseudoTerminal::PseudoTerminal()
    : m_master_fd(invalid_fd), m_slave_fd(invalid_fd) {}

//----------------------------------------------------------------------
// Destructor
// The master and slave file descriptors will get closed if they are
// valid. Call the ReleaseMasterFD()/ReleaseSlaveFD() member functions
// to release any file descriptors that are needed beyond the lifespan
// of this object.
//----------------------------------------------------------------------
PseudoTerminal::~PseudoTerminal() {
  CloseMaster();
  CloseSlave();
}

//----------------------------------------------------------------------
// Close the master file descriptor if it is valid.
//----------------------------------------------------------------------
void PseudoTerminal::CloseMaster() {
  if (m_master_fd > 0) {
    ::close(m_master_fd);
    m_master_fd = invalid_fd;
  }
}

//----------------------------------------------------------------------
// Close the slave file descriptor if it is valid.
//----------------------------------------------------------------------
void PseudoTerminal::CloseSlave() {
  if (m_slave_fd > 0) {
    ::close(m_slave_fd);
    m_slave_fd = invalid_fd;
  }
}

//----------------------------------------------------------------------
// Open the first available pseudo terminal with OFLAG as the
// permissions. The file descriptor is store in the m_master_fd member
// variable and can be accessed via the MasterFD() or ReleaseMasterFD()
// accessors.
//
// Suggested value for oflag is O_RDWR|O_NOCTTY
//
// RETURNS:
//  Zero when successful, non-zero indicating an error occurred.
//----------------------------------------------------------------------
PseudoTerminal::Error PseudoTerminal::OpenFirstAvailableMaster(int oflag) {
  // Open the master side of a pseudo terminal
  m_master_fd = ::posix_openpt(oflag);
  if (m_master_fd < 0) {
    return err_posix_openpt_failed;
  }

  // Grant access to the slave pseudo terminal
  if (::grantpt(m_master_fd) < 0) {
    CloseMaster();
    return err_grantpt_failed;
  }

  // Clear the lock flag on the slave pseudo terminal
  if (::unlockpt(m_master_fd) < 0) {
    CloseMaster();
    return err_unlockpt_failed;
  }

  return success;
}

//----------------------------------------------------------------------
// Open the slave pseudo terminal for the current master pseudo
// terminal. A master pseudo terminal should already be valid prior to
// calling this function (see PseudoTerminal::OpenFirstAvailableMaster()).
// The file descriptor is stored in the m_slave_fd member variable and
// can be accessed via the SlaveFD() or ReleaseSlaveFD() accessors.
//
// RETURNS:
//  Zero when successful, non-zero indicating an error occurred.
//----------------------------------------------------------------------
PseudoTerminal::Error PseudoTerminal::OpenSlave(int oflag) {
  CloseSlave();

  // Open the master side of a pseudo terminal
  const char *slave_name = SlaveName();

  if (slave_name == NULL)
    return err_ptsname_failed;

  m_slave_fd = ::open(slave_name, oflag);

  if (m_slave_fd < 0)
    return err_open_slave_failed;

  return success;
}

//----------------------------------------------------------------------
// Get the name of the slave pseudo terminal. A master pseudo terminal
// should already be valid prior to calling this function (see
// PseudoTerminal::OpenFirstAvailableMaster()).
//
// RETURNS:
//  NULL if no valid master pseudo terminal or if ptsname() fails.
//  The name of the slave pseudo terminal as a NULL terminated C string
//  that comes from static memory, so a copy of the string should be
//  made as subsequent calls can change this value.
//----------------------------------------------------------------------
const char *PseudoTerminal::SlaveName() const {
  if (m_master_fd < 0)
    return NULL;
  return ::ptsname(m_master_fd);
}

//----------------------------------------------------------------------
// Fork a child process that and have its stdio routed to a pseudo
// terminal.
//
// In the parent process when a valid pid is returned, the master file
// descriptor can be used as a read/write access to stdio of the
// child process.
//
// In the child process the stdin/stdout/stderr will already be routed
// to the slave pseudo terminal and the master file descriptor will be
// closed as it is no longer needed by the child process.
//
// This class will close the file descriptors for the master/slave
// when the destructor is called, so be sure to call ReleaseMasterFD()
// or ReleaseSlaveFD() if any file descriptors are going to be used
// past the lifespan of this object.
//
// RETURNS:
//  in the parent process: the pid of the child, or -1 if fork fails
//  in the child process: zero
//----------------------------------------------------------------------

pid_t PseudoTerminal::Fork(PseudoTerminal::Error &error) {
  pid_t pid = invalid_pid;
  error = OpenFirstAvailableMaster(O_RDWR | O_NOCTTY);

  if (error == 0) {
    // Successfully opened our master pseudo terminal

    pid = ::fork();
    if (pid < 0) {
      // Fork failed
      error = err_fork_failed;
    } else if (pid == 0) {
      // Child Process
      ::setsid();

      error = OpenSlave(O_RDWR);
      if (error == 0) {
        // Successfully opened slave
        // We are done with the master in the child process so lets close it
        CloseMaster();

#if defined(TIOCSCTTY)
        // Acquire the controlling terminal
        if (::ioctl(m_slave_fd, TIOCSCTTY, (char *)0) < 0)
          error = err_failed_to_acquire_controlling_terminal;
#endif
        // Duplicate all stdio file descriptors to the slave pseudo terminal
        if (::dup2(m_slave_fd, STDIN_FILENO) != STDIN_FILENO)
          error = error ? error : err_dup2_failed_on_stdin;
        if (::dup2(m_slave_fd, STDOUT_FILENO) != STDOUT_FILENO)
          error = error ? error : err_dup2_failed_on_stdout;
        if (::dup2(m_slave_fd, STDERR_FILENO) != STDERR_FILENO)
          error = error ? error : err_dup2_failed_on_stderr;
      }
    } else {
      // Parent Process
      // Do nothing and let the pid get returned!
    }
  }
  return pid;
}
