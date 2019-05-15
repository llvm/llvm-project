//===-- PseudoTerminal.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 1/8/08.
//
//===----------------------------------------------------------------------===//

#ifndef __PseudoTerminal_h__
#define __PseudoTerminal_h__

#include <fcntl.h>
#include <string>
#include <termios.h>

class PseudoTerminal {
public:
  enum { invalid_fd = -1, invalid_pid = -1 };

  enum Status {
    success = 0,
    err_posix_openpt_failed = -2,
    err_grantpt_failed = -3,
    err_unlockpt_failed = -4,
    err_ptsname_failed = -5,
    err_open_slave_failed = -6,
    err_fork_failed = -7,
    err_setsid_failed = -8,
    err_failed_to_acquire_controlling_terminal = -9,
    err_dup2_failed_on_stdin = -10,
    err_dup2_failed_on_stdout = -11,
    err_dup2_failed_on_stderr = -12
  };
  // Constructors and Destructors
  PseudoTerminal();
  ~PseudoTerminal();

  void CloseMaster();
  void CloseSlave();
  Status OpenFirstAvailableMaster(int oflag);
  Status OpenSlave(int oflag);
  int MasterFD() const { return m_master_fd; }
  int SlaveFD() const { return m_slave_fd; }
  int ReleaseMasterFD() {
    // Release ownership of the master pseudo terminal file
    // descriptor without closing it. (the destructor for this
    // class will close it otherwise!)
    int fd = m_master_fd;
    m_master_fd = invalid_fd;
    return fd;
  }
  int ReleaseSlaveFD() {
    // Release ownership of the slave pseudo terminal file
    // descriptor without closing it (the destructor for this
    // class will close it otherwise!)
    int fd = m_slave_fd;
    m_slave_fd = invalid_fd;
    return fd;
  }

  const char *SlaveName() const;

  pid_t Fork(Status &error);

protected:
  // Classes that inherit from PseudoTerminal can see and modify these
  int m_master_fd;
  int m_slave_fd;

private:
  PseudoTerminal(const PseudoTerminal &rhs) = delete;
  PseudoTerminal &operator=(const PseudoTerminal &rhs) = delete;
};

#endif // #ifndef __PseudoTerminal_h__
