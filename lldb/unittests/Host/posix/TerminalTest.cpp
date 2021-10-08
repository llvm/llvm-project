//===-- TerminalTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/Terminal.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <termios.h>
#include <unistd.h>

using namespace lldb_private;

class TerminalTest : public ::testing::Test {
protected:
  PseudoTerminal m_pty;
  int m_fd;
  Terminal m_term;

  void SetUp() override {
    ASSERT_THAT_ERROR(m_pty.OpenFirstAvailablePrimary(O_RDWR | O_NOCTTY),
                      llvm::Succeeded());
    ASSERT_THAT_ERROR(m_pty.OpenSecondary(O_RDWR | O_NOCTTY),
                      llvm::Succeeded());
    m_fd = m_pty.GetSecondaryFileDescriptor();
    ASSERT_NE(m_fd, -1);
    m_term.SetFileDescriptor(m_fd);
  }
};

TEST_F(TerminalTest, PtyIsATerminal) {
  EXPECT_EQ(m_term.IsATerminal(), true);
}

TEST_F(TerminalTest, PipeIsNotATerminal) {
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  Terminal pipeterm{pipefd[0]};
  EXPECT_EQ(pipeterm.IsATerminal(), false);
  close(pipefd[0]);
  close(pipefd[1]);
}

TEST_F(TerminalTest, SetEcho) {
  struct termios terminfo;

  ASSERT_EQ(m_term.SetEcho(true), true);
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_lflag & ECHO, 0U);

  ASSERT_EQ(m_term.SetEcho(false), true);
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_lflag & ECHO, 0U);
}

TEST_F(TerminalTest, SetCanonical) {
  struct termios terminfo;

  ASSERT_EQ(m_term.SetCanonical(true), true);
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_lflag & ICANON, 0U);

  ASSERT_EQ(m_term.SetCanonical(false), true);
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_lflag & ICANON, 0U);
}

TEST_F(TerminalTest, SaveRestoreRAII) {
  struct termios orig_terminfo;
  struct termios terminfo;
  ASSERT_EQ(tcgetattr(m_fd, &orig_terminfo), 0);

  {
    TerminalState term_state{m_term};
    terminfo = orig_terminfo;

    // make an arbitrary change
    cfsetispeed(&terminfo,
                cfgetispeed(&orig_terminfo) == B9600 ? B4800 : B9600);
    cfsetospeed(&terminfo,
                cfgetospeed(&orig_terminfo) == B9600 ? B4800 : B9600);

    ASSERT_EQ(tcsetattr(m_fd, TCSANOW, &terminfo),
              0);
  }

  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  ASSERT_EQ(memcmp(&terminfo, &orig_terminfo, sizeof(terminfo)), 0);
}

TEST_F(TerminalTest, SaveRestore) {
  TerminalState term_state;

  struct termios orig_terminfo;
  struct termios terminfo;
  ASSERT_EQ(tcgetattr(m_fd, &orig_terminfo), 0);

  term_state.Save(m_term, false);
  terminfo = orig_terminfo;

  // make an arbitrary change
  cfsetispeed(&terminfo, cfgetispeed(&orig_terminfo) == B9600 ? B4800 : B9600);
  cfsetospeed(&terminfo, cfgetospeed(&orig_terminfo) == B9600 ? B4800 : B9600);

  ASSERT_EQ(tcsetattr(m_fd, TCSANOW, &terminfo), 0);

  term_state.Restore();
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  ASSERT_EQ(memcmp(&terminfo, &orig_terminfo, sizeof(terminfo)), 0);
}
