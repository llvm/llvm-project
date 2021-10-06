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

  void SetUp() override {
    EXPECT_THAT_ERROR(m_pty.OpenFirstAvailablePrimary(O_RDWR | O_NOCTTY),
                      llvm::Succeeded());
  }
};

TEST_F(TerminalTest, PtyIsATerminal) {
  Terminal term{m_pty.GetPrimaryFileDescriptor()};
  EXPECT_EQ(term.IsATerminal(), true);
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
  Terminal term{m_pty.GetPrimaryFileDescriptor()};

  ASSERT_EQ(term.SetEcho(true), true);
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &terminfo), 0);
  EXPECT_NE(terminfo.c_lflag & ECHO, 0U);

  ASSERT_EQ(term.SetEcho(false), true);
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &terminfo), 0);
  EXPECT_EQ(terminfo.c_lflag & ECHO, 0U);
}

TEST_F(TerminalTest, SetCanonical) {
  struct termios terminfo;
  Terminal term{m_pty.GetPrimaryFileDescriptor()};

  ASSERT_EQ(term.SetCanonical(true), true);
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &terminfo), 0);
  EXPECT_NE(terminfo.c_lflag & ICANON, 0U);

  ASSERT_EQ(term.SetCanonical(false), true);
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &terminfo), 0);
  EXPECT_EQ(terminfo.c_lflag & ICANON, 0U);
}

TEST_F(TerminalTest, SaveRestoreRAII) {
  struct termios orig_terminfo;
  struct termios terminfo;
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &orig_terminfo), 0);

  Terminal term{m_pty.GetPrimaryFileDescriptor()};

  {
    TerminalState term_state{term};
    terminfo = orig_terminfo;

    // make some arbitrary changes
    terminfo.c_iflag ^= IGNPAR | INLCR;
    terminfo.c_oflag ^= OPOST | OCRNL;
    terminfo.c_cflag ^= PARENB | PARODD;
    terminfo.c_lflag ^= ICANON | ECHO;
    terminfo.c_cc[VEOF] ^= 8;
    terminfo.c_cc[VEOL] ^= 4;
    cfsetispeed(&terminfo, B9600);
    cfsetospeed(&terminfo, B9600);

    ASSERT_EQ(tcsetattr(m_pty.GetPrimaryFileDescriptor(), TCSANOW, &terminfo),
              0);
  }

  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &terminfo), 0);
  ASSERT_EQ(memcmp(&terminfo, &orig_terminfo, sizeof(terminfo)), 0);
}

TEST_F(TerminalTest, SaveRestore) {
  TerminalState term_state;

  struct termios orig_terminfo;
  struct termios terminfo;
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &orig_terminfo), 0);

  Terminal term{m_pty.GetPrimaryFileDescriptor()};
  term_state.Save(term, false);
  terminfo = orig_terminfo;

  // make some arbitrary changes
  terminfo.c_iflag ^= IGNPAR | INLCR;
  terminfo.c_oflag ^= OPOST | OCRNL;
  terminfo.c_cflag ^= PARENB | PARODD;
  terminfo.c_lflag ^= ICANON | ECHO;
  terminfo.c_cc[VEOF] ^= 8;
  terminfo.c_cc[VEOL] ^= 4;
  cfsetispeed(&terminfo, B9600);
  cfsetospeed(&terminfo, B9600);

  ASSERT_EQ(tcsetattr(m_pty.GetPrimaryFileDescriptor(), TCSANOW, &terminfo), 0);

  term_state.Restore();
  ASSERT_EQ(tcgetattr(m_pty.GetPrimaryFileDescriptor(), &terminfo), 0);
  ASSERT_EQ(memcmp(&terminfo, &orig_terminfo, sizeof(terminfo)), 0);
}
