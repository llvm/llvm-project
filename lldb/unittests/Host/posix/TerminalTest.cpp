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

  ASSERT_THAT_ERROR(m_term.SetEcho(true), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_lflag & ECHO, 0U);

  ASSERT_THAT_ERROR(m_term.SetEcho(false), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_lflag & ECHO, 0U);
}

TEST_F(TerminalTest, SetCanonical) {
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetCanonical(true), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_lflag & ICANON, 0U);

  ASSERT_THAT_ERROR(m_term.SetCanonical(false), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_lflag & ICANON, 0U);
}

TEST_F(TerminalTest, SetRaw) {
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetRaw(), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  // NB: cfmakeraw() on glibc disables IGNBRK, on FreeBSD sets it
  EXPECT_EQ(terminfo.c_iflag &
                (BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON),
            0U);
  EXPECT_EQ(terminfo.c_oflag & OPOST, 0U);
  EXPECT_EQ(terminfo.c_lflag & (ICANON | ECHO | ISIG | IEXTEN), 0U);
  EXPECT_EQ(terminfo.c_cflag & (CSIZE | PARENB), 0U | CS8);
  EXPECT_EQ(terminfo.c_cc[VMIN], 1);
  EXPECT_EQ(terminfo.c_cc[VTIME], 0);
}

TEST_F(TerminalTest, SetBaudRate) {
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetBaudRate(38400), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(cfgetispeed(&terminfo), static_cast<speed_t>(B38400));
  EXPECT_EQ(cfgetospeed(&terminfo), static_cast<speed_t>(B38400));

  ASSERT_THAT_ERROR(m_term.SetBaudRate(115200), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(cfgetispeed(&terminfo), static_cast<speed_t>(B115200));
  EXPECT_EQ(cfgetospeed(&terminfo), static_cast<speed_t>(B115200));

  // uncommon value
#if defined(B153600)
  ASSERT_THAT_ERROR(m_term.SetBaudRate(153600), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(cfgetispeed(&terminfo), static_cast<speed_t>(B153600));
  EXPECT_EQ(cfgetospeed(&terminfo), static_cast<speed_t>(B153600));
#else
  ASSERT_THAT_ERROR(m_term.SetBaudRate(153600),
                    llvm::Failed<llvm::ErrorInfoBase>(testing::Property(
                        &llvm::ErrorInfoBase::message,
                        "baud rate 153600 unsupported by the platform")));
#endif
}

TEST_F(TerminalTest, SetStopBits) {
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetStopBits(1), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_cflag & CSTOPB, 0U);

  ASSERT_THAT_ERROR(m_term.SetStopBits(2), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_cflag & CSTOPB, 0U);

  ASSERT_THAT_ERROR(m_term.SetStopBits(0),
                    llvm::Failed<llvm::ErrorInfoBase>(testing::Property(
                        &llvm::ErrorInfoBase::message,
                        "invalid stop bit count: 0 (must be 1 or 2)")));
  ASSERT_THAT_ERROR(m_term.SetStopBits(3),
                    llvm::Failed<llvm::ErrorInfoBase>(testing::Property(
                        &llvm::ErrorInfoBase::message,
                        "invalid stop bit count: 3 (must be 1 or 2)")));
}

TEST_F(TerminalTest, SetParity) {
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::No), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_cflag & PARENB, 0U);

#if !defined(__linux__) // Linux pty devices do not support setting parity
  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::Even),
                    llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_cflag & PARENB, 0U);
  EXPECT_EQ(terminfo.c_cflag & PARODD, 0U);
#if defined(CMSPAR)
  EXPECT_EQ(terminfo.c_cflag & CMSPAR, 0U);
#endif

  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::Odd), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_cflag & PARENB, 0U);
  EXPECT_NE(terminfo.c_cflag & PARODD, 0U);
#if defined(CMSPAR)
  EXPECT_EQ(terminfo.c_cflag & CMSPAR, 0U);
#endif

#if defined(CMSPAR)
  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::Space),
                    llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_cflag & PARENB, 0U);
  EXPECT_EQ(terminfo.c_cflag & PARODD, 0U);
  EXPECT_NE(terminfo.c_cflag & CMSPAR, 0U);

  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::Mark),
                    llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_cflag & PARENB, 0U);
  EXPECT_NE(terminfo.c_cflag & PARODD, 0U);
  EXPECT_NE(terminfo.c_cflag & CMSPAR, 0U);
#endif // defined(CMSPAR)
#endif // !defined(__linux__)

#if !defined(CMSPAR)
  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::Space),
                    llvm::Failed<llvm::ErrorInfoBase>(testing::Property(
                        &llvm::ErrorInfoBase::message,
                        "space/mark parity is not supported by the platform")));
  ASSERT_THAT_ERROR(m_term.SetParity(Terminal::Parity::Mark),
                    llvm::Failed<llvm::ErrorInfoBase>(testing::Property(
                        &llvm::ErrorInfoBase::message,
                        "space/mark parity is not supported by the platform")));
#endif
}

TEST_F(TerminalTest, SetParityCheck) {
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetParityCheck(Terminal::ParityCheck::No),
                    llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_iflag & (IGNPAR | PARMRK | INPCK), 0U);

  ASSERT_THAT_ERROR(
      m_term.SetParityCheck(Terminal::ParityCheck::ReplaceWithNUL),
      llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_iflag & INPCK, 0U);
  EXPECT_EQ(terminfo.c_iflag & (IGNPAR | PARMRK), 0U);

  ASSERT_THAT_ERROR(m_term.SetParityCheck(Terminal::ParityCheck::Ignore),
                    llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_iflag & IGNPAR, 0U);
  EXPECT_EQ(terminfo.c_iflag & PARMRK, 0U);
  EXPECT_NE(terminfo.c_iflag & INPCK, 0U);

  ASSERT_THAT_ERROR(m_term.SetParityCheck(Terminal::ParityCheck::Mark),
                    llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_iflag & IGNPAR, 0U);
  EXPECT_NE(terminfo.c_iflag & PARMRK, 0U);
  EXPECT_NE(terminfo.c_iflag & INPCK, 0U);
}

TEST_F(TerminalTest, SetHardwareFlowControl) {
#if defined(CRTSCTS)
  struct termios terminfo;

  ASSERT_THAT_ERROR(m_term.SetHardwareFlowControl(true), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_NE(terminfo.c_cflag & CRTSCTS, 0U);

  ASSERT_THAT_ERROR(m_term.SetHardwareFlowControl(false), llvm::Succeeded());
  ASSERT_EQ(tcgetattr(m_fd, &terminfo), 0);
  EXPECT_EQ(terminfo.c_cflag & CRTSCTS, 0U);
#else
  ASSERT_THAT_ERROR(
      m_term.SetHardwareFlowControl(true),
      llvm::Failed<llvm::ErrorInfoBase>(testing::Property(
          &llvm::ErrorInfoBase::message,
          "hardware flow control is not supported by the platform")));
  ASSERT_THAT_ERROR(m_term.SetHardwareFlowControl(false), llvm::Succeeded());
#endif
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
