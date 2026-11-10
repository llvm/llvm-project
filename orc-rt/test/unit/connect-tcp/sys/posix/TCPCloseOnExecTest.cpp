//===- TCPCloseOnExecTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that the POSIX TCP primitives do not leak sockets across exec.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCP.h"

#include "ErrorMatchers.h"
#include "connect-tcp/TCPTestUtils.h"

#include "gtest/gtest.h"

#include <fcntl.h>

using namespace orc_rt;
using namespace orc_rt::test;

namespace {

using TCPCloseOnExecTest = LoopbackTCPTest<IPFamily::IPv4>;

bool isCloseOnExec(const SocketHandle &Sock) {
  int Flags = ::fcntl(Sock.get(), F_GETFD);
  return Flags != -1 && (Flags & FD_CLOEXEC);
}

TEST_F(TCPCloseOnExecTest, ConnectedAndAcceptedSocketsAreCloseOnExec) {
  auto L = TCPListener::listen("127.0.0.1", 0);
  ASSERT_THAT_EXPECTED(L, Succeeded());
  auto Client = tcpConnect("127.0.0.1", L->port());
  ASSERT_THAT_EXPECTED(Client, Succeeded());
  auto Server = L->accept();
  ASSERT_THAT_EXPECTED(Server, Succeeded());

  EXPECT_TRUE(isCloseOnExec(*Client));
  EXPECT_TRUE(isCloseOnExec(*Server));
}

} // namespace
