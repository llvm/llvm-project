//===- TCPTest.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the TCP connect and listen primitives.
//
// Every test that connects does so over loopback, so none needs a network.
// Those tests skip on machines without loopback TCP, which the suites without
// a fixture do not need.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCP.h"

#include "ErrorMatchers.h"
#include "connect-socket/SocketTestUtils.h"
#include "connect-tcp/TCPTestUtils.h"

#include "gtest/gtest.h"

#include <string>

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace {

using TCPIPv4Test = LoopbackTCPTest<IPFamily::IPv4>;
using TCPIPv6Test = LoopbackTCPTest<IPFamily::IPv6>;

/// Checks that bytes go both ways between a connected pair.
void expectConnected(const SocketHandle &A, const SocketHandle &B) {
  char Buf[4];

  ASSERT_THAT_ERROR(sendAll(A.get(), "ping", 4), Succeeded());
  auto N = recvAll(B.get(), Buf, sizeof(Buf));
  ASSERT_THAT_EXPECTED(N, Succeeded());
  EXPECT_EQ(std::string(Buf, *N), "ping");

  ASSERT_THAT_ERROR(sendAll(B.get(), "pong", 4), Succeeded());
  N = recvAll(A.get(), Buf, sizeof(Buf));
  ASSERT_THAT_EXPECTED(N, Succeeded());
  EXPECT_EQ(std::string(Buf, *N), "pong");
}

/// Listens on Host, connects to the listener at ConnectHost, and checks that
/// the two ends are connected to each other.
void connectAndAccept(const char *ListenHost, const char *ConnectHost) {
  auto L = TCPListener::listen(ListenHost, 0);
  ASSERT_THAT_EXPECTED(L, Succeeded());

  // The connection completes in the listen backlog, so there is no need to be
  // accepting before connecting.
  auto Client = tcpConnect(ConnectHost, L->port());
  ASSERT_THAT_EXPECTED(Client, Succeeded());
  auto Server = L->accept();
  ASSERT_THAT_EXPECTED(Server, Succeeded());

  expectConnected(*Client, *Server);
}

TEST_F(TCPIPv4Test, ListenOnPortZeroReportsTheChosenPort) {
  auto L = TCPListener::listen("127.0.0.1", 0);
  ASSERT_THAT_EXPECTED(L, Succeeded());
  EXPECT_NE(L->port(), 0);
}

TEST_F(TCPIPv4Test, ConnectAndAccept) {
  connectAndAccept("127.0.0.1", "127.0.0.1");
}

TEST_F(TCPIPv4Test, ConnectByName) {
  if (!localhostResolvesTo(IPFamily::IPv4))
    GTEST_SKIP() << "\"localhost\" does not resolve to an IPv4 address";

  // "localhost" may resolve to ::1 before 127.0.0.1. In that case the first
  // attempt is refused, and this passes only if the next is tried.
  connectAndAccept("127.0.0.1", "localhost");
}

TEST_F(TCPIPv4Test, ListenerAcceptsSuccessivePeers) {
  auto L = TCPListener::listen("127.0.0.1", 0);
  ASSERT_THAT_EXPECTED(L, Succeeded());

  for (int I = 0; I != 2; ++I) {
    auto Client = tcpConnect("127.0.0.1", L->port());
    ASSERT_THAT_EXPECTED(Client, Succeeded());
    auto Server = L->accept();
    ASSERT_THAT_EXPECTED(Server, Succeeded());
    expectConnected(*Client, *Server);
  }
}

TEST_F(TCPIPv4Test, ConnectRefusedReportsEachAttempt) {
  uint16_t Port;
  {
    auto L = TCPListener::listen("127.0.0.1", 0);
    ASSERT_THAT_EXPECTED(L, Succeeded());
    Port = L->port();
  }

  // Nothing listens on Port now that the listener is gone, barring a race
  // with another process that binds it in the meantime.
  std::string Endpoint = "127.0.0.1:" + std::to_string(Port);
  EXPECT_THAT_EXPECTED(
      tcpConnect("127.0.0.1", Port),
      FailedWithMessage(AllOf(HasSubstr("Could not connect to " + Endpoint),
                              HasSubstr(Endpoint + ": connect failed"))));
}

TEST_F(TCPIPv4Test, ListenOnPortInUseFails) {
  auto First = TCPListener::listen("127.0.0.1", 0);
  ASSERT_THAT_EXPECTED(First, Succeeded());

  EXPECT_THAT_EXPECTED(TCPListener::listen("127.0.0.1", First->port()),
                       FailedWithMessage(HasSubstr("bind failed")));
}

TEST_F(TCPIPv6Test, ConnectAndAccept) { connectAndAccept("::1", "::1"); }

// The tests below fail before any connection is attempted, or whether or not
// one can be made, so they need no fixture and run everywhere.

TEST(TCPTest, IPv6EndpointsAreBracketedInErrors) {
  EXPECT_THAT_EXPECTED(tcpConnect("::1", 0),
                       FailedWithMessage(HasSubstr("[::1]:0")));
}

TEST(TCPTest, ConnectRejectsEmptyHost) {
  EXPECT_THAT_EXPECTED(tcpConnect("", 1),
                       FailedWithMessage(HasSubstr("no host given")));
}

TEST(TCPTest, ListenRejectsEmptyHost) {
  EXPECT_THAT_EXPECTED(TCPListener::listen("", 0),
                       FailedWithMessage(HasSubstr("no host given")));
}

TEST(TCPTest, ListenRequiresANumericAddress) {
  // "localhost" is the case that matters: it may resolve to both ::1 and
  // 127.0.0.1, and a listener on one is unreachable at the other.
  for (const char *Host : {"localhost", "256.0.0.1"}) {
    SCOPED_TRACE(Host);
    EXPECT_THAT_EXPECTED(
        TCPListener::listen(Host, 0),
        FailedWithMessage(HasSubstr(std::string("\"") + Host +
                                    "\" is not a numeric address")));
  }
}

} // namespace
