//===- TCPConnectorTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the "tcp" connectors.
//
// Specs are validated before anything touches the network, so those tests need
// no fixture. The tests that connect do so over loopback, and skip on machines
// without it.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCPConnector.h"

#include "orc-rt/bedrock/Session.h"
#include "orc-rt/connect-tcp/TCP.h"
#include "orc-rt/connect/ConnectionSpec.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"
#include "ErrorMatchers.h"
#include "connect-socket/SocketTestUtils.h"
#include "connect-tcp/TCPTestUtils.h"

#include "gtest/gtest.h"

#include <chrono>
#include <future>
#include <string>
#include <thread>

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::HasSubstr;

namespace {

using RegisterFn = Error (*)(ConnectorRegistry &) noexcept;

using TCPConnectorIPv4Test = LoopbackTCPTest<IPFamily::IPv4>;
using TCPConnectorIPv6Test = LoopbackTCPTest<IPFamily::IPv6>;

/// Runs Spec through the connectors that Register installs, targeting a fresh
/// Session. For specs that fail before attaching, so the Session never
/// connects (and noErrors would catch an unexpected attach failure).
Error connectWith(RegisterFn Register, const std::string &Spec) {
  ConnectorRegistry R;
  if (auto Err = Register(R))
    return Err;
  auto CS = ConnectionSpec::parse(Spec);
  if (!CS)
    return CS.takeError();
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  return R.connect(*CS, S, BootstrapInfo(S));
}

/// Checks that the executor at the far end of Controller has started the
/// conversation. What it sends is the CA's business, and tested there.
void expectConversationStarted(const SocketHandle &Controller) {
  char Byte;
  auto N = recvAll(Controller.get(), &Byte, 1);
  ASSERT_THAT_EXPECTED(N, Succeeded());
  EXPECT_EQ(*N, 1u) << "the executor sent nothing";
}

/// Returns a port that nothing was listening on a moment ago. Another process
/// could take it before the caller does, but on loopback that is unlikely.
uint16_t unusedPort(const char *Host) {
  auto L = TCPListener::listen(Host, 0);
  if (!L) {
    ADD_FAILURE() << toString(L.takeError());
    return 0;
  }
  return L->port();
}

/// Connects to Port on Host, retrying until something listens there or a few
/// seconds pass. Returns an empty handle if nothing did.
SocketHandle connectWhenListening(const char *Host, uint16_t Port) {
  for (int Attempt = 0; Attempt != 500; ++Attempt) {
    auto Sock = tcpConnect(Host, Port);
    if (Sock)
      return std::move(*Sock);
    consumeError(Sock.takeError());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return SocketHandle();
}

/// Plays the controller for an executor told "tcp:connect=<SpecHost>:<port>",
/// listening on ListenHost.
void connectAttachesASession(const char *ListenHost, const char *SpecHost) {
  auto L = TCPListener::listen(ListenHost, 0);
  ASSERT_THAT_EXPECTED(L, Succeeded());

  SocketHandle Controller;
  std::future<Error> Disconnected;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.setOnDisconnect(waitFor(Disconnected));

  ConnectorRegistry R;
  ASSERT_THAT_ERROR(registerTCPConnect(R), Succeeded());
  auto CS = ConnectionSpec::parse(std::string("tcp:connect=") + SpecHost + ":" +
                                  std::to_string(L->port()));
  ASSERT_THAT_EXPECTED(CS, Succeeded());

  // The connection completes in the listen backlog, so the executor can
  // connect before the controller accepts.
  ASSERT_THAT_ERROR(R.connect(*CS, S, BootstrapInfo(S)), Succeeded());
  auto Accepted = L->accept();
  ASSERT_THAT_EXPECTED(Accepted, Succeeded());
  Controller = std::move(*Accepted);

  expectConversationStarted(Controller);

  S.detach();
  EXPECT_THAT_ERROR(Disconnected.get(), Succeeded());
}

TEST_F(TCPConnectorIPv4Test, ConnectAttachesASession) {
  connectAttachesASession("127.0.0.1", "127.0.0.1");
}

TEST_F(TCPConnectorIPv6Test, ConnectToBracketedAddressAttachesASession) {
  connectAttachesASession("::1", "[::1]");
}

TEST_F(TCPConnectorIPv4Test, ListenAcceptsOneControllerAndStopsListening) {
  uint16_t Port = unusedPort("127.0.0.1");
  ASSERT_NE(Port, 0);

  SocketHandle Controller;
  std::future<Error> Disconnected;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.setOnDisconnect(waitFor(Disconnected));

  ConnectorRegistry R;
  ASSERT_THAT_ERROR(registerTCPConnectAndListen(R), Succeeded());
  auto CS =
      ConnectionSpec::parse("tcp:listen=127.0.0.1:" + std::to_string(Port));
  ASSERT_THAT_EXPECTED(CS, Succeeded());

  // The connector blocks until a controller connects, so the controller runs
  // on its own thread, retrying until the executor is listening.
  std::thread ControllerThread(
      [&] { Controller = connectWhenListening("127.0.0.1", Port); });
  Error ConnectErr = R.connect(*CS, S, BootstrapInfo(S));
  ControllerThread.join();
  ASSERT_THAT_ERROR(std::move(ConnectErr), Succeeded());
  ASSERT_TRUE(!!Controller) << "the controller could not connect";

  expectConversationStarted(Controller);

  // The executor stopped listening once its controller arrived.
  EXPECT_THAT_EXPECTED(tcpConnect("127.0.0.1", Port), Failed());

  S.detach();
  EXPECT_THAT_ERROR(Disconnected.get(), Succeeded());
}

TEST(TCPConnectorTest, ConnectOnlyRegistrationRejectsListen) {
  EXPECT_THAT_ERROR(
      connectWith(registerTCPConnect, "tcp:listen=127.0.0.1:20000"),
      FailedWithMessage(
          HasSubstr("supports only the \"connect\" action for the \"tcp\"")));
}

TEST(TCPConnectorTest, RejectsOtherActions) {
  for (const char *Spec : {"tcp:adopt=3", "tcp=127.0.0.1:20000"}) {
    SCOPED_TRACE(Spec);
    EXPECT_THAT_ERROR(
        connectWith(registerTCPConnectAndListen, Spec),
        FailedWithMessage(
            HasSubstr("supports only the \"connect\" and \"listen\" actions")));
  }
}

TEST(TCPConnectorTest, RejectsMalformedDescriptors) {
  struct {
    const char *Spec;
    const char *Reason;
  } Cases[] = {
      {"tcp:connect=localhost", "expected <host>:<port>"},
      {"tcp:connect=:20000", "no host given"},
      {"tcp:connect=::1:20000", "an IPv6 address must be bracketed"},
      {"tcp:connect=[::1", "missing ']'"},
      {"tcp:connect=[::1]20000", "expected ':' and a port after ']'"},
      {"tcp:connect=localhost:", "\"\" is not a port number"},
      {"tcp:connect=localhost:http", "\"http\" is not a port number"},
      {"tcp:connect=localhost:-1", "\"-1\" is not a port number"},
      {"tcp:connect=localhost:65536", "\"65536\" is not a port number"},
      {"tcp:connect=localhost:0", "port 0 cannot be connected to"},
      {"tcp:listen=127.0.0.1:0", "a listening executor must name its port"},
  };
  for (auto &C : Cases) {
    SCOPED_TRACE(C.Spec);
    EXPECT_THAT_ERROR(connectWith(registerTCPConnectAndListen, C.Spec),
                      FailedWithMessage(HasSubstr(C.Reason)));
  }
}

TEST(TCPConnectorTest, ListenRequiresANumericAddress) {
  // Rejected before anything is bound, and without looking the name up.
  EXPECT_THAT_ERROR(
      connectWith(registerTCPConnectAndListen, "tcp:listen=localhost:20000"),
      FailedWithMessage(HasSubstr("\"localhost\" is not a numeric address")));
}

TEST(TCPConnectorTest, ErrorsQuoteTheSpec) {
  EXPECT_THAT_ERROR(
      connectWith(registerTCPConnect, "tcp:connect=localhost"),
      FailedWithMessage(
          HasSubstr("Invalid connection spec \"tcp:connect=localhost\": ")));
}

TEST(TCPConnectorTest, RegistrationsAreAlternatives) {
  // Both register the "tcp" transport, so a process gets one or the other.
  ConnectorRegistry R;
  ASSERT_THAT_ERROR(registerTCPConnect(R), Succeeded());
  EXPECT_THAT_ERROR(registerTCPConnectAndListen(R),
                    FailedWithMessage(HasSubstr("already registered")));
}

} // namespace
