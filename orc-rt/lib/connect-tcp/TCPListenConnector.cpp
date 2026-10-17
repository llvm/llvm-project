//===- TCPListenConnector.cpp - The "tcp:listen" connector ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "listen" action of the "tcp" transport, and the registration that
// supports both connect and listen.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCPConnector.h"

#include "TCPConnectorHelpers.h"

#include "orc-rt/connect-tcp/TCP.h"

using namespace orc_rt;
using namespace orc_rt::detail;

namespace {

Error tcpListenAction(const ConnectionSpec &CS, Session &S, BootstrapInfo BI) {
  auto EP = parseTCPEndpoint(CS);
  if (!EP)
    return EP.takeError();

  // An executor has one controller, so stop listening as soon as it arrives:
  // the listener is closed before the conversation starts, and no one else
  // can connect.
  auto Sock = [&]() -> Expected<SocketHandle> {
    auto L = TCPListener::listen(EP->Host, EP->Port);
    if (!L)
      return L.takeError();
    return L->accept();
  }();
  if (!Sock)
    return Sock.takeError();

  return attachTCPPeer(S, std::move(*Sock), std::move(BI));
}

// BI is taken by rvalue reference rather than by value; see tcpConnectOnly in
// TCPConnectConnector.cpp.
Error tcpConnectOrListen(const ConnectionSpec &CS, Session &S,
                         BootstrapInfo &&BI) noexcept {
  if (CS.action() == "connect")
    return tcpConnectAction(CS, S, std::move(BI));
  if (CS.action() == "listen")
    return tcpListenAction(CS, S, std::move(BI));
  return invalidTCPSpec(CS, "the \"tcp\" transport supports only the "
                            "\"connect\" and \"listen\" actions");
}

} // namespace

namespace orc_rt {

Error registerTCPConnectAndListen(ConnectorRegistry &R) noexcept {
  return R.registerConnector("tcp", tcpConnectOrListen);
}

} // namespace orc_rt
