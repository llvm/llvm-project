//===- TCPConnectConnector.cpp - The "tcp:connect" connector --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "connect" action of the "tcp" transport, and the connect-only
// registration.
//
// Kept apart from TCPListenConnector.cpp so that a process that registers only
// connect does not link the code to listen.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCPConnector.h"

#include "TCPConnectorHelpers.h"

#include "orc-rt/connect-tcp/TCP.h"

#include <cassert>

using namespace orc_rt;
using namespace orc_rt::detail;

namespace {

// BI is taken by rvalue reference rather than by value: a ConnectorFn must be
// nothrow-invocable, and moving a BootstrapInfo can throw where moving its
// std::unordered_maps can (as in MSVC's standard library).
Error tcpConnectOnly(const ConnectionSpec &CS, Session &S,
                     BootstrapInfo &&BI) noexcept {
  // Checked here rather than left to the registry, so that a process built to
  // reach out but not to be reached says which it is.
  if (CS.action() != "connect")
    return invalidTCPSpec(CS, "this process supports only the \"connect\" "
                              "action for the \"tcp\" transport");
  return tcpConnectAction(CS, S, std::move(BI));
}

} // namespace

namespace orc_rt {

Error detail::tcpConnectAction(const ConnectionSpec &CS, Session &S,
                               BootstrapInfo BI) {
  assert(CS.action() == "connect" && "Not a tcp:connect spec");
  auto EP = parseTCPEndpoint(CS);
  if (!EP)
    return EP.takeError();
  auto Sock = tcpConnect(EP->Host, EP->Port);
  if (!Sock)
    return Sock.takeError();
  return attachTCPPeer(S, std::move(*Sock), std::move(BI));
}

Error registerTCPConnect(ConnectorRegistry &R) noexcept {
  return R.registerConnector("tcp", tcpConnectOnly);
}

} // namespace orc_rt
