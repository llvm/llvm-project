//===- TCPConnector.h - Connectors for the "tcp" transport ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Connectors for the "tcp" transport.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_CONNECT_TCP_TCPCONNECTOR_H
#define ORC_RT_CONNECT_TCP_TCPCONNECTOR_H

#include "orc-rt/connect/ConnectorRegistry.h"

namespace orc_rt {

/// Registers a connector for the "tcp" transport that supports only the
/// "connect" action.
///
/// "tcp:connect=<host>:<port>" connects to the controller at that endpoint.
/// <host> is a name or a numeric address, with IPv6 addresses bracketed (e.g.
/// "tcp:connect=[::1]:20000"). Connecting blocks until the connection is made
/// or fails.
///
/// TCP does not identify the peer, so it is trusted as unchecked (see
/// VettedPeer::unchecked). Anyone who can impersonate the controller's
/// endpoint can run code in this process.
///
/// Any other action is rejected. To accept "tcp:listen" as well, call
/// registerTCPConnectAndListen instead. The two can't both be registered.
Error registerTCPConnect(ConnectorRegistry &R) noexcept;

/// Registers a connector for the "tcp" transport that supports both the
/// "connect" action (see registerTCPConnect) and the "listen" action.
///
/// "tcp:listen=<address>:<port>" listens on that endpoint, accepts one
/// controller, and stops listening. Listening blocks until a controller
/// connects. The address must be numeric, IPv6 addresses bracketed (e.g.
/// "tcp:listen=127.0.0.1:20000" or "tcp:listen=[::1]:20000"): a host name is
/// rejected, as for TCPListener::listen. The port must not be 0, since nothing
/// would tell the controller which port the system chose.
///
/// The accepted peer is trusted as unchecked (see VettedPeer::unchecked), and
/// whoever connects first is the peer. So anyone who can reach the endpoint
/// can run code in this process: listen on a loopback address unless the
/// controller is on another machine.
Error registerTCPConnectAndListen(ConnectorRegistry &R) noexcept;

} // namespace orc_rt

#endif // ORC_RT_CONNECT_TCP_TCPCONNECTOR_H
