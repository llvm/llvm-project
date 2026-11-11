//===- TCPConnectorHelpers.h - Shared TCP connector helpers -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by the "tcp" connectors.
//
// Private to orc-rt-connect-tcp, like sys/posix/TCPSocket.h.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_LIB_CONNECT_TCP_TCPCONNECTORHELPERS_H
#define ORC_RT_LIB_CONNECT_TCP_TCPCONNECTORHELPERS_H

#include "orc-rt/bedrock/BootstrapInfo.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/connect-socket/SocketHandle.h"
#include "orc-rt/connect/ConnectionSpec.h"
#include "orc-rt/support/Error.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace orc_rt::detail {

/// An error saying that CS is invalid, for Reason.
Error invalidTCPSpec(const ConnectionSpec &CS, std::string_view Reason);

/// The endpoint named by a "tcp" connection spec's descriptor.
struct TCPEndpoint {
  std::string Host;
  uint16_t Port = 0;
};

/// Parses CS's descriptor as "<host>:<port>", or "[<IPv6 address>]:<port>".
/// Rejects port 0, which neither action can use.
Expected<TCPEndpoint> parseTCPEndpoint(const ConnectionSpec &CS);

/// Starts a SimpleRemote conversation over Sock, whose peer TCP does not
/// identify, and attaches it to S.
Error attachTCPPeer(Session &S, SocketHandle Sock, BootstrapInfo BI);

/// Performs a "tcp:connect" spec, for either registration.
Error tcpConnectAction(const ConnectionSpec &CS, Session &S, BootstrapInfo BI);

} // namespace orc_rt::detail

#endif // ORC_RT_LIB_CONNECT_TCP_TCPCONNECTORHELPERS_H
