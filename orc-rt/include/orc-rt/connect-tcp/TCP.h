//===- TCP.h - TCP connection establishment ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Primitives for establishing TCP connections.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_CONNECT_TCP_TCP_H
#define ORC_RT_CONNECT_TCP_TCP_H

#include "orc-rt/connect-socket/SocketHandle.h"
#include "orc-rt/support/Error.h"

#include <cstdint>
#include <string_view>

namespace orc_rt {

/// Connects to Port on Host and returns the connected socket.
///
/// Host is a name or a numeric address, IPv6 addresses unbracketed (e.g.
/// "::1"). A name may resolve to several addresses, and each is tried in turn
/// until one connects. The error, if none does, reports every attempt.
///
/// Host must not be empty: the system would read an empty host as loopback,
/// and a caller that meant loopback should say so.
///
/// The socket is blocking and close-on-exec. Nothing is known about the peer
/// beyond the address it answered on: TCP does not identify it.
Expected<SocketHandle> tcpConnect(std::string_view Host,
                                  uint16_t Port) noexcept;

/// A TCP socket listening for connections.
///
/// A listener accepts any peer that can reach it: whoever connects is the one
/// you get, and TCP says nothing about who that is. What can reach it depends
/// on the address it listens on. Loopback addresses are reachable from this
/// machine only, and the wildcard addresses "0.0.0.0" and "::" from any network
/// this machine is on.
class TCPListener {
public:
  /// Listens on Port at Address.
  ///
  /// Address must be a numeric address, IPv6 addresses unbracketed (e.g.
  /// "127.0.0.1" or "::1"). A host name is an error: what a listener can be
  /// reached on should be exactly what its caller named, not whatever name
  /// resolution makes of it, and a name may also resolve to several
  /// addresses. Port 0 asks the system to choose a free port, which port()
  /// then reports.
  ///
  /// Address must not be empty. The system would read an empty address as
  /// every interface, which should never happen by omission. Pass "0.0.0.0" or
  /// "::" to ask for that explicitly.
  static Expected<TCPListener> listen(std::string_view Address,
                                      uint16_t Port) noexcept;

  /// The port this listener is bound to.
  uint16_t port() const noexcept { return Port; }

  /// Waits for a peer to connect and returns the connected socket.
  ///
  /// Blocks until a peer connects. A peer that connects and resets before it is
  /// accepted is skipped, not reported. The socket is blocking and
  /// close-on-exec.
  Expected<SocketHandle> accept() noexcept;

private:
  TCPListener(SocketHandle Sock, uint16_t Port) noexcept
      : Sock(std::move(Sock)), Port(Port) {}

  SocketHandle Sock;
  uint16_t Port;
};

} // namespace orc_rt

#endif // ORC_RT_CONNECT_TCP_TCP_H
