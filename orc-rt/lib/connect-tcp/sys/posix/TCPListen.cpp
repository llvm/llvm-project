//===- TCPListen.cpp - TCP listen on POSIX --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Listening for TCP connections on POSIX systems.
//
// Kept apart from TCPConnect.cpp so that a process that only connects does not
// link the code to listen.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCP.h"

#include "TCPSocket.h"

#include "orc-rt-c/support/Compiler.h"

#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>

using namespace orc_rt;
using namespace orc_rt::sys::posix;

namespace {

Error bindAndListen(const SocketHandle &Sock, const addrinfo &AI) {
  if (::bind(Sock.get(), AI.ai_addr, AI.ai_addrlen) != 0)
    return make_error<StringError>(opFailed("bind", errno));
  if (::listen(Sock.get(), SOMAXCONN) != 0)
    return make_error<StringError>(opFailed("listen", errno));
  return Error::success();
}

Expected<uint16_t> boundPort(const SocketHandle &Sock) {
  sockaddr_storage Addr;
  socklen_t Len = sizeof(Addr);
  if (::getsockname(Sock.get(), reinterpret_cast<sockaddr *>(&Addr), &Len) != 0)
    return make_error<StringError>(opFailed("getsockname", errno));

  switch (Addr.ss_family) {
  case AF_INET:
    return ntohs(reinterpret_cast<const sockaddr_in &>(Addr).sin_port);
  case AF_INET6:
    return ntohs(reinterpret_cast<const sockaddr_in6 &>(Addr).sin6_port);
  default:
    ORC_RT_UNREACHABLE("TCP socket is neither IPv4 nor IPv6");
  }
}

} // namespace

namespace orc_rt {

Expected<TCPListener> TCPListener::listen(std::string_view Address,
                                          uint16_t Port) noexcept {
  // Numeric only, so that what this listens on is exactly what the caller
  // named, and not whatever name resolution makes of it.
  auto Addrs = resolveTCP(Address, Port, HostKind::NumericAddress, "listen on");
  if (!Addrs)
    return Addrs.takeError();

  AttemptLog Log;
  for (const addrinfo *AI = Addrs->get(); AI; AI = AI->ai_next) {
    auto Sock = openSocket(*AI);
    if (!Sock) {
      Log.add(*AI, toString(Sock.takeError()));
      continue;
    }
    if (auto Err = bindAndListen(*Sock, *AI)) {
      Log.add(*AI, toString(std::move(Err)));
      continue;
    }
    auto BoundPort = boundPort(*Sock);
    if (!BoundPort)
      return BoundPort.takeError();
    return TCPListener(std::move(*Sock), *BoundPort);
  }

  return Log.fail("Could not listen on " + endpointString(Address, Port));
}

Expected<SocketHandle> TCPListener::accept() noexcept {
  for (;;) {
    SocketHandle Peer(::accept(Sock.get(), nullptr, nullptr));
    if (Peer) {
      if (auto Err = setCloseOnExec(Peer.get()))
        return std::move(Err);
      return Peer;
    }
    // Neither a signal nor a peer that reset while it was queued is a failure
    // of this listener, so wait for the next peer.
    int ErrNum = errno;
    if (ErrNum != EINTR && ErrNum != ECONNABORTED)
      return make_error<StringError>(opFailed("accept", ErrNum));
  }
}

} // namespace orc_rt
