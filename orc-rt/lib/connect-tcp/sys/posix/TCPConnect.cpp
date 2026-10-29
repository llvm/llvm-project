//===- TCPConnect.cpp - TCP connect on POSIX ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Outbound TCP connections on POSIX systems.
//
// Kept apart from TCPListen.cpp so that a process that only connects does not
// link the code to listen.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCP.h"

#include "TCPSocket.h"

#include <cerrno>
#include <poll.h>
#include <sys/socket.h>

using namespace orc_rt;
using namespace orc_rt::sys::posix;

namespace {

Error connectTo(const SocketHandle &Sock, const addrinfo &AI) {
  if (::connect(Sock.get(), AI.ai_addr, AI.ai_addrlen) == 0)
    return Error::success();
  if (errno != EINTR)
    return make_error<StringError>(opFailed("connect", errno));

  // An interrupted connect carries on in the background, and calling connect
  // again would fail with EALREADY. Instead, wait for it to finish, then ask
  // how it went.
  pollfd PFD = {Sock.get(), POLLOUT, 0};
  while (::poll(&PFD, 1, -1) == -1)
    if (errno != EINTR)
      return make_error<StringError>(opFailed("poll", errno));

  int SockErr = 0;
  socklen_t Len = sizeof(SockErr);
  if (::getsockopt(Sock.get(), SOL_SOCKET, SO_ERROR, &SockErr, &Len) != 0)
    return make_error<StringError>(opFailed("getsockopt(SO_ERROR)", errno));
  if (SockErr != 0)
    return make_error<StringError>(opFailed("connect", SockErr));
  return Error::success();
}

} // namespace

namespace orc_rt {

Expected<SocketHandle> tcpConnect(std::string_view Host,
                                  uint16_t Port) noexcept {
  auto Addrs = resolveTCP(Host, Port, HostKind::NameOrAddress, "connect to");
  if (!Addrs)
    return Addrs.takeError();

  AttemptLog Log;
  for (const addrinfo *AI = Addrs->get(); AI; AI = AI->ai_next) {
    auto Sock = openSocket(*AI);
    if (!Sock) {
      Log.add(*AI, toString(Sock.takeError()));
      continue;
    }
    if (auto Err = connectTo(*Sock, *AI)) {
      Log.add(*AI, toString(std::move(Err)));
      continue;
    }
    return std::move(*Sock);
  }

  return Log.fail("Could not connect to " + endpointString(Host, Port));
}

} // namespace orc_rt
