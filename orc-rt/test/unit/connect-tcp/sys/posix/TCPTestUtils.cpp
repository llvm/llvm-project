//===- TCPTestUtils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// POSIX definitions of connect-tcp/TCPTestUtils.h.
//
//===----------------------------------------------------------------------===//

#include "connect-tcp/TCPTestUtils.h"

#include "orc-rt-internal/support/sys/Errno.h"
#include "orc-rt/connect-socket/SocketHandle.h"

#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <utility>

namespace orc_rt::test {

static const char *familyName(IPFamily F) {
  return F == IPFamily::IPv4 ? "IPv4" : "IPv6";
}

/// Tries to listen on Addr. Returns the call that failed and its errno, or
/// nullopt if listening succeeded.
///
/// Each errno is read as its call fails, before Sock is closed: close can
/// change errno.
static std::optional<std::pair<const char *, int>>
tryListen(const sockaddr_storage &Addr, socklen_t AddrLen) {
  SocketHandle Sock(::socket(Addr.ss_family, SOCK_STREAM, IPPROTO_TCP));
  if (!Sock)
    return std::make_pair("socket", errno);
  if (::bind(Sock.get(), reinterpret_cast<const sockaddr *>(&Addr), AddrLen) !=
      0)
    return std::make_pair("bind", errno);
  if (::listen(Sock.get(), 1) != 0)
    return std::make_pair("listen", errno);
  return std::nullopt;
}

std::optional<std::string> whyLoopbackUnavailable(IPFamily F) {
  sockaddr_storage Addr = {};
  socklen_t AddrLen;
  if (F == IPFamily::IPv4) {
    auto &A = reinterpret_cast<sockaddr_in &>(Addr);
    A.sin_family = AF_INET;
    A.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    AddrLen = sizeof(A);
  } else {
    auto &A = reinterpret_cast<sockaddr_in6 &>(Addr);
    A.sin6_family = AF_INET6;
    A.sin6_addr = in6addr_loopback;
    AddrLen = sizeof(A);
  }

  auto Failure = tryListen(Addr, AddrLen);
  if (!Failure)
    return std::nullopt;

  // The errors LLDB's socket tests treat as "no loopback here"; see
  // lldb/unittests/TestingSupport/Host/SocketTestUtilities.cpp.
  auto [Op, ErrNum] = *Failure;
  if (ErrNum != EAFNOSUPPORT && ErrNum != EADDRNOTAVAIL && ErrNum != EPERM)
    return std::nullopt;

  return std::string(familyName(F)) + " loopback unavailable: " + Op +
         " failed: " + sys::strError(ErrNum);
}

bool localhostResolvesTo(IPFamily F) {
  addrinfo Hints = {};
  Hints.ai_family = F == IPFamily::IPv4 ? AF_INET : AF_INET6;
  Hints.ai_socktype = SOCK_STREAM;
  addrinfo *AI = nullptr;
  if (::getaddrinfo("localhost", nullptr, &Hints, &AI) != 0)
    return false;
  ::freeaddrinfo(AI);
  return true;
}

} // namespace orc_rt::test
