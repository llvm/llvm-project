//===- TCPSocket.cpp - TCP socket helpers on POSIX ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by the POSIX TCP connect and listen implementations.
//
//===----------------------------------------------------------------------===//

#include "TCPSocket.h"

#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt-internal/support/sys/Errno.h"

#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

namespace orc_rt::sys::posix {

static std::string formatEndpoint(std::string_view Host,
                                  std::string_view Port) {
  StringOutputStream OS;
  if (Host.find(':') != std::string_view::npos)
    OS << '[' << Host << ']';
  else
    OS << Host;
  OS << ':' << Port;
  return std::move(OS).str();
}

std::string endpointString(std::string_view Host, uint16_t Port) {
  return formatEndpoint(Host, std::to_string(Port));
}

std::string opFailed(const char *Op, int ErrNum) {
  return std::string(Op) + " failed: " + strError(ErrNum);
}

Expected<AddrInfoList> resolveTCP(std::string_view Host, uint16_t Port,
                                  HostKind Kind, const char *Verb) {
  auto Fail = [&](std::string_view Reason) {
    return make_error<StringError>(
        (StringOutputStream() << "Could not " << Verb << ' '
                              << endpointString(Host, Port) << ": " << Reason)
            .str());
  };

  if (Host.empty())
    return Fail("no host given");

  addrinfo Hints = {};
  Hints.ai_family = AF_UNSPEC;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_protocol = IPPROTO_TCP;
  Hints.ai_flags = AI_NUMERICSERV;
  if (Kind == HostKind::NumericAddress)
    Hints.ai_flags |= AI_NUMERICHOST;

  std::string HostStr(Host);
  std::string PortStr = std::to_string(Port);
  addrinfo *AI = nullptr;
  if (int Ret = ::getaddrinfo(HostStr.c_str(), PortStr.c_str(), &Hints, &AI)) {
    // With AI_NUMERICHOST, a host that is not a numeric address is reported as
    // one that could not be found. Say what is actually wrong.
    if (Ret == EAI_NONAME && Kind == HostKind::NumericAddress)
      return Fail((StringOutputStream()
                   << "\"" << Host << "\" is not a numeric address")
                      .str());
    // errno says what went wrong only for EAI_SYSTEM.
    if (Ret == EAI_SYSTEM)
      return Fail(opFailed("getaddrinfo", errno));
    return Fail(std::string("getaddrinfo failed: ") + ::gai_strerror(Ret));
  }
  assert(AI && "getaddrinfo succeeded without resolving anything?");
  return AddrInfoList(AI);
}

Expected<SocketHandle> openSocket(const addrinfo &AI) {
  SocketHandle Sock(::socket(AI.ai_family, AI.ai_socktype, AI.ai_protocol));
  if (!Sock)
    return make_error<StringError>(opFailed("socket", errno));
  if (auto Err = setCloseOnExec(Sock.get()))
    return std::move(Err);
  return Sock;
}

Error setCloseOnExec(int FD) {
  // Set after the fact rather than with SOCK_CLOEXEC, which Darwin lacks. So a
  // thread that forks and execs in between can still leak the descriptor into
  // the child.
  int Flags = ::fcntl(FD, F_GETFD);
  if (Flags == -1)
    return make_error<StringError>(opFailed("fcntl(F_GETFD)", errno));
  if (::fcntl(FD, F_SETFD, Flags | FD_CLOEXEC) == -1)
    return make_error<StringError>(opFailed("fcntl(F_SETFD)", errno));
  return Error::success();
}

void AttemptLog::add(const addrinfo &AI, std::string_view Reason) {
  if (!Attempts.empty())
    Attempts += "; ";

  char Host[NI_MAXHOST];
  char Port[NI_MAXSERV];
  if (::getnameinfo(AI.ai_addr, AI.ai_addrlen, Host, sizeof(Host), Port,
                    sizeof(Port), NI_NUMERICHOST | NI_NUMERICSERV) == 0)
    Attempts += formatEndpoint(Host, Port);
  else
    Attempts += "<unprintable address>";

  Attempts += ": ";
  Attempts += Reason;
}

Error AttemptLog::fail(std::string_view Summary) const {
  assert(!Attempts.empty() && "No attempts to report");
  return make_error<StringError>(std::string(Summary) + ": " + Attempts);
}

} // namespace orc_rt::sys::posix
