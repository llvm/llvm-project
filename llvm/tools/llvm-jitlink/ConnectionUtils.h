//===- ConnectionUtils.h - Connection helpers for llvm-jitlink tools -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Connection-establishment helpers shared between llvm-jitlink and
// llvm-jitlink-executor. Header-only: the two tools are separate
// executables and never link against each other, so there is no shared
// object to put these definitions in.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_JITLINK_CONNECTIONUTILS_H
#define LLVM_TOOLS_LLVM_JITLINK_CONNECTIONUTILS_H

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ExponentialBackoff.h"

#include <cstring>
#include <memory>
#include <string>

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

namespace llvm {

/// ConnectFn must make one attempt to connect to the executor, returning an
/// Expected<T>, where T is the connection handle type (e.g. T=int for a
/// socket).
/// If Retry is non-null this function will retry with exponential backoff
/// until Retry's timeout elapses.
/// If Retry is null this function will make a single attempt to connect and
/// return the result.
template <typename ConnectorFn>
decltype(auto)
connectWithRetry(ConnectorFn &&Connect,
                 std::unique_ptr<ExponentialBackoff> Retry = nullptr) {
  if (!Retry)
    return Connect();

  while (true) {
    if (auto Handle = Connect())
      return Handle; // Return success.
    else if (Retry->waitForNextAttempt())
      consumeError(Handle.takeError()); // Ignore error and retry.
    else
      return Handle; // Returns error.
  }

  llvm_unreachable("should exit from loop above");
}

#ifdef LLVM_ON_UNIX

/// Connects to Host:PortStr over TCP. Returns the connected socket
/// descriptor.
inline Expected<int> connectTCPSocket(StringRef Host, StringRef PortStr) {
  addrinfo Hints{};
  Hints.ai_family = AF_INET;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_flags = AI_NUMERICSERV;

  addrinfo *AI;
  if (int EC =
          getaddrinfo(Host.str().c_str(), PortStr.str().c_str(), &Hints, &AI))
    return make_error<StringError>(Twine("Address resolution failed for '") +
                                       Host + ":" + PortStr +
                                       "': " + gai_strerror(EC),
                                   inconvertibleErrorCode());
  auto FreeAI = scope_exit([&]() { freeaddrinfo(AI); });

  // Cycle through the returned addrinfo structures and connect to the first
  // reachable endpoint.
  int SockFD = -1;
  addrinfo *Server;
  for (Server = AI; Server != nullptr; Server = Server->ai_next) {
    // socket might fail, e.g. if the address family is not supported. Skip
    // to the next addrinfo structure in such a case.
    if ((SockFD = socket(Server->ai_family, Server->ai_socktype,
                         Server->ai_protocol)) < 0)
      continue;

    // If connect returns 0 we exit the loop with a working socket.
    if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0)
      break;

    close(SockFD);
  }

  // If we reached the end of the loop without connecting to a valid
  // endpoint, report the last error logged by socket() or connect().
  if (Server == nullptr)
    return make_error<StringError>(Twine("Failed to connect to '") + Host +
                                       ":" + PortStr +
                                       "': " + std::strerror(errno),
                                   inconvertibleErrorCode());

  return SockFD;
}

/// Binds and listens for a single incoming TCP connection on Host:PortStr.
/// Host may be empty to bind the wildcard address; PortStr may be "0" to
/// request an OS-assigned ephemeral port. Returns the listening socket
/// descriptor -- pass it to acceptTCPConnection to accept the connection.
/// If ResolvedPortStr is non-null it is set to the concrete bound port,
/// which is required to learn the real port when PortStr was "0".
inline Expected<int> listenTCPSocket(StringRef Host, StringRef PortStr,
                                     std::string *ResolvedPortStr = nullptr) {
  addrinfo Hints{};
  Hints.ai_family = AF_INET;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_flags = AI_PASSIVE;

  std::string HostStr = Host.str();
  const char *Node = Host.empty() ? nullptr : HostStr.c_str();

  addrinfo *AI;
  if (int EC = getaddrinfo(Node, PortStr.str().c_str(), &Hints, &AI))
    return make_error<StringError>(Twine("Address resolution failed for '") +
                                       Host + ":" + PortStr +
                                       "': " + gai_strerror(EC),
                                   inconvertibleErrorCode());
  auto FreeAI = scope_exit([&]() { freeaddrinfo(AI); });

  int SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol);
  if (SockFD < 0)
    return make_error<StringError>(Twine("Error creating socket: ") +
                                       std::strerror(errno),
                                   inconvertibleErrorCode());
  auto CloseSockFD = scope_exit([&]() { close(SockFD); });

  // Avoid "Address already in use" errors.
  const int Yes = 1;
  if (setsockopt(SockFD, SOL_SOCKET, SO_REUSEADDR, &Yes, sizeof(int)) == -1)
    return make_error<StringError>(Twine("Error calling setsockopt: ") +
                                       std::strerror(errno),
                                   inconvertibleErrorCode());

  if (bind(SockFD, AI->ai_addr, AI->ai_addrlen) < 0)
    return make_error<StringError>(Twine("Error binding to port '") + PortStr +
                                       "': " + std::strerror(errno),
                                   inconvertibleErrorCode());

  static constexpr int ConnectionQueueLen = 1;
  if (listen(SockFD, ConnectionQueueLen) < 0)
    return make_error<StringError>(Twine("Error listening on port '") +
                                       PortStr + "': " + std::strerror(errno),
                                   inconvertibleErrorCode());

  if (ResolvedPortStr) {
    sockaddr_in BoundAddr{};
    socklen_t BoundAddrLen = sizeof(BoundAddr);
    if (getsockname(SockFD, reinterpret_cast<sockaddr *>(&BoundAddr),
                    &BoundAddrLen) < 0)
      return make_error<StringError>(Twine("Error resolving bound port: ") +
                                         std::strerror(errno),
                                     inconvertibleErrorCode());
    *ResolvedPortStr = std::to_string(ntohs(BoundAddr.sin_port));
  }

  CloseSockFD.release();
  return SockFD;
}

/// Accepts a single incoming connection on ListeningSockFD (as returned by
/// listenTCPSocket) and closes the listening socket, whether or not the
/// accept succeeds -- listenTCPSocket only ever queues one connection.
/// Returns the accepted connection's socket descriptor.
inline Expected<int> acceptTCPConnection(int ListeningSockFD) {
  auto CloseListeningSockFD = scope_exit([&]() { close(ListeningSockFD); });

  int FD = accept(ListeningSockFD, nullptr, nullptr);
  if (FD < 0)
    return make_error<StringError>(Twine("Error accepting connection: ") +
                                       std::strerror(errno),
                                   inconvertibleErrorCode());

  return FD;
}

#endif // LLVM_ON_UNIX

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_JITLINK_CONNECTIONUTILS_H
