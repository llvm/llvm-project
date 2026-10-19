//===- TCPSocket.h - TCP socket helpers on POSIX ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by the POSIX TCP connect and listen implementations.
//
// Private to orc-rt-connect-tcp: this lives beside its users rather than under
// include/orc-rt-internal/, so that nothing outside lib/connect-tcp can include
// it.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_LIB_CONNECT_TCP_SYS_POSIX_TCPSOCKET_H
#define ORC_RT_LIB_CONNECT_TCP_SYS_POSIX_TCPSOCKET_H

#include "orc-rt/connect-socket/SocketHandle.h"
#include "orc-rt/support/Error.h"

#include <cstdint>
#include <memory>
#include <netdb.h>
#include <string>
#include <string_view>

namespace orc_rt::sys::posix {

struct AddrInfoDeleter {
  void operator()(addrinfo *AI) const noexcept { ::freeaddrinfo(AI); }
};

/// The list getaddrinfo returns, freed on destruction.
using AddrInfoList = std::unique_ptr<addrinfo, AddrInfoDeleter>;

/// Formats Host and Port as "<host>:<port>", bracketing a Host that contains
/// ':' so that an IPv6 address stays readable.
std::string endpointString(std::string_view Host, uint16_t Port);

/// Formats "<Op> failed: <description of ErrNum>".
std::string opFailed(const char *Op, int ErrNum);

/// What a host passed to resolveTCP may be.
enum class HostKind {
  /// A name or a numeric address. A name may resolve to several addresses.
  NameOrAddress,
  /// A numeric address only, which resolves to exactly one address, without
  /// consulting the system's name resolution.
  NumericAddress,
};

/// Resolves Host and Port to the TCP addresses to try, in the order to try
/// them. Errors if Host is empty, is not of kind Kind, or cannot be resolved.
/// Verb says what the caller wants the addresses for ("connect to", "listen
/// on"), for the error.
Expected<AddrInfoList> resolveTCP(std::string_view Host, uint16_t Port,
                                  HostKind Kind, const char *Verb);

/// Opens a close-on-exec socket of AI's family, type and protocol.
Expected<SocketHandle> openSocket(const addrinfo &AI);

/// Sets close-on-exec on FD.
Error setCloseOnExec(int FD);

/// Collects the failed attempts of an operation that tries several addresses,
/// so that the error, if none succeeds, reports each.
class AttemptLog {
public:
  /// Records that the attempt on AI's address failed for Reason.
  void add(const addrinfo &AI, std::string_view Reason);

  /// An error for when every attempt has failed: Summary, then each attempt.
  Error fail(std::string_view Summary) const;

private:
  std::string Attempts;
};

} // namespace orc_rt::sys::posix

#endif // ORC_RT_LIB_CONNECT_TCP_SYS_POSIX_TCPSOCKET_H
