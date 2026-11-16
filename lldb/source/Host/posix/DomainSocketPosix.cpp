//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/DomainSocketPosix.h"
#include "lldb/Utility/LLDBLog.h"

#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <fcntl.h>
#include <memory>
#include <sys/socket.h>

using namespace lldb;
using namespace lldb_private;

llvm::Expected<DomainSocket::Pair> DomainSocketPosix::CreatePair() {
  int sockets[2];
  int type = SOCK_STREAM;
#ifdef SOCK_CLOEXEC
  type |= SOCK_CLOEXEC;
#endif
  if (socketpair(AF_UNIX, type, 0, sockets) == -1)
    return llvm::errorCodeToError(llvm::errnoAsErrorCode());

#ifndef SOCK_CLOEXEC
  for (int s : sockets) {
    int r = fcntl(s, F_SETFD, FD_CLOEXEC | fcntl(s, F_GETFD));
    assert(r == 0);
    (void)r;
  }
#endif

#if defined(SO_NOSIGPIPE)
  Log *log = GetLog(LLDBLog::Host);
  if (Socket::SetOption(sockets[0], SOL_SOCKET, SO_NOSIGPIPE, 1) == -1)
    LLDB_LOG(log, "failed to set NO_SIGPIPE on fd {0}: {1}", sockets[0],
             llvm::sys::StrError());
  if (Socket::SetOption(sockets[1], SOL_SOCKET, SO_NOSIGPIPE, 1) == -1)
    LLDB_LOG(log, "failed to set NO_SIGPIPE on fd {0}: {1}", sockets[1],
             llvm::sys::StrError());
#endif

  return Pair(std::unique_ptr<DomainSocket>(new DomainSocketPosix(
                  ProtocolUnixDomain, sockets[0], /*should_close=*/true)),
              std::unique_ptr<DomainSocket>(new DomainSocketPosix(
                  ProtocolUnixDomain, sockets[1], /*should_close=*/true)));
}
