//===- SocketTestUtils.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// POSIX definitions of bedrock/SocketTestUtils.h.
//
//===----------------------------------------------------------------------===//

#include "bedrock/SocketTestUtils.h"

#include "orc-rt-internal/support/sys/Errno.h"

#include <cerrno>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace orc_rt::test {

static Error makeError(const char *Op, int ErrNum) {
  return make_error<StringError>(std::string(Op) + ": " +
                                 sys::strError(ErrNum));
}

std::optional<NativeSocketHandle> makeNativeSocket() {
  // Unbound, so this needs no network, peer or filesystem entry.
  NativeSocketHandle H = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (H == InvalidNativeSocketHandle)
    return std::nullopt;
  return H;
}

std::optional<NativeSocketHandle> makeNativeNonStreamSocket() {
  NativeSocketHandle H = ::socket(AF_UNIX, SOCK_DGRAM, 0);
  if (H == InvalidNativeSocketHandle)
    return std::nullopt;
  return H;
}

bool isNativeSocketOpen(NativeSocketHandle H) {
  // A zero-length send moves no data and needs no peer. An unconnected socket
  // refuses it with ENOTCONN, which still says the descriptor is there; only a
  // closed one reports EBADF.
  return ::send(H, "", 0, 0) != -1 || errno != EBADF;
}

void closeNativeSocket(NativeSocketHandle H) { ::close(H); }

Expected<std::pair<SocketHandle, SocketHandle>> makeStreamSocketPair() {
  // AF_UNIX gives the close semantics the header requires, and its default
  // buffers are far smaller than StallingPayloadSize.
  int FDs[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, FDs) != 0)
    return makeError("socketpair", errno);
  return std::make_pair(SocketHandle(FDs[0]), SocketHandle(FDs[1]));
}

Error sendAll(NativeSocketHandle H, const char *Buf, size_t Size) {
  while (Size) {
    // MSG_NOSIGNAL: a peer that has gone should fail the send, not kill the
    // test with SIGPIPE.
    ssize_t N = ::send(H, Buf, Size, MSG_NOSIGNAL);
    if (N < 0) {
      if (errno == EINTR)
        continue;
      return makeError("send", errno);
    }
    Buf += N;
    Size -= N;
  }
  return Error::success();
}

Expected<size_t> recvAll(NativeSocketHandle H, char *Buf, size_t Size) {
  size_t Got = 0;
  while (Got < Size) {
    ssize_t N = ::recv(H, Buf + Got, Size - Got, 0);
    if (N == 0)
      return Got;
    if (N < 0) {
      if (errno == EINTR)
        continue;
      return makeError("recv", errno);
    }
    Got += N;
  }
  return Got;
}

} // namespace orc_rt::test
