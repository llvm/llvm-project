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

#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>

using namespace orc_rt;

std::optional<NativeSocketHandle> makeNativeSocket() {
  // Unbound, so this needs no network, peer or filesystem entry.
  NativeSocketHandle H = ::socket(AF_UNIX, SOCK_STREAM, 0);
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
