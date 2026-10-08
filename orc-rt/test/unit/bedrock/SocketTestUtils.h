//===- SocketTestUtils.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sockets for tests of the socket APIs, defined once per system under
// sys/<system>/ so that a test needing one stays portable.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_BEDROCK_SOCKETTESTUTILS_H
#define ORC_RT_UNITTEST_BEDROCK_SOCKETTESTUTILS_H

#include "orc-rt/bedrock/SocketHandle.h"
#include "orc-rt/support/Error.h"

#include <cstddef>
#include <optional>
#include <utility>

namespace orc_rt::test {

/// Creates a socket for a test to own, or nullopt if the system refuses one.
/// The socket is neither bound nor connected.
std::optional<NativeSocketHandle> makeNativeSocket();

/// Creates a socket that is not a stream socket, for a test to own, or nullopt
/// if the system refuses one. The socket is neither bound nor connected.
std::optional<NativeSocketHandle> makeNativeNonStreamSocket();

/// True if H names a socket this process still has open.
///
/// Only meaningful while nothing else in the process is opening sockets: a
/// closed handle's value can be reissued to the next caller, which is
/// indistinguishable from the original still being open.
bool isNativeSocketOpen(NativeSocketHandle H);

/// Closes H, which must be open and owned by no SocketHandle.
void closeNativeSocket(NativeSocketHandle H);

/// A payload size that neither direction of a makeStreamSocketPair pair can
/// buffer, so writing this much without the peer reading stalls the writer.
inline constexpr size_t StallingPayloadSize = 1 << 20;

/// Creates a connected pair of blocking stream sockets.
///
/// Tests of the socket transports rely on two further properties, which the
/// definition for each system must provide:
///
///   - Neither direction buffers StallingPayloadSize bytes.
///
///   - Closing one end still delivers everything already sent from it, even
///     if that end has unread data of its own. AF_UNIX sockets do this; a
///     loopback TCP pair may reset the connection instead, discarding it.
Expected<std::pair<SocketHandle, SocketHandle>> makeStreamSocketPair();

/// Sends all Size bytes of Buf over H, which must be blocking.
Error sendAll(NativeSocketHandle H, const char *Buf, size_t Size);

/// Receives Size bytes into Buf from H, which must be blocking. Returns fewer
/// only if the peer closed first.
Expected<size_t> recvAll(NativeSocketHandle H, char *Buf, size_t Size);

} // namespace orc_rt::test

#endif // ORC_RT_UNITTEST_BEDROCK_SOCKETTESTUTILS_H
