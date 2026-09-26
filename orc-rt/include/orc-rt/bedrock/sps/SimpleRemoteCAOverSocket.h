//===- SimpleRemoteCAOverSocket.h - SimpleRemote CA over socket -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A ControllerAccess speaking the SimpleRemote protocol over a connected
// socket.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_SIMPLEREMOTECAOVERSOCKET_H
#define ORC_RT_BEDROCK_SPS_SIMPLEREMOTECAOVERSOCKET_H

#include "orc-rt/bedrock/Session.h"
#include "orc-rt/bedrock/SocketHandle.h"
#include "orc-rt/support/Error.h"

#include <memory>

namespace orc_rt {

/// Creates a ControllerAccess that carries SimpleRemote messages over Sock,
/// taking ownership of it. Fails if Sock is not a stream socket. Sock must be
/// connected.
///
/// The result is ready to hand to Session::attach, which is what starts the
/// conversation; nothing is sent before then.
///
/// A factory rather than a class, because how the messages are pumped is the
/// platform's business and not the caller's: the POSIX implementation owns a
/// reactor thread built on poll(2) and a wake socket, and a Windows one will
/// need something else entirely. The wire format is the same either way, and
/// matches LLVM's SimpleRemoteEPC.
Expected<std::shared_ptr<Session::ControllerAccess>>
createSimpleRemoteCAOverSocket(Session &S, SocketHandle Sock);

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SPS_SIMPLEREMOTECAOVERSOCKET_H
