//===- SimpleRemoteCAOverSocket.cpp - Windows stub ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placeholder until Windows supports the socket transport: creating the
// ControllerAccess always fails.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-socket/sps/SimpleRemoteCAOverSocket.h"

namespace orc_rt {

Expected<std::shared_ptr<Session::ControllerAccess>>
createSimpleRemoteCAOverSocket(Session &, VettedPeer<SocketHandle>) {
  return make_error<StringError>(
      "The SimpleRemote socket transport is not supported on Windows yet");
}

} // namespace orc_rt
