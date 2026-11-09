//===- TCPListen.cpp - TCP listen on Windows (stub) -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placeholder until Windows supports TCP: listening always fails.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCP.h"

namespace orc_rt {

Expected<TCPListener> TCPListener::listen(std::string_view, uint16_t) noexcept {
  return make_error<StringError>(
      "TCP listening is not supported on Windows yet");
}

Expected<SocketHandle> TCPListener::accept() noexcept {
  // listen never succeeds here, so there is no listener to accept on. Fail
  // rather than assert, since this stub can't be tested where it is built.
  return make_error<StringError>(
      "TCP listening is not supported on Windows yet");
}

} // namespace orc_rt
