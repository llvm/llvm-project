//===- TCPConnect.cpp - TCP connect on Windows (stub) ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placeholder until Windows supports TCP: connecting always fails.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/connect-tcp/TCP.h"

namespace orc_rt {

Expected<SocketHandle> tcpConnect(std::string_view, uint16_t) noexcept {
  return make_error<StringError>(
      "TCP connections are not supported on Windows yet");
}

} // namespace orc_rt
