//===- SocketConnector.cpp - Socket connector on Windows --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SocketConnector.h"

using namespace orc_rt;

namespace orc_rt {

Error registerSocketConnector(ConnectorRegistry &R) noexcept {
  return R.registerConnector(
      "socket",
      [](const ConnectionSpec &, Session &, BootstrapInfo &&) noexcept -> Error {
        return make_error<StringError>("socket connector not implemented");
      });
}

} // namespace orc_rt
