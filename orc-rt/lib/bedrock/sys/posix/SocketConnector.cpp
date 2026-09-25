//===- SocketConnector.cpp - Socket connector on POSIX ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Adoption of an inherited socket on POSIX systems.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SocketConnector.h"

#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt-internal/support/sys/Errno.h"
#include "orc-rt/bedrock/sps/SimpleRemoteCAOverSocket.h"

#include <cerrno>
#include <charconv>
#include <sys/socket.h>

using namespace orc_rt;

namespace {

Error socketConnector(ConnectorRegistry::GetAttachInfoFn GetAttachInfo,
                      const ConnectionSpec &CS) noexcept {
  auto BadCS = [&](const std::string &Reason) noexcept {
    return make_error<StringError>((StringOutputStream()
                                    << "Invalid connection spec \"" << CS.str()
                                    << "\": " << Reason)
                                       .str());
  };

  if (CS.action() != "adopt")
    return BadCS("the socket transport supports only the \"adopt\" action");

  std::string_view FDStr = CS.descriptor();
  int FD;
  auto [Ptr, ErrC] =
      std::from_chars(FDStr.data(), FDStr.data() + FDStr.size(), FD);
  if (auto EC = std::make_error_code(ErrC))
    return BadCS(std::string(FDStr) + " is not a descriptor (" + EC.message() +
                 ")");
  if (Ptr != FDStr.data() + FDStr.size())
    return BadCS("trailing characters after file descriptor \"" +
                 std::string(FDStr) + "\"");
  if (FD < 0)
    return BadCS("file descriptor " + std::string(FDStr) + " is negative");

  // A descriptor that is not a socket is not ours to close, so it is checked
  // before being wrapped. One that is a socket is ours from here on, and is
  // closed if anything below fails.
  int Type;
  socklen_t TypeLen = sizeof(Type);
  if (::getsockopt(FD, SOL_SOCKET, SO_TYPE, &Type, &TypeLen) != 0) {
    int ErrNum = errno;
    return BadCS("file descriptor " + std::string(FDStr) +
                 " is not a socket (" + sys::strError(ErrNum) + ")");
  }
  SocketHandle Sock(FD);

  auto AI = GetAttachInfo();
  if (!AI)
    return AI.takeError();
  auto CA = createSimpleRemoteCAOverSocket(AI->S, std::move(Sock));
  if (!CA)
    return CA.takeError();

  AI->S.attach(std::move(*CA), std::move(AI->BI));
  return Error::success();
}

} // namespace

namespace orc_rt {

Error registerSocketConnector(ConnectorRegistry &R) noexcept {
  return R.registerConnector("socket", socketConnector);
}

} // namespace orc_rt
