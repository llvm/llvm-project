//===- TCPConnectorHelpers.cpp - Shared TCP connector helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by the "tcp" connectors.
//
//===----------------------------------------------------------------------===//

#include "TCPConnectorHelpers.h"

#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt/connect-socket/sps/SimpleRemoteCAOverSocket.h"
#include "orc-rt/connect/VettedPeer.h"

#include <charconv>
#include <limits>

namespace orc_rt::detail {

Error invalidTCPSpec(const ConnectionSpec &CS, std::string_view Reason) {
  return make_error<StringError>((StringOutputStream()
                                  << "Invalid connection spec \"" << CS.str()
                                  << "\": " << Reason)
                                     .str());
}

Expected<TCPEndpoint> parseTCPEndpoint(const ConnectionSpec &CS) {
  std::string_view D = CS.descriptor();
  std::string_view Host;
  std::string_view PortStr;

  if (!D.empty() && D.front() == '[') {
    size_t Close = D.find(']');
    if (Close == std::string_view::npos)
      return invalidTCPSpec(CS, "missing ']' after the IPv6 address");
    Host = D.substr(1, Close - 1);
    if (Close + 1 == D.size() || D[Close + 1] != ':')
      return invalidTCPSpec(CS, "expected ':' and a port after ']'");
    PortStr = D.substr(Close + 2);
  } else {
    size_t Colon = D.find(':');
    if (Colon == std::string_view::npos)
      return invalidTCPSpec(CS, "expected <host>:<port>");
    Host = D.substr(0, Colon);
    PortStr = D.substr(Colon + 1);
    if (PortStr.find(':') != std::string_view::npos)
      return invalidTCPSpec(CS, "an IPv6 address must be bracketed, e.g. "
                                "\"[::1]:20000\"");
  }

  if (Host.empty())
    return invalidTCPSpec(CS, "no host given");

  unsigned Port = 0;
  auto [Ptr, EC] =
      std::from_chars(PortStr.data(), PortStr.data() + PortStr.size(), Port);
  if (PortStr.empty() || EC != std::errc() ||
      Ptr != PortStr.data() + PortStr.size() ||
      Port > std::numeric_limits<uint16_t>::max())
    return invalidTCPSpec(CS, (StringOutputStream()
                               << "\"" << PortStr << "\" is not a port number")
                                  .str());

  if (Port == 0)
    return invalidTCPSpec(
        CS, CS.action() == "listen"
                ? "a listening executor must name its port, since nothing "
                  "would report a port the system chose"
                : "port 0 cannot be connected to");

  return TCPEndpoint{std::string(Host), static_cast<uint16_t>(Port)};
}

Error attachTCPPeer(Session &S, SocketHandle Sock, BootstrapInfo BI) {
  // Unchecked: TCP knows the peer only by its address, which identifies no
  // one.
  auto CA = createSimpleRemoteCAOverSocket(
      S, VettedPeer<SocketHandle>::unchecked(std::move(Sock)));
  if (!CA)
    return CA.takeError();
  S.attach(std::move(*CA), std::move(BI));
  return Error::success();
}

} // namespace orc_rt::detail
