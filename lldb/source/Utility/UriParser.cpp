//===-- UriParser.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/UriParser.h"

#include <string>

#include <cstdint>
#include <tuple>

using namespace lldb_private;

// UriParser::Parse
bool UriParser::Parse(llvm::StringRef uri, llvm::StringRef &scheme,
                      llvm::StringRef &hostname, llvm::Optional<uint16_t> &port,
                      llvm::StringRef &path) {
  llvm::StringRef tmp_scheme, tmp_hostname, tmp_path;

  const llvm::StringRef kSchemeSep("://");
  auto pos = uri.find(kSchemeSep);
  if (pos == std::string::npos)
    return false;

  // Extract path.
  tmp_scheme = uri.substr(0, pos);
  auto host_pos = pos + kSchemeSep.size();
  auto path_pos = uri.find('/', host_pos);
  if (path_pos != std::string::npos)
    tmp_path = uri.substr(path_pos);
  else
    tmp_path = "/";

  auto host_port = uri.substr(
      host_pos,
      ((path_pos != std::string::npos) ? path_pos : uri.size()) - host_pos);

  // Extract hostname
  if (!host_port.empty() && host_port[0] == '[') {
    // hostname is enclosed with square brackets.
    pos = host_port.rfind(']');
    if (pos == std::string::npos)
      return false;

    tmp_hostname = host_port.substr(1, pos - 1);
    host_port = host_port.drop_front(pos + 1);
    if (!host_port.empty() && !host_port.consume_front(":"))
      return false;
  } else {
    std::tie(tmp_hostname, host_port) = host_port.split(':');
  }

  // Extract port
  if (!host_port.empty()) {
    uint16_t port_value = 0;
    if (host_port.getAsInteger(0, port_value))
      return false;
    port = port_value;
  } else
    port = llvm::None;

  scheme = tmp_scheme;
  hostname = tmp_hostname;
  path = tmp_path;
  return true;
}
