//===-- UriParser.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Utility/UriParser.h"

// C Includes

// C++ Includes
#include <cstring>

// Other libraries and framework includes
// Project includes
#include "lldb/Host/StringConvert.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// UriParser::Parse
//----------------------------------------------------------------------
bool UriParser::Parse(const std::string &uri, std::string &scheme,
                      std::string &hostname, int &port, std::string &path) {
  std::string tmp_scheme, tmp_hostname, tmp_port, tmp_path;

  static const char *kSchemeSep = "://";
  auto pos = uri.find(kSchemeSep);
  if (pos == std::string::npos)
    return false;

  // Extract path.
  tmp_scheme = uri.substr(0, pos);
  auto host_pos = pos + strlen(kSchemeSep);
  auto path_pos = uri.find('/', host_pos);
  if (path_pos != std::string::npos)
    tmp_path = uri.substr(path_pos);
  else
    tmp_path = "/";

  auto host_port = uri.substr(
      host_pos,
      ((path_pos != std::string::npos) ? path_pos : uri.size()) - host_pos);

  // Extract hostname
  if (host_port[0] == '[') {
    // hostname is enclosed with square brackets.
    pos = host_port.find(']');
    if (pos == std::string::npos)
      return false;

    tmp_hostname = host_port.substr(1, pos - 1);
    host_port.erase(0, pos + 1);
  } else {
    pos = host_port.find(':');
    tmp_hostname = host_port.substr(
        0, (pos != std::string::npos) ? pos : host_port.size());
    host_port.erase(0, (pos != std::string::npos) ? pos : host_port.size());
  }

  // Extract port
  tmp_port = host_port;
  if (!tmp_port.empty()) {
    if (tmp_port[0] != ':')
      return false;
    tmp_port = tmp_port.substr(1);
    bool success = false;
    auto port_tmp =
        StringConvert::ToUInt32(tmp_port.c_str(), UINT32_MAX, 10, &success);
    if (!success || port_tmp > 65535) {
      // there are invalid characters in port_buf
      return false;
    }
    port = port_tmp;
  } else
    port = -1;

  scheme = tmp_scheme;
  hostname = tmp_hostname;
  path = tmp_path;
  return true;
}
