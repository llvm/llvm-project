//===-- UriParser.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_UriParser_h_
#define utility_UriParser_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes

class UriParser {
public:
  // Parses
  // RETURN VALUE
  //   if url is valid, function returns true and
  //   scheme/hostname/port/path are set to the parsed values
  //   port it set to -1 if it is not included in the URL
  //
  //   if the url is invalid, function returns false and
  //   output parameters remain unchanged
  static bool Parse(const std::string &uri, std::string &scheme,
                    std::string &hostname, int &port, std::string &path);
};

#endif // utility_UriParser_h_
