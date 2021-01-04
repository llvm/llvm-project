//===------- SourceInfo.h - Target independent OpenMP target RTL -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Methods used to describe source information in target regions
//
//===----------------------------------------------------------------------===//

#ifndef _SOURCE_INFO_H_
#define _SOURCE_INFO_H_

#include <string>

#ifdef _WIN32
static const bool OS_WINDOWS = true;
#else
static const bool OS_WINDOWS = false;
#endif

/// Type alias for source location information for variable mappings with
/// data layout ";name;filename;row;col;;\0" from clang.
using map_var_info_t = void *;

/// The ident structure that describes a source location from kmp.h. with
/// source location string data as ";filename;function;line;column;;\0".
struct ident_t {
  // Ident_t flags described in kmp.h.
  int32_t reserved_1;
  int32_t flags;
  int32_t reserved_2;
  int32_t reserved_3;
  char const *psource;
};

/// Struct to hold source individual location information.
class SourceInfo {
  /// Underlying string copy of the original source information.
  const std::string sourceStr;

  /// Location fields extracted from the source information string.
  const std::string name;
  const std::string filename;
  const int32_t line;
  const int32_t column;

  std::string initStr(const void *name) {
    if (!name)
      return ";unknown;unknown;0;0;;";
    else
      return std::string(reinterpret_cast<const char *>(name));
  }

  std::string initStr(const ident_t *loc) {
    if (!loc)
      return ";unknown;unknown;0;0;;";
    else
      return std::string(reinterpret_cast<const char *>(loc->psource));
  }

  /// Get n-th substring in an expression separated by ;.
  std::string getSubstring(const int n) const {
    std::size_t begin = sourceStr.find(';');
    std::size_t end = sourceStr.find(';', begin + 1);
    for (int i = 0; i < n; i++) {
      begin = end;
      end = sourceStr.find(';', begin + 1);
    }
    return sourceStr.substr(begin + 1, end - begin - 1);
  };

  /// Get the filename from a full path.
  std::string removePath(const std::string &path) const {
    std::size_t pos = (OS_WINDOWS) ? path.rfind('\\') : path.rfind('/');
    return path.substr(pos + 1);
  };

public:
  SourceInfo(const ident_t *loc)
      : sourceStr(initStr(loc)), name(getSubstring(1)),
        filename(removePath(getSubstring(0))), line(std::stoi(getSubstring(2))),
        column(std::stoi(getSubstring(3))) {}

  SourceInfo(const map_var_info_t name)
      : sourceStr(initStr(name)), name(getSubstring(0)),
        filename(removePath(getSubstring(1))), line(std::stoi(getSubstring(2))),
        column(std::stoi(getSubstring(3))) {}

  const char *getName() const { return name.c_str(); }
  const char *getFilename() const { return filename.c_str(); }
  int32_t getLine() const { return line; }
  int32_t getColumn() const { return column; }
  bool isAvailible() const { return (line || column); }
};

/// Standalone function for getting the variable name of a mapping.
static inline std::string getNameFromMapping(const map_var_info_t name) {
  if (!name)
    return "unknown";

  const std::string name_str(reinterpret_cast<const char *>(name));
  std::size_t begin = name_str.find(';');
  std::size_t end = name_str.find(';', begin + 1);
  return name_str.substr(begin + 1, end - begin - 1);
}

#endif
