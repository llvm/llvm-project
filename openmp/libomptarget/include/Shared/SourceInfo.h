//===-- SharedSourceInfo.h - Target independent OpenMP target RTL - C++ -*-===//
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

#ifndef OMPTARGET_SHARED_SOURCE_INFO_H
#define OMPTARGET_SHARED_SOURCE_INFO_H

#include <string>

#ifdef _WIN32
constexpr bool OSWindows = true;
#else
constexpr bool OSWindows = false;
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
  const std::string SourceStr;

  /// Location fields extracted from the source information string.
  const std::string Name;
  const std::string Filename;
  const int32_t Line;
  const int32_t Column;

  std::string initStr(const void *Name) {
    if (!Name)
      return ";unknown;unknown;0;0;;";

    std::string Str = std::string(reinterpret_cast<const char *>(Name));
    if (Str.find(';') == std::string::npos)
      return ";" + Str + ";unknown;0;0;;";
    return Str;
  }

  std::string initStr(const ident_t *Loc) {
    if (!Loc)
      return ";unknown;unknown;0;0;;";
    return std::string(reinterpret_cast<const char *>(Loc->psource));
  }

  /// Get n-th substring in an expression separated by ;.
  std::string getSubstring(const unsigned N) const {
    std::size_t Begin = SourceStr.find(';');
    std::size_t End = SourceStr.find(';', Begin + 1);
    for (unsigned I = 0; I < N; I++) {
      Begin = End;
      End = SourceStr.find(';', Begin + 1);
    }
    return SourceStr.substr(Begin + 1, End - Begin - 1);
  };

  /// Get the filename from a full path.
  std::string removePath(const std::string &Path) const {
    std::size_t Pos = (OSWindows) ? Path.rfind('\\') : Path.rfind('/');
    return Path.substr(Pos + 1);
  };

public:
  SourceInfo(const ident_t *Loc)
      : SourceStr(initStr(Loc)), Name(getSubstring(1)),
        Filename(removePath(getSubstring(0))), Line(std::stoi(getSubstring(2))),
        Column(std::stoi(getSubstring(3))) {}

  SourceInfo(const map_var_info_t Name)
      : SourceStr(initStr(Name)), Name(getSubstring(0)),
        Filename(removePath(getSubstring(1))), Line(std::stoi(getSubstring(2))),
        Column(std::stoi(getSubstring(3))) {}

  const char *getName() const { return Name.c_str(); }
  const char *getFilename() const { return Filename.c_str(); }
  const char *getProfileLocation() const { return SourceStr.data(); }
  int32_t getLine() const { return Line; }
  int32_t getColumn() const { return Column; }
  bool isAvailible() const { return (Line || Column); }
};

/// Standalone function for getting the variable name of a mapping.
static inline std::string getNameFromMapping(const map_var_info_t Name) {
  if (!Name)
    return "unknown";

  const std::string NameStr(reinterpret_cast<const char *>(Name));
  std::size_t Begin = NameStr.find(';');
  std::size_t End = NameStr.find(';', Begin + 1);
  return NameStr.substr(Begin + 1, End - Begin - 1);
}

#endif // OMPTARGET_SHARED_SOURCE_INFO_H
