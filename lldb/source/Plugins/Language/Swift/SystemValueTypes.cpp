//===-- SystemValueTypes.cpp ------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SystemValueTypes.h"

#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

#include <algorithm>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

bool lldb_private::formatters::swift::FilePath_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {

  // public struct FilePath {
  //   internal var _storage: SystemString
  // }
  // internal struct SystemString {
  //   internal var nullTerminatedStorage: [SystemChar]
  // }
  // internal struct SystemChar {
  //   internal var rawValue: CInterop.PlatformChar  // Int8 on Unix, UInt16 on Windows
  // }

  static constexpr llvm::StringLiteral g__storage("_storage");
  static constexpr llvm::StringLiteral g_nullTerminatedStorage(
      "nullTerminatedStorage");
  static constexpr llvm::StringLiteral g_rawValue("rawValue");

  ValueObjectSP storage_sp =
      valobj.GetChildAtNamePath({g__storage, g_nullTerminatedStorage});
  if (!storage_sp)
    return false;

  storage_sp = storage_sp->GetSyntheticValue();
  if (!storage_sp)
    return false;

  uint32_t num_children = storage_sp->GetNumChildrenIgnoringErrors();
  if (num_children == 0)
    return false;

  // The storage array is null-terminated, so read num_children - 1 elements.
  // Also guard against overly long paths.
  const uint32_t max_path_length = 4096;
  uint32_t path_length = std::min(num_children - 1, max_path_length);

  std::string path;
  path.reserve(path_length);

  for (uint32_t i = 0; i < path_length; ++i) {
    ValueObjectSP char_sp = storage_sp->GetChildAtIndex(i);
    if (!char_sp)
      return false;

    // Get the rawValue from SystemChar
    ValueObjectSP raw_value_sp = char_sp->GetChildAtNamePath({g_rawValue});
    if (!raw_value_sp)
      return false;

    raw_value_sp = raw_value_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!raw_value_sp)
      return false;

    int64_t byte_value = raw_value_sp->GetValueAsSigned(0);

    // Stop if we reach an early null-terminator
    if (byte_value == 0)
      break;

    // On Windows, SystemChar is UInt16 (UTF-16). This truncates non-ASCII.
    path.push_back(static_cast<char>(byte_value));
  }

  stream << '"' << path << '"';
  return true;
}

bool lldb_private::formatters::swift::SystemChar_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {

  // internal struct SystemChar {
  //   internal var rawValue: CInterop.PlatformChar // Int8 on Unix, UInt16 on Windows
  // }

  static constexpr llvm::StringLiteral g_rawValue("rawValue");

  ValueObjectSP raw_value_sp = valobj.GetChildAtNamePath({g_rawValue});
  if (!raw_value_sp)
    return false;

  raw_value_sp = raw_value_sp->GetQualifiedRepresentationIfAvailable(
      lldb::eDynamicDontRunTarget, true);
  if (!raw_value_sp)
    return false;

  int64_t byte_value = raw_value_sp->GetValueAsSigned(0);

  // For printable ASCII characters (0x20-0x7E), show the character.
  // For other values, just show the numeric value.
  if (byte_value >= 0x20 && byte_value <= 0x7E) {
    stream.Printf("'%c' (%d)", static_cast<char>(byte_value),
                  static_cast<int>(byte_value));
  } else if (byte_value == 0) {
    stream.Printf("'\\0' (0)");
  } else {
    stream.Printf("%d", static_cast<int>(byte_value));
  }

  return true;
}
