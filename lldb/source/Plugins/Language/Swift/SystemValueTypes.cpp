//===-- SystemValueTypes.cpp ----------------------------------------------===//
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
#include <cinttypes>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

// Writes a character value to the stream with appropriate escaping.
// Printable ASCII characters (0x20-0x7E) are written directly, except for
// backslash and double-quote which are escaped. Non-printable characters
// use "\x{XX}" format (or "\x{XXXX}" on 16-bit wide char platforms).
static void WriteChar(Stream &stream, uint64_t value, bool is_wide_char) {
  // Handle special escape characters
  switch (value) {
  case '\n':
    stream << "\\n";
    return;
  case '\r':
    stream << "\\r";
    return;
  case '\t':
    stream << "\\t";
    return;
  case '\\':
    stream << "\\\\";
    return;
  case '"':
    stream << "\\\"";
    return;
  default:
    break;
  }

  // Printable ASCII (0x20 space through 0x7E tilde)
  if (value >= 0x20 && value <= 0x7E) {
    stream << static_cast<char>(value);
    return;
  }

  // Non-printable: use "\x{XX}" format (or "\x{XXXX}" for 16-bit wide char)
  if (is_wide_char)
    stream.Printf("\\x{%04" PRIX64 "}", value & 0xFFFF);
  else
    stream.Printf("\\x{%02" PRIX64 "}", value & 0xFF);
}

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

  // Detect if we're dealing with 16-bit characters (Windows) by
  // checking the byte size of the first character's rawValue.
  bool is_wide_char = false;
  if (path_length > 0) {
    if (ValueObjectSP first_char_sp = storage_sp->GetChildAtIndex(0)) {
      if (ValueObjectSP first_raw_sp =
              first_char_sp->GetChildAtNamePath({g_rawValue})) {
        if (auto byte_size = first_raw_sp->GetByteSize())
          is_wide_char = *byte_size > 1;
      }
    }
  }

  stream << '"';

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

    uint64_t value = raw_value_sp->GetValueAsUnsigned(0);

    // Stop if we reach an early null-terminator
    if (value == 0)
      break;

    WriteChar(stream, value, is_wide_char);
  }

  stream << '"';
  return true;
}
