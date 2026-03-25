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

#include "lldb/Target/Target.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

#include <algorithm>
#include <cinttypes>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

// public struct FilePath {
//   internal var _storage: SystemString
// }
// internal struct SystemString {
//   internal var nullTerminatedStorage: [SystemChar]
// }
// internal struct SystemChar {
//   internal var rawValue: CInterop.PlatformChar // Int8 on Unix, UInt16 on Windows
// }

static constexpr llvm::StringLiteral g__storage("_storage");
static constexpr llvm::StringLiteral g_nullTerminatedStorage("nullTerminatedStorage");
static constexpr llvm::StringLiteral g_rawValue("rawValue");

// Given a synthetic array of `SystemChar` elements, extract each `rawValue`
// into a vector `values` and detect whether the elements are wide (16-bit).
// Returns false on failure.
static bool ExtractSystemChars(ValueObjectSP storage_sp,
                               std::vector<uint64_t> &values,
                               bool &is_wide_char) {
  storage_sp = storage_sp->GetSyntheticValue();
  if (!storage_sp)
    return false;

  uint32_t num_children = storage_sp->GetNumChildrenIgnoringErrors();
  if (num_children == 0)
    return false;

  // Detect if we're dealing with 16-bit characters (Windows) by
  // checking the byte size of the first character's rawValue.
  is_wide_char = false;
  if (ValueObjectSP first_char_sp = storage_sp->GetChildAtIndex(0)) {
    if (ValueObjectSP first_raw_sp =
            first_char_sp->GetChildAtNamePath({g_rawValue})) {
      if (auto byte_size = first_raw_sp->GetByteSize())
        is_wide_char = *byte_size > 1;
    }
  }

  // Use the target's max-children-count, or 4096 (~PATH_MAX) by default
  uint32_t max_length = 4096;
  if (auto target_sp = storage_sp->GetTargetSP())
    max_length = target_sp->GetMaximumNumberOfChildrenToDisplay();
  if (max_length == 0)
    max_length = 4096;
  uint32_t length = std::min(num_children, max_length);

  values.clear();
  values.reserve(length);

  for (uint32_t i = 0; i < length; ++i) {
    ValueObjectSP char_sp = storage_sp->GetChildAtIndex(i);
    if (!char_sp)
      return false;

    ValueObjectSP raw_value_sp = char_sp->GetChildAtNamePath({g_rawValue});
    if (!raw_value_sp)
      return false;

    raw_value_sp = raw_value_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicDontRunTarget, true);
    if (!raw_value_sp)
      return false;

    values.push_back(raw_value_sp->GetValueAsUnsigned(0));
  }

  return true;
}

// Extract SystemChar values from a FilePath's _storage.nullTerminatedStorage.
static bool ExtractFilePathChars(ValueObject &valobj,
                                 std::vector<uint64_t> &values,
                                 bool &is_wide_char) {
  ValueObjectSP storage_sp =
      valobj.GetChildAtNamePath({g__storage, g_nullTerminatedStorage});
  if (!storage_sp)
    return false;
  return ExtractSystemChars(storage_sp, values, is_wide_char);
}

// Check if all characters before the null-terminator are printable ASCII.
static bool IsAllPrintableASCII(const std::vector<uint64_t> &values) {
  for (uint64_t value : values) {
    if (value == 0)
      break;
    if (value < 0x20 || value > 0x7E)
      return false;
  }
  return true;
}

bool lldb_private::formatters::swift::FilePathSummaryProvider::FormatObject(
    ValueObject *valobj, std::string &dest,
    const TypeSummaryOptions &options) {
  if (!valobj)
    return false;

  std::vector<uint64_t> values;
  bool is_wide_char = false;
  if (!ExtractFilePathChars(*valobj, values, is_wide_char))
    return false;

  std::string path;
  path.reserve(values.size());

  for (uint64_t value : values) {
    // Stop at null-terminator.
    if (value == 0)
      break;
    // On Windows, SystemChar is UInt16 (UTF-16). This truncates non-ASCII.
    path.push_back(static_cast<char>(value));
  }

  dest = "\"" + path + "\"";
  return true;
}

bool lldb_private::formatters::swift::FilePathSummaryProvider::
    DoesPrintChildren(ValueObject *valobj) const {
  if (!valobj)
    return false;

  std::vector<uint64_t> values;
  bool is_wide_char = false;
  if (!ExtractFilePathChars(*valobj, values, is_wide_char))
    return false;

  return !IsAllPrintableASCII(values);
}

std::string
lldb_private::formatters::swift::FilePathSummaryProvider::GetDescription() {
  return "FilePath summary provider";
}

std::string
lldb_private::formatters::swift::FilePathSummaryProvider::GetName() {
  return "FilePathSummaryProvider";
}

bool lldb_private::formatters::swift::SystemString_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {

  // internal struct SystemString {
  //   internal var nullTerminatedStorage: [SystemChar]
  // }

  ValueObjectSP storage_sp =
      valobj.GetChildAtNamePath({g_nullTerminatedStorage});
  if (!storage_sp)
    return false;

  std::vector<uint64_t> values;
  bool is_wide_char = false;
  if (!ExtractSystemChars(storage_sp, values, is_wide_char))
    return false;

  // Display as an array of bytes, e.g: ['/', 'b', 'i', 'n', 0x00]
  // Printable ASCII (0x20-0x7E) are shown as characters, others as hex.

  stream << '[';

  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      stream << ", ";

    uint64_t value = values[i];

    if (value >= 0x20 && value <= 0x7E) {
      stream.Printf("'%c'", static_cast<char>(value));
    } else if (is_wide_char) {
      stream.Printf("0x%04" PRIX64, value & 0xFFFF);
    } else {
      stream.Printf("0x%02" PRIX64, value & 0xFF);
    }
  }

  stream << ']';
  return true;
}
