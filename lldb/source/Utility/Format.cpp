//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Format.h"
#include "llvm/ADT/StringRef.h"
#include <iterator>

using namespace lldb;

namespace {
struct FormatInfo {
  Format format;
  const char format_char;  // One or more format characters that can be used for
                           // this format.
  const char *format_name; // Long format name that can be used to specify the
                           // current format
};

static constexpr FormatInfo g_format_infos[] = {
    {eFormatDefault, '\0', "default"},
    {eFormatBoolean, 'B', "boolean"},
    {eFormatBinary, 'b', "binary"},
    {eFormatBytes, 'y', "bytes"},
    {eFormatBytesWithASCII, 'Y', "bytes with ASCII"},
    {eFormatChar, 'c', "character"},
    {eFormatCharPrintable, 'C', "printable character"},
    {eFormatComplexFloat, 'F', "complex float"},
    {eFormatCString, 's', "c-string"},
    {eFormatDecimal, 'd', "decimal"},
    {eFormatEnum, 'E', "enumeration"},
    {eFormatHex, 'x', "hex"},
    {eFormatHexUppercase, 'X', "uppercase hex"},
    {eFormatFloat, 'f', "float"},
    {eFormatOctal, 'o', "octal"},
    {eFormatOSType, 'O', "OSType"},
    {eFormatUnicode16, 'U', "unicode16"},
    {eFormatUnicode32, '\0', "unicode32"},
    {eFormatUnsigned, 'u', "unsigned decimal"},
    {eFormatPointer, 'p', "pointer"},
    {eFormatVectorOfChar, '\0', "char[]"},
    {eFormatVectorOfSInt8, '\0', "int8_t[]"},
    {eFormatVectorOfUInt8, '\0', "uint8_t[]"},
    {eFormatVectorOfSInt16, '\0', "int16_t[]"},
    {eFormatVectorOfUInt16, '\0', "uint16_t[]"},
    {eFormatVectorOfSInt32, '\0', "int32_t[]"},
    {eFormatVectorOfUInt32, '\0', "uint32_t[]"},
    {eFormatVectorOfSInt64, '\0', "int64_t[]"},
    {eFormatVectorOfUInt64, '\0', "uint64_t[]"},
    {eFormatVectorOfFloat16, '\0', "float16[]"},
    {eFormatVectorOfFloat32, '\0', "float32[]"},
    {eFormatVectorOfFloat64, '\0', "float64[]"},
    {eFormatVectorOfUInt128, '\0', "uint128_t[]"},
    {eFormatComplexInteger, 'I', "complex integer"},
    {eFormatCharArray, 'a', "character array"},
    {eFormatAddressInfo, 'A', "address"},
    {eFormatHexFloat, '\0', "hex float"},
    {eFormatInstruction, 'i', "instruction"},
    {eFormatVoid, 'v', "void"},
    {eFormatUnicode8, 'u', "unicode8"},
    {eFormatFloat128, '\0', "float128"},
};

static_assert((sizeof(g_format_infos) / sizeof(g_format_infos[0])) ==
                  kNumFormats,
              "All formats must have a corresponding info entry.");

static uint32_t g_num_format_infos = std::size(g_format_infos);

static bool GetFormatFromFormatChar(char format_char, Format &format) {
  for (uint32_t i = 0; i < g_num_format_infos; ++i) {
    if (g_format_infos[i].format_char == format_char) {
      format = g_format_infos[i].format;
      return true;
    }
  }
  format = eFormatInvalid;
  return false;
}

static bool GetFormatFromFormatName(llvm::StringRef format_name,
                                    Format &format) {
  uint32_t i;
  for (i = 0; i < g_num_format_infos; ++i) {
    if (format_name.equals_insensitive(g_format_infos[i].format_name)) {
      format = g_format_infos[i].format;
      return true;
    }
  }

  for (i = 0; i < g_num_format_infos; ++i) {
    if (llvm::StringRef(g_format_infos[i].format_name)
            .starts_with_insensitive(format_name)) {
      format = g_format_infos[i].format;
      return true;
    }
  }
  format = eFormatInvalid;
  return false;
}

} // namespace

bool lldb_private::GetFormatFromCString(const char *format_cstr,
                                         lldb::Format &format) {
  bool success = false;
  if (format_cstr && format_cstr[0]) {
    if (format_cstr[1] == '\0') {
      success = GetFormatFromFormatChar(format_cstr[0], format);
      if (success)
        return true;
    }

    success = GetFormatFromFormatName(format_cstr, format);
  }
  if (!success)
    format = eFormatInvalid;
  return success;
}

char lldb_private::GetFormatAsFormatChar(lldb::Format format) {
  for (uint32_t i = 0; i < g_num_format_infos; ++i) {
    if (g_format_infos[i].format == format)
      return g_format_infos[i].format_char;
  }
  return '\0';
}

const char *lldb_private::GetFormatAsCString(Format format) {
  if (format >= eFormatDefault && format < kNumFormats)
    return g_format_infos[format].format_name;
  return nullptr;
}

