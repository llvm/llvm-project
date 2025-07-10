//===-- MsvcStl.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/FormatEntity.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

#include "Plugins/Language/CPlusPlus/CxxStringTypes.h"

#include "lldb/lldb-forward.h"
#include <optional>
#include <tuple>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

using StringElementType = StringPrinter::StringElementType;

template <StringElementType element_type>
static constexpr uint64_t StringElementByteSize() {
  switch (element_type) {
  case StringElementType::ASCII:
  case StringElementType::UTF8:
    return 1;
  case StringElementType::UTF16:
    return 2;
  case StringElementType::UTF32:
    return 3;
  }
  return 0;
}

static ValueObjectSP ExtractMsvcStlStringData(ValueObject &valobj) {
  return valobj.GetChildAtNamePath({"_Mypair", "_Myval2"});
}

/// Determine the size in bytes of \p valobj (a MSVC STL std::string object) and
/// extract its data payload. Return the size + payload pair.
static std::optional<std::pair<uint64_t, ValueObjectSP>>
ExtractMsvcStlStringInfo(ValueObject &valobj, uint64_t element_size) {
  ValueObjectSP valobj_pair_sp = ExtractMsvcStlStringData(valobj);
  if (!valobj_pair_sp || !valobj_pair_sp->GetError().Success())
    return {};

  ValueObjectSP size_sp = valobj_pair_sp->GetChildMemberWithName("_Mysize");
  ValueObjectSP capacity_sp = valobj_pair_sp->GetChildMemberWithName("_Myres");
  ValueObjectSP bx_sp = valobj_pair_sp->GetChildMemberWithName("_Bx");
  if (!size_sp || !capacity_sp || !bx_sp)
    return {};

  bool success = false;
  uint64_t size = size_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return {};
  uint64_t capacity = capacity_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return {};

  size_t bufSize = std::max<size_t>(16 / element_size, 1);
  bool isShortString = capacity < bufSize;

  if (isShortString) {
    ValueObjectSP buf_sp = bx_sp->GetChildMemberWithName("_Buf");
    if (buf_sp)
      return std::make_pair(size, buf_sp);
    return {};
  }
  ValueObjectSP ptr_sp = bx_sp->GetChildMemberWithName("_Ptr");
  if (ptr_sp)
    return std::make_pair(size, ptr_sp);
  return {};
}

template <StringPrinter::StringElementType element_type>
static bool
MsvcStlStringSummaryProviderImpl(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &summary_options,
                                 std::string prefix_token) {
  auto string_info =
      ExtractMsvcStlStringInfo(valobj, StringElementByteSize<element_type>());
  if (!string_info)
    return false;
  auto [size, location_sp] = *string_info;

  return StringBufferSummaryProvider<element_type>(
      stream, summary_options, location_sp, size, prefix_token);
}
template <StringPrinter::StringElementType element_type>
static bool formatStringImpl(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &summary_options,
                             std::string prefix_token) {
  StreamString scratch_stream;
  const bool success = MsvcStlStringSummaryProviderImpl<element_type>(
      valobj, scratch_stream, summary_options, prefix_token);
  if (success)
    stream << scratch_stream.GetData();
  else
    stream << "Summary Unavailable";
  return true;
}

bool lldb_private::formatters::IsMsvcStlStringType(ValueObject &valobj) {
  std::vector<uint32_t> indexes;
  return valobj.GetCompilerType().GetIndexOfChildMemberWithName("_Mypair", true,
                                                                indexes) > 0;
}

bool lldb_private::formatters::MsvcStlWStringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options) {
  return formatStringImpl<StringElementType::UTF16>(valobj, stream,
                                                    summary_options, "L");
}

template <>
bool lldb_private::formatters::MsvcStlStringSummaryProvider<
    StringElementType::ASCII>(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &summary_options) {
  return MsvcStlStringSummaryProviderImpl<StringElementType::ASCII>(
      valobj, stream, summary_options, "");
}
template <>
bool lldb_private::formatters::MsvcStlStringSummaryProvider<
    StringElementType::UTF8>(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &summary_options) {
  return MsvcStlStringSummaryProviderImpl<StringElementType::UTF8>(
      valobj, stream, summary_options, "u8");
}
template <>
bool lldb_private::formatters::MsvcStlStringSummaryProvider<
    StringElementType::UTF16>(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &summary_options) {
  return MsvcStlStringSummaryProviderImpl<StringElementType::UTF16>(
      valobj, stream, summary_options, "u");
}
template <>
bool lldb_private::formatters::MsvcStlStringSummaryProvider<
    StringElementType::UTF32>(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &summary_options) {
  return MsvcStlStringSummaryProviderImpl<StringElementType::UTF32>(
      valobj, stream, summary_options, "U");
}
