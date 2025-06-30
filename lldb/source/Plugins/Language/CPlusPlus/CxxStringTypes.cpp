//===-- CxxStringTypes.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CxxStringTypes.h"

#include "llvm/Support/ConvertUTF.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Host/Time.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"

#include <algorithm>
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

using StringElementType = StringPrinter::StringElementType;

static constexpr std::pair<const char *, Format>
getElementTraits(StringElementType ElemType) {
  switch (ElemType) {
  case StringElementType::UTF8:
    return std::make_pair("u8", lldb::eFormatUnicode8);
  case StringElementType::UTF16:
    return std::make_pair("u", lldb::eFormatUnicode16);
  case StringElementType::UTF32:
    return std::make_pair("U", lldb::eFormatUnicode32);
  default:
    return std::make_pair(nullptr, lldb::eFormatInvalid);
  }
}

template <StringElementType ElemType>
static bool CharStringSummaryProvider(ValueObject &valobj, Stream &stream) {
  Address valobj_addr = GetArrayAddressOrPointerValue(valobj);
  if (!valobj_addr.IsValid())
    return false;

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(valobj_addr);
  options.SetTargetSP(valobj.GetTargetSP());
  options.SetStream(&stream);
  options.SetPrefixToken(getElementTraits(ElemType).first);

  if (!StringPrinter::ReadStringAndDumpToStream<ElemType>(options))
    stream.Printf("Summary Unavailable");

  return true;
}

template <StringElementType ElemType>
static bool CharSummaryProvider(ValueObject &valobj, Stream &stream) {
  DataExtractor data;
  Status error;
  valobj.GetData(data, error);

  if (error.Fail())
    return false;

  std::string value;
  StringPrinter::ReadBufferAndDumpToStreamOptions options(valobj);

  constexpr auto ElemTraits = getElementTraits(ElemType);
  valobj.GetValueAsCString(ElemTraits.second, value);

  if (!value.empty())
    stream.Printf("%s ", value.c_str());

  options.SetData(std::move(data));
  options.SetStream(&stream);
  options.SetPrefixToken(ElemTraits.first);
  options.SetQuote('\'');
  options.SetSourceSize(1);
  options.SetBinaryZeroIsTerminator(false);

  return StringPrinter::ReadBufferAndDumpToStream<ElemType>(options);
}

bool lldb_private::formatters::Char8StringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  return CharStringSummaryProvider<StringElementType::UTF8>(valobj, stream);
}

bool lldb_private::formatters::Char16StringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  return CharStringSummaryProvider<StringElementType::UTF16>(valobj, stream);
}

bool lldb_private::formatters::Char32StringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  return CharStringSummaryProvider<StringElementType::UTF32>(valobj, stream);
}

bool lldb_private::formatters::WCharStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  Address valobj_addr = GetArrayAddressOrPointerValue(valobj);
  if (!valobj_addr.IsValid())
    return false;

  // Get a wchar_t basic type from the current type system
  std::optional<uint64_t> size = GetWCharByteSize(valobj);
  if (!size)
    return false;
  const uint32_t wchar_size = *size;

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(valobj_addr);
  options.SetTargetSP(valobj.GetTargetSP());
  options.SetStream(&stream);
  options.SetPrefixToken("L");

  switch (wchar_size) {
  case 1:
    return StringPrinter::ReadStringAndDumpToStream<StringElementType::UTF8>(
        options);
  case 2:
    return StringPrinter::ReadStringAndDumpToStream<StringElementType::UTF16>(
        options);
  case 4:
    return StringPrinter::ReadStringAndDumpToStream<StringElementType::UTF32>(
        options);
  default:
    stream.Printf("size for wchar_t is not valid");
    return true;
  }
  return true;
}

bool lldb_private::formatters::Char8SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  return CharSummaryProvider<StringElementType::UTF8>(valobj, stream);
}

bool lldb_private::formatters::Char16SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  return CharSummaryProvider<StringElementType::UTF16>(valobj, stream);
}

bool lldb_private::formatters::Char32SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  return CharSummaryProvider<StringElementType::UTF32>(valobj, stream);
}

bool lldb_private::formatters::WCharSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  DataExtractor data;
  Status error;
  valobj.GetData(data, error);

  if (error.Fail())
    return false;

  // Get a wchar_t basic type from the current type system
  std::optional<uint64_t> size = GetWCharByteSize(valobj);
  if (!size)
    return false;
  const uint32_t wchar_size = *size;

  StringPrinter::ReadBufferAndDumpToStreamOptions options(valobj);
  options.SetData(std::move(data));
  options.SetStream(&stream);
  options.SetPrefixToken("L");
  options.SetQuote('\'');
  options.SetSourceSize(1);
  options.SetBinaryZeroIsTerminator(false);

  switch (wchar_size) {
  case 1:
    return StringPrinter::ReadBufferAndDumpToStream<StringElementType::UTF8>(
        options);
  case 2:
    return StringPrinter::ReadBufferAndDumpToStream<StringElementType::UTF16>(
        options);
  case 4:
    return StringPrinter::ReadBufferAndDumpToStream<StringElementType::UTF32>(
        options);
  default:
    stream.Printf("size for wchar_t is not valid");
    return true;
  }
  return true;
}

std::optional<uint64_t>
lldb_private::formatters::GetWCharByteSize(ValueObject &valobj) {
  return llvm::expectedToOptional(
      valobj.GetCompilerType()
          .GetBasicTypeFromAST(lldb::eBasicTypeWChar)
          .GetByteSize(nullptr));
}

template <StringPrinter::StringElementType element_type>
bool lldb_private::formatters::StringBufferSummaryProvider(
    Stream &stream, const TypeSummaryOptions &summary_options,
    lldb::ValueObjectSP location_sp, uint64_t size, std::string prefix_token) {

  if (size == 0) {
    stream.PutCString(prefix_token);
    stream.PutCString("\"\"");
    return true;
  }

  if (!location_sp)
    return false;

  StringPrinter::ReadBufferAndDumpToStreamOptions options(*location_sp);

  if (summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped) {
    const auto max_size =
        location_sp->GetTargetSP()->GetMaximumSizeOfStringSummary();
    if (size > max_size) {
      size = max_size;
      options.SetIsTruncated(true);
    }
  }

  {
    DataExtractor extractor;
    const size_t bytes_read = location_sp->GetPointeeData(extractor, 0, size);
    if (bytes_read < size)
      return false;

    options.SetData(std::move(extractor));
  }
  options.SetStream(&stream);
  if (prefix_token.empty())
    options.SetPrefixToken(nullptr);
  else
    options.SetPrefixToken(prefix_token);
  options.SetQuote('"');
  options.SetSourceSize(size);
  options.SetBinaryZeroIsTerminator(false);
  return StringPrinter::ReadBufferAndDumpToStream<element_type>(options);
}

// explicit instantiations for all string element types
template bool
lldb_private::formatters::StringBufferSummaryProvider<StringElementType::ASCII>(
    Stream &, const TypeSummaryOptions &, lldb::ValueObjectSP, uint64_t,
    std::string);
template bool
lldb_private::formatters::StringBufferSummaryProvider<StringElementType::UTF8>(
    Stream &, const TypeSummaryOptions &, lldb::ValueObjectSP, uint64_t,
    std::string);
template bool
lldb_private::formatters::StringBufferSummaryProvider<StringElementType::UTF16>(
    Stream &, const TypeSummaryOptions &, lldb::ValueObjectSP, uint64_t,
    std::string);
template bool
lldb_private::formatters::StringBufferSummaryProvider<StringElementType::UTF32>(
    Stream &, const TypeSummaryOptions &, lldb::ValueObjectSP, uint64_t,
    std::string);
