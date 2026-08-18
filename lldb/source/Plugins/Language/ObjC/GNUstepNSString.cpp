//===-- GNUstepNSString.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepFormatters.h"

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

/// The character encodings a string may hold. gnustep-base's 8-bit strings
/// use its "internal encoding", which defaults to ISO Latin-1
/// (Source/GSString.m), so one byte is one code point.
enum class Encoding { Latin1, UTF8, UTF16, UTF32 };

/// Where the characters live and how many there are.
struct StringContents {
  addr_t address = LLDB_INVALID_ADDRESS;
  /// Number of code units of the encoding, not bytes.
  uint64_t count = 0;
  Encoding encoding = Encoding::Latin1;
};

/// A ValueObject for the union `GSCharPtr _contents` (or a plain pointer)
/// yields the buffer address either way.
addr_t GetPointerValue(ValueObject &value) {
  if (value.GetCompilerType().IsPointerType())
    return value.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  // A union: any member is the same pointer.
  if (ValueObjectSP first_sp = value.GetChildAtIndex(0))
    return first_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  return LLDB_INVALID_ADDRESS;
}

/// NSConstantString with the gnustep-2.x string ABI: `uint32_t flags`
/// (low two bits: 0 ASCII, 1 UTF-8, 2 UTF-16, 3 UTF-32), `uint32_t nxcslen`
/// (characters), `uint32_t size` (bytes), `uint32_t hash`, `const char
/// *nxcsptr` (Headers/Foundation/NSString.h). The legacy ABI has only
/// `nxcsptr` and a byte count `nxcslen`.
std::optional<StringContents> ReadConstantString(ValueObject &valobj) {
  ValueObjectSP ptr_sp = GNUstepGetIvar(valobj, "nxcsptr");
  ValueObjectSP len_sp = GNUstepGetIvar(valobj, "nxcslen");
  if (!ptr_sp || !len_sp)
    return std::nullopt;
  StringContents contents;
  contents.address = ptr_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  if (contents.address == LLDB_INVALID_ADDRESS)
    return std::nullopt;

  ValueObjectSP flags_sp = GNUstepGetIvar(valobj, "flags");
  ValueObjectSP size_sp = GNUstepGetIvar(valobj, "size");
  if (!flags_sp || !size_sp) {
    // Legacy ABI: nxcslen is a byte count of UTF-8 data.
    contents.encoding = Encoding::UTF8;
    contents.count = len_sp->GetValueAsUnsigned(0);
    return contents;
  }
  const uint64_t bytes = size_sp->GetValueAsUnsigned(0);
  switch (flags_sp->GetValueAsUnsigned(0) & 3) {
  case 0:
  case 1:
    contents.encoding = Encoding::UTF8;
    contents.count = bytes;
    break;
  case 2:
    contents.encoding = Encoding::UTF16;
    contents.count = bytes / 2;
    break;
  default:
    contents.encoding = Encoding::UTF32;
    contents.count = bytes / 4;
    break;
  }
  return contents;
}

/// `_flags` is a bitfield struct, which libobjc2's encodings cannot describe,
/// so without debug info read `wide` out of the byte it occupies.
std::optional<bool> IsWide(ValueObject &valobj) {
  if (ValueObjectSP flags_sp = GNUstepGetIvar(valobj, "_flags"))
    if (ValueObjectSP wide_sp = flags_sp->GetChildMemberWithName("wide"))
      return wide_sp->GetValueAsUnsigned(0) != 0;

  std::optional<addr_t> flags_addr = GNUstepGetIvarAddress(valobj, "_flags");
  ProcessSP process_sp = valobj.GetProcessSP();
  if (!flags_addr || !process_sp)
    return std::nullopt;
  Status error;
  uint8_t byte = 0;
  if (process_sp->ReadMemory(*flags_addr, &byte, sizeof(byte), error) !=
          sizeof(byte) ||
      error.Fail())
    return std::nullopt;
  return GNUstepDecodeWideFlag(byte, process_sp->GetByteOrder());
}

/// GSString and everything derived from it, plus GSMutableString: the buffer
/// pointer `_contents`, the character count `_count`, and `_flags` whose bit
/// 0 (`wide`) selects 16-bit characters (Source/GSPrivate.h). The buffer is
/// never NUL-terminated.
std::optional<StringContents> ReadGSString(ValueObject &valobj) {
  ValueObjectSP contents_sp = GNUstepGetIvar(valobj, "_contents");
  ValueObjectSP count_sp = GNUstepGetIvar(valobj, "_count");
  if (!contents_sp || !count_sp)
    return std::nullopt;
  StringContents contents;
  contents.address = GetPointerValue(*contents_sp);
  if (contents.address == LLDB_INVALID_ADDRESS)
    return std::nullopt;
  contents.count = count_sp->GetValueAsUnsigned(0);
  std::optional<bool> wide = IsWide(valobj);
  if (!wide)
    return std::nullopt;
  contents.encoding = *wide ? Encoding::UTF16 : Encoding::Latin1;
  return contents;
}

bool DumpContents(ValueObject &valobj, Stream &stream,
                  const TypeSummaryOptions &summary_options,
                  const StringContents &contents) {
  static constexpr llvm::StringLiteral g_TypeHint("NSString");
  llvm::StringRef prefix, suffix;
  if (Language *language = Language::FindPlugin(summary_options.GetLanguage()))
    std::tie(prefix, suffix) = language->GetFormatterPrefixSuffix(g_TypeHint);

  if (contents.count == 0) {
    stream << prefix << "\"\"" << suffix;
    return true;
  }

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(Address(contents.address));
  options.SetTargetSP(valobj.GetTargetSP());
  options.SetStream(&stream);
  options.SetPrefixToken(prefix.str());
  options.SetSuffixToken(suffix.str());
  options.SetQuote('"');
  options.SetSourceSize(contents.count);
  options.SetHasSourceSize(true);
  options.SetZeroTermination(StringPrinter::ZeroTermination::Ignore);
  options.SetIgnoreMaxLength(summary_options.GetCapping() ==
                             TypeSummaryCapping::eTypeSummaryUncapped);

  switch (contents.encoding) {
  case Encoding::Latin1: {
    // Read exactly `count` bytes and transcode to UTF-8 ourselves: the ASCII
    // path of the printer reads a C string (dropping the last byte for a
    // terminator the buffer does not have) and cannot represent code points
    // above 0x7f.
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
      return false;
    const uint64_t max_size =
        valobj.GetTargetSP()->GetMaximumSizeOfStringSummary();
    uint64_t to_read = contents.count;
    bool truncated = false;
    if (!options.GetIgnoreMaxLength() && to_read > max_size) {
      to_read = max_size;
      truncated = true;
    }
    std::vector<uint8_t> latin1(to_read);
    Status error;
    if (to_read && process_sp->ReadMemory(contents.address, latin1.data(),
                                          to_read, error) != to_read)
      return false;
    std::string utf8;
    utf8.reserve(to_read * 2);
    for (uint8_t byte : latin1) {
      if (byte < 0x80) {
        utf8.push_back(static_cast<char>(byte));
      } else {
        utf8.push_back(static_cast<char>(0xC0 | (byte >> 6)));
        utf8.push_back(static_cast<char>(0x80 | (byte & 0x3F)));
      }
    }
    StringPrinter::ReadBufferAndDumpToStreamOptions dump_options(options);
    dump_options.SetData(DataExtractor(utf8.data(), utf8.size(),
                                       process_sp->GetByteOrder(),
                                       process_sp->GetAddressByteSize()));
    dump_options.SetSourceSize(utf8.size());
    dump_options.SetIsTruncated(truncated);
    return StringPrinter::ReadBufferAndDumpToStream<
        StringPrinter::StringElementType::UTF8>(dump_options);
  }
  case Encoding::UTF8:
    return StringPrinter::ReadStringAndDumpToStream<
        StringPrinter::StringElementType::UTF8>(options);
  case Encoding::UTF16:
    return StringPrinter::ReadStringAndDumpToStream<
        StringPrinter::StringElementType::UTF16>(options);
  case Encoding::UTF32:
    return StringPrinter::ReadStringAndDumpToStream<
        StringPrinter::StringElementType::UTF32>(options);
  }
  return false;
}

} // namespace

bool lldb_private::formatters::GNUstepNSStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  const uint64_t ptr = valobj.GetValueAsUnsigned(0);
  if (ptr == 0)
    return false;

  // Up to eight ASCII characters live in the pointer itself; clang emits
  // such literals directly (CGObjCGNU.cpp) and the runtime never allocates a
  // GSTinyString object.
  ProcessSP tiny_process_sp = valobj.GetProcessSP();
  const uint32_t pointer_size =
      tiny_process_sp ? tiny_process_sp->GetAddressByteSize() : 8;
  if (std::optional<std::string> tiny =
          GNUstepDecodeTinyString(ptr, pointer_size)) {
    static constexpr llvm::StringLiteral g_TypeHint("NSString");
    llvm::StringRef prefix, suffix;
    if (Language *language = Language::FindPlugin(options.GetLanguage()))
      std::tie(prefix, suffix) = language->GetFormatterPrefixSuffix(g_TypeHint);
    // Through the printer rather than straight to the stream: a tiny string
    // may hold any 7-bit character, including a quote, a backslash or a
    // control character, and those have to be escaped exactly as they are on
    // every other path here. The bytes are 7-bit, so they are already UTF-8.
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
      return false;
    StringPrinter::ReadBufferAndDumpToStreamOptions dump_options;
    dump_options.SetStream(&stream);
    dump_options.SetPrefixToken(prefix.str());
    dump_options.SetSuffixToken(suffix.str());
    dump_options.SetQuote('"');
    dump_options.SetSourceSize(tiny->size());
    dump_options.SetData(DataExtractor(tiny->data(), tiny->size(),
                                       process_sp->GetByteOrder(),
                                       process_sp->GetAddressByteSize()));
    return StringPrinter::ReadBufferAndDumpToStream<
        StringPrinter::StringElementType::UTF8>(dump_options);
  }
  if (ptr & GNUstepSmallObjectMask(pointer_size))
    return false;

  ProcessSP process_sp = valobj.GetProcessSP();
  ObjCLanguageRuntime *runtime =
      process_sp ? ObjCLanguageRuntime::Get(*process_sp) : nullptr;
  if (!runtime)
    return false;
  ObjCLanguageRuntime::ClassDescriptorSP descriptor =
      runtime->GetClassDescriptor(valobj);
  if (!descriptor || !descriptor->IsValid())
    return false;

  std::optional<StringContents> contents;
  if (descriptor->GetClassName() == "NSConstantString")
    contents = ReadConstantString(valobj);
  else
    contents = ReadGSString(valobj);
  if (!contents)
    return false;
  return DumpContents(valobj, stream, options, *contents);
}
