//===-- SwiftFormatters.cpp -------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftFormatters.h"
#include "Plugins/Language/Swift/SwiftStringIndex.h"
#include "Plugins/LanguageRuntime/Swift/ReflectionContextInterface.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timer.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "swift/ABI/Task.h"
#include "swift/AST/Types.h"
#include "swift/Concurrency/Actor.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/ManglingMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

// FIXME: we should not need this
#include "Plugins/Language/CPlusPlus/CxxStringTypes.h"
#include "Plugins/Language/ObjC/Cocoa.h"
#include "Plugins/Language/ObjC/NSString.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;
using namespace llvm;

bool lldb_private::formatters::swift::Character_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__str("_str");

  ValueObjectSP str_sp = valobj.GetChildMemberWithName(g__str, true);
  if (!str_sp)
    return false;

  return String_SummaryProvider(*str_sp, stream, options);
}

bool lldb_private::formatters::swift::UnicodeScalar_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_value("_value");
  ValueObjectSP value_sp(valobj.GetChildMemberWithName(g_value, true));
  if (!value_sp)
    return false;
  return Char32SummaryProvider(*value_sp.get(), stream, options);
}

bool lldb_private::formatters::swift::StringGuts_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return StringGuts_SummaryProvider(
      valobj, stream, options,
      StringPrinter::ReadStringAndDumpToStreamOptions());
}

bool lldb_private::formatters::swift::SwiftSharedString_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return SwiftSharedString_SummaryProvider_2(
      valobj, stream, options,
      StringPrinter::ReadStringAndDumpToStreamOptions());
}

struct StringSlice {
  uint64_t start, end;
};

template <typename AddrT>
static void applySlice(AddrT &address, uint64_t &length,
                       std::optional<StringSlice> slice) {
  if (!slice)
    return;

  // No slicing is performed when the slice starts beyond the string's bounds.
  if (slice->start > length)
    return;

  // The slicing logic does handle the corner case where slice->start == length.

  auto offset = slice->start;
  auto slice_length = slice->end - slice->start;

  // Adjust from the start.
  address += offset;
  length -= offset;

  // Reduce to the slice length, unless it's larger than the remaining length.
  length = std::min(slice_length, length);
}

static bool readStringFromAddress(
    uint64_t startAddress, uint64_t length, ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {
  if (length == 0) {
    stream.Printf("\"\"");
    return true;
  }

  read_options.SetLocation(startAddress);
  read_options.SetTargetSP(valobj.GetTargetSP());
  read_options.SetStream(&stream);
  read_options.SetSourceSize(length);
  read_options.SetHasSourceSize(true);
  read_options.SetNeedsZeroTermination(false);
  read_options.SetIgnoreMaxLength(summary_options.GetCapping() ==
                                  lldb::eTypeSummaryUncapped);
  read_options.SetBinaryZeroIsTerminator(false);
  read_options.SetEscapeStyle(StringPrinter::EscapeStyle::Swift);

  return StringPrinter::ReadStringAndDumpToStream<
      StringPrinter::StringElementType::UTF8>(read_options);
};

static bool makeStringGutsSummary(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options,
    std::optional<StringSlice> slice = std::nullopt) {
  static ConstString g__object("_object");
  static ConstString g__storage("_storage");
  static ConstString g__value("_value");

  auto error = [&](std::string message) {
    stream << "<cannot decode string: " << message << ">";
    return true;
  };

  ProcessSP process(valobj.GetProcessSP());
  if (!process)
    return error("no live process");

  auto ptrSize = process->GetAddressByteSize();

  auto object_sp = valobj.GetChildMemberWithName(g__object, true);
  if (!object_sp)
    return error("unexpected layout");

  // We retrieve String contents by first extracting the
  // platform-independent 128-bit raw value representation from
  // _StringObject, then interpreting that.
  Status status;
  uint64_t raw0;
  uint64_t raw1;

  if (ptrSize == 8) {
    // On 64-bit platforms, we simply need to get the raw integer
    // values of the two stored properties.
    static ConstString g__countAndFlagsBits("_countAndFlagsBits");

    auto countAndFlagsBits = object_sp->GetChildAtNamePath(
      {g__countAndFlagsBits, g__value});
    if (!countAndFlagsBits)
      return error("unexpected layout");
    raw0 = countAndFlagsBits->GetValueAsUnsigned(0);

    auto object = object_sp->GetChildMemberWithName(g__object, true);
    if (!object)
      return error("unexpected layout (object)");
    raw1 = object->GetValueAsUnsigned(0);
  } else if (ptrSize == 4) {
    // On 32-bit platforms, we emulate what `_StringObject.rawBits`
    // does. It involves inspecting the variant and rearranging bits
    // to match the 64-bit representation.
    static ConstString g__count("_count");
    static ConstString g__variant("_variant");
    static ConstString g__discriminator("_discriminator");
    static ConstString g__flags("_flags");
    static ConstString g_immortal("immortal");

    auto count_sp = object_sp->GetChildAtNamePath({g__count, g__value});
    if (!count_sp)
      return error("unexpected layout (count)");
    uint64_t count = count_sp->GetValueAsUnsigned(0);

    auto discriminator_sp =
        object_sp->GetChildAtNamePath({g__discriminator, g__value});
    if (!discriminator_sp)
      return error("unexpected layout (discriminator)");
    uint64_t discriminator = discriminator_sp->GetValueAsUnsigned(0) & 0xff;

    auto flags_sp = object_sp->GetChildAtNamePath({g__flags, g__value});
    if (!flags_sp)
      return error("unexpected layout (flags)");
    uint64_t flags = flags_sp->GetValueAsUnsigned(0) & 0xffff;

    auto variant_sp = object_sp->GetChildMemberWithName(g__variant, true);
    if (!variant_sp)
      return error("unexpected layout (variant)");

    llvm::StringRef variantCase = variant_sp->GetValueAsCString();

    ValueObjectSP payload_sp;
    if (variantCase == "immortal" || variantCase == "native" ||
        variantCase == "bridged") {
      payload_sp = variant_sp->GetSyntheticValue();
      if (!payload_sp)
        return error("unexpected layout (no variant)");
      payload_sp = payload_sp->GetChildAtIndex(0);
      if (!payload_sp)
        return error("unexpected layout (no variant payload)");
      payload_sp = payload_sp->GetSyntheticValue();
    } else {
      return error("unknown variant");
    }
    if (variantCase == "bridged") {
      if (!payload_sp)
        return error("unexpected layout (bridged)");
      payload_sp = payload_sp->GetChildAtIndex(0, true); // "instance"
    }
    if (!payload_sp)
      return error("no payload");

    uint64_t pointerBits = payload_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    if (pointerBits == LLDB_INVALID_ADDRESS)
      return error("invalid payload");

    if ((discriminator & 0xB0) == 0xA0) {
      raw0 = count | (pointerBits << 32);
      raw1 = flags | (discriminator << 56);
    } else {
      raw0 = count | (flags << 48);
      raw1 = pointerBits | (discriminator << 56);
    }
  } else {
    return error("unsupported pointer size");
  }

  // Copied from StringObject.swift
  //
  // TODO: Hyperlink to final set of documentation diagrams instead
  //
  /*
  On 64-bit platforms, the discriminator is the most significant 4 bits of the
  bridge object.

  ┌─────────────────────╥─────┬─────┬─────┬─────┐
  │ Form                ║ b63 │ b62 │ b61 │ b60 │
  ╞═════════════════════╬═════╪═════╪═════╪═════╡
  │ Immortal, Small     ║  1  │ASCII│  1  │  0  │
  ├─────────────────────╫─────┼─────┼─────┼─────┤
  │ Immortal, Large     ║  1  │  0  │  0  │  0  │
  ├─────────────────────╫─────┼─────┼─────┼─────┤
  │ Immortal, Bridged   ║  1  │  1  │  0  │  0  │
  ╞═════════════════════╬═════╪═════╪═════╪═════╡
  │ Native              ║  0  │  0  │  0  │  0  │
  ├─────────────────────╫─────┼─────┼─────┼─────┤
  │ Shared              ║  x  │  0  │  0  │  0  │
  ├─────────────────────╫─────┼─────┼─────┼─────┤
  │ Shared, Bridged     ║  0  │  1  │  0  │  0  │
  ╞═════════════════════╬═════╪═════╪═════╪═════╡
  │ Foreign             ║  x  │  0  │  0  │  1  │
  ├─────────────────────╫─────┼─────┼─────┼─────┤
  │ Foreign, Bridged    ║  0  │  1  │  0  │  1  │
  └─────────────────────╨─────┴─────┴─────┴─────┘

  b63: isImmortal: Should the Swift runtime skip ARC
    - Small strings are just values, always immortal
    - Large strings can sometimes be immortal, e.g. literals
  b62: (large) isBridged / (small) isASCII
    - For large strings, this means lazily-bridged NSString: perform ObjC ARC
    - Small strings repurpose this as a dedicated bit to remember ASCII-ness
  b61: isSmall: Dedicated bit to denote small strings
  b60: isForeign: aka isSlow, cannot provide access to contiguous UTF-8

 All non-small forms share the same structure for the other half of the bits
 (i.e. non-object bits) as a word containing code unit count and various
 performance flags. The top 16 bits are for performance flags, which are not
 semantically relevant but communicate that some operations can be done more
 efficiently on this particular string, and the lower 48 are the code unit
 count (aka endIndex).

┌─────────┬───────┬──────────────────┬─────────────────┬────────┬───────┐
│   b63   │  b62  │       b61        │       b60       │ b59:48 │ b47:0 │
├─────────┼───────┼──────────────────┼─────────────────┼────────┼───────┤
│ isASCII │ isNFC │ isNativelyStored │ isTailAllocated │  TBD   │ count │
└─────────┴───────┴──────────────────┴─────────────────┴────────┴───────┘

 isASCII: set when all code units are known to be ASCII, enabling:
   - Trivial Unicode scalars, they're just the code units
   - Trivial UTF-16 transcoding (just bit-extend)
   - Also, isASCII always implies isNFC
 isNFC: set when the contents are in normal form C
   - Enables trivial lexicographical comparisons: just memcmp
   - `isASCII` always implies `isNFC`, but not vice versa
 isNativelyStored: set for native stored strings
   - `largeAddressBits` holds an instance of `_StringStorage`.
   - I.e. the start of the code units is at the stored address + `nativeBias`
 isTailAllocated: start of the code units is at the stored address + `nativeBias`
   - `isNativelyStored` always implies `isTailAllocated`, but not vice versa
      (e.g. literals)
 TBD: Reserved for future usage
   - Setting a TBD bit to 1 must be semantically equivalent to 0
   - I.e. it can only be used to "cache" fast-path information in the future
 count: stores the number of code units, corresponds to `endIndex`.
  */

  uint8_t discriminator = raw1 >> 56;

  if ((discriminator & 0b1011'0000) == 0b1010'0000) { // 1x10xxxx: Small string
    uint64_t count = (raw1 >> 56) & 0b1111;
    uint64_t maxCount = (ptrSize == 8 ? 15 : 10);
    if (count > maxCount)
      return error("count > maxCount");

    uint64_t rawBuffer[2] = {raw0, raw1};
    auto *buffer = (uint8_t *)&rawBuffer;
    applySlice(buffer, count, slice);

    StringPrinter::ReadBufferAndDumpToStreamOptions options(read_options);
    options.SetData(lldb_private::DataExtractor(
        buffer, count, process->GetByteOrder(), ptrSize));
    options.SetStream(&stream);
    options.SetSourceSize(count);
    options.SetBinaryZeroIsTerminator(false);
    options.SetEscapeStyle(StringPrinter::EscapeStyle::Swift);
    return StringPrinter::ReadBufferAndDumpToStream<
        StringPrinter::StringElementType::UTF8>(options);
  }

  uint64_t count = raw0 & 0x0000FFFFFFFFFFFF;
  uint16_t flags = raw0 >> 48;
  lldb::addr_t objectAddress = (raw1 & 0x0FFFFFFFFFFFFFFF);
  // Catch a zero-initialized string.
  if (!objectAddress) {
    stream << "<uninitialized>";
    return true;
  }

  if ((flags & 0x1000) != 0) { // Tail-allocated / biased address
    // Tail-allocation is only for natively stored or literals.
    if ((discriminator & 0b0111'0000) != 0)
      return error("unexpected discriminator");
    uint64_t bias = (ptrSize == 8 ? 32 : 20);
    auto address = objectAddress + bias;
    applySlice(address, count, slice);
    return readStringFromAddress(
      address, count, valobj, stream, summary_options, read_options);
  }

  if ((discriminator & 0b1111'0000) == 0) { // Shared string
    // FIXME: Verify that there is a __SharedStringStorage instance at `address`.
    // Shared strings must not be tail-allocated or natively stored.
    if ((flags & 0x3000) != 0)
      return false;
    uint64_t startOffset = (ptrSize == 8 ? 24 : 12);
    auto address = objectAddress + startOffset;
    lldb::addr_t start = process->ReadPointerFromMemory(address, status);
    if (status.Fail())
      return error(status.AsCString());

    applySlice(address, count, slice);
    return readStringFromAddress(
      start, count, valobj, stream, summary_options, read_options);
  }

  // Native/shared strings should already have been handled.
  if ((discriminator & 0b0111'0000) == 0)
    return error("unexpected discriminator");

  if ((discriminator & 0b0110'0000) == 0b0100'0000) { // x10xxxxx: Bridged
    TypeSystemClangSP clang_ts_sp =
        ScratchTypeSystemClang::GetForTarget(process->GetTarget());
    if (!clang_ts_sp)
      return error("no Clang type system");

    CompilerType id_type = clang_ts_sp->GetBasicType(lldb::eBasicTypeObjCID);

    // We may have an NSString pointer inline, so try formatting it directly.
    lldb_private::DataExtractor DE(&objectAddress, ptrSize,
                                   process->GetByteOrder(), ptrSize);
    auto nsstring = ValueObject::CreateValueObjectFromData(
        "nsstring", DE, valobj.GetExecutionContextRef(), id_type);
    if (!nsstring || nsstring->GetError().Fail())
      return error("could not create NSString value object");

    return NSStringSummaryProvider(*nsstring.get(), stream, summary_options);
  }

  if ((discriminator & 0b1111'1000) == 0b0001'1000) { // 0001xxxx: Foreign
    // Not currently generated: Foreign non-bridged strings are not currently
    // used in Swift.
    return error("unexpected discriminator");
  }

  // Invalid discriminator.
  return error("invalid discriminator");
}

bool lldb_private::formatters::swift::StringGuts_SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {
  return makeStringGutsSummary(valobj, stream, summary_options, read_options);
}

bool lldb_private::formatters::swift::String_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return String_SummaryProvider(
      valobj, stream, options,
      StringPrinter::ReadStringAndDumpToStreamOptions());
}

bool lldb_private::formatters::swift::String_SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {
  static ConstString g_guts("_guts");
  ValueObjectSP guts_sp = valobj.GetChildMemberWithName(g_guts, true);
  if (guts_sp)
    return StringGuts_SummaryProvider(*guts_sp, stream, summary_options,
                                      read_options);
  return false;
}

bool lldb_private::formatters::swift::Substring_SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options) {
  static ConstString g__slice("_slice");
  static ConstString g__base("_base");
  static ConstString g__startIndex("_startIndex");
  static ConstString g__endIndex("_endIndex");
  static ConstString g__rawBits("_rawBits");
  auto slice_sp = valobj.GetChildMemberWithName(g__slice, true);
  if (!slice_sp)
    return false;
  auto base_sp = slice_sp->GetChildMemberWithName(g__base, true);
  if (!base_sp)
    return false;

  auto get_index =
      [&slice_sp](ConstString index_name) -> std::optional<StringIndex> {
    auto raw_bits_sp = slice_sp->GetChildAtNamePath({index_name, g__rawBits});
    if (!raw_bits_sp)
      return std::nullopt;
    bool success = false;
    StringIndex index =
        raw_bits_sp->GetSyntheticValue()->GetValueAsUnsigned(0, &success);
    if (!success)
      return std::nullopt;
    return index;
  };

  std::optional<StringIndex> start_index = get_index(g__startIndex);
  std::optional<StringIndex> end_index = get_index(g__endIndex);
  if (!start_index || !end_index)
    return false;

  if (!start_index->matchesEncoding(*end_index))
    return false;

  static ConstString g_guts("_guts");
  auto guts_sp = base_sp->GetChildMemberWithName(g_guts, true);
  if (!guts_sp)
    return false;

  StringPrinter::ReadStringAndDumpToStreamOptions read_options;
  StringSlice slice{start_index->encodedOffset(), end_index->encodedOffset()};
  return makeStringGutsSummary(*guts_sp, stream, summary_options, read_options,
                               slice);
}

bool lldb_private::formatters::swift::StringIndex_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__rawBits("_rawBits");
  auto raw_bits_sp = valobj.GetChildMemberWithName(g__rawBits, true);
  if (!raw_bits_sp)
    return false;

  bool success = false;
  StringIndex index =
      raw_bits_sp->GetSyntheticValue()->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;

  stream.Printf("%" PRIu64 "[%s]", index.encodedOffset(), index.encodingName());
  if (index.transcodedOffset() != 0)
    stream.Printf("+%u", index.transcodedOffset());

  return true;
}

bool lldb_private::formatters::swift::StaticString_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return StaticString_SummaryProvider(
      valobj, stream, options,
      StringPrinter::ReadStringAndDumpToStreamOptions());
}

bool lldb_private::formatters::swift::StaticString_SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {
  static ConstString g__startPtrOrData("_startPtrOrData");
  static ConstString g__byteSize("_utf8CodeUnitCount");
  static ConstString g__flags("_flags");

  ValueObjectSP flags_sp(valobj.GetChildMemberWithName(g__flags, true));
  if (!flags_sp)
    return false;

  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;

  // 0 == pointer representation
  InferiorSizedWord flags(flags_sp->GetValueAsUnsigned(0), *process_sp);
  if (0 != (flags & 0x1).GetValue())
    return false;

  ValueObjectSP startptr_sp(
      valobj.GetChildMemberWithName(g__startPtrOrData, true));
  ValueObjectSP bytesize_sp(valobj.GetChildMemberWithName(g__byteSize, true));
  if (!startptr_sp || !bytesize_sp)
    return false;

  lldb::addr_t start_ptr =
      startptr_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  uint64_t size = bytesize_sp->GetValueAsUnsigned(0);

  if (start_ptr == LLDB_INVALID_ADDRESS || start_ptr == 0)
    return false;

  if (size == 0) {
    stream.Printf("\"\"");
    return true;
  }

  read_options.SetTargetSP(valobj.GetTargetSP());
  read_options.SetLocation(start_ptr);
  read_options.SetSourceSize(size);
  read_options.SetHasSourceSize(true);
  read_options.SetBinaryZeroIsTerminator(false);
  read_options.SetNeedsZeroTermination(false);
  read_options.SetStream(&stream);
  read_options.SetIgnoreMaxLength(summary_options.GetCapping() ==
                                  lldb::eTypeSummaryUncapped);
  read_options.SetEscapeStyle(StringPrinter::EscapeStyle::Swift);

  return StringPrinter::ReadStringAndDumpToStream<
      StringPrinter::StringElementType::UTF8>(read_options);
}

bool lldb_private::formatters::swift::SwiftSharedString_SummaryProvider_2(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {
  ProcessSP process(valobj.GetProcessSP());
  if (!process)
    return false;

  Status error;
  auto ptr_size = process->GetAddressByteSize();

  lldb::addr_t raw1 = valobj.GetPointerValue().address;
  lldb::addr_t address = (raw1 & 0x00FFFFFFFFFFFFFF);
  uint64_t startOffset = (ptr_size == 8 ? 24 : 12);

  lldb::addr_t start =
      process->ReadPointerFromMemory(address + startOffset, error);
  if (error.Fail())
    return false;
  lldb::addr_t raw0 =
      process->ReadPointerFromMemory(address + startOffset + ptr_size, error);
  if (error.Fail())
    return false;

  uint64_t count = raw0 & 0x0000FFFFFFFFFFFF;

  return readStringFromAddress(start, count, valobj, stream, summary_options,
                               read_options);
}

bool lldb_private::formatters::swift::SwiftStringStorage_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ProcessSP process(valobj.GetProcessSP());
  if (!process)
    return false;
  auto ptrSize = process->GetAddressByteSize();
  uint64_t bias = (ptrSize == 8 ? 32 : 20);
  uint64_t raw0_offset = (ptrSize == 8 ? 24 : 12);
  lldb::addr_t raw1 = valobj.GetPointerValue().address;
  lldb::addr_t address = (raw1 & 0x00FFFFFFFFFFFFFF) + bias;

  Status error;
  lldb::addr_t raw0 = process->ReadPointerFromMemory(raw1 + raw0_offset, error);
  if (error.Fail())
    return false;
  uint64_t count = raw0 & 0x0000FFFFFFFFFFFF;
  return readStringFromAddress(
      address, count, valobj, stream, options,
      StringPrinter::ReadStringAndDumpToStreamOptions());
}

bool lldb_private::formatters::swift::Bool_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_value("_value");
  ValueObjectSP value_child(
      valobj.GetNonSyntheticValue()->GetChildMemberWithName(g_value, true));
  if (!value_child)
    return false;

  // Swift Bools are stored in a byte, but only the LSB of the byte is
  // significant.  The swift::irgen::FixedTypeInfo structure represents
  // this information by providing a mask of the "extra bits" for the type.
  // But at present CompilerType has no way to represent that information.
  // So for now we hard code it.
  uint64_t value = value_child->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  const uint64_t mask = 1 << 0;
  value &= mask;

  switch (value) {
  case 0:
    stream.Printf("false");
    return true;
  case 1:
    stream.Printf("true");
    return true;
  case LLDB_INVALID_ADDRESS:
    return false;
  default:
    stream.Printf("<invalid> (0x%" PRIx8 ")", (uint8_t)value);
    return true;
  }
}

bool lldb_private::formatters::swift::DarwinBoolean_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__value("_value");
  ValueObjectSP value_child(valobj.GetChildMemberWithName(g__value, true));
  if (!value_child)
    return false;
  auto value = value_child->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  switch (value) {
  case 0:
    stream.Printf("false");
    return true;
  default:
    stream.Printf("true");
    return true;
  }
}

static bool RangeFamily_SummaryProvider(ValueObject &valobj, Stream &stream,
                                        const TypeSummaryOptions &options,
                                        bool isHalfOpen) {
  static ConstString g_lowerBound("lowerBound");
  static ConstString g_upperBound("upperBound");

  ValueObjectSP lowerBound_sp(
      valobj.GetChildMemberWithName(g_lowerBound, true));
  ValueObjectSP upperBound_sp(
      valobj.GetChildMemberWithName(g_upperBound, true));

  if (!lowerBound_sp || !upperBound_sp)
    return false;

  lowerBound_sp = lowerBound_sp->GetQualifiedRepresentationIfAvailable(
      lldb::eDynamicDontRunTarget, true);
  upperBound_sp = upperBound_sp->GetQualifiedRepresentationIfAvailable(
      lldb::eDynamicDontRunTarget, true);

  auto start_summary = lowerBound_sp->GetValueAsCString();
  auto end_summary = upperBound_sp->GetValueAsCString();

  // the Range should not have a summary unless both start and end indices have
  // one - or it will look awkward
  if (!start_summary || !start_summary[0] || !end_summary || !end_summary[0])
    return false;

  stream.Printf("%s%s%s", start_summary, isHalfOpen ? "..<" : "...",
                end_summary);

  return true;
}

bool lldb_private::formatters::swift::Range_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return RangeFamily_SummaryProvider(valobj, stream, options, true);
}

bool lldb_private::formatters::swift::CountableRange_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return RangeFamily_SummaryProvider(valobj, stream, options, true);
}

bool lldb_private::formatters::swift::ClosedRange_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return RangeFamily_SummaryProvider(valobj, stream, options, false);
}

bool lldb_private::formatters::swift::CountableClosedRange_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return RangeFamily_SummaryProvider(valobj, stream, options, false);
}

bool lldb_private::formatters::swift::BuiltinObjC_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  stream.Printf("0x%" PRIx64 " ", valobj.GetValueAsUnsigned(0));
  llvm::Expected<std::string> desc = valobj.GetObjectDescription();
  if (desc)
    stream << toString(desc.takeError());
  else
    stream << *desc;
  return true;
}

namespace lldb_private {
namespace formatters {
namespace swift {

/// The size of Swift Tasks. Fragments are tail allocated.
static constexpr size_t AsyncTaskSize = sizeof(::swift::AsyncTask);
/// The offset of ChildFragment, which is the first fragment of an AsyncTask.
static constexpr offset_t ChildFragmentOffset = AsyncTaskSize;

class EnumSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  EnumSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  llvm::Expected<uint32_t> CalculateNumChildren() override;
  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;
  lldb::ChildCacheState Update() override;
  bool MightHaveChildren() override;
  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override;

private:
  ValueObjectSP m_projected;
  lldb::DynamicValueType m_dynamic = eNoDynamicValues;
  bool m_indirect = false;
};

static std::string mangledTypenameForTasksTuple(size_t count) {
  /*
  Global > TypeMangling > Type > Tuple
    TupleElement > Type > Structure
      Module, text="Swift"
      Identifier, text="UnsafeCurrentTask"
  */
  using namespace ::swift::Demangle;
  using Kind = Node::Kind;
  NodeFactory factory;
  auto [root, tuple] = swift_demangle::MakeNodeChain(
      {Kind::TypeMangling, Kind::Type, Kind::Tuple}, factory);

  // Make a TupleElement subtree N times, where N is the number of subtasks.
  for (size_t i = 0; i < count; ++i) {
    auto *structure = swift_demangle::MakeNodeChain(
        tuple, {Kind::TupleElement, Kind::Type, Kind::Structure}, factory);
    if (structure) {
      structure->addChild(
          factory.createNode(Kind::Module, ::swift::STDLIB_NAME), factory);
      structure->addChild(
          factory.createNode(Kind::Identifier, "UnsafeCurrentTask"), factory);
    }
  }

  return mangleNode(root).result();
}

/// Synthetic provider for `Swift.Task`.
///
/// As seen by lldb, a `Task` instance is an opaque pointer, with neither type
/// metadata nor an AST to describe it. To implement this synthetic provider, a
/// `Task`'s state is retrieved from a `ReflectionContext`, and that data is
/// used to manually construct `ValueObject` children.
class TaskSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  TaskSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
    auto target_sp = m_backend.GetTargetSP();
    auto ts_or_err =
        target_sp->GetScratchTypeSystemForLanguage(eLanguageTypeSwift);
    if (auto err = ts_or_err.takeError()) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
                     std::move(err),
                     "could not get Swift type system for Task synthetic "
                     "provider: {0}");
      return;
    }
    m_ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts_or_err->get());
  }

  constexpr static StringLiteral TaskChildren[] = {
      // clang-format off
      "address",
      "id",
      "enqueuePriority",
      "parent",
      "children",

      // Children below this point are hidden.
      "isChildTask",
      "isFuture",
      "isGroupChildTask",
      "isAsyncLetTask",
      "isCancelled",
      "isStatusRecordLocked",
      "isEscalated",
      "isEnqueued",
      "isComplete",
      "isSuspended",
      "isRunning",
      // clang-format on
  };

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    // Show only the first five children
    // address/id/enqueuePriority/parent/children.
    return 5;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    auto target_sp = m_backend.GetTargetSP();

    // TypeMangling for "Swift.Bool"
    CompilerType bool_type =
        m_ts->GetTypeFromMangledTypename(ConstString("$sSbD"));

#define RETURN_CHILD(FIELD, NAME, TYPE)                                        \
  if (!FIELD) {                                                                \
    auto value = m_task_info.NAME;                                             \
    DataExtractor data{reinterpret_cast<const void *>(&value), sizeof(value),  \
                       endian::InlHostByteOrder(), sizeof(void *)};            \
    FIELD = ValueObject::CreateValueObjectFromData(                            \
        #NAME, data, m_backend.GetExecutionContextRef(), TYPE);                \
  }                                                                            \
  return FIELD;

    switch (idx) {
    case 0:
      if (!m_address_sp) {
        // TypeMangling for "Swift.UnsafeRawPointer"
        CompilerType raw_pointer_type =
            m_ts->GetTypeFromMangledTypename(ConstString("$sSVD"));

        addr_t value = m_task_ptr;
        if (auto process_sp = m_backend.GetProcessSP())
          value = process_sp->FixDataAddress(value);
        DataExtractor data{reinterpret_cast<const void *>(&value),
                           sizeof(value), endian::InlHostByteOrder(),
                           sizeof(void *)};
        m_address_sp = ValueObject::CreateValueObjectFromData(
            "address", data, m_backend.GetExecutionContextRef(),
            raw_pointer_type);
      }
      return m_address_sp;
    case 1: {
      // TypeMangling for "Swift.UInt64"
      CompilerType uint64_type =
          m_ts->GetTypeFromMangledTypename(ConstString("$ss6UInt64VD"));
      RETURN_CHILD(m_id_sp, id, uint64_type);
    }
    case 2: {
      // TypeMangling for "Swift.TaskPriority"
      CompilerType priority_type =
          m_ts->GetTypeFromMangledTypename(ConstString("$sScPD"));
      RETURN_CHILD(m_enqueue_priority_sp, enqueuePriority, priority_type);
    }
    case 3: {
      if (!m_parent_task_sp) {
        auto process_sp = m_backend.GetProcessSP();
        if (!process_sp)
          return {};

        // TypeMangling for "Swift.Optional<Swift.UnsafeRawPointer>"
        CompilerType raw_pointer_type =
            m_ts->GetTypeFromMangledTypename(ConstString("$sSVSgD"));

        addr_t parent_addr = 0;
        if (m_task_info.isChildTask) {
          // Read ChildFragment::Parent, the first field of the ChildFragment.
          Status status;
          parent_addr = process_sp->ReadPointerFromMemory(
              m_task_ptr + ChildFragmentOffset, status);
          if (status.Fail() || parent_addr == LLDB_INVALID_ADDRESS)
            parent_addr = 0;
        }

        addr_t value = process_sp->FixDataAddress(parent_addr);
        DataExtractor data{reinterpret_cast<const void *>(&value),
                           sizeof(value), endian::InlHostByteOrder(),
                           sizeof(void *)};
        m_parent_task_sp = ValueObject::CreateValueObjectFromData(
            "parent", data, m_backend.GetExecutionContextRef(),
            raw_pointer_type);
      }
      return m_parent_task_sp;
    }
    case 4: {
      if (!m_child_tasks_sp) {
        using task_type = decltype(m_task_info.childTasks)::value_type;
        std::vector<task_type> tasks = m_task_info.childTasks;

        // Remove any bogus child tasks.
        // Very rarely, the child tasks include a bogus task which has an
        // invalid task id of 0.
        if (auto reflection_ctx = GetReflectionContext())
          llvm::erase_if(tasks, [&](auto task_ptr) {
            if (auto task_info =
                    expectedToOptional(reflection_ctx->asyncTaskInfo(task_ptr)))
              return task_info->id == 0;
            // Don't filter children with errors here. Let these tasks reach the
            // formatter's existing error handling.
            return false;
          });

        std::string mangled_typename =
            mangledTypenameForTasksTuple(tasks.size());
        CompilerType tasks_tuple_type =
            m_ts->GetTypeFromMangledTypename(ConstString(mangled_typename));
        DataExtractor data{tasks.data(), tasks.size() * sizeof(task_type),
                           endian::InlHostByteOrder(), sizeof(void *)};
        m_child_tasks_sp = ValueObject::CreateValueObjectFromData(
            "children", data, m_backend.GetExecutionContextRef(),
            tasks_tuple_type);
      }
      return m_child_tasks_sp;
    }
    case 5:
      RETURN_CHILD(m_is_child_task_sp, isChildTask, bool_type);
    case 6:
      RETURN_CHILD(m_is_future_sp, isFuture, bool_type);
    case 7:
      RETURN_CHILD(m_is_group_child_task_sp, isGroupChildTask, bool_type);
    case 8:
      RETURN_CHILD(m_is_async_let_task_sp, isAsyncLetTask, bool_type);
    case 9:
      RETURN_CHILD(m_is_cancelled_sp, isCancelled, bool_type);
    case 10:
      RETURN_CHILD(m_is_status_record_locked_sp, isStatusRecordLocked,
                   bool_type);
    case 11:
      RETURN_CHILD(m_is_escalated_sp, isEscalated, bool_type);
    case 12:
      RETURN_CHILD(m_is_enqueued_sp, isEnqueued, bool_type);
    case 13:
      RETURN_CHILD(m_is_complete_sp, isComplete, bool_type);
    case 14:
      RETURN_CHILD(m_is_suspended_sp, isSuspended, bool_type);
    case 15: {
      if (m_task_info.hasIsRunning)
        RETURN_CHILD(m_is_running_sp, isRunning, bool_type);
      return {};
    }
    default:
      return {};
    }

#undef RETURN_CHILD
  }

  lldb::ChildCacheState Update() override {
    if (auto reflection_ctx = GetReflectionContext()) {
      ValueObjectSP task_obj_sp = m_backend.GetChildMemberWithName("_task");
      if (!task_obj_sp)
        return ChildCacheState::eRefetch;
      m_task_ptr = task_obj_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
      if (m_task_ptr != LLDB_INVALID_ADDRESS) {
        llvm::Expected<ReflectionContextInterface::AsyncTaskInfo> task_info =
            reflection_ctx->asyncTaskInfo(m_task_ptr);
        if (auto err = task_info.takeError()) {
          LLDB_LOG_ERROR(
              GetLog(LLDBLog::DataFormatters | LLDBLog::Types), std::move(err),
              "could not get info for async task {0:x}: {1}", m_task_ptr);
        } else {
          m_task_info = *task_info;
          for (auto child :
               {m_address_sp, m_id_sp, m_kind_sp, m_enqueue_priority_sp,
                m_is_child_task_sp, m_is_future_sp, m_is_group_child_task_sp,
                m_is_async_let_task_sp, m_is_cancelled_sp,
                m_is_status_record_locked_sp, m_is_escalated_sp,
                m_is_enqueued_sp, m_is_complete_sp, m_is_suspended_sp,
                m_parent_task_sp, m_child_tasks_sp, m_is_running_sp})
            child.reset();
        }
      }
    }
    return ChildCacheState::eRefetch;
  }

  bool MightHaveChildren() override { return true; }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    ArrayRef children = TaskChildren;
    const auto *it = llvm::find(children, name);
    if (it == children.end())
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    return std::distance(children.begin(), it);
  }

private:
  ThreadSafeReflectionContext GetReflectionContext() {
    if (auto *runtime = SwiftLanguageRuntime::Get(m_backend.GetProcessSP()))
      return runtime->GetReflectionContext();
    return {};
  }

  TypeSystemSwiftTypeRef *m_ts = nullptr;
  addr_t m_task_ptr = LLDB_INVALID_ADDRESS;
  ReflectionContextInterface::AsyncTaskInfo m_task_info;
  ValueObjectSP m_address_sp;
  ValueObjectSP m_id_sp;
  ValueObjectSP m_kind_sp;
  ValueObjectSP m_enqueue_priority_sp;
  ValueObjectSP m_is_child_task_sp;
  ValueObjectSP m_is_future_sp;
  ValueObjectSP m_is_group_child_task_sp;
  ValueObjectSP m_is_async_let_task_sp;
  ValueObjectSP m_is_cancelled_sp;
  ValueObjectSP m_is_status_record_locked_sp;
  ValueObjectSP m_is_escalated_sp;
  ValueObjectSP m_is_enqueued_sp;
  ValueObjectSP m_is_complete_sp;
  ValueObjectSP m_is_suspended_sp;
  ValueObjectSP m_parent_task_sp;
  ValueObjectSP m_child_tasks_sp;
  ValueObjectSP m_is_running_sp;
};

class UnsafeContinuationSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  UnsafeContinuationSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
    if (auto target_sp = m_backend.GetTargetSP()) {
      if (auto ts_or_err =
              target_sp->GetScratchTypeSystemForLanguage(eLanguageTypeSwift)) {
        if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
                ts_or_err->get()))
          // TypeMangling for "Swift.UnsafeCurrentTask"
          m_task_type = ts->GetTypeFromMangledTypename(ConstString("$sSctD"));
      } else {
        LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
                       ts_or_err.takeError(),
                       "could not get Swift type system for UnsafeContinuation "
                       "synthetic provider: {0}");
      }
    }
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!m_task_sp)
      return m_backend.GetNumChildren();

    return 1;
  }

  bool MightHaveChildren() override { return true; }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_task_sp)
      return m_backend.GetChildAtIndex(idx);

    if (idx == 0)
      return m_task_sp;

    return {};
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_task_sp)
      return m_backend.GetIndexOfChildWithName(name);

    if (name == "task")
      return 0;

    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }

  lldb::ChildCacheState Update() override {
    if (auto context_sp = m_backend.GetChildMemberWithName("context"))
      if (addr_t task_addr = context_sp->GetValueAsUnsigned(0)) {
        m_task_sp = ValueObject::CreateValueObjectFromAddress(
            "task", task_addr, m_backend.GetExecutionContextRef(), m_task_type,
            false);
        if (auto synthetic_sp = m_task_sp->GetSyntheticValue())
          m_task_sp = synthetic_sp;
      }
    return ChildCacheState::eRefetch;
  }

private:
  CompilerType m_task_type;
  ValueObjectSP m_task_sp;
};

class CheckedContinuationSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  CheckedContinuationSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
    bool is_64bit = false;
    if (auto target_sp = m_backend.GetTargetSP())
      is_64bit = target_sp->GetArchitecture().GetTriple().isArch64Bit();

    std::optional<uint32_t> concurrency_version;
    if (auto process_sp = m_backend.GetProcessSP())
      concurrency_version =
          SwiftLanguageRuntime::FindConcurrencyDebugVersion(*process_sp);

    bool is_supported_target = is_64bit && concurrency_version.value_or(0) == 1;
    if (!is_supported_target)
      return;

    if (auto target_sp = m_backend.GetTargetSP()) {
      if (auto ts_or_err =
              target_sp->GetScratchTypeSystemForLanguage(eLanguageTypeSwift)) {
        if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
                ts_or_err->get()))
          // TypeMangling for "Swift.UnsafeCurrentTask"
          m_task_type = ts->GetTypeFromMangledTypename(ConstString("$sSctD"));
      } else {
        LLDB_LOG_ERROR(
            GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
            ts_or_err.takeError(),
            "could not get Swift type system for CheckedContinuation "
            "synthetic provider: {0}");
      }
    }
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!m_task_sp)
      return m_backend.GetNumChildren();

    return 1;
  }

  bool MightHaveChildren() override { return true; }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_task_sp)
      return m_backend.GetChildAtIndex(idx);

    if (idx == 0)
      return m_task_sp;

    return {};
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_task_sp)
      return m_backend.GetIndexOfChildWithName(name);

    if (name == "task")
      return 0;

    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }

  lldb::ChildCacheState Update() override {
    if (!m_task_type)
      return ChildCacheState::eReuse;

    size_t canary_task_offset = 0x10;
    Status status;
    if (auto canary_sp = m_backend.GetChildMemberWithName("canary"))
      if (addr_t canary_addr = canary_sp->GetValueAsUnsigned(0))
        if (addr_t task_addr = m_backend.GetProcessSP()->ReadPointerFromMemory(
                canary_addr + canary_task_offset, status))
          m_task_sp = ValueObject::CreateValueObjectFromAddress(
              "task", task_addr, m_backend.GetExecutionContextRef(),
              m_task_type, false);

    if (m_task_sp)
      if (auto synthetic_sp = m_task_sp->GetSyntheticValue())
        m_task_sp = synthetic_sp;
    return ChildCacheState::eRefetch;
  }

private:
  CompilerType m_task_type;
  ValueObjectSP m_task_sp;
};

class TaskGroupSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  TaskGroupSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
    bool is_64bit = false;
    if (auto target_sp = m_backend.GetTargetSP())
      is_64bit = target_sp->GetArchitecture().GetTriple().isArch64Bit();

    std::optional<uint32_t> concurrency_version;
    if (auto process_sp = m_backend.GetProcessSP())
      concurrency_version =
          SwiftLanguageRuntime::FindConcurrencyDebugVersion(*process_sp);

    m_is_supported_target = is_64bit && concurrency_version.value_or(0) == 1;
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!m_is_supported_target)
      return m_backend.GetNumChildren();

    return m_task_addrs.size();
  }

  bool MightHaveChildren() override { return true; }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_is_supported_target)
      return m_backend.GetChildAtIndex(idx);

    if (!m_task_type || idx >= m_task_addrs.size())
      return {};

    if (auto valobj_sp = m_children[idx])
      return valobj_sp;

    addr_t task_addr = m_task_addrs[idx];
    auto child_name = ("[" + Twine(idx) + "]").str();
    auto task_sp = ValueObject::CreateValueObjectFromAddress(
        child_name, task_addr, m_backend.GetExecutionContextRef(), m_task_type,
        false);
    if (auto synthetic_sp = task_sp->GetSyntheticValue())
      task_sp = synthetic_sp;

    m_children[idx] = task_sp;
    return task_sp;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (!m_is_supported_target)
      return m_backend.GetIndexOfChildWithName(name);

    StringRef buf = name.GetStringRef();
    size_t idx = UINT32_MAX;
    if (buf.consume_front("[") && !buf.consumeInteger(10, idx) && buf == "]")
      return idx;
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }

  lldb::ChildCacheState Update() override {
    if (!m_is_supported_target)
      return ChildCacheState::eReuse;

    m_task_addrs.clear();
    m_children.clear();

    if (!m_task_type)
      if (auto target_sp = m_backend.GetTargetSP()) {
        if (auto ts_or_err = target_sp->GetScratchTypeSystemForLanguage(
                eLanguageTypeSwift)) {
          if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
                  ts_or_err->get()))
            // TypeMangling for "Swift.UnsafeCurrentTask"
            m_task_type = ts->GetTypeFromMangledTypename(ConstString("$sSctD"));
        } else {
          LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
                         ts_or_err.takeError(),
                         "could not get Swift type system for Task synthetic "
                         "provider: {0}");
          return ChildCacheState::eReuse;
        }
      }

    if (!m_task_type)
      return ChildCacheState::eReuse;

    // Get the (opaque) pointer to the `TaskGroupBase`.
    addr_t task_group_ptr = LLDB_INVALID_ADDRESS;
    if (auto opaque_group_ptr_sp = m_backend.GetChildMemberWithName("_group"))
      task_group_ptr =
          opaque_group_ptr_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    TaskGroupBase task_group{m_backend.GetProcessSP(), task_group_ptr};

    // Get the TaskGroup's child tasks by getting all tasks in the range
    // [FirstChild, LastChild].
    //
    // Child tasks are connected together using ChildFragment::NextChild.
    Status status;
    auto current_task = task_group.getFirstChild(status);
    auto last_task = task_group.getLastChild(status);
    while (current_task) {
      m_task_addrs.push_back(current_task.addr);
      if (current_task == last_task)
        break;
      current_task = current_task.getNextChild(status);
    }

    // Populate the child cache with null values.
    m_children.resize(m_task_addrs.size());

    if (status.Fail()) {
      LLDB_LOG(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
               "could not read TaskGroup's child task pointers: {0}",
               status.AsCString());
      return ChildCacheState::eReuse;
    }

    return ChildCacheState::eRefetch;
  }

private:
  /// Lightweight Task pointer wrapper, for the purpose of traversing to the
  /// Task's next sibling (via `ChildFragment::NextChild`).
  struct Task {
    ProcessSP process_sp;
    addr_t addr;

    operator bool() const { return addr && addr != LLDB_INVALID_ADDRESS; }

    bool operator==(const Task &other) const { return addr == other.addr; }
    bool operator!=(const Task &other) const { return !(*this == other); }

    static constexpr offset_t NextChildOffset = ChildFragmentOffset + 0x8;

    Task getNextChild(Status &status) {
      addr_t next_task = LLDB_INVALID_ADDRESS;
      if (status.Success())
        next_task =
            process_sp->ReadPointerFromMemory(addr + NextChildOffset, status);
      return {process_sp, next_task};
    }
  };

  /// Lightweight wrapper around TaskGroup opaque pointers (`TaskGroupBase`),
  /// for the purpose of traversing its child tasks.
  struct TaskGroupBase {
    ProcessSP process_sp;
    addr_t addr;

    // FirstChild offset for a TaskGroupBase instance.
    static constexpr offset_t FirstChildOffset = 0x18;
    static constexpr offset_t LastChildOffset = 0x20;

    Task getFirstChild(Status &status) {
      addr_t first_child = LLDB_INVALID_ADDRESS;
      if (status.Success())
        first_child =
            process_sp->ReadPointerFromMemory(addr + FirstChildOffset, status);
      return {process_sp, first_child};
    }

    Task getLastChild(Status &status) {
      addr_t last_child = LLDB_INVALID_ADDRESS;
      if (status.Success())
        last_child =
            process_sp->ReadPointerFromMemory(addr + LastChildOffset, status);
      return {process_sp, last_child};
    }
  };

private:
  bool m_is_supported_target = false;
  // Type for Swift.UnsafeCurrentTask.
  CompilerType m_task_type;
  // The TaskGroup's list of child task addresses.
  std::vector<addr_t> m_task_addrs;
  // Cache and storage of constructed child values.
  std::vector<ValueObjectSP> m_children;
};

/// Offset of ActiveActorStatus from _$defaultActor_.
///
/// DefaultActorImpl has the following (labeled) layout.
///
/// DefaultActorImpl:
///   0: HeapObject
/// $defaultActor:
///   16/0: isDistributedRemoteActor
///   17/1: <alignment padding>
///   32/16: StatusStorage
///
/// As shown, the $defaultActor field does not point to the start of the
/// DefaultActorImpl.
///
/// The StatusStorage is at offset of +32 from the start of DefaultActorImpl, or
/// at +16 relative to $defaultActor. The formatters are based on $defaultActor,
/// and as such use the relative offset.
static constexpr offset_t ActiveActorStatusOffset = 16;

class ActorSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  ActorSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp)
      : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
    bool is_64bit = false;
    if (auto target_sp = m_backend.GetTargetSP())
      is_64bit = target_sp->GetArchitecture().GetTriple().isArch64Bit();

    std::optional<uint32_t> concurrency_version;
    if (auto process_sp = m_backend.GetProcessSP())
      concurrency_version =
          SwiftLanguageRuntime::FindConcurrencyDebugVersion(*process_sp);

    m_is_supported_target = is_64bit && concurrency_version.value_or(0) == 1;
    if (!m_is_supported_target)
      return;

    auto target_sp = m_backend.GetTargetSP();
    auto ts_or_err =
        target_sp->GetScratchTypeSystemForLanguage(eLanguageTypeSwift);
    if (auto err = ts_or_err.takeError()) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
                     std::move(err),
                     "could not get Swift type system for Task synthetic "
                     "provider: {0}");
      return;
    }
    m_ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts_or_err->get());
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_is_supported_target ? 1 : 0;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    if (!m_is_supported_target || idx != 0)
      return {};

    if (!m_unprioritised_jobs_sp) {
      std::string mangled_typename =
          mangledTypenameForTasksTuple(m_job_addrs.size());
      CompilerType tasks_tuple_type =
          m_ts->GetTypeFromMangledTypename(ConstString(mangled_typename));
      DataExtractor data{m_job_addrs.data(),
                         m_job_addrs.size() * sizeof(addr_t),
                         endian::InlHostByteOrder(), sizeof(void *)};
      m_unprioritised_jobs_sp = ValueObject::CreateValueObjectFromData(
          "unprioritised_jobs", data, m_backend.GetExecutionContextRef(),
          tasks_tuple_type);
    }
    return m_unprioritised_jobs_sp;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    if (m_is_supported_target && name == "unprioritised_jobs")
      return 0;
    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }

  lldb::ChildCacheState Update() override {
    if (!m_is_supported_target)
      return ::eReuse;

    m_job_addrs.clear();

    if (!m_task_type)
      if (auto target_sp = m_backend.GetTargetSP()) {
        if (auto ts_or_err = target_sp->GetScratchTypeSystemForLanguage(
                eLanguageTypeSwift)) {
          if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
                  ts_or_err->get()))
            // TypeMangling for "Swift.UnsafeCurrentTask"
            m_task_type = ts->GetTypeFromMangledTypename(ConstString("$sSctD"));
        } else {
          LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
                         ts_or_err.takeError(),
                         "could not get Swift type system for Task synthetic "
                         "provider: {0}");
          return ChildCacheState::eReuse;
        }
      }

    if (!m_task_type)
      return ChildCacheState::eReuse;

    // Get the actor's queue of unprioritized jobs (tasks) by following the
    // "linked list" embedded in storage provided by SchedulerPrivate.
    DefaultActorImpl actor{m_backend.GetProcessSP(),
                           m_backend.GetLoadAddress()};
    Status status;
    Job first_job = actor.getFirstJob(status);
    Job current_job = first_job;
    while (current_job) {
      m_job_addrs.push_back(current_job.addr);
      current_job = current_job.getNextScheduledJob(status);
    }

    if (status.Fail()) {
      LLDB_LOG(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
               "could not read actor's job pointers: {0}", status.AsCString());
      return ChildCacheState::eReuse;
    }

    return ChildCacheState::eRefetch;
  }

  bool MightHaveChildren() override { return m_is_supported_target; }

private:
  /// Lightweight wrapper around Job pointers, for the purpose of traversing to
  /// the next scheduled Job.
  struct Job {
    ProcessSP process_sp;
    addr_t addr;

    operator bool() const { return addr && addr != LLDB_INVALID_ADDRESS; }

    // void *SchedulerPrivate[2] is the first Job specific field, its layout
    // follows the HeapObject base class (size 16).
    static constexpr offset_t SchedulerPrivateOffset = 16;
    static constexpr offset_t NextJobOffset = SchedulerPrivateOffset;

    Job getNextScheduledJob(Status &status) {
      addr_t next_job = LLDB_INVALID_ADDRESS;
      if (status.Success())
        next_job =
            process_sp->ReadPointerFromMemory(addr + NextJobOffset, status);
      return {process_sp, next_job};
    }
  };

  /// Lightweight wrapper around DefaultActorImpl/$defaultActor, for the purpose
  /// of accessing contents of ActiveActorStatus.
  struct DefaultActorImpl {
    ProcessSP process_sp;
    addr_t addr;

    // FirstJob's offset within ActiveActorStatus.
    static constexpr offset_t FirstJobOffset = ActiveActorStatusOffset + 8;

    Job getFirstJob(Status &status) {
      addr_t first_job = LLDB_INVALID_ADDRESS;
      if (status.Success())
        first_job =
            process_sp->ReadPointerFromMemory(addr + FirstJobOffset, status);
      return {process_sp, first_job};
    }
  };

private:
  bool m_is_supported_target = false;
  TypeSystemSwiftTypeRef *m_ts = nullptr;
  std::vector<addr_t> m_job_addrs;
  CompilerType m_task_type;
  ValueObjectSP m_unprioritised_jobs_sp;
};
}
}
}

lldb_private::formatters::swift::EnumSyntheticFrontEnd::EnumSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp.get()) {
  if (valobj_sp)
    Update();
}

llvm::Expected<uint32_t>
lldb_private::formatters::swift::EnumSyntheticFrontEnd::CalculateNumChildren() {
  if (m_indirect && m_projected)
    return m_projected->GetNumChildren();
  return m_projected ? 1 : 0;
}

lldb::ValueObjectSP
lldb_private::formatters::swift::EnumSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  ValueObjectSP value_sp;
  // Hide the indirection.
  if (m_indirect && m_projected) {
    value_sp = m_projected->GetChildAtIndex(idx);
  } else {
    if (idx != 0)
      return {};
    value_sp = m_projected;
  }
  if (!value_sp)
    return {};

  return value_sp;
}

lldb::ChildCacheState
lldb_private::formatters::swift::EnumSyntheticFrontEnd::Update() {
  auto *runtime = SwiftLanguageRuntime::Get(m_backend.GetProcessSP());
  if (!runtime)
    return ChildCacheState::eRefetch;

  llvm::Expected<ValueObjectSP> projected =
      runtime->SwiftLanguageRuntime::ProjectEnum(m_backend);
  if (!projected) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters), projected.takeError(),
                   "{0}");
    return ChildCacheState::eRefetch;
  }

  m_dynamic = m_backend.GetDynamicValueType();
  m_projected = *projected;
  if (m_projected &&
      m_projected->GetName().GetStringRef().starts_with("$indirect."))
    m_indirect = true;
  else
    m_indirect = false;

  if (!m_projected)
    return ChildCacheState::eRefetch;

  if ((m_projected->GetCompilerType().GetTypeInfo() & eTypeIsEnumeration))
    if (auto synthetic_sp = m_projected->GetSyntheticValue())
      m_projected = synthetic_sp;

  if (m_dynamic != eNoDynamicValues)
    if (auto dynamic_sp = m_projected->GetDynamicValue(m_dynamic))
      m_projected = dynamic_sp;
  return ChildCacheState::eRefetch;
}

bool lldb_private::formatters::swift::EnumSyntheticFrontEnd::
    MightHaveChildren() {
  return m_projected ? true : false;
}

llvm::Expected<size_t>
lldb_private::formatters::swift::EnumSyntheticFrontEnd::GetIndexOfChildWithName(
    ConstString name) {
  // Hide the indirection.
  if (m_indirect && m_projected)
    return m_projected->GetIndexOfChildWithName(name);
  if (m_projected && name == m_projected->GetName())
    return 0;
  return llvm::createStringError("Type has no child named '%s'",
                                 name.AsCString());
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::EnumSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return NULL;
  return (new EnumSyntheticFrontEnd(valobj_sp));
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::TaskSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return NULL;
  return new TaskSyntheticFrontEnd(valobj_sp);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::UnsafeContinuationSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new UnsafeContinuationSyntheticFrontEnd(valobj_sp);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::CheckedContinuationSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new CheckedContinuationSyntheticFrontEnd(valobj_sp);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::TaskGroupSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new TaskGroupSyntheticFrontEnd(valobj_sp);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::ActorSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return nullptr;
  return new ActorSyntheticFrontEnd(valobj_sp);
}

bool lldb_private::formatters::swift::ObjC_Selector_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_ptr("ptr");
  static ConstString g__rawValue("_rawValue");

  ValueObjectSP ptr_sp(valobj.GetChildAtNamePath({g_ptr, g__rawValue}));
  if (!ptr_sp)
    return false;

  auto ptr_value = ptr_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

  if (0 == ptr_value || LLDB_INVALID_ADDRESS == ptr_value)
    return false;

  StringPrinter::ReadStringAndDumpToStreamOptions read_options;
  read_options.SetLocation(ptr_value);
  read_options.SetTargetSP(valobj.GetTargetSP());
  read_options.SetStream(&stream);
  read_options.SetQuote('"');
  read_options.SetNeedsZeroTermination(true);
  read_options.SetEscapeStyle(StringPrinter::EscapeStyle::Swift);

  return StringPrinter::ReadStringAndDumpToStream<
      StringPrinter::StringElementType::ASCII>(read_options);
}

template <int Key> struct TypePreservingNSNumber;

template <> struct TypePreservingNSNumber<0> {
  typedef int64_t SixtyFourValueType;
  typedef int32_t ThirtyTwoValueType;

  static constexpr const char *FormatString = "Int(%" PRId64 ")";
};

template <> struct TypePreservingNSNumber<1> {
  typedef int64_t ValueType;
  static constexpr const char *FormatString = "Int64(%" PRId64 ")";
};

template <> struct TypePreservingNSNumber<2> {
  typedef int32_t ValueType;
  static constexpr const char *FormatString = "Int32(%" PRId32 ")";
};

template <> struct TypePreservingNSNumber<3> {
  typedef int16_t ValueType;
  static constexpr const char *FormatString = "Int16(%" PRId16 ")";
};

template <> struct TypePreservingNSNumber<4> {
  typedef int8_t ValueType;
  static constexpr const char *FormatString = "Int8(%" PRId8 ")";
};

template <> struct TypePreservingNSNumber<5> {
  typedef uint64_t SixtyFourValueType;
  typedef uint32_t ThirtyTwoValueType;

  static constexpr const char *FormatString = "UInt(%" PRIu64 ")";
};

template <> struct TypePreservingNSNumber<6> {
  typedef uint64_t ValueType;
  static constexpr const char *FormatString = "UInt64(%" PRIu64 ")";
};

template <> struct TypePreservingNSNumber<7> {
  typedef uint32_t ValueType;
  static constexpr const char *FormatString = "UInt32(%" PRIu32 ")";
};

template <> struct TypePreservingNSNumber<8> {
  typedef uint16_t ValueType;
  static constexpr const char *FormatString = "UInt16(%" PRIu16 ")";
};

template <> struct TypePreservingNSNumber<9> {
  typedef uint8_t ValueType;
  static constexpr const char *FormatString = "UInt8(%" PRIu8 ")";
};

template <> struct TypePreservingNSNumber<10> {
  typedef float ValueType;
  static constexpr const char *FormatString = "Float(%f)";
};

template <> struct TypePreservingNSNumber<11> {
  typedef double ValueType;
  static constexpr const char *FormatString = "Double(%f)";
};

template <> struct TypePreservingNSNumber<12> {
  typedef double SixtyFourValueType;
  typedef float ThirtyTwoValueType;

  static constexpr const char *FormatString = "CGFloat(%f)";
};

template <> struct TypePreservingNSNumber<13> {
  typedef bool ValueType;
  static constexpr const char *FormatString = "Bool(%d)";
};

template <int Key,
          typename Value = typename TypePreservingNSNumber<Key>::ValueType>
bool PrintTypePreservingNSNumber(DataBufferSP buffer_sp, Stream &stream) {
  Value value;
  memcpy(&value, buffer_sp->GetBytes(), sizeof(value));
  stream.Printf(TypePreservingNSNumber<Key>::FormatString, value);
  return true;
}

template <>
bool PrintTypePreservingNSNumber<13, void>(DataBufferSP buffer_sp,
                                           Stream &stream) {
  typename TypePreservingNSNumber<13>::ValueType value;
  memcpy(&value, buffer_sp->GetBytes(), sizeof(value));
  stream.PutCString(value ? "true" : "false");
  return true;
}

template <int Key, typename SixtyFour =
                       typename TypePreservingNSNumber<Key>::SixtyFourValueType,
          typename ThirtyTwo =
              typename TypePreservingNSNumber<Key>::ThirtyTwoValueType>
bool PrintTypePreservingNSNumber(DataBufferSP buffer_sp, ProcessSP process_sp,
                                 Stream &stream) {
  switch (process_sp->GetAddressByteSize()) {
  case 4: {
    ThirtyTwo value;
    memcpy(&value, buffer_sp->GetBytes(), sizeof(value));
    stream.Printf(TypePreservingNSNumber<Key>::FormatString, (SixtyFour)value);
    return true;
  }
  case 8: {
    SixtyFour value;
    memcpy(&value, buffer_sp->GetBytes(), sizeof(value));
    stream.Printf(TypePreservingNSNumber<Key>::FormatString, value);
    return true;
  }
  }

  llvm_unreachable("unknown address byte size");
}

bool lldb_private::formatters::swift::TypePreservingNSNumber_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  lldb::addr_t ptr_value(valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS));
  if (ptr_value == LLDB_INVALID_ADDRESS)
    return false;

  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;

  uint32_t ptr_size = process_sp->GetAddressByteSize();
  const uint32_t size_of_tag = 1;
  const uint32_t size_of_payload = 8;

  lldb::addr_t addr_of_payload = ptr_value + ptr_size;
  lldb::addr_t addr_of_tag = addr_of_payload + size_of_payload;

  Status read_error;
  uint64_t tag = process_sp->ReadUnsignedIntegerFromMemory(
      addr_of_tag, size_of_tag, 0, read_error);
  if (read_error.Fail())
    return false;

  WritableDataBufferSP buffer_sp(new DataBufferHeap(size_of_payload, 0));
  process_sp->ReadMemoryFromInferior(addr_of_payload, buffer_sp->GetBytes(),
                                     size_of_payload, read_error);
  if (read_error.Fail())
    return false;

#define PROCESS_DEPENDENT_TAG(Key)                                             \
  case Key:                                                                    \
    return PrintTypePreservingNSNumber<Key>(buffer_sp, process_sp, stream);
#define PROCESS_INDEPENDENT_TAG(Key)                                           \
  case Key:                                                                    \
    return PrintTypePreservingNSNumber<Key>(buffer_sp, stream);

  switch (tag) {
    PROCESS_DEPENDENT_TAG(0);
    PROCESS_INDEPENDENT_TAG(1);
    PROCESS_INDEPENDENT_TAG(2);
    PROCESS_INDEPENDENT_TAG(3);
    PROCESS_INDEPENDENT_TAG(4);
    PROCESS_DEPENDENT_TAG(5);
    PROCESS_INDEPENDENT_TAG(6);
    PROCESS_INDEPENDENT_TAG(7);
    PROCESS_INDEPENDENT_TAG(8);
    PROCESS_INDEPENDENT_TAG(9);
    PROCESS_INDEPENDENT_TAG(10);
    PROCESS_INDEPENDENT_TAG(11);
    PROCESS_DEPENDENT_TAG(12);
    PROCESS_INDEPENDENT_TAG(13);
  default:
    break;
  }

#undef PROCESS_DEPENDENT_TAG
#undef PROCESS_INDEPENDENT_TAG

  return false;
}

bool lldb_private::formatters::swift::TaskPriority_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  uint64_t raw_value = UINT8_MAX;
  if (auto child_sp = valobj.GetChildMemberWithName("rawValue"))
    if (auto synthetic_sp = child_sp->GetSyntheticValue())
      raw_value = synthetic_sp->GetValueAsUnsigned(UINT8_MAX);
  if (raw_value >= UINT8_MAX)
    return false;

  switch (raw_value) {
  case 0x19:
    // Also .userInitiated
    stream.PutCString(".high");
    break;
  case 0x15:
    // Also .default (deprecated)
    stream.PutCString(".medium");
    break;
  case 0x11:
    // Also .utilitiy
    stream.PutCString(".low");
    break;
  case 0x09:
    stream.PutCString(".background");
    break;
  default:
    stream.Format("{0}", raw_value);
    break;
  }
  return true;
}

static const std::pair<StringRef, StringRef> TASK_FLAGS[] = {
    {"isComplete", "complete"},
    {"isSuspended", "suspended"},
    {"isRunning", "running"},
    {"isCancelled", "cancelled"},
    {"isEscalated", "escalated"},
    {"isEnqueued", "enqueued"},
    {"isGroupChildTask", "groupChildTask"},
    {"isAsyncLetTask", "asyncLetTask"},
    {"isStatusRecordLocked", "statusRecordLocked"},
};

bool lldb_private::formatters::swift::Task_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  auto get_member = [&valobj](StringRef name) -> uint64_t {
    if (auto member_sp = valobj.GetChildMemberWithName(name))
      if (auto synthetic_sp = member_sp->GetSyntheticValue())
        return synthetic_sp->GetValueAsUnsigned(0);
    return 0;
  };

  addr_t task_addr = get_member("address");
  if (auto process_sp = valobj.GetProcessSP()) {
    if (auto name_or_err = GetTaskName(task_addr, *process_sp)) {
      if (auto maybe_name = *name_or_err)
        stream.Format("\"{0}\" ", *maybe_name);
    } else {
      LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters | LLDBLog::Types),
                     name_or_err.takeError(),
                     "failed to determine name of task {1:x}: {0}", task_addr);
    }
  }

  stream.Format("id:{0}", get_member("id"));

  std::vector<StringRef> flags;
  for (auto [member, flag] : TASK_FLAGS)
    if (get_member(member))
      flags.push_back(flag);

  if (!flags.empty())
    // Append the flags in an `|` separated list. This matches the format used
    // by swift-inspect dump-concurrency.
    stream.Format(" flags:{0:$[|]}", iterator_range(flags));

  return true;
}

bool lldb_private::formatters::swift::Actor_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static constexpr offset_t FlagsOffset = ActiveActorStatusOffset;
  auto addr = valobj.GetLoadAddress();
  if (addr == LLDB_INVALID_ADDRESS)
    return false;

  auto flags_addr = addr + FlagsOffset;
  Status status;
  uint64_t flags = 0;
  if (auto process_sp = valobj.GetProcessSP())
    flags = process_sp->ReadUnsignedIntegerFromMemory(flags_addr, 4, 0, status);

  if (status.Fail()) {
    stream.PutCString("<could not read actor state>");
    return true;
  }

  using namespace ::swift::concurrency::ActorFlagConstants;
  uint8_t state = flags & ActorStateMask;
  static_assert(Zombie_ReadyForDeallocation == 3);
  if (state > Zombie_ReadyForDeallocation) {
    stream << "<unknown actor state: " << Twine(state).str() << ">";
    return true;
  }

  static const StringRef states[] = {"idle", "scheduled", "running", "zombie"};
  stream.PutCString(states[state]);
  return true;
}

namespace {

/// Enumerate the kinds of SIMD elements.
enum class SIMDElementKind {
  Int32,
  UInt32,
  Float32,
  Float64
};

/// A helper for formatting a kind of SIMD element.
class SIMDElementFormatter {
  SIMDElementKind m_kind;

public:
  SIMDElementFormatter(SIMDElementKind kind) : m_kind(kind) {}

  /// Create a string representation of a SIMD element given a pointer to it.
  std::string Format(const uint8_t *data) const {
    std::string S;
    llvm::raw_string_ostream OS(S);
    switch (m_kind) {
    case SIMDElementKind::Int32: {
      auto *p = reinterpret_cast<const int32_t *>(data);
      OS << *p;
      break;
    }
    case SIMDElementKind::UInt32: {
      auto *p = reinterpret_cast<const uint32_t *>(data);
      OS << *p;
      break;
    }
    case SIMDElementKind::Float32: {
      auto *p = reinterpret_cast<const float *>(data);
      OS << *p;
      break;
    }
    case SIMDElementKind::Float64: {
      auto *p = reinterpret_cast<const double *>(data);
      OS << *p;
      break;
    }
    }
    return S;
  }

  /// Get the size in bytes of this kind of SIMD element.
  unsigned getElementSize() const {
    return (m_kind == SIMDElementKind::Float64) ? 8 : 4;
  }
};

/// Read a vector from a buffer target.
std::optional<std::vector<std::string>>
ReadVector(const SIMDElementFormatter &formatter, const uint8_t *buffer,
           unsigned len, unsigned offset, unsigned num_elements) {
  unsigned elt_size = formatter.getElementSize();
  if ((offset + num_elements * elt_size) > len)
    return std::nullopt;
  std::vector<std::string> elements;
  for (unsigned I = 0; I < num_elements; ++I)
    elements.emplace_back(formatter.Format(buffer + offset + (I * elt_size)));
  return elements;
}

/// Read a SIMD vector from the target.
std::optional<std::vector<std::string>>
ReadVector(Process &process, ValueObject &valobj,
           const SIMDElementFormatter &formatter, unsigned num_elements) {
  Status error;
  static ConstString g_storage("_storage");
  static ConstString g_value("_value");
  ValueObjectSP value_sp = valobj.GetChildAtNamePath({g_storage, g_value});
  if (!value_sp)
    return std::nullopt;

  // The layout of the vector is the same as what you'd expect for a C-style
  // array. It's a contiguous bag of bytes with no padding.
  lldb_private::DataExtractor data;
  uint64_t len = value_sp->GetData(data, error);
  if (error.Fail())
    return std::nullopt;

  const uint8_t *buffer = data.GetDataStart();
  return ReadVector(formatter, buffer, len, 0, num_elements);
}

/// Print a vector of elements as a row, if possible.
bool PrintRow(Stream &stream, std::optional<std::vector<std::string>> vec) {
  if (!vec)
    return false;

  std::string joined = llvm::join(*vec, ", ");
  stream.Printf("(%s)", joined.c_str());
  return true;
}

void PrintMatrix(Stream &stream,
                 const std::vector<std::vector<std::string>> &matrix,
                 int num_columns, int num_rows) {
  // Print each row.
  stream.Printf("\n[ ");
  for (int J = 0; J < num_rows; ++J) {
    // Join the J-th row's elements with commas.
    std::vector<std::string> row;
    for (int I = 0; I < num_columns; ++I)
      row.emplace_back(std::move(matrix[I][J]));
    std::string joined = llvm::join(row, ", ");

    // Add spacing and punctuation to 1) make it possible to copy the matrix
    // into a Python repl and 2) to avoid writing '[[' in FileCheck tests.
    if (J > 0)
      stream.Printf("  ");
    stream.Printf("[%s]", joined.c_str());
    if (J != (num_rows - 1))
      stream.Printf(",\n");
    else
      stream.Printf(" ]\n");
  }
}

} // end anonymous namespace

bool lldb_private::formatters::swift::SIMDVector_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  // SIMD vector contains an inner member `_storage` which is an opaque
  // container. Given SIMD is always in the form SIMDX<Type> where X is a
  // positive integer, we can calculate the number of elements and the
  // dynamic archetype (and hence its size). Everything follows naturally
  // as the elements are laid out in a contigous buffer without padding.
  CompilerType simd_type = valobj.GetCompilerType().GetCanonicalType();
  auto ts = simd_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return false;

  ExecutionContext exe_ctx = valobj.GetExecutionContextRef().Lock(true);
  std::optional<uint64_t> opt_type_size = llvm::expectedToOptional(
      simd_type.GetByteSize(exe_ctx.GetBestExecutionContextScope()));
  if (!opt_type_size)
    return false;
  uint64_t type_size = *opt_type_size;

  lldbassert(simd_type.GetNumTemplateArguments() == 1 && "broken SIMD type");
  if (simd_type.GetNumTemplateArguments() != 1)
    return false;

  auto arg_type = ts->GetGenericArgumentType(simd_type.GetOpaqueQualType(), 0);
  lldbassert(arg_type && "Unexpected invalid SIMD generic argument type");
  if (!arg_type)
    return false;

  std::optional<uint64_t> opt_arg_size = llvm::expectedToOptional(
      arg_type.GetByteSize(exe_ctx.GetBestExecutionContextScope()));
  if (!opt_arg_size)
    return false;
  uint64_t arg_size = *opt_arg_size;

  DataExtractor storage_buf;
  Status error;
  uint64_t len = valobj.GetData(storage_buf, error);
  lldbassert(len == type_size && "extracted less bytes than requested");
  if (len < type_size)
    return false;

  // We deduce the number of elements looking at the size of the swift
  // type and the size of the generic argument, as we know the type is
  // laid out contiguosly in memory. SIMD3, though, has an element of
  // padding. Given this is the only type in the standard library with
  // padding, we special-case it.
  ConstString full_type_name = simd_type.GetTypeName();
  llvm::StringRef type_name = full_type_name.GetStringRef();
  uint64_t num_elements = type_size / arg_size;
  auto generic_pos = type_name.find("<");
  if (generic_pos != llvm::StringRef::npos)
    type_name = type_name.slice(0, generic_pos);
  if (type_name == "Swift.SIMD3")
    num_elements = 3;

  std::vector<std::string> elem_vector;
  for (uint64_t i = 0; i < num_elements; ++i) {
    DataExtractor elem_extractor(storage_buf, i * arg_size, arg_size);
    auto simd_elem = ValueObject::CreateValueObjectFromData(
        "simd_elem", elem_extractor, valobj.GetExecutionContextRef(), arg_type);
    if (!simd_elem || simd_elem->GetError().Fail())
      return false;

    auto synthetic = simd_elem->GetSyntheticValue();
    if (!synthetic)
      return false;
    const char *value_string = synthetic->GetValueAsCString();
    elem_vector.push_back(value_string);
  }

  return PrintRow(stream, elem_vector);
}

bool lldb_private::formatters::swift::LegacySIMD_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  Status error;
  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;
  Process &process = *process_sp.get();

  // Get the type name without the "simd.simd_" prefix.
  ConstString full_type_name = valobj.GetTypeName();

  llvm::StringRef type_name = full_type_name.GetStringRef();
  if (type_name.starts_with("simd."))
    type_name = type_name.drop_front(5);
  if (type_name.starts_with("simd_"))
    type_name = type_name.drop_front(5);

  // Get the type of object this is.
  bool is_quaternion = type_name.starts_with("quat");
  bool is_matrix = type_name[type_name.size() - 2] == 'x';
  bool is_vector = !is_matrix && !is_quaternion;

  // Get the kind of SIMD element inside of this object.
  std::optional<SIMDElementKind> kind = std::nullopt;
  if (type_name.starts_with("int"))
    kind = SIMDElementKind::Int32;
  else if (type_name.starts_with("uint"))
    kind = SIMDElementKind::UInt32;
  else if ((is_quaternion && type_name.ends_with("f")) ||
           type_name.starts_with("float"))
    kind = SIMDElementKind::Float32;
  else if ((is_quaternion && type_name.ends_with("d")) ||
           type_name.starts_with("double"))
    kind = SIMDElementKind::Float64;
  if (!kind)
    return false;

  SIMDElementFormatter formatter(*kind);

  if (is_vector) {
    unsigned num_elements = llvm::hexDigitValue(type_name.back());
    return PrintRow(stream,
                    ReadVector(process, valobj, formatter, num_elements));
  } else if (is_quaternion) {
    static ConstString g_vector("vector");
    ValueObjectSP vec_sp = valobj.GetChildAtNamePath({g_vector});
    if (!vec_sp)
      return false;

    return PrintRow(stream, ReadVector(process, *vec_sp.get(), formatter, 4));
  } else if (is_matrix) {
    static ConstString g_columns("columns");
    ValueObjectSP columns_sp = valobj.GetChildAtNamePath({g_columns});
    if (!columns_sp)
      return false;

    unsigned num_columns = llvm::hexDigitValue(type_name[type_name.size() - 3]);
    unsigned num_rows = llvm::hexDigitValue(type_name[type_name.size() - 1]);

    // SIMD matrices are stored column-major. Collect each column vector as a
    // precursor for row-by-row pretty-printing.
    std::vector<std::vector<std::string>> columns;
    for (unsigned I = 0; I < num_columns; ++I) {
      std::string col_num_str = llvm::utostr(I);
      ConstString col_num_const_str(col_num_str.c_str());
      ValueObjectSP column_sp =
          columns_sp->GetChildAtNamePath({col_num_const_str});
      if (!column_sp)
        return false;

      auto vec = ReadVector(process, *column_sp.get(), formatter, num_rows);
      if (!vec)
        return false;

      columns.emplace_back(std::move(*vec));
    }

    PrintMatrix(stream, columns, num_columns, num_rows);
    return true;
  }

  return false;
}

bool lldb_private::formatters::swift::GLKit_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  // Get the type name without the "GLKit." prefix.
  ConstString full_type_name = valobj.GetTypeName();
  llvm::StringRef type_name = full_type_name.GetStringRef();
  if (type_name.starts_with("GLKit."))
    type_name = type_name.drop_front(6);

  // Get the type of object this is.
  bool is_quaternion = type_name == "GLKQuaternion";
  bool is_matrix = type_name.starts_with("GLKMatrix");
  bool is_vector = type_name.starts_with("GLKVector");

  if (!(is_quaternion || is_matrix || is_vector))
    return false;

  SIMDElementFormatter formatter(SIMDElementKind::Float32);

  unsigned num_elements =
      is_quaternion ? 4 : llvm::hexDigitValue(type_name.back());
  DataExtractor data;
  Status error;
  uint64_t len = valobj.GetData(data, error);
  const uint8_t *buffer = data.GetDataStart();
  if (!is_matrix) {
    return PrintRow(stream,
                    ReadVector(formatter, buffer, len, 0, num_elements));
  }

  // GLKit matrices are stored column-major. Collect each column vector as a
  // precursor for row-by-row pretty-printing.
  std::vector<std::vector<std::string>> columns;
  for (unsigned I = 0; I < num_elements; ++I) {
    auto vec =
        ReadVector(formatter, buffer, len, I * 4 * num_elements, num_elements);
    if (!vec)
      return false;

    columns.emplace_back(std::move(*vec));
  }

  PrintMatrix(stream, columns, num_elements, num_elements);
  return true;
}
