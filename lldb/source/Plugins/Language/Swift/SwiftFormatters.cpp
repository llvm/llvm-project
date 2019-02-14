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
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Status.h"
#include "swift/AST/Types.h"
#include "swift/Demangling/ManglingMacros.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

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

/// Get an Obj-C object's class name, if one is present.
static Optional<StringRef> getObjC_ClassName(ValueObject &valobj,
                                             Process &process) {
  ObjCLanguageRuntime *runtime =
      (ObjCLanguageRuntime *)process.GetLanguageRuntime(
          lldb::eLanguageTypeObjC);
  if (!runtime)
    return None;

  ObjCLanguageRuntime::ClassDescriptorSP descriptor(
      runtime->GetClassDescriptor(valobj));

  if (!descriptor.get() || !descriptor->IsValid())
    return None;

  ConstString class_name_cs = descriptor->GetClassName();
  return class_name_cs.GetStringRef();
}

static bool readStringFromAddress(
    uint64_t startAddress, uint64_t length, ProcessSP process, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {
  if (length == 0) {
    stream.Printf("\"\"");
    return true;
  }

  read_options.SetLocation(startAddress);
  read_options.SetProcessSP(process);
  read_options.SetStream(&stream);
  read_options.SetSourceSize(length);
  read_options.SetNeedsZeroTermination(false);
  read_options.SetIgnoreMaxLength(summary_options.GetCapping() ==
                                  lldb::eTypeSummaryUncapped);
  read_options.SetBinaryZeroIsTerminator(false);
  read_options.SetLanguage(lldb::eLanguageTypeSwift);

  return StringPrinter::ReadStringAndDumpToStream<
      StringPrinter::StringElementType::UTF8>(read_options);
};

bool lldb_private::formatters::swift::StringGuts_SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summary_options,
    StringPrinter::ReadStringAndDumpToStreamOptions read_options) {

  static ConstString g__object("_object");
  static ConstString g__storage("_storage");
  static ConstString g__value("_value");

  ProcessSP process(valobj.GetProcessSP());
  if (!process)
    return false;

  auto ptrSize = process->GetAddressByteSize();

  auto object_sp = valobj.GetChildMemberWithName(g__object, true);
  if (!object_sp)
    return false;

  // We retrieve String contents by first extracting the
  // platform-independent 128-bit raw value representation from
  // _StringObject, then interpreting that.
  Status error;
  uint64_t raw0;
  uint64_t raw1;

  if (ptrSize == 8) {
    // On 64-bit platforms, we simply need to get the raw integer
    // values of the two stored properties.
    static ConstString g__countAndFlagsBits("_countAndFlagsBits");

    auto countAndFlagsBits = object_sp->GetChildAtNamePath(
      {g__countAndFlagsBits, g__value});
    if (!countAndFlagsBits)
      return false;
    raw0 = countAndFlagsBits->GetValueAsUnsigned(0);

    auto object = object_sp->GetChildMemberWithName(g__object, true);
    if (!object)
      return false;
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
      return false;
    uint64_t count = count_sp->GetValueAsUnsigned(0);

    auto discriminator_sp =
        object_sp->GetChildAtNamePath({g__discriminator, g__value});
    if (!discriminator_sp)
      return false;
    uint64_t discriminator = discriminator_sp->GetValueAsUnsigned(0) & 0xff;

    auto flags_sp = object_sp->GetChildAtNamePath({g__flags, g__value});
    if (!flags_sp)
      return false;
    uint64_t flags = flags_sp->GetValueAsUnsigned(0) & 0xffff;

    auto variant_sp = object_sp->GetChildMemberWithName(g__variant, true);
    if (!variant_sp)
      return false;

    llvm::StringRef variantCase = variant_sp->GetValueAsCString();

    ValueObjectSP payload_sp;
    if (variantCase.startswith("immortal")) {
      payload_sp = variant_sp->GetChildAtNamePath({g_immortal, g__value});
    } else if (variantCase.startswith("native")) {
      payload_sp = variant_sp->GetChildAtNamePath({g_immortal, g__value});
    } else if (variantCase.startswith("bridged")) {
      static ConstString g_bridged("bridged");
      auto anyobject_sp = variant_sp->GetChildMemberWithName(g_bridged, true);
      if (!anyobject_sp)
        return false;
      payload_sp = anyobject_sp->GetChildAtIndex(0, true); // "instance"
    } else {
      lldbassert(false && "Uknown variant");
      return false;
    }
    if (!payload_sp)
      return false;
    uint64_t pointerBits = payload_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);

    if (pointerBits == LLDB_INVALID_ADDRESS)
      return false;

    if ((discriminator & 0xB0) == 0xA0) {
      raw0 = count | (pointerBits << 32);
      raw1 = flags | (discriminator << 56);
    } else {
      raw0 = count | (flags << 48);
      raw1 = pointerBits | (discriminator << 56);
    }
  } else {
    lldbassert(false && "Unsupported arch?");
    return false;
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

  if ((discriminator & 0xB0) == 0xA0) { // 1x10xxxx: Small string
    uint64_t count = (raw1 >> 56) & 0x0F;
    uint64_t maxCount = (ptrSize == 8 ? 15 : 10);
    if (count > maxCount)
      return false;

    uint64_t buffer[2] = {raw0, raw1};
    DataExtractor data(buffer, count, process->GetByteOrder(), ptrSize);

    StringPrinter::ReadBufferAndDumpToStreamOptions options(read_options);
    options.SetData(data)
        .SetStream(&stream)
        .SetSourceSize(count)
        .SetBinaryZeroIsTerminator(false)
        .SetLanguage(lldb::eLanguageTypeSwift);

    return StringPrinter::ReadBufferAndDumpToStream<
        StringPrinter::StringElementType::UTF8>(options);

  }

  uint64_t count = raw0 & 0x0000FFFFFFFFFFFF;
  uint16_t flags = raw0 >> 48;
  lldb::addr_t objectAddress = (raw1 & 0x0FFFFFFFFFFFFFFF);
  if ((flags & 0x1000) != 0) { // Tail-allocated / biased address
    lldbassert((discriminator & 0x70) == 0 &&
      "tail-allocation is only for natively stored or literals");
    uint64_t bias = (ptrSize == 8 ? 32 : 20);
    auto address = objectAddress + bias;
    return readStringFromAddress(
      address, count, process, stream, summary_options, read_options);
  }

  if ((discriminator & 0xF0) == 0x00) { // Shared string
    // FIXME: Verify that there is a __SharedStringStorage instance at `address`.
    lldbassert((flags & 0x3000) == 0);
    uint64_t startOffset = (ptrSize == 8 ? 24 : 12);
    auto address = objectAddress + startOffset;
    lldb::addr_t start = process->ReadPointerFromMemory(address, error);
    if (error.Fail())
      return false;

    return readStringFromAddress(
      start, count, process, stream, summary_options, read_options);
  }

  lldbassert((discriminator & 0x70) != 0 &&
    "native/shared strings already handled");

  if ((discriminator & 0xE0) == 0x40) { // 010xxxxx: Bridged
    CompilerType id_type =
        process->GetTarget().GetScratchClangASTContext()->GetBasicType(
            lldb::eBasicTypeObjCID);

    // We may have an NSString pointer inline, so try formatting it directly.
    DataExtractor DE(&objectAddress, ptrSize, process->GetByteOrder(), ptrSize);
    auto nsstring = ValueObject::CreateValueObjectFromData(
        "nsstring", DE, valobj.GetExecutionContextRef(), id_type);
    if (!nsstring || nsstring->GetError().Fail())
      return false;

    return NSStringSummaryProvider(*nsstring.get(), stream, summary_options);
  }

  if ((discriminator & 0xF8) == 0x18) { // 0001xxxx: Foreign
    // Not currently generated
    lldbassert(
      false && "Foreign non-bridged strings are not currently used in Swift");
    return false;
  }

  lldbassert(false && "Invalid discriminator");
  return false;
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

  read_options.SetProcessSP(process_sp);
  read_options.SetLocation(start_ptr);
  read_options.SetSourceSize(size);
  read_options.SetBinaryZeroIsTerminator(false);
  read_options.SetNeedsZeroTermination(false);
  read_options.SetStream(&stream);
  read_options.SetIgnoreMaxLength(summary_options.GetCapping() ==
                                  lldb::eTypeSummaryUncapped);
  read_options.SetLanguage(lldb::eLanguageTypeSwift);

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

  lldb::addr_t raw1 = valobj.GetPointerValue();
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

  return readStringFromAddress(start, count, process, stream, summary_options,
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
  lldb::addr_t raw1 = valobj.GetPointerValue();
  lldb::addr_t address = (raw1 & 0x00FFFFFFFFFFFFFF) + bias;

  Status error;
  lldb::addr_t raw0 = process->ReadPointerFromMemory(raw1 + raw0_offset, error);
  if (error.Fail())
    return false;
  uint64_t count = raw0 & 0x0000FFFFFFFFFFFF;
  return readStringFromAddress(
      address, count, process, stream, options,
      StringPrinter::ReadStringAndDumpToStreamOptions());
}

bool lldb_private::formatters::swift::Bool_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g_value("_value");
  ValueObjectSP value_child(valobj.GetChildMemberWithName(g_value, true));
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

bool lldb_private::formatters::swift::StridedRangeGenerator_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  static ConstString g__bounds("_bounds");
  static ConstString g__stride("_stride");

  ValueObjectSP bounds_sp(valobj.GetChildMemberWithName(g__bounds, true));
  ValueObjectSP stride_sp(valobj.GetChildMemberWithName(g__stride, true));

  if (!bounds_sp || !stride_sp)
    return false;

  auto bounds_summary = bounds_sp->GetSummaryAsCString();
  auto stride_summary = stride_sp->GetValueAsCString();

  if (!bounds_summary || !bounds_summary[0] || !stride_summary ||
      !stride_summary[0])
    return false;

  stream.Printf("(%s).by(%s)", bounds_summary, stride_summary);

  return true;
}

bool lldb_private::formatters::swift::BuiltinObjC_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  stream.Printf("0x%" PRIx64 " ", valobj.GetValueAsUnsigned(0));
  stream.Printf("%s", valobj.GetObjectDescription());
  return true;
}

namespace lldb_private {
namespace formatters {
namespace swift {
class EnumSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  EnumSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp);

  virtual size_t CalculateNumChildren();

  virtual lldb::ValueObjectSP GetChildAtIndex(size_t idx);

  virtual bool Update();

  virtual bool MightHaveChildren();

  virtual size_t GetIndexOfChildWithName(const ConstString &name);

  virtual ~EnumSyntheticFrontEnd() = default;

private:
  ExecutionContextRef m_exe_ctx_ref;
  ConstString m_element_name;
  size_t m_child_index;
};
}
}
}

lldb_private::formatters::swift::EnumSyntheticFrontEnd::EnumSyntheticFrontEnd(
    lldb::ValueObjectSP valobj_sp)
    : SyntheticChildrenFrontEnd(*valobj_sp.get()), m_exe_ctx_ref(),
      m_element_name(nullptr), m_child_index(UINT32_MAX) {
  if (valobj_sp)
    Update();
}

size_t
lldb_private::formatters::swift::EnumSyntheticFrontEnd::CalculateNumChildren() {
  return m_child_index != UINT32_MAX ? 1 : 0;
}

lldb::ValueObjectSP
lldb_private::formatters::swift::EnumSyntheticFrontEnd::GetChildAtIndex(
    size_t idx) {
  if (idx)
    return ValueObjectSP();
  if (m_child_index == UINT32_MAX)
    return ValueObjectSP();
  return m_backend.GetChildAtIndex(m_child_index, true);
}

bool lldb_private::formatters::swift::EnumSyntheticFrontEnd::Update() {
  m_element_name.Clear();
  m_child_index = UINT32_MAX;
  m_exe_ctx_ref = m_backend.GetExecutionContextRef();
  m_element_name.SetCString(m_backend.GetValueAsCString());
  m_child_index = m_backend.GetIndexOfChildWithName(m_element_name);
  return false;
}

bool lldb_private::formatters::swift::EnumSyntheticFrontEnd::
    MightHaveChildren() {
  return m_child_index != UINT32_MAX;
}

size_t
lldb_private::formatters::swift::EnumSyntheticFrontEnd::GetIndexOfChildWithName(
    const ConstString &name) {
  if (name == m_element_name)
    return 0;
  return UINT32_MAX;
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::EnumSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (!valobj_sp)
    return NULL;
  return (new EnumSyntheticFrontEnd(valobj_sp));
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
  read_options.SetLocation(ptr_value)
      .SetProcessSP(valobj.GetProcessSP())
      .SetStream(&stream)
      .SetQuote('"')
      .SetNeedsZeroTermination(true)
      .SetLanguage(lldb::eLanguageTypeSwift);

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

  DataBufferSP buffer_sp(new DataBufferHeap(size_of_payload, 0));
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
llvm::Optional<std::vector<std::string>>
ReadVector(const SIMDElementFormatter &formatter, const uint8_t *buffer,
           unsigned len, unsigned offset, unsigned num_elements) {
  unsigned elt_size = formatter.getElementSize();
  if ((offset + num_elements * elt_size) > len)
    return llvm::None;
  std::vector<std::string> elements;
  for (unsigned I = 0; I < num_elements; ++I)
    elements.emplace_back(formatter.Format(buffer + offset + (I * elt_size)));
  return elements;
}

/// Read a SIMD vector from the target.
llvm::Optional<std::vector<std::string>>
ReadVector(Process &process, ValueObject &valobj,
           const SIMDElementFormatter &formatter, unsigned num_elements) {
  Status error;
  static ConstString g_storage("_storage");
  static ConstString g_value("_value");
  ValueObjectSP value_sp = valobj.GetChildAtNamePath({g_storage, g_value});
  if (!value_sp)
    return llvm::None;

  // The layout of the vector is the same as what you'd expect for a C-style
  // array. It's a contiguous bag of bytes with no padding.
  DataExtractor data;
  uint64_t len = value_sp->GetData(data, error);
  unsigned elt_size = formatter.getElementSize();
  if (error.Fail())
    return llvm::None;

  const uint8_t *buffer = data.GetDataStart();
  return ReadVector(formatter, buffer, len, 0, num_elements);
}

/// Print a vector of elements as a row, if possible.
bool PrintRow(Stream &stream, llvm::Optional<std::vector<std::string>> vec) {
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
  for (unsigned J = 0; J < num_rows; ++J) {
    // Join the J-th row's elements with commas.
    std::vector<std::string> row;
    for (unsigned I = 0; I < num_columns; ++I)
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
  Status error;
  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;
  Process &process = *process_sp.get();

  // SIMD vector contains an inner member `_storage` which is an opaque
  // container. Given SIMD is always in the form SIMDX<Type> where X is a
  // positive integer, we can calculate the number of elements and the
  // dynamic archetype (and hence its size). Everything follows naturally
  // as the elements are laid out in a contigous buffer without padding.
  CompilerType simd_type = valobj.GetCompilerType();
  void *type_buffer = reinterpret_cast<void *>(simd_type.GetOpaqueQualType());
  llvm::Optional<uint64_t> opt_type_size = simd_type.GetByteSize(nullptr);
  if (!opt_type_size)
    return false;
  uint64_t type_size = *opt_type_size;

  auto swift_type = reinterpret_cast<::swift::TypeBase *>(type_buffer);
  auto bound_type = dyn_cast<::swift::BoundGenericType>(swift_type);
  if (!bound_type)
    return false;
  auto generic_args = bound_type->getGenericArgs();
  lldbassert(generic_args.size() == 1 && "broken SIMD type");
  if (generic_args.size() != 1)
    return false;
  auto swift_arg_type = generic_args[0];
  CompilerType arg_type(swift_arg_type);

  llvm::Optional<uint64_t> opt_arg_size = arg_type.GetByteSize(nullptr);
  if (!opt_arg_size)
    return false;
  uint64_t arg_size = *opt_arg_size;

  lldb::addr_t object_address = valobj.GetAddressOf();
  if (object_address == LLDB_INVALID_ADDRESS)
    return false;

  uint64_t num_elements = type_size / arg_size;
  std::vector<std::string> elem_string;
  for (uint64_t i = 0; i < num_elements; ++i) {
    DataBufferSP buffer_sp(new DataBufferHeap(arg_size, 0));
    uint8_t *data_ptr = buffer_sp->GetBytes();
    process.ReadMemory(object_address, data_ptr, arg_size, error);
    if (error.Fail())
      return false;
    DataExtractor DE(buffer_sp, process_sp->GetByteOrder(),
                     process_sp->GetAddressByteSize());

    auto simd_elem = ValueObject::CreateValueObjectFromData(
        "simd_elem", DE, valobj.GetExecutionContextRef(), arg_type);
    if (!simd_elem || simd_elem->GetError().Fail())
      return false;

    auto synthetic = simd_elem->GetSyntheticValue();
    const char *value_string = synthetic->GetValueAsCString();
    if (!value_string)
      value_string = synthetic->GetSummaryAsCString();
    if (!value_string)
      return false;
    elem_string.push_back(std::string(value_string));
    object_address += arg_size;
  }
  return PrintRow(stream, elem_string);
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
  if (type_name.startswith("simd."))
    type_name = type_name.drop_front(5);
  if (type_name.startswith("simd_"))
    type_name = type_name.drop_front(5);

  // Get the type of object this is.
  bool is_quaternion = type_name.startswith("quat");
  bool is_matrix = type_name[type_name.size() - 2] == 'x';
  bool is_vector = !is_matrix && !is_quaternion;

  // Get the kind of SIMD element inside of this object.
  llvm::Optional<SIMDElementKind> kind = llvm::None;
  if (type_name.startswith("int"))
    kind = SIMDElementKind::Int32;
  else if (type_name.startswith("uint"))
    kind = SIMDElementKind::UInt32;
  else if ((is_quaternion && type_name.endswith("f")) ||
           type_name.startswith("float"))
    kind = SIMDElementKind::Float32;
  else if ((is_quaternion && type_name.endswith("d")) ||
           type_name.startswith("double"))
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
  Status error;

  ProcessSP process_sp(valobj.GetProcessSP());
  if (!process_sp)
    return false;

  Process &process = *process_sp.get();

  // Get the type name without the "simd.simd_" prefix.
  ConstString full_type_name = valobj.GetTypeName();
  llvm::StringRef type_name = full_type_name.GetStringRef();

  // Get the type of object this is.
  bool is_quaternion = type_name == "GLKQuaternion";
  bool is_matrix = type_name.startswith("GLKMatrix");
  bool is_vector = type_name.startswith("GLKVector");

  if (!(is_quaternion || is_matrix || is_vector))
    return false;

  SIMDElementFormatter formatter(SIMDElementKind::Float32);

  unsigned num_elements =
      is_quaternion ? 4 : llvm::hexDigitValue(type_name.back());
  DataExtractor data;
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
