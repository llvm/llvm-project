//===-- GNUstepFormatters.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepFormatters.h"
#include "Cocoa.h"

#include "Plugins/LanguageRuntime/ObjC/GNUstepObjCRuntime/GNUstepObjCRuntime.h"
#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"

#include <cmath>
#include <cstring>
#include <ctime>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool lldb_private::formatters::IsGNUstepObjCRuntime(ValueObject &valobj) {
  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;
  return llvm::isa_and_nonnull<GNUstepObjCRuntime>(
      ObjCLanguageRuntime::Get(*process_sp));
}

ValueObjectSP lldb_private::formatters::GNUstepGetIvar(ValueObject &valobj,
                                                       llvm::StringRef name) {
  // Formatters usually receive the dynamic value already, but the static
  // value arrives when a provider registered under an abstract class name
  // (NSArray) dispatches here; the ivars live on the concrete class, so ask
  // for the dynamic value in that case.
  ValueObjectSP object_sp = valobj.GetSP();
  if (!object_sp)
    return {};
  // A summary runs on the value the synthetic children were attached to,
  // whose "children" are the elements; the ivars are on the value beneath.
  if (ValueObjectSP non_synthetic_sp = object_sp->GetNonSyntheticValue())
    object_sp = non_synthetic_sp;
  if (ValueObjectSP dynamic_sp =
          object_sp->GetDynamicValue(lldb::eDynamicDontRunTarget))
    object_sp = dynamic_sp;
  return object_sp->GetChildMemberWithName(name);
}

std::optional<double>
lldb_private::formatters::GNUstepGetFloatValue(ValueObject &valobj) {
  llvm::Expected<llvm::APFloat> value = valobj.GetValueAsAPFloat();
  if (!value) {
    llvm::consumeError(value.takeError());
    return std::nullopt;
  }
  bool ignored = false;
  llvm::APFloat as_double(*value);
  as_double.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloat::rmNearestTiesToEven, &ignored);
  return as_double.convertToDouble();
}

// --- Small object decoding -------------------------------------------------

std::optional<std::string>
lldb_private::formatters::GNUstepDecodeTinyString(uint64_t ptr,
                                                  uint32_t pointer_size) {
  // A 32-bit pointer has one tag bit, so tag 4 cannot be encoded and there
  // are no tiny strings to find.
  if (pointer_size != 8)
    return std::nullopt;
  if ((ptr & g_gnustep_small_object_mask_64) != 4)
    return std::nullopt;
  // struct { uintptr_t char0..char7 : 7 each; length : 5; tag : 3; }: the
  // characters occupy the high bits, character i at bits [57-7i, 64-7i).
  const uint64_t length = (ptr >> 3) & 0x1f;
  // Nine means eight characters and an implicit terminator.
  if (length > 9)
    return std::nullopt;
  std::string result;
  for (uint64_t i = 0; i < length && i < 8; ++i)
    result.push_back(static_cast<char>((ptr >> (57 - 7 * i)) & 0x7f));
  return result;
}

int64_t lldb_private::formatters::GNUstepDecodeSmallInt(uint64_t ptr,
                                                        uint32_t pointer_size) {
  // The payload is shifted by the width of the tag, which is the whole
  // difference between the two data models here.
  const uint32_t shift = pointer_size == 8 ? 3 : 1;
  if (pointer_size == 4)
    ptr = static_cast<uint64_t>(
        static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(ptr))));
  return static_cast<int64_t>(ptr) >> shift;
}

double
lldb_private::formatters::GNUstepDecodeSmallExtendedDouble(uint64_t ptr) {
  // The tag displaced the low three mantissa bits, which were all equal to
  // bit 3; restore them from it.
  const uint64_t low_bit = ptr & 8;
  const uint64_t bits = (ptr & ~g_gnustep_small_object_mask_64) |
                        (low_bit >> 1) | (low_bit >> 2) | (low_bit >> 3);
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

double
lldb_private::formatters::GNUstepDecodeSmallRepeatingDouble(uint64_t ptr) {
  // Bits 3-5 hold the three mantissa bits displaced by the tag.
  const uint64_t moved = ptr & 56;
  const uint64_t bits = (ptr & ~g_gnustep_small_object_mask_64) | (moved >> 3);
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

double lldb_private::formatters::GNUstepDecodeSmallDate(uint64_t ptr) {
  // union CompressedDouble { tag:3; fraction:52; exponent:8 (signed); sign:1 }
  // with the exponent rebased on 0x3EF (Source/NSDate.m).
  const uint64_t fraction = (ptr >> 3) & ((1ULL << 52) - 1);
  const int64_t exponent = static_cast<int8_t>((ptr >> 55) & 0xff);
  const uint64_t sign = (ptr >> 63) & 1;
  const uint64_t bits =
      (sign << 63) | ((static_cast<uint64_t>(exponent + 0x3EF) & 0x7ff) << 52) |
      fraction;
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// --- Small providers -------------------------------------------------------

bool lldb_private::formatters::GNUstepNSNullSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  stream.PutCString("<null>");
  return true;
}

bool lldb_private::formatters::GNUstepNSDataSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  // NSDataStatic and its subclasses keep `NSUInteger length` (Source/NSData.m).
  ValueObjectSP length_sp = GNUstepGetIvar(valobj, "length");
  if (!length_sp)
    return false;
  bool success = false;
  const uint64_t length = length_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;
  stream.Printf("%" PRIu64 " byte%s", length, length == 1 ? "" : "s");
  return true;
}

bool lldb_private::formatters::GNUstepNSDateSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  // NSCalendarDate is registered by both runtimes into the shared "objc"
  // category, and for a given type name the last registration wins. Returning
  // false here would leave the value with no summary at all rather than
  // falling through, so hand back to the Apple provider explicitly.
  if (!IsGNUstepObjCRuntime(valobj))
    return NSDateSummaryProvider(valobj, stream, options);
  const uint64_t ptr = valobj.GetValueAsUnsigned(0);
  double seconds_since_2001 = 0.0;
  if ((ptr & g_gnustep_small_object_mask_64) == 6) {
    seconds_since_2001 = GNUstepDecodeSmallDate(ptr);
  } else {
    // NSCalendarDate (and NSGDate on targets without small objects) keep the
    // interval in _seconds_since_ref.
    ValueObjectSP seconds_sp = GNUstepGetIvar(valobj, "_seconds_since_ref");
    if (!seconds_sp)
      return false;
    std::optional<double> seconds = GNUstepGetFloatValue(*seconds_sp);
    if (!seconds)
      return false;
    seconds_since_2001 = *seconds;
  }
  // Same rendering as the Apple NSDate summary: seconds since 2001-01-01
  // converted through the Unix epoch, printed as UTC.
  constexpr time_t g_seconds_from_1970_to_2001 = 978307200;
  time_t epoch = g_seconds_from_1970_to_2001 +
                 static_cast<time_t>(std::floor(seconds_since_2001));
  tm *tm_date = gmtime(&epoch);
  if (!tm_date)
    return false;
  stream.Printf("%04d-%02d-%02d %02d:%02d:%02d UTC", tm_date->tm_year + 1900,
                tm_date->tm_mon + 1, tm_date->tm_mday, tm_date->tm_hour,
                tm_date->tm_min, tm_date->tm_sec);
  return true;
}

template <bool is_sel_ptr>
bool lldb_private::formatters::GNUstepObjCSELSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  // SEL is registered by both runtimes into the shared "objc" category, and
  // for a given type name the last registration wins. Hand back to the Apple
  // provider for any other runtime rather than leaving the value unsummarized.
  if (!IsGNUstepObjCRuntime(valobj))
    return ObjCSELSummaryProvider<is_sel_ptr>(valobj, stream, options);

  ProcessSP process_sp = valobj.GetProcessSP();
  auto *runtime = llvm::dyn_cast_or_null<GNUstepObjCRuntime>(
      ObjCLanguageRuntime::Get(*process_sp));
  if (!runtime)
    return false;

  // Apple's provider special-cases SEL* by loading through the pointer; a
  // plain SEL holds the selector's address as its value.
  lldb::addr_t sel_addr = LLDB_INVALID_ADDRESS;
  if (is_sel_ptr) {
    const lldb::addr_t ptr_addr =
        valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (ptr_addr == LLDB_INVALID_ADDRESS)
      return false;
    Status error;
    sel_addr = process_sp->ReadPointerFromMemory(ptr_addr, error);
    if (error.Fail())
      return false;
  } else {
    sel_addr = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  }
  if (sel_addr == 0 || sel_addr == LLDB_INVALID_ADDRESS)
    return false;

  ConstString name = runtime->GetSelectorName(sel_addr);
  if (!name)
    return false;
  stream.Printf("\"%s\"", name.AsCString(""));
  return true;
}

template bool lldb_private::formatters::GNUstepObjCSELSummaryProvider<true>(
    ValueObject &, Stream &, const TypeSummaryOptions &);
template bool lldb_private::formatters::GNUstepObjCSELSummaryProvider<false>(
    ValueObject &, Stream &, const TypeSummaryOptions &);

// --- Registration ----------------------------------------------------------

void lldb_private::formatters::LoadGNUstepFormatters(
    TypeCategoryImplSP objc_category_sp) {
  if (!objc_category_sp)
    return;

  TypeSummaryImpl::Flags summary_flags;
  summary_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(false)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  SyntheticChildren::Flags synth_flags;
  synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);

  // The names below are gnustep-base's concrete classes: the runtime reports
  // them for a value and ObjCLanguage offers them as formatter candidates.
  // Placeholder classes are omitted on purpose; they are what +alloc returns
  // before -init has run and carry no contents.

  // Strings (Source/GSString.m, Headers/Foundation/NSString.h).
  static constexpr const char *g_string_classes[] = {
      "GSTinyString",    "NSConstantString",   "GSString",
      "GSCString",       "GSUnicodeString",    "GSCInlineString",
      "GSUInlineString", "GSCBufferString",    "GSUnicodeBufferString",
      "GSCSubString",    "GSUnicodeSubString", "GSMutableString",
  };
  for (const char *name : g_string_classes)
    AddCXXSummary(objc_category_sp, GNUstepNSStringSummaryProvider,
                  "GNUstep NSString summary provider", name, summary_flags);

  // Numbers (Source/NSNumber.m).
  static constexpr const char *g_number_classes[] = {
      "NSSmallInt",
      "NSSmallExtendedDouble",
      "NSSmallRepeatingDouble",
      "NSSmallFloat",
      "NSIntNumber",
      "NSBoolNumber",
      "NSLongLongNumber",
      "NSUnsignedLongLongNumber",
      "NSFloatNumber",
      "NSDoubleNumber",
  };
  for (const char *name : g_number_classes)
    AddCXXSummary(objc_category_sp, GNUstepNSNumberSummaryProvider,
                  "GNUstep NSNumber summary provider", name, summary_flags);

  // Dates (Source/NSDate.m, Headers/Foundation/NSCalendarDate.h).
  for (const char *name : {"GSSmallDate", "NSGDate", "NSCalendarDate"})
    AddCXXSummary(objc_category_sp, GNUstepNSDateSummaryProvider,
                  "GNUstep NSDate summary provider", name, summary_flags);

  // Arrays (Source/GSArray.m).
  for (const char *name : {"GSArray", "GSInlineArray", "GSMutableArray"}) {
    AddCXXSummary(objc_category_sp, GNUstepNSArraySummaryProvider,
                  "GNUstep NSArray summary provider", name, summary_flags);
    AddCXXSynthetic(objc_category_sp, GNUstepNSArraySyntheticFrontEndCreator,
                    "GNUstep NSArray synthetic children", name, synth_flags);
  }

  // Dictionaries (Source/GSDictionary.m).
  for (const char *name :
       {"GSDictionary", "GSMutableDictionary", "GSCachedDictionary"}) {
    AddCXXSummary(objc_category_sp, GNUstepNSDictionarySummaryProvider,
                  "GNUstep NSDictionary summary provider", name, summary_flags);
    AddCXXSynthetic(
        objc_category_sp, GNUstepNSDictionarySyntheticFrontEndCreator,
        "GNUstep NSDictionary synthetic children", name, synth_flags);
  }

  // Sets (Source/GSSet.m, Source/GSCountedSet.m).
  for (const char *name : {"GSSet", "GSMutableSet", "GSCountedSet"}) {
    AddCXXSummary(objc_category_sp, GNUstepNSSetSummaryProvider,
                  "GNUstep NSSet summary provider", name, summary_flags);
    AddCXXSynthetic(objc_category_sp, GNUstepNSSetSyntheticFrontEndCreator,
                    "GNUstep NSSet synthetic children", name, synth_flags);
  }

  // Data (Source/NSData.m).
  for (const char *name : {"NSDataStatic", "NSDataEmpty", "NSDataMalloc",
                           "NSDataWithDeallocatorBlock", "NSMutableDataMalloc",
                           "NSMutableDataWithDeallocatorBlock"})
    AddCXXSummary(objc_category_sp, GNUstepNSDataSummaryProvider,
                  "GNUstep NSData summary provider", name, summary_flags);

  AddCXXSummary(objc_category_sp, GNUstepNSNullSummaryProvider,
                "GNUstep NSNull summary provider", "NSNull", summary_flags);

  // Not covered: NSURL. gnustep-base declares its ivars behind
  // GS_EXPOSE(NSURL), so they are in neither the debug info nor the
  // __objc_ivar_offset symbols of a normal build; `po` still describes it.

  // SEL. The Apple registrations skip pointers because SEL* is special-cased
  // when its value is retrieved; mirror that so the two runtimes stay on the
  // same flag set for the same type names.
  TypeSummaryImpl::Flags sel_flags(summary_flags);
  sel_flags.SetSkipPointers(true);
  for (const char *name : {"SEL", "struct objc_selector", "objc_selector"})
    AddCXXSummary(objc_category_sp, GNUstepObjCSELSummaryProvider<false>,
                  "GNUstep SEL summary provider", name, sel_flags);
  for (const char *name : {"objc_selector *", "SEL *"})
    AddCXXSummary(objc_category_sp, GNUstepObjCSELSummaryProvider<true>,
                  "GNUstep SEL summary provider", name, sel_flags);
}
