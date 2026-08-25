//===-- GNUstepNSNumber.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepFormatters.h"

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/Stream.h"

#include <cstdarg>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

void PrintWithHint(Stream &stream, lldb::LanguageType lang,
                   llvm::StringRef hint, const char *format, ...)
    __attribute__((format(printf, 4, 5)));

void PrintWithHint(Stream &stream, lldb::LanguageType lang,
                   llvm::StringRef hint, const char *format, ...) {
  llvm::StringRef prefix, suffix;
  if (Language *language = Language::FindPlugin(lang))
    std::tie(prefix, suffix) = language->GetFormatterPrefixSuffix(hint);
  stream << prefix;
  va_list args;
  va_start(args, format);
  stream.PrintfVarArg(format, args);
  va_end(args);
  stream << suffix;
}

} // namespace

bool lldb_private::formatters::GNUstepNSNumberSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  if (!IsGNUstepObjCRuntime(valobj))
    return false;
  const uint64_t ptr = valobj.GetValueAsUnsigned(0);
  if (ptr == 0)
    return false;
  const lldb::LanguageType lang = options.GetLanguage();

  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;
  const uint32_t pointer_size = process_sp->GetAddressByteSize();

  // Small objects: the tag selects the class and the payload is in the
  // pointer (Source/NSNumber.m). A 32-bit target has a single tag bit, so
  // NSSmallInt is the only class it can express and the wider layouts below
  // are unreachable there.
  switch (ptr & GNUstepSmallObjectMask(pointer_size)) {
  case 1: // NSSmallInt
    PrintWithHint(stream, lang, "NSNumber:long", "%" PRId64,
                  GNUstepDecodeSmallInt(ptr, pointer_size));
    return true;
  case 2: // NSSmallExtendedDouble
    PrintWithHint(stream, lang, "NSNumber:double", "%g",
                  GNUstepDecodeSmallExtendedDouble(ptr));
    return true;
  case 3: // NSSmallRepeatingDouble
    PrintWithHint(stream, lang, "NSNumber:double", "%g",
                  GNUstepDecodeSmallRepeatingDouble(ptr));
    return true;
  case 5: // NSSmallFloat: same encoding, single precision when created
    PrintWithHint(stream, lang, "NSNumber:float", "%f",
                  static_cast<float>(GNUstepDecodeSmallRepeatingDouble(ptr)));
    return true;
  case 0:
    break;
  default:
    return false;
  }

  ObjCLanguageRuntime *runtime = ObjCLanguageRuntime::Get(*process_sp);
  if (!runtime)
    return false;
  ObjCLanguageRuntime::ClassDescriptorSP descriptor =
      runtime->GetClassDescriptor(valobj);
  if (!descriptor || !descriptor->IsValid())
    return false;
  llvm::StringRef class_name = descriptor->GetClassName().GetStringRef();

  // Every heap NSNumber subclass has exactly one ivar, `value`, whose C type
  // is what the class name says (Source/NSNumber.m).
  ValueObjectSP value_sp = GNUstepGetIvar(valobj, "value");
  if (!value_sp)
    return false;

  if (class_name == "NSBoolNumber") {
    stream.PutCString(value_sp->GetValueAsUnsigned(0) ? "YES" : "NO");
    return true;
  }
  if (class_name == "NSIntNumber") {
    PrintWithHint(stream, lang, "NSNumber:int", "%d",
                  static_cast<int>(value_sp->GetValueAsSigned(0)));
    return true;
  }
  if (class_name == "NSLongLongNumber") {
    PrintWithHint(stream, lang, "NSNumber:long", "%" PRId64,
                  value_sp->GetValueAsSigned(0));
    return true;
  }
  if (class_name == "NSUnsignedLongLongNumber") {
    PrintWithHint(stream, lang, "NSNumber:long", "%" PRIu64,
                  value_sp->GetValueAsUnsigned(0));
    return true;
  }
  if (class_name == "NSFloatNumber" || class_name == "NSDoubleNumber") {
    std::optional<double> value = GNUstepGetFloatValue(*value_sp);
    if (!value)
      return false;
    if (class_name == "NSFloatNumber")
      PrintWithHint(stream, lang, "NSNumber:float", "%f",
                    static_cast<float>(*value));
    else
      PrintWithHint(stream, lang, "NSNumber:double", "%g", *value);
    return true;
  }
  return false;
}
