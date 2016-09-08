//===-- JavaFormatterFunctions.cpp-------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "JavaFormatterFunctions.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool lldb_private::formatters::JavaStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &opts) {
  if (valobj.IsPointerOrReferenceType()) {
    Error error;
    ValueObjectSP deref = valobj.Dereference(error);
    if (error.Fail())
      return false;
    return JavaStringSummaryProvider(*deref, stream, opts);
  }

  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;

  ConstString data_name("value");
  ConstString length_name("count");

  ValueObjectSP length_sp = valobj.GetChildMemberWithName(length_name, true);
  ValueObjectSP data_sp = valobj.GetChildMemberWithName(data_name, true);
  if (!data_sp || !length_sp)
    return false;

  bool success = false;
  uint64_t length = length_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;

  if (length == 0) {
    stream.Printf("\"\"");
    return true;
  }
  lldb::addr_t valobj_addr = data_sp->GetAddressOf();

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(valobj_addr);
  options.SetProcessSP(process_sp);
  options.SetStream(&stream);
  options.SetSourceSize(length);
  options.SetNeedsZeroTermination(false);
  options.SetLanguage(eLanguageTypeJava);

  if (StringPrinter::ReadStringAndDumpToStream<
          StringPrinter::StringElementType::UTF16>(options))
    return true;

  stream.Printf("Summary Unavailable");
  return true;
}
