//===-- GoFormatterFunctions.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GoFormatterFunctions.h"

#include "lldb/Core/Address.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

ValueObject *GetGoStringValue(ValueObject &valobj, ValueObjectSP &storage) {
  if (!valobj.GetCompilerType().IsPointerType())
    return &valobj;

  Status error;
  storage = valobj.Dereference(error);
  if (error.Fail())
    return nullptr;
  return storage.get();
}

} // namespace

bool lldb_private::formatters::IsGoString(ValueObject &valobj) {
  ValueObjectSP storage;
  ValueObject *value = GetGoStringValue(valobj, storage);
  if (!value || value->GetCompilerType().GetTypeName() != "string")
    return false;

  return value->GetChildMemberWithName(ConstString("str"), true) &&
         value->GetChildMemberWithName(ConstString("len"), true);
}

bool lldb_private::formatters::GoStringSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  ValueObjectSP storage;
  ValueObject *value = GetGoStringValue(valobj, storage);
  if (!value)
    return false;

  ValueObjectSP data_sp =
      value->GetChildMemberWithName(ConstString("str"), true);
  ValueObjectSP len_sp =
      value->GetChildMemberWithName(ConstString("len"), true);
  if (!data_sp || !len_sp)
    return false;

  bool success = false;
  const lldb::addr_t address = data_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;

  const uint64_t length = len_sp->GetValueAsUnsigned(0, &success);
  if (!success)
    return false;
  if (length == 0) {
    stream.PutCString("\"\"");
    return true;
  }

  StringPrinter::ReadStringAndDumpToStreamOptions options(valobj);
  options.SetLocation(Address(address));
  options.SetTargetSP(valobj.GetTargetSP());
  options.SetStream(&stream);
  options.SetSourceSize(length);
  options.SetHasSourceSize(true);
  options.SetZeroTermination(StringPrinter::ZeroTermination::Ignore);

  if (!StringPrinter::ReadStringAndDumpToStream<
          StringPrinter::StringElementType::UTF8>(options))
    stream.PutCString("Summary Unavailable");
  return true;
}
