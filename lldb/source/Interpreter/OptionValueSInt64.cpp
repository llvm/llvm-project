//===-- OptionValueSInt64.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueSInt64.h"

#include "lldb/Host/StringConvert.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

void OptionValueSInt64::DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                                  uint32_t dump_mask) {
  // printf ("%p: DumpValue (exe_ctx=%p, strm, mask) m_current_value = %"
  // PRIi64
  // "\n", this, exe_ctx, m_current_value);
  if (dump_mask & eDumpOptionType)
    strm.Printf("(%s)", GetTypeAsCString());
  //    if (dump_mask & eDumpOptionName)
  //        DumpQualifiedName (strm);
  if (dump_mask & eDumpOptionValue) {
    if (dump_mask & eDumpOptionType)
      strm.PutCString(" = ");
    strm.Printf("%" PRIi64, m_current_value);
  }
}

Status OptionValueSInt64::SetValueFromString(llvm::StringRef value_ref,
                                             VarSetOperationType op) {
  Status error;
  switch (op) {
  case eVarSetOperationClear:
    Clear();
    NotifyValueChanged();
    break;

  case eVarSetOperationReplace:
  case eVarSetOperationAssign: {
    bool success = false;
    std::string value_str = value_ref.trim().str();
    int64_t value = StringConvert::ToSInt64(value_str.c_str(), 0, 0, &success);
    if (success) {
      if (value >= m_min_value && value <= m_max_value) {
        m_value_was_set = true;
        m_current_value = value;
        NotifyValueChanged();
      } else
        error.SetErrorStringWithFormat(
            "%" PRIi64 " is out of range, valid values must be between %" PRIi64
            " and %" PRIi64 ".",
            value, m_min_value, m_max_value);
    } else {
      error.SetErrorStringWithFormat("invalid int64_t string value: '%s'",
                                     value_ref.str().c_str());
    }
  } break;

  case eVarSetOperationInsertBefore:
  case eVarSetOperationInsertAfter:
  case eVarSetOperationRemove:
  case eVarSetOperationAppend:
  case eVarSetOperationInvalid:
    error = OptionValue::SetValueFromString(value_ref, op);
    break;
  }
  return error;
}
