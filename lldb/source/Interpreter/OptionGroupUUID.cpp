//===-- OptionGroupUUID.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupUUID.h"

#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_uuid
#include "InterpreterOptions.inc"

llvm::ArrayRef<OptionDefinition> OptionGroupUUID::GetDefinitions() {
  return llvm::ArrayRef(g_uuid_options);
}

Status OptionGroupUUID::SetOptionValue(uint32_t option_idx,
                                       llvm::StringRef option_arg,
                                       ExecutionContext *execution_context) {
  Status error;
  const int short_option = g_uuid_options[option_idx].short_option;

  switch (short_option) {
  case 'u':
    error = m_uuid.SetValueFromString(option_arg);
    if (error.Success())
      m_uuid.SetOptionWasSet();
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void OptionGroupUUID::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_uuid.Clear();
}
