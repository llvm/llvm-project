//===-- OptionGroupOutputFile.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupOutputFile.h"

#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_output_file
#include "InterpreterOptions.inc"

OptionGroupOutputFile::OptionGroupOutputFile() : m_append(false, false) {}

llvm::ArrayRef<OptionDefinition> OptionGroupOutputFile::GetDefinitions() {
  return llvm::ArrayRef(g_output_file_options);
}

Status
OptionGroupOutputFile::SetOptionValue(uint32_t option_idx,
                                      llvm::StringRef option_arg,
                                      ExecutionContext *execution_context) {
  Status error;
  const int short_option = g_output_file_options[option_idx].short_option;

  switch (short_option) {
  case 'o':
    error = m_file.SetValueFromString(option_arg);
    break;

  case '\x01': // --append-outfile
    m_append.SetCurrentValue(true);
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void OptionGroupOutputFile::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_file.Clear();
  m_append.Clear();
}
