//===-- OptionGroupDirection.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupDirection.h"

#include "lldb/Host/OptionParser.h"

using namespace lldb;
using namespace lldb_private;

static constexpr OptionDefinition g_direction_options[] = {
    {LLDB_OPT_SET_1,
     false,
     "forward",
     'F',
     OptionParser::eNoArgument,
     nullptr,
     {},
     eNoCompletion,
     eArgTypeNone,
     "Forward execute the operation."},
    {LLDB_OPT_SET_2,
     false,
     "reverse",
     'R',
     OptionParser::eNoArgument,
     nullptr,
     {},
     eNoCompletion,
     eArgTypeNone,
     "Reverse execute the operation."},
};

OptionGroupDirection::OptionGroupDirection() : m_direction(eRunForward) {}

Status
OptionGroupDirection::SetOptionValue(uint32_t option_idx,
                                     llvm::StringRef option_arg,
                                     ExecutionContext *execution_context) {
  Status error;
  char short_option = g_direction_options[option_idx].short_option;
  switch (short_option) {
  case 'F':
    m_direction = lldb::RunDirection::eRunForward;
    break;
  case 'R':
    m_direction = lldb::RunDirection::eRunReverse;
    break;
  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

void OptionGroupDirection::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_direction.reset();
}

llvm::ArrayRef<OptionDefinition> OptionGroupDirection::GetDefinitions() {
  return g_direction_options;
}
