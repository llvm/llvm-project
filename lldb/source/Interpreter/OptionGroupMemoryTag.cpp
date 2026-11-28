//===-- OptionGroupMemoryTag.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupMemoryTag.h"

#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_memory_tag
#include "InterpreterOptions.inc"

#define LLDB_OPTIONS_memory_tag_binary
#include "InterpreterOptions.inc"

OptionGroupMemoryTag::OptionGroupMemoryTag(bool note_binary /*=false*/)
    : m_show_tags(false, false),
      m_option_definition(note_binary ? g_memory_tag_binary_options[0]
                                      : g_memory_tag_options[0]) {}

llvm::ArrayRef<OptionDefinition> OptionGroupMemoryTag::GetDefinitions() {
  return llvm::ArrayRef(m_option_definition);
}

Status
OptionGroupMemoryTag::SetOptionValue(uint32_t option_idx,
                                     llvm::StringRef option_arg,
                                     ExecutionContext *execution_context) {
  assert(option_idx == 0 && "Only one option in memory tag group!");

  switch (m_option_definition.short_option) {
  case '\x02': // --show-tags
    m_show_tags.SetCurrentValue(true);
    m_show_tags.SetOptionWasSet();
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }

  return {};
}

void OptionGroupMemoryTag::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_show_tags.Clear();
}
