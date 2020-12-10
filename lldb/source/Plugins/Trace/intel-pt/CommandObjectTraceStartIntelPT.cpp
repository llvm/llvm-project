//===-- CommandObjectTraceStartIntelPT.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectTraceStartIntelPT.h"

#include "lldb/Host/OptionParser.h"
#include "lldb/Target/Trace.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

#define LLDB_OPTIONS_thread_trace_start_intel_pt
#include "TraceIntelPTCommandOptions.inc"

Status CommandObjectTraceStartIntelPT::CommandOptions::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option = m_getopt_table[option_idx].val;

  switch (short_option) {
  case 's': {
    int32_t size_in_kb;
    if (option_arg.empty() || option_arg.getAsInteger(0, size_in_kb) ||
        size_in_kb < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_size_in_kb = size_in_kb;
    break;
  }
  case 'c': {
    int32_t custom_config;
    if (option_arg.empty() || option_arg.getAsInteger(0, custom_config) ||
        custom_config < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_custom_config = custom_config;
    break;
  }
  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

void CommandObjectTraceStartIntelPT::CommandOptions::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_size_in_kb = 4;
  m_custom_config = 0;
}

llvm::ArrayRef<OptionDefinition>
CommandObjectTraceStartIntelPT::CommandOptions::GetDefinitions() {
  return llvm::makeArrayRef(g_thread_trace_start_intel_pt_options);
}

bool CommandObjectTraceStartIntelPT::HandleOneThread(
    lldb::tid_t tid, CommandReturnObject &result) {
  result.AppendMessageWithFormat(
      "would trace tid %" PRIu64 " with size_in_kb %zu and custom_config %d\n",
      tid, m_options.m_size_in_kb, m_options.m_custom_config);
  result.SetStatus(eReturnStatusSuccessFinishResult);
  return result.Succeeded();
}
