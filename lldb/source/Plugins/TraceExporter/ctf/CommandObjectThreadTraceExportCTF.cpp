//===-- CommandObjectThreadTraceExportCTF.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectThreadTraceExportCTF.h"

#include "lldb/Host/OptionParser.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::ctf;
using namespace llvm;

// CommandObjectThreadTraceExportCTF

#define LLDB_OPTIONS_thread_trace_export_ctf
#include "TraceExporterCTFCommandOptions.inc"

Status CommandObjectThreadTraceExportCTF::CommandOptions::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option = m_getopt_table[option_idx].val;

  switch (short_option) {
  case 't': {
    int64_t thread_index;
    if (option_arg.empty() || option_arg.getAsInteger(0, thread_index) ||
        thread_index < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_thread_index = thread_index;
    break;
  }
  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

void CommandObjectThreadTraceExportCTF::CommandOptions::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_thread_index = None;
}

llvm::ArrayRef<OptionDefinition>
CommandObjectThreadTraceExportCTF::CommandOptions::GetDefinitions() {
  return llvm::makeArrayRef(g_thread_trace_export_ctf_options);
}

bool CommandObjectThreadTraceExportCTF::DoExecute(Args &command,
                                                  CommandReturnObject &result) {
  Stream &s = result.GetOutputStream();
  // TODO: create an actual instance of the exporter and invoke it
  if (m_options.m_thread_index)
    s.Printf("got thread index %d\n", (int)m_options.m_thread_index.getValue());
  else
    s.Printf("didn't get a thread index\n");

  return result.Succeeded();
}
