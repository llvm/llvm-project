//===-- CommandObjectTraceStartIntelPT.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectTraceStartIntelPT.h"

#include "TraceIntelPT.h"
#include "TraceIntelPTConstants.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Trace.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

// CommandObjectThreadTraceStartIntelPT

#define LLDB_OPTIONS_thread_trace_start_intel_pt
#include "TraceIntelPTCommandOptions.inc"

Status CommandObjectThreadTraceStartIntelPT::CommandOptions::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option = m_getopt_table[option_idx].val;

  switch (short_option) {
  case 's': {
    int64_t ipt_trace_size;
    if (option_arg.empty() || option_arg.getAsInteger(0, ipt_trace_size) ||
        ipt_trace_size < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_ipt_trace_size = ipt_trace_size;
    break;
  }
  case 't': {
    m_enable_tsc = true;
    break;
  }
  case 'p': {
    int64_t psb_period;
    if (option_arg.empty() || option_arg.getAsInteger(0, psb_period) ||
        psb_period < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_psb_period = psb_period;
    break;
  }
  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

void CommandObjectThreadTraceStartIntelPT::CommandOptions::
    OptionParsingStarting(ExecutionContext *execution_context) {
  m_ipt_trace_size = kDefaultIptTraceSize;
  m_enable_tsc = kDefaultEnableTscValue;
  m_psb_period = kDefaultPsbPeriod;
}

llvm::ArrayRef<OptionDefinition>
CommandObjectThreadTraceStartIntelPT::CommandOptions::GetDefinitions() {
  return llvm::makeArrayRef(g_thread_trace_start_intel_pt_options);
}

bool CommandObjectThreadTraceStartIntelPT::DoExecuteOnThreads(
    Args &command, CommandReturnObject &result,
    llvm::ArrayRef<lldb::tid_t> tids) {
  if (Error err = m_trace.Start(tids, m_options.m_ipt_trace_size,
                                m_options.m_enable_tsc, m_options.m_psb_period))
    result.SetError(Status(std::move(err)));
  else
    result.SetStatus(eReturnStatusSuccessFinishResult);

  return result.Succeeded();
}

/// CommandObjectProcessTraceStartIntelPT

#define LLDB_OPTIONS_process_trace_start_intel_pt
#include "TraceIntelPTCommandOptions.inc"

Status CommandObjectProcessTraceStartIntelPT::CommandOptions::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option = m_getopt_table[option_idx].val;

  switch (short_option) {
  case 's': {
    int64_t ipt_trace_size;
    if (option_arg.empty() || option_arg.getAsInteger(0, ipt_trace_size) ||
        ipt_trace_size < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_ipt_trace_size = ipt_trace_size;
    break;
  }
  case 'l': {
    int64_t process_buffer_size_limit;
    if (option_arg.empty() ||
        option_arg.getAsInteger(0, process_buffer_size_limit) ||
        process_buffer_size_limit < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_process_buffer_size_limit = process_buffer_size_limit;
    break;
  }
  case 't': {
    m_enable_tsc = true;
    break;
  }
  case 'c': {
    m_per_cpu_tracing = true;
    break;
  }
  case 'p': {
    int64_t psb_period;
    if (option_arg.empty() || option_arg.getAsInteger(0, psb_period) ||
        psb_period < 0)
      error.SetErrorStringWithFormat("invalid integer value for option '%s'",
                                     option_arg.str().c_str());
    else
      m_psb_period = psb_period;
    break;
  }
  default:
    llvm_unreachable("Unimplemented option");
  }
  return error;
}

void CommandObjectProcessTraceStartIntelPT::CommandOptions::
    OptionParsingStarting(ExecutionContext *execution_context) {
  m_ipt_trace_size = kDefaultIptTraceSize;
  m_process_buffer_size_limit = kDefaultProcessBufferSizeLimit;
  m_enable_tsc = kDefaultEnableTscValue;
  m_psb_period = kDefaultPsbPeriod;
  m_per_cpu_tracing = kDefaultPerCpuTracing;
}

llvm::ArrayRef<OptionDefinition>
CommandObjectProcessTraceStartIntelPT::CommandOptions::GetDefinitions() {
  return llvm::makeArrayRef(g_process_trace_start_intel_pt_options);
}

bool CommandObjectProcessTraceStartIntelPT::DoExecute(
    Args &command, CommandReturnObject &result) {
  if (Error err = m_trace.Start(m_options.m_ipt_trace_size,
                                m_options.m_process_buffer_size_limit,
                                m_options.m_enable_tsc, m_options.m_psb_period,
                                m_options.m_per_cpu_tracing))
    result.SetError(Status(std::move(err)));
  else
    result.SetStatus(eReturnStatusSuccessFinishResult);

  return result.Succeeded();
}
