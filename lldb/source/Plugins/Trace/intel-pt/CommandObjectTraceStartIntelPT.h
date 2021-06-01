//===-- CommandObjectTraceStartIntelPT.h ----------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_COMMANDOBJECTTRACESTARTINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_COMMANDOBJECTTRACESTARTINTELPT_H

#include "../../../../source/Commands/CommandObjectTrace.h"
#include "TraceIntelPT.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

namespace lldb_private {
namespace trace_intel_pt {

class CommandObjectThreadTraceStartIntelPT
    : public CommandObjectMultipleThreads {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() { OptionParsingStarting(nullptr); }

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override;

    void OptionParsingStarting(ExecutionContext *execution_context) override;

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

    size_t m_thread_buffer_size;
  };

  CommandObjectThreadTraceStartIntelPT(TraceIntelPT &trace,
                                       CommandInterpreter &interpreter)
      : CommandObjectMultipleThreads(
            interpreter, "thread trace start",
            "Start tracing one or more threads with intel-pt. "
            "Defaults to the current thread. Thread indices can be "
            "specified as arguments.\n Use the thread-index \"all\" to trace "
            "all threads including future threads.",
            "thread trace start [<thread-index> <thread-index> ...] "
            "[<intel-pt-options>]",
            lldb::eCommandRequiresProcess | lldb::eCommandTryTargetAPILock |
                lldb::eCommandProcessMustBeLaunched |
                lldb::eCommandProcessMustBePaused),
        m_trace(trace), m_options() {}

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecuteOnThreads(Args &command, CommandReturnObject &result,
                          llvm::ArrayRef<lldb::tid_t> tids) override;

  TraceIntelPT &m_trace;
  CommandOptions m_options;
};

class CommandObjectProcessTraceStartIntelPT : public CommandObjectParsed {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() { OptionParsingStarting(nullptr); }

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override;

    void OptionParsingStarting(ExecutionContext *execution_context) override;

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

    size_t m_thread_buffer_size;
    size_t m_process_buffer_size_limit;
  };

  CommandObjectProcessTraceStartIntelPT(TraceIntelPT &trace,
                                        CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "process trace start",
            "Start tracing this process with intel-pt, including future "
            "threads. "
            "This is implemented by tracing each thread independently. "
            "Threads traced with the \"thread trace start\" command are left "
            "unaffected ant not retraced.",
            "process trace start [<intel-pt-options>]",
            lldb::eCommandRequiresProcess | lldb::eCommandTryTargetAPILock |
                lldb::eCommandProcessMustBeLaunched |
                lldb::eCommandProcessMustBePaused),
        m_trace(trace), m_options() {}

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override;

  TraceIntelPT &m_trace;
  CommandOptions m_options;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_COMMANDOBJECTTRACESTARTINTELPT_H
