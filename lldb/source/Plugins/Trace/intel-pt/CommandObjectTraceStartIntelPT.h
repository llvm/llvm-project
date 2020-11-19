//===-- CommandObjectTraceStartIntelPT.h ----------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_COMMANDOBJECTTRACESTARTINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_COMMANDOBJECTTRACESTARTINTELPT_H

#include "../../../../source/Commands/CommandObjectThreadUtil.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

namespace lldb_private {
namespace trace_intel_pt {

class CommandObjectTraceStartIntelPT : public CommandObjectIterateOverThreads {
public:
  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() { OptionParsingStarting(nullptr); }

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override;

    void OptionParsingStarting(ExecutionContext *execution_context) override;

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

    size_t m_size_in_kb;
    uint32_t m_custom_config;
  };

  CommandObjectTraceStartIntelPT(CommandInterpreter &interpreter)
      : CommandObjectIterateOverThreads(
            interpreter, "thread trace start",
            "Start tracing one or more threads with intel-pt. "
            "Defaults to the current thread. Thread indices can be "
            "specified as arguments.\n Use the thread-index \"all\" to trace "
            "all threads.",
            "thread trace start [<thread-index> <thread-index> ...] "
            "[<intel-pt-options>]",
            lldb::eCommandRequiresProcess | lldb::eCommandTryTargetAPILock |
                lldb::eCommandProcessMustBeLaunched |
                lldb::eCommandProcessMustBePaused),
        m_options() {}

  ~CommandObjectTraceStartIntelPT() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  bool HandleOneThread(lldb::tid_t tid, CommandReturnObject &result) override;

  CommandOptions m_options;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_COMMANDOBJECTTRACESTARTINTELPT_H
