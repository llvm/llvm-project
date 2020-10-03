//===-- ThreadIntelPT.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_THREADINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_THREADINTELPT_H

#include "lldb/Target/Thread.h"

namespace lldb_private {
namespace trace_intel_pt {

class ThreadIntelPT : public Thread {
public:
  /// Create an Intel PT-traced thread.
  ///
  /// \param[in] process
  ///     The process that owns this thread.
  ///
  /// \param[in] tid
  ///     The thread id of this thread.
  ///
  /// \param[in] trace_file
  ///     The trace file for this thread.
  ///
  /// \param[in] pt_cpu
  ///     The Intel CPU information required to decode the \a trace_file.
  ThreadIntelPT(Process &process, lldb::tid_t tid, const FileSpec &trace_file)
      : Thread(process, tid), m_trace_file(trace_file) {}

  void RefreshStateAfterStop() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

protected:
  bool CalculateStopInfo() override;

  lldb::RegisterContextSP m_thread_reg_ctx_sp;

private:
  FileSpec m_trace_file;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_THREADINTELPT_H
