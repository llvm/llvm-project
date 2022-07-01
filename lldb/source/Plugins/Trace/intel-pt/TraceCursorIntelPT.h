//===-- TraceCursorIntelPT.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACECURSORINTELPT_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACECURSORINTELPT_H

#include "ThreadDecoder.h"

namespace lldb_private {
namespace trace_intel_pt {

class TraceCursorIntelPT : public TraceCursor {
public:
  TraceCursorIntelPT(lldb::ThreadSP thread_sp,
                     DecodedThreadSP decoded_thread_sp);

  bool Seek(int64_t offset, SeekType origin) override;

  void Next() override;

  bool HasValue() const override;

  const char *GetError() override;

  lldb::addr_t GetLoadAddress() override;

  llvm::Optional<uint64_t> GetCounter(lldb::TraceCounter counter_type) override;

  lldb::TraceEvents GetEvents() override;

  lldb::TraceInstructionControlFlowType
  GetInstructionControlFlowType() override;

  bool IsError() override;

  bool GoToId(lldb::user_id_t id) override;

  lldb::user_id_t GetId() const override;

  bool HasId(lldb::user_id_t id) const override;

private:
  /// \return
  ///   The number of instructions and errors in the trace.
  int64_t GetItemsCount() const;

  /// Calculate the tsc range for the current position if needed.
  void CalculateTscRange();

  /// Storage of the actual instructions
  DecodedThreadSP m_decoded_thread_sp;
  /// Internal instruction index currently pointing at.
  int64_t m_pos;
  /// Tsc range covering the current instruction.
  llvm::Optional<DecodedThread::TscRange> m_tsc_range;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACECURSORINTELPT_H
