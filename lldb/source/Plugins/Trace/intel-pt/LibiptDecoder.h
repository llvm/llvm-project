//===-- LibiptDecoder.h --======---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_LIBIPT_DECODER_H
#define LLDB_SOURCE_PLUGINS_TRACE_LIBIPT_DECODER_H

#include "DecodedThread.h"
#include "PerfContextSwitchDecoder.h"
#include "forward-declarations.h"

#include "intel-pt.h"

namespace lldb_private {
namespace trace_intel_pt {

struct IntelPTThreadSubtrace {
  uint64_t tsc;
  uint64_t psb_offset;
};

/// This struct represents a continuous execution of a thread in a core,
/// delimited by a context switch in and out, and a list of Intel PT subtraces
/// that belong to this execution.
struct IntelPTThreadContinousExecution {
  ThreadContinuousExecution thread_execution;
  std::vector<IntelPTThreadSubtrace> intelpt_subtraces;

  IntelPTThreadContinousExecution(
      const ThreadContinuousExecution &thread_execution)
      : thread_execution(thread_execution) {}

  /// Comparator by time
  bool operator<(const IntelPTThreadContinousExecution &o) const;
};

/// Decode a raw Intel PT trace given in \p buffer and append the decoded
/// instructions and errors in \p decoded_thread. It uses the low level libipt
/// library underneath.
void DecodeTrace(DecodedThread &decoded_thread, TraceIntelPT &trace_intel_pt,
                 llvm::ArrayRef<uint8_t> buffer);

void DecodeTrace(
    DecodedThread &decoded_thread, TraceIntelPT &trace_intel_pt,
    const llvm::DenseMap<lldb::core_id_t, llvm::ArrayRef<uint8_t>> &buffers,
    const std::vector<IntelPTThreadContinousExecution> &executions);

llvm::Expected<std::vector<IntelPTThreadSubtrace>>
SplitTraceInContinuousExecutions(TraceIntelPT &trace_intel_pt,
                                 llvm::ArrayRef<uint8_t> buffer);

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_LIBIPT_DECODER_H
