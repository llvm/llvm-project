//===-- IntelPTDecoder.h --======--------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODER_H

#include "intel-pt.h"

#include "DecodedThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {
namespace trace_intel_pt {

/// \a lldb_private::ThreadTrace decoder that stores the output from decoding,
/// avoiding recomputations, as decoding is expensive.
class ThreadTraceDecoder {
public:
  /// \param[in] trace_thread
  ///     The thread whose trace file will be decoded.
  ///
  /// \param[in] pt_cpu
  ///     The libipt cpu used when recording the trace.
  ThreadTraceDecoder(const std::shared_ptr<ThreadTrace> &trace_thread,
                     const pt_cpu &pt_cpu)
      : m_trace_thread(trace_thread), m_pt_cpu(pt_cpu), m_decoded_thread() {}

  /// Decode the thread and store the result internally.
  ///
  /// \return
  ///     A \a DecodedThread instance.
  const DecodedThread &Decode();

private:
  ThreadTraceDecoder(const ThreadTraceDecoder &other) = delete;
  ThreadTraceDecoder &operator=(const ThreadTraceDecoder &other) = delete;

  std::shared_ptr<ThreadTrace> m_trace_thread;
  pt_cpu m_pt_cpu;
  llvm::Optional<DecodedThread> m_decoded_thread;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODER_H
