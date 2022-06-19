//===-- ThreadDecoder.h --======---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_THREAD_DECODER_H
#define LLDB_SOURCE_PLUGINS_TRACE_THREAD_DECODER_H

#include "intel-pt.h"

#include "DecodedThread.h"
#include "forward-declarations.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {
namespace trace_intel_pt {

/// Class that handles the decoding of a thread and caches the result.
class ThreadDecoder {
public:
  /// \param[in] thread_sp
  ///     The thread whose intel pt trace buffer will be decoded.
  ///
  /// \param[in] trace
  ///     The main Trace object who owns this decoder and its data.
  ThreadDecoder(const lldb::ThreadSP &thread_sp, TraceIntelPT &trace);

  /// Decode the thread and store the result internally, to avoid
  /// recomputations.
  ///
  /// \return
  ///     A \a DecodedThread instance.
  DecodedThreadSP Decode();

  ThreadDecoder(const ThreadDecoder &other) = delete;
  ThreadDecoder &operator=(const ThreadDecoder &other) = delete;

private:
  DecodedThreadSP DoDecode();

  lldb::ThreadSP m_thread_sp;
  TraceIntelPT &m_trace;
  llvm::Optional<DecodedThreadSP> m_decoded_thread;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_THREAD_DECODER_H
