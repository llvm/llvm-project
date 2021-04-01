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
#include "forward-declarations.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {
namespace trace_intel_pt {

/// Base class that handles the decoding of a thread and caches the result.
class ThreadDecoder {
public:
  virtual ~ThreadDecoder() = default;

  ThreadDecoder() = default;

  /// Decode the thread and store the result internally, to avoid
  /// recomputations.
  ///
  /// \return
  ///     A \a DecodedThread instance.
  const DecodedThread &Decode();

  ThreadDecoder(const ThreadDecoder &other) = delete;
  ThreadDecoder &operator=(const ThreadDecoder &other) = delete;

protected:
  /// Decode the thread.
  ///
  /// \return
  ///     A \a DecodedThread instance.
  virtual DecodedThread DoDecode() = 0;

  llvm::Optional<DecodedThread> m_decoded_thread;
};

/// Decoder implementation for \a lldb_private::ThreadPostMortemTrace, which are
/// non-live processes that come trace session files.
class PostMortemThreadDecoder : public ThreadDecoder {
public:
  /// \param[in] trace_thread
  ///     The thread whose trace file will be decoded.
  ///
  /// \param[in] trace
  ///     The main Trace object who owns this decoder and its data.
  PostMortemThreadDecoder(const lldb::ThreadPostMortemTraceSP &trace_thread,
                          TraceIntelPT &trace);

private:
  DecodedThread DoDecode() override;

  lldb::ThreadPostMortemTraceSP m_trace_thread;
  TraceIntelPT &m_trace;
};

class LiveThreadDecoder : public ThreadDecoder {
public:
  /// \param[in] thread
  ///     The thread whose traces will be decoded.
  ///
  /// \param[in] trace
  ///     The main Trace object who owns this decoder and its data.
  LiveThreadDecoder(Thread &thread, TraceIntelPT &trace);

private:
  DecodedThread DoDecode() override;

  lldb::ThreadSP m_thread_sp;
  TraceIntelPT &m_trace;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODER_H
