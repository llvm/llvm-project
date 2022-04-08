//===-- ThreadDecoder.cpp --======-----------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadDecoder.h"

#include "llvm/Support/MemoryBuffer.h"

#include "../common/ThreadPostMortemTrace.h"
#include "LibiptDecoder.h"
#include "TraceIntelPT.h"

#include <utility>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

// ThreadDecoder ====================

DecodedThreadSP ThreadDecoder::Decode() {
  if (!m_decoded_thread.hasValue())
    m_decoded_thread = DoDecode();
  return *m_decoded_thread;
}

// LiveThreadDecoder ====================

LiveThreadDecoder::LiveThreadDecoder(Thread &thread, TraceIntelPT &trace)
    : m_thread_sp(thread.shared_from_this()), m_trace(trace) {}

DecodedThreadSP LiveThreadDecoder::DoDecode() {
  DecodedThreadSP decoded_thread_sp =
      std::make_shared<DecodedThread>(m_thread_sp);

  Expected<std::vector<uint8_t>> buffer =
      m_trace.GetLiveThreadBuffer(m_thread_sp->GetID());
  if (!buffer) {
    decoded_thread_sp->AppendError(buffer.takeError());
    return decoded_thread_sp;
  }

  decoded_thread_sp->SetRawTraceSize(buffer->size());
  DecodeTrace(*decoded_thread_sp, m_trace, MutableArrayRef<uint8_t>(*buffer));
  return decoded_thread_sp;
}

// PostMortemThreadDecoder =======================

PostMortemThreadDecoder::PostMortemThreadDecoder(
    const lldb::ThreadPostMortemTraceSP &trace_thread, TraceIntelPT &trace)
    : m_trace_thread(trace_thread), m_trace(trace) {}

DecodedThreadSP PostMortemThreadDecoder::DoDecode() {
  DecodedThreadSP decoded_thread_sp =
      std::make_shared<DecodedThread>(m_trace_thread);

  ErrorOr<std::unique_ptr<MemoryBuffer>> trace_or_error =
      MemoryBuffer::getFile(m_trace_thread->GetTraceFile().GetPath());
  if (std::error_code err = trace_or_error.getError()) {
    decoded_thread_sp->AppendError(errorCodeToError(err));
    return decoded_thread_sp;
  }

  MemoryBuffer &trace = **trace_or_error;
  MutableArrayRef<uint8_t> trace_data(
      // The libipt library does not modify the trace buffer, hence the
      // following cast is safe.
      reinterpret_cast<uint8_t *>(const_cast<char *>(trace.getBufferStart())),
      trace.getBufferSize());
  decoded_thread_sp->SetRawTraceSize(trace_data.size());

  DecodeTrace(*decoded_thread_sp, m_trace, trace_data);
  return decoded_thread_sp;
}
