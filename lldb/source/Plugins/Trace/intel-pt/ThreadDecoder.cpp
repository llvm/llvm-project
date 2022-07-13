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

ThreadDecoder::ThreadDecoder(const ThreadSP &thread_sp, TraceIntelPT &trace)
    : m_thread_sp(thread_sp), m_trace(trace) {}

Expected<DecodedThreadSP> ThreadDecoder::Decode() {
  if (!m_decoded_thread.hasValue()) {
    if (Expected<DecodedThreadSP> decoded_thread = DoDecode()) {
      m_decoded_thread = *decoded_thread;
    } else {
      return decoded_thread.takeError();
    }
  }
  return *m_decoded_thread;
}

llvm::Expected<DecodedThreadSP> ThreadDecoder::DoDecode() {
  return m_trace.GetThreadTimer(m_thread_sp->GetID())
      .TimeTask(
          "Decoding instructions", [&]() -> Expected<DecodedThreadSP> {
            DecodedThreadSP decoded_thread_sp =
                std::make_shared<DecodedThread>(m_thread_sp);

            Error err = m_trace.OnThreadBufferRead(
                m_thread_sp->GetID(), [&](llvm::ArrayRef<uint8_t> data) {
                  return DecodeSingleTraceForThread(*decoded_thread_sp, m_trace,
                                                    data);
                });

            if (err)
              return std::move(err);
            return decoded_thread_sp;
          });
}
