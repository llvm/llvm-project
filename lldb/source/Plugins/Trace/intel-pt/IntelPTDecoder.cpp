//===-- IntelPTDecoder.cpp --======----------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTDecoder.h"

#include "llvm/Support/MemoryBuffer.h"

#include "../common/ThreadPostMortemTrace.h"
#include "DecodedThread.h"
#include "TraceIntelPT.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StringExtractor.h"
#include <utility>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

/// Move the decoder forward to the next synchronization point (i.e. next PSB
/// packet).
///
/// Once the decoder is at that sync. point, it can start decoding instructions.
///
/// \return
///   A negative number with the libipt error if we couldn't synchronize.
///   Otherwise, a positive number with the synchronization status will be
///   returned.
static int FindNextSynchronizationPoint(pt_insn_decoder &decoder) {
  // Try to sync the decoder. If it fails, then get
  // the decoder_offset and try to sync again from
  // the next synchronization point. If the
  // new_decoder_offset is same as decoder_offset
  // then we can't move to the next synchronization
  // point. Otherwise, keep resyncing until either
  // end of trace stream (eos) is reached or
  // pt_insn_sync_forward() passes.
  int errcode = pt_insn_sync_forward(&decoder);

  if (errcode != -pte_eos && errcode < 0) {
    uint64_t decoder_offset = 0;
    int errcode_off = pt_insn_get_offset(&decoder, &decoder_offset);
    if (errcode_off >= 0) { // we could get the offset
      while (true) {
        errcode = pt_insn_sync_forward(&decoder);
        if (errcode >= 0 || errcode == -pte_eos)
          break;

        uint64_t new_decoder_offset = 0;
        errcode_off = pt_insn_get_offset(&decoder, &new_decoder_offset);
        if (errcode_off < 0)
          break; // We can't further synchronize.
        else if (new_decoder_offset <= decoder_offset) {
          // We tried resyncing the decoder and
          // decoder didn't make any progress because
          // the offset didn't change. We will not
          // make any progress further. Hence,
          // stopping in this situation.
          break;
        }
        // We'll try again starting from a new offset.
        decoder_offset = new_decoder_offset;
      }
    }
  }

  return errcode;
}

/// Before querying instructions, we need to query the events associated that
/// instruction e.g. timing events like ptev_tick, or paging events like
/// ptev_paging.
///
/// \return
///   0 if there were no errors processing the events, or a negative libipt
///   error code in case of errors.
static int ProcessPTEvents(pt_insn_decoder &decoder, int errcode) {
  while (errcode & pts_event_pending) {
    pt_event event;
    errcode = pt_insn_event(&decoder, &event, sizeof(event));
    if (errcode < 0)
      return errcode;
  }
  return 0;
}

/// Decode all the instructions from a configured decoder.
/// The decoding flow is based on
/// https://github.com/intel/libipt/blob/master/doc/howto_libipt.md#the-instruction-flow-decode-loop
/// but with some relaxation to allow for gaps in the trace.
///
/// Error codes returned by libipt while decoding are:
/// - negative: actual errors
/// - positive or zero: not an error, but a list of bits signaling the status of
/// the decoder
///
/// \param[in] decoder
///   A configured libipt \a pt_insn_decoder.
static void DecodeInstructions(pt_insn_decoder &decoder,
                               DecodedThreadSP &decoded_thread_sp) {
  while (true) {
    int errcode = FindNextSynchronizationPoint(decoder);
    if (errcode == -pte_eos)
      break;

    if (errcode < 0) {
      decoded_thread_sp->AppendError(make_error<IntelPTError>(errcode));
      break;
    }

    // We have synchronized, so we can start decoding
    // instructions and events.
    while (true) {
      errcode = ProcessPTEvents(decoder, errcode);
      if (errcode < 0) {
        decoded_thread_sp->AppendError(make_error<IntelPTError>(errcode));
        break;
      }

      pt_insn insn;
      errcode = pt_insn_next(&decoder, &insn, sizeof(insn));
      if (errcode == -pte_eos)
        break;

      if (errcode < 0) {
        decoded_thread_sp->AppendError(
            make_error<IntelPTError>(errcode, insn.ip));
        break;
      }

      uint64_t time;
      int time_error = pt_insn_time(&decoder, &time, nullptr, nullptr);
      if (time_error == -pte_invalid) {
        // This happens if we invoke the pt_insn_time method incorrectly,
        // but the instruction is good though.
        decoded_thread_sp->AppendError(
            make_error<IntelPTError>(time_error, insn.ip));
        decoded_thread_sp->AppendInstruction(insn);
        break;
      }

      if (time_error == -pte_no_time) {
        // We simply don't have time information, i.e. None of TSC, MTC or CYC
        // was enabled.
        decoded_thread_sp->AppendInstruction(insn);
      } else {
        decoded_thread_sp->AppendInstruction(insn, time);
      }
    }
  }
}

/// Callback used by libipt for reading the process memory.
///
/// More information can be found in
/// https://github.com/intel/libipt/blob/master/doc/man/pt_image_set_callback.3.md
static int ReadProcessMemory(uint8_t *buffer, size_t size,
                             const pt_asid * /* unused */, uint64_t pc,
                             void *context) {
  Process *process = static_cast<Process *>(context);

  Status error;
  int bytes_read = process->ReadMemory(pc, buffer, size, error);
  if (error.Fail())
    return -pte_nomap;
  return bytes_read;
}

static void DecodeInMemoryTrace(DecodedThreadSP &decoded_thread_sp,
                                TraceIntelPT &trace_intel_pt,
                                MutableArrayRef<uint8_t> buffer) {
  Expected<pt_cpu> cpu_info = trace_intel_pt.GetCPUInfo();
  if (!cpu_info) {
    return decoded_thread_sp->AppendError(cpu_info.takeError());
  }

  pt_config config;
  pt_config_init(&config);
  config.cpu = *cpu_info;

  if (int errcode = pt_cpu_errata(&config.errata, &config.cpu))
    return decoded_thread_sp->AppendError(make_error<IntelPTError>(errcode));

  config.begin = buffer.data();
  config.end = buffer.data() + buffer.size();

  pt_insn_decoder *decoder = pt_insn_alloc_decoder(&config);
  if (!decoder)
    return decoded_thread_sp->AppendError(make_error<IntelPTError>(-pte_nomem));

  pt_image *image = pt_insn_get_image(decoder);

  int errcode =
      pt_image_set_callback(image, ReadProcessMemory,
                            decoded_thread_sp->GetThread()->GetProcess().get());
  assert(errcode == 0);
  (void)errcode;

  DecodeInstructions(*decoder, decoded_thread_sp);
  pt_insn_free_decoder(decoder);
}
// ---------------------------

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
  DecodeInMemoryTrace(decoded_thread_sp, m_trace,
                      MutableArrayRef<uint8_t>(*buffer));
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

  DecodeInMemoryTrace(decoded_thread_sp, m_trace, trace_data);
  return decoded_thread_sp;
}
