//===-- LibiptDecoder.cpp --======-----------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibiptDecoder.h"

#include "TraceIntelPT.h"

#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

// Simple struct used by the decoder to keep the state of the most
// recent TSC and a flag indicating whether TSCs are enabled, not enabled
// or we just don't yet.
struct TscInfo {
  uint64_t tsc = 0;
  LazyBool has_tsc = eLazyBoolCalculate;

  explicit operator bool() const { return has_tsc == eLazyBoolYes; }
};

static inline bool IsLibiptError(int libipt_status) {
  return libipt_status < 0;
}

static inline bool IsEndOfStream(int libipt_status) {
  return libipt_status == -pte_eos;
}

static inline bool IsTscUnavailable(int libipt_status) {
  return libipt_status == -pte_no_time;
}

/// Class that decodes a raw buffer for a single thread using the low level
/// libipt library.
///
/// Throughout this code, the status of the decoder will be used to identify
/// events needed to be processed or errors in the decoder. The values can be
/// - negative: actual errors
/// - positive or zero: not an error, but a list of bits signaling the status
/// of the decoder, e.g. whether there are events that need to be decoded or
/// not.
class LibiptDecoder {
public:
  /// \param[in] decoder
  ///     A well configured decoder. Using the current state of that decoder,
  ///     decoding will start at its next valid PSB. It's not assumed that the
  ///     decoder is already pointing at a valid PSB.
  ///
  /// \param[in] decoded_thread
  ///     A \a DecodedThread object where the decoded instructions will be
  ///     appended to. It might have already some instructions.
  LibiptDecoder(pt_insn_decoder &decoder, DecodedThread &decoded_thread)
      : m_decoder(decoder), m_decoded_thread(decoded_thread) {}

  /// Decode all the instructions until the end of the trace.
  /// The decoding flow is based on
  /// https://github.com/intel/libipt/blob/master/doc/howto_libipt.md#the-instruction-flow-decode-loop
  /// but with some relaxation to allow for gaps in the trace.
  void DecodeUntilEndOfTrace() {
    int status = pte_ok;
    while (!IsLibiptError(status = FindNextSynchronizationPoint())) {
      // We have synchronized, so we can start decoding instructions and
      // events.
      // Multiple loops indicate gaps in the trace.
      DecodeInstructionsAndEvents(status);
    }
  }

private:
  /// Decode all the instructions and events until an error is found or the end
  /// of the trace is reached.
  ///
  /// \param[in] status
  ///   The status that was result of synchronizing to the most recent PSB.
  void DecodeInstructionsAndEvents(int status) {
    pt_insn insn;
    while (ProcessPTEvents(status)) {
      status = pt_insn_next(&m_decoder, &insn, sizeof(insn));
      // The status returned by pt_insn_next will need to be processed by
      // ProcessPTEvents in the next loop.
      if (FoundErrors(status, insn.ip))
        break;
      AppendInstruction(insn);
    }
  }

  /// Move the decoder forward to the next synchronization point (i.e. next PSB
  /// packet).
  ///
  /// Once the decoder is at that synchronization point, it can start decoding
  /// instructions.
  ///
  /// \return
  ///   The libipt decoder status after moving to the next PSB. Negative if
  ///   no PSB was found.
  int FindNextSynchronizationPoint() {
    // Try to sync the decoder. If it fails, then get the decoder_offset and
    // try to sync again from the next synchronization point. If the
    // new_decoder_offset is same as decoder_offset then we can't move to the
    // next synchronization point. Otherwise, keep resyncing until either end
    // of trace stream (eos) is reached or pt_insn_sync_forward() passes.
    int status = pt_insn_sync_forward(&m_decoder);

    if (!IsEndOfStream(status) && IsLibiptError(status)) {
      uint64_t decoder_offset = 0;
      int errcode_off = pt_insn_get_offset(&m_decoder, &decoder_offset);
      if (!IsLibiptError(errcode_off)) { // we could get the offset
        while (true) {
          status = pt_insn_sync_forward(&m_decoder);
          if (!IsLibiptError(status) || IsEndOfStream(status))
            break;

          uint64_t new_decoder_offset = 0;
          errcode_off = pt_insn_get_offset(&m_decoder, &new_decoder_offset);
          if (IsLibiptError(errcode_off))
            break; // We can't further synchronize.
          else if (new_decoder_offset <= decoder_offset) {
            // We tried resyncing the decoder and it didn't make any progress
            // because the offset didn't change. We will not make any further
            // progress. Hence, we stop in this situation.
            break;
          }
          // We'll try again starting from a new offset.
          decoder_offset = new_decoder_offset;
        }
      }
    }

    // We make this call to record any synchronization errors.
    FoundErrors(status);
    return status;
  }

  /// Before querying instructions, we need to query the events associated that
  /// instruction e.g. timing events like ptev_tick, or paging events like
  /// ptev_paging.
  ///
  /// \return
  ///   \b true if we could process the events, \b false if errors were found.
  bool ProcessPTEvents(int status) {
    while (status & pts_event_pending) {
      pt_event event;
      status = pt_insn_event(&m_decoder, &event, sizeof(event));
      if (IsLibiptError(status))
        break;
    }

    // We refresh the TSC that might have changed after processing the events.
    // See
    // https://github.com/intel/libipt/blob/master/doc/man/pt_evt_next.3.md
    RefreshTscInfo();
    return !FoundErrors(status);
  }

  /// Query the decoder for the most recent TSC timestamp and update
  /// the inner tsc information accordingly.
  void RefreshTscInfo() {
    if (m_tsc_info.has_tsc == eLazyBoolNo)
      return;

    uint64_t new_tsc;
    int tsc_status;
    if (IsLibiptError(tsc_status = pt_insn_time(&m_decoder, &new_tsc, nullptr,
                                                nullptr))) {
      if (IsTscUnavailable(tsc_status)) {
        // We now know that the trace doesn't support TSC, so we won't try
        // again.
        // See
        // https://github.com/intel/libipt/blob/master/doc/man/pt_qry_time.3.md
        m_tsc_info.has_tsc = eLazyBoolNo;
      } else {
        // We don't add TSC decoding errors in the decoded trace itself to
        // prevent creating unnecessary gaps, but we can count how many of
        // these errors happened. In this case we reuse the previous correct
        // TSC we saw, as it's better than no TSC at all.
        m_decoded_thread.RecordTscError(tsc_status);
      }
    } else {
      m_tsc_info.tsc = new_tsc;
      m_tsc_info.has_tsc = eLazyBoolYes;
    }
  }

  /// Check if the given libipt status signals any errors. If errors were found,
  /// they will be recorded in the decoded trace.
  ///
  /// \param[in] ip
  ///     An optional ip address can be passed if the error is associated with
  ///     the decoding of a specific instruction.
  ///
  /// \return
  ///     \b true if errors were found, \b false otherwise.
  bool FoundErrors(int status, lldb::addr_t ip = LLDB_INVALID_ADDRESS) {
    if (!IsLibiptError(status))
      return false;

    // We signal a gap only if it's not "end of stream", as that's not a proper
    // error.
    if (!IsEndOfStream(status)) {
      if (m_tsc_info) {
        m_decoded_thread.AppendError(make_error<IntelPTError>(status, ip),
                                     m_tsc_info.tsc);
      } else {
        m_decoded_thread.AppendError(make_error<IntelPTError>(status, ip));
      }
    }
    return true;
  }

  void AppendInstruction(const pt_insn &insn) {
    if (m_tsc_info)
      m_decoded_thread.AppendInstruction(insn, m_tsc_info.tsc);
    else
      m_decoded_thread.AppendInstruction(insn);
  }

private:
  pt_insn_decoder &m_decoder;
  DecodedThread &m_decoded_thread;
  TscInfo m_tsc_info;
};

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

// RAII deleter for libipt's decoder
auto DecoderDeleter = [](pt_insn_decoder *decoder) {
  pt_insn_free_decoder(decoder);
};

using PtInsnDecoderUP =
    std::unique_ptr<pt_insn_decoder, decltype(DecoderDeleter)>;

static Expected<PtInsnDecoderUP>
CreateInstructionDecoder(DecodedThread &decoded_thread,
                         TraceIntelPT &trace_intel_pt,
                         ArrayRef<uint8_t> buffer) {
  Expected<pt_cpu> cpu_info = trace_intel_pt.GetCPUInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  pt_config config;
  pt_config_init(&config);
  config.cpu = *cpu_info;
  int status = pte_ok;

  if (IsLibiptError(status = pt_cpu_errata(&config.errata, &config.cpu)))
    return make_error<IntelPTError>(status);

  // The libipt library does not modify the trace buffer, hence the
  // following casts are safe.
  config.begin = const_cast<uint8_t *>(buffer.data());
  config.end = const_cast<uint8_t *>(buffer.data() + buffer.size());

  pt_insn_decoder *decoder_ptr = pt_insn_alloc_decoder(&config);
  if (!decoder_ptr)
    return make_error<IntelPTError>(-pte_nomem);
  PtInsnDecoderUP decoder_up(decoder_ptr, DecoderDeleter);

  pt_image *image = pt_insn_get_image(decoder_ptr);
  Process *process = decoded_thread.GetThread()->GetProcess().get();

  if (IsLibiptError(
          status = pt_image_set_callback(image, ReadProcessMemory, process)))
    return make_error<IntelPTError>(status);
  return decoder_up;
}

void lldb_private::trace_intel_pt::DecodeTrace(DecodedThread &decoded_thread,
                                               TraceIntelPT &trace_intel_pt,
                                               ArrayRef<uint8_t> buffer) {
  decoded_thread.SetRawTraceSize(buffer.size());

  Expected<PtInsnDecoderUP> decoder_up =
      CreateInstructionDecoder(decoded_thread, trace_intel_pt, buffer);
  if (!decoder_up)
    return decoded_thread.AppendError(decoder_up.takeError());

  LibiptDecoder libipt_decoder(*decoder_up.get(), decoded_thread);
  libipt_decoder.DecodeUntilEndOfTrace();
}
