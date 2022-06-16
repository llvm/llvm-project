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
  /// https://github.com/intel/libipt/blob/master/doc/howto_libipt.md#the-instruction-flow-decode-loop.
  void DecodeUntilEndOfTrace() {
    // Multiple loops indicate gaps in the trace, which are found by the inner
    // call to DecodeInstructionsAndEvents.
    while (true) {
      int status = pt_insn_sync_forward(&m_decoder);

      if (IsLibiptError(status)) {
        m_decoded_thread.Append(DecodedInstruction(status));
        break;
      }

      DecodeInstructionsAndEvents(status);
    }
  }

  /// Decode all the instructions that belong to the same PSB packet given its
  /// offset.
  void DecodePSB(uint64_t psb_offset) {
    int status = pt_insn_sync_set(&m_decoder, psb_offset);
    if (IsLibiptError(status)) {
      m_decoded_thread.Append(DecodedInstruction(status));
      return;
    }
    DecodeInstructionsAndEvents(status, /*stop_on_psb_change=*/true);
  }

private:
  /// Invoke the low level function \a pt_insn_next and store the decoded
  /// instruction in the given \a DecodedInstruction.
  ///
  /// \param[out] insn
  ///   The instruction builder where the pt_insn information will be stored.
  ///
  /// \return
  ///   The status returned by pt_insn_next.
  int DecodeNextInstruction(DecodedInstruction &insn) {
    return pt_insn_next(&m_decoder, &insn.pt_insn, sizeof(insn.pt_insn));
  }

  /// Decode all the instructions and events until an error is found, the end
  /// of the trace is reached, or optionally a new PSB is reached.
  ///
  /// \param[in] status
  ///   The status that was result of synchronizing to the most recent PSB.
  ///
  /// \param[in] stop_on_psb_change
  ///   If \b true, decoding
  ///   An optional offset to a given PSB. Decoding stops if a different PSB is
  ///   reached.
  void DecodeInstructionsAndEvents(int status,
                                   bool stop_on_psb_change = false) {
    uint64_t psb_offset;
    pt_insn_get_sync_offset(&m_decoder,
                            &psb_offset); // this can't fail because we got here

    while (true) {
      DecodedInstruction insn = ProcessPTEvents(status);
      if (!insn) {
        m_decoded_thread.Append(insn);
        break;
      }

      if (stop_on_psb_change) {
        uint64_t cur_psb_offset;
        pt_insn_get_sync_offset(
            &m_decoder, &cur_psb_offset); // this can't fail because we got here
        if (cur_psb_offset != psb_offset)
          break;
      }

      // The status returned by DecodeNextInstruction will need to be processed
      // by ProcessPTEvents in the next loop if it is not an error.
      if (IsLibiptError(status = DecodeNextInstruction(insn))) {
        insn.libipt_error = status;
        m_decoded_thread.Append(insn);
        break;
      }
      m_decoded_thread.Append(insn);
    }
  }

  /// Move the decoder forward to the next synchronization point (i.e. next PSB
  /// packet).
  ///
  /// Once the decoder is at that synchronization point, it can start decoding
  /// instructions.
  ///
  /// If errors are found, they will be appended to the trace.
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

    // We make this call to record any synchronization errors.
    if (IsLibiptError(status))
      m_decoded_thread.Append(DecodedInstruction(status));

    return status;
  }

  /// Before querying instructions, we need to query the events associated that
  /// instruction e.g. timing events like ptev_tick, or paging events like
  /// ptev_paging.
  ///
  /// \param[in] status
  ///   The status gotten from the previous instruction decoding or PSB
  ///   synchronization.
  ///
  /// \return
  ///   A \a DecodedInstruction with event, tsc and error information.
  DecodedInstruction ProcessPTEvents(int status) {
    DecodedInstruction insn;
    while (status & pts_event_pending) {
      pt_event event;
      status = pt_insn_event(&m_decoder, &event, sizeof(event));
      if (IsLibiptError(status)) {
        insn.libipt_error = status;
        break;
      }

      switch (event.type) {
      case ptev_enabled:
        // The kernel started or resumed tracing the program.
        break;
      case ptev_disabled:
        // The CPU paused tracing the program, e.g. due to ip filtering.
      case ptev_async_disabled:
        // The kernel or user code paused tracing the program, e.g.
        // a breakpoint or a ioctl invocation pausing the trace, or a
        // context switch happened.

        if (m_decoded_thread.GetInstructionsCount() > 0) {
          // A paused event before the first instruction can be safely
          // discarded.
          insn.events |= eTraceEventPaused;
        }
        break;
      case ptev_overflow:
        // The CPU internal buffer had an overflow error and some instructions
        // were lost.
        insn.libipt_error = -pte_overflow;
        break;
      default:
        break;
      }
    }

    // We refresh the TSC that might have changed after processing the events.
    // See
    // https://github.com/intel/libipt/blob/master/doc/man/pt_evt_next.3.md
    RefreshTscInfo();
    if (m_tsc_info)
      insn.tsc = m_tsc_info.tsc;
    return insn;
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
CreateInstructionDecoder(TraceIntelPT &trace_intel_pt,
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

  return PtInsnDecoderUP(decoder_ptr, DecoderDeleter);
}

static Error SetupMemoryImage(PtInsnDecoderUP &decoder_up, Process &process) {
  pt_image *image = pt_insn_get_image(decoder_up.get());

  int status = pte_ok;
  if (IsLibiptError(
          status = pt_image_set_callback(image, ReadProcessMemory, &process)))
    return make_error<IntelPTError>(status);
  return Error::success();
}

void lldb_private::trace_intel_pt::DecodeTrace(DecodedThread &decoded_thread,
                                               TraceIntelPT &trace_intel_pt,
                                               ArrayRef<uint8_t> buffer) {
  Expected<PtInsnDecoderUP> decoder_up =
      CreateInstructionDecoder(trace_intel_pt, buffer);
  if (!decoder_up)
    return decoded_thread.SetAsFailed(decoder_up.takeError());

  if (Error err = SetupMemoryImage(*decoder_up,
                                   *decoded_thread.GetThread()->GetProcess()))
    return decoded_thread.SetAsFailed(std::move(err));

  LibiptDecoder libipt_decoder(*decoder_up.get(), decoded_thread);
  libipt_decoder.DecodeUntilEndOfTrace();
}

void lldb_private::trace_intel_pt::DecodeTrace(
    DecodedThread &decoded_thread, TraceIntelPT &trace_intel_pt,
    const DenseMap<lldb::core_id_t, llvm::ArrayRef<uint8_t>> &buffers,
    const std::vector<IntelPTThreadContinousExecution> &executions) {
  DenseMap<lldb::core_id_t, LibiptDecoder> decoders;
  for (auto &core_id_buffer : buffers) {
    Expected<PtInsnDecoderUP> decoder_up =
        CreateInstructionDecoder(trace_intel_pt, core_id_buffer.second);
    if (!decoder_up)
      return decoded_thread.SetAsFailed(decoder_up.takeError());

    if (Error err = SetupMemoryImage(*decoder_up,
                                     *decoded_thread.GetThread()->GetProcess()))
      return decoded_thread.SetAsFailed(std::move(err));

    decoders.try_emplace(core_id_buffer.first,
                         LibiptDecoder(*decoder_up->release(), decoded_thread));
  }

  bool has_seen_psbs = false;
  for (size_t i = 0; i < executions.size(); i++) {
    const IntelPTThreadContinousExecution &execution = executions[i];

    auto variant = execution.thread_execution.variant;
    // If we haven't seen a PSB yet, then it's fine not to show errors
    if (has_seen_psbs) {
      if (execution.intelpt_subtraces.empty()) {
        decoded_thread.AppendError(createStringError(
            inconvertibleErrorCode(),
            formatv("Unable to find intel pt data for thread execution with "
                    "tsc = {0} on core id = {1}",
                    execution.thread_execution.GetLowestKnownTSC(),
                    execution.thread_execution.core_id)));
      }

      // If the first execution is incomplete because it doesn't have a previous
      // context switch in its cpu, all good.
      if (variant == ThreadContinuousExecution::Variant::OnlyEnd ||
          variant == ThreadContinuousExecution::Variant::HintedStart) {
        decoded_thread.AppendError(createStringError(
            inconvertibleErrorCode(),
            formatv("Thread execution starting at tsc = {0} on core id = {1} "
                    "doesn't have a matching context switch in.",
                    execution.thread_execution.GetLowestKnownTSC(),
                    execution.thread_execution.core_id)));
      }
    }

    LibiptDecoder &decoder =
        decoders.find(execution.thread_execution.core_id)->second;
    for (const IntelPTThreadSubtrace &intel_pt_execution :
         execution.intelpt_subtraces) {
      has_seen_psbs = true;
      decoder.DecodePSB(intel_pt_execution.psb_offset);
    }

    // If we haven't seen a PSB yet, then it's fine not to show errors
    if (has_seen_psbs) {
      // If the last execution is incomplete because it doesn't have a following
      // context switch in its cpu, all good.
      if ((variant == ThreadContinuousExecution::Variant::OnlyStart &&
           i + 1 != executions.size()) ||
          variant == ThreadContinuousExecution::Variant::HintedEnd) {
        decoded_thread.AppendError(createStringError(
            inconvertibleErrorCode(),
            formatv("Thread execution starting at tsc = {0} on core id = {1} "
                    "doesn't have a matching context switch out",
                    execution.thread_execution.GetLowestKnownTSC(),
                    execution.thread_execution.core_id)));
      }
    }
  }
}

bool IntelPTThreadContinousExecution::operator<(
    const IntelPTThreadContinousExecution &o) const {
  // As the context switch might be incomplete, we look first for the first real
  // PSB packet, which is a valid TSC. Otherwise, We query the thread execution
  // itself for some tsc.
  auto get_tsc = [](const IntelPTThreadContinousExecution &exec) {
    return exec.intelpt_subtraces.empty()
               ? exec.thread_execution.GetLowestKnownTSC()
               : exec.intelpt_subtraces.front().tsc;
  };

  return get_tsc(*this) < get_tsc(o);
}

Expected<std::vector<IntelPTThreadSubtrace>>
lldb_private::trace_intel_pt::SplitTraceInContinuousExecutions(
    TraceIntelPT &trace_intel_pt, llvm::ArrayRef<uint8_t> buffer) {
  Expected<PtInsnDecoderUP> decoder_up =
      CreateInstructionDecoder(trace_intel_pt, buffer);
  if (!decoder_up)
    return decoder_up.takeError();

  pt_insn_decoder *decoder = decoder_up.get().get();

  std::vector<IntelPTThreadSubtrace> executions;

  int status = pte_ok;
  while (!IsLibiptError(status = pt_insn_sync_forward(decoder))) {
    uint64_t tsc;
    if (IsLibiptError(pt_insn_time(decoder, &tsc, nullptr, nullptr)))
      return createStringError(inconvertibleErrorCode(),
                               "intel pt trace doesn't have TSC timestamps");

    uint64_t psb_offset;
    pt_insn_get_sync_offset(decoder,
                            &psb_offset); // this can't fail because we got here

    executions.push_back({
        tsc,
        psb_offset,
    });
  }
  return executions;
}
