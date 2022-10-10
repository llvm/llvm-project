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

bool IsLibiptError(int status) { return status < 0; }

bool IsEndOfStream(int status) {
  assert(status >= 0 && "We can't check if we reached the end of the stream if "
                        "we got a failed status");
  return status & pts_eos;
}

bool HasEvents(int status) {
  assert(status >= 0 && "We can't check for events if we got a failed status");
  return status & pts_event_pending;
}

// RAII deleter for libipt's decoders
auto InsnDecoderDeleter = [](pt_insn_decoder *decoder) {
  pt_insn_free_decoder(decoder);
};

auto QueryDecoderDeleter = [](pt_query_decoder *decoder) {
  pt_qry_free_decoder(decoder);
};

using PtInsnDecoderUP =
    std::unique_ptr<pt_insn_decoder, decltype(InsnDecoderDeleter)>;

using PtQueryDecoderUP =
    std::unique_ptr<pt_query_decoder, decltype(QueryDecoderDeleter)>;

/// Create a basic configuration object limited to a given buffer that can be
/// used for many different decoders.
static Expected<pt_config> CreateBasicLibiptConfig(TraceIntelPT &trace_intel_pt,
                                                   ArrayRef<uint8_t> buffer) {
  Expected<pt_cpu> cpu_info = trace_intel_pt.GetCPUInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  pt_config config;
  pt_config_init(&config);
  config.cpu = *cpu_info;

  int status = pt_cpu_errata(&config.errata, &config.cpu);
  if (IsLibiptError(status))
    return make_error<IntelPTError>(status);

  // The libipt library does not modify the trace buffer, hence the
  // following casts are safe.
  config.begin = const_cast<uint8_t *>(buffer.data());
  config.end = const_cast<uint8_t *>(buffer.data() + buffer.size());
  return config;
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

/// Set up the memory image callback for the given decoder.
static Error SetupMemoryImage(pt_insn_decoder *decoder, Process &process) {
  pt_image *image = pt_insn_get_image(decoder);

  int status = pt_image_set_callback(image, ReadProcessMemory, &process);
  if (IsLibiptError(status))
    return make_error<IntelPTError>(status);
  return Error::success();
}

/// Create an instruction decoder for the given buffer and the given process.
static Expected<PtInsnDecoderUP>
CreateInstructionDecoder(TraceIntelPT &trace_intel_pt, ArrayRef<uint8_t> buffer,
                         Process &process) {
  Expected<pt_config> config = CreateBasicLibiptConfig(trace_intel_pt, buffer);
  if (!config)
    return config.takeError();

  pt_insn_decoder *decoder_ptr = pt_insn_alloc_decoder(&*config);
  if (!decoder_ptr)
    return make_error<IntelPTError>(-pte_nomem);

  PtInsnDecoderUP decoder_up(decoder_ptr, InsnDecoderDeleter);

  if (Error err = SetupMemoryImage(decoder_ptr, process))
    return std::move(err);

  return decoder_up;
}

/// Create a query decoder for the given buffer. The query decoder is the
/// highest level decoder that operates directly on packets and doesn't perform
/// actual instruction decoding. That's why it can be useful for inspecting a
/// raw trace without pinning it to a particular process.
static Expected<PtQueryDecoderUP>
CreateQueryDecoder(TraceIntelPT &trace_intel_pt, ArrayRef<uint8_t> buffer) {
  Expected<pt_config> config = CreateBasicLibiptConfig(trace_intel_pt, buffer);
  if (!config)
    return config.takeError();

  pt_query_decoder *decoder_ptr = pt_qry_alloc_decoder(&*config);
  if (!decoder_ptr)
    return make_error<IntelPTError>(-pte_nomem);

  return PtQueryDecoderUP(decoder_ptr, QueryDecoderDeleter);
}

/// Class that decodes a raw buffer for a single PSB block using the low level
/// libipt library. It assumes that kernel and user mode instructions are not
/// mixed in the same PSB block.
///
/// Throughout this code, the status of the decoder will be used to identify
/// events needed to be processed or errors in the decoder. The values can be
/// - negative: actual errors
/// - positive or zero: not an error, but a list of bits signaling the status
/// of the decoder, e.g. whether there are events that need to be decoded or
/// not.
class PSBBlockDecoder {
public:
  /// \param[in] decoder
  ///     A decoder configured to start and end within the boundaries of the
  ///     given \p psb_block.
  ///
  /// \param[in] psb_block
  ///     The PSB block to decode.
  ///
  /// \param[in] next_block_ip
  ///     The starting ip at the next PSB block of the same thread if available.
  ///
  /// \param[in] decoded_thread
  ///     A \a DecodedThread object where the decoded instructions will be
  ///     appended to. It might have already some instructions.
  PSBBlockDecoder(PtInsnDecoderUP &&decoder_up, const PSBBlock &psb_block,
                  Optional<lldb::addr_t> next_block_ip,
                  DecodedThread &decoded_thread)
      : m_decoder_up(std::move(decoder_up)), m_psb_block(psb_block),
        m_next_block_ip(next_block_ip), m_decoded_thread(decoded_thread) {}

  /// \param[in] trace_intel_pt
  ///     The main Trace object that own the PSB block.
  ///
  /// \param[in] decoder
  ///     A decoder configured to start and end within the boundaries of the
  ///     given \p psb_block.
  ///
  /// \param[in] psb_block
  ///     The PSB block to decode.
  ///
  /// \param[in] buffer
  ///     The raw intel pt trace for this block.
  ///
  /// \param[in] process
  ///     The process to decode. It provides the memory image to use for
  ///     decoding.
  ///
  /// \param[in] next_block_ip
  ///     The starting ip at the next PSB block of the same thread if available.
  ///
  /// \param[in] decoded_thread
  ///     A \a DecodedThread object where the decoded instructions will be
  ///     appended to. It might have already some instructions.
  static Expected<PSBBlockDecoder>
  Create(TraceIntelPT &trace_intel_pt, const PSBBlock &psb_block,
         ArrayRef<uint8_t> buffer, Process &process,
         Optional<lldb::addr_t> next_block_ip, DecodedThread &decoded_thread) {
    Expected<PtInsnDecoderUP> decoder_up =
        CreateInstructionDecoder(trace_intel_pt, buffer, process);
    if (!decoder_up)
      return decoder_up.takeError();

    return PSBBlockDecoder(std::move(*decoder_up), psb_block, next_block_ip,
                           decoded_thread);
  }

  void DecodePSBBlock() {
    int status = pt_insn_sync_forward(m_decoder_up.get());
    assert(status >= 0 &&
           "Synchronization shouldn't fail because this PSB was previously "
           "decoded correctly.");

    // We emit a TSC before a sync event to more easily associate a timestamp to
    // the sync event. If present, the current block's TSC would be the first
    // TSC we'll see when processing events.
    if (m_psb_block.tsc)
      m_decoded_thread.NotifyTsc(*m_psb_block.tsc);

    m_decoded_thread.NotifySyncPoint(m_psb_block.psb_offset);

    DecodeInstructionsAndEvents(status);
  }

private:
  /// Decode all the instructions and events of the given PSB block.
  ///
  /// \param[in] status
  ///   The status that was result of synchronizing to the most recent PSB.
  void DecodeInstructionsAndEvents(int status) {
    pt_insn insn;
    while (true) {
      status = ProcessPTEvents(status);

      if (IsLibiptError(status))
        return;
      else if (IsEndOfStream(status))
        break;

      // The status returned by pt_insn_next will need to be processed
      // by ProcessPTEvents in the next loop if it is not an error.
      std::memset(&insn, 0, sizeof insn);
      status = pt_insn_next(m_decoder_up.get(), &insn, sizeof(insn));

      if (IsLibiptError(status)) {
        m_decoded_thread.AppendError(IntelPTError(status, insn.ip));
        return;
      } else if (IsEndOfStream(status)) {
        break;
      }
      m_decoded_thread.AppendInstruction(insn);
    }

    // We need to keep querying non-branching instructions until we hit the
    // starting point of the next PSB. We won't see events at this point. This
    // is based on
    // https://github.com/intel/libipt/blob/master/doc/howto_libipt.md#parallel-decode
    if (m_next_block_ip && insn.ip != 0) {
      while (insn.ip != *m_next_block_ip) {
        m_decoded_thread.AppendInstruction(insn);

        status = pt_insn_next(m_decoder_up.get(), &insn, sizeof(insn));

        if (IsLibiptError(status)) {
          m_decoded_thread.AppendError(IntelPTError(status, insn.ip));
          return;
        }
      }
    }
  }

  /// Before querying instructions, we need to query the events associated with
  /// that instruction, e.g. timing and trace disablement events.
  ///
  /// \param[in] status
  ///   The status gotten from the previous instruction decoding or PSB
  ///   synchronization.
  ///
  /// \return
  ///     The pte_status after decoding events.
  int ProcessPTEvents(int status) {
    while (HasEvents(status)) {
      pt_event event;
      std::memset(&event, 0, sizeof event);
      status = pt_insn_event(m_decoder_up.get(), &event, sizeof(event));

      if (IsLibiptError(status)) {
        m_decoded_thread.AppendError(IntelPTError(status));
        return status;
      }

      if (event.has_tsc)
        m_decoded_thread.NotifyTsc(event.tsc);

      switch (event.type) {
      case ptev_disabled:
        // The CPU paused tracing the program, e.g. due to ip filtering.
        m_decoded_thread.AppendEvent(lldb::eTraceEventDisabledHW);
        break;
      case ptev_async_disabled:
        // The kernel or user code paused tracing the program, e.g.
        // a breakpoint or a ioctl invocation pausing the trace, or a
        // context switch happened.
        m_decoded_thread.AppendEvent(lldb::eTraceEventDisabledSW);
        break;
      case ptev_overflow:
        // The CPU internal buffer had an overflow error and some instructions
        // were lost. A OVF packet comes with an FUP packet (harcoded address)
        // according to the documentation, so we'll continue seeing instructions
        // after this event.
        m_decoded_thread.AppendError(IntelPTError(-pte_overflow));
        break;
      default:
        break;
      }
    }

    return status;
  }

private:
  PtInsnDecoderUP m_decoder_up;
  PSBBlock m_psb_block;
  Optional<lldb::addr_t> m_next_block_ip;
  DecodedThread &m_decoded_thread;
};

Error lldb_private::trace_intel_pt::DecodeSingleTraceForThread(
    DecodedThread &decoded_thread, TraceIntelPT &trace_intel_pt,
    ArrayRef<uint8_t> buffer) {
  Expected<std::vector<PSBBlock>> blocks =
      SplitTraceIntoPSBBlock(trace_intel_pt, buffer, /*expect_tscs=*/false);
  if (!blocks)
    return blocks.takeError();

  for (size_t i = 0; i < blocks->size(); i++) {
    PSBBlock &block = blocks->at(i);

    Expected<PSBBlockDecoder> decoder = PSBBlockDecoder::Create(
        trace_intel_pt, block, buffer.slice(block.psb_offset, block.size),
        *decoded_thread.GetThread()->GetProcess(),
        i + 1 < blocks->size() ? blocks->at(i + 1).starting_ip : None,
        decoded_thread);
    if (!decoder)
      return decoder.takeError();

    decoder->DecodePSBBlock();
  }

  return Error::success();
}

Error lldb_private::trace_intel_pt::DecodeSystemWideTraceForThread(
    DecodedThread &decoded_thread, TraceIntelPT &trace_intel_pt,
    const DenseMap<lldb::cpu_id_t, llvm::ArrayRef<uint8_t>> &buffers,
    const std::vector<IntelPTThreadContinousExecution> &executions) {
  bool has_seen_psbs = false;
  for (size_t i = 0; i < executions.size(); i++) {
    const IntelPTThreadContinousExecution &execution = executions[i];

    auto variant = execution.thread_execution.variant;

    // We emit the first valid tsc
    if (execution.psb_blocks.empty()) {
      decoded_thread.NotifyTsc(execution.thread_execution.GetLowestKnownTSC());
    } else {
      assert(execution.psb_blocks.front().tsc &&
             "per cpu decoding expects TSCs");
      decoded_thread.NotifyTsc(
          std::min(execution.thread_execution.GetLowestKnownTSC(),
                   *execution.psb_blocks.front().tsc));
    }

    // We then emit the CPU, which will be correctly associated with a tsc.
    decoded_thread.NotifyCPU(execution.thread_execution.cpu_id);

    // If we haven't seen a PSB yet, then it's fine not to show errors
    if (has_seen_psbs) {
      if (execution.psb_blocks.empty()) {
        decoded_thread.AppendCustomError(
            formatv("Unable to find intel pt data a thread "
                    "execution on cpu id = {0}",
                    execution.thread_execution.cpu_id)
                .str());
      }

      // A hinted start is a non-initial execution that doesn't have a switch
      // in. An only end is an initial execution that doesn't have a switch in.
      // Any of those cases represent a gap because we have seen a PSB before.
      if (variant == ThreadContinuousExecution::Variant::HintedStart ||
          variant == ThreadContinuousExecution::Variant::OnlyEnd) {
        decoded_thread.AppendCustomError(
            formatv("Unable to find the context switch in for a thread "
                    "execution on cpu id = {0}",
                    execution.thread_execution.cpu_id)
                .str());
      }
    }

    for (size_t j = 0; j < execution.psb_blocks.size(); j++) {
      const PSBBlock &psb_block = execution.psb_blocks[j];

      Expected<PSBBlockDecoder> decoder = PSBBlockDecoder::Create(
          trace_intel_pt, psb_block,
          buffers.lookup(executions[i].thread_execution.cpu_id)
              .slice(psb_block.psb_offset, psb_block.size),
          *decoded_thread.GetThread()->GetProcess(),
          j + 1 < execution.psb_blocks.size()
              ? execution.psb_blocks[j + 1].starting_ip
              : None,
          decoded_thread);
      if (!decoder)
        return decoder.takeError();

      has_seen_psbs = true;
      decoder->DecodePSBBlock();
    }

    // If we haven't seen a PSB yet, then it's fine not to show errors
    if (has_seen_psbs) {
      // A hinted end is a non-ending execution that doesn't have a switch out.
      // An only start is an ending execution that doesn't have a switch out.
      // Any of those cases represent a gap if we still have executions to
      // process and we have seen a PSB before.
      if (i + 1 != executions.size() &&
          (variant == ThreadContinuousExecution::Variant::OnlyStart ||
           variant == ThreadContinuousExecution::Variant::HintedEnd)) {
        decoded_thread.AppendCustomError(
            formatv("Unable to find the context switch out for a thread "
                    "execution on cpu id = {0}",
                    execution.thread_execution.cpu_id)
                .str());
      }
    }
  }
  return Error::success();
}

bool IntelPTThreadContinousExecution::operator<(
    const IntelPTThreadContinousExecution &o) const {
  // As the context switch might be incomplete, we look first for the first real
  // PSB packet, which is a valid TSC. Otherwise, We query the thread execution
  // itself for some tsc.
  auto get_tsc = [](const IntelPTThreadContinousExecution &exec) {
    return exec.psb_blocks.empty() ? exec.thread_execution.GetLowestKnownTSC()
                                   : exec.psb_blocks.front().tsc;
  };

  return get_tsc(*this) < get_tsc(o);
}

Expected<std::vector<PSBBlock>>
lldb_private::trace_intel_pt::SplitTraceIntoPSBBlock(
    TraceIntelPT &trace_intel_pt, llvm::ArrayRef<uint8_t> buffer,
    bool expect_tscs) {
  // This follows
  // https://github.com/intel/libipt/blob/master/doc/howto_libipt.md#parallel-decode

  Expected<PtQueryDecoderUP> decoder_up =
      CreateQueryDecoder(trace_intel_pt, buffer);
  if (!decoder_up)
    return decoder_up.takeError();

  pt_query_decoder *decoder = decoder_up.get().get();

  std::vector<PSBBlock> executions;

  while (true) {
    uint64_t maybe_ip = LLDB_INVALID_ADDRESS;
    int decoding_status = pt_qry_sync_forward(decoder, &maybe_ip);
    if (IsLibiptError(decoding_status))
      break;

    uint64_t psb_offset;
    int offset_status = pt_qry_get_sync_offset(decoder, &psb_offset);
    assert(offset_status >= 0 &&
           "This can't fail because we were able to synchronize");

    Optional<uint64_t> ip;
    if (!(pts_ip_suppressed & decoding_status))
      ip = maybe_ip;

    Optional<uint64_t> tsc;
    // Now we fetch the first TSC that comes after the PSB.
    while (HasEvents(decoding_status)) {
      pt_event event;
      decoding_status = pt_qry_event(decoder, &event, sizeof(event));
      if (IsLibiptError(decoding_status))
        break;
      if (event.has_tsc) {
        tsc = event.tsc;
        break;
      }
    }
    if (IsLibiptError(decoding_status)) {
      // We continue to the next PSB. This effectively merges this PSB with the
      // previous one, and that should be fine because this PSB might be the
      // direct continuation of the previous thread and it's better to show an
      // error in the decoded thread than to hide it. If this is the first PSB,
      // we are okay losing it. Besides that, an error at processing events
      // means that we wouldn't be able to get any instruction out of it.
      continue;
    }

    if (expect_tscs && !tsc)
      return createStringError(inconvertibleErrorCode(),
                               "Found a PSB without TSC.");

    executions.push_back({
        psb_offset,
        tsc,
        0,
        ip,
    });
  }
  if (!executions.empty()) {
    // We now adjust the sizes of each block
    executions.back().size = buffer.size() - executions.back().psb_offset;
    for (int i = (int)executions.size() - 2; i >= 0; i--) {
      executions[i].size =
          executions[i + 1].psb_offset - executions[i].psb_offset;
    }
  }
  return executions;
}

Expected<Optional<uint64_t>>
lldb_private::trace_intel_pt::FindLowestTSCInTrace(TraceIntelPT &trace_intel_pt,
                                                   ArrayRef<uint8_t> buffer) {
  Expected<PtQueryDecoderUP> decoder_up =
      CreateQueryDecoder(trace_intel_pt, buffer);
  if (!decoder_up)
    return decoder_up.takeError();

  pt_query_decoder *decoder = decoder_up.get().get();
  uint64_t ip = LLDB_INVALID_ADDRESS;
  int status = pt_qry_sync_forward(decoder, &ip);
  if (IsLibiptError(status))
    return None;

  while (HasEvents(status)) {
    pt_event event;
    status = pt_qry_event(decoder, &event, sizeof(event));
    if (IsLibiptError(status))
      return None;
    if (event.has_tsc)
      return event.tsc;
  }
  return None;
}
