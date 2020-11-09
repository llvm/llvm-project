//===-- IntelPTDecoder.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTDecoder.h"

#include "llvm/Support/MemoryBuffer.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadTrace.h"

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
};

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
///
/// \return
///   The decoded instructions.
static std::vector<IntelPTInstruction>
DecodeInstructions(pt_insn_decoder &decoder) {
  std::vector<IntelPTInstruction> instructions;

  while (true) {
    int errcode = FindNextSynchronizationPoint(decoder);
    if (errcode == -pte_eos)
      break;

    if (errcode < 0) {
      instructions.emplace_back(make_error<IntelPTError>(errcode));
      break;
    }

    // We have synchronized, so we can start decoding
    // instructions and events.
    while (true) {
      errcode = ProcessPTEvents(decoder, errcode);
      if (errcode < 0) {
        instructions.emplace_back(make_error<IntelPTError>(errcode));
        break;
      }
      pt_insn insn;

      errcode = pt_insn_next(&decoder, &insn, sizeof(insn));
      if (errcode == -pte_eos)
        break;

      if (errcode < 0) {
        instructions.emplace_back(make_error<IntelPTError>(errcode, insn.ip));
        break;
      }

      instructions.emplace_back(insn);
    }
  }

  return instructions;
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

static std::vector<IntelPTInstruction> makeInstructionListFromError(Error err) {
  std::vector<IntelPTInstruction> instructions;
  instructions.emplace_back(std::move(err));
  return instructions;
}

static std::vector<IntelPTInstruction>
CreateDecoderAndDecode(Process &process, const pt_cpu &pt_cpu,
                       const FileSpec &trace_file) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> trace_or_error =
      MemoryBuffer::getFile(trace_file.GetPath());
  if (std::error_code err = trace_or_error.getError())
    return makeInstructionListFromError(errorCodeToError(err));

  MemoryBuffer &trace = **trace_or_error;

  pt_config config;
  pt_config_init(&config);
  config.cpu = pt_cpu;

  if (int errcode = pt_cpu_errata(&config.errata, &config.cpu))
    return makeInstructionListFromError(make_error<IntelPTError>(errcode));

  // The libipt library does not modify the trace buffer, hence the following
  // cast is safe.
  config.begin =
      reinterpret_cast<uint8_t *>(const_cast<char *>(trace.getBufferStart()));
  config.end =
      reinterpret_cast<uint8_t *>(const_cast<char *>(trace.getBufferEnd()));

  pt_insn_decoder *decoder = pt_insn_alloc_decoder(&config);
  if (!decoder)
    return makeInstructionListFromError(make_error<IntelPTError>(-pte_nomem));

  pt_image *image = pt_insn_get_image(decoder);

  int errcode = pt_image_set_callback(image, ReadProcessMemory, &process);
  assert(errcode == 0);
  (void)errcode;

  std::vector<IntelPTInstruction> instructions = DecodeInstructions(*decoder);

  pt_insn_free_decoder(decoder);
  return instructions;
}

const DecodedThread &ThreadTraceDecoder::Decode() {
  if (!m_decoded_thread.hasValue()) {
    m_decoded_thread = DecodedThread(
        CreateDecoderAndDecode(*m_trace_thread->GetProcess(), m_pt_cpu,
                               m_trace_thread->GetTraceFile()));
  }

  return *m_decoded_thread;
}
