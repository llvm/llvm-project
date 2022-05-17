//===-- IntelPTSingleBufferTrace.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelPTSingleBufferTrace.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"

#include <sstream>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

const char *kOSEventIntelPTTypeFile =
    "/sys/bus/event_source/devices/intel_pt/type";

const char *kPSBPeriodCapFile =
    "/sys/bus/event_source/devices/intel_pt/caps/psb_cyc";

const char *kPSBPeriodValidValuesFile =
    "/sys/bus/event_source/devices/intel_pt/caps/psb_periods";

const char *kPSBPeriodBitOffsetFile =
    "/sys/bus/event_source/devices/intel_pt/format/psb_period";

const char *kTSCBitOffsetFile =
    "/sys/bus/event_source/devices/intel_pt/format/tsc";

enum IntelPTConfigFileType {
  Hex = 0,
  // 0 or 1
  ZeroOne,
  Decimal,
  // a bit index file always starts with the prefix config: following by an int,
  // which represents the offset of the perf_event_attr.config value where to
  // store a given configuration.
  BitOffset
};

static Expected<uint32_t> ReadIntelPTConfigFile(const char *file,
                                                IntelPTConfigFileType type) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> stream =
      MemoryBuffer::getFileAsStream(file);

  if (!stream)
    return createStringError(inconvertibleErrorCode(),
                             "Can't open the file '%s'", file);

  uint32_t value = 0;
  StringRef text_buffer = stream.get()->getBuffer();

  if (type == BitOffset) {
    const char *prefix = "config:";
    if (!text_buffer.startswith(prefix))
      return createStringError(inconvertibleErrorCode(),
                               "The file '%s' contents doesn't start with '%s'",
                               file, prefix);
    text_buffer = text_buffer.substr(strlen(prefix));
  }

  auto getRadix = [&]() {
    switch (type) {
    case Hex:
      return 16;
    case ZeroOne:
    case Decimal:
    case BitOffset:
      return 10;
    }
    llvm_unreachable("Fully covered switch above!");
  };

  auto createError = [&](const char *expected_value_message) {
    return createStringError(
        inconvertibleErrorCode(),
        "The file '%s' has an invalid value. It should be %s.", file,
        expected_value_message);
  };

  if (text_buffer.trim().consumeInteger(getRadix(), value) ||
      (type == ZeroOne && value != 0 && value != 1)) {
    switch (type) {
    case Hex:
      return createError("an unsigned hexadecimal int");
    case ZeroOne:
      return createError("0 or 1");
    case Decimal:
    case BitOffset:
      return createError("an unsigned decimal int");
    }
  }
  return value;
}

/// Return the Linux perf event type for Intel PT.
Expected<uint32_t> process_linux::GetIntelPTOSEventType() {
  return ReadIntelPTConfigFile(kOSEventIntelPTTypeFile,
                               IntelPTConfigFileType::Decimal);
}

static Error CheckPsbPeriod(size_t psb_period) {
  Expected<uint32_t> cap =
      ReadIntelPTConfigFile(kPSBPeriodCapFile, IntelPTConfigFileType::ZeroOne);
  if (!cap)
    return cap.takeError();
  if (*cap == 0)
    return createStringError(inconvertibleErrorCode(),
                             "psb_period is unsupported in the system.");

  Expected<uint32_t> valid_values = ReadIntelPTConfigFile(
      kPSBPeriodValidValuesFile, IntelPTConfigFileType::Hex);
  if (!valid_values)
    return valid_values.takeError();

  if (valid_values.get() & (1 << psb_period))
    return Error::success();

  std::ostringstream error;
  // 0 is always a valid value
  error << "Invalid psb_period. Valid values are: 0";
  uint32_t mask = valid_values.get();
  while (mask) {
    int index = __builtin_ctz(mask);
    if (index > 0)
      error << ", " << index;
    // clear the lowest bit
    mask &= mask - 1;
  }
  error << ".";
  return createStringError(inconvertibleErrorCode(), error.str().c_str());
}

#ifdef PERF_ATTR_SIZE_VER5
static Expected<uint64_t>
GeneratePerfEventConfigValue(bool enable_tsc, Optional<uint64_t> psb_period) {
  uint64_t config = 0;
  // tsc is always supported
  if (enable_tsc) {
    if (Expected<uint32_t> offset = ReadIntelPTConfigFile(
            kTSCBitOffsetFile, IntelPTConfigFileType::BitOffset))
      config |= 1 << *offset;
    else
      return offset.takeError();
  }
  if (psb_period) {
    if (Error error = CheckPsbPeriod(*psb_period))
      return std::move(error);

    if (Expected<uint32_t> offset = ReadIntelPTConfigFile(
            kPSBPeriodBitOffsetFile, IntelPTConfigFileType::BitOffset))
      config |= *psb_period << *offset;
    else
      return offset.takeError();
  }
  return config;
}

/// Create a \a perf_event_attr configured for
/// an IntelPT event.
///
/// \return
///   A \a perf_event_attr if successful,
///   or an \a llvm::Error otherwise.
static Expected<perf_event_attr>
CreateIntelPTPerfEventConfiguration(bool enable_tsc,
                                    llvm::Optional<uint64_t> psb_period) {
  perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.exclude_kernel = 1;
  attr.sample_type = PERF_SAMPLE_TIME;
  attr.sample_id_all = 1;
  attr.exclude_hv = 1;
  attr.exclude_idle = 1;
  attr.mmap = 1;

  if (Expected<uint64_t> config_value =
          GeneratePerfEventConfigValue(enable_tsc, psb_period))
    attr.config = *config_value;
  else
    return config_value.takeError();

  if (Expected<uint32_t> intel_pt_type = GetIntelPTOSEventType())
    attr.type = *intel_pt_type;
  else
    return intel_pt_type.takeError();

  return attr;
}
#endif

size_t IntelPTSingleBufferTrace::GetTraceBufferSize() const {
  return m_perf_event.GetAuxBuffer().size();
}

Expected<std::vector<uint8_t>>
IntelPTSingleBufferTrace::GetTraceBuffer(size_t offset, size_t size) const {
  auto fd = m_perf_event.GetFd();
  perf_event_mmap_page &mmap_metadata = m_perf_event.GetMetadataPage();
  // Disable the perf event to force a flush out of the CPU's internal buffer.
  // Besides, we can guarantee that the CPU won't override any data as we are
  // reading the buffer.
  //
  // The Intel documentation says:
  //
  // Packets are first buffered internally and then written out asynchronously.
  // To collect packet output for postprocessing, a collector needs first to
  // ensure that all packet data has been flushed from internal buffers.
  // Software can ensure this by stopping packet generation by clearing
  // IA32_RTIT_CTL.TraceEn (see “Disabling Packet Generation” in
  // Section 35.2.7.2).
  //
  // This is achieved by the PERF_EVENT_IOC_DISABLE ioctl request, as mentioned
  // in the man page of perf_event_open.
  ioctl(fd, PERF_EVENT_IOC_DISABLE);

  Log *log = GetLog(POSIXLog::Trace);
  Status error;
  uint64_t head = mmap_metadata.aux_head;

  LLDB_LOG(log, "Aux size -{0} , Head - {1}", mmap_metadata.aux_size, head);

  /**
   * When configured as ring buffer, the aux buffer keeps wrapping around
   * the buffer and its not possible to detect how many times the buffer
   * wrapped. Initially the buffer is filled with zeros,as shown below
   * so in order to get complete buffer we first copy firstpartsize, followed
   * by any left over part from beginning to aux_head
   *
   * aux_offset [d,d,d,d,d,d,d,d,0,0,0,0,0,0,0,0,0,0,0] aux_size
   *                 aux_head->||<- firstpartsize  ->|
   *
   * */

  std::vector<uint8_t> data(size, 0);
  MutableArrayRef<uint8_t> buffer(data);
  ReadCyclicBuffer(buffer, m_perf_event.GetAuxBuffer(),
                   static_cast<size_t>(head), offset);

  // Reenable tracing now we have read the buffer
  ioctl(fd, PERF_EVENT_IOC_ENABLE);
  return data;
}

Expected<IntelPTSingleBufferTraceUP>
IntelPTSingleBufferTrace::Start(const TraceIntelPTStartRequest &request,
                                Optional<lldb::tid_t> tid,
                                Optional<core_id_t> core_id) {
#ifndef PERF_ATTR_SIZE_VER5
  return createStringError(inconvertibleErrorCode(),
                           "Intel PT Linux perf event not supported");
#else
  Log *log = GetLog(POSIXLog::Trace);

  LLDB_LOG(log, "Will start tracing thread id {0} and cpu id {1}", tid,
           core_id);

  if (__builtin_popcount(request.trace_buffer_size) != 1 ||
      request.trace_buffer_size < 4096) {
    return createStringError(
        inconvertibleErrorCode(),
        "The trace buffer size must be a power of 2 greater than or equal to "
        "4096 (2^12) bytes. It was %" PRIu64 ".",
        request.trace_buffer_size);
  }
  uint64_t page_size = getpagesize();
  uint64_t buffer_numpages = static_cast<uint64_t>(llvm::PowerOf2Floor(
      (request.trace_buffer_size + page_size - 1) / page_size));

  Expected<perf_event_attr> attr = CreateIntelPTPerfEventConfiguration(
      request.enable_tsc, request.psb_period.map([](int value) {
        return static_cast<uint64_t>(value);
      }));
  if (!attr)
    return attr.takeError();

  LLDB_LOG(log, "Will create trace buffer of size {0}",
           request.trace_buffer_size);

  if (Expected<PerfEvent> perf_event = PerfEvent::Init(*attr, tid, core_id)) {
    if (Error mmap_err = perf_event->MmapMetadataAndBuffers(buffer_numpages,
                                                            buffer_numpages)) {
      return std::move(mmap_err);
    }
    return IntelPTSingleBufferTraceUP(
        new IntelPTSingleBufferTrace(std::move(*perf_event)));
  } else {
    return perf_event.takeError();
  }
#endif
}
