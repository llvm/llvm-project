//===-- ProcessorTrace.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <fstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "ProcessorTrace.h"
#include "lldb/Host/linux/Support.h"

#include <sys/ioctl.h>
#include <sys/syscall.h>

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

lldb::user_id_t ProcessorTraceMonitor::m_trace_num = 1;
const char *kOSEventIntelPTTypeFile =
    "/sys/bus/event_source/devices/intel_pt/type";

Status ProcessorTraceMonitor::GetTraceConfig(TraceOptions &config) const {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("perf event not supported");
#else
  Status error;

  config.setType(lldb::TraceType::eTraceTypeProcessorTrace);
  config.setMetaDataBufferSize(m_mmap_meta->data_size);

  config.setTraceBufferSize(m_mmap_meta->aux_size);

  error = GetCPUType(config);

  return error;
#endif
}

Expected<uint32_t> ProcessorTraceMonitor::GetOSEventType() {
  auto intel_pt_type_text =
      llvm::MemoryBuffer::getFileAsStream(kOSEventIntelPTTypeFile);

  if (!intel_pt_type_text)
    return createStringError(inconvertibleErrorCode(),
                             "Can't open the file '%s'",
                             kOSEventIntelPTTypeFile);

  uint32_t intel_pt_type = 0;
  StringRef buffer = intel_pt_type_text.get()->getBuffer();
  if (buffer.trim().getAsInteger(10, intel_pt_type))
    return createStringError(
        inconvertibleErrorCode(),
        "The file '%s' has a invalid value. It should be an unsigned int.",
        kOSEventIntelPTTypeFile);
  return intel_pt_type;
}

bool ProcessorTraceMonitor::IsSupported() { return (bool)GetOSEventType(); }

Status ProcessorTraceMonitor::StartTrace(lldb::pid_t pid, lldb::tid_t tid,
                                         const TraceOptions &config) {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("perf event not supported");
#else
  Status error;
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  LLDB_LOG(log, "called thread id {0}", tid);
  uint64_t page_size = getpagesize();
  uint64_t bufsize = config.getTraceBufferSize();
  uint64_t metabufsize = config.getMetaDataBufferSize();

  uint64_t numpages = static_cast<uint64_t>(
      llvm::PowerOf2Floor((bufsize + page_size - 1) / page_size));
  numpages = std::max<uint64_t>(1, numpages);
  bufsize = page_size * numpages;

  numpages = static_cast<uint64_t>(
      llvm::PowerOf2Floor((metabufsize + page_size - 1) / page_size));
  metabufsize = page_size * numpages;

  perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.exclude_kernel = 1;
  attr.sample_type = PERF_SAMPLE_TIME;
  attr.sample_id_all = 1;
  attr.exclude_hv = 1;
  attr.exclude_idle = 1;
  attr.mmap = 1;

  Expected<uint32_t> intel_pt_type = GetOSEventType();

  if (!intel_pt_type) {
    error = intel_pt_type.takeError();
    return error;
  }

  LLDB_LOG(log, "intel pt type {0}", *intel_pt_type);
  attr.type = *intel_pt_type;

  LLDB_LOG(log, "meta buffer size {0}", metabufsize);
  LLDB_LOG(log, "buffer size {0} ", bufsize);

  if (error.Fail()) {
    LLDB_LOG(log, "Status in custom config");

    return error;
  }

  errno = 0;
  auto fd =
      syscall(SYS_perf_event_open, &attr, static_cast<::tid_t>(tid), -1, -1, 0);
  if (fd == -1) {
    LLDB_LOG(log, "syscall error {0}", errno);
    error.SetErrorString("perf event syscall Failed");
    return error;
  }

  m_fd = std::unique_ptr<int, file_close>(new int(fd), file_close());

  errno = 0;
  auto base =
      mmap(nullptr, (metabufsize + page_size), PROT_WRITE, MAP_SHARED, fd, 0);

  if (base == MAP_FAILED) {
    LLDB_LOG(log, "mmap base error {0}", errno);
    error.SetErrorString("Meta buffer allocation failed");
    return error;
  }

  m_mmap_meta = std::unique_ptr<perf_event_mmap_page, munmap_delete>(
      reinterpret_cast<perf_event_mmap_page *>(base),
      munmap_delete(metabufsize + page_size));

  m_mmap_meta->aux_offset = m_mmap_meta->data_offset + m_mmap_meta->data_size;
  m_mmap_meta->aux_size = bufsize;

  errno = 0;
  auto mmap_aux = mmap(nullptr, bufsize, PROT_READ, MAP_SHARED, fd,
                       static_cast<long int>(m_mmap_meta->aux_offset));

  if (mmap_aux == MAP_FAILED) {
    LLDB_LOG(log, "second mmap done {0}", errno);
    error.SetErrorString("Trace buffer allocation failed");
    return error;
  }
  m_mmap_aux = std::unique_ptr<uint8_t, munmap_delete>(
      reinterpret_cast<uint8_t *>(mmap_aux), munmap_delete(bufsize));
  return error;
#endif
}

llvm::MutableArrayRef<uint8_t> ProcessorTraceMonitor::GetDataBuffer() {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("perf event not supported");
#else
  return MutableArrayRef<uint8_t>(
      (reinterpret_cast<uint8_t *>(m_mmap_meta.get()) +
       m_mmap_meta->data_offset),
      m_mmap_meta->data_size);
#endif
}

llvm::MutableArrayRef<uint8_t> ProcessorTraceMonitor::GetAuxBuffer() {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("perf event not supported");
#else
  return MutableArrayRef<uint8_t>(m_mmap_aux.get(), m_mmap_meta->aux_size);
#endif
}

Status ProcessorTraceMonitor::GetCPUType(TraceOptions &config) {

  Status error;
  uint64_t cpu_family = -1;
  uint64_t model = -1;
  uint64_t stepping = -1;
  std::string vendor_id;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  auto BufferOrError = getProcFile("cpuinfo");
  if (!BufferOrError)
    return BufferOrError.getError();

  LLDB_LOG(log, "GetCPUType Function");

  StringRef Rest = BufferOrError.get()->getBuffer();
  while (!Rest.empty()) {
    StringRef Line;
    std::tie(Line, Rest) = Rest.split('\n');

    SmallVector<StringRef, 2> columns;
    Line.split(columns, StringRef(":"), -1, false);

    if (columns.size() < 2)
      continue; // continue searching

    columns[1] = columns[1].trim(" ");
    if (columns[0].contains("cpu family") &&
        columns[1].getAsInteger(10, cpu_family))
      continue;

    else if (columns[0].contains("model") && columns[1].getAsInteger(10, model))
      continue;

    else if (columns[0].contains("stepping") &&
             columns[1].getAsInteger(10, stepping))
      continue;

    else if (columns[0].contains("vendor_id")) {
      vendor_id = columns[1].str();
      if (!vendor_id.empty())
        continue;
    }
    LLDB_LOG(log, "{0}:{1}:{2}:{3}", cpu_family, model, stepping, vendor_id);

    if ((cpu_family != static_cast<uint64_t>(-1)) &&
        (model != static_cast<uint64_t>(-1)) &&
        (stepping != static_cast<uint64_t>(-1)) && (!vendor_id.empty())) {
      auto params_dict = std::make_shared<StructuredData::Dictionary>();
      params_dict->AddIntegerItem("cpu_family", cpu_family);
      params_dict->AddIntegerItem("cpu_model", model);
      params_dict->AddIntegerItem("cpu_stepping", stepping);
      params_dict->AddStringItem("cpu_vendor", vendor_id);

      llvm::StringRef intel_custom_params_key("intel-pt");

      auto intel_custom_params = std::make_shared<StructuredData::Dictionary>();
      intel_custom_params->AddItem(
          intel_custom_params_key,
          StructuredData::ObjectSP(std::move(params_dict)));

      config.setTraceParams(intel_custom_params);
      return error; // we are done
    }
  }

  error.SetErrorString("cpu info not found");
  return error;
}

llvm::Expected<ProcessorTraceMonitorUP>
ProcessorTraceMonitor::Create(lldb::pid_t pid, lldb::tid_t tid,
                              const TraceOptions &config,
                              bool useProcessSettings) {

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  Status error;
  if (tid == LLDB_INVALID_THREAD_ID) {
    error.SetErrorString("thread not specified");
    return error.ToError();
  }

  ProcessorTraceMonitorUP pt_monitor_up(new ProcessorTraceMonitor);

  error = pt_monitor_up->StartTrace(pid, tid, config);
  if (error.Fail())
    return error.ToError();

  pt_monitor_up->SetThreadID(tid);

  if (useProcessSettings) {
    pt_monitor_up->SetTraceID(0);
  } else {
    pt_monitor_up->SetTraceID(m_trace_num++);
    LLDB_LOG(log, "Trace ID {0}", m_trace_num);
  }
  return std::move(pt_monitor_up);
}

Status
ProcessorTraceMonitor::ReadPerfTraceAux(llvm::MutableArrayRef<uint8_t> &buffer,
                                        size_t offset) {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("perf event not supported");
#else
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
  ioctl(*m_fd, PERF_EVENT_IOC_DISABLE);

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  Status error;
  uint64_t head = m_mmap_meta->aux_head;

  LLDB_LOG(log, "Aux size -{0} , Head - {1}", m_mmap_meta->aux_size, head);

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

  ReadCyclicBuffer(buffer, GetAuxBuffer(), static_cast<size_t>(head), offset);
  LLDB_LOG(log, "ReadCyclic BUffer Done");

  // Reenable tracing now we have read the buffer
  ioctl(*m_fd, PERF_EVENT_IOC_ENABLE);
  return error;
#endif
}

Status
ProcessorTraceMonitor::ReadPerfTraceData(llvm::MutableArrayRef<uint8_t> &buffer,
                                         size_t offset) {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("perf event not supported");
#else
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  uint64_t bytes_remaining = buffer.size();
  Status error;

  uint64_t head = m_mmap_meta->data_head;

  /*
   * The data buffer and aux buffer have different implementations
   * with respect to their definition of head pointer. In the case
   * of Aux data buffer the head always wraps around the aux buffer
   * and we don't need to care about it, whereas the data_head keeps
   * increasing and needs to be wrapped by modulus operator
   */

  LLDB_LOG(log, "bytes_remaining - {0}", bytes_remaining);

  auto data_buffer = GetDataBuffer();

  if (head > data_buffer.size()) {
    head = head % data_buffer.size();
    LLDB_LOG(log, "Data size -{0} Head - {1}", m_mmap_meta->data_size, head);

    ReadCyclicBuffer(buffer, data_buffer, static_cast<size_t>(head), offset);
    bytes_remaining -= buffer.size();
  } else {
    LLDB_LOG(log, "Head - {0}", head);
    if (offset >= head) {
      LLDB_LOG(log, "Invalid Offset ");
      error.SetErrorString("invalid offset");
      buffer = buffer.slice(buffer.size());
      return error;
    }

    auto data = data_buffer.slice(offset, (head - offset));
    auto remaining = std::copy(data.begin(), data.end(), buffer.begin());
    bytes_remaining -= (remaining - buffer.begin());
  }
  buffer = buffer.drop_back(bytes_remaining);
  return error;
#endif
}

void ProcessorTraceMonitor::ReadCyclicBuffer(
    llvm::MutableArrayRef<uint8_t> &dst, llvm::MutableArrayRef<uint8_t> src,
    size_t src_cyc_index, size_t offset) {

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  if (dst.empty() || src.empty()) {
    dst = dst.drop_back(dst.size());
    return;
  }

  if (dst.data() == nullptr || src.data() == nullptr) {
    dst = dst.drop_back(dst.size());
    return;
  }

  if (src_cyc_index > src.size()) {
    dst = dst.drop_back(dst.size());
    return;
  }

  if (offset >= src.size()) {
    LLDB_LOG(log, "Too Big offset ");
    dst = dst.drop_back(dst.size());
    return;
  }

  llvm::SmallVector<MutableArrayRef<uint8_t>, 2> parts = {
      src.slice(src_cyc_index), src.take_front(src_cyc_index)};

  if (offset > parts[0].size()) {
    parts[1] = parts[1].slice(offset - parts[0].size());
    parts[0] = parts[0].drop_back(parts[0].size());
  } else if (offset == parts[0].size()) {
    parts[0] = parts[0].drop_back(parts[0].size());
  } else {
    parts[0] = parts[0].slice(offset);
  }
  auto next = dst.begin();
  auto bytes_left = dst.size();
  for (auto part : parts) {
    size_t chunk_size = std::min(part.size(), bytes_left);
    next = std::copy_n(part.begin(), chunk_size, next);
    bytes_left -= chunk_size;
  }
  dst = dst.drop_back(bytes_left);
}
