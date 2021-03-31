//===-- IntelPTManager.cpp ------------------------------------------------===//
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

#include "IntelPTManager.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/linux/Support.h"
#include "lldb/Utility/StreamString.h"

#include <sys/ioctl.h>
#include <sys/syscall.h>

using namespace lldb;
using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

const char *kOSEventIntelPTTypeFile =
    "/sys/bus/event_source/devices/intel_pt/type";

/// Return the Linux perf event type for Intel PT.
static Expected<uint32_t> GetOSEventType() {
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

size_t IntelPTThreadTrace::GetTraceBufferSize() const {
  return m_mmap_meta->aux_size;
}

Error IntelPTThreadTrace::StartTrace(lldb::pid_t pid, lldb::tid_t tid,
                                     uint64_t buffer_size) {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("Intel PT Linux perf event not supported");
#else
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  m_tid = tid;
  LLDB_LOG(log, "called thread id {0}", tid);
  uint64_t page_size = getpagesize();

  if (__builtin_popcount(buffer_size) != 1 || buffer_size < 4096) {
    return createStringError(
        inconvertibleErrorCode(),
        "The trace buffer size must be a power of 2 greater than or equal to "
        "4096 (2^12) bytes. It was %" PRIu64 ".",
        buffer_size);
  }
  uint64_t numpages = static_cast<uint64_t>(
      llvm::PowerOf2Floor((buffer_size + page_size - 1) / page_size));
  numpages = std::max<uint64_t>(1, numpages);
  buffer_size = page_size * numpages;

  perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.exclude_kernel = 1;
  attr.sample_type = PERF_SAMPLE_TIME;
  attr.sample_id_all = 1;
  attr.exclude_hv = 1;
  attr.exclude_idle = 1;
  attr.mmap = 1;
  attr.config = 0;

  Expected<uint32_t> intel_pt_type = GetOSEventType();

  if (!intel_pt_type)
    return intel_pt_type.takeError();

  LLDB_LOG(log, "intel pt type {0}", *intel_pt_type);
  attr.type = *intel_pt_type;

  LLDB_LOG(log, "buffer size {0} ", buffer_size);

  errno = 0;
  auto fd =
      syscall(SYS_perf_event_open, &attr, static_cast<::tid_t>(tid), -1, -1, 0);
  if (fd == -1) {
    LLDB_LOG(log, "syscall error {0}", errno);
    return createStringError(inconvertibleErrorCode(),
                             "perf event syscall failed");
  }

  m_fd = std::unique_ptr<int, file_close>(new int(fd), file_close());

  errno = 0;
  auto base =
      mmap(nullptr, (buffer_size + page_size), PROT_WRITE, MAP_SHARED, fd, 0);

  if (base == MAP_FAILED) {
    LLDB_LOG(log, "mmap base error {0}", errno);
    return createStringError(inconvertibleErrorCode(),
                             "Meta buffer allocation failed");
  }

  m_mmap_meta = std::unique_ptr<perf_event_mmap_page, munmap_delete>(
      reinterpret_cast<perf_event_mmap_page *>(base),
      munmap_delete(buffer_size + page_size));

  m_mmap_meta->aux_offset = m_mmap_meta->data_offset + m_mmap_meta->data_size;
  m_mmap_meta->aux_size = buffer_size;

  errno = 0;
  auto mmap_aux = mmap(nullptr, buffer_size, PROT_READ, MAP_SHARED, fd,
                       static_cast<long int>(m_mmap_meta->aux_offset));

  if (mmap_aux == MAP_FAILED) {
    LLDB_LOG(log, "second mmap done {0}", errno);
    return createStringError(inconvertibleErrorCode(),
                             "Trace buffer allocation failed");
  }
  m_mmap_aux = std::unique_ptr<uint8_t, munmap_delete>(
      reinterpret_cast<uint8_t *>(mmap_aux), munmap_delete(buffer_size));
  return Error::success();
#endif
}

llvm::MutableArrayRef<uint8_t> IntelPTThreadTrace::GetDataBuffer() const {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("Intel PT Linux perf event not supported");
#else
  return MutableArrayRef<uint8_t>(
      (reinterpret_cast<uint8_t *>(m_mmap_meta.get()) +
       m_mmap_meta->data_offset),
      m_mmap_meta->data_size);
#endif
}

llvm::MutableArrayRef<uint8_t> IntelPTThreadTrace::GetAuxBuffer() const {
#ifndef PERF_ATTR_SIZE_VER5
  llvm_unreachable("Intel PT Linux perf event not supported");
#else
  return MutableArrayRef<uint8_t>(m_mmap_aux.get(), m_mmap_meta->aux_size);
#endif
}

Expected<ArrayRef<uint8_t>> IntelPTThreadTrace::GetCPUInfo() {
  static llvm::Optional<std::vector<uint8_t>> cpu_info;
  if (!cpu_info) {
    auto buffer_or_error = getProcFile("cpuinfo");
    if (!buffer_or_error)
      return Status(buffer_or_error.getError()).ToError();
    MemoryBuffer &buffer = **buffer_or_error;
    cpu_info = std::vector<uint8_t>(
        reinterpret_cast<const uint8_t *>(buffer.getBufferStart()),
        reinterpret_cast<const uint8_t *>(buffer.getBufferEnd()));
  }
  return *cpu_info;
}

llvm::Expected<IntelPTThreadTraceUP>
IntelPTThreadTrace::Create(lldb::pid_t pid, lldb::tid_t tid,
                           size_t buffer_size) {
  IntelPTThreadTraceUP thread_trace_up(new IntelPTThreadTrace());

  if (llvm::Error err = thread_trace_up->StartTrace(pid, tid, buffer_size))
    return std::move(err);

  return std::move(thread_trace_up);
}

Expected<std::vector<uint8_t>>
IntelPTThreadTrace::GetIntelPTBuffer(size_t offset, size_t size) const {
  std::vector<uint8_t> data(size, 0);
  MutableArrayRef<uint8_t> buffer_ref(data);
  Status error = ReadPerfTraceAux(buffer_ref, 0);
  if (error.Fail())
    return error.ToError();
  return data;
}

Status
IntelPTThreadTrace::ReadPerfTraceAux(llvm::MutableArrayRef<uint8_t> &buffer,
                                     size_t offset) const {
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
IntelPTThreadTrace::ReadPerfTraceData(llvm::MutableArrayRef<uint8_t> &buffer,
                                      size_t offset) const {
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

void IntelPTThreadTrace::ReadCyclicBuffer(llvm::MutableArrayRef<uint8_t> &dst,
                                          llvm::MutableArrayRef<uint8_t> src,
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

TraceThreadState IntelPTThreadTrace::GetState() const {
  return {static_cast<int64_t>(m_tid),
          {TraceBinaryData{"threadTraceBuffer",
                           static_cast<int64_t>(GetTraceBufferSize())}}};
}

/// IntelPTThreadTraceCollection

bool IntelPTThreadTraceCollection::TracesThread(lldb::tid_t tid) const {
  return m_thread_traces.count(tid);
}

Error IntelPTThreadTraceCollection::TraceStop(lldb::tid_t tid) {
  auto it = m_thread_traces.find(tid);
  if (it == m_thread_traces.end())
    return createStringError(inconvertibleErrorCode(),
                             "Thread %" PRIu64 " not currently traced", tid);
  m_total_buffer_size -= it->second->GetTraceBufferSize();
  m_thread_traces.erase(tid);
  return Error::success();
}

Error IntelPTThreadTraceCollection::TraceStart(
    lldb::tid_t tid, const TraceIntelPTStartRequest &request) {
  if (TracesThread(tid))
    return createStringError(inconvertibleErrorCode(),
                             "Thread %" PRIu64 " already traced", tid);

  Expected<IntelPTThreadTraceUP> trace_up =
      IntelPTThreadTrace::Create(m_pid, tid, request.threadBufferSize);
  if (!trace_up)
    return trace_up.takeError();

  m_total_buffer_size += (*trace_up)->GetTraceBufferSize();
  m_thread_traces.try_emplace(tid, std::move(*trace_up));
  return Error::success();
}

size_t IntelPTThreadTraceCollection::GetTotalBufferSize() const {
  return m_total_buffer_size;
}

std::vector<TraceThreadState>
IntelPTThreadTraceCollection::GetThreadStates() const {
  std::vector<TraceThreadState> states;
  for (const auto &it : m_thread_traces)
    states.push_back(it.second->GetState());
  return states;
}

Expected<const IntelPTThreadTrace &>
IntelPTThreadTraceCollection::GetTracedThread(lldb::tid_t tid) const {
  auto it = m_thread_traces.find(tid);
  if (it == m_thread_traces.end())
    return createStringError(inconvertibleErrorCode(),
                             "Thread %" PRIu64 " not currently traced", tid);
  return *it->second.get();
}

void IntelPTThreadTraceCollection::Clear() {
  m_thread_traces.clear();
  m_total_buffer_size = 0;
}

/// IntelPTProcessTrace

bool IntelPTProcessTrace::TracesThread(lldb::tid_t tid) const {
  return m_thread_traces.TracesThread(tid);
}

Error IntelPTProcessTrace::TraceStop(lldb::tid_t tid) {
  return m_thread_traces.TraceStop(tid);
}

Error IntelPTProcessTrace::TraceStart(lldb::tid_t tid) {
  if (m_thread_traces.GetTotalBufferSize() + m_tracing_params.threadBufferSize >
      static_cast<size_t>(*m_tracing_params.processBufferSizeLimit))
    return createStringError(
        inconvertibleErrorCode(),
        "Thread %" PRIu64 " can't be traced as the process trace size limit "
        "has been reached. Consider retracing with a higher "
        "limit.",
        tid);

  return m_thread_traces.TraceStart(tid, m_tracing_params);
}

const IntelPTThreadTraceCollection &
IntelPTProcessTrace::GetThreadTraces() const {
  return m_thread_traces;
}

/// IntelPTManager

Error IntelPTManager::TraceStop(lldb::tid_t tid) {
  if (IsProcessTracingEnabled() && m_process_trace->TracesThread(tid))
    return m_process_trace->TraceStop(tid);
  return m_thread_traces.TraceStop(tid);
}

Error IntelPTManager::TraceStop(const TraceStopRequest &request) {
  if (request.IsProcessTracing()) {
    if (!IsProcessTracingEnabled()) {
      return createStringError(inconvertibleErrorCode(),
                               "Process not currently traced");
    }
    ClearProcessTracing();
    return Error::success();
  } else {
    Error error = Error::success();
    for (int64_t tid : *request.tids)
      error = joinErrors(std::move(error),
                         TraceStop(static_cast<lldb::tid_t>(tid)));
    return error;
  }
}

Error IntelPTManager::TraceStart(
    const TraceIntelPTStartRequest &request,
    const std::vector<lldb::tid_t> &process_threads) {
  if (request.IsProcessTracing()) {
    if (IsProcessTracingEnabled()) {
      return createStringError(
          inconvertibleErrorCode(),
          "Process currently traced. Stop process tracing first");
    }
    m_process_trace = IntelPTProcessTrace(m_pid, request);

    Error error = Error::success();
    for (lldb::tid_t tid : process_threads)
      error = joinErrors(std::move(error), m_process_trace->TraceStart(tid));
    return error;
  } else {
    Error error = Error::success();
    for (int64_t tid : *request.tids)
      error = joinErrors(std::move(error),
                         m_thread_traces.TraceStart(tid, request));
    return error;
  }
}

Error IntelPTManager::OnThreadCreated(lldb::tid_t tid) {
  if (!IsProcessTracingEnabled())
    return Error::success();
  return m_process_trace->TraceStart(tid);
}

Error IntelPTManager::OnThreadDestroyed(lldb::tid_t tid) {
  if (IsProcessTracingEnabled() && m_process_trace->TracesThread(tid))
    return m_process_trace->TraceStop(tid);
  else if (m_thread_traces.TracesThread(tid))
    return m_thread_traces.TraceStop(tid);
  return Error::success();
}

Expected<json::Value> IntelPTManager::GetState() const {
  Expected<ArrayRef<uint8_t>> cpu_info = IntelPTThreadTrace::GetCPUInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  TraceGetStateResponse state;
  state.processBinaryData.push_back(
      {"cpuInfo", static_cast<int64_t>(cpu_info->size())});

  std::vector<TraceThreadState> thread_states =
      m_thread_traces.GetThreadStates();
  state.tracedThreads.insert(state.tracedThreads.end(), thread_states.begin(),
                             thread_states.end());

  if (IsProcessTracingEnabled()) {
    thread_states = m_process_trace->GetThreadTraces().GetThreadStates();
    state.tracedThreads.insert(state.tracedThreads.end(), thread_states.begin(),
                               thread_states.end());
  }
  return toJSON(state);
}

Expected<const IntelPTThreadTrace &>
IntelPTManager::GetTracedThread(lldb::tid_t tid) const {
  if (IsProcessTracingEnabled() && m_process_trace->TracesThread(tid))
    return m_process_trace->GetThreadTraces().GetTracedThread(tid);
  return m_thread_traces.GetTracedThread(tid);
}

Expected<std::vector<uint8_t>>
IntelPTManager::GetBinaryData(const TraceGetBinaryDataRequest &request) const {
  if (request.kind == "threadTraceBuffer") {
    if (Expected<const IntelPTThreadTrace &> trace =
            GetTracedThread(*request.tid))
      return trace->GetIntelPTBuffer(request.offset, request.size);
    else
      return trace.takeError();
  } else if (request.kind == "cpuInfo") {
    return IntelPTThreadTrace::GetCPUInfo();
  }
  return createStringError(inconvertibleErrorCode(),
                           "Unsuported trace binary data kind: %s",
                           request.kind.c_str());
}

void IntelPTManager::ClearProcessTracing() { m_process_trace = None; }

bool IntelPTManager::IsSupported() { return (bool)GetOSEventType(); }

bool IntelPTManager::IsProcessTracingEnabled() const {
  return (bool)m_process_trace;
}

void IntelPTManager::Clear() {
  ClearProcessTracing();
  m_thread_traces.Clear();
}
