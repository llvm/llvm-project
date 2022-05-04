//===-- Perf.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Perf.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/linux/Support.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

void lldb_private::process_linux::ReadCyclicBuffer(
    llvm::MutableArrayRef<uint8_t> &dst, llvm::ArrayRef<uint8_t> src,
    size_t src_cyc_index, size_t offset) {

  Log *log = GetLog(POSIXLog::Trace);

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

  llvm::SmallVector<ArrayRef<uint8_t>, 2> parts = {
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

Expected<LinuxPerfZeroTscConversion>
lldb_private::process_linux::LoadPerfTscConversionParameters() {
  lldb::pid_t pid = getpid();
  perf_event_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);
  attr.type = PERF_TYPE_SOFTWARE;
  attr.config = PERF_COUNT_SW_DUMMY;

  Expected<PerfEvent> perf_event = PerfEvent::Init(attr, pid);
  if (!perf_event)
    return perf_event.takeError();
  if (Error mmap_err = perf_event->MmapMetadataAndBuffers(/*num_data_pages*/ 0,
                                                          /*num_aux_pages*/ 0))
    return std::move(mmap_err);

  perf_event_mmap_page &mmap_metada = perf_event->GetMetadataPage();
  if (mmap_metada.cap_user_time && mmap_metada.cap_user_time_zero) {
    return LinuxPerfZeroTscConversion{
        mmap_metada.time_mult, mmap_metada.time_shift, mmap_metada.time_zero};
  } else {
    auto err_cap =
        !mmap_metada.cap_user_time ? "cap_user_time" : "cap_user_time_zero";
    std::string err_msg =
        llvm::formatv("Can't get TSC to real time conversion values. "
                      "perf_event capability '{0}' not supported.",
                      err_cap);
    return llvm::createStringError(llvm::inconvertibleErrorCode(), err_msg);
  }
}

void resource_handle::MmapDeleter::operator()(void *ptr) {
  if (m_bytes && ptr != nullptr)
    munmap(ptr, m_bytes);
}

void resource_handle::FileDescriptorDeleter::operator()(long *ptr) {
  if (ptr == nullptr)
    return;
  if (*ptr == -1)
    return;
  close(*ptr);
  std::default_delete<long>()(ptr);
}

llvm::Expected<PerfEvent> PerfEvent::Init(perf_event_attr &attr,
                                          Optional<lldb::pid_t> pid,
                                          Optional<lldb::core_id_t> cpu,
                                          Optional<int> group_fd,
                                          unsigned long flags) {
  errno = 0;
  long fd = syscall(SYS_perf_event_open, &attr, pid.getValueOr(-1),
                    cpu.getValueOr(-1), group_fd.getValueOr(-1), flags);
  if (fd == -1) {
    std::string err_msg =
        llvm::formatv("perf event syscall failed: {0}", std::strerror(errno));
    return llvm::createStringError(llvm::inconvertibleErrorCode(), err_msg);
  }
  return PerfEvent{fd};
}

llvm::Expected<PerfEvent> PerfEvent::Init(perf_event_attr &attr,
                                          Optional<lldb::pid_t> pid,
                                          Optional<lldb::core_id_t> cpu) {
  return Init(attr, pid, cpu, -1, 0);
}

llvm::Expected<resource_handle::MmapUP>
PerfEvent::DoMmap(void *addr, size_t length, int prot, int flags,
                  long int offset, llvm::StringRef buffer_name) {
  errno = 0;
  auto mmap_result = ::mmap(nullptr, length, prot, flags, GetFd(), offset);

  if (mmap_result == MAP_FAILED) {
    std::string err_msg =
        llvm::formatv("perf event mmap allocation failed for {0}: {1}",
                      buffer_name, std::strerror(errno));
    return createStringError(inconvertibleErrorCode(), err_msg);
  }
  return resource_handle::MmapUP(mmap_result, length);
}

llvm::Error PerfEvent::MmapMetadataAndDataBuffer(size_t num_data_pages) {
  size_t mmap_size = (num_data_pages + 1) * getpagesize();
  if (Expected<resource_handle::MmapUP> mmap_metadata_data =
          DoMmap(nullptr, mmap_size, PROT_WRITE, MAP_SHARED, 0,
                 "metadata and data buffer")) {
    m_metadata_data_base = std::move(mmap_metadata_data.get());
    return Error::success();
  } else
    return mmap_metadata_data.takeError();
}

llvm::Error PerfEvent::MmapAuxBuffer(size_t num_aux_pages) {
  if (num_aux_pages == 0)
    return Error::success();

  perf_event_mmap_page &metadata_page = GetMetadataPage();
  metadata_page.aux_offset =
      metadata_page.data_offset + metadata_page.data_size;
  metadata_page.aux_size = num_aux_pages * getpagesize();

  if (Expected<resource_handle::MmapUP> mmap_aux =
          DoMmap(nullptr, metadata_page.aux_size, PROT_READ, MAP_SHARED,
                 metadata_page.aux_offset, "aux buffer")) {
    m_aux_base = std::move(mmap_aux.get());
    return Error::success();
  } else
    return mmap_aux.takeError();
}

llvm::Error PerfEvent::MmapMetadataAndBuffers(size_t num_data_pages,
                                              size_t num_aux_pages) {
  if (num_data_pages != 0 && !isPowerOf2_64(num_data_pages))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::formatv("Number of data pages must be a power of 2, got: {0}",
                      num_data_pages));
  if (num_aux_pages != 0 && !isPowerOf2_64(num_aux_pages))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::formatv("Number of aux pages must be a power of 2, got: {0}",
                      num_aux_pages));
  if (Error err = MmapMetadataAndDataBuffer(num_data_pages))
    return err;
  if (Error err = MmapAuxBuffer(num_aux_pages))
    return err;
  return Error::success();
}

long PerfEvent::GetFd() const { return *(m_fd.get()); }

perf_event_mmap_page &PerfEvent::GetMetadataPage() const {
  return *reinterpret_cast<perf_event_mmap_page *>(m_metadata_data_base.get());
}

ArrayRef<uint8_t> PerfEvent::GetDataBuffer() const {
  perf_event_mmap_page &mmap_metadata = GetMetadataPage();
  return {reinterpret_cast<uint8_t *>(m_metadata_data_base.get()) +
              mmap_metadata.data_offset,
           static_cast<size_t>(mmap_metadata.data_size)};
}

ArrayRef<uint8_t> PerfEvent::GetAuxBuffer() const {
  perf_event_mmap_page &mmap_metadata = GetMetadataPage();
  return {reinterpret_cast<uint8_t *>(m_aux_base.get()),
           static_cast<size_t>(mmap_metadata.aux_size)};
}

Error PerfEvent::DisableWithIoctl() const {
  if (ioctl(*m_fd, PERF_EVENT_IOC_DISABLE) < 0)
    return createStringError(inconvertibleErrorCode(),
                             "Can't disable perf event. %s",
                             std::strerror(errno));
  return Error::success();
}

Error PerfEvent::EnableWithIoctl() const {
  if (ioctl(*m_fd, PERF_EVENT_IOC_ENABLE) < 0)
    return createStringError(inconvertibleErrorCode(),
                             "Can't disable perf event. %s",
                             std::strerror(errno));
  return Error::success();
}
