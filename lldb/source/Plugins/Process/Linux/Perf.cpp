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
  if (Error mmap_err =
          perf_event->MmapMetadataAndBuffers(/*num_data_pages=*/0,
                                             /*num_aux_pages=*/0,
                                             /*data_buffer_write=*/false))
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
  return PerfEvent(fd, attr.disabled ? CollectionState::Disabled
                                     : CollectionState::Enabled);
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
  auto mmap_result = ::mmap(addr, length, prot, flags, GetFd(), offset);

  if (mmap_result == MAP_FAILED) {
    std::string err_msg =
        llvm::formatv("perf event mmap allocation failed for {0}: {1}",
                      buffer_name, std::strerror(errno));
    return createStringError(inconvertibleErrorCode(), err_msg);
  }
  return resource_handle::MmapUP(mmap_result, length);
}

llvm::Error PerfEvent::MmapMetadataAndDataBuffer(size_t num_data_pages,
                                                 bool data_buffer_write) {
  size_t mmap_size = (num_data_pages + 1) * getpagesize();
  if (Expected<resource_handle::MmapUP> mmap_metadata_data = DoMmap(
          nullptr, mmap_size, PROT_READ | (data_buffer_write ? PROT_WRITE : 0),
          MAP_SHARED, 0, "metadata and data buffer")) {
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
                                              size_t num_aux_pages,
                                              bool data_buffer_write) {
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
  if (Error err = MmapMetadataAndDataBuffer(num_data_pages, data_buffer_write))
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

Expected<std::vector<uint8_t>>
PerfEvent::ReadFlushedOutDataCyclicBuffer(size_t offset, size_t size) {
  CollectionState previous_state = m_collection_state;
  if (Error err = DisableWithIoctl())
    return std::move(err);

  /**
   * The data buffer and aux buffer have different implementations
   * with respect to their definition of head pointer. In the case
   * of Aux data buffer the head always wraps around the aux buffer
   * and we don't need to care about it, whereas the data_head keeps
   * increasing and needs to be wrapped by modulus operator
   */
  perf_event_mmap_page &mmap_metadata = GetMetadataPage();

  ArrayRef<uint8_t> data = GetDataBuffer();
  uint64_t data_head = mmap_metadata.data_head;
  uint64_t data_size = mmap_metadata.data_size;
  std::vector<uint8_t> output;
  output.reserve(size);

  if (data_head > data_size) {
    uint64_t actual_data_head = data_head % data_size;
    // The buffer has wrapped
    for (uint64_t i = actual_data_head + offset;
         i < data_size && output.size() < size; i++)
      output.push_back(data[i]);

    // We need to find the starting position for the left part if the offset was
    // too big
    uint64_t left_part_start = actual_data_head + offset >= data_size
                                   ? actual_data_head + offset - data_size
                                   : 0;
    for (uint64_t i = left_part_start;
         i < actual_data_head && output.size() < size; i++)
      output.push_back(data[i]);
  } else {
    for (uint64_t i = offset; i < data_head && output.size() < size; i++)
      output.push_back(data[i]);
  }

  if (previous_state == CollectionState::Enabled) {
    if (Error err = EnableWithIoctl())
      return std::move(err);
  }

  if (output.size() != size)
    return createStringError(inconvertibleErrorCode(),
                             formatv("Requested {0} bytes of perf_event data "
                                     "buffer but only {1} are available",
                                     size, output.size()));

  return output;
}

Expected<std::vector<uint8_t>>
PerfEvent::ReadFlushedOutAuxCyclicBuffer(size_t offset, size_t size) {
  CollectionState previous_state = m_collection_state;
  if (Error err = DisableWithIoctl())
    return std::move(err);

  perf_event_mmap_page &mmap_metadata = GetMetadataPage();

  ArrayRef<uint8_t> data = GetAuxBuffer();
  uint64_t aux_head = mmap_metadata.aux_head;
  uint64_t aux_size = mmap_metadata.aux_size;
  std::vector<uint8_t> output;
  output.reserve(size);

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

  for (uint64_t i = aux_head + offset; i < aux_size && output.size() < size;
       i++)
    output.push_back(data[i]);

  // We need to find the starting position for the left part if the offset was
  // too big
  uint64_t left_part_start =
      aux_head + offset >= aux_size ? aux_head + offset - aux_size : 0;
  for (uint64_t i = left_part_start; i < aux_head && output.size() < size; i++)
    output.push_back(data[i]);

  if (previous_state == CollectionState::Enabled) {
    if (Error err = EnableWithIoctl())
      return std::move(err);
  }

  if (output.size() != size)
    return createStringError(inconvertibleErrorCode(),
                             formatv("Requested {0} bytes of perf_event aux "
                                     "buffer but only {1} are available",
                                     size, output.size()));

  return output;
}

Error PerfEvent::DisableWithIoctl() {
  if (m_collection_state == CollectionState::Disabled)
    return Error::success();

  if (ioctl(*m_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0)
    return createStringError(inconvertibleErrorCode(),
                             "Can't disable perf event. %s",
                             std::strerror(errno));

  m_collection_state = CollectionState::Disabled;
  return Error::success();
}

Error PerfEvent::EnableWithIoctl() {
  if (m_collection_state == CollectionState::Enabled)
    return Error::success();

  if (ioctl(*m_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
    return createStringError(inconvertibleErrorCode(),
                             "Can't enable perf event. %s",
                             std::strerror(errno));

  m_collection_state = CollectionState::Enabled;
  return Error::success();
}

size_t PerfEvent::GetEffectiveDataBufferSize() const {
  perf_event_mmap_page &mmap_metadata = GetMetadataPage();
  if (mmap_metadata.data_head < mmap_metadata.data_size)
    return mmap_metadata.data_head;
  else
    return mmap_metadata.data_size; // The buffer has wrapped.
}
