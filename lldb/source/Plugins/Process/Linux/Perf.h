//===-- Perf.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains a thin wrapper of the perf_event_open API
/// and classes to handle the destruction of file descriptors
/// and mmap pointers.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_LINUX_PERF_H
#define LLDB_SOURCE_PLUGINS_PROCESS_LINUX_PERF_H

#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"
#include "lldb/lldb-types.h"

#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <linux/perf_event.h>

namespace lldb_private {
namespace process_linux {
namespace resource_handle {

/// Custom deleter for the pointer returned by \a mmap.
///
/// This functor type is provided to \a unique_ptr to properly
/// unmap the region at destruction time.
class MmapDeleter {
public:
  /// Construct new \a MmapDeleter.
  ///
  /// \param[in] bytes
  ///   Size of the mmap'ed region in bytes.
  MmapDeleter(size_t bytes = 0) : m_bytes(bytes) {}

  /// Unmap the mmap'ed region.
  ///
  /// If \a m_bytes==0 or \a ptr==nullptr, nothing is unmmapped.
  ///
  /// \param[in] ptr
  ///   pointer to the region to be unmmapped.
  void operator()(void *ptr);

private:
  /// Size of the mmap'ed region, in bytes, to be unmapped.
  size_t m_bytes;
};

/// Custom deleter for a file descriptor.
///
/// This functor type is provided to \a unique_ptr to properly release
/// the resources associated with the file descriptor at destruction time.
class FileDescriptorDeleter {
public:
  /// Close and free the memory associated with the file descriptor pointer.
  ///
  /// Effectively a no-op if \a ptr==nullptr or \a*ptr==-1.
  ///
  /// \param[in] ptr
  ///   Pointer to the file descriptor.
  void operator()(long *ptr);
};

using FileDescriptorUP =
    std::unique_ptr<long, resource_handle::FileDescriptorDeleter>;
using MmapUP = std::unique_ptr<void, resource_handle::MmapDeleter>;

} // namespace resource_handle

/// Read data from a cyclic buffer
///
/// \param[in] [out] buf
///     Destination buffer, the buffer will be truncated to written size.
///
/// \param[in] src
///     Source buffer which must be a cyclic buffer.
///
/// \param[in] src_cyc_index
///     The index pointer (start of the valid data in the cyclic
///     buffer).
///
/// \param[in] offset
///     The offset to begin reading the data in the cyclic buffer.
void ReadCyclicBuffer(llvm::MutableArrayRef<uint8_t> &dst,
                      llvm::ArrayRef<uint8_t> src, size_t src_cyc_index,
                      size_t offset);

/// Thin wrapper of the perf_event_open API.
///
/// Exposes the metadata page and data and aux buffers of a perf event.
/// Handles the management of the event's file descriptor and mmap'ed
/// regions.
class PerfEvent {
public:
  /// Create a new performance monitoring event via the perf_event_open syscall.
  ///
  /// The parameters are directly forwarded to a perf_event_open syscall,
  /// for additional information on the parameters visit
  /// https://man7.org/linux/man-pages/man2/perf_event_open.2.html.
  ///
  /// \param[in] attr
  ///     Configuration information for the event.
  ///
  /// \param[in] pid
  ///     The process or thread to be monitored by the event. If \b None, then
  ///     all processes and threads are monitored.
  ///
  /// \param[in] cpu
  ///     The cpu to be monitored by the event. If \b None, then all cpus are
  ///     monitored.
  ///
  /// \param[in] group_fd
  ///     File descriptor of the group leader. If \b None, then this perf_event
  ///     doesn't belong to a preexisting group.
  ///
  /// \param[in] flags
  ///     Bitmask of additional configuration flags.
  ///
  /// \return
  ///     If the perf_event_open syscall was successful, a minimal \a PerfEvent
  ///     instance, or an \a llvm::Error otherwise.
  static llvm::Expected<PerfEvent> Init(perf_event_attr &attr,
                                        llvm::Optional<lldb::pid_t> pid,
                                        llvm::Optional<lldb::core_id_t> cpu,
                                        llvm::Optional<int> group_fd,
                                        unsigned long flags);

  /// Create a new performance monitoring event via the perf_event_open syscall
  /// with "default" values for the cpu, group_fd and flags arguments.
  ///
  /// Convenience method to be used when the perf event requires minimal
  /// configuration. It handles the default values of all other arguments.
  ///
  /// \param[in] attr
  ///     Configuration information for the event.
  ///
  /// \param[in] pid
  ///     The process or thread to be monitored by the event. If \b None, then
  ///     all threads and processes are monitored.
  static llvm::Expected<PerfEvent>
  Init(perf_event_attr &attr, llvm::Optional<lldb::pid_t> pid,
       llvm::Optional<lldb::core_id_t> core = llvm::None);

  /// Mmap the metadata page and the data and aux buffers of the perf event and
  /// expose them through \a PerfEvent::GetMetadataPage() , \a
  /// PerfEvent::GetDataBuffer() and \a PerfEvent::GetAuxBuffer().
  ///
  /// This uses mmap underneath, which means that the number of pages mmap'ed
  /// must be less than the actual data available by the kernel. The metadata
  /// page is always mmap'ed.
  ///
  /// Mmap is needed because the underlying data might be changed by the kernel
  /// dynamically.
  ///
  /// \param[in] num_data_pages
  ///     Number of pages in the data buffer to mmap, must be a power of 2.
  ///     A value of 0 is useful for "dummy" events that only want to access
  ///     the metadata, \a perf_event_mmap_page, or the aux buffer.
  ///
  /// \param[in] num_aux_pages
  ///     Number of pages in the aux buffer to mmap, must be a power of 2.
  ///     A value of 0 effectively is a no-op and no data is mmap'ed for this
  ///     buffer.
  ///
  /// \return
  ///   \a llvm::Error::success if the mmap operations succeeded,
  ///   or an \a llvm::Error otherwise.
  llvm::Error MmapMetadataAndBuffers(size_t num_data_pages,
                                     size_t num_aux_pages);

  /// Get the file descriptor associated with the perf event.
  long GetFd() const;

  /// Get the metadata page from the data section's mmap buffer.
  ///
  /// The metadata page is always mmap'ed, even when \a num_data_pages is 0.
  ///
  /// This should be called only after \a PerfEvent::MmapMetadataAndBuffers,
  /// otherwise a failure might happen.
  ///
  /// \return
  ///   The data section's \a perf_event_mmap_page.
  perf_event_mmap_page &GetMetadataPage() const;

  /// Get the data buffer from the data section's mmap buffer.
  ///
  /// The data buffer is the region of the data section's mmap buffer where
  /// perf sample data is located.
  ///
  /// This should be called only after \a PerfEvent::MmapMetadataAndBuffers,
  /// otherwise a failure might happen.
  ///
  /// \return
  ///   \a ArrayRef<uint8_t> extending \a data_size bytes from \a data_offset.
  llvm::ArrayRef<uint8_t> GetDataBuffer() const;

  /// Get the AUX buffer.
  ///
  /// AUX buffer is a region for high-bandwidth data streams
  /// such as IntelPT. This is separate from the metadata and data buffer.
  ///
  /// This should be called only after \a PerfEvent::MmapMetadataAndBuffers,
  /// otherwise a failure might happen.
  ///
  /// \return
  ///   \a ArrayRef<uint8_t> extending \a aux_size bytes from \a aux_offset.
  llvm::ArrayRef<uint8_t> GetAuxBuffer() const;

  /// Use the ioctl API to disable the perf event. This doesn't terminate the
  /// perf event.
  ///
  /// \return
  ///   An Error if the perf event couldn't be disabled.
  llvm::Error DisableWithIoctl() const;

  /// Use the ioctl API to enable the perf event.
  ///
  /// \return
  ///   An Error if the perf event couldn't be enabled.
  llvm::Error EnableWithIoctl() const;

private:
  /// Create new \a PerfEvent.
  ///
  /// \param[in] fd
  ///   File descriptor of the perf event.
  PerfEvent(long fd)
      : m_fd(new long(fd), resource_handle::FileDescriptorDeleter()),
        m_metadata_data_base(), m_aux_base() {}

  /// Wrapper for \a mmap to provide custom error messages.
  ///
  /// The parameters are directly forwarded to a \a mmap syscall,
  /// for information on the parameters visit
  /// https://man7.org/linux/man-pages/man2/mmap.2.html.
  ///
  /// The value of \a GetFd() is passed as the \a fd argument to \a mmap.
  llvm::Expected<resource_handle::MmapUP> DoMmap(void *addr, size_t length,
                                                 int prot, int flags,
                                                 long int offset,
                                                 llvm::StringRef buffer_name);

  /// Mmap the data buffer of the perf event.
  ///
  /// \param[in] num_data_pages
  ///     Number of pages in the data buffer to mmap, must be a power of 2.
  ///     A value of 0 is useful for "dummy" events that only want to access
  ///     the metadata, \a perf_event_mmap_page, or the aux buffer.
  llvm::Error MmapMetadataAndDataBuffer(size_t num_data_pages);

  /// Mmap the aux buffer of the perf event.
  ///
  /// \param[in] num_aux_pages
  ///   Number of pages in the aux buffer to mmap, must be a power of 2.
  ///   A value of 0 effectively is a no-op and no data is mmap'ed for this
  ///   buffer.
  llvm::Error MmapAuxBuffer(size_t num_aux_pages);

  /// The file descriptor representing the perf event.
  resource_handle::FileDescriptorUP m_fd;
  /// Metadata page and data section where perf samples are stored.
  resource_handle::MmapUP m_metadata_data_base;
  /// AUX buffer is a separate region for high-bandwidth data streams
  /// such as IntelPT.
  resource_handle::MmapUP m_aux_base;
};

/// Load \a PerfTscConversionParameters from \a perf_event_mmap_page, if
/// available.
llvm::Expected<LinuxPerfZeroTscConversion> LoadPerfTscConversionParameters();

} // namespace process_linux
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_LINUX_PERF_H
