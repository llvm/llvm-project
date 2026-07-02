//===-- OutputRedirector.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "OutputRedirector.h"
#include "DAP.h"
#include "DAPLog.h"
#include "lldb/Host/File.h"
#include "lldb/Host/MainLoopBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstring>
#include <system_error>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace llvm;
using namespace lldb_private;

namespace lldb_dap {

int OutputRedirector::kInvalidDescriptor = -1;

OutputRedirector::OutputRedirector()
    : m_fd(kInvalidDescriptor), m_original_fd(kInvalidDescriptor),
      m_restore_fd(kInvalidDescriptor) {}

Expected<int> OutputRedirector::GetWriteFileDescriptor() {
  if (m_fd == kInvalidDescriptor)
    return createStringError(std::errc::bad_file_descriptor,
                             "write handle is not open for writing");
  return m_fd;
}

Error OutputRedirector::RedirectTo(MainLoopBase &loop, std::FILE *file_override,
                                   std::function<void(StringRef)> callback,
                                   Log &log) {
  assert(m_fd == kInvalidDescriptor && "OutputRedirector already started.");
  int new_fd[2];

#if defined(_WIN32)
  if (::_pipe(new_fd, OutputBufferSize, O_TEXT) == -1) {
#else
  if (::pipe(new_fd) == -1) {
#endif
    int error = errno;
    return createStringError(inconvertibleErrorCode(),
                             "Couldn't create new pipe %s", strerror(error));
  }

  int read_fd = new_fd[0];
  m_fd = new_fd[1];

  m_read_obj = std::make_shared<NativeFile>(read_fd, File::eOpenOptionReadOnly,
                                            NativeFile::Owned);

  if (file_override) {
    int override_fd = fileno(file_override);

    // Backup the FD to restore once redirection is complete.
    m_original_fd = override_fd;
    m_restore_fd = dup(override_fd);

    // Override the existing fd the new write end of the pipe.
    if (::dup2(m_fd, override_fd) == -1)
      return llvm::errorCodeToError(llvm::errnoAsErrorCode());
  }

  auto read_callback = [callback = std::move(callback), this,
                        &log](MainLoopBase &) {
    std::array<char, OutputBufferSize> buffer;
    size_t num_bytes = buffer.size();

    const Status status = m_read_obj->Read(buffer.data(), num_bytes);
    if (status.Fail()) {
      DAP_LOG_ERROR(log, status.ToError(),
                    "OutputRedirector read failed (handle {1}): error: {0}",
                    m_read_obj->GetWaitableHandle());
      m_read_handle.reset();
      return;
    }
    if (num_bytes == 0) { // EOF
      m_read_handle.reset();
      return;
    }

    const llvm::StringRef data(buffer.data(), num_bytes);
    callback(data);
  };

  Status status;
  m_read_handle =
      loop.RegisterReadObject(m_read_obj, std::move(read_callback), status);

  return status.takeError();
}

void OutputRedirector::Stop() {
  // Stop polling.
  m_read_handle.reset();
  m_read_obj.reset();

  if (m_fd != kInvalidDescriptor) {
    int fd = m_fd;
    m_fd = kInvalidDescriptor;
    ::close(fd);

    // Restore the fd back to its original state since we stopped the
    // redirection.
    if (m_restore_fd != kInvalidDescriptor &&
        m_original_fd != kInvalidDescriptor) {
      int restore_fd = m_restore_fd;
      m_restore_fd = kInvalidDescriptor;
      int original_fd = m_original_fd;
      m_original_fd = kInvalidDescriptor;
      ::dup2(restore_fd, original_fd);
      ::close(restore_fd);
    }
  }
}

} // namespace lldb_dap
