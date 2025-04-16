//===-- OutputRedirector.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "OutputRedirector.h"
#include "DAP.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <system_error>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

using lldb_private::Pipe;
using llvm::createStringError;
using llvm::Error;
using llvm::Expected;
using llvm::inconvertibleErrorCode;
using llvm::StringRef;

namespace lldb_dap {

int OutputRedirector::kInvalidDescriptor = -1;

OutputRedirector::OutputRedirector() : m_fd(kInvalidDescriptor) {}

Expected<int> OutputRedirector::GetWriteFileDescriptor() {
  if (m_fd == kInvalidDescriptor)
    return createStringError(std::errc::bad_file_descriptor,
                             "write handle is not open for writing");
  return m_fd;
}

Error OutputRedirector::RedirectTo(std::function<void(StringRef)> callback) {
  assert(m_fd == kInvalidDescriptor && "Output readirector already started.");
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
  m_forwarder = std::thread([this, callback, read_fd]() {
    char buffer[OutputBufferSize];
    while (!m_stopped) {
      ssize_t bytes_count = ::read(read_fd, &buffer, sizeof(buffer));
      // EOF detected.
      if (bytes_count == 0)
        break;
      if (bytes_count == -1) {
        // Skip non-fatal errors.
        if (errno == EAGAIN || errno == EINTR || errno == EWOULDBLOCK)
          continue;
        break;
      }

      callback(StringRef(buffer, bytes_count));
    }
    ::close(read_fd);
  });

  return Error::success();
}

void OutputRedirector::Stop() {
  m_stopped = true;

  if (m_fd != kInvalidDescriptor) {
    int fd = m_fd;
    m_fd = kInvalidDescriptor;
    // Closing the pipe may not be sufficient to wake up the thread in case the
    // write descriptor is duplicated (to stdout/err or to another process).
    // Write a null byte to ensure the read call returns.
    char buf[] = "\0";
    ::write(fd, buf, sizeof(buf));
    ::close(fd);
    m_forwarder.join();
  }
}

} // namespace lldb_dap
