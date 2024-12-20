//===-- OutputRedirector.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "llvm/Support/Error.h"
#include <system_error>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include "DAP.h"
#include "OutputRedirector.h"
#include "llvm/ADT/StringRef.h"

using lldb_private::Pipe;
using lldb_private::Status;
using llvm::createStringError;
using llvm::Error;
using llvm::Expected;
using llvm::StringRef;

namespace lldb_dap {

Expected<int> OutputRedirector::GetWriteFileDescriptor() {
  if (!m_pipe.CanWrite())
    return createStringError(std::errc::bad_file_descriptor,
                             "write handle is not open for writing");
  return m_pipe.GetWriteFileDescriptor();
}

Error OutputRedirector::RedirectTo(std::function<void(StringRef)> callback) {
  Status status = m_pipe.CreateNew(/*child_process_inherit=*/false);
  if (status.Fail())
    return status.takeError();

  m_forwarder = std::thread([this, callback]() {
    char buffer[OutputBufferSize];
    while (m_pipe.CanRead() && !m_stopped) {
      size_t bytes_read;
      Status status = m_pipe.Read(&buffer, sizeof(buffer), bytes_read);
      if (status.Fail())
        continue;

      // EOF detected
      if (bytes_read == 0)
        break;

      callback(StringRef(buffer, bytes_read));
    }
  });

  return Error::success();
}

void OutputRedirector::Stop() {
  m_stopped = true;

  if (m_pipe.CanWrite()) {
    // If the fd is waiting for input and is closed it may not return from the
    // current select/poll/kqueue/etc. asyncio wait operation. Write a null byte
    // to ensure the read fd wakes to detect the closed FD.
    char buf[] = "\0";
    size_t bytes_written;
    m_pipe.Write(buf, sizeof(buf), bytes_written);
    m_pipe.CloseWriteFileDescriptor();
    m_forwarder.join();
  }
}

} // namespace lldb_dap
