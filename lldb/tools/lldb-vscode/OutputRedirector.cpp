//===-- OutputRedirector.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "OutputRedirector.h"

using namespace llvm;

namespace lldb_vscode {

Error RedirectFd(int fd, std::function<void(llvm::StringRef)> callback) {
#if !defined(_WIN32)
  int new_fd[2];
  if (pipe(new_fd) == -1) {
    int error = errno;
    return createStringError(inconvertibleErrorCode(),
                             "Couldn't create new pipe for fd %d. %s", fd,
                             strerror(error));
  }

  if (dup2(new_fd[1], fd) == -1) {
    int error = errno;
    return createStringError(inconvertibleErrorCode(),
                             "Couldn't override the fd %d. %s", fd,
                             strerror(error));
  }

  int read_fd = new_fd[0];
  std::thread t([read_fd, callback]() {
    char buffer[4096];
    while (true) {
      ssize_t bytes_count = read(read_fd, &buffer, sizeof(buffer));
      if (bytes_count == 0)
        return;
      if (bytes_count == -1) {
        if (errno == EAGAIN || errno == EINTR)
          continue;
        break;
      }
      callback(StringRef(buffer, bytes_count).str());
    }
  });
  t.detach();
#endif
  return Error::success();
}

} // namespace lldb_vscode
