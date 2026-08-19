//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/FilePosix.h"

#include <cassert>
#include <climits>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Process.h"

using namespace lldb_private;

#if defined(__APPLE__)
// Darwin kernels only can read/write <= INT_MAX bytes.  Match the chunking
// limit used by NativeFileBase::Read/Write.
#define MAX_READ_SIZE INT_MAX
#define MAX_WRITE_SIZE INT_MAX
#endif

NativeFilePosix::NativeFilePosix(FILE *fh, OpenOptions options,
                                 bool transfer_ownership)
    : NativeFileBase(fh, options, transfer_ownership) {
#ifndef NDEBUG
  int fd = fileno(fh);
  if (fd != -1) {
    int required_mode = ConvertOpenOptionsForPOSIXOpen(options) & O_ACCMODE;
    int mode = fcntl(fd, F_GETFL);
    if (mode != -1) {
      mode &= O_ACCMODE;
      // Check that the file is open with a valid subset of the requested file
      // access mode, e.g. if we expected the file to be writable then ensure it
      // was opened with O_WRONLY or O_RDWR.
      assert(
          (required_mode == O_RDWR && mode == O_RDWR) ||
          (required_mode == O_RDONLY && (mode == O_RDWR || mode == O_RDONLY) ||
           (required_mode == O_WRONLY &&
            (mode == O_RDWR || mode == O_WRONLY))) &&
              "invalid file access mode");
    }
  }
#endif
}

NativeFilePosix::NativeFilePosix(int fd, OpenOptions options,
                                 bool transfer_ownership)
    : NativeFileBase(fd, options, transfer_ownership) {}

IOObject::WaitableHandle NativeFilePosix::GetWaitableHandle() {
  // POSIX poll/select/epoll work directly on file descriptors.
  return GetDescriptor();
}

Status NativeFilePosix::Sync() {
  Status error;
  if (ValueGuard descriptor_guard = DescriptorIsValid()) {
    if (llvm::sys::RetryAfterSignal(-1, ::fsync, m_descriptor) == -1)
      error = Status::FromErrno();
  } else {
    error = Status::FromErrorString("invalid file handle");
  }
  return error;
}

void NativeFilePosix::CalculateInteractiveAndTerminal() {
  const int fd = GetDescriptor();
  if (!File::DescriptorIsValid(fd)) {
    m_is_interactive = eLazyBoolNo;
    m_is_real_terminal = eLazyBoolNo;
    m_supports_colors = eLazyBoolNo;
    return;
  }
  m_is_interactive = eLazyBoolNo;
  m_is_real_terminal = eLazyBoolNo;
  if (isatty(fd)) {
    m_is_interactive = eLazyBoolYes;
    struct winsize window_size;
    if (::ioctl(fd, TIOCGWINSZ, &window_size) == 0) {
      if (window_size.ws_col > 0) {
        m_is_real_terminal = eLazyBoolYes;
        if (llvm::sys::Process::FileDescriptorHasColors(fd))
          m_supports_colors = eLazyBoolYes;
      }
    }
  }
}

Status NativeFilePosix::GetFileSpec(FileSpec &file_spec) const {
  Status error;
#ifdef F_GETPATH
  if (IsValid()) {
    char path[PATH_MAX];
    if (::fcntl(GetDescriptor(), F_GETPATH, path) == -1)
      error = Status::FromErrno();
    else
      file_spec.SetFile(path, FileSpec::Style::native);
  } else {
    error = Status::FromErrorString("invalid file handle");
  }
#elif defined(__linux__)
  char proc[64];
  char path[PATH_MAX];
  if (::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", GetDescriptor()) < 0)
    error = Status::FromErrorString("cannot resolve file descriptor");
  else {
    ssize_t len;
    if ((len = ::readlink(proc, path, sizeof(path) - 1)) == -1)
      error = Status::FromErrno();
    else {
      path[len] = '\0';
      file_spec.SetFile(path, FileSpec::Style::native);
    }
  }
#else
  error = Status::FromErrorString(
      "NativeFile::GetFileSpec is not supported on this platform");
#endif

  if (error.Fail())
    file_spec.Clear();
  return error;
}

Status NativeFilePosix::Read(void *buf, size_t &num_bytes, off_t &offset) {
  Status error;

#if defined(MAX_READ_SIZE)
  if (num_bytes > MAX_READ_SIZE) {
    uint8_t *p = (uint8_t *)buf;
    size_t bytes_left = num_bytes;
    // Init the num_bytes read to zero
    num_bytes = 0;

    while (bytes_left > 0) {
      size_t curr_num_bytes;
      if (bytes_left > MAX_READ_SIZE)
        curr_num_bytes = MAX_READ_SIZE;
      else
        curr_num_bytes = bytes_left;

      error = Read(p + num_bytes, curr_num_bytes, offset);

      // Update how many bytes were read
      num_bytes += curr_num_bytes;
      if (bytes_left < curr_num_bytes)
        bytes_left = 0;
      else
        bytes_left -= curr_num_bytes;

      if (error.Fail())
        break;
    }
    return error;
  }
#endif

  int fd = GetDescriptor();
  if (fd != kInvalidDescriptor) {
    ssize_t bytes_read =
        llvm::sys::RetryAfterSignal(-1, ::pread, fd, buf, num_bytes, offset);
    if (bytes_read < 0) {
      num_bytes = 0;
      error = Status::FromErrno();
    } else {
      offset += bytes_read;
      num_bytes = bytes_read;
    }
  } else {
    num_bytes = 0;
    error = Status::FromErrorString("invalid file handle");
  }
  return error;
}

Status NativeFilePosix::Write(const void *buf, size_t &num_bytes,
                              off_t &offset) {
  Status error;

#if defined(MAX_WRITE_SIZE)
  if (num_bytes > MAX_WRITE_SIZE) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t bytes_left = num_bytes;
    // Init the num_bytes written to zero
    num_bytes = 0;

    while (bytes_left > 0) {
      size_t curr_num_bytes;
      if (bytes_left > MAX_WRITE_SIZE)
        curr_num_bytes = MAX_WRITE_SIZE;
      else
        curr_num_bytes = bytes_left;

      error = Write(p + num_bytes, curr_num_bytes, offset);

      // Update how many bytes were read
      num_bytes += curr_num_bytes;
      if (bytes_left < curr_num_bytes)
        bytes_left = 0;
      else
        bytes_left -= curr_num_bytes;

      if (error.Fail())
        break;
    }
    return error;
  }
#endif

  int fd = GetDescriptor();
  if (fd != kInvalidDescriptor) {
    ssize_t bytes_written = llvm::sys::RetryAfterSignal(
        -1, ::pwrite, m_descriptor, buf, num_bytes, offset);
    if (bytes_written < 0) {
      num_bytes = 0;
      error = Status::FromErrno();
    } else {
      offset += bytes_written;
      num_bytes = bytes_written;
    }
  } else {
    num_bytes = 0;
    error = Status::FromErrorString("invalid file handle");
  }
  return error;
}

char NativeFilePosix::ID = 0;