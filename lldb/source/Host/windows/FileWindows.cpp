//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/FileWindows.h"

#include "lldb/Host/windows/windows.h"

#include <climits>
#include <io.h>
#include <mutex>
#include <stdio.h>

#include "lldb/Utility/Status.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;

NativeFileWindows::NativeFileWindows(FILE *fh, OpenOptions options,
                                     bool transfer_ownership)
    : NativeFileBase(fh, options, transfer_ownership) {
  HANDLE h = INVALID_HANDLE_VALUE;
  if (fh == stdin)
    h = ::GetStdHandle(STD_INPUT_HANDLE);
  else if (fh == stdout)
    h = ::GetStdHandle(STD_OUTPUT_HANDLE);
  else if (fh == stderr)
    h = ::GetStdHandle(STD_ERROR_HANDLE);
  m_is_windows_console =
      h != INVALID_HANDLE_VALUE && ::GetFileType(h) == FILE_TYPE_CHAR;
}

NativeFileWindows::NativeFileWindows(int fd, OpenOptions options,
                                     bool transfer_ownership)
    : NativeFileBase(fd, options, transfer_ownership) {
  HANDLE h = INVALID_HANDLE_VALUE;
  if (fd == STDIN_FILENO)
    h = ::GetStdHandle(STD_INPUT_HANDLE);
  else if (fd == STDOUT_FILENO)
    h = ::GetStdHandle(STD_OUTPUT_HANDLE);
  else if (fd == STDERR_FILENO)
    h = ::GetStdHandle(STD_ERROR_HANDLE);
  m_is_windows_console =
      h != INVALID_HANDLE_VALUE && ::GetFileType(h) == FILE_TYPE_CHAR;
}

void NativeFileWindows::CalculateInteractiveAndTerminal() {
  const int fd = GetDescriptor();
  if (!File::DescriptorIsValid(fd)) {
    m_is_interactive = eLazyBoolNo;
    m_is_real_terminal = eLazyBoolNo;
    m_supports_colors = eLazyBoolNo;
    return;
  }
  m_is_interactive = eLazyBoolNo;
  m_is_real_terminal = eLazyBoolNo;
  if (_isatty(fd)) {
    m_is_interactive = eLazyBoolYes;
    m_is_real_terminal = eLazyBoolYes;
#if defined(ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    m_supports_colors = eLazyBoolYes;
#endif
  }
}

int NativeFileWindows::Fileno(FILE *fh) const { return ::_fileno(fh); }

int NativeFileWindows::Dup(int fd) const { return ::_dup(fd); }

IOObject::WaitableHandle NativeFileWindows::GetWaitableHandle() {
  return (HANDLE)_get_osfhandle(GetDescriptor());
}

Status NativeFileWindows::Sync() {
  Status error;
  if (ValueGuard descriptor_guard = DescriptorIsValid()) {
    if (FlushFileBuffers((HANDLE)_get_osfhandle(m_descriptor)) == 0)
      error = Status::FromErrorString("unknown error");
  } else {
    error = Status::FromErrorString("invalid file handle");
  }
  return error;
}

bool NativeFileWindows::TryWriteStreamUnlocked(const void *buf,
                                               size_t &num_bytes,
                                               Status &error) {
  if (!m_is_windows_console)
    return false;
  // Bypass fwrite for console output: use raw_fd_ostream so that the Windows
  // console renders non-ASCII characters via its UTF-16 path.
  llvm::raw_fd_ostream(_fileno(m_stream), false)
      .write((const char *)buf, num_bytes);
  return true;
}

Status NativeFileWindows::Read(void *buf, size_t &num_bytes, off_t &offset) {
  Status error;

  int fd = GetDescriptor();
  if (fd != kInvalidDescriptor) {
    // Win32 has no pread(); emulate it by saving the current offset, seeking,
    // reading, and restoring.
    std::lock_guard<std::mutex> guard(offset_access_mutex);
    long cur = ::lseek(m_descriptor, 0, SEEK_CUR);
    SeekFromStart(offset);
    error = NativeFileBase::Read(buf, num_bytes);
    if (!error.Fail())
      SeekFromStart(cur);
  } else {
    num_bytes = 0;
    error = Status::FromErrorString("invalid file handle");
  }
  return error;
}

Status NativeFileWindows::Write(const void *buf, size_t &num_bytes,
                                off_t &offset) {
  Status error;

  int fd = GetDescriptor();
  if (fd != kInvalidDescriptor) {
    // Win32 has no pwrite(); same trick as Read above, but the post-write
    // file position is what the caller wants reported back via `offset`.
    std::lock_guard<std::mutex> guard(offset_access_mutex);
    long cur = ::lseek(m_descriptor, 0, SEEK_CUR);
    SeekFromStart(offset);
    error = NativeFileBase::Write(buf, num_bytes);
    long after = ::lseek(m_descriptor, 0, SEEK_CUR);

    if (!error.Fail())
      SeekFromStart(cur);

    offset = after;
  } else {
    num_bytes = 0;
    error = Status::FromErrorString("invalid file handle");
  }
  return error;
}

char NativeFileWindows::ID = 0;
