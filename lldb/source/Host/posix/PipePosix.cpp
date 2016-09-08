//===-- PipePosix.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/PipePosix.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#if defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
#ifndef _GLIBCXX_USE_NANOSLEEP
#define _GLIBCXX_USE_NANOSLEEP
#endif
#endif

#include <functional>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;

int PipePosix::kInvalidDescriptor = -1;

enum PIPES { READ, WRITE }; // Constants 0 and 1 for READ and WRITE

// pipe2 is supported by a limited set of platforms
// TODO: Add more platforms that support pipe2.
#if defined(__linux__) || (defined(__FreeBSD__) && __FreeBSD__ >= 10) ||       \
    defined(__NetBSD__)
#define PIPE2_SUPPORTED 1
#else
#define PIPE2_SUPPORTED 0
#endif

namespace {

constexpr auto OPEN_WRITER_SLEEP_TIMEOUT_MSECS = 100;

#if defined(FD_CLOEXEC) && !PIPE2_SUPPORTED
bool SetCloexecFlag(int fd) {
  int flags = ::fcntl(fd, F_GETFD);
  if (flags == -1)
    return false;
  return (::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0);
}
#endif

std::chrono::time_point<std::chrono::steady_clock> Now() {
  return std::chrono::steady_clock::now();
}

Error SelectIO(int handle, bool is_read,
               const std::function<Error(bool &)> &io_handler,
               const std::chrono::microseconds &timeout) {
  Error error;
  fd_set fds;
  bool done = false;

  using namespace std::chrono;

  const auto finish_time = Now() + timeout;

  while (!done) {
    struct timeval tv = {0, 0};
    if (timeout != microseconds::zero()) {
      const auto remaining_dur =
          duration_cast<microseconds>(finish_time - Now());
      if (remaining_dur.count() <= 0) {
        error.SetErrorString("timeout exceeded");
        break;
      }
      const auto dur_secs = duration_cast<seconds>(remaining_dur);
      const auto dur_usecs = remaining_dur % seconds(1);

      tv.tv_sec = dur_secs.count();
      tv.tv_usec = dur_usecs.count();
    } else
      tv.tv_sec = 1;

    FD_ZERO(&fds);
    FD_SET(handle, &fds);

    const auto retval = ::select(handle + 1, (is_read) ? &fds : nullptr,
                                 (is_read) ? nullptr : &fds, nullptr, &tv);
    if (retval == -1) {
      if (errno == EINTR)
        continue;
      error.SetErrorToErrno();
      break;
    }
    if (retval == 0) {
      error.SetErrorString("timeout exceeded");
      break;
    }
    if (!FD_ISSET(handle, &fds)) {
      error.SetErrorString("invalid state");
      break;
    }

    error = io_handler(done);
    if (error.Fail()) {
      if (error.GetError() == EINTR)
        continue;
      break;
    }
  }
  return error;
}
}

PipePosix::PipePosix()
    : m_fds{PipePosix::kInvalidDescriptor, PipePosix::kInvalidDescriptor} {}

PipePosix::PipePosix(int read_fd, int write_fd) : m_fds{read_fd, write_fd} {}

PipePosix::PipePosix(PipePosix &&pipe_posix)
    : PipeBase{std::move(pipe_posix)},
      m_fds{pipe_posix.ReleaseReadFileDescriptor(),
            pipe_posix.ReleaseWriteFileDescriptor()} {}

PipePosix &PipePosix::operator=(PipePosix &&pipe_posix) {
  PipeBase::operator=(std::move(pipe_posix));
  m_fds[READ] = pipe_posix.ReleaseReadFileDescriptor();
  m_fds[WRITE] = pipe_posix.ReleaseWriteFileDescriptor();
  return *this;
}

PipePosix::~PipePosix() { Close(); }

Error PipePosix::CreateNew(bool child_processes_inherit) {
  if (CanRead() || CanWrite())
    return Error(EINVAL, eErrorTypePOSIX);

  Error error;
#if PIPE2_SUPPORTED
  if (::pipe2(m_fds, (child_processes_inherit) ? 0 : O_CLOEXEC) == 0)
    return error;
#else
  if (::pipe(m_fds) == 0) {
#ifdef FD_CLOEXEC
    if (!child_processes_inherit) {
      if (!SetCloexecFlag(m_fds[0]) || !SetCloexecFlag(m_fds[1])) {
        error.SetErrorToErrno();
        Close();
        return error;
      }
    }
#endif
    return error;
  }
#endif

  error.SetErrorToErrno();
  m_fds[READ] = PipePosix::kInvalidDescriptor;
  m_fds[WRITE] = PipePosix::kInvalidDescriptor;
  return error;
}

Error PipePosix::CreateNew(llvm::StringRef name, bool child_process_inherit) {
  if (CanRead() || CanWrite())
    return Error("Pipe is already opened");

  Error error;
  if (::mkfifo(name.data(), 0660) != 0)
    error.SetErrorToErrno();

  return error;
}

Error PipePosix::CreateWithUniqueName(llvm::StringRef prefix,
                                      bool child_process_inherit,
                                      llvm::SmallVectorImpl<char> &name) {
  llvm::SmallString<PATH_MAX> named_pipe_path;
  llvm::SmallString<PATH_MAX> pipe_spec((prefix + ".%%%%%%").str());
  FileSpec tmpdir_file_spec;
  tmpdir_file_spec.Clear();
  if (HostInfo::GetLLDBPath(ePathTypeLLDBTempSystemDir, tmpdir_file_spec)) {
    tmpdir_file_spec.AppendPathComponent(pipe_spec.c_str());
  } else {
    tmpdir_file_spec.AppendPathComponent("/tmp");
    tmpdir_file_spec.AppendPathComponent(pipe_spec.c_str());
  }

  // It's possible that another process creates the target path after we've
  // verified it's available but before we create it, in which case we
  // should try again.
  Error error;
  do {
    llvm::sys::fs::createUniqueFile(tmpdir_file_spec.GetPath().c_str(),
                                    named_pipe_path);
    error = CreateNew(named_pipe_path, child_process_inherit);
  } while (error.GetError() == EEXIST);

  if (error.Success())
    name = named_pipe_path;
  return error;
}

Error PipePosix::OpenAsReader(llvm::StringRef name,
                              bool child_process_inherit) {
  if (CanRead() || CanWrite())
    return Error("Pipe is already opened");

  int flags = O_RDONLY | O_NONBLOCK;
  if (!child_process_inherit)
    flags |= O_CLOEXEC;

  Error error;
  int fd = ::open(name.data(), flags);
  if (fd != -1)
    m_fds[READ] = fd;
  else
    error.SetErrorToErrno();

  return error;
}

Error PipePosix::OpenAsWriterWithTimeout(
    llvm::StringRef name, bool child_process_inherit,
    const std::chrono::microseconds &timeout) {
  if (CanRead() || CanWrite())
    return Error("Pipe is already opened");

  int flags = O_WRONLY | O_NONBLOCK;
  if (!child_process_inherit)
    flags |= O_CLOEXEC;

  using namespace std::chrono;
  const auto finish_time = Now() + timeout;

  while (!CanWrite()) {
    if (timeout != microseconds::zero()) {
      const auto dur = duration_cast<microseconds>(finish_time - Now()).count();
      if (dur <= 0)
        return Error("timeout exceeded - reader hasn't opened so far");
    }

    errno = 0;
    int fd = ::open(name.data(), flags);
    if (fd == -1) {
      const auto errno_copy = errno;
      // We may get ENXIO if a reader side of the pipe hasn't opened yet.
      if (errno_copy != ENXIO)
        return Error(errno_copy, eErrorTypePOSIX);

      std::this_thread::sleep_for(
          milliseconds(OPEN_WRITER_SLEEP_TIMEOUT_MSECS));
    } else {
      m_fds[WRITE] = fd;
    }
  }

  return Error();
}

int PipePosix::GetReadFileDescriptor() const { return m_fds[READ]; }

int PipePosix::GetWriteFileDescriptor() const { return m_fds[WRITE]; }

int PipePosix::ReleaseReadFileDescriptor() {
  const int fd = m_fds[READ];
  m_fds[READ] = PipePosix::kInvalidDescriptor;
  return fd;
}

int PipePosix::ReleaseWriteFileDescriptor() {
  const int fd = m_fds[WRITE];
  m_fds[WRITE] = PipePosix::kInvalidDescriptor;
  return fd;
}

void PipePosix::Close() {
  CloseReadFileDescriptor();
  CloseWriteFileDescriptor();
}

Error PipePosix::Delete(llvm::StringRef name) {
  return FileSystem::Unlink(FileSpec{name.data(), true});
}

bool PipePosix::CanRead() const {
  return m_fds[READ] != PipePosix::kInvalidDescriptor;
}

bool PipePosix::CanWrite() const {
  return m_fds[WRITE] != PipePosix::kInvalidDescriptor;
}

void PipePosix::CloseReadFileDescriptor() {
  if (CanRead()) {
    close(m_fds[READ]);
    m_fds[READ] = PipePosix::kInvalidDescriptor;
  }
}

void PipePosix::CloseWriteFileDescriptor() {
  if (CanWrite()) {
    close(m_fds[WRITE]);
    m_fds[WRITE] = PipePosix::kInvalidDescriptor;
  }
}

Error PipePosix::ReadWithTimeout(void *buf, size_t size,
                                 const std::chrono::microseconds &timeout,
                                 size_t &bytes_read) {
  bytes_read = 0;
  if (!CanRead())
    return Error(EINVAL, eErrorTypePOSIX);

  auto handle = GetReadFileDescriptor();
  return SelectIO(handle, true,
                  [=, &bytes_read](bool &done) {
                    Error error;
                    auto result = ::read(handle, reinterpret_cast<char *>(buf) +
                                                     bytes_read,
                                         size - bytes_read);
                    if (result != -1) {
                      bytes_read += result;
                      if (bytes_read == size || result == 0)
                        done = true;
                    } else
                      error.SetErrorToErrno();

                    return error;
                  },
                  timeout);
}

Error PipePosix::Write(const void *buf, size_t size, size_t &bytes_written) {
  bytes_written = 0;
  if (!CanWrite())
    return Error(EINVAL, eErrorTypePOSIX);

  auto handle = GetWriteFileDescriptor();
  return SelectIO(handle, false,
                  [=, &bytes_written](bool &done) {
                    Error error;
                    auto result =
                        ::write(handle, reinterpret_cast<const char *>(buf) +
                                            bytes_written,
                                size - bytes_written);
                    if (result != -1) {
                      bytes_written += result;
                      if (bytes_written == size)
                        done = true;
                    } else
                      error.SetErrorToErrno();

                    return error;
                  },
                  std::chrono::microseconds::zero());
}
