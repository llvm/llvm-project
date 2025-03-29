//===-- PipePosix.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/PipePosix.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/SelectHelper.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <system_error>
#include <thread>

#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;

int PipePosix::kInvalidDescriptor = -1;

enum PIPES { READ, WRITE }; // Constants 0 and 1 for READ and WRITE

// pipe2 is supported by a limited set of platforms
// TODO: Add more platforms that support pipe2.
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) ||       \
    defined(__OpenBSD__)
#define PIPE2_SUPPORTED 1
#else
#define PIPE2_SUPPORTED 0
#endif

static constexpr auto OPEN_WRITER_SLEEP_TIMEOUT_MSECS = 100;

#if defined(FD_CLOEXEC) && !PIPE2_SUPPORTED
static bool SetCloexecFlag(int fd) {
  int flags = ::fcntl(fd, F_GETFD);
  if (flags == -1)
    return false;
  return (::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0);
}
#endif

static std::chrono::time_point<std::chrono::steady_clock> Now() {
  return std::chrono::steady_clock::now();
}

PipePosix::PipePosix()
    : m_fds{PipePosix::kInvalidDescriptor, PipePosix::kInvalidDescriptor} {}

PipePosix::PipePosix(lldb::pipe_t read, lldb::pipe_t write)
    : m_fds{read, write} {}

PipePosix::PipePosix(PipePosix &&pipe_posix)
    : PipeBase{std::move(pipe_posix)},
      m_fds{pipe_posix.ReleaseReadFileDescriptor(),
            pipe_posix.ReleaseWriteFileDescriptor()} {}

PipePosix &PipePosix::operator=(PipePosix &&pipe_posix) {
  std::scoped_lock<std::mutex, std::mutex, std::mutex, std::mutex> guard(
      m_read_mutex, m_write_mutex, pipe_posix.m_read_mutex,
      pipe_posix.m_write_mutex);

  PipeBase::operator=(std::move(pipe_posix));
  m_fds[READ] = pipe_posix.ReleaseReadFileDescriptorUnlocked();
  m_fds[WRITE] = pipe_posix.ReleaseWriteFileDescriptorUnlocked();
  return *this;
}

PipePosix::~PipePosix() { Close(); }

Status PipePosix::CreateNew(bool child_processes_inherit) {
  std::scoped_lock<std::mutex, std::mutex> guard(m_read_mutex, m_write_mutex);
  if (CanReadUnlocked() || CanWriteUnlocked())
    return Status(EINVAL, eErrorTypePOSIX);

  Status error;
#if PIPE2_SUPPORTED
  if (::pipe2(m_fds, (child_processes_inherit) ? 0 : O_CLOEXEC) == 0)
    return error;
#else
  if (::pipe(m_fds) == 0) {
#ifdef FD_CLOEXEC
    if (!child_processes_inherit) {
      if (!SetCloexecFlag(m_fds[0]) || !SetCloexecFlag(m_fds[1])) {
        error = Status::FromErrno();
        CloseUnlocked();
        return error;
      }
    }
#endif
    return error;
  }
#endif

  error = Status::FromErrno();
  m_fds[READ] = PipePosix::kInvalidDescriptor;
  m_fds[WRITE] = PipePosix::kInvalidDescriptor;
  return error;
}

Status PipePosix::CreateNew(llvm::StringRef name, bool child_process_inherit) {
  std::scoped_lock<std::mutex, std::mutex> guard(m_read_mutex, m_write_mutex);
  if (CanReadUnlocked() || CanWriteUnlocked())
    return Status::FromErrorString("Pipe is already opened");

  Status error;
  if (::mkfifo(name.str().c_str(), 0660) != 0)
    error = Status::FromErrno();
  return error;
}

Status PipePosix::CreateWithUniqueName(llvm::StringRef prefix,
                                       bool child_process_inherit,
                                       llvm::SmallVectorImpl<char> &name) {
  llvm::SmallString<128> named_pipe_path;
  llvm::SmallString<128> pipe_spec((prefix + ".%%%%%%").str());
  FileSpec tmpdir_file_spec = HostInfo::GetProcessTempDir();
  if (!tmpdir_file_spec)
    tmpdir_file_spec.AppendPathComponent("/tmp");
  tmpdir_file_spec.AppendPathComponent(pipe_spec);

  // It's possible that another process creates the target path after we've
  // verified it's available but before we create it, in which case we should
  // try again.
  Status error;
  do {
    llvm::sys::fs::createUniquePath(tmpdir_file_spec.GetPath(), named_pipe_path,
                                    /*MakeAbsolute=*/false);
    error = CreateNew(named_pipe_path, child_process_inherit);
  } while (error.GetError() == EEXIST);

  if (error.Success())
    name = named_pipe_path;
  return error;
}

Status PipePosix::OpenAsReader(llvm::StringRef name,
                               bool child_process_inherit) {
  std::scoped_lock<std::mutex, std::mutex> guard(m_read_mutex, m_write_mutex);

  if (CanReadUnlocked() || CanWriteUnlocked())
    return Status::FromErrorString("Pipe is already opened");

  int flags = O_RDONLY | O_NONBLOCK;
  if (!child_process_inherit)
    flags |= O_CLOEXEC;

  Status error;
  int fd = FileSystem::Instance().Open(name.str().c_str(), flags);
  if (fd != -1)
    m_fds[READ] = fd;
  else
    error = Status::FromErrno();

  return error;
}

llvm::Error PipePosix::OpenAsWriter(llvm::StringRef name,
                                    bool child_process_inherit,
                                    const Timeout<std::micro> &timeout) {
  std::lock_guard<std::mutex> guard(m_write_mutex);
  if (CanReadUnlocked() || CanWriteUnlocked())
    return llvm::createStringError("Pipe is already opened");

  int flags = O_WRONLY | O_NONBLOCK;
  if (!child_process_inherit)
    flags |= O_CLOEXEC;

  using namespace std::chrono;
  std::optional<time_point<steady_clock>> finish_time;
  if (timeout)
    finish_time = Now() + *timeout;

  while (!CanWriteUnlocked()) {
    if (timeout) {
      if (Now() > finish_time)
        return llvm::createStringError(
            std::make_error_code(std::errc::timed_out),
            "timeout exceeded - reader hasn't opened so far");
    }

    errno = 0;
    int fd = ::open(name.str().c_str(), flags);
    if (fd == -1) {
      const auto errno_copy = errno;
      // We may get ENXIO if a reader side of the pipe hasn't opened yet.
      if (errno_copy != ENXIO && errno_copy != EINTR)
        return llvm::errorCodeToError(
            std::error_code(errno_copy, std::generic_category()));

      std::this_thread::sleep_for(
          milliseconds(OPEN_WRITER_SLEEP_TIMEOUT_MSECS));
    } else {
      m_fds[WRITE] = fd;
    }
  }

  return llvm::Error::success();
}

int PipePosix::GetReadFileDescriptor() const {
  std::lock_guard<std::mutex> guard(m_read_mutex);
  return GetReadFileDescriptorUnlocked();
}

int PipePosix::GetReadFileDescriptorUnlocked() const {
  return m_fds[READ];
}

int PipePosix::GetWriteFileDescriptor() const {
  std::lock_guard<std::mutex> guard(m_write_mutex);
  return GetWriteFileDescriptorUnlocked();
}

int PipePosix::GetWriteFileDescriptorUnlocked() const {
  return m_fds[WRITE];
}

int PipePosix::ReleaseReadFileDescriptor() {
  std::lock_guard<std::mutex> guard(m_read_mutex);
  return ReleaseReadFileDescriptorUnlocked();
}

int PipePosix::ReleaseReadFileDescriptorUnlocked() {
  const int fd = m_fds[READ];
  m_fds[READ] = PipePosix::kInvalidDescriptor;
  return fd;
}

int PipePosix::ReleaseWriteFileDescriptor() {
  std::lock_guard<std::mutex> guard(m_write_mutex);
  return ReleaseWriteFileDescriptorUnlocked();
}

int PipePosix::ReleaseWriteFileDescriptorUnlocked() {
  const int fd = m_fds[WRITE];
  m_fds[WRITE] = PipePosix::kInvalidDescriptor;
  return fd;
}

void PipePosix::Close() {
  std::scoped_lock<std::mutex, std::mutex> guard(m_read_mutex, m_write_mutex);
  CloseUnlocked();
}

void PipePosix::CloseUnlocked() {
  CloseReadFileDescriptorUnlocked();
  CloseWriteFileDescriptorUnlocked();
}

Status PipePosix::Delete(llvm::StringRef name) {
  return llvm::sys::fs::remove(name);
}

bool PipePosix::CanRead() const {
  std::lock_guard<std::mutex> guard(m_read_mutex);
  return CanReadUnlocked();
}

bool PipePosix::CanReadUnlocked() const {
  return m_fds[READ] != PipePosix::kInvalidDescriptor;
}

bool PipePosix::CanWrite() const {
  std::lock_guard<std::mutex> guard(m_write_mutex);
  return CanWriteUnlocked();
}

bool PipePosix::CanWriteUnlocked() const {
  return m_fds[WRITE] != PipePosix::kInvalidDescriptor;
}

void PipePosix::CloseReadFileDescriptor() {
  std::lock_guard<std::mutex> guard(m_read_mutex);
  CloseReadFileDescriptorUnlocked();
}
void PipePosix::CloseReadFileDescriptorUnlocked() {
  if (CanReadUnlocked()) {
    close(m_fds[READ]);
    m_fds[READ] = PipePosix::kInvalidDescriptor;
  }
}

void PipePosix::CloseWriteFileDescriptor() {
  std::lock_guard<std::mutex> guard(m_write_mutex);
  CloseWriteFileDescriptorUnlocked();
}

void PipePosix::CloseWriteFileDescriptorUnlocked() {
  if (CanWriteUnlocked()) {
    close(m_fds[WRITE]);
    m_fds[WRITE] = PipePosix::kInvalidDescriptor;
  }
}

llvm::Expected<size_t> PipePosix::Read(void *buf, size_t size,
                                       const Timeout<std::micro> &timeout) {
  std::lock_guard<std::mutex> guard(m_read_mutex);
  if (!CanReadUnlocked())
    return llvm::errorCodeToError(
        std::make_error_code(std::errc::invalid_argument));

  const int fd = GetReadFileDescriptorUnlocked();

  SelectHelper select_helper;
  if (timeout)
    select_helper.SetTimeout(*timeout);
  select_helper.FDSetRead(fd);

  if (llvm::Error error = select_helper.Select().takeError())
    return error;

  ssize_t result = ::read(fd, buf, size);
  if (result == -1)
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));

  return result;
}

llvm::Expected<size_t> PipePosix::Write(const void *buf, size_t size,
                                        const Timeout<std::micro> &timeout) {
  std::lock_guard<std::mutex> guard(m_write_mutex);
  if (!CanWriteUnlocked())
    return llvm::errorCodeToError(
        std::make_error_code(std::errc::invalid_argument));

  const int fd = GetWriteFileDescriptorUnlocked();
  SelectHelper select_helper;
  if (timeout)
    select_helper.SetTimeout(*timeout);
  select_helper.FDSetWrite(fd);

  if (llvm::Error error = select_helper.Select().takeError())
    return error;

  ssize_t result = ::write(fd, buf, size);
  if (result == -1)
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));

  return result;
}
