//===-- PipeWindows.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PipeWindows.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <fcntl.h>
#include <io.h>
#include <rpc.h>

#include <atomic>
#include <string>

using namespace lldb;
using namespace lldb_private;

static std::atomic<uint32_t> g_pipe_serial(0);
static constexpr llvm::StringLiteral g_pipe_name_prefix = "\\\\.\\Pipe\\";

PipeWindows::PipeWindows()
    : m_read(INVALID_HANDLE_VALUE), m_write(INVALID_HANDLE_VALUE),
      m_read_fd(PipeWindows::kInvalidDescriptor),
      m_write_fd(PipeWindows::kInvalidDescriptor) {
  ZeroMemory(&m_read_overlapped, sizeof(m_read_overlapped));
  ZeroMemory(&m_write_overlapped, sizeof(m_write_overlapped));
}

PipeWindows::PipeWindows(pipe_t read, pipe_t write)
    : m_read((HANDLE)read), m_write((HANDLE)write),
      m_read_fd(PipeWindows::kInvalidDescriptor),
      m_write_fd(PipeWindows::kInvalidDescriptor) {
  assert(read != LLDB_INVALID_PIPE || write != LLDB_INVALID_PIPE);

  // Don't risk in passing file descriptors and getting handles from them by
  // _get_osfhandle since the retrieved handles are highly likely unrecognized
  // in the current process and usually crashes the program.  Pass handles
  // instead since the handle can be inherited.

  if (read != LLDB_INVALID_PIPE) {
    m_read_fd = _open_osfhandle((intptr_t)read, _O_RDONLY);
    // Make sure the fd and native handle are consistent.
    if (m_read_fd < 0)
      m_read = INVALID_HANDLE_VALUE;
  }

  if (write != LLDB_INVALID_PIPE) {
    m_write_fd = _open_osfhandle((intptr_t)write, _O_WRONLY);
    if (m_write_fd < 0)
      m_write = INVALID_HANDLE_VALUE;
  }

  ZeroMemory(&m_read_overlapped, sizeof(m_read_overlapped));
  m_read_overlapped.hEvent = ::CreateEventA(nullptr, TRUE, FALSE, nullptr);

  ZeroMemory(&m_write_overlapped, sizeof(m_write_overlapped));
  m_write_overlapped.hEvent = ::CreateEventA(nullptr, TRUE, FALSE, nullptr);
}

PipeWindows::~PipeWindows() { Close(); }

Status PipeWindows::CreateNew() {
  // Even for anonymous pipes, we open a named pipe.  This is because you
  // cannot get overlapped i/o on Windows without using a named pipe.  So we
  // synthesize a unique name.
  uint32_t serial = g_pipe_serial.fetch_add(1);
  std::string pipe_name = llvm::formatv(
      "lldb.pipe.{0}.{1}.{2}", GetCurrentProcessId(), &g_pipe_serial, serial);

  return CreateNew(pipe_name.c_str());
}

Status PipeWindows::CreateNew(llvm::StringRef name) {
  if (name.empty())
    return Status(ERROR_INVALID_PARAMETER, eErrorTypeWin32);

  if (CanRead() || CanWrite())
    return Status(ERROR_ALREADY_EXISTS, eErrorTypeWin32);

  std::string pipe_path = g_pipe_name_prefix.str();
  pipe_path.append(name.str());

  // We always create inheritable handles, but we won't pass them to a child
  // process unless explicitly requested (cf. ProcessLauncherWindows.cpp).
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), 0, TRUE};

  // Always open for overlapped i/o.  We implement blocking manually in Read
  // and Write.
  DWORD read_mode = FILE_FLAG_OVERLAPPED;
  m_read =
      ::CreateNamedPipeA(pipe_path.c_str(), PIPE_ACCESS_INBOUND | read_mode,
                         PIPE_TYPE_BYTE | PIPE_WAIT, /*nMaxInstances=*/1,
                         /*nOutBufferSize=*/1024,
                         /*nInBufferSize=*/1024,
                         /*nDefaultTimeOut=*/0, &sa);
  if (INVALID_HANDLE_VALUE == m_read)
    return Status(::GetLastError(), eErrorTypeWin32);
  m_read_fd = _open_osfhandle((intptr_t)m_read, _O_RDONLY);
  ZeroMemory(&m_read_overlapped, sizeof(m_read_overlapped));
  m_read_overlapped.hEvent = ::CreateEvent(nullptr, TRUE, FALSE, nullptr);

  // Open the write end of the pipe. Note that closing either the read or 
  // write end of the pipe could directly close the pipe itself.
  Status result = OpenNamedPipe(name, false);
  if (!result.Success()) {
    CloseReadFileDescriptor();
    return result;
  }

  return result;
}

Status PipeWindows::CreateWithUniqueName(llvm::StringRef prefix,
                                         llvm::SmallVectorImpl<char> &name) {
  llvm::SmallString<128> pipe_name;
  Status error;
  ::UUID unique_id;
  RPC_CSTR unique_string;
  RPC_STATUS status = ::UuidCreate(&unique_id);
  if (status == RPC_S_OK || status == RPC_S_UUID_LOCAL_ONLY)
    status = ::UuidToStringA(&unique_id, &unique_string);
  if (status == RPC_S_OK) {
    pipe_name = prefix;
    pipe_name += "-";
    pipe_name += reinterpret_cast<char *>(unique_string);
    ::RpcStringFreeA(&unique_string);
    error = CreateNew(pipe_name);
  } else {
    error = Status(status, eErrorTypeWin32);
  }
  if (error.Success())
    name = pipe_name;
  return error;
}

Status PipeWindows::OpenAsReader(llvm::StringRef name) {
  if (CanRead())
    return Status(); // Note the name is ignored.

  return OpenNamedPipe(name, true);
}

llvm::Error PipeWindows::OpenAsWriter(llvm::StringRef name,
                                      const Timeout<std::micro> &timeout) {
  if (CanWrite())
    return llvm::Error::success(); // Note the name is ignored.

  return OpenNamedPipe(name, false).takeError();
}

Status PipeWindows::OpenNamedPipe(llvm::StringRef name, bool is_read) {
  if (name.empty())
    return Status(ERROR_INVALID_PARAMETER, eErrorTypeWin32);

  assert(is_read ? !CanRead() : !CanWrite());

  // We always create inheritable handles, but we won't pass them to a child
  // process unless explicitly requested (cf. ProcessLauncherWindows.cpp).
  SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), 0, TRUE};

  std::string pipe_path = g_pipe_name_prefix.str();
  pipe_path.append(name.str());

  if (is_read) {
    m_read = ::CreateFileA(pipe_path.c_str(), GENERIC_READ, 0, &attributes,
                           OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    if (INVALID_HANDLE_VALUE == m_read)
      return Status(::GetLastError(), eErrorTypeWin32);

    m_read_fd = _open_osfhandle((intptr_t)m_read, _O_RDONLY);

    ZeroMemory(&m_read_overlapped, sizeof(m_read_overlapped));
    m_read_overlapped.hEvent = ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
  } else {
    m_write = ::CreateFileA(pipe_path.c_str(), GENERIC_WRITE, 0, &attributes,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    if (INVALID_HANDLE_VALUE == m_write)
      return Status(::GetLastError(), eErrorTypeWin32);

    m_write_fd = _open_osfhandle((intptr_t)m_write, _O_WRONLY);

    ZeroMemory(&m_write_overlapped, sizeof(m_write_overlapped));
    m_write_overlapped.hEvent = ::CreateEventA(nullptr, TRUE, FALSE, nullptr);
  }

  return Status();
}

int PipeWindows::GetReadFileDescriptor() const { return m_read_fd; }

int PipeWindows::GetWriteFileDescriptor() const { return m_write_fd; }

int PipeWindows::ReleaseReadFileDescriptor() {
  if (!CanRead())
    return PipeWindows::kInvalidDescriptor;
  int result = m_read_fd;
  m_read_fd = PipeWindows::kInvalidDescriptor;
  if (m_read_overlapped.hEvent)
    ::CloseHandle(m_read_overlapped.hEvent);
  m_read = INVALID_HANDLE_VALUE;
  ZeroMemory(&m_read_overlapped, sizeof(m_read_overlapped));
  return result;
}

int PipeWindows::ReleaseWriteFileDescriptor() {
  if (!CanWrite())
    return PipeWindows::kInvalidDescriptor;
  int result = m_write_fd;
  m_write_fd = PipeWindows::kInvalidDescriptor;
  if (m_write_overlapped.hEvent)
    ::CloseHandle(m_write_overlapped.hEvent);
  m_write = INVALID_HANDLE_VALUE;
  ZeroMemory(&m_write_overlapped, sizeof(m_write_overlapped));
  return result;
}

void PipeWindows::CloseReadFileDescriptor() {
  if (!CanRead())
    return;

  if (m_read_overlapped.hEvent)
    ::CloseHandle(m_read_overlapped.hEvent);

  _close(m_read_fd);
  m_read = INVALID_HANDLE_VALUE;
  m_read_fd = PipeWindows::kInvalidDescriptor;
  ZeroMemory(&m_read_overlapped, sizeof(m_read_overlapped));
}

void PipeWindows::CloseWriteFileDescriptor() {
  if (!CanWrite())
    return;

  if (m_write_overlapped.hEvent)
    ::CloseHandle(m_write_overlapped.hEvent);

  _close(m_write_fd);
  m_write = INVALID_HANDLE_VALUE;
  m_write_fd = PipeWindows::kInvalidDescriptor;
  ZeroMemory(&m_write_overlapped, sizeof(m_write_overlapped));
}

void PipeWindows::Close() {
  CloseReadFileDescriptor();
  CloseWriteFileDescriptor();
}

Status PipeWindows::Delete(llvm::StringRef name) { return Status(); }

bool PipeWindows::CanRead() const { return (m_read != INVALID_HANDLE_VALUE); }

bool PipeWindows::CanWrite() const { return (m_write != INVALID_HANDLE_VALUE); }

HANDLE
PipeWindows::GetReadNativeHandle() { return m_read; }

HANDLE
PipeWindows::GetWriteNativeHandle() { return m_write; }

llvm::Expected<size_t> PipeWindows::Read(void *buf, size_t size,
                                         const Timeout<std::micro> &timeout) {
  if (!CanRead())
    return Status(ERROR_INVALID_HANDLE, eErrorTypeWin32).takeError();

  DWORD bytes_read = 0;
  BOOL result = ::ReadFile(m_read, buf, size, &bytes_read, &m_read_overlapped);
  if (result)
    return bytes_read;

  DWORD failure_error = ::GetLastError();
  if (failure_error != ERROR_IO_PENDING)
    return Status(failure_error, eErrorTypeWin32).takeError();

  DWORD timeout_msec =
      timeout ? std::chrono::ceil<std::chrono::milliseconds>(*timeout).count()
              : INFINITE;
  DWORD wait_result =
      ::WaitForSingleObject(m_read_overlapped.hEvent, timeout_msec);
  if (wait_result != WAIT_OBJECT_0) {
    // The operation probably failed.  However, if it timed out, we need to
    // cancel the I/O. Between the time we returned from WaitForSingleObject
    // and the time we call CancelIoEx, the operation may complete.  If that
    // hapens, CancelIoEx will fail and return ERROR_NOT_FOUND. If that
    // happens, the original operation should be considered to have been
    // successful.
    bool failed = true;
    failure_error = ::GetLastError();
    if (wait_result == WAIT_TIMEOUT) {
      BOOL cancel_result = ::CancelIoEx(m_read, &m_read_overlapped);
      if (!cancel_result && ::GetLastError() == ERROR_NOT_FOUND)
        failed = false;
    }
    if (failed)
      return Status(failure_error, eErrorTypeWin32).takeError();
  }

  // Now we call GetOverlappedResult setting bWait to false, since we've
  // already waited as long as we're willing to.
  if (!::GetOverlappedResult(m_read, &m_read_overlapped, &bytes_read, FALSE))
    return Status(::GetLastError(), eErrorTypeWin32).takeError();

  return bytes_read;
}

llvm::Expected<size_t> PipeWindows::Write(const void *buf, size_t size,
                                          const Timeout<std::micro> &timeout) {
  if (!CanWrite())
    return Status(ERROR_INVALID_HANDLE, eErrorTypeWin32).takeError();

  DWORD bytes_written = 0;
  BOOL result =
      ::WriteFile(m_write, buf, size, &bytes_written, &m_write_overlapped);
  if (result)
    return bytes_written;

  DWORD failure_error = ::GetLastError();
  if (failure_error != ERROR_IO_PENDING)
    return Status(failure_error, eErrorTypeWin32).takeError();

  DWORD timeout_msec =
      timeout ? std::chrono::ceil<std::chrono::milliseconds>(*timeout).count()
              : INFINITE;
  DWORD wait_result =
      ::WaitForSingleObject(m_write_overlapped.hEvent, timeout_msec);
  if (wait_result != WAIT_OBJECT_0) {
    // The operation probably failed.  However, if it timed out, we need to
    // cancel the I/O. Between the time we returned from WaitForSingleObject
    // and the time we call CancelIoEx, the operation may complete.  If that
    // hapens, CancelIoEx will fail and return ERROR_NOT_FOUND. If that
    // happens, the original operation should be considered to have been
    // successful.
    bool failed = true;
    failure_error = ::GetLastError();
    if (wait_result == WAIT_TIMEOUT) {
      BOOL cancel_result = ::CancelIoEx(m_write, &m_write_overlapped);
      if (!cancel_result && ::GetLastError() == ERROR_NOT_FOUND)
        failed = false;
    }
    if (failed)
      return Status(failure_error, eErrorTypeWin32).takeError();
  }

  // Now we call GetOverlappedResult setting bWait to false, since we've
  // already waited as long as we're willing to.
  if (!::GetOverlappedResult(m_write, &m_write_overlapped, &bytes_written,
                             FALSE))
    return Status(::GetLastError(), eErrorTypeWin32).takeError();

  return bytes_written;
}
