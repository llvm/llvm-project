//===-- ConnectionGenericFileWindows.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ConnectionGenericFileWindows.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timeout.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

using namespace lldb;
using namespace lldb_private;

ConnectionGenericFile::ConnectionGenericFile()
    : m_file(INVALID_HANDLE_VALUE), m_owns_file(false) {
  ::ZeroMemory(&m_overlapped, sizeof(m_overlapped));
  ::ZeroMemory(&m_file_position, sizeof(m_file_position));
  InitializeEventHandles();
}

ConnectionGenericFile::ConnectionGenericFile(lldb::file_t file, bool owns_file)
    : m_file(file), m_owns_file(owns_file) {
  ::ZeroMemory(&m_overlapped, sizeof(m_overlapped));
  ::ZeroMemory(&m_file_position, sizeof(m_file_position));
  InitializeEventHandles();
}

ConnectionGenericFile::~ConnectionGenericFile() {
  if (m_owns_file && IsConnected())
    ::CloseHandle(m_file);

  ::CloseHandle(m_event_handles[kBytesAvailableEvent]);
  ::CloseHandle(m_event_handles[kInterruptEvent]);
}

void ConnectionGenericFile::InitializeEventHandles() {
  m_event_handles[kInterruptEvent] = CreateEvent(NULL, FALSE, FALSE, NULL);

  // Note, we should use a manual reset event for the hEvent argument of the
  // OVERLAPPED.  This is because both WaitForMultipleObjects and
  // GetOverlappedResult (if you set the bWait argument to TRUE) will wait for
  // the event to be signalled.  If we use an auto-reset event,
  // WaitForMultipleObjects will reset the event, return successfully, and then
  // GetOverlappedResult will block since the event is no longer signalled.
  m_event_handles[kBytesAvailableEvent] =
      ::CreateEvent(NULL, TRUE, FALSE, NULL);
}

bool ConnectionGenericFile::IsConnected() const {
  return m_file && (m_file != INVALID_HANDLE_VALUE);
}

lldb::ConnectionStatus ConnectionGenericFile::Connect(llvm::StringRef path,
                                                      Status *error_ptr) {
  Log *log = GetLog(LLDBLog::Connection);
  LLDB_LOGF(log, "%p ConnectionGenericFile::Connect (url = '%s')",
            static_cast<void *>(this), path.str().c_str());

  if (!path.consume_front("file://")) {
    if (error_ptr)
      *error_ptr = Status::FromErrorStringWithFormat(
          "unsupported connection URL: '%s'", path.str().c_str());
    return eConnectionStatusError;
  }

  if (IsConnected()) {
    ConnectionStatus status = Disconnect(error_ptr);
    if (status != eConnectionStatusSuccess)
      return status;
  }

  // Open the file for overlapped access.  If it does not exist, create it.  We
  // open it overlapped so that we can issue asynchronous reads and then use
  // WaitForMultipleObjects to allow the read to be interrupted by an event
  // object.
  std::wstring wpath;
  if (!llvm::ConvertUTF8toWide(path, wpath)) {
    if (error_ptr)
      *error_ptr = Status(1, eErrorTypeGeneric);
    return eConnectionStatusError;
  }
  m_file = ::CreateFileW(wpath.c_str(), GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ, NULL, OPEN_ALWAYS,
                         FILE_FLAG_OVERLAPPED, NULL);
  if (m_file == INVALID_HANDLE_VALUE) {
    if (error_ptr)
      *error_ptr = Status(::GetLastError(), eErrorTypeWin32);
    return eConnectionStatusError;
  }

  m_owns_file = true;
  m_uri = path.str();
  return eConnectionStatusSuccess;
}

lldb::ConnectionStatus ConnectionGenericFile::Disconnect(Status *error_ptr) {
  Log *log = GetLog(LLDBLog::Connection);
  LLDB_LOGF(log, "%p ConnectionGenericFile::Disconnect ()",
            static_cast<void *>(this));

  if (!IsConnected())
    return eConnectionStatusSuccess;

  // Reset the handle so that after we unblock any pending reads, subsequent
  // calls to Read() will see a disconnected state.
  HANDLE old_file = m_file;
  m_file = INVALID_HANDLE_VALUE;

  // Set the disconnect event so that any blocking reads unblock, then cancel
  // any pending IO operations.
  ::CancelIoEx(old_file, &m_overlapped);

  // Close the file handle if we owned it, but don't close the event handles.
  // We could always reconnect with the same Connection instance.
  if (m_owns_file)
    ::CloseHandle(old_file);

  ::ZeroMemory(&m_file_position, sizeof(m_file_position));
  m_owns_file = false;
  m_uri.clear();
  return eConnectionStatusSuccess;
}

size_t ConnectionGenericFile::Read(void *dst, size_t dst_len,
                                   const Timeout<std::micro> &timeout,
                                   lldb::ConnectionStatus &status,
                                   Status *error_ptr) {
  if (error_ptr)
    error_ptr->Clear();

  auto finish = [&](size_t bytes, ConnectionStatus s, DWORD error_code) {
    m_read_pending = s == eConnectionStatusInterrupted;
    status = s;
    if (error_ptr)
      *error_ptr = Status(error_code, eErrorTypeWin32);

    // kBytesAvailableEvent is a manual reset event.  Make sure it gets reset
    // here so that any subsequent operations don't immediately see bytes
    // available.
    ResetEvent(m_event_handles[kBytesAvailableEvent]);
    IncrementFilePointer(bytes);
    Log *log = GetLog(LLDBLog::Connection);
    LLDB_LOGF(log,
              "%p ConnectionGenericFile::Read()  handle = %p, dst = %p, "
              "dst_len = %zu) => %zu, error = %s",
              static_cast<void *>(this), m_file, dst, dst_len, bytes,
              error_code ? Status(error_code, eErrorTypeWin32).AsCString()
                         : "");
    return bytes;
  };

  if (!IsConnected())
    return finish(0, eConnectionStatusNoConnection, ERROR_INVALID_HANDLE);

  BOOL read_result;
  DWORD read_error;
  if (!m_read_pending) {
    m_overlapped.hEvent = m_event_handles[kBytesAvailableEvent];
    read_result = ::ReadFile(m_file, dst, dst_len, NULL, &m_overlapped);
    read_error = ::GetLastError();
  }

  if (!m_read_pending && !read_result && read_error != ERROR_IO_PENDING) {
    if (read_error == ERROR_BROKEN_PIPE) {
      // The write end of a pipe was closed.  This is equivalent to EOF.
      return finish(0, eConnectionStatusEndOfFile, 0);
    }
    // An unknown error occurred.  Fail out.
    return finish(0, eConnectionStatusError, ::GetLastError());
  }

  if (!read_result || m_read_pending) {
    // The expected return path.  The operation is pending.  Wait for the
    // operation to complete or be interrupted.
    DWORD milliseconds =
        timeout
            ? std::chrono::duration_cast<std::chrono::milliseconds>(*timeout)
                  .count()
            : INFINITE;
    DWORD wait_result = ::WaitForMultipleObjects(
        std::size(m_event_handles), m_event_handles, FALSE, milliseconds);
    // All of the events are manual reset events, so make sure we reset them
    // to non-signalled.
    switch (wait_result) {
    case WAIT_OBJECT_0 + kBytesAvailableEvent:
      break;
    case WAIT_OBJECT_0 + kInterruptEvent:
      return finish(0, eConnectionStatusInterrupted, 0);
    case WAIT_TIMEOUT:
      return finish(0, eConnectionStatusTimedOut, 0);
    case WAIT_FAILED:
      return finish(0, eConnectionStatusError, ::GetLastError());
    }
  }

  // The data is ready.  Figure out how much was read and return;
  DWORD bytes_read = 0;
  if (!::GetOverlappedResult(m_file, &m_overlapped, &bytes_read, FALSE)) {
    DWORD result_error = ::GetLastError();
    // ERROR_OPERATION_ABORTED occurs when someone calls Disconnect() during
    // a blocking read. This triggers a call to CancelIoEx, which causes the
    // operation to complete and the result to be ERROR_OPERATION_ABORTED.
    if (result_error == ERROR_HANDLE_EOF ||
        result_error == ERROR_OPERATION_ABORTED ||
        result_error == ERROR_BROKEN_PIPE)
      return finish(bytes_read, eConnectionStatusEndOfFile, 0);
    return finish(bytes_read, eConnectionStatusError, result_error);
  }

  if (bytes_read == 0)
    return finish(0, eConnectionStatusEndOfFile, 0);
  return finish(bytes_read, eConnectionStatusSuccess, 0);
}

size_t ConnectionGenericFile::Write(const void *src, size_t src_len,
                                    lldb::ConnectionStatus &status,
                                    Status *error_ptr) {
  if (error_ptr)
    error_ptr->Clear();

  auto finish = [&](size_t bytes, ConnectionStatus s, DWORD error_code) {
    status = s;
    if (error_ptr)
      *error_ptr = Status(error_code, eErrorTypeWin32);
    IncrementFilePointer(bytes);
    Log *log = GetLog(LLDBLog::Connection);
    LLDB_LOGF(log,
              "%p ConnectionGenericFile::Write()  handle = %p, src = %p, "
              "src_len = %zu) => %zu, error = %s",
              static_cast<void *>(this), m_file, src, src_len, bytes,
              Status(error_code, eErrorTypeWin32).AsCString());
    return bytes;
  };

  if (!IsConnected())
    return finish(0, eConnectionStatusNoConnection, ERROR_INVALID_HANDLE);

  m_overlapped.hEvent = NULL;

  DWORD bytes_written = 0;
  BOOL result = ::WriteFile(m_file, src, src_len, NULL, &m_overlapped);
  if (!result && ::GetLastError() != ERROR_IO_PENDING)
    return finish(0, eConnectionStatusError, ::GetLastError());

  if (!::GetOverlappedResult(m_file, &m_overlapped, &bytes_written, TRUE))
    return finish(bytes_written, eConnectionStatusError, ::GetLastError());

  return finish(bytes_written, eConnectionStatusSuccess, 0);
}

std::string ConnectionGenericFile::GetURI() { return m_uri; }

bool ConnectionGenericFile::InterruptRead() {
  return ::SetEvent(m_event_handles[kInterruptEvent]);
}

void ConnectionGenericFile::IncrementFilePointer(DWORD amount) {
  LARGE_INTEGER old_pos;
  old_pos.HighPart = m_overlapped.OffsetHigh;
  old_pos.LowPart = m_overlapped.Offset;
  old_pos.QuadPart += amount;
  m_overlapped.Offset = old_pos.LowPart;
  m_overlapped.OffsetHigh = old_pos.HighPart;
}
