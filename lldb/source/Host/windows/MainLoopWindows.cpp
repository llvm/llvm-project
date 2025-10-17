//===-- MainLoopWindows.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/MainLoopWindows.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/Status.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/WindowsError.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <ctime>
#include <io.h>
#include <synchapi.h>
#include <thread>
#include <vector>
#include <winbase.h>
#include <winerror.h>
#include <winsock2.h>

using namespace lldb;
using namespace lldb_private;

static DWORD ToTimeout(std::optional<MainLoopWindows::TimePoint> point) {
  using namespace std::chrono;

  if (!point)
    return WSA_INFINITE;

  nanoseconds dur = (std::max)(*point - steady_clock::now(), nanoseconds(0));
  return ceil<milliseconds>(dur).count();
}

namespace {

class PipeEvent : public MainLoopWindows::IOEvent {
public:
  explicit PipeEvent(HANDLE handle)
      : IOEvent(CreateEventW(NULL, /*bManualReset=*/TRUE,
                             /*bInitialState=*/FALSE, NULL)),
        m_handle(handle), m_ready(CreateEventW(NULL, /*bManualReset=*/TRUE,
                                               /*bInitialState=*/FALSE, NULL)) {
    assert(m_event && m_ready);
    m_monitor_thread = std::thread(&PipeEvent::Monitor, this);
  }

  ~PipeEvent() override {
    if (m_monitor_thread.joinable()) {
      m_stopped = true;
      SetEvent(m_ready);
      CancelIoEx(m_handle, /*lpOverlapped=*/NULL);
      m_monitor_thread.join();
    }
    CloseHandle(m_event);
    CloseHandle(m_ready);
  }

  void WillPoll() override {
    if (WaitForSingleObject(m_event, /*dwMilliseconds=*/0) != WAIT_TIMEOUT) {
      // The thread has already signalled that the data is available. No need
      // for further polling until we consume that event.
      return;
    }
    if (WaitForSingleObject(m_ready, /*dwMilliseconds=*/0) != WAIT_TIMEOUT) {
      // The thread is already waiting for data to become available.
      return;
    }
    // Start waiting.
    SetEvent(m_ready);
  }

  void Disarm() override { ResetEvent(m_event); }

  /// Monitors the handle performing a zero byte read to determine when data is
  /// avaiable.
  void Monitor() {
    // Wait until the MainLoop tells us to start.
    WaitForSingleObject(m_ready, INFINITE);

    do {
      char buf[1];
      DWORD bytes_read = 0;
      OVERLAPPED ov;
      ZeroMemory(&ov, sizeof(ov));
      // Block on a 0-byte read; this will only resume when data is
      // available in the pipe. The pipe must be PIPE_WAIT or this thread
      // will spin.
      BOOL success =
          ReadFile(m_handle, buf, /*nNumberOfBytesToRead=*/0, &bytes_read, &ov);
      DWORD bytes_available = 0;
      DWORD err = GetLastError();
      if (!success && err == ERROR_IO_PENDING) {
        success = GetOverlappedResult(m_handle, &ov, &bytes_read,
                                      /*bWait=*/TRUE);
        err = GetLastError();
      }
      if (success) {
        success =
            PeekNamedPipe(m_handle, NULL, 0, NULL, &bytes_available, NULL);
        err = GetLastError();
      }
      if (success) {
        if (bytes_available == 0) {
          // This can happen with a zero-byte write. Try again.
          continue;
        }
      } else if (err == ERROR_NO_DATA) {
        // The pipe is nonblocking. Try again.
        Sleep(0);
        continue;
      } else if (err == ERROR_OPERATION_ABORTED) {
        // Read may have been cancelled, try again.
        continue;
      }

      // Notify that data is available on the pipe. It's important to set this
      // before clearing m_ready to avoid a race with WillPoll.
      SetEvent(m_event);
      // Stop polling until we're told to resume.
      ResetEvent(m_ready);

      // Wait until the current read is consumed before doing the next read.
      WaitForSingleObject(m_ready, INFINITE);
    } while (!m_stopped);
  }

private:
  HANDLE m_handle;
  HANDLE m_ready;
  std::thread m_monitor_thread;
  std::atomic<bool> m_stopped = false;
};

class SocketEvent : public MainLoopWindows::IOEvent {
public:
  explicit SocketEvent(SOCKET socket)
      : IOEvent(WSACreateEvent()), m_socket(socket) {
    assert(m_event != WSA_INVALID_EVENT);
  }

  ~SocketEvent() override { WSACloseEvent(m_event); }

  void WillPoll() override {
    int result =
        WSAEventSelect(m_socket, m_event, FD_READ | FD_ACCEPT | FD_CLOSE);
    assert(result == 0);
    UNUSED_IF_ASSERT_DISABLED(result);
  }

  void DidPoll() override {
    int result = WSAEventSelect(m_socket, WSA_INVALID_EVENT, 0);
    assert(result == 0);
    UNUSED_IF_ASSERT_DISABLED(result);
  }

  void Disarm() override { WSAResetEvent(m_event); }

  SOCKET m_socket;
};

} // namespace

MainLoopWindows::MainLoopWindows() {
  m_interrupt_event = WSACreateEvent();
  assert(m_interrupt_event != WSA_INVALID_EVENT);
}

MainLoopWindows::~MainLoopWindows() {
  assert(m_read_fds.empty());
  BOOL result = WSACloseEvent(m_interrupt_event);
  assert(result == TRUE);
  UNUSED_IF_ASSERT_DISABLED(result);
}

llvm::Expected<size_t> MainLoopWindows::Poll() {
  std::vector<HANDLE> events;
  events.reserve(m_read_fds.size() + 1);
  for (auto &[_, fd_info] : m_read_fds) {
    fd_info.event->WillPoll();
    events.push_back(fd_info.event->GetHandle());
  }
  events.push_back(m_interrupt_event);

  DWORD result =
      WSAWaitForMultipleEvents(events.size(), events.data(), FALSE,
                               ToTimeout(GetNextWakeupTime()), FALSE);

  for (auto &[_, fd_info] : m_read_fds)
    fd_info.event->DidPoll();

  if (result >= WSA_WAIT_EVENT_0 && result < WSA_WAIT_EVENT_0 + events.size())
    return result - WSA_WAIT_EVENT_0;

  // A timeout is treated as a (premature) signalization of the interrupt event.
  if (result == WSA_WAIT_TIMEOUT)
    return events.size() - 1;

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "WSAWaitForMultipleEvents failed");
}

MainLoopWindows::ReadHandleUP
MainLoopWindows::RegisterReadObject(const IOObjectSP &object_sp,
                                    const Callback &callback, Status &error) {
  if (!object_sp || !object_sp->IsValid()) {
    error = Status::FromErrorString("IO object is not valid.");
    return nullptr;
  }

  IOObject::WaitableHandle waitable_handle = object_sp->GetWaitableHandle();
  assert(waitable_handle != IOObject::kInvalidHandleValue);

  if (m_read_fds.find(waitable_handle) != m_read_fds.end()) {
    error = Status::FromErrorStringWithFormat(
        "File descriptor %p already monitored.", waitable_handle);
    return nullptr;
  }

  if (object_sp->GetFdType() == IOObject::eFDTypeSocket) {
    m_read_fds[waitable_handle] = {
        std::make_unique<SocketEvent>(
            reinterpret_cast<SOCKET>(waitable_handle)),
        callback};
  } else {
    DWORD file_type = GetFileType(waitable_handle);
    if (file_type != FILE_TYPE_PIPE) {
      error = Status::FromErrorStringWithFormat("Unsupported file type %ld",
                                                file_type);
      return nullptr;
    }

    m_read_fds[waitable_handle] = {std::make_unique<PipeEvent>(waitable_handle),
                                   callback};
  }

  return CreateReadHandle(object_sp);
}

void MainLoopWindows::UnregisterReadObject(IOObject::WaitableHandle handle) {
  auto it = m_read_fds.find(handle);
  assert(it != m_read_fds.end());
  m_read_fds.erase(it);
}

Status MainLoopWindows::Run() {
  m_terminate_request = false;

  Status error;

  while (!m_terminate_request) {
    llvm::Expected<size_t> signaled_event = Poll();
    if (!signaled_event)
      return Status::FromError(signaled_event.takeError());

    if (*signaled_event < m_read_fds.size()) {
      auto &KV = *std::next(m_read_fds.begin(), *signaled_event);
      KV.second.event->Disarm();
      KV.second.callback(*this); // Do the work.
    } else {
      assert(*signaled_event == m_read_fds.size());
      WSAResetEvent(m_interrupt_event);
    }
    ProcessCallbacks();
  }
  return Status();
}

void MainLoopWindows::Interrupt() { WSASetEvent(m_interrupt_event); }
