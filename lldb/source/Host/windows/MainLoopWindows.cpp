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
#include "lldb/Utility/Status.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WindowsError.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <ctime>
#include <vector>
#include <winsock2.h>

using namespace lldb;
using namespace lldb_private;

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
  for (auto &[fd, fd_info] : m_read_fds) {
    // short circuit waiting if a handle is already ready.
    if (fd_info.object_sp->HasReadableData())
      return events.size();
    events.push_back(fd);
  }
  events.push_back(m_interrupt_event);

  while (true) {
    DWORD timeout = INFINITY;
    std::optional<lldb_private::MainLoopBase::TimePoint> deadline =
        GetNextWakeupTime();
    if (deadline) {
      // Check how much time is remaining, we may have woken up early for an
      // unrelated reason on a file descriptor (e.g. a stat was triggered).
      std::chrono::milliseconds remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline.value() - std::chrono::steady_clock::now());
      if (remaining.count() <= 0)
        return events.size() - 1;
      timeout = remaining.count();
    }

    DWORD result =
        WaitForMultipleObjects(events.size(), events.data(), FALSE, timeout);

    // A timeout is treated as a (premature) signalization of the interrupt
    // event.
    if (result == WAIT_TIMEOUT)
      return events.size() - 1;

    if (result == WAIT_FAILED)
      return llvm::createStringError(llvm::mapLastWindowsError(),
                                     "WaitForMultipleObjects failed");

    // check if interrupt requested.
    if (result == WAIT_OBJECT_0 + events.size())
      return result - WAIT_OBJECT_0;

    // An object may be signaled before data is ready for reading, verify it has
    // data.
    if (result >= WAIT_OBJECT_0 &&
        result < WAIT_OBJECT_0 + (events.size() - 1) &&
        std::next(m_read_fds.begin(), result - WAIT_OBJECT_0)
            ->second.object_sp->HasReadableData())
      return result - WAIT_OBJECT_0;

    // If no handles are actually ready then yield the thread to allow the CPU
    // to progress.
    std::this_thread::yield();
  }

  llvm_unreachable();
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

  const bool inserted =
      m_read_fds.try_emplace(waitable_handle, FdInfo{object_sp, callback})
          .second;
  if (!inserted) {
    error = Status::FromErrorStringWithFormat(
        "File descriptor %d already monitored.", waitable_handle);
    return nullptr;
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
