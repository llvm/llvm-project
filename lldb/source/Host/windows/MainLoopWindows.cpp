//===-- MainLoopWindows.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/MainLoopWindows.h"
#include "lldb/Host/Config.h"
#include "lldb/Utility/Status.h"
#include "llvm/Config/llvm-config.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <ctime>
#include <vector>
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
  std::vector<WSAEVENT> events;
  events.reserve(m_read_fds.size() + 1);
  for (auto &[fd, info] : m_read_fds) {
    int result = WSAEventSelect(fd, info.event, FD_READ | FD_ACCEPT | FD_CLOSE);
    assert(result == 0);
    UNUSED_IF_ASSERT_DISABLED(result);

    events.push_back(info.event);
  }
  events.push_back(m_interrupt_event);

  DWORD result =
      WSAWaitForMultipleEvents(events.size(), events.data(), FALSE,
                               ToTimeout(GetNextWakeupTime()), FALSE);

  for (auto &fd : m_read_fds) {
    int result = WSAEventSelect(fd.first, WSA_INVALID_EVENT, 0);
    assert(result == 0);
    UNUSED_IF_ASSERT_DISABLED(result);
  }

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
  if (object_sp->GetFdType() != IOObject::eFDTypeSocket) {
    error = Status::FromErrorString(
        "MainLoopWindows: non-socket types unsupported on Windows");
    return nullptr;
  }

  WSAEVENT event = WSACreateEvent();
  if (event == WSA_INVALID_EVENT) {
    error =
        Status::FromErrorStringWithFormat("Cannot create monitoring event.");
    return nullptr;
  }

  const bool inserted =
      m_read_fds
          .try_emplace(object_sp->GetWaitableHandle(), FdInfo{event, callback})
          .second;
  if (!inserted) {
    WSACloseEvent(event);
    error = Status::FromErrorStringWithFormat(
        "File descriptor %d already monitored.",
        object_sp->GetWaitableHandle());
    return nullptr;
  }

  return CreateReadHandle(object_sp);
}

void MainLoopWindows::UnregisterReadObject(IOObject::WaitableHandle handle) {
  auto it = m_read_fds.find(handle);
  assert(it != m_read_fds.end());
  BOOL result = WSACloseEvent(it->second.event);
  assert(result == TRUE);
  UNUSED_IF_ASSERT_DISABLED(result);
  m_read_fds.erase(it);
}

void MainLoopWindows::ProcessReadObject(IOObject::WaitableHandle handle) {
  auto it = m_read_fds.find(handle);
  if (it != m_read_fds.end())
    it->second.callback(*this); // Do the work
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
      WSAResetEvent(KV.second.event);
      ProcessReadObject(KV.first);
    } else {
      assert(*signaled_event == m_read_fds.size());
      WSAResetEvent(m_interrupt_event);
    }
    ProcessCallbacks();
  }
  return Status();
}

void MainLoopWindows::Interrupt() { WSASetEvent(m_interrupt_event); }
