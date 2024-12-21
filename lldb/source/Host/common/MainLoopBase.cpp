//===-- MainLoopBase.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MainLoopBase.h"
#include <chrono>

using namespace lldb;
using namespace lldb_private;

void MainLoopBase::AddCallback(const Callback &callback, TimePoint point) {
  bool interrupt_needed;
  {
    std::lock_guard<std::mutex> lock{m_callback_mutex};
    // We need to interrupt the main thread if this callback is scheduled to
    // execute at an earlier time than the earliest callback registered so far.
    interrupt_needed = m_callbacks.empty() || point < m_callbacks.top().first;
    m_callbacks.emplace(point, callback);
  }
  if (interrupt_needed)
    Interrupt();
}

void MainLoopBase::ProcessCallbacks() {
  while (true) {
    Callback callback;
    {
      std::lock_guard<std::mutex> lock{m_callback_mutex};
      if (m_callbacks.empty() ||
          std::chrono::steady_clock::now() < m_callbacks.top().first)
        return;
      callback = std::move(m_callbacks.top().second);
      m_callbacks.pop();
    }

    callback(*this);
  }
}

std::optional<MainLoopBase::TimePoint> MainLoopBase::GetNextWakeupTime() {
  std::lock_guard<std::mutex> lock(m_callback_mutex);
  if (m_callbacks.empty())
    return std::nullopt;
  return m_callbacks.top().first;
}
