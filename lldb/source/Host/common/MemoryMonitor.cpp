//===-- MemoryMonitor.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MemoryMonitor.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace lldb_private;

class MemoryMonitorPoll : public MemoryMonitor {
public:
  using MemoryMonitor::MemoryMonitor;

  lldb::thread_result_t MonitorThread() {
#if defined(__linux__)
    struct pollfd fds;
    fds.fd = open("/proc/pressure/memory", O_RDWR | O_NONBLOCK);
    if (fds.fd < 0)
      return {};
    fds.events = POLLPRI;

    auto cleanup = llvm::make_scope_exit([&]() { close(fds.fd); });

    // Detect a 50ms stall in a 2 second time window.
    const char trig[] = "some 50000 2000000";
    if (write(fds.fd, trig, strlen(trig) + 1) < 0)
      return {};

    while (!m_done) {
      int n = poll(&fds, 1, g_timeout);
      if (n > 0) {
        if (fds.revents & POLLERR)
          return {};
        if (fds.revents & POLLPRI)
          m_callback();
      }
    }
#endif

#if defined(_WIN32)
    HANDLE low_memory_notification =
        CreateMemoryResourceNotification(LowMemoryResourceNotification);
    if (!low_memory_notification)
      return {};

    while (!m_done) {
      if (WaitForSingleObject(low_memory_notification, g_timeout) ==
          WAIT_OBJECT_0) {
        m_callback();
      }
    }
#endif

    return {};
  }

  void Start() override {
    llvm::Expected<HostThread> memory_monitor_thread =
        ThreadLauncher::LaunchThread("lldb.debugger.memory-monitor",
                                     [this] { return MonitorThread(); });
    if (memory_monitor_thread) {
      m_memory_monitor_thread = *memory_monitor_thread;
    } else {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Host), memory_monitor_thread.takeError(),
                     "failed to launch host thread: {0}");
    }
  }

  void Stop() override {
    if (m_memory_monitor_thread.IsJoinable()) {
      m_done = true;
      m_memory_monitor_thread.Join(nullptr);
    }
  }

private:
  static constexpr uint32_t g_timeout = 1000;
  std::atomic<bool> m_done = false;
  HostThread m_memory_monitor_thread;
};

#if !defined(__APPLE__)
std::unique_ptr<MemoryMonitor> MemoryMonitor::Create(Callback callback) {
  return std::make_unique<MemoryMonitorPoll>(callback);
}
#endif
