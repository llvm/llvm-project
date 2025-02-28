//===-- MemoryMonitor.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryMonitor.h"
#include "llvm/ADT/ScopeExit.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

using namespace lldb_dap;

#if defined(__APPLE__)
class MemoryMonitorDarwin : public MemoryMonitor {
  using MemoryMonitor::MemoryMonitor;
  void Start() override {
    m_memory_pressure_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,
        0, // system-wide monitoring
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));

    if (!m_memory_pressure_source)
      return;

    dispatch_source_set_event_handler(m_memory_pressure_source, ^{
      dispatch_source_memorypressure_flags_t pressureLevel =
          dispatch_source_get_data(m_memory_pressure_source);
      if (pressureLevel &
          (DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL)) {
        m_callback();
      }
    });
  }

  void Stop() override {
    if (m_memory_pressure_source) {
      dispatch_source_cancel(m_memory_pressure_source);
      dispatch_release(m_memory_pressure_source);
    }
  }

private:
  dispatch_source_t m_memory_pressure_source;
};
#endif

#if defined(__linux__)
static void MonitorThread(std::atomic<bool> &done,
                          MemoryMonitor::Callback callback) {
  struct pollfd fds;
  fds.fd = open("/proc/pressure/memory", O_RDWR | O_NONBLOCK);
  if (fds.fd < 0)
    return;
  fds.events = POLLPRI;

  auto cleanup = llvm::make_scope_exit([&]() { close(fds.fd); });

  // Detect a 50ms stall in a 2 second time window.
  const char trig[] = "some 50000 2000000";
  if (write(fds.fd, trig, strlen(trig) + 1) < 0)
    return;

  while (!done) {
    int n = poll(&fds, 1, 1000);
    if (n > 0) {
      if (fds.revents & POLLERR)
        return;
      if (fds.revents & POLLPRI)
        callback();
    }
  }
}

class MemoryMonitorLinux : public MemoryMonitor {
public:
  using MemoryMonitor::MemoryMonitor;

  void Start() override {
    m_memory_pressure_thread =
        std::thread(MonitorThread, std::ref(m_done), m_callback);
  }

  void Stop() override {
    if (m_memory_pressure_thread.joinable()) {
      m_done = true;
      m_memory_pressure_thread.join();
    }
  }

private:
  std::atomic<bool> m_done = false;
  std::thread m_memory_pressure_thread;
};
#endif

std::unique_ptr<MemoryMonitor> MemoryMonitor::Create(Callback callback) {
#if defined(__APPLE__)
  return std::make_unique<MemoryMonitorDarwin>(callback);
#endif

#if defined(__linux__)
  return std::make_unique<MemoryMonitorLinux>(callback);
#endif

  return nullptr;
}
