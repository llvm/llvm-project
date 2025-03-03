//===-- MemoryMonitor.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MemoryMonitor.h"
#include "llvm/ADT/ScopeExit.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

using namespace lldb_private;

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

#if !defined(__APPLE__)
std::unique_ptr<MemoryMonitor> MemoryMonitor::Create(Callback callback) {
#if defined(__linux__)
  return std::make_unique<MemoryMonitorLinux>(callback);
#endif
  return nullptr;
}
#endif
