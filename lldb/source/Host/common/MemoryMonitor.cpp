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
#include <cstddef>
#include <cstdio>
#include <cstring>

#if defined(__linux__)
#include "lldb/Host/posix/Support.h"
#include "llvm/Support/LineIterator.h"
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/poll.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#include <atomic>
#include <windows.h>
#endif

using namespace lldb_private;

#if defined(__linux__)
class MemoryMonitorLinux : public MemoryMonitor {
public:
  using MemoryMonitor::MemoryMonitor;

  explicit MemoryMonitorLinux(Callback callback)
      : MemoryMonitor(std::move(callback)),
        m_stop_fd(::eventfd(0, EFD_NONBLOCK)) {}

  ~MemoryMonitorLinux() {
    if (m_memory_monitor_thread.IsJoinable())
      m_memory_monitor_thread.Join(nullptr);
    if (m_stop_fd != -1)
      ::close(m_stop_fd);
  }

  void Start() override {
    if (m_stop_fd < 0) {
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::Host),
          llvm::errorCodeToError(llvm::errnoAsErrorCode()),
          "failed to create stop file descriptor for memory monitor: {0}");
      return;
    }

    llvm::Expected<HostThread> memory_monitor_thread =
        ThreadLauncher::LaunchThread("memory.monitor",
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
      if (m_stop_fd != -1)
        ::eventfd_write(m_stop_fd, 1);
      m_memory_monitor_thread.Join(nullptr);
    }
  }

private:
  lldb::thread_result_t MonitorThread() {
    constexpr size_t pressure_idx = 0;
    constexpr size_t stop_idx = 1;
    constexpr size_t fd_count = 2;
    std::array<pollfd, fd_count> pfds{};

    // Setup stop file descriptor.
    pfds[stop_idx].fd = m_stop_fd;
    pfds[stop_idx].events = POLLIN;

    // Setup pressure file descriptor.
    pfds[pressure_idx].fd =
        ::open("/proc/pressure/memory", O_RDWR | O_NONBLOCK);
    if (pfds[pressure_idx].fd < 0)
      return {};
    pfds[pressure_idx].events = POLLPRI;

    llvm::scope_exit cleanup([&]() { ::close(pfds[pressure_idx].fd); });

    // Detect a 200ms stall in a 2 second time window.
    constexpr llvm::StringRef trigger = "some 200000 2000000";
    if (::write(pfds[pressure_idx].fd, trigger.data(), trigger.size() + 1) < 0)
      return {};

    while (true) {
      constexpr int timeout_infinite = -1;
      const int n = ::poll(pfds.data(), pfds.size(), timeout_infinite);
      if (n > 0) {
        // Handle stop event.
        if (pfds[stop_idx].revents & (POLLIN | POLLERR))
          return {};

        const short pressure_revents = pfds[stop_idx].revents;
        if (pressure_revents & POLLERR)
          return {};
        if (pressure_revents & POLLPRI) {
          if (const std::optional<bool> is_low_opt = IsLowMemory();
              is_low_opt && *is_low_opt)
            m_callback();
        }
      }
    }
    return {};
  }

  static std::optional<bool> IsLowMemory() {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_err =
        getProcFile("meminfo");

    if (!buffer_or_err)
      return std::nullopt;

    uint64_t mem_total = 0;
    uint64_t mem_available = 0;
    const int radix = 10;
    bool parse_error = false;

    for (llvm::line_iterator iter(**buffer_or_err, true); !iter.is_at_end();
         ++iter) {
      llvm::StringRef line = *iter;
      if (line.consume_front("MemTotal:"))
        parse_error = line.ltrim().consumeInteger(radix, mem_total);
      else if (line.consume_front("MemAvailable:"))
        parse_error = line.ltrim().consumeInteger(radix, mem_available);

      if (parse_error)
        return std::nullopt;

      if (mem_total && mem_available)
        break;
    }

    if (mem_total == 0)
      return std::nullopt;

    if (mem_available == 0) // We are actually out of memory.
      return true;

    const uint64_t approx_memory_percent = (mem_available * 100) / mem_total;
    const uint64_t low_memory_percent = 20;
    return approx_memory_percent < low_memory_percent;
  }

  int m_stop_fd = -1;
  HostThread m_memory_monitor_thread;
};
#elif defined(_WIN32)

class MemoryMonitorWindows : public MemoryMonitor {
public:
  using MemoryMonitor::MemoryMonitor;

  lldb::thread_result_t MonitorThread() {
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
#endif

#if !defined(__APPLE__)
std::unique_ptr<MemoryMonitor> MemoryMonitor::Create(Callback callback) {
#if defined(__linux__)
  return std::make_unique<MemoryMonitorLinux>(std::move(callback));
#elif defined(_WIN32)
  return std::make_unique<MemoryMonitorWindows>(std::move(callback));
#else
  return nullptr;
#endif
}
#endif
