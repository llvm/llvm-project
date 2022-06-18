#include "TaskTimer.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

void ScopedTaskTimer::ForEachTimedTask(
    std::function<void(const std::string &event,
                       std::chrono::milliseconds duration)>
        callback) {
  for (const auto &kv : m_timed_tasks) {
    callback(kv.first, kv.second);
  }
}

ScopedTaskTimer &TaskTimer::ForThread(lldb::tid_t tid) {
  auto it = m_thread_timers.find(tid);
  if (it == m_thread_timers.end())
    it = m_thread_timers.try_emplace(tid, ScopedTaskTimer{}).first;
  return it->second;
}

ScopedTaskTimer &TaskTimer::ForGlobal() { return m_global_timer; }
