//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "DAPSessionManager.h"
#include "DAP.h"
#include "EventHelper.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBTarget.h"
#include "lldb/Host/MainLoopBase.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/WithColor.h"

#include <chrono>
#include <mutex>

namespace lldb_dap {

ManagedEventThread::ManagedEventThread(lldb::SBBroadcaster broadcaster,
                                       std::thread t)
    : m_broadcaster(broadcaster), m_event_thread(std::move(t)) {}

ManagedEventThread::~ManagedEventThread() {
  if (m_event_thread.joinable()) {
    m_broadcaster.BroadcastEventByType(eBroadcastBitStopEventThread);
    m_event_thread.join();
  }
}

DAPSessionManager &DAPSessionManager::GetInstance() {
  static std::once_flag initialized;
  static DAPSessionManager *instance =
      nullptr; // NOTE: intentional leak to avoid issues with C++ destructor
               // chain

  std::call_once(initialized, []() { instance = new DAPSessionManager(); });

  return *instance;
}

void DAPSessionManager::RegisterSession(lldb_private::MainLoop *loop,
                                        DAP *dap) {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  m_active_sessions[loop] = dap;
}

void DAPSessionManager::UnregisterSession(lldb_private::MainLoop *loop) {
  std::unique_lock<std::mutex> lock(m_sessions_mutex);
  m_active_sessions.erase(loop);
  std::notify_all_at_thread_exit(m_sessions_condition, std::move(lock));
}

std::vector<DAP *> DAPSessionManager::GetActiveSessions() {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  std::vector<DAP *> sessions;
  for (const auto &[loop, dap] : m_active_sessions)
    if (dap)
      sessions.emplace_back(dap);
  return sessions;
}

void DAPSessionManager::DisconnectAllSessions() {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  m_client_failed = false;
  for (auto [loop, dap] : m_active_sessions) {
    if (dap) {
      if (llvm::Error error = dap->Disconnect()) {
        m_client_failed = true;
        llvm::WithColor::error() << "DAP client disconnected failed: "
                                 << llvm::toString(std::move(error)) << "\n";
      }
      loop->AddPendingCallback(
          [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
    }
  }
}

llvm::Error DAPSessionManager::WaitForAllSessionsToDisconnect() {
  std::unique_lock<std::mutex> lock(m_sessions_mutex);
  m_sessions_condition.wait(lock, [this] { return m_active_sessions.empty(); });

  // Check if any disconnection failed and return appropriate error.
  if (m_client_failed)
    return llvm::make_error<llvm::StringError>(
        "disconnecting all clients failed", llvm::inconvertibleErrorCode());

  return llvm::Error::success();
}

std::shared_ptr<ManagedEventThread>
DAPSessionManager::GetEventThreadForDebugger(lldb::SBDebugger debugger,
                                             DAP *requesting_dap) {
  lldb::user_id_t debugger_id = debugger.GetID();
  std::lock_guard<std::mutex> lock(m_sessions_mutex);

  // Try to use shared event thread, if it exists.
  if (auto it = m_debugger_event_threads.find(debugger_id);
      it != m_debugger_event_threads.end()) {
    if (std::shared_ptr<ManagedEventThread> thread_sp = it->second.lock())
      return thread_sp;
    // Our weak pointer has expired.
    m_debugger_event_threads.erase(it);
  }

  // Create a new event thread and store it.
  auto new_thread_sp = std::make_shared<ManagedEventThread>(
      requesting_dap->broadcaster,
      std::thread(EventThread, debugger, requesting_dap->broadcaster,
                  requesting_dap->m_client_name,
                  std::ref(requesting_dap->log)));
  m_debugger_event_threads[debugger_id] = new_thread_sp;
  return new_thread_sp;
}

DAP *DAPSessionManager::FindDAPForTarget(lldb::SBTarget target) {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);

  for (const auto &[loop, dap] : m_active_sessions)
    if (dap && dap->target.IsValid() && dap->target == target)
      return dap;

  return nullptr;
}

void DAPSessionManager::ReleaseExpiredEventThreads() {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  for (auto it = m_debugger_event_threads.begin();
       it != m_debugger_event_threads.end();) {
    // Check if the weak_ptr has expired (no DAP instances are using it
    // anymore).
    if (it->second.expired()) {
      it = m_debugger_event_threads.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace lldb_dap
