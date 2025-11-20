//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "DAPSessionManager.h"
#include "DAP.h"
#include "DAPLog.h"
#include "EventHelper.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBTarget.h"
#include "llvm/Support/WithColor.h"
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

SessionManager &SessionManager::GetInstance() {
  static std::once_flag initialized;
  static SessionManager *instance = nullptr; // NOTE: intentional leak to avoid
                                             // issues with C++ destructor chain

  std::call_once(initialized, []() { instance = new SessionManager(); });

  return *instance;
}

SessionManager::SessionHandle SessionManager::Register(DAP &dap) {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  m_active_sessions.insert(&dap);
  return SessionHandle(*this, dap);
}

std::vector<DAP *> SessionManager::GetActiveSessions() {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  return std::vector<DAP *>(m_active_sessions.begin(), m_active_sessions.end());
}

void SessionManager::DisconnectAllSessions() {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);
  m_client_failed = false;
  for (auto *dap : m_active_sessions)
    if (llvm::Error error = dap->Disconnect()) {
      m_client_failed = true;
      llvm::WithColor::error() << "DAP client disconnected failed: "
                               << llvm::toString(std::move(error)) << "\n";
    }
}

llvm::Error SessionManager::WaitForAllSessionsToDisconnect() {
  std::unique_lock<std::mutex> lock(m_sessions_mutex);
  m_sessions_condition.wait(lock, [this] { return m_active_sessions.empty(); });

  // Check if any disconnection failed and return appropriate error.
  if (m_client_failed)
    return llvm::make_error<llvm::StringError>(
        "disconnecting all clients failed", llvm::inconvertibleErrorCode());

  return llvm::Error::success();
}

std::shared_ptr<ManagedEventThread>
SessionManager::GetEventThreadForDebugger(lldb::SBDebugger debugger,
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
                  requesting_dap->GetClientName(),
                  std::ref(requesting_dap->log)));
  m_debugger_event_threads[debugger_id] = new_thread_sp;
  return new_thread_sp;
}

DAP *SessionManager::FindDAPForTarget(lldb::SBTarget target) {
  std::lock_guard<std::mutex> lock(m_sessions_mutex);

  for (auto *dap : m_active_sessions)
    if (dap && dap->target.IsValid() && dap->target == target)
      return dap;

  return nullptr;
}

void SessionManager::ReleaseExpiredEventThreads() {
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
