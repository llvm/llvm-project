//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the DAPSessionManager and
/// ManagedEventThread classes, which are used to multiple concurrent DAP
/// sessions in a single lldb-dap process.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H
#define LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H

#include "DAPForward.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace lldb_dap {

class ManagedEventThread {
public:
  // Constructor declaration
  ManagedEventThread(lldb::SBBroadcaster broadcaster, std::thread t);

  ~ManagedEventThread();

  ManagedEventThread(const ManagedEventThread &) = delete;
  ManagedEventThread &operator=(const ManagedEventThread &) = delete;

private:
  lldb::SBBroadcaster m_broadcaster;
  std::thread m_event_thread;
};

/// Global DAP session manager that manages multiple concurrent DAP sessions in
/// a single lldb-dap process. Handles session lifecycle tracking, coordinates
/// shared debugger event threads, and facilitates target handoff between
/// sessions for dynamically created targets.
class SessionManager {
public:
  /// Get the singleton instance of the DAP session manager.
  static SessionManager &GetInstance();

  /// RAII Session tracking.
  class SessionHandle {
  public:
    SessionManager &manager;
    DAP &dap;

    SessionHandle(SessionManager &manager, DAP &dap)
        : manager(manager), dap(dap) {}

    ~SessionHandle() {
      std::scoped_lock<std::mutex> lock(manager.m_sessions_mutex);

      auto index = manager.m_active_sessions.find(&dap);
      assert(index != manager.m_active_sessions.end());
      manager.m_active_sessions.erase(index);
      manager.m_sessions_condition.notify_all();
    }
  };

  /// Register a DAP session.
  SessionHandle Register(DAP &dap);

  /// Get all active DAP sessions.
  std::vector<DAP *> GetActiveSessions();

  /// Disconnect all registered sessions by calling Disconnect() on
  /// each and requesting their event loops to terminate. Used during
  /// shutdown to force all sessions to begin disconnecting.
  void DisconnectAllSessions();

  /// Block until all sessions disconnect and unregister. Returns an error if
  /// DisconnectAllSessions() was called and any disconnection failed.
  llvm::Error WaitForAllSessionsToDisconnect();

  /// Get or create event thread for a specific debugger.
  std::shared_ptr<ManagedEventThread>
  GetEventThreadForDebugger(lldb::SBDebugger debugger, DAP *requesting_dap);

  /// Find the DAP instance that owns the given target.
  DAP *FindDAPForTarget(lldb::SBTarget target);

  /// Static convenience method for FindDAPForTarget.
  static DAP *FindDAP(lldb::SBTarget target) {
    return GetInstance().FindDAPForTarget(target);
  }

  /// Clean up expired event threads from the collection.
  void ReleaseExpiredEventThreads();

private:
  SessionManager() = default;
  ~SessionManager() = default;

  // Non-copyable and non-movable.
  SessionManager(const SessionManager &) = delete;
  SessionManager &operator=(const SessionManager &) = delete;
  SessionManager(SessionManager &&) = delete;
  SessionManager &operator=(SessionManager &&) = delete;

  bool m_client_failed = false;
  std::mutex m_sessions_mutex;
  std::condition_variable m_sessions_condition;
  std::set<DAP *> m_active_sessions;

  /// Map from debugger ID to its event thread, used when multiple DAP sessions
  /// share the same debugger instance.
  std::map<lldb::user_id_t, std::weak_ptr<ManagedEventThread>>
      m_debugger_event_threads;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H
