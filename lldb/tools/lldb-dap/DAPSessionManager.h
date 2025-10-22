//===-- DAPSessionManager.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H
#define LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace lldb_dap {

// Forward declarations
struct DAP;

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
class DAPSessionManager {
public:
  /// Get the singleton instance of the DAP session manager.
  static DAPSessionManager &GetInstance();

  /// Register a DAP session.
  void RegisterSession(lldb_private::MainLoop *loop, DAP *dap);

  /// Unregister a DAP session. Called by sessions when they complete their
  /// disconnection, which unblocks WaitForAllSessionsToDisconnect().
  void UnregisterSession(lldb_private::MainLoop *loop);

  /// Get all active DAP sessions.
  std::vector<DAP *> GetActiveSessions();

  /// Disconnect all registered sessions by calling Disconnect() on
  /// each and requesting their event loops to terminate. Used during
  /// shutdown to force all sessions to begin disconnecting.
  void DisconnectAllSessions();

  /// Block until all sessions disconnect and unregister. Returns an error if
  /// DisconnectAllSessions() was called and any disconnection failed.
  llvm::Error WaitForAllSessionsToDisconnect();

  /// Store a target for later retrieval by another session.
  ///
  /// When a new target is dynamically created (e.g., via scripting hooks or
  /// child process debugging), it needs to be handed off to a new DAP session.
  /// This method stores the target in a temporary map using its globally unique
  /// ID so that the new session can retrieve it via TakeTargetById().
  ///
  /// \param target_id The globally unique ID of the target (from
  ///                  SBTarget::GetGloballyUniqueID()).
  /// \param target The target to store.
  void StoreTargetById(lldb::user_id_t target_id, lldb::SBTarget target);

  /// Retrieve and remove a stored target by its globally unique target ID.
  ///
  /// This method is called during the attach request when a new DAP session
  /// wants to attach to a dynamically created target. The target is removed
  /// from the map after retrieval because:
  /// 1. Each target should only be attached to by one DAP session
  /// 2. Once attached, the DAP session takes ownership of the target
  /// 3. Keeping the target in the map would prevent proper cleanup
  ///
  /// \param target_id The globally unique ID of the target to retrieve.
  /// \return The target if found, std::nullopt otherwise.
  std::optional<lldb::SBTarget> TakeTargetById(lldb::user_id_t target_id);

  /// Get or create event thread for a specific debugger.
  std::shared_ptr<ManagedEventThread>
  GetEventThreadForDebugger(lldb::SBDebugger debugger, DAP *requesting_dap);

  /// Find the DAP instance that owns the given target.
  DAP *FindDAPForTarget(lldb::SBTarget target);

  /// Static convenience method for FindDAPForTarget.
  static DAP *FindDAP(lldb::SBTarget target) {
    return GetInstance().FindDAPForTarget(target);
  }

  /// Clean up shared resources when the last session exits.
  void CleanupSharedResources();

  /// Clean up expired event threads from the collection.
  void ReleaseExpiredEventThreads();

private:
  DAPSessionManager() = default;
  ~DAPSessionManager() = default;

  // Non-copyable and non-movable.
  DAPSessionManager(const DAPSessionManager &) = delete;
  DAPSessionManager &operator=(const DAPSessionManager &) = delete;
  DAPSessionManager(DAPSessionManager &&) = delete;
  DAPSessionManager &operator=(DAPSessionManager &&) = delete;

  bool m_client_failed = false;
  std::mutex m_sessions_mutex;
  std::condition_variable m_sessions_condition;
  std::map<lldb_private::MainLoop *, DAP *> m_active_sessions;

  /// Map from target ID to target for handing off newly created targets
  /// between sessions.
  std::map<lldb::user_id_t, lldb::SBTarget> m_target_map;

  /// Map from debugger ID to its event thread, used when multiple DAP sessions
  /// share the same debugger instance.
  std::map<lldb::user_id_t, std::weak_ptr<ManagedEventThread>>
      m_debugger_event_threads;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H
