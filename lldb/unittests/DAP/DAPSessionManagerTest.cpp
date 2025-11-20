//===-- SessionManagerTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPSessionManager.h"
#include "TestBase.h"
#include "lldb/API/SBDebugger.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <future>

using namespace lldb_dap;
using namespace lldb;
using namespace lldb_dap_tests;

class SessionManagerTest : public DAPTestBase {};

TEST_F(SessionManagerTest, GetInstanceReturnsSameSingleton) {
  SessionManager &instance1 = SessionManager::GetInstance();
  SessionManager &instance2 = SessionManager::GetInstance();

  EXPECT_EQ(&instance1, &instance2);
}

// UnregisterSession uses std::notify_all_at_thread_exit, so it must be called
// from a separate thread to properly release the mutex on thread exit.
TEST_F(SessionManagerTest, RegisterAndUnregisterSession) {
  SessionManager &manager = SessionManager::GetInstance();

  // Initially not registered.
  std::vector<DAP *> sessions_before = manager.GetActiveSessions();
  EXPECT_EQ(
      std::count(sessions_before.begin(), sessions_before.end(), dap.get()), 0);

  {
    auto handle = manager.Register(*dap.get());

    // Should be in active sessions after registration.
    std::vector<DAP *> sessions_after = manager.GetActiveSessions();
    EXPECT_EQ(
        std::count(sessions_after.begin(), sessions_after.end(), dap.get()), 1);
  }

  // There should no longer be active sessions.
  std::vector<DAP *> sessions_final = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions_final.begin(), sessions_final.end(), dap.get()),
            0);
}

TEST_F(SessionManagerTest, DisconnectAllSessions) {
  SessionManager &manager = SessionManager::GetInstance();

  auto handle = manager.Register(*dap.get());

  std::vector<DAP *> sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 1);

  manager.DisconnectAllSessions();

  // DisconnectAllSessions shutdown but doesn't wait for
  // sessions to complete or remove them from the active sessions map.
  sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 1);
}

TEST_F(SessionManagerTest, WaitForAllSessionsToDisconnect) {
  SessionManager &manager = SessionManager::GetInstance();

  std::promise<void> registered_promise;
  std::promise<void> unregistered_promise;

  // Register the session after a delay to test blocking behavior.
  std::thread session_thread([&]() {
    auto handle = manager.Register(*dap.get());
    registered_promise.set_value();
    unregistered_promise.get_future().wait();
  });

  registered_promise.get_future().wait();

  std::vector<DAP *> sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 1);

  // Trigger the session_thread to return, which should unregister the session.
  unregistered_promise.set_value();
  // WaitForAllSessionsToDisconnect should block until unregistered.
  llvm::Error err = manager.WaitForAllSessionsToDisconnect();
  EXPECT_FALSE(err);

  // Session should be unregistered now.
  sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 0);

  session_thread.join();
}
