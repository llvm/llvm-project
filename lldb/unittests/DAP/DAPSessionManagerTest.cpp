//===-- DAPSessionManagerTest.cpp ----------------------------------------===//
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

using namespace lldb_dap;
using namespace lldb;
using namespace lldb_dap_tests;

class DAPSessionManagerTest : public DAPTestBase {};

TEST_F(DAPSessionManagerTest, GetInstanceReturnsSameSingleton) {
  DAPSessionManager &instance1 = DAPSessionManager::GetInstance();
  DAPSessionManager &instance2 = DAPSessionManager::GetInstance();

  EXPECT_EQ(&instance1, &instance2);
}

// UnregisterSession uses std::notify_all_at_thread_exit, so it must be called
// from a separate thread to properly release the mutex on thread exit.
TEST_F(DAPSessionManagerTest, RegisterAndUnregisterSession) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  // Initially not registered.
  std::vector<DAP *> sessions_before = manager.GetActiveSessions();
  EXPECT_EQ(
      std::count(sessions_before.begin(), sessions_before.end(), dap.get()), 0);

  manager.RegisterSession(&loop, dap.get());

  // Should be in active sessions after registration.
  std::vector<DAP *> sessions_after = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions_after.begin(), sessions_after.end(), dap.get()),
            1);

  // Unregister.
  std::thread unregister_thread([&]() { manager.UnregisterSession(&loop); });

  unregister_thread.join();

  // There should no longer be active sessions.
  std::vector<DAP *> sessions_final = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions_final.begin(), sessions_final.end(), dap.get()),
            0);
}

TEST_F(DAPSessionManagerTest, DisconnectAllSessions) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  manager.RegisterSession(&loop, dap.get());

  std::vector<DAP *> sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 1);

  manager.DisconnectAllSessions();

  // DisconnectAllSessions shutdown but doesn't wait for
  // sessions to complete or remove them from the active sessions map.
  sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 1);

  std::thread unregister_thread([&]() { manager.UnregisterSession(&loop); });
  unregister_thread.join();
}

TEST_F(DAPSessionManagerTest, WaitForAllSessionsToDisconnect) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  manager.RegisterSession(&loop, dap.get());

  std::vector<DAP *> sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 1);

  // Unregister after a delay to test blocking behavior.
  std::thread unregister_thread([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    manager.UnregisterSession(&loop);
  });

  // WaitForAllSessionsToDisconnect should block until unregistered.
  auto start = std::chrono::steady_clock::now();
  llvm::Error err = manager.WaitForAllSessionsToDisconnect();
  EXPECT_FALSE(err);
  auto duration = std::chrono::steady_clock::now() - start;

  // Verify it waited at least 100ms.
  EXPECT_GE(duration, std::chrono::milliseconds(100));

  // Session should be unregistered now.
  sessions = manager.GetActiveSessions();
  EXPECT_EQ(std::count(sessions.begin(), sessions.end(), dap.get()), 0);

  unregister_thread.join();
}
