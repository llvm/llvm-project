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

TEST_F(DAPSessionManagerTest, StoreAndTakeTarget) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  SBDebugger debugger = SBDebugger::Create();
  ASSERT_TRUE(debugger.IsValid());
  // Create an empty target (no executable) since we're only testing session
  // management functionality, not actual debugging operations.
  SBTarget target = debugger.CreateTarget("");
  ASSERT_TRUE(target.IsValid());

  uint32_t target_id = target.GetGloballyUniqueID();
  manager.StoreTargetById(target_id, target);

  // Retrieval consumes the target (removes from map).
  std::optional<SBTarget> retrieved = manager.TakeTargetById(target_id);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_TRUE(retrieved->IsValid());
  // Verify we got the correct target back.
  EXPECT_EQ(retrieved->GetDebugger().GetID(), debugger.GetID());
  EXPECT_EQ(*retrieved, target);

  // Second retrieval should fail (one-time retrieval semantics).
  std::optional<SBTarget> second_retrieval = manager.TakeTargetById(target_id);
  EXPECT_FALSE(second_retrieval.has_value());

  SBDebugger::Destroy(debugger);
}

TEST_F(DAPSessionManagerTest, GetTargetWithInvalidId) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  std::optional<SBTarget> result = manager.TakeTargetById(99999);

  EXPECT_FALSE(result.has_value());
}

TEST_F(DAPSessionManagerTest, MultipleTargetsWithDifferentIds) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  SBDebugger debugger1 = SBDebugger::Create();
  SBDebugger debugger2 = SBDebugger::Create();
  ASSERT_TRUE(debugger1.IsValid());
  ASSERT_TRUE(debugger2.IsValid());
  // Create empty targets (no executable) since we're only testing session
  // management functionality, not actual debugging operations.
  SBTarget target1 = debugger1.CreateTarget("");
  SBTarget target2 = debugger2.CreateTarget("");
  ASSERT_TRUE(target1.IsValid());
  ASSERT_TRUE(target2.IsValid());

  uint32_t target_id_1 = target1.GetGloballyUniqueID();
  uint32_t target_id_2 = target2.GetGloballyUniqueID();

  // Sanity check the targets have distinct IDs.
  EXPECT_NE(target_id_1, target_id_2);

  manager.StoreTargetById(target_id_1, target1);
  manager.StoreTargetById(target_id_2, target2);

  std::optional<SBTarget> retrieved1 = manager.TakeTargetById(target_id_1);
  ASSERT_TRUE(retrieved1.has_value());
  EXPECT_TRUE(retrieved1->IsValid());
  // Verify we got the correct target by comparing debugger and target IDs.
  EXPECT_EQ(retrieved1->GetDebugger().GetID(), debugger1.GetID());
  EXPECT_EQ(*retrieved1, target1);

  std::optional<SBTarget> retrieved2 = manager.TakeTargetById(target_id_2);
  ASSERT_TRUE(retrieved2.has_value());
  EXPECT_TRUE(retrieved2->IsValid());
  EXPECT_EQ(retrieved2->GetDebugger().GetID(), debugger2.GetID());
  EXPECT_EQ(*retrieved2, target2);

  SBDebugger::Destroy(debugger1);
  SBDebugger::Destroy(debugger2);
}

TEST_F(DAPSessionManagerTest, CleanupSharedResources) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();

  SBDebugger debugger = SBDebugger::Create();
  ASSERT_TRUE(debugger.IsValid());
  // Create empty targets (no executable) since we're only testing session
  // management functionality, not actual debugging operations.
  SBTarget target1 = debugger.CreateTarget("");
  SBTarget target2 = debugger.CreateTarget("");
  ASSERT_TRUE(target1.IsValid());
  ASSERT_TRUE(target2.IsValid());

  uint32_t target_id_1 = target1.GetGloballyUniqueID();
  uint32_t target_id_2 = target2.GetGloballyUniqueID();

  // Sanity check the targets have distinct IDs.
  EXPECT_NE(target_id_1, target_id_2);

  // Store multiple targets to verify cleanup removes them all.
  manager.StoreTargetById(target_id_1, target1);
  manager.StoreTargetById(target_id_2, target2);

  // Cleanup should remove all stored targets.
  manager.CleanupSharedResources();

  // Verify both targets are gone after cleanup.
  std::optional<SBTarget> result1 = manager.TakeTargetById(target_id_1);
  EXPECT_FALSE(result1.has_value());

  std::optional<SBTarget> result2 = manager.TakeTargetById(target_id_2);
  EXPECT_FALSE(result2.has_value());

  SBDebugger::Destroy(debugger);
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
