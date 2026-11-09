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

TEST_F(DAPSessionManagerTest, SharedDebuggerReplAcrossSessions) {
  DAPSessionManager &manager = DAPSessionManager::GetInstance();
  CreateDebugger();
  manager.RegisterSession(&loop, dap.get());

  lldb_private::MainLoop loop2;
  auto [to_client2, to_server2] = TestDAPTransport::createPair(loop2);
  auto dap2 = std::make_unique<DAP>(
      /*log=*/*log,
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<protocol::String>(),
      /*no_lldbinit=*/true,
      /*client_name=*/"test_client_2",
      /*transport=*/*to_client2, /*loop=*/loop2);

  protocol::DAPSession session;
  SBTarget second_target = dap->debugger.CreateTarget("");
  ASSERT_TRUE(second_target.IsValid());
  session.debuggerId = dap->debugger.GetID();
  session.targetId = second_target.GetGloballyUniqueID();
  ASSERT_THAT_ERROR(dap2->InitializeDebugger(session), llvm::Succeeded());

  // Evaluate a command from the second session using the shared REPL.
  llvm::Expected<SBCommandReturnObject> res1 = dap2->EvaluateCommand("version");
  ASSERT_THAT_EXPECTED(res1, llvm::Succeeded());
  EXPECT_TRUE(res1->Succeeded());

  // Enter an interactive sub-IOHandler from dap2 and verify both sessions see
  // the sub-IOHandler as active, then exit it.
  llvm::Expected<SBCommandReturnObject> script_start =
      dap2->EvaluateCommand("script -l python --");
  ASSERT_THAT_EXPECTED(script_start, llvm::Succeeded());
  EXPECT_FALSE(dap->IsCommandInterpreterActive());
  EXPECT_FALSE(dap2->IsCommandInterpreterActive());

  llvm::Expected<SBCommandReturnObject> script_exit =
      dap2->EvaluateCommand("quit()");
  ASSERT_THAT_EXPECTED(script_exit, llvm::Succeeded());
  EXPECT_TRUE(dap->IsCommandInterpreterActive());
  EXPECT_TRUE(dap2->IsCommandInterpreterActive());

  // Unregister and destroy the first session while dap2 is still active; dap2
  // should continue to be able to evaluate REPL commands without a dangling
  // print callback or closed input pipe.
  std::thread unregister_thread([&]() { manager.UnregisterSession(&loop); });
  unregister_thread.join();
  dap.reset();

  llvm::Expected<SBCommandReturnObject> res2 = dap2->EvaluateCommand("version");
  ASSERT_THAT_EXPECTED(res2, llvm::Succeeded());
  EXPECT_TRUE(res2->Succeeded());
}
