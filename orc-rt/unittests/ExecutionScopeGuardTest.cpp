//===- ExecutionScopeGuardTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests the ExecutionScopeGuard RAII class.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ExecutionScopeGuard.h"
#include "orc-rt/Session.h"
#include "gtest/gtest.h"

#include <atomic>
#include <thread>

using namespace orc_rt;

namespace {

static void noErrors(Error Err) { cantFail(std::move(Err)); }

TEST(ExecutionScopeGuardTest, DefaultConstruction) {
  ExecutionScopeGuard ESG;
  EXPECT_FALSE(ESG);
  EXPECT_EQ(ESG.getSession(), nullptr);
}

TEST(ExecutionScopeGuardTest, ConstructFromSession) {
  Session S(noErrors);
  ExecutionScopeGuard ESG(S);
  EXPECT_TRUE(ESG);
  EXPECT_EQ(ESG.getSession(), &S);
}

TEST(ExecutionScopeGuardTest, CopyConstruction) {
  Session S(noErrors);
  ExecutionScopeGuard ESG1(S);
  ExecutionScopeGuard ESG2(ESG1);

  EXPECT_TRUE(ESG1);
  EXPECT_TRUE(ESG2);
  EXPECT_EQ(ESG1.getSession(), &S);
  EXPECT_EQ(ESG2.getSession(), &S);
}

TEST(ExecutionScopeGuardTest, CopyAssignment) {
  Session S(noErrors);
  ExecutionScopeGuard ESG1(S);
  ExecutionScopeGuard ESG2;

  EXPECT_FALSE(ESG2);
  ESG2 = ESG1;
  EXPECT_TRUE(ESG1);
  EXPECT_TRUE(ESG2);
  EXPECT_EQ(ESG2.getSession(), &S);
}

TEST(ExecutionScopeGuardTest, CopyAssignmentSelfAssign) {
  Session S(noErrors);
  ExecutionScopeGuard ESG(S);

  ESG = static_cast<ExecutionScopeGuard &>(ESG); // Self-assignment
  EXPECT_TRUE(ESG);
  EXPECT_EQ(ESG.getSession(), &S);
}

TEST(ExecutionScopeGuardTest, CopyAssignmentReplacesExisting) {
  Session S1(noErrors);
  Session S2(noErrors);
  ExecutionScopeGuard ESG1(S1);
  ExecutionScopeGuard ESG2(S2);

  ESG1 = ESG2;
  EXPECT_EQ(ESG1.getSession(), &S2);
  EXPECT_EQ(ESG2.getSession(), &S2);
}

TEST(ExecutionScopeGuardTest, MoveConstruction) {
  Session S(noErrors);
  ExecutionScopeGuard ESG1(S);
  ExecutionScopeGuard ESG2(std::move(ESG1));

  EXPECT_FALSE(ESG1);
  EXPECT_TRUE(ESG2);
  EXPECT_EQ(ESG1.getSession(), nullptr);
  EXPECT_EQ(ESG2.getSession(), &S);
}

TEST(ExecutionScopeGuardTest, MoveAssignment) {
  Session S(noErrors);
  ExecutionScopeGuard ESG1(S);
  ExecutionScopeGuard ESG2;

  ESG2 = std::move(ESG1);
  EXPECT_FALSE(ESG1);
  EXPECT_TRUE(ESG2);
  EXPECT_EQ(ESG1.getSession(), nullptr);
  EXPECT_EQ(ESG2.getSession(), &S);
}

TEST(ExecutionScopeGuardTest, MoveAssignmentReplacesExisting) {
  Session S1(noErrors);
  Session S2(noErrors);
  ExecutionScopeGuard ESG1(S1);
  ExecutionScopeGuard ESG2(S2);

  ESG1 = std::move(ESG2);
  EXPECT_EQ(ESG1.getSession(), &S2);
  EXPECT_EQ(ESG2.getSession(), nullptr);
}

TEST(ExecutionScopeGuardTest, Reset) {
  Session S(noErrors);
  ExecutionScopeGuard ESG(S);

  EXPECT_TRUE(ESG);
  ESG.reset();
  EXPECT_FALSE(ESG);
  EXPECT_EQ(ESG.getSession(), nullptr);
}

TEST(ExecutionScopeGuardTest, ResetOnDefaultConstructed) {
  ExecutionScopeGuard ESG;
  ESG.reset(); // Should be safe on default-constructed
  EXPECT_FALSE(ESG);
}

TEST(ExecutionScopeGuardTest, ShutdownWaitsForExecutionScope) {
  Session S(noErrors);
  std::atomic<bool> ShutdownStarted = false;
  std::atomic<bool> ShutdownComplete = false;

  {
    ExecutionScopeGuard ESG(S);

    std::thread ShutdownThread([&] {
      ShutdownStarted = true;
      S.shutdown([&] { ShutdownComplete = true; });
    });

    // Give the shutdown thread time to start waiting
    while (!ShutdownStarted)
      std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Shutdown should not complete while we hold the execution scope
    EXPECT_FALSE(ShutdownComplete);

    ShutdownThread.detach();
    // ESG goes out of scope here, releasing the execution scope
  }

  // Now shutdown can complete
  S.waitForShutdown();
  EXPECT_TRUE(ShutdownComplete);
}

TEST(ExecutionScopeGuardTest, MultipleGuards) {
  Session S(noErrors);
  std::atomic<bool> ShutdownComplete = false;

  {
    ExecutionScopeGuard ESG1(S);
    {
      ExecutionScopeGuard ESG2(S);
      ExecutionScopeGuard ESG3(ESG1); // Copy

      std::thread ShutdownThread(
          [&] { S.shutdown([&] { ShutdownComplete = true; }); });
      ShutdownThread.detach();

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      EXPECT_FALSE(ShutdownComplete); // Still have 3 guards
      // ESG2 and ESG3 go out of scope
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(ShutdownComplete); // Still have ESG1
    // ESG1 goes out of scope
  }

  S.waitForShutdown();
  EXPECT_TRUE(ShutdownComplete);
}

} // end anonymous namespace
