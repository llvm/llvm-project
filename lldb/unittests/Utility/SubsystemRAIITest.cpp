//===-- SubsystemRAIITest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

#include "TestingSupport/SubsystemRAII.h"

using namespace lldb_private;

namespace {

enum class SystemState {
  /// Start state of the subsystem.
  Start,
  /// Initialize has been called but Terminate hasn't been called yet.
  Initialized,
  /// Terminate has been called.
  Terminated
};

struct TestSubsystem {
  static SystemState state;
  static void Initialize() {
    assert(state == SystemState::Start);
    state = SystemState::Initialized;
  }
  static void Terminate() {
    assert(state == SystemState::Initialized);
    state = SystemState::Terminated;
  }
};
} // namespace

SystemState TestSubsystem::state = SystemState::Start;

TEST(SubsystemRAIITest, NormalSubsystem) {
  // Tests that SubsystemRAII handles Initialize functions that return void.
  EXPECT_EQ(SystemState::Start, TestSubsystem::state);
  {
    SubsystemRAII<TestSubsystem> subsystem;
    EXPECT_EQ(SystemState::Initialized, TestSubsystem::state);
  }
  EXPECT_EQ(SystemState::Terminated, TestSubsystem::state);
}

static const char *SubsystemErrorString = "Initialize failed";

namespace {
struct TestSubsystemWithError {
  static SystemState state;
  static bool will_fail;
  static llvm::Error Initialize() {
    assert(state == SystemState::Start);
    state = SystemState::Initialized;
    if (will_fail)
      return llvm::make_error<llvm::StringError>(
          SubsystemErrorString, llvm::inconvertibleErrorCode());
    return llvm::Error::success();
  }
  static void Terminate() {
    assert(state == SystemState::Initialized);
    state = SystemState::Terminated;
  }
  /// Reset the subsystem to the default state for testing.
  static void Reset() { state = SystemState::Start; }
};
} // namespace

SystemState TestSubsystemWithError::state = SystemState::Start;
bool TestSubsystemWithError::will_fail = false;

TEST(SubsystemRAIITest, SubsystemWithErrorSuccess) {
  // Tests that SubsystemRAII handles llvm::success() returned from
  // Initialize.
  TestSubsystemWithError::Reset();
  EXPECT_EQ(SystemState::Start, TestSubsystemWithError::state);
  {
    TestSubsystemWithError::will_fail = false;
    SubsystemRAII<TestSubsystemWithError> subsystem;
    EXPECT_EQ(SystemState::Initialized, TestSubsystemWithError::state);
  }
  EXPECT_EQ(SystemState::Terminated, TestSubsystemWithError::state);
}

TEST(SubsystemRAIITest, SubsystemWithErrorFailure) {
  // Tests that SubsystemRAII handles any errors returned from
  // Initialize.
  TestSubsystemWithError::Reset();
  EXPECT_EQ(SystemState::Start, TestSubsystemWithError::state);
  TestSubsystemWithError::will_fail = true;
  EXPECT_FATAL_FAILURE(SubsystemRAII<TestSubsystemWithError> subsystem,
                       SubsystemErrorString);
}
