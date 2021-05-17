//===- DebugActionTest.cpp - Debug Action Tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugAction.h"
#include "gmock/gmock.h"

// DebugActionManager is only enabled in DEBUG mode.
#ifndef NDEBUG

using namespace mlir;

namespace {
struct SimpleAction : public DebugAction<> {
  static StringRef getTag() { return "simple-action"; }
  static StringRef getDescription() { return "simple-action-description"; }
};
struct ParametricAction : public DebugAction<bool> {
  static StringRef getTag() { return "param-action"; }
  static StringRef getDescription() { return "param-action-description"; }
};

TEST(DebugActionTest, GenericHandler) {
  DebugActionManager manager;

  // A generic handler that always executes the simple action, but not the
  // parametric action.
  struct GenericHandler : public DebugActionManager::GenericHandler {
    FailureOr<bool> shouldExecute(StringRef tag, StringRef desc) final {
      if (tag == SimpleAction::getTag()) {
        EXPECT_EQ(desc, SimpleAction::getDescription());
        return true;
      }

      EXPECT_EQ(tag, ParametricAction::getTag());
      EXPECT_EQ(desc, ParametricAction::getDescription());
      return false;
    }
  };
  manager.registerActionHandler<GenericHandler>();

  EXPECT_TRUE(manager.shouldExecute<SimpleAction>());
  EXPECT_FALSE(manager.shouldExecute<ParametricAction>(true));
}

TEST(DebugActionTest, ActionSpecificHandler) {
  DebugActionManager manager;

  // Handler that simply uses the input as the decider.
  struct ActionSpecificHandler : public ParametricAction::Handler {
    FailureOr<bool> shouldExecute(bool shouldExecuteParam) final {
      return shouldExecuteParam;
    }
  };
  manager.registerActionHandler<ActionSpecificHandler>();

  EXPECT_TRUE(manager.shouldExecute<ParametricAction>(true));
  EXPECT_FALSE(manager.shouldExecute<ParametricAction>(false));

  // There is no handler for the simple action, so it is always executed.
  EXPECT_TRUE(manager.shouldExecute<SimpleAction>());
}

TEST(DebugActionTest, DebugCounterHandler) {
  DebugActionManager manager;

  // Handler that uses the number of action executions as the decider.
  struct DebugCounterHandler : public SimpleAction::Handler {
    FailureOr<bool> shouldExecute() final {
      return numExecutions++ < 3;
    }
    unsigned numExecutions = 0;
  };
  manager.registerActionHandler<DebugCounterHandler>();

  // Check that the action is executed 3 times, but no more after.
  EXPECT_TRUE(manager.shouldExecute<SimpleAction>());
  EXPECT_TRUE(manager.shouldExecute<SimpleAction>());
  EXPECT_TRUE(manager.shouldExecute<SimpleAction>());
  EXPECT_FALSE(manager.shouldExecute<SimpleAction>());
  EXPECT_FALSE(manager.shouldExecute<SimpleAction>());
}

} // namespace

#endif
