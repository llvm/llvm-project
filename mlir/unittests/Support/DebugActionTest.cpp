//===- DebugActionTest.cpp - Debug Action Tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugAction.h"
#include "mlir/Support/TypeID.h"
#include "gmock/gmock.h"

using namespace mlir;

namespace {
struct SimpleAction : DebugAction<SimpleAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleAction)
  static StringRef getTag() { return "simple-action"; }
  static StringRef getDescription() { return "simple-action-description"; }
};
struct OtherSimpleAction : DebugAction<OtherSimpleAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherSimpleAction)
  static StringRef getTag() { return "other-simple-action"; }
  static StringRef getDescription() {
    return "other-simple-action-description";
  }
};
struct ParametricAction : DebugAction<ParametricAction, bool> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParametricAction)
  ParametricAction(bool executeParam) : executeParam(executeParam) {}
  bool executeParam;
  static StringRef getTag() { return "param-action"; }
  static StringRef getDescription() { return "param-action-description"; }
};

TEST(DebugActionTest, GenericHandler) {
  DebugActionManager manager;

  // A generic handler that always executes the simple action, but not the
  // parametric action.
  struct GenericHandler : DebugActionManager::GenericHandler {
    FailureOr<bool> execute(llvm::function_ref<void()> transform,
                            const DebugActionBase &action) final {
      StringRef desc = action.getDescription();
      if (isa<SimpleAction>(action)) {
        EXPECT_EQ(desc, SimpleAction::getDescription());
        transform();
        return true;
      }

      EXPECT_TRUE(isa<ParametricAction>(action));
      EXPECT_EQ(desc, ParametricAction::getDescription());
      return false;
    }
  };
  manager.registerActionHandler<GenericHandler>();

  auto noOp = []() { return; };
  EXPECT_TRUE(manager.execute<SimpleAction>(noOp));
  EXPECT_FALSE(manager.execute<ParametricAction>(noOp, true));
}

TEST(DebugActionTest, ActionSpecificHandler) {
  DebugActionManager manager;

  // Handler that simply uses the input as the decider.
  struct ActionSpecificHandler : ParametricAction::Handler {
    FailureOr<bool> execute(llvm::function_ref<void()> transform,
                            const ParametricAction &action) final {
      if (action.executeParam)
        transform();
      return action.executeParam;
    }
  };
  manager.registerActionHandler<ActionSpecificHandler>();

  int count = 0;
  auto incCount = [&]() { count++; };
  EXPECT_TRUE(manager.execute<ParametricAction>(incCount, true));
  EXPECT_EQ(count, 1);
  EXPECT_FALSE(manager.execute<ParametricAction>(incCount, false));
  EXPECT_EQ(count, 1);

  // There is no handler for the simple action, so it is always executed.
  EXPECT_TRUE(manager.execute<SimpleAction>(incCount));
  EXPECT_EQ(count, 2);
}

TEST(DebugActionTest, DebugCounterHandler) {
  DebugActionManager manager;

  // Handler that uses the number of action executions as the decider.
  struct DebugCounterHandler : SimpleAction::Handler {
    FailureOr<bool> execute(llvm::function_ref<void()> transform,
                            const SimpleAction &action) final {
      bool shouldExecute = numExecutions++ < 3;
      if (shouldExecute)
        transform();
      return shouldExecute;
    }
    unsigned numExecutions = 0;
  };
  manager.registerActionHandler<DebugCounterHandler>();

  // Check that the action is executed 3 times, but no more after.
  auto noOp = []() { return; };
  EXPECT_TRUE(manager.execute<SimpleAction>(noOp));
  EXPECT_TRUE(manager.execute<SimpleAction>(noOp));
  EXPECT_TRUE(manager.execute<SimpleAction>(noOp));
  EXPECT_FALSE(manager.execute<SimpleAction>(noOp));
  EXPECT_FALSE(manager.execute<SimpleAction>(noOp));
}

TEST(DebugActionTest, NonOverlappingActionSpecificHandlers) {
  DebugActionManager manager;

  // One handler returns true and another returns false
  struct SimpleActionHandler : SimpleAction::Handler {
    FailureOr<bool> execute(llvm::function_ref<void()> transform,
                            const SimpleAction &action) final {
      transform();
      return true;
    }
  };
  struct OtherSimpleActionHandler : OtherSimpleAction::Handler {
    FailureOr<bool> execute(llvm::function_ref<void()> transform,
                            const OtherSimpleAction &action) final {
      transform();
      return false;
    }
  };
  manager.registerActionHandler<SimpleActionHandler>();
  manager.registerActionHandler<OtherSimpleActionHandler>();
  auto noOp = []() { return; };
  EXPECT_TRUE(manager.execute<SimpleAction>(noOp));
  EXPECT_FALSE(manager.execute<OtherSimpleAction>(noOp));
}

} // namespace
