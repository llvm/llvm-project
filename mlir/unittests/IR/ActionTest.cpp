//===- ActionTest.cpp - Debug Action Tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Action.h"
#include "mlir/Support/TypeID.h"
#include "gmock/gmock.h"

using namespace mlir;
using namespace mlir::tracing;

namespace {
struct SimpleAction : ActionImpl<SimpleAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleAction)
  static constexpr StringLiteral tag = "simple-action";
};
struct OtherSimpleAction : ActionImpl<OtherSimpleAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherSimpleAction)
  static constexpr StringLiteral tag = "other-simple-action";
};
struct ParametricAction : ActionImpl<ParametricAction, bool> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParametricAction)
  ParametricAction(bool executeParam) : executeParam(executeParam) {}
  bool executeParam;
  static constexpr StringLiteral tag = "param-action";
};

TEST(ActionTest, GenericHandler) {
  ActionManager manager;

  // A generic handler that always executes the simple action, but not the
  // parametric action.
  struct GenericHandler : ActionManager::GenericHandler {
    FailureOr<bool> execute(llvm::function_ref<void()> transform,
                            const Action &action) final {
      StringRef tag = action.getTag();
      if (isa<SimpleAction>(action)) {
        EXPECT_EQ(tag, SimpleAction::tag);
        transform();
        return true;
      }

      EXPECT_TRUE(isa<ParametricAction>(action));
      return false;
    }
  };
  manager.registerActionHandler<GenericHandler>();

  auto noOp = []() { return; };
  EXPECT_TRUE(manager.execute<SimpleAction>(noOp));
  EXPECT_FALSE(manager.execute<ParametricAction>(noOp, true));
}

TEST(ActionTest, ActionSpecificHandler) {
  ActionManager manager;

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

TEST(ActionTest, DebugCounterHandler) {
  ActionManager manager;

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

TEST(ActionTest, NonOverlappingActionSpecificHandlers) {
  ActionManager manager;

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
