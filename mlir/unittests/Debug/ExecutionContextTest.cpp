//===- ExecutionContextTest.cpp - Debug Execution Context first impl ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"
#include "llvm/ADT/MapVector.h"
#include "gmock/gmock.h"

using namespace mlir;
using namespace mlir::tracing;

namespace {
struct DebuggerAction : public ActionImpl<DebuggerAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebuggerAction)
  static constexpr StringLiteral tag = "debugger-action";
};
struct OtherAction : public ActionImpl<OtherAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OtherAction)
  static constexpr StringLiteral tag = "other-action";
};
struct ThirdAction : public ActionImpl<ThirdAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThirdAction)
  static constexpr StringLiteral tag = "third-action";
};

// Simple action that does nothing.
void noOp() {}

/// This test executes a stack of nested action and check that the backtrace is
/// as expect.
TEST(ExecutionContext, ActionActiveStackTest) {

  // We'll break three time, once on each action, the backtraces should match
  // each of the entries here.
  std::vector<std::vector<StringRef>> expectedStacks = {
      {DebuggerAction::tag},
      {OtherAction::tag, DebuggerAction::tag},
      {ThirdAction::tag, OtherAction::tag, DebuggerAction::tag}};

  auto checkStacks = [&](const ActionActiveStack *backtrace,
                         const std::vector<StringRef> &currentStack) {
    ASSERT_EQ((int)currentStack.size(), backtrace->getDepth() + 1);
    for (StringRef stackEntry : currentStack) {
      ASSERT_NE(backtrace, nullptr);
      ASSERT_EQ(stackEntry, backtrace->getAction().getTag());
      backtrace = backtrace->getParent();
    }
  };

  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Step, ExecutionContext::Apply};
  int idx = 0;
  StringRef current;
  int currentDepth = -1;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    current = backtrace->getAction().getTag();
    currentDepth = backtrace->getDepth();
    checkStacks(backtrace, expectedStacks[idx]);
    return controlSequence[idx++];
  };

  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  std::vector<TagBreakpoint *> breakpoints;
  breakpoints.push_back(simpleManager.addBreakpoint(DebuggerAction::tag));
  breakpoints.push_back(simpleManager.addBreakpoint(OtherAction::tag));
  breakpoints.push_back(simpleManager.addBreakpoint(ThirdAction::tag));

  auto third = [&]() {
    EXPECT_EQ(current, ThirdAction::tag);
    EXPECT_EQ(currentDepth, 2);
  };
  auto nested = [&]() {
    EXPECT_EQ(current, OtherAction::tag);
    EXPECT_EQ(currentDepth, 1);
    executionCtx(third, ThirdAction{});
  };
  auto original = [&]() {
    EXPECT_EQ(current, DebuggerAction::tag);
    EXPECT_EQ(currentDepth, 0);
    executionCtx(nested, OtherAction{});
    return;
  };

  executionCtx(original, DebuggerAction{});
}

TEST(ExecutionContext, DebuggerTest) {
  // Check matching and non matching breakpoints, with various enable/disable
  // schemes.
  int match = 0;
  auto onBreakpoint = [&match](const ActionActiveStack *backtrace) {
    match++;
    return ExecutionContext::Skip;
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);

  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 0);

  Breakpoint *dbgBreakpoint = simpleManager.addBreakpoint(DebuggerAction::tag);
  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 1);

  dbgBreakpoint->disable();
  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 1);

  dbgBreakpoint->enable();
  executionCtx(noOp, DebuggerAction{});
  EXPECT_EQ(match, 2);

  executionCtx(noOp, OtherAction{});
  EXPECT_EQ(match, 2);
}

TEST(ExecutionContext, ApplyTest) {
  // Test the "apply" control.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() { EXPECT_EQ(counter, 1); };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 1);
}

TEST(ExecutionContext, SkipTest) {
  // Test the "skip" control.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Apply, ExecutionContext::Skip};
  int idx = 0, counter = 0, executionCounter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() { ++executionCounter; };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 2);
  EXPECT_EQ(executionCounter, 1);
}

TEST(ExecutionContext, StepApplyTest) {
  // Test the "step" control with a nested action.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag, OtherAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);
  auto nested = [&]() { EXPECT_EQ(counter, 2); };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    executionCtx(nested, OtherAction{});
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, StepNothingInsideTest) {
  // Test the "step" control without a nested action.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Step};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() { EXPECT_EQ(counter, 1); };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, NextTest) {
  // Test the "next" control.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Next, ExecutionContext::Next};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() { EXPECT_EQ(counter, 1); };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, FinishTest) {
  // Test the "finish" control.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag, OtherAction::tag,
                                        DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Step, ExecutionContext::Finish,
      ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);
  auto nested = [&]() { EXPECT_EQ(counter, 2); };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    executionCtx(nested, OtherAction{});
    EXPECT_EQ(counter, 2);
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 3);
}

TEST(ExecutionContext, FinishBreakpointInNestedTest) {
  // Test the "finish" control with a breakpoint in the nested action.
  std::vector<StringRef> tagSequence = {OtherAction::tag, DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Finish, ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(OtherAction::tag);

  auto nested = [&]() { EXPECT_EQ(counter, 1); };
  auto original = [&]() {
    EXPECT_EQ(counter, 0);
    executionCtx(nested, OtherAction{});
    EXPECT_EQ(counter, 1);
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 2);
}

TEST(ExecutionContext, FinishNothingBackTest) {
  // Test the "finish" control without a nested action.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Finish};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };
  auto callback = [&]() { EXPECT_EQ(counter, 1); };
  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);

  executionCtx(callback, DebuggerAction{});
  EXPECT_EQ(counter, 1);
}

TEST(ExecutionContext, EnableDisableBreakpointOnCallback) {
  // Test enabling and disabling breakpoints while executing the action.
  std::vector<StringRef> tagSequence = {DebuggerAction::tag, ThirdAction::tag,
                                        OtherAction::tag, DebuggerAction::tag};
  std::vector<ExecutionContext::Control> controlSequence = {
      ExecutionContext::Apply, ExecutionContext::Finish,
      ExecutionContext::Finish, ExecutionContext::Apply};
  int idx = 0, counter = 0;
  auto onBreakpoint = [&](const ActionActiveStack *backtrace) {
    ++counter;
    EXPECT_EQ(tagSequence[idx], backtrace->getAction().getTag());
    return controlSequence[idx++];
  };

  TagBreakpointManager simpleManager;
  ExecutionContext executionCtx(onBreakpoint);
  executionCtx.addBreakpointManager(&simpleManager);
  simpleManager.addBreakpoint(DebuggerAction::tag);
  Breakpoint *toBeDisabled = simpleManager.addBreakpoint(OtherAction::tag);

  auto third = [&]() { EXPECT_EQ(counter, 2); };
  auto nested = [&]() {
    EXPECT_EQ(counter, 1);
    executionCtx(third, ThirdAction{});
    EXPECT_EQ(counter, 2);
  };
  auto original = [&]() {
    EXPECT_EQ(counter, 1);
    toBeDisabled->disable();
    simpleManager.addBreakpoint(ThirdAction::tag);
    executionCtx(nested, OtherAction{});
    EXPECT_EQ(counter, 3);
  };

  executionCtx(original, DebuggerAction{});
  EXPECT_EQ(counter, 4);
}
} // namespace
