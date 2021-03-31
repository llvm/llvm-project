//===- DebugCounterTest.cpp - Debug Counter Tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugCounter.h"
#include "gmock/gmock.h"

using namespace mlir;

// DebugActionManager is only enabled in DEBUG mode.
#ifndef NDEBUG

namespace {

struct CounterAction : public DebugAction<> {
  static StringRef getTag() { return "counter-action"; }
  static StringRef getDescription() { return "Test action for debug counters"; }
};

TEST(DebugCounterTest, CounterTest) {
  std::unique_ptr<DebugCounter> counter = std::make_unique<DebugCounter>();
  counter->addCounter(CounterAction::getTag(), /*countToSkip=*/1,
                      /*countToStopAfter=*/3);

  DebugActionManager manager;
  manager.registerActionHandler(std::move(counter));

  // The first execution is skipped.
  EXPECT_FALSE(manager.shouldExecute<CounterAction>());

  // The counter stops after 3 successful executions.
  EXPECT_TRUE(manager.shouldExecute<CounterAction>());
  EXPECT_TRUE(manager.shouldExecute<CounterAction>());
  EXPECT_TRUE(manager.shouldExecute<CounterAction>());
  EXPECT_FALSE(manager.shouldExecute<CounterAction>());
}

} // namespace

#endif
