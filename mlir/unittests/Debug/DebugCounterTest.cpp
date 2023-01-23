//===- DebugCounterTest.cpp - Debug Counter Tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/Counter.h"
#include "mlir/Support/TypeID.h"
#include "gmock/gmock.h"

using namespace mlir;
using namespace mlir::tracing;

namespace {

struct CounterAction : public ActionImpl<CounterAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterAction)
  static constexpr StringLiteral tag = "counter-action";
};

TEST(DebugCounterTest, CounterTest) {
  std::unique_ptr<DebugCounter> counter = std::make_unique<DebugCounter>();
  counter->addCounter(CounterAction::tag, /*countToSkip=*/1,
                      /*countToStopAfter=*/3);

  ActionManager manager;
  manager.registerActionHandler(std::move(counter));

  auto noOp = []() { return; };

  // The first execution is skipped.
  EXPECT_FALSE(manager.execute<CounterAction>(noOp));

  // The counter stops after 3 successful executions.
  EXPECT_TRUE(manager.execute<CounterAction>(noOp));
  EXPECT_TRUE(manager.execute<CounterAction>(noOp));
  EXPECT_TRUE(manager.execute<CounterAction>(noOp));
  EXPECT_FALSE(manager.execute<CounterAction>(noOp));
}

} // namespace
