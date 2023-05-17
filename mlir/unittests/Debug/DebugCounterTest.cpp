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
  DebugCounter counter;
  counter.addCounter(CounterAction::tag, /*countToSkip=*/1,
                     /*countToStopAfter=*/3);

  int count = 0;
  auto noOp = [&]() {
    ++count;
    return;
  };

  // The first execution is skipped.
  counter(noOp, CounterAction{});
  EXPECT_EQ(count, 0);

  // The counter stops after 3 successful executions.
  counter(noOp, CounterAction{});
  EXPECT_EQ(count, 1);
  counter(noOp, CounterAction{});
  EXPECT_EQ(count, 2);
  counter(noOp, CounterAction{});
  EXPECT_EQ(count, 3);
  counter(noOp, CounterAction{});
  EXPECT_EQ(count, 3);
}

} // namespace
