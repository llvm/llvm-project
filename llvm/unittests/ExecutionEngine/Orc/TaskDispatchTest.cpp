//===----------- TaskDispatchTest.cpp - Test TaskDispatch APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;

TEST(InPlaceTaskDispatchTest, GenericNamedTask) {
  auto D = std::make_unique<InPlaceTaskDispatcher>();
  bool B = false;
  D->dispatch(makeGenericNamedTask([&]() { B = true; }));
  EXPECT_TRUE(B);
}

#if LLVM_ENABLE_THREADS
TEST(DynamicThreadPoolDispatchTest, GenericNamedTask) {
  auto D = std::make_unique<DynamicThreadPoolTaskDispatcher>();
  std::promise<bool> P;
  auto F = P.get_future();
  D->dispatch(makeGenericNamedTask(
      [P = std::move(P)]() mutable { P.set_value(true); }));
  EXPECT_TRUE(F.get());
}
#endif
