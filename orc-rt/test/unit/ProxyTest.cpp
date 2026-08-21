//===- ProxyTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's Proxy.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Proxy.h"

#include "CommonTestUtils.h"

#include "gtest/gtest.h"

#include <optional>
#include <utility>

using namespace orc_rt;

namespace {

// Reports the callee's void result (success), exercising the void -> Error
// return mapping.
void voidDispatch(move_only_function<void(Error)> OnComplete, Session &,
                  const void *) {
  OnComplete(Error::success());
}

// Returns its argument plus one, exercising argument forwarding and the
// T -> Expected<T> return mapping.
void addOneDispatch(move_only_function<void(Expected<int>)> OnComplete,
                    Session &, const void *, const int &X) {
  OnComplete(X + 1);
}

// Returns the callee tag it was handed, exercising tag forwarding through
// operator().
void returnTagDispatch(
    move_only_function<void(Expected<const void *>)> OnComplete, Session &,
    const void *Tag) {
  OnComplete(Tag);
}

} // namespace

TEST(ProxyTest, DefaultConstructedProxyIsNull) {
  Proxy<void()> P;
  EXPECT_FALSE(P);
}

TEST(ProxyTest, DispatchReportsVoidResult) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  int Tag = 0;
  Proxy<void()> P(voidDispatch, &Tag);
  EXPECT_TRUE(P);

  bool Completed = false;
  P(
      [&](Error Err) {
        cantFail(std::move(Err));
        Completed = true;
      },
      S);
  EXPECT_TRUE(Completed);
}

TEST(ProxyTest, DispatchForwardsArgsAndResult) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  int Tag = 0;
  Proxy<int(int)> P(addOneDispatch, &Tag);

  std::optional<int> Result;
  P([&](Expected<int> R) { Result = cantFail(std::move(R)); }, S, 42);
  ASSERT_TRUE(Result.has_value());
  EXPECT_EQ(*Result, 43);
}

TEST(ProxyTest, DispatchForwardsCalleeTag) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  int Tag = 0;
  Proxy<const void *()> P(returnTagDispatch, &Tag);

  std::optional<const void *> Result;
  P([&](Expected<const void *> R) { Result = cantFail(std::move(R)); }, S);
  ASSERT_TRUE(Result.has_value());
  EXPECT_EQ(*Result, &Tag);
}
