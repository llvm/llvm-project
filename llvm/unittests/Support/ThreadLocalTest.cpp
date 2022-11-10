//===- llvm/unittest/Support/ThreadLocalTest.cpp - ThreadLocal tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadLocal.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace llvm;
using namespace sys;

namespace {

class ThreadLocalTest : public ::testing::Test {
};

struct S {
  int i;
};

TEST_F(ThreadLocalTest, Basics) {
  ThreadLocal<const S> x;

  static_assert(std::is_const_v<std::remove_pointer_t<decltype(x.get())>>,
                "ThreadLocal::get didn't return a pointer to const object");

  EXPECT_EQ(nullptr, x.get());

  S s;
  x.set(&s);
  EXPECT_EQ(&s, x.get());

  x.erase();
  EXPECT_EQ(nullptr, x.get());

  ThreadLocal<S> y;

  static_assert(!std::is_const_v<std::remove_pointer_t<decltype(y.get())>>,
                "ThreadLocal::get returned a pointer to const object");

  EXPECT_EQ(nullptr, y.get());

  y.set(&s);
  EXPECT_EQ(&s, y.get());

  y.erase();
  EXPECT_EQ(nullptr, y.get());
}

}
