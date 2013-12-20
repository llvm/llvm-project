//===- llvm/unittest/Support/ThreadLocalTest.cpp - Therad Local tests   ---===//
//
//		       The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadLocal.h"
#include "gtest/gtest.h"

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

  EXPECT_EQ(0, x.get());

  S s;
  x.set(&s);
  EXPECT_EQ(&s, x.get());

  x.erase();
  EXPECT_EQ(0, x.get());
}

}
