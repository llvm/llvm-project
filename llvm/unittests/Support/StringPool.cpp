//===- llvm/unittest/Support/StringPoiil.cpp - StringPool tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StringPool.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(PooledStringPtrTest, OperatorEquals) {
  StringPool pool;
  const PooledStringPtr a = pool.intern("a");
  const PooledStringPtr b = pool.intern("b");
  EXPECT_FALSE(a == b);
}

TEST(PooledStringPtrTest, OperatorNotEquals) {
  StringPool pool;
  const PooledStringPtr a = pool.intern("a");
  const PooledStringPtr b = pool.intern("b");
  EXPECT_TRUE(a != b);
}

}
