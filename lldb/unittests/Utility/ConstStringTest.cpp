//===-- ConstStringTest.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ConstString.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(ConstStringTest, format_provider) {
  EXPECT_EQ("foo", llvm::formatv("{0}", ConstString("foo")).str());
}

TEST(ConstStringTest, MangledCounterpart) {
  ConstString foo("foo");
  ConstString counterpart;
  EXPECT_FALSE(foo.GetMangledCounterpart(counterpart));
  EXPECT_EQ("", counterpart.GetStringRef());

  ConstString bar;
  bar.SetStringWithMangledCounterpart("bar", foo);
  EXPECT_EQ("bar", bar.GetStringRef());

  EXPECT_TRUE(bar.GetMangledCounterpart(counterpart));
  EXPECT_EQ("foo", counterpart.GetStringRef());

  EXPECT_TRUE(foo.GetMangledCounterpart(counterpart));
  EXPECT_EQ("bar", counterpart.GetStringRef());
}

TEST(ConstStringTest, NullAndEmptyStates) {
  ConstString foo("foo");
  EXPECT_FALSE(!foo);
  EXPECT_FALSE(foo.IsEmpty());
  EXPECT_FALSE(foo.IsNull());

  ConstString empty("");
  EXPECT_TRUE(!empty);
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_FALSE(empty.IsNull());

  ConstString null;
  EXPECT_TRUE(!null);
  EXPECT_TRUE(null.IsEmpty());
  EXPECT_TRUE(null.IsNull());
}
