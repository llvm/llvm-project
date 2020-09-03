//===-- ConstStringTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ConstString.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/YAMLParser.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(ConstStringTest, format_provider) {
  EXPECT_EQ("foo", llvm::formatv("{0}", ConstString("foo")).str());
}

TEST(ConstStringTest, MangledCounterpart) {
  ConstString uvw("uvw");
  ConstString counterpart;
  EXPECT_FALSE(uvw.GetMangledCounterpart(counterpart));
  EXPECT_EQ("", counterpart.GetStringRef());

  ConstString xyz;
  xyz.SetStringWithMangledCounterpart("xyz", uvw);
  EXPECT_EQ("xyz", xyz.GetStringRef());

  EXPECT_TRUE(xyz.GetMangledCounterpart(counterpart));
  EXPECT_EQ("uvw", counterpart.GetStringRef());

  EXPECT_TRUE(uvw.GetMangledCounterpart(counterpart));
  EXPECT_EQ("xyz", counterpart.GetStringRef());
}

TEST(ConstStringTest, UpdateMangledCounterpart) {
  { // Add counterpart
    ConstString some1;
    some1.SetStringWithMangledCounterpart("some", ConstString(""));
  }
  { // Overwrite empty string
    ConstString some2;
    some2.SetStringWithMangledCounterpart("some", ConstString("one"));
  }
  { // Overwrite with identical value
    ConstString some2;
    some2.SetStringWithMangledCounterpart("some", ConstString("one"));
  }
  { // Check counterpart is set
    ConstString counterpart;
    EXPECT_TRUE(ConstString("some").GetMangledCounterpart(counterpart));
    EXPECT_EQ("one", counterpart.GetStringRef());
  }
}

TEST(ConstStringTest, FromMidOfBufferStringRef) {
  // StringRef's into bigger buffer: no null termination
  const char *buffer = "abcdefghi";
  llvm::StringRef foo_ref(buffer, 3);
  llvm::StringRef bar_ref(buffer + 3, 3);

  ConstString foo(foo_ref);

  ConstString bar;
  bar.SetStringWithMangledCounterpart(bar_ref, foo);
  EXPECT_EQ("def", bar.GetStringRef());

  ConstString counterpart;
  EXPECT_TRUE(bar.GetMangledCounterpart(counterpart));
  EXPECT_EQ("abc", counterpart.GetStringRef());

  EXPECT_TRUE(foo.GetMangledCounterpart(counterpart));
  EXPECT_EQ("def", counterpart.GetStringRef());
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

TEST(ConstStringTest, CompareConstString) {
  ConstString foo("foo");
  ConstString foo2("foo");
  ConstString bar("bar");

  EXPECT_TRUE(foo == foo2);
  EXPECT_TRUE(foo2 == foo);
  EXPECT_TRUE(foo == ConstString("foo"));

  EXPECT_FALSE(foo == bar);
  EXPECT_FALSE(foo2 == bar);
  EXPECT_FALSE(foo == ConstString("bar"));
  EXPECT_FALSE(foo == ConstString("different"));
  EXPECT_FALSE(foo == ConstString(""));
  EXPECT_FALSE(foo == ConstString());

  ConstString empty("");
  EXPECT_FALSE(empty == ConstString("bar"));
  EXPECT_FALSE(empty == ConstString());
  EXPECT_TRUE(empty == ConstString(""));

  ConstString null;
  EXPECT_FALSE(null == ConstString("bar"));
  EXPECT_TRUE(null == ConstString());
  EXPECT_FALSE(null == ConstString(""));
}

TEST(ConstStringTest, CompareStringRef) {
  ConstString foo("foo");

  EXPECT_TRUE(foo == "foo");
  EXPECT_TRUE(foo != "");
  EXPECT_FALSE(foo == static_cast<const char *>(nullptr));
  EXPECT_TRUE(foo != "bar");

  ConstString empty("");
  EXPECT_FALSE(empty == "foo");
  EXPECT_FALSE(empty != "");
  EXPECT_FALSE(empty == static_cast<const char *>(nullptr));
  EXPECT_TRUE(empty != "bar");

  ConstString null;
  EXPECT_FALSE(null == "foo");
  EXPECT_TRUE(null != "");
  EXPECT_TRUE(null == static_cast<const char *>(nullptr));
  EXPECT_TRUE(null != "bar");
}

TEST(ConstStringTest, YAML) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);

  // Serialize.
  std::vector<ConstString> strings = {ConstString("foo"), ConstString("bar"),
                                      ConstString("")};
  llvm::yaml::Output yout(os);
  yout << strings;
  os.flush();

  // Deserialize.
  std::vector<ConstString> deserialized;
  llvm::yaml::Input yin(buffer);
  yin >> deserialized;

  EXPECT_EQ(strings, deserialized);
}
