//===- StringExtrasTest.cpp - Unit tests for String extras ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(StringExtrasTest, Join) {
  std::vector<std::string> Items;
  EXPECT_EQ("", join(Items.begin(), Items.end(), " <sep> "));

  Items = {"foo"};
  EXPECT_EQ("foo", join(Items.begin(), Items.end(), " <sep> "));

  Items = {"foo", "bar"};
  EXPECT_EQ("foo <sep> bar", join(Items.begin(), Items.end(), " <sep> "));

  Items = {"foo", "bar", "baz"};
  EXPECT_EQ("foo <sep> bar <sep> baz",
            join(Items.begin(), Items.end(), " <sep> "));
}

TEST(StringExtrasTest, JoinItems) {
  const char *Foo = "foo";
  std::string Bar = "bar";
  llvm::StringRef Baz = "baz";
  char X = 'x';

  EXPECT_EQ("", join_items(" <sep> "));
  EXPECT_EQ("", join_items('/'));

  EXPECT_EQ("foo", join_items(" <sep> ", Foo));
  EXPECT_EQ("foo", join_items('/', Foo));

  EXPECT_EQ("foo <sep> bar", join_items(" <sep> ", Foo, Bar));
  EXPECT_EQ("foo/bar", join_items('/', Foo, Bar));

  EXPECT_EQ("foo <sep> bar <sep> baz", join_items(" <sep> ", Foo, Bar, Baz));
  EXPECT_EQ("foo/bar/baz", join_items('/', Foo, Bar, Baz));

  EXPECT_EQ("foo <sep> bar <sep> baz <sep> x",
            join_items(" <sep> ", Foo, Bar, Baz, X));

  EXPECT_EQ("foo/bar/baz/x", join_items('/', Foo, Bar, Baz, X));
}
