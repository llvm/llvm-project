//===-- PlatformDarwinTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"

#include "llvm/ADT/StringRef.h"

#include <tuple>

using namespace lldb;
using namespace lldb_private;

struct PlatformDarwinTester : public PlatformDarwin {
public:
  using PlatformDarwin::FindComponentInPath;
};

TEST(PlatformDarwinTest, TestParseVersionBuildDir) {
  llvm::VersionTuple V;
  llvm::StringRef D;

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("1.2.3 (test1)");
  EXPECT_EQ(llvm::VersionTuple(1, 2, 3), V);
  EXPECT_EQ("test1", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("2.3 (test2)");
  EXPECT_EQ(llvm::VersionTuple(2, 3), V);
  EXPECT_EQ("test2", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("3 (test3)");
  EXPECT_EQ(llvm::VersionTuple(3), V);
  EXPECT_EQ("test3", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("1.2.3 (test");
  EXPECT_EQ(llvm::VersionTuple(1, 2, 3), V);
  EXPECT_EQ("test", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("2.3.4 test");
  EXPECT_EQ(llvm::VersionTuple(2, 3, 4), V);
  EXPECT_EQ("", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("3.4.5");
  EXPECT_EQ(llvm::VersionTuple(3, 4, 5), V);
}

TEST(PlatformDarwinTest, FindComponentInPath) {
  EXPECT_EQ("/path/to/foo",
            PlatformDarwinTester::FindComponentInPath("/path/to/foo/", "foo"));

  EXPECT_EQ("/path/to/foo",
            PlatformDarwinTester::FindComponentInPath("/path/to/foo", "foo"));

  EXPECT_EQ("/path/to/foobar", PlatformDarwinTester::FindComponentInPath(
                                   "/path/to/foobar", "foo"));

  EXPECT_EQ("/path/to/foobar", PlatformDarwinTester::FindComponentInPath(
                                   "/path/to/foobar", "bar"));

  EXPECT_EQ("",
            PlatformDarwinTester::FindComponentInPath("/path/to/foo", "bar"));
}
