//===-- PlatformDarwinTest.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"

#include "llvm/ADT/StringRef.h"

#include <tuple>

using namespace lldb;
using namespace lldb_private;

TEST(PlatformDarwinTest, TestParseVersionBuildDir) {
  uint32_t A, B, C;
  llvm::StringRef D;

  std::tie(A, B, C, D) = PlatformDarwin::ParseVersionBuildDir("1.2.3 (test1)");
  EXPECT_EQ(1u, A);
  EXPECT_EQ(2u, B);
  EXPECT_EQ(3u, C);
  EXPECT_EQ("test1", D);

  std::tie(A, B, C, D) = PlatformDarwin::ParseVersionBuildDir("2.3 (test2)");
  EXPECT_EQ(2u, A);
  EXPECT_EQ(3u, B);
  EXPECT_EQ("test2", D);

  std::tie(A, B, C, D) = PlatformDarwin::ParseVersionBuildDir("3 (test3)");
  EXPECT_EQ(3u, A);
  EXPECT_EQ("test3", D);

  std::tie(A, B, C, D) = PlatformDarwin::ParseVersionBuildDir("1.2.3 (test");
  EXPECT_EQ(1u, A);
  EXPECT_EQ(2u, B);
  EXPECT_EQ(3u, C);
  EXPECT_EQ("test", D);

  std::tie(A, B, C, D) = PlatformDarwin::ParseVersionBuildDir("2.3.4 test");
  EXPECT_EQ(2u, A);
  EXPECT_EQ(3u, B);
  EXPECT_EQ(4u, C);
  EXPECT_EQ("", D);

  std::tie(A, B, C, D) = PlatformDarwin::ParseVersionBuildDir("3.4.5");
  EXPECT_EQ(3u, A);
  EXPECT_EQ(4u, B);
  EXPECT_EQ(5u, C);
}
