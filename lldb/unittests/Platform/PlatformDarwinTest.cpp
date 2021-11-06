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
  using PlatformDarwin::GetCompatibleArch;
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

TEST(PlatformDarwinTest, GetCompatibleArchARM64) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_arm64;
  EXPECT_STREQ("arm64", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("armv4", PlatformDarwinTester::GetCompatibleArch(core, 10));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 11));
  EXPECT_STREQ("thumbv7", PlatformDarwinTester::GetCompatibleArch(core, 12));
  EXPECT_STREQ("thumbv4t", PlatformDarwinTester::GetCompatibleArch(core, 21));
  EXPECT_STREQ("thumb", PlatformDarwinTester::GetCompatibleArch(core, 22));
  EXPECT_EQ(nullptr, PlatformDarwinTester::GetCompatibleArch(core, 23));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv7f) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv7f;
  EXPECT_STREQ("armv7f", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 6));
  EXPECT_STREQ("thumbv7f", PlatformDarwinTester::GetCompatibleArch(core, 7));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv7k) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv7k;
  EXPECT_STREQ("armv7k", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 6));
  EXPECT_STREQ("thumbv7k", PlatformDarwinTester::GetCompatibleArch(core, 7));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv7s) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv7s;
  EXPECT_STREQ("armv7s", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 6));
  EXPECT_STREQ("thumbv7s", PlatformDarwinTester::GetCompatibleArch(core, 7));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv7m) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv7m;
  EXPECT_STREQ("armv7m", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 6));
  EXPECT_STREQ("thumbv7m", PlatformDarwinTester::GetCompatibleArch(core, 7));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv7em) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv7em;
  EXPECT_STREQ("armv7em", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 6));
  EXPECT_STREQ("thumbv7em", PlatformDarwinTester::GetCompatibleArch(core, 7));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv7) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv7;
  EXPECT_STREQ("armv7", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv6m", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 5));
  EXPECT_STREQ("thumbv7", PlatformDarwinTester::GetCompatibleArch(core, 6));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv6m) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv6m;
  EXPECT_STREQ("armv6m", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv6", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 4));
  EXPECT_STREQ("thumbv6m", PlatformDarwinTester::GetCompatibleArch(core, 5));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv6) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv6;
  EXPECT_STREQ("armv6", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv5", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 3));
  EXPECT_STREQ("thumbv6", PlatformDarwinTester::GetCompatibleArch(core, 4));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv5) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv5;
  EXPECT_STREQ("armv5", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("armv4", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 2));
  EXPECT_STREQ("thumbv5", PlatformDarwinTester::GetCompatibleArch(core, 3));
}

TEST(PlatformDarwinTest, GetCompatibleArchARMv4) {
  const ArchSpec::Core core = ArchSpec::eCore_arm_armv4;
  EXPECT_STREQ("armv4", PlatformDarwinTester::GetCompatibleArch(core, 0));
  EXPECT_STREQ("arm", PlatformDarwinTester::GetCompatibleArch(core, 1));
  EXPECT_STREQ("thumbv4t", PlatformDarwinTester::GetCompatibleArch(core, 2));
  EXPECT_STREQ("thumb", PlatformDarwinTester::GetCompatibleArch(core, 3));
}
