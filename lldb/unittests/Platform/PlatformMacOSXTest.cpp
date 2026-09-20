//===-- PlatformMacOSXTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"

using namespace lldb;
using namespace lldb_private;

class PlatformMacOSXTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;
};

#ifdef __APPLE__
static bool containsArch(const std::vector<ArchSpec> &archs,
                         const ArchSpec &arch) {
  return std::find_if(archs.begin(), archs.end(), [&](const ArchSpec &other) {
           return arch.IsExactMatch(other);
         }) != archs.end();
}

TEST_F(PlatformMacOSXTest, TestGetSupportedArchitectures) {
  PlatformMacOSX platform;

  const ArchSpec x86_macosx_arch("x86_64-apple-macosx");

  EXPECT_TRUE(containsArch(platform.GetSupportedArchitectures(x86_macosx_arch),
                           x86_macosx_arch));
  EXPECT_TRUE(
      containsArch(platform.GetSupportedArchitectures({}), x86_macosx_arch));

#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
  const ArchSpec arm64_macosx_arch("arm64-apple-macosx");
  const ArchSpec arm64_ios_arch("arm64-apple-ios");

  EXPECT_TRUE(containsArch(
      platform.GetSupportedArchitectures(arm64_macosx_arch), arm64_ios_arch));
  EXPECT_TRUE(
      containsArch(platform.GetSupportedArchitectures({}), arm64_ios_arch));
  EXPECT_FALSE(containsArch(platform.GetSupportedArchitectures(arm64_ios_arch),
                            arm64_ios_arch));
#endif
}

struct NameAndResult {
  std::string dir;
  llvm::VersionTuple version;
  std::string build;
};

TEST_F(PlatformMacOSXTest, TestDeviceSupportDirectoryNames) {
  PlatformMacOSX platform;

  NameAndResult tests[] = {
      {"10.0 (21R329) universal", llvm::VersionTuple(10, 0), "21R329"},
      {"17.0 (23X1010104078) universal", llvm::VersionTuple(17, 0),
       "23X1010104078"},
      {"17.0 (23A200) arm64e", llvm::VersionTuple(17, 0), "23A200"},
      {"17.0 (20A352)", llvm::VersionTuple(17, 0), "20A352"},
      {"Watch4,2 10.0 (21R329)", llvm::VersionTuple(10, 0), "21R329"},
      {"iPhone11,2 26.0 (23A276)", llvm::VersionTuple(26, 0), "23A276"},
      {"iPhone13,2 17.0 (18C22)", llvm::VersionTuple(17, 0), "18C22"},
  };
  for (size_t i = 0; i < std::size(tests); i++) {
    llvm::VersionTuple version;
    llvm::StringRef build_str;
    std::tie(version, build_str) = platform.ParseVersionBuildDir(tests[i].dir);
    EXPECT_EQ(tests[i].version, version);
    EXPECT_EQ(tests[i].build, build_str);
  }
}

#endif
