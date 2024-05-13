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
#endif
