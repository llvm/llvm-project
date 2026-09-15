//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Windows/Common/LoadedModuleList.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
const FileSpec kDll("C:\\a\\one.dll", FileSpec::Style::windows);
const FileSpec kOther("C:\\a\\two.dll", FileSpec::Style::windows);
} // namespace

TEST(LoadedModuleListTest, ReportsTheMappingItWasGiven) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);

  EXPECT_EQ(modules.GetSize(), 1u);
  EXPECT_EQ(modules.GetBaseAddress(kDll), 0x1000u);
  EXPECT_EQ(*modules.FindFile(kDll), kDll);
}

TEST(LoadedModuleListTest, IsEmptyUntilSomethingIsMapped) {
  LoadedModuleList modules;
  EXPECT_TRUE(modules.IsEmpty());
  modules.Add(kDll, 0x1000);
  EXPECT_FALSE(modules.IsEmpty());
}

TEST(LoadedModuleListTest, DoesNotFindAnUnmappedFile) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);

  EXPECT_EQ(modules.GetBaseAddress(kOther), std::nullopt);
  EXPECT_EQ(modules.FindFile(kOther), nullptr);
}

TEST(LoadedModuleListTest, ReportsTheFirstOfSeveralMappings) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);
  modules.Add(kDll, 0x9000);

  // One entry per file, at the address the loader mapped it first.
  EXPECT_EQ(modules.GetSize(), 1u);
  EXPECT_EQ(modules.GetBaseAddress(kDll), 0x1000u);
}

TEST(LoadedModuleListTest, KeepsTheFileWhileAnyMappingRemains) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);
  modules.Add(kDll, 0x9000);

  // Unmapping the second mapping must not retire the file: the program is
  // still running from the first one.
  EXPECT_FALSE(modules.Remove(0x9000));
  EXPECT_EQ(modules.GetBaseAddress(kDll), 0x1000u);

  EXPECT_EQ(modules.Remove(0x1000), kDll);
  EXPECT_TRUE(modules.IsEmpty());
}

TEST(LoadedModuleListTest, FallsBackToASurvivingMapping) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);
  modules.Add(kDll, 0x9000);

  // Dropping the reported mapping leaves the other one to report.
  EXPECT_FALSE(modules.Remove(0x1000));
  EXPECT_EQ(modules.GetBaseAddress(kDll), 0x9000u);
}

TEST(LoadedModuleListTest, RemovesOnlyTheGivenMapping) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);
  modules.Add(kOther, 0x2000);

  EXPECT_EQ(modules.Remove(0x1000), kDll);
  EXPECT_EQ(modules.GetSize(), 1u);
  EXPECT_EQ(modules.GetBaseAddress(kOther), 0x2000u);
}

TEST(LoadedModuleListTest, IgnoresAnUnknownAddress) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);

  EXPECT_FALSE(modules.Remove(0xdead));
  EXPECT_EQ(modules.GetSize(), 1u);
}

TEST(LoadedModuleListTest, SameAddressTwiceIsNotOneMapping) {
  LoadedModuleList modules;
  modules.Add(kDll, 0x1000);
  modules.Add(kDll, 0x1000);

  // Two reported loads owe two reported unloads, even at the same base.
  EXPECT_FALSE(modules.Remove(0x1000));
  EXPECT_EQ(modules.Remove(0x1000), kDll);
  EXPECT_TRUE(modules.IsEmpty());
}
