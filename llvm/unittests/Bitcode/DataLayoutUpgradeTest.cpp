//===- DataLayoutUpgradeTest.cpp - Tests for DataLayout upgrades ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/AutoUpgrade.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DataLayoutUpgradeTest, ValidDataLayoutUpgrade) {
  std::string DL1 =
      UpgradeDataLayoutString("e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128",
                              "x86_64-unknown-linux-gnu");
  std::string DL2 = UpgradeDataLayoutString(
      "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32", "i686-pc-windows-msvc");
  std::string DL3 = UpgradeDataLayoutString("e-m:o-i64:64-i128:128-n32:64-S128",
                                            "x86_64-apple-macosx");
  EXPECT_EQ(DL1, "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64"
                 "-f80:128-n8:16:32:64-S128");
  EXPECT_EQ(DL2, "e-m:w-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64"
                 "-f80:128-n8:16:32-S32");
  EXPECT_EQ(DL3, "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128"
                 "-n32:64-S128");

  // Check that AMDGPU targets add -G1 if it's not present.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "r600"), "e-p:32:32-G1");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:64:64", "amdgcn"), "e-p:64:64-G1");

  // Check that RISCV64 upgrades -n64 to -n32:64.
  EXPECT_EQ(UpgradeDataLayoutString("e-m:e-p:64:64-i64:64-i128:128-n64-S128",
                                    "riscv64"),
            "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128");
}

TEST(DataLayoutUpgradeTest, NoDataLayoutUpgrade) {
  std::string DL1 = UpgradeDataLayoutString(
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32"
      "-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
      "-n8:16:32:64-S128",
      "x86_64-unknown-linux-gnu");
  std::string DL2 = UpgradeDataLayoutString("e-p:32:32", "i686-apple-darwin9");
  std::string DL3 = UpgradeDataLayoutString("e-m:e-i64:64-n32:64",
                                            "powerpc64le-unknown-linux-gnu");
  std::string DL4 =
      UpgradeDataLayoutString("e-m:o-i64:64-i128:128-n32:64-S128", "aarch64--");
  EXPECT_EQ(DL1, "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                 "-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64"
                 "-f80:128:128-n8:16:32:64-S128");
  EXPECT_EQ(DL2, "e-p:32:32");
  EXPECT_EQ(DL3, "e-m:e-i64:64-n32:64");
  EXPECT_EQ(DL4, "e-m:o-i64:64-i128:128-n32:64-S128");

  // Check that AMDGPU targets don't add -G1 if there is already a -G flag.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G2", "r600"), "e-p:32:32-G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2", "r600"), "G2");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:64:64-G2", "amdgcn"), "e-p:64:64-G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2-e-p:64:64", "amdgcn"), "G2-e-p:64:64");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:64:64-G0", "amdgcn"), "e-p:64:64-G0");
}

TEST(DataLayoutUpgradeTest, EmptyDataLayout) {
  std::string DL1 = UpgradeDataLayoutString("", "x86_64-unknown-linux-gnu");
  std::string DL2 = UpgradeDataLayoutString(
      "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128", "");
  EXPECT_EQ(DL1, "");
  EXPECT_EQ(DL2, "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128");

  // Check that AMDGPU targets add G1 if it's not present.
  EXPECT_EQ(UpgradeDataLayoutString("", "r600"), "G1");
  EXPECT_EQ(UpgradeDataLayoutString("", "amdgcn"), "G1");
}

} // end namespace
