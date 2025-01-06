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
  std::string DL3 = UpgradeDataLayoutString(
      "e-m:o-i64:64-f80:128-n8:16:32:64-S128", "x86_64-apple-macosx");
  std::string DL4 =
      UpgradeDataLayoutString("e-m:o-i64:64-i128:128-n32:64-S128", "aarch64--");
  std::string DL5 = UpgradeDataLayoutString(
      "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", "aarch64--");
  EXPECT_EQ(DL1,
            "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128"
            "-f80:128-n8:16:32:64-S128");
  EXPECT_EQ(DL2,
            "e-m:w-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128"
            "-f80:128-n8:16:32-S32");
  EXPECT_EQ(DL3, "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:"
                 "128-n8:16:32:64-S128");
  EXPECT_EQ(DL4, "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:"
                 "64-S128-Fn32");
  EXPECT_EQ(DL5, "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:"
                 "64-i128:128-n32:64-S128-Fn32");

  // Check that AMDGPU targets add -G1 if it's not present.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "r600"), "e-p:32:32-G1");
  // and that ANDGCN adds p7 and p8 as well.
  EXPECT_EQ(
      UpgradeDataLayoutString("e-p:64:64", "amdgcn"),
      "e-p:64:64-G1-ni:7:8:9-p7:160:256:256:32-p8:128:128-p9:192:256:256:32");
  EXPECT_EQ(
      UpgradeDataLayoutString("e-p:64:64-G1", "amdgcn"),
      "e-p:64:64-G1-ni:7:8:9-p7:160:256:256:32-p8:128:128-p9:192:256:256:32");
  // but that r600 does not.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G1", "r600"), "e-p:32:32-G1");

  // Ensure that the non-integral direction for address space 8 doesn't get
  // added in to pointer declarations.
  EXPECT_EQ(
      UpgradeDataLayoutString(
          "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:"
          "64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-"
          "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7",
          "amdgcn"),
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-"
      "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
      "1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9-p7:160:256:256:32-p8:128:128-"
      "p9:192:256:256:32");

  // Check that RISCV64 upgrades -n64 to -n32:64.
  EXPECT_EQ(UpgradeDataLayoutString("e-m:e-p:64:64-i64:64-i128:128-n64-S128",
                                    "riscv64"),
            "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128");

  // Check that LoongArch64 upgrades -n64 to -n32:64.
  EXPECT_EQ(UpgradeDataLayoutString("e-m:e-p:64:64-i64:64-i128:128-n64-S128",
                                    "loongarch64"),
            "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128");

  // Check that SPARC targets add -i128:128.
  EXPECT_EQ(
      UpgradeDataLayoutString("E-m:e-p:32:32-i64:64-f128:64-n32-S64", "sparc"),
      "E-m:e-p:32:32-i64:64-i128:128-f128:64-n32-S64");
  EXPECT_EQ(UpgradeDataLayoutString("E-m:e-i64:64-n32:64-S128", "sparcv9"),
            "E-m:e-i64:64-i128:128-n32:64-S128");

  // Check that MIPS64 targets add -i128:128.
  EXPECT_EQ(UpgradeDataLayoutString(
                "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128", "mips64"),
            "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  EXPECT_EQ(UpgradeDataLayoutString(
                "e-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128", "mips64el"),
            "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  // but that MIPS64 with o32 does not.
  EXPECT_EQ(UpgradeDataLayoutString(
                "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64", "mips64el"),
            "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64");

  // Check that PowerPC64 targets add -i128:128.
  EXPECT_EQ(UpgradeDataLayoutString("e-m:e-i64:64-n32:64", "powerpc64le-linux"),
            "e-m:e-i64:64-i128:128-n32:64");
  EXPECT_EQ(
      UpgradeDataLayoutString("E-m:e-Fn32-i64:64-n32:64", "powerpc64-linux"),
      "E-m:e-Fn32-i64:64-i128:128-n32:64");
  EXPECT_EQ(
      UpgradeDataLayoutString("E-m:a-Fi64-i64:64-n32:64", "powerpc64-ibm-aix"),
      "E-m:a-Fi64-i64:64-i128:128-n32:64");

  // Check that WebAssembly targets add -i128:128.
  EXPECT_EQ(
      UpgradeDataLayoutString(
          "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20",
          "wasm32"),
      "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20");
  EXPECT_EQ(
      UpgradeDataLayoutString(
          "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20",
          "wasm64"),
      "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20");

  // Check that SPIR && SPIRV targets add -G1 if it's not present.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "spir"), "e-p:32:32-G1");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "spir64"), "e-p:32:32-G1");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "spirv32"), "e-p:32:32-G1");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "spirv64"), "e-p:32:32-G1");
  // but that SPIRV Logical does not.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32", "spirv"), "e-p:32:32");
}

TEST(DataLayoutUpgradeTest, NoDataLayoutUpgrade) {
  std::string DL1 = UpgradeDataLayoutString(
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-"
      "f32:32:32"
      "-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
      "-n8:16:32:64-S128",
      "x86_64-unknown-linux-gnu");
  std::string DL3 = UpgradeDataLayoutString(
      "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32",
      "aarch64--");
  EXPECT_EQ(
      DL1,
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128"
      "-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64"
      "-f80:128:128-n8:16:32:64-S128");
  EXPECT_EQ(DL3, "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:"
                 "64-S128-Fn32");

  // Check that AMDGPU targets don't add -G1 if there is already a -G flag.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G2", "r600"), "e-p:32:32-G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2", "r600"), "G2");
  EXPECT_EQ(
      UpgradeDataLayoutString("e-p:64:64-G2", "amdgcn"),
      "e-p:64:64-G2-ni:7:8:9-p7:160:256:256:32-p8:128:128-p9:192:256:256:32");
  EXPECT_EQ(
      UpgradeDataLayoutString("G2-e-p:64:64", "amdgcn"),
      "G2-e-p:64:64-ni:7:8:9-p7:160:256:256:32-p8:128:128-p9:192:256:256:32");
  EXPECT_EQ(
      UpgradeDataLayoutString("e-p:64:64-G0", "amdgcn"),
      "e-p:64:64-G0-ni:7:8:9-p7:160:256:256:32-p8:128:128-p9:192:256:256:32");

  // Check that AMDGCN targets don't add already declared address space 7.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:64:64-p7:64:64", "amdgcn"),
            "e-p:64:64-p7:64:64-G1-ni:7:8:9-p8:128:128-p9:192:256:256:32");
  EXPECT_EQ(UpgradeDataLayoutString("p7:64:64-G2-e-p:64:64", "amdgcn"),
            "p7:64:64-G2-e-p:64:64-ni:7:8:9-p8:128:128-p9:192:256:256:32");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:64:64-p7:64:64-G1", "amdgcn"),
            "e-p:64:64-p7:64:64-G1-ni:7:8:9-p8:128:128-p9:192:256:256:32");

  // Check that SPIR & SPIRV targets don't add -G1 if there is already a -G
  // flag.
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G2", "spir"), "e-p:32:32-G2");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G2", "spir64"), "e-p:32:32-G2");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G2", "spirv32"), "e-p:32:32-G2");
  EXPECT_EQ(UpgradeDataLayoutString("e-p:32:32-G2", "spirv64"), "e-p:32:32-G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2", "spir"), "G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2", "spir64"), "G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2", "spirv32"), "G2");
  EXPECT_EQ(UpgradeDataLayoutString("G2", "spirv64"), "G2");

  // Check that PowerPC32 targets don't add -i128:128.
  EXPECT_EQ(UpgradeDataLayoutString("e-m:e-i64:64-n32", "powerpcle-linux"),
            "e-m:e-i64:64-n32");
  EXPECT_EQ(UpgradeDataLayoutString("E-m:e-Fn32-i64:64-n32", "powerpc-linux"),
            "E-m:e-Fn32-i64:64-n32");
  EXPECT_EQ(UpgradeDataLayoutString("E-m:a-Fi64-i64:64-n32", "powerpc-aix"),
            "E-m:a-Fi64-i64:64-n32");
}

TEST(DataLayoutUpgradeTest, EmptyDataLayout) {
  std::string DL1 = UpgradeDataLayoutString("", "x86_64-unknown-linux-gnu");
  std::string DL2 = UpgradeDataLayoutString(
      "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128", "");
  EXPECT_EQ(DL1, "");
  EXPECT_EQ(DL2, "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128");

  // Check that AMDGPU targets add G1 if it's not present.
  EXPECT_EQ(UpgradeDataLayoutString("", "r600"), "G1");
  EXPECT_EQ(UpgradeDataLayoutString("", "amdgcn"),
            "G1-ni:7:8:9-p7:160:256:256:32-p8:128:128-p9:192:256:256:32");

  // Check that SPIR & SPIRV targets add G1 if it's not present.
  EXPECT_EQ(UpgradeDataLayoutString("", "spir"), "G1");
  EXPECT_EQ(UpgradeDataLayoutString("", "spir64"), "G1");
  EXPECT_EQ(UpgradeDataLayoutString("", "spirv32"), "G1");
  EXPECT_EQ(UpgradeDataLayoutString("", "spirv64"), "G1");
  // but SPIRV Logical does not.
  EXPECT_EQ(UpgradeDataLayoutString("", "spirv"), "");
}

} // end namespace
