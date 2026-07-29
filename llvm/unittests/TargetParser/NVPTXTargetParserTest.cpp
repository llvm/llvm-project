//===----------- NVPTXTargetParserTest.cpp - NVPTX Target Parser ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/NVPTXTargetParser.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(NVPTXTargetParserTest, ParseArch) {
  EXPECT_EQ(NVPTX::parseArch("sm_90"), NVPTX::GK_SM_90);
  EXPECT_EQ(NVPTX::parseArch("sm_90a"), NVPTX::GK_SM_90a);
  EXPECT_EQ(NVPTX::parseArch("sm_100f"), NVPTX::GK_SM_100f);
  // sm_32 uses the underscore-suffixed enumerator internally but the canonical
  // name has no trailing underscore.
  EXPECT_EQ(NVPTX::parseArch("sm_32"), NVPTX::GK_SM_32_);
  EXPECT_EQ(NVPTX::parseArch("gfx900"), NVPTX::GK_NONE);
  EXPECT_EQ(NVPTX::parseArch(""), NVPTX::GK_NONE);
}

TEST(NVPTXTargetParserTest, ArchNames) {
  EXPECT_EQ(NVPTX::getArchName(NVPTX::GK_SM_90), "sm_90");
  EXPECT_EQ(NVPTX::getArchName(NVPTX::GK_SM_32_), "sm_32");
  EXPECT_EQ(NVPTX::getVirtualArch(NVPTX::GK_SM_90), "compute_90");
  // sm_21 shares the compute_20 virtual arch.
  EXPECT_EQ(NVPTX::getVirtualArch(NVPTX::GK_SM_21), "compute_20");
  EXPECT_EQ(NVPTX::getArchName(NVPTX::GK_NONE), "");
  EXPECT_EQ(NVPTX::getVirtualArch(NVPTX::GK_NONE), "");
}

TEST(NVPTXTargetParserTest, SmVersion) {
  EXPECT_EQ(NVPTX::getSmVersion(NVPTX::GK_SM_90), 900u);
  EXPECT_EQ(NVPTX::getSmVersion(NVPTX::GK_SM_90a), 900u);
  EXPECT_EQ(NVPTX::getSmVersion(NVPTX::GK_SM_100f), 1000u);
  EXPECT_EQ(NVPTX::getSmVersion(NVPTX::GK_NONE), 0u);
}

TEST(NVPTXTargetParserTest, ArchSuffix) {
  EXPECT_FALSE(NVPTX::isAcceleratedArch(NVPTX::GK_SM_90));
  EXPECT_TRUE(NVPTX::isAcceleratedArch(NVPTX::GK_SM_90a));
  EXPECT_FALSE(NVPTX::isAcceleratedArch(NVPTX::GK_SM_100f));

  // Family-specific covers both 'f' and 'a' variants.
  EXPECT_FALSE(NVPTX::isFamilySpecificArch(NVPTX::GK_SM_90));
  EXPECT_TRUE(NVPTX::isFamilySpecificArch(NVPTX::GK_SM_100f));
  EXPECT_TRUE(NVPTX::isFamilySpecificArch(NVPTX::GK_SM_90a));
}

// Every parseable name must round-trip back to the same canonical name.
TEST(NVPTXTargetParserTest, RoundTrip) {
  static const char *const Names[] = {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX) NAME,
#include "llvm/TargetParser/NVPTXTargetParser.def"
  };
  for (const char *Name : Names) {
    NVPTX::GPUKind Kind = NVPTX::parseArch(Name);
    EXPECT_NE(Kind, NVPTX::GK_NONE) << Name;
    EXPECT_EQ(NVPTX::getArchName(Kind), Name);
  }
}

} // namespace
