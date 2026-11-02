//===- ScaledWMMAMicroOpTest.cpp - Tests for Scaled WMMA micro-op modeling ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCoExecInfo.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

TEST(ScaledWMMAMicroOpTest, XSlotBlocksEverything) {
  // X slot (LD_SCALE cycle): nothing can coexecute.
  auto Info = CoExecInfo::build(8, 11, "X0EEIEEIIVV");

  EXPECT_EQ(Info.getType(0), CoExecStageType::X);
  EXPECT_FALSE(Info.canCoExec(CoExecMask::VALU, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::TRANS, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::DS, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::SALU, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::WMMA, 0));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::CTRL, 0));
}

TEST(ScaledWMMAMicroOpTest, ScaledWMMASlotLayout) {
  // F8F6F4 16x16x128 scaled: X0EEIEEIIVV (11 cycles)
  auto Info = CoExecInfo::build(8, 11, "X0EEIEEIIVV");

  EXPECT_EQ(Info.UnitOccupancy, 8u);
  EXPECT_EQ(Info.TotalWindow, 11u);

  // Stage layout
  EXPECT_EQ(Info.getType(0), CoExecStageType::X);
  EXPECT_EQ(Info.getType(1), CoExecStageType::E0);
  EXPECT_EQ(Info.getType(2), CoExecStageType::E);
  EXPECT_EQ(Info.getType(3), CoExecStageType::E);
  EXPECT_EQ(Info.getType(4), CoExecStageType::I);
  EXPECT_EQ(Info.getType(5), CoExecStageType::E);
  EXPECT_EQ(Info.getType(6), CoExecStageType::E);
  EXPECT_EQ(Info.getType(7), CoExecStageType::I);
  EXPECT_EQ(Info.getType(8), CoExecStageType::I);
  EXPECT_EQ(Info.getType(9), CoExecStageType::V);
  EXPECT_EQ(Info.getType(10), CoExecStageType::V);
}

TEST(ScaledWMMAMicroOpTest, NonScaledWMMASlotLayout) {
  auto Info = CoExecInfo::build(8, 10, "0EEIEEIIVV");

  EXPECT_EQ(Info.UnitOccupancy, 8u);
  EXPECT_EQ(Info.TotalWindow, 10u);

  // Stage 7 is a regular I slot, NOT an S slot
  EXPECT_EQ(Info.getMask(7), CoExecMask::StageI);
  EXPECT_TRUE(Info.canCoExec(CoExecMask::VALU, 7));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::TRANS, 7));
  // WMMA is NOT allowed in I slot
  EXPECT_FALSE(Info.canCoExec(CoExecMask::WMMA, 7));
}

TEST(ScaledWMMAMicroOpTest, ScaledWMMACanIssueAtISlot) {
  // In a scaled WMMA's I slot, another scaled WMMA can begin
  // because its LD_SCALE uses VALU.
  auto Info = CoExecInfo::build(8, 11, "X0EEIEEIIVV");

  // I slots at stages 4, 7, 8
  // Using the IsScaledWMMA overload, WMMA should be allowed at I slots
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 4, true));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 7, true));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 8, true));

  // Non-scaled WMMA should NOT be allowed at I slots
  EXPECT_FALSE(Info.canCoExec(CoExecMask::WMMA, 4, false));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::WMMA, 7, false));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::WMMA, 8, false));

  // Both should be allowed at V slots
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 9, true));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 9, false));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 10, true));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 10, false));
}

TEST(ScaledWMMAMicroOpTest, FindNextWMMASlotScaled) {
  auto Info = CoExecInfo::build(8, 11, "X0EEIEEIIVV");

  // For scaled WMMA: first allowed slot from stage 0 should be stage 4 (I)
  auto Next = Info.findNextAllowedStage(CoExecMask::WMMA, 0, true);
  ASSERT_TRUE(Next.has_value());
  EXPECT_EQ(*Next, 4u);

  // For non-scaled WMMA: first allowed from stage 0 is stage 9 (V)
  Next = Info.findNextAllowedStage(CoExecMask::WMMA, 0, false);
  ASSERT_TRUE(Next.has_value());
  EXPECT_EQ(*Next, 9u);
}

TEST(ScaledWMMAMicroOpTest, FindNextWMMASlotNonScaled) {
  auto Info = CoExecInfo::build(8, 10, "0EEIEEIIVV");

  // From stage 0, first slot allowing WMMA is stage 8 (V slot)
  auto Next = Info.findNextAllowedStage(CoExecMask::WMMA, 0);
  ASSERT_TRUE(Next.has_value());
  EXPECT_EQ(*Next, 8u);
}

TEST(ScaledWMMAMicroOpTest, ShortScaledWMMAPattern) {
  // F4xF4: X0EEIVV (7 cycles)
  auto Info = CoExecInfo::build(4, 7, "X0EEIVV");

  EXPECT_EQ(Info.UnitOccupancy, 4u);
  EXPECT_EQ(Info.TotalWindow, 7u);

  EXPECT_EQ(Info.getType(0), CoExecStageType::X);
  EXPECT_EQ(Info.getType(1), CoExecStageType::E0);
  EXPECT_EQ(Info.getType(4), CoExecStageType::I);
  EXPECT_EQ(Info.getType(5), CoExecStageType::V);
  EXPECT_EQ(Info.getType(6), CoExecStageType::V);

  // Scaled WMMA can issue at I slot (stage 4)
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 4, true));
  EXPECT_FALSE(Info.canCoExec(CoExecMask::WMMA, 4, false));
}

TEST(ScaledWMMAMicroOpTest, CoExecSlotCountComparison) {
  auto ScaledInfo = CoExecInfo::build(8, 11, "X0EEIEEIIVV");
  auto NonScaledInfo = CoExecInfo::build(8, 10, "0EEIEEIIVV");

  // Non-scaled: 3 VALU slots (I at stages 3, 6, 7)
  unsigned NonScaledVALUSlots = NonScaledInfo.getCoExecStageCount(CoExecMask::VALU);
  EXPECT_EQ(NonScaledVALUSlots, 3u);

  // Scaled: 3 VALU slots (I at stages 4, 7, 8)
  unsigned ScaledVALUSlots = ScaledInfo.getCoExecStageCount(CoExecMask::VALU);
  EXPECT_EQ(ScaledVALUSlots, 3u);

  // Non-scaled: 2 WMMA slots (V at stages 8, 9)
  unsigned NonScaledWMMASlots = NonScaledInfo.getCoExecStageCount(CoExecMask::WMMA);
  EXPECT_EQ(NonScaledWMMASlots, 2u);

  // Scaled: 2 WMMA slots (V at stages 9, 10) - I slots don't allow WMMA
  // in the basic canCoExec (without IsScaledWMMA flag)
  unsigned ScaledWMMASlots = ScaledInfo.getCoExecStageCount(CoExecMask::WMMA);
  EXPECT_EQ(ScaledWMMASlots, 2u);
}

TEST(ScaledWMMAMicroOpTest, SSlotStillWorks) {
  // Verify old S-slot patterns still work (backwards compatibility)
  auto Info = CoExecInfo::build(8, 10, "0EEIEEISVV");

  EXPECT_EQ(Info.getMask(7), CoExecMask::StageIS);
  EXPECT_TRUE(Info.canCoExec(CoExecMask::VALU, 7));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::WMMA, 7));
  EXPECT_TRUE(Info.canCoExec(CoExecMask::TRANS, 7));
}

} // namespace
