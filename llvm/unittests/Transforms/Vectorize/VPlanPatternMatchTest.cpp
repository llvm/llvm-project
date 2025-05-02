//===- llvm/unittests/Transforms/Vectorize/VPlanPatternMatchTest.cpp ------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanPatternMatch.h"
#include "../lib/Transforms/Vectorize/LoopVectorizationPlanner.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanHelpers.h"
#include "VPlanTestBase.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
using VPPatternMatchTest = VPlanTestBase;

TEST_F(VPPatternMatchTest, ScalarIVSteps) {
  VPlan &Plan = getPlan();
  VPBasicBlock *VPBB = Plan.createVPBasicBlock("");
  VPBuilder Builder(VPBB);

  IntegerType *I64Ty = IntegerType::get(C, 64);
  VPValue *StartV = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 0));
  auto *CanonicalIVPHI = new VPCanonicalIVPHIRecipe(StartV, DebugLoc());
  Builder.insert(CanonicalIVPHI);

  VPValue *Inc = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 1));
  VPValue *VF = &Plan.getVF();
  VPValue *Steps = Builder.createScalarIVSteps(
      Instruction::Add, nullptr, CanonicalIVPHI, Inc, VF, DebugLoc());

  using namespace VPlanPatternMatch;

  ASSERT_TRUE(match(Steps, m_ScalarIVSteps(m_Specific(CanonicalIVPHI),
                                           m_SpecificInt(1), m_Specific(VF))));
}

} // namespace
} // namespace llvm
