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

  VPValue *Inc2 = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 2));
  VPValue *Steps2 = Builder.createScalarIVSteps(
      Instruction::Add, nullptr, CanonicalIVPHI, Inc2, VF, DebugLoc());

  using namespace VPlanPatternMatch;

  ASSERT_TRUE(match(Steps, m_ScalarIVSteps(m_Specific(CanonicalIVPHI),
                                           m_SpecificInt(1), m_Specific(VF))));
  ASSERT_FALSE(
      match(Steps2, m_ScalarIVSteps(m_Specific(CanonicalIVPHI),
                                    m_SpecificInt(1), m_Specific(VF))));
  ASSERT_TRUE(match(Steps2, m_ScalarIVSteps(m_Specific(CanonicalIVPHI),
                                            m_SpecificInt(2), m_Specific(VF))));
}

TEST_F(VPPatternMatchTest, GetElementPtr) {
  VPlan &Plan = getPlan();
  VPBasicBlock *VPBB = Plan.createVPBasicBlock("entry");
  VPBuilder Builder(VPBB);

  IntegerType *I64Ty = IntegerType::get(C, 64);
  VPValue *One = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 1));
  VPValue *Two = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 2));
  VPValue *Ptr =
      Plan.getOrAddLiveIn(Constant::getNullValue(PointerType::get(C, 0)));

  VPInstruction *PtrAdd = Builder.createPtrAdd(Ptr, One);
  VPInstruction *WidePtrAdd = Builder.createWidePtrAdd(Ptr, Two);

  using namespace VPlanPatternMatch;
  ASSERT_TRUE(
      match(PtrAdd, m_GetElementPtr(m_Specific(Ptr), m_SpecificInt(1))));
  ASSERT_FALSE(
      match(PtrAdd, m_GetElementPtr(m_Specific(Ptr), m_SpecificInt(2))));
  ASSERT_TRUE(
      match(WidePtrAdd, m_GetElementPtr(m_Specific(Ptr), m_SpecificInt(2))));
  ASSERT_FALSE(
      match(WidePtrAdd, m_GetElementPtr(m_Specific(Ptr), m_SpecificInt(1))));
}
} // namespace
} // namespace llvm
