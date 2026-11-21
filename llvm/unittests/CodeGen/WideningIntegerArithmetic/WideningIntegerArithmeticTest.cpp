//===- WideningIntegerArithmeticTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/WideningIntegerArithmeticInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

using Solution = WideningIntegerSolutionInfo;

class WIASolutionTest : public testing::Test {
protected:
  LLVMContext Context;
  Value *FirstValue = ConstantInt::get(Type::getInt8Ty(Context), 1);
  Value *SecondValue = ConstantInt::get(Type::getInt8Ty(Context), 2);

  std::unique_ptr<Solution> makeSolution(unsigned short PayloadWidth,
                                         InstructionCost Cost,
                                         Value *V = nullptr,
                                         IntegerFillType Fill = SIGN,
                                         unsigned short UpdatedWidth = 64) {
    return std::make_unique<Solution>(Instruction::Add, Instruction::Add, Fill,
                                      PayloadWidth, 8, UpdatedWidth, Cost,
                                      WIAK_UNKNOWN, V ? V : FirstValue);
  }
};

TEST_F(WIASolutionTest, DominanceComparesBothCostAndPayload) {
  struct TestCase {
    unsigned short BetterPayload;
    InstructionCost BetterCost;
    unsigned short WorsePayload;
    InstructionCost WorseCost;
  };
  const TestCase Cases[] = {{8, 1, 8, 3}, {4, 3, 8, 3}, {4, 1, 8, 3}};

  for (const TestCase &C : Cases) {
    SCOPED_TRACE(testing::Message()
                 << C.BetterPayload << ", " << C.BetterCost.getValue());
    auto Better = makeSolution(C.BetterPayload, C.BetterCost);
    auto Worse = makeSolution(C.WorsePayload, C.WorseCost);
    EXPECT_EQ(Better->IsRedudant(*Worse), -1);
    EXPECT_EQ(Worse->IsRedudant(*Better), 1);
  }
}

TEST_F(WIASolutionTest, ValueFillAndUpdatedWidthKeepCandidatesDistinct) {
  auto Original = makeSolution(8, 3);
  auto OtherValue = makeSolution(4, 1, SecondValue);
  auto OtherFill = makeSolution(4, 1, FirstValue, ZEROS);
  auto OtherWidth = makeSolution(4, 1, FirstValue, SIGN, 32);

  for (Solution *Other :
       {OtherValue.get(), OtherFill.get(), OtherWidth.get()}) {
    EXPECT_EQ(Original->IsRedudant(*Other), 0);
    EXPECT_EQ(Other->IsRedudant(*Original), 0);
  }
}

// make solution(Payload, cost, value, Sign, updatedW)

TEST_F(WIASolutionTest, CostAndPayLoadTradeOffKeepCandidates) {
  auto Sol1 = makeSolution(8, 2);
  auto Sol2 = makeSolution(4, 3);

  EXPECT_EQ(Sol1->IsRedudant(*Sol2), 0);
  EXPECT_EQ(Sol2->IsRedudant(*Sol1), 0);
}

} // namespace
