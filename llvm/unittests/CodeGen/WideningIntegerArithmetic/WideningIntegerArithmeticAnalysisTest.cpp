//===- WideningIntegerArithmeticAnalysisTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../../lib/CodeGen/WideningIntegerArithmetic.cpp"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using WIA = WideningIntegerArithmetic;
using Solution = WideningIntegerSolutionInfo;

class WIAAnalysisTest : public testing::Test {
protected:
  LLVMContext Context;
  DataLayout Layout{"e-p:64:64-p1:32:32"};
  Module TestModule{"wia", Context};
  IRBuilder<NoFolder> Builder{TestModule};
  WIA Analysis;

  WIAAnalysisTest() { Analysis.DL = &Layout; }

  auto &solutions() { return Analysis.AvailableSolutions; }

  Solution *makeSolution(Value *V, unsigned short PayloadWidth,
                         InstructionCost Cost) {
    return Analysis.SolutionAllocator.allocateSolution<Solution>(
        Instruction::Add, Instruction::Add, SIGN, PayloadWidth, 8, 64, Cost,
        WIAK_UNKNOWN, V);
  }
};

TEST_F(WIAAnalysisTest, AddNonRedundantPruneBadSols) {
  auto *V = Builder.getInt8(1);
  auto *LowerCost = makeSolution(V, 8, 1);
  auto *LowerPayload = makeSolution(V, 4, 3);
  auto *Better = makeSolution(V, 4, 1);
  ASSERT_TRUE(Analysis.addNonRedundant(V, LowerCost));
  ASSERT_TRUE(Analysis.addNonRedundant(V, LowerPayload));
  ASSERT_EQ(solutions().at(V).size(), 2u);
  EXPECT_TRUE(Analysis.addNonRedundant(V, Better));
  ASSERT_EQ(solutions().at(V).size(), 1u);
  EXPECT_EQ(solutions().at(V).front(), Better);
  EXPECT_FALSE(Analysis.addNonRedundant(V, makeSolution(V, 4, 1)));
  EXPECT_FALSE(Analysis.addNonRedundant(V, makeSolution(V, 8, 3)));
  EXPECT_EQ(solutions().at(V).front(), Better);
}

TEST_F(WIAAnalysisTest, RepeatedUndefOrPoisonVisitsKeepsSols) {

  auto *V = PoisonValue::get(Builder.getInt8Ty());
  Analysis.visitUndefOrPoison(V);
  ASSERT_FALSE(solutions().at(V).empty());
  const auto Sols = solutions().at(V);
  // Calling visit undef second time shouldn't change anything.
  Analysis.visitUndefOrPoison(V);
  EXPECT_EQ(Sols, solutions().at(V));
}

} // namespace
