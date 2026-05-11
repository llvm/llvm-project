//===- RegionWithScoreTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/RegionWithScore.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct RegionWithScoreTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  std::unique_ptr<TargetTransformInfo> TTI;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    TTI = std::make_unique<TargetTransformInfo>(M->getDataLayout());
    if (!M)
      Err.print("RegionTest", errs());
  }
};

TEST_F(RegionWithScoreTest, RegionCost) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2) {
  %add0 = add i8 %v0, 1
  %add1 = add i8 %v1, 2
  %add2 = add i8 %v2, 3
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  auto *LLVMBB = &*LLVMF->begin();
  auto LLVMIt = LLVMBB->begin();
  auto *LLVMAdd0 = &*LLVMIt++;
  auto *LLVMAdd1 = &*LLVMIt++;
  auto *LLVMAdd2 = &*LLVMIt++;

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Add0 = cast<sandboxir::Instruction>(&*It++);
  auto *Add1 = cast<sandboxir::Instruction>(&*It++);
  auto *Add2 = cast<sandboxir::Instruction>(&*It++);

  sandboxir::RegionWithScore Rgn(Ctx, *TTI);
  const auto &SB = Rgn.getScoreboard();
  EXPECT_EQ(SB.getAfterCost(), 0);
  EXPECT_EQ(SB.getBeforeCost(), 0);

  auto GetCost = [this](llvm::Instruction *LLVMI) {
    constexpr static TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
    SmallVector<const llvm::Value *> Operands(LLVMI->operands());
    return TTI->getInstructionCost(LLVMI, Operands, CostKind);
  };
  // Add `Add0` to the region, should be counted in "After".
  sandboxir::RegionInternalsAttorney::add(Rgn, Add0);
  EXPECT_EQ(SB.getBeforeCost(), 0);
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd0));
  // Same for `Add1`.
  sandboxir::RegionInternalsAttorney::add(Rgn, Add1);
  EXPECT_EQ(SB.getBeforeCost(), 0);
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd0) + GetCost(LLVMAdd1));
  // Remove `Add0`, should be subtracted from "After".
  sandboxir::RegionInternalsAttorney::remove(Rgn, Add0);
  EXPECT_EQ(SB.getBeforeCost(), 0);
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd1));
  // Remove `Add2` which was never in the region, should counted in "Before".
  sandboxir::RegionInternalsAttorney::remove(Rgn, Add2);
  EXPECT_EQ(SB.getBeforeCost(), GetCost(LLVMAdd2));
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd1));
}
