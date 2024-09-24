//===- SeedCollectorTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SeedCollector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SeedBundleTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("LegalityTest", errs());
  }
};

TEST_F(SeedBundleTest, SeedBundle) {
  parseIR(C, R"IR(
define void @foo(float %v0, float %v1) {
bb:
  %add0 = fadd float %v0, %v1
  %add1 = fadd float %v0, %v1
  %add2 = fadd float %v0, %v1
  %add3 = fadd float %v0, %v1
  %add4 = fadd float %v0, %v1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  DataLayout DL(M->getDataLayout());
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  // Assume test instructions are identical in the number of bits.
  const unsigned kFloatBits = sandboxir::Utils::getNumBits(I0, DL);
  // Constructor
  sandboxir::SeedBundle SBO(I0, DL);
  EXPECT_EQ(*SBO.begin(), I0);
  // getNumUnusedBits after constructor
  EXPECT_EQ(SBO.getNumUnusedBits(), kFloatBits);
  // setUsed
  SBO.setUsed(I0, DL);
  // allUsed
  EXPECT_TRUE(SBO.allUsed());
  // isUsed
  EXPECT_TRUE(SBO.isUsed(0));
  // getNumUnusedBits after setUsed
  EXPECT_EQ(SBO.getNumUnusedBits(), 0u);
  // insertAt
  SBO.insertAt(SBO.end(), I1, DL);
  EXPECT_NE(*SBO.begin(), I1);
  // getNumUnusedBits after insertAt
  EXPECT_EQ(SBO.getNumUnusedBits(), kFloatBits);
  // allUsed
  EXPECT_FALSE(SBO.allUsed());
  // getFirstUnusedElement
  EXPECT_EQ(SBO.getFirstUnusedElementIdx(), 1u);

  sandboxir::SeedBundle::SeedList Seeds;
  It = BB->begin();
  Seeds.push_back(&*It++);
  Seeds.push_back(&*It++);
  Seeds.push_back(&*It++);
  Seeds.push_back(&*It++);
  Seeds.push_back(&*It++);
  // Constructor
  sandboxir::SeedBundle SB1(std::move(Seeds), DL);
  // getNumUnusedBits after constructor
  EXPECT_EQ(SB1.getNumUnusedBits(), 5 * kFloatBits);
  // setUsed with index
  SB1.setUsed(1, DL);
  // getFirstUnusedElementIdx
  EXPECT_EQ(SB1.getFirstUnusedElementIdx(), 0u);
  EXPECT_EQ(SB1.getNumUnusedBits(), 4 * kFloatBits);
  SB1.setUsed(unsigned(0), DL);
  // getFirstUnusedElementIdx not at end
  EXPECT_EQ(SB1.getFirstUnusedElementIdx(), 2u);
  // getSlice
  auto Slice0 = SB1.getSlice(2, /* MaxVecRegBits */ kFloatBits * 2,
                             /* ForcePowerOf2 */ true, DL);
  EXPECT_EQ(Slice0.size(), 2u);
  auto Slice1 = SB1.getSlice(2, /* MaxVecRegBits */ kFloatBits * 3,
                             /* ForcePowerOf2 */ false, DL);
  EXPECT_EQ(Slice1.size(), 3u);
}
