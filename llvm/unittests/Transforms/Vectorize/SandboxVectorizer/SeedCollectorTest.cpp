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
#include "llvm/Testing/Support/SupportHelpers.h"
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
define void @foo(float %v0, i32 %i0, i16 %i1, i8 %i2) {
bb:
  %add0 = fadd float %v0, %v0
  %add1 = fadd float %v0, %v0
  %add2 = add i8 %i2, %i2
  %add3 = add i16 %i1, %i1
  %add4 = add i32 %i0, %i0
  %add5 = add i16 %i1, %i1
  %add6 = add i8 %i2, %i2
  %add7 = add i8 %i2, %i2
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
  // Assume first two instructions are identical in the number of bits.
  const unsigned IOBits = sandboxir::Utils::getNumBits(I0, DL);
  // Constructor
  sandboxir::SeedBundle SBO(I0);
  EXPECT_EQ(*SBO.begin(), I0);
  // getNumUnusedBits after constructor
  EXPECT_EQ(SBO.getNumUnusedBits(), IOBits);
  // setUsed
  SBO.setUsed(I0);
  // allUsed
  EXPECT_TRUE(SBO.allUsed());
  // isUsed
  EXPECT_TRUE(SBO.isUsed(0));
  // getNumUnusedBits after setUsed
  EXPECT_EQ(SBO.getNumUnusedBits(), 0u);
  // insertAt
  SBO.insertAt(SBO.end(), I1);
  EXPECT_NE(*SBO.begin(), I1);
  // getNumUnusedBits after insertAt
  EXPECT_EQ(SBO.getNumUnusedBits(), IOBits);
  // allUsed
  EXPECT_FALSE(SBO.allUsed());
  // getFirstUnusedElement
  EXPECT_EQ(SBO.getFirstUnusedElementIdx(), 1u);

  SmallVector<sandboxir::Instruction *> Insts;
  // add2 through add7
  Insts.push_back(&*It++);
  Insts.push_back(&*It++);
  Insts.push_back(&*It++);
  Insts.push_back(&*It++);
  Insts.push_back(&*It++);
  Insts.push_back(&*It++);
  unsigned BundleBits = 0;
  for (auto &S : Insts)
    BundleBits += sandboxir::Utils::getNumBits(S);
  // Ensure the instructions are as expected.
  EXPECT_EQ(BundleBits, 88u);
  auto Seeds = Insts;
  // Constructor
  sandboxir::SeedBundle SB1(std::move(Seeds));
  // getNumUnusedBits after constructor
  EXPECT_EQ(SB1.getNumUnusedBits(), BundleBits);
  // setUsed with index
  SB1.setUsed(1);
  // getFirstUnusedElementIdx
  EXPECT_EQ(SB1.getFirstUnusedElementIdx(), 0u);
  SB1.setUsed(unsigned(0));
  // getFirstUnusedElementIdx not at end
  EXPECT_EQ(SB1.getFirstUnusedElementIdx(), 2u);

  // getSlice is (StartIdx, MaxVecRegBits, ForcePowerOf2). It's easier to
  // compare test cases without the parameter-name comments inline.
  auto Slice0 = SB1.getSlice(2, 64, true);
  EXPECT_THAT(Slice0,
              testing::ElementsAre(Insts[2], Insts[3], Insts[4], Insts[5]));
  auto Slice1 = SB1.getSlice(2, 72, true);
  EXPECT_THAT(Slice1,
              testing::ElementsAre(Insts[2], Insts[3], Insts[4], Insts[5]));
  auto Slice2 = SB1.getSlice(2, 80, true);
  EXPECT_THAT(Slice2,
              testing::ElementsAre(Insts[2], Insts[3], Insts[4], Insts[5]));

  SB1.setUsed(2);
  auto Slice3 = SB1.getSlice(3, 64, false);
  EXPECT_THAT(Slice3, testing::ElementsAre(Insts[3], Insts[4], Insts[5]));
  // getSlice empty case
  SB1.setUsed(3);
  auto Slice4 = SB1.getSlice(4, /* MaxVecRegBits */ 8,
                             /* ForcePowerOf2 */ true);
  EXPECT_EQ(Slice4.size(), 0u);
}
