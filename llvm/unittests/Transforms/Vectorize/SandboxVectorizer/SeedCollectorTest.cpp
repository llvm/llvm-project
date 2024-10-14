//===- SeedCollectorTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SeedCollector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
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
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

// Stub class to make the abstract base class testable.
class SeedBundleForTest : public sandboxir::SeedBundle {
public:
  using sandboxir::SeedBundle::SeedBundle;
  void insert(sandboxir::Instruction *I, ScalarEvolution &SE) override {
    insertAt(Seeds.end(), I);
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
  SeedBundleForTest SBO(I0);
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
  SeedBundleForTest SB1(std::move(Seeds));
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

TEST_F(SeedBundleTest, MemSeedBundle) {
  parseIR(C, R"IR(
define void @foo(ptr %ptrA, float %val, ptr %ptr) {
bb:
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %gep2 = getelementptr float, ptr %ptr, i32 3
  %gep3 = getelementptr float, ptr %ptr, i32 4
  store float %val, ptr %gep0
  store float %val, ptr %gep1
  store float %val, ptr %gep2
  store float %val, ptr %gep3

  load float, ptr %gep0
  load float, ptr %gep1
  load float, ptr %gep2
  load float, ptr %gep3

  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");

  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = std::next(BB->begin(), 4);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);

  // Single instruction constructor; test insert out of memory order
  sandboxir::StoreSeedBundle SB(S3);
  SB.insert(S1, SE);
  SB.insert(S2, SE);
  SB.insert(S0, SE);
  EXPECT_THAT(SB, testing::ElementsAre(S0, S1, S2, S3));

  // Instruction list constructor; test list out of order
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);
  SmallVector<sandboxir::Instruction *> Loads;
  Loads.push_back(L1);
  Loads.push_back(L3);
  Loads.push_back(L2);
  Loads.push_back(L0);
  sandboxir::LoadSeedBundle LB(std::move(Loads), SE);
  EXPECT_THAT(LB, testing::ElementsAre(L0, L1, L2, L3));
}

TEST_F(SeedBundleTest, Container) {
  parseIR(C, R"IR(
define void @foo(ptr %ptrA, float %val, ptr %ptrB) {
bb:
  %gepA0 = getelementptr float, ptr %ptrA, i32 0
  %gepA1 = getelementptr float, ptr %ptrA, i32 1
  %gepB0 = getelementptr float, ptr %ptrB, i32 0
  %gepB1 = getelementptr float, ptr %ptrB, i32 1
  store float %val, ptr %gepA0
  store float %val, ptr %gepA1
  store float %val, ptr %gepB0
  store float %val, ptr %gepB1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");

  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *F.begin();
  auto It = std::next(BB.begin(), 4);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);
  sandboxir::SeedContainer SC(SE);
  // Check begin() end() when empty.
  EXPECT_EQ(SC.begin(), SC.end());

  SC.insert(S0);
  SC.insert(S1);
  SC.insert(S2);
  SC.insert(S3);
  unsigned Cnt = 0;
  SmallVector<sandboxir::SeedBundle *> Bndls;
  for (auto &SeedBndl : SC) {
    EXPECT_EQ(SeedBndl.size(), 2u);
    ++Cnt;
    Bndls.push_back(&SeedBndl);
  }
  EXPECT_EQ(Cnt, 2u);

  // Mark them "Used" to check if operator++ skips them in the next loop.
  for (auto *SeedBndl : Bndls)
    for (auto Lane : seq<unsigned>(SeedBndl->size()))
      SeedBndl->setUsed(Lane);
  // Check if iterator::operator++ skips used lanes.
  Cnt = 0;
  for (auto &SeedBndl : SC) {
    (void)SeedBndl;
    ++Cnt;
  }
  EXPECT_EQ(Cnt, 0u);
}
