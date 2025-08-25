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

// TODO: gcc-10 has a bug that causes the below line not to compile due to some
// macro-magic in gunit in combination with a class with pure-virtual
// function. Once gcc-10 is no longer supported, replace this function with
// something like the following:
//
// EXPECT_THAT(SB, testing::ElementsAre(St0, St1, St2, St3));
static void
ExpectThatElementsAre(sandboxir::SeedBundle &SR,
                      llvm::ArrayRef<sandboxir::Instruction *> Contents) {
  EXPECT_EQ(range_size(SR), Contents.size());
  auto CI = Contents.begin();
  if (range_size(SR) == Contents.size())
    for (auto &S : SR)
      EXPECT_EQ(S, *CI++);
}

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
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
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
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
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

  SC.insert(S0, /*AllowDiffTypes=*/false);
  SC.insert(S1, /*AllowDiffTypes=*/false);
  SC.insert(S2, /*AllowDiffTypes=*/false);
  SC.insert(S3, /*AllowDiffTypes=*/false);
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

TEST_F(SeedBundleTest, ConsecutiveStores) {
  // Where "Consecutive" means the stores address consecutive locations in
  // memory, but not in program order. Check to see that the collector puts them
  // in the proper order for vectorization.
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, float %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ptr2 = getelementptr float, ptr %ptr, i32 2
  %ptr3 = getelementptr float, ptr %ptr, i32 3
  store float %val, ptr %ptr0
  store float %val, ptr %ptr2
  store float %val, ptr %ptr1
  store float %val, ptr %ptr3
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto BB = F.begin();
  sandboxir::SeedCollector SC(&*BB, SE, /*CollectStores=*/true,
                              /*CollectLoads=*/false);

  // Find the stores
  auto It = std::next(BB->begin(), 4);
  // StX with X as the order by offset in memory
  auto *St0 = &*It++;
  auto *St2 = &*It++;
  auto *St1 = &*It++;
  auto *St3 = &*It++;

  auto StoreSeedsRange = SC.getStoreSeeds();
  auto &SB = *StoreSeedsRange.begin();
  //  Expect just one vector of store seeds
  EXPECT_EQ(range_size(StoreSeedsRange), 1u);
  ExpectThatElementsAre(SB, {St0, St1, St2, St3});
}

TEST_F(SeedBundleTest, StoresWithGaps) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, float %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 3
  %ptr2 = getelementptr float, ptr %ptr, i32 5
  %ptr3 = getelementptr float, ptr %ptr, i32 7
  store float %val, ptr %ptr0
  store float %val, ptr %ptr2
  store float %val, ptr %ptr1
  store float %val, ptr %ptr3
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto BB = F.begin();
  sandboxir::SeedCollector SC(&*BB, SE, /*CollectStores=*/true,
                              /*CollectLoads=*/false);

  // Find the stores
  auto It = std::next(BB->begin(), 4);
  // StX with X as the order by offset in memory
  auto *St0 = &*It++;
  auto *St2 = &*It++;
  auto *St1 = &*It++;
  auto *St3 = &*It++;

  auto StoreSeedsRange = SC.getStoreSeeds();
  auto &SB = *StoreSeedsRange.begin();
  // Expect just one vector of store seeds
  EXPECT_EQ(range_size(StoreSeedsRange), 1u);
  ExpectThatElementsAre(SB, {St0, St1, St2, St3});
  // Check that the EraseInstr callback works.

  // TODO: Range_size counts fully used-bundles even though the iterator skips
  // them. Further, iterating over anything other than the Bundles in a
  // SeedContainer includes used seeds. So for now just check that removing all
  // the seeds from a bundle also empties the bundle.
  St0->eraseFromParent();
  St1->eraseFromParent();
  St2->eraseFromParent();
  St3->eraseFromParent();
  size_t nonEmptyBundleCount = 0;
  for (auto &B : SC.getStoreSeeds()) {
    (void)B;
    nonEmptyBundleCount++;
  }
  EXPECT_EQ(nonEmptyBundleCount, 0u);
}

TEST_F(SeedBundleTest, VectorStores) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, <2 x float> %val0, i64 %val1) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ptr2 = getelementptr i64, ptr %ptr, i32 2
  store <2 x float> %val0, ptr %ptr1
  store <2 x float> %val0, ptr %ptr0
  store atomic i64 %val1, ptr %ptr2 unordered, align 8
  store volatile i64 %val1, ptr %ptr2

  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto BB = F.begin();
  sandboxir::SeedCollector SC(&*BB, SE, /*CollectStores=*/true,
                              /*CollectLoads=*/false);

  // Find the stores
  auto It = std::next(BB->begin(), 3);
  // StX with X as the order by offset in memory
  auto *St1 = &*It++;
  auto *St0 = &*It++;

  auto StoreSeedsRange = SC.getStoreSeeds();
  EXPECT_EQ(range_size(StoreSeedsRange), 1u);
  auto &SB = *StoreSeedsRange.begin();
  // isValidMemSeed check: The atomic and volatile stores should not
  // be included in the bundle, but the vector stores should be.
  ExpectThatElementsAre(SB, {St0, St1});
}

TEST_F(SeedBundleTest, MixedScalarVectors) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, float %v, <2 x float> %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ptr3 = getelementptr float, ptr %ptr, i32 3
  store float %v, ptr %ptr0
  store float %v, ptr %ptr3
  store <2 x float> %val, ptr %ptr1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto BB = F.begin();
  sandboxir::SeedCollector SC(&*BB, SE, /*CollectStores=*/true,
                              /*CollectLoads=*/false);

  // Find the stores
  auto It = std::next(BB->begin(), 3);
  // StX with X as the order by offset in memory
  auto *St0 = &*It++;
  auto *St3 = &*It++;
  auto *St1 = &*It++;

  auto StoreSeedsRange = SC.getStoreSeeds();
  EXPECT_EQ(range_size(StoreSeedsRange), 1u);
  auto &SB = *StoreSeedsRange.begin();
  // isValidMemSeedCheck here: all of the three stores should be included.
  ExpectThatElementsAre(SB, {St0, St1, St3});
}

TEST_F(SeedBundleTest, DiffTypes) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, i8 %v, i16 %v16) {
bb:
  %ptr0 = getelementptr i8, ptr %ptr, i32 0
  %ptr1 = getelementptr i8, ptr %ptr, i32 1
  %ptr3 = getelementptr i8, ptr %ptr, i32 3
  store i8 %v, ptr %ptr0
  store i8 %v, ptr %ptr3
  store i16 %v16, ptr %ptr1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto BB = F.begin();
  auto It = std::next(BB->begin(), 3);
  auto *St0 = &*It++;
  auto *St3 = &*It++;
  auto *St1 = &*It++;

  sandboxir::SeedCollector SC(&*BB, SE, /*CollectStores=*/true,
                              /*CollectLoads=*/false, /*AllowDiffTypes=*/true);

  auto StoreSeedsRange = SC.getStoreSeeds();
  EXPECT_EQ(range_size(StoreSeedsRange), 1u);
  auto &SB = *StoreSeedsRange.begin();
  ExpectThatElementsAre(SB, {St0, St1, St3});
}

TEST_F(SeedBundleTest, VectorLoads) {
  parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, <2 x float> %val0) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %r0 = load <2 x float>, ptr %ptr0
  %r1 = load <2 x float>, ptr %ptr1
  %r2 = load atomic i64, ptr %ptr0 unordered, align 8
  %r3 = load volatile i64, ptr %ptr1
  %r4 = load void()*, ptr %ptr1

  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  DominatorTree DT(LLVMF);
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M->getDataLayout());
  LoopInfo LI(DT);
  AssumptionCache AC(LLVMF);
  ScalarEvolution SE(LLVMF, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto BB = F.begin();
  sandboxir::SeedCollector SC(&*BB, SE, /*CollectStores=*/false,
                              /*CollectLoads=*/true);

  // Find the loads
  auto It = std::next(BB->begin(), 2);
  // StX with X as the order by offset in memory
  auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  auto *Ld1 = cast<sandboxir::LoadInst>(&*It++);

  auto LoadSeedsRange = SC.getLoadSeeds();
  EXPECT_EQ(range_size(LoadSeedsRange), 2u);
  auto &SB = *LoadSeedsRange.begin();
  // isValidMemSeed check: The atomic and volatile loads should not
  // be included in the bundle, the vector stores should be, but the
  // void-typed load should not.
  ExpectThatElementsAre(SB, {Ld0, Ld1});
}
