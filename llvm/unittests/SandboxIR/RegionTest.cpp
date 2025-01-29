//===- RegionTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Region.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct RegionTest : public testing::Test {
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

TEST_F(RegionTest, Basic) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1
  ret i8 %t1
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *Ret = cast<sandboxir::Instruction>(&*It++);
  sandboxir::Region Rgn(Ctx, *TTI);

  // Check getContext.
  EXPECT_EQ(&Ctx, &Rgn.getContext());

  // Check add / remove / empty.
  EXPECT_TRUE(Rgn.empty());
  Rgn.add(T0);
  EXPECT_FALSE(Rgn.empty());
  Rgn.remove(T0);
  EXPECT_TRUE(Rgn.empty());

  // Check iteration.
  Rgn.add(T0);
  Rgn.add(T1);
  Rgn.add(Ret);
  // Use an ordered matcher because we're supposed to preserve the insertion
  // order for determinism.
  EXPECT_THAT(Rgn.insts(), testing::ElementsAre(T0, T1, Ret));

  // Check contains
  EXPECT_TRUE(Rgn.contains(T0));
  Rgn.remove(T0);
  EXPECT_FALSE(Rgn.contains(T0));

#ifndef NDEBUG
  // Check equality comparison. Insert in reverse order into `Other` to check
  // that comparison is order-independent.
  sandboxir::Region Other(Ctx, *TTI);
  Other.add(Ret);
  EXPECT_NE(Rgn, Other);
  Other.add(T1);
  EXPECT_EQ(Rgn, Other);
#endif
}

TEST_F(RegionTest, CallbackUpdates) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1, ptr %ptr) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1
  ret i8 %t0
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *Ptr = F->getArg(2);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *Ret = cast<sandboxir::Instruction>(&*It++);
  sandboxir::Region Rgn(Ctx, *TTI);
  Rgn.add(T0);
  Rgn.add(T1);

  // Test creation.
  auto *NewI = sandboxir::StoreInst::create(T0, Ptr, /*Align=*/std::nullopt,
                                            Ret->getIterator(), Ctx);
  EXPECT_THAT(Rgn.insts(), testing::ElementsAre(T0, T1, NewI));

  // Test deletion.
  T1->eraseFromParent();
  EXPECT_THAT(Rgn.insts(), testing::ElementsAre(T0, NewI));
}

TEST_F(RegionTest, MetadataFromIR) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1, !sandboxvec !0
  %t1 = add i8 %t0, %v1, !sandboxvec !1
  %t2 = add i8 %t1, %v1, !sandboxvec !1
  ret i8 %t2
}

!0 = distinct !{!"sandboxregion"}
!1 = distinct !{!"sandboxregion"}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *T2 = cast<sandboxir::Instruction>(&*It++);

  SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
      sandboxir::Region::createRegionsFromMD(*F, *TTI);
  EXPECT_THAT(Regions[0]->insts(), testing::UnorderedElementsAre(T0));
  EXPECT_THAT(Regions[1]->insts(), testing::UnorderedElementsAre(T1, T2));
}

TEST_F(RegionTest, NonContiguousRegion) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1, !sandboxvec !0
  %t1 = add i8 %t0, %v1
  %t2 = add i8 %t1, %v1, !sandboxvec !0
  ret i8 %t2
}

!0 = distinct !{!"sandboxregion"}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  [[maybe_unused]] auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *T2 = cast<sandboxir::Instruction>(&*It++);

  SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
      sandboxir::Region::createRegionsFromMD(*F, *TTI);
  EXPECT_THAT(Regions[0]->insts(), testing::UnorderedElementsAre(T0, T2));
}

TEST_F(RegionTest, DumpedMetadata) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1
  %t2 = add i8 %t1, %v1
  ret i8 %t1
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  [[maybe_unused]] auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *T2 = cast<sandboxir::Instruction>(&*It++);
  [[maybe_unused]] auto *Ret = cast<sandboxir::Instruction>(&*It++);
  sandboxir::Region Rgn(Ctx, *TTI);
  Rgn.add(T0);
  sandboxir::Region Rgn2(Ctx, *TTI);
  Rgn2.add(T2);

  std::string output;
  llvm::raw_string_ostream RSO(output);
  M->print(RSO, nullptr, /*ShouldPreserveUseListOrder=*/true,
           /*IsForDebug=*/true);

  // TODO: Replace this with a lit test, which is more suitable for this kind
  // of IR comparison.
  std::string expected = R"(; ModuleID = '<string>'
source_filename = "<string>"

define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1, !sandboxvec !0
  %t1 = add i8 %t0, %v1
  %t2 = add i8 %t1, %v1, !sandboxvec !1
  ret i8 %t1
}

!0 = distinct !{!"sandboxregion"}
!1 = distinct !{!"sandboxregion"}
)";
  EXPECT_EQ(expected, output);
}

TEST_F(RegionTest, MetadataRoundTrip) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1
  ret i8 %t1
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);

  sandboxir::Region Rgn(Ctx, *TTI);
  Rgn.add(T0);
  Rgn.add(T1);

  SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
      sandboxir::Region::createRegionsFromMD(*F, *TTI);
  ASSERT_EQ(1U, Regions.size());
#ifndef NDEBUG
  EXPECT_EQ(Rgn, *Regions[0].get());
#endif
}

TEST_F(RegionTest, RegionCost) {
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

  sandboxir::Region Rgn(Ctx, *TTI);
  const auto &SB = Rgn.getScoreboard();
  EXPECT_EQ(SB.getAfterCost(), 0);
  EXPECT_EQ(SB.getBeforeCost(), 0);

  auto GetCost = [this](llvm::Instruction *LLVMI) {
    constexpr static TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
    SmallVector<const llvm::Value *> Operands(LLVMI->operands());
    return TTI->getInstructionCost(LLVMI, Operands, CostKind);
  };
  // Add `Add0` to the region, should be counted in "After".
  Rgn.add(Add0);
  EXPECT_EQ(SB.getBeforeCost(), 0);
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd0));
  // Same for `Add1`.
  Rgn.add(Add1);
  EXPECT_EQ(SB.getBeforeCost(), 0);
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd0) + GetCost(LLVMAdd1));
  // Remove `Add0`, should be subtracted from "After".
  Rgn.remove(Add0);
  EXPECT_EQ(SB.getBeforeCost(), 0);
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd1));
  // Remove `Add2` which was never in the region, should counted in "Before".
  Rgn.remove(Add2);
  EXPECT_EQ(SB.getBeforeCost(), GetCost(LLVMAdd2));
  EXPECT_EQ(SB.getAfterCost(), GetCost(LLVMAdd1));
}

TEST_F(RegionTest, Aux) {
  parseIR(C, R"IR(
define void @foo(i8 %v) {
  %t0 = add i8 %v, 0, !sandboxvec !0, !sbaux !2
  %t1 = add i8 %v, 1, !sandboxvec !0, !sbaux !3
  %t2 = add i8 %v, 2, !sandboxvec !1
  %t3 = add i8 %v, 3, !sandboxvec !1, !sbaux !2
  %t4 = add i8 %v, 4, !sandboxvec !1, !sbaux !4
  %t5 = add i8 %v, 5, !sandboxvec !1, !sbaux !3
  ret void
}

!0 = distinct !{!"sandboxregion"}
!1 = distinct !{!"sandboxregion"}

!2 = !{i32 0}
!3 = !{i32 1}
!4 = !{i32 2}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);
  auto *T2 = cast<sandboxir::Instruction>(&*It++);
  auto *T3 = cast<sandboxir::Instruction>(&*It++);
  auto *T4 = cast<sandboxir::Instruction>(&*It++);
  auto *T5 = cast<sandboxir::Instruction>(&*It++);

  SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
      sandboxir::Region::createRegionsFromMD(*F, *TTI);
  // Check that the regions are correct.
  EXPECT_THAT(Regions[0]->insts(), testing::UnorderedElementsAre(T0, T1));
  EXPECT_THAT(Regions[1]->insts(),
              testing::UnorderedElementsAre(T2, T3, T4, T5));
  // Check aux.
  EXPECT_THAT(Regions[0]->getAux(), testing::ElementsAre(T0, T1));
  EXPECT_THAT(Regions[1]->getAux(), testing::ElementsAre(T3, T5, T4));
}

// Check that Aux is well-formed.
TEST_F(RegionTest, AuxVerify) {
  parseIR(C, R"IR(
define void @foo(i8 %v) {
  %t0 = add i8 %v, 0, !sandboxvec !0, !sbaux !2
  %t1 = add i8 %v, 1, !sandboxvec !0, !sbaux !3
  ret void
}

!0 = distinct !{!"sandboxregion"}
!2 = !{i32 0}
!3 = !{i32 2}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
#ifndef NDEBUG
  EXPECT_DEATH(sandboxir::Region::createRegionsFromMD(*F, *TTI), ".*Gap*");
#endif
}

TEST_F(RegionTest, AuxRoundTrip) {
  parseIR(C, R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1
  ret i8 %t1
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *T0 = cast<sandboxir::Instruction>(&*It++);
  auto *T1 = cast<sandboxir::Instruction>(&*It++);

  sandboxir::Region Rgn(Ctx, *TTI);
  Rgn.add(T0);
  Rgn.add(T1);
#ifndef NDEBUG
  EXPECT_DEATH(Rgn.setAux({T0, T0}), ".*already.*");
#endif
  Rgn.setAux({T1, T0});

  SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
      sandboxir::Region::createRegionsFromMD(*F, *TTI);
  ASSERT_EQ(1U, Regions.size());
#ifndef NDEBUG
  EXPECT_EQ(Rgn, *Regions[0].get());
#endif
  EXPECT_THAT(Rgn.getAux(), testing::ElementsAre(T1, T0));
}
