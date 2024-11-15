//===- LegalityTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct LegalityTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<TargetLibraryInfoImpl> TLII;
  std::unique_ptr<TargetLibraryInfo> TLI;
  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<LoopInfo> LI;
  std::unique_ptr<ScalarEvolution> SE;
  std::unique_ptr<BasicAAResult> BAA;
  std::unique_ptr<AAResults> AA;

  void getAnalyses(llvm::Function &LLVMF) {
    DT = std::make_unique<DominatorTree>(LLVMF);
    TLII = std::make_unique<TargetLibraryInfoImpl>();
    TLI = std::make_unique<TargetLibraryInfo>(*TLII);
    AC = std::make_unique<AssumptionCache>(LLVMF);
    LI = std::make_unique<LoopInfo>(*DT);
    SE = std::make_unique<ScalarEvolution>(LLVMF, *TLI, *AC, *DT, *LI);
    BAA = std::make_unique<BasicAAResult>(LLVMF.getParent()->getDataLayout(),
                                          LLVMF, *TLI, *AC, DT.get());
    AA = std::make_unique<AAResults>(*TLI);
    AA->addAAResult(*BAA);
  }

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("LegalityTest", errs());
  }
};

TEST_F(LegalityTest, LegalitySkipSchedule) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x float> %vec2, <3 x float> %vec3, i8 %arg, float %farg0, float %farg1, i64 %v0, i64 %v1, i32 %v2) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %gep3 = getelementptr float, ptr %ptr, i32 3
  %ld0 = load float, ptr %gep0
  %ld0b = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld3 = load float, ptr %gep3
  store float %ld0, ptr %gep0
  store float %ld1, ptr %gep1
  store <2 x float> %vec2, ptr %gep1
  store <3 x float> %vec3, ptr %gep3
  store i8 %arg, ptr %gep1
  %fadd0 = fadd float %farg0, %farg0
  %fadd1 = fadd fast float %farg1, %farg1
  %trunc0 = trunc nuw nsw i64 %v0 to i8
  %trunc1 = trunc nsw i64 %v1 to i8
  %trunc64to8 = trunc i64 %v0 to i8
  %trunc32to8 = trunc i32 %v2 to i8
  %cmpSLT = icmp slt i64 %v0, %v1
  %cmpSGT = icmp sgt i64 %v0, %v1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  getAnalyses(*LLVMF);
  const auto &DL = M->getDataLayout();

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  [[maybe_unused]] auto *Gep0 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep1 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep3 = cast<sandboxir::GetElementPtrInst>(&*It++);
  auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  auto *Ld0b = cast<sandboxir::LoadInst>(&*It++);
  auto *Ld1 = cast<sandboxir::LoadInst>(&*It++);
  auto *Ld3 = cast<sandboxir::LoadInst>(&*It++);
  auto *St0 = cast<sandboxir::StoreInst>(&*It++);
  auto *St1 = cast<sandboxir::StoreInst>(&*It++);
  auto *StVec2 = cast<sandboxir::StoreInst>(&*It++);
  auto *StVec3 = cast<sandboxir::StoreInst>(&*It++);
  auto *StI8 = cast<sandboxir::StoreInst>(&*It++);
  auto *FAdd0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *FAdd1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Trunc0 = cast<sandboxir::TruncInst>(&*It++);
  auto *Trunc1 = cast<sandboxir::TruncInst>(&*It++);
  auto *Trunc64to8 = cast<sandboxir::TruncInst>(&*It++);
  auto *Trunc32to8 = cast<sandboxir::TruncInst>(&*It++);
  auto *CmpSLT = cast<sandboxir::CmpInst>(&*It++);
  auto *CmpSGT = cast<sandboxir::CmpInst>(&*It++);

  sandboxir::LegalityAnalysis Legality(*AA, *SE, DL, Ctx);
  const auto &Result =
      Legality.canVectorize({St0, St1}, /*SkipScheduling=*/true);
  EXPECT_TRUE(isa<sandboxir::Widen>(Result));

  {
    // Check NotInstructions
    auto &Result = Legality.canVectorize({F, St0}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::NotInstructions);
  }
  {
    // Check DiffOpcodes
    const auto &Result =
        Legality.canVectorize({St0, Ld0}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffOpcodes);
  }
  {
    // Check DiffTypes
    EXPECT_TRUE(isa<sandboxir::Widen>(
        Legality.canVectorize({St0, StVec2}, /*SkipScheduling=*/true)));
    EXPECT_TRUE(isa<sandboxir::Widen>(
        Legality.canVectorize({StVec2, StVec3}, /*SkipScheduling=*/true)));

    const auto &Result =
        Legality.canVectorize({St0, StI8}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffTypes);
  }
  {
    // Check DiffMathFlags
    const auto &Result =
        Legality.canVectorize({FAdd0, FAdd1}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffMathFlags);
  }
  {
    // Check DiffWrapFlags
    const auto &Result =
        Legality.canVectorize({Trunc0, Trunc1}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffWrapFlags);
  }
  {
    // Check DiffTypes for unary operands that have a different type.
    const auto &Result = Legality.canVectorize({Trunc64to8, Trunc32to8},
                                               /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffTypes);
  }
  {
    // Check DiffOpcodes for CMPs with different predicates.
    const auto &Result =
        Legality.canVectorize({CmpSLT, CmpSGT}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffOpcodes);
  }
  {
    // Check NotConsecutive Ld0,Ld0b
    const auto &Result =
        Legality.canVectorize({Ld0, Ld0b}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::NotConsecutive);
  }
  {
    // Check NotConsecutive Ld0,Ld3
    const auto &Result =
        Legality.canVectorize({Ld0, Ld3}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::NotConsecutive);
  }
  {
    // Check Widen Ld0,Ld1
    const auto &Result =
        Legality.canVectorize({Ld0, Ld1}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Widen>(Result));
  }
}

TEST_F(LegalityTest, LegalitySchedule) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float %ld0, ptr %gep1
  %ld1 = load float, ptr %gep1
  store float %ld0, ptr %gep0
  store float %ld1, ptr %gep1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  getAnalyses(*LLVMF);
  const auto &DL = M->getDataLayout();

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  [[maybe_unused]] auto *Gep0 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep1 = cast<sandboxir::GetElementPtrInst>(&*It++);
  auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *ConflictingSt = cast<sandboxir::StoreInst>(&*It++);
  auto *Ld1 = cast<sandboxir::LoadInst>(&*It++);
  auto *St0 = cast<sandboxir::StoreInst>(&*It++);
  auto *St1 = cast<sandboxir::StoreInst>(&*It++);

  sandboxir::LegalityAnalysis Legality(*AA, *SE, DL, Ctx);
  {
    // Can vectorize St0,St1.
    const auto &Result = Legality.canVectorize({St0, St1});
    EXPECT_TRUE(isa<sandboxir::Widen>(Result));
  }
  {
    // Can't vectorize Ld0,Ld1 because of conflicting store.
    auto &Result = Legality.canVectorize({Ld0, Ld1});
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::CantSchedule);
  }
}

#ifndef NDEBUG
TEST_F(LegalityTest, LegalityResultDump) {
  parseIR(C, R"IR(
define void @foo() {
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  getAnalyses(*LLVMF);
  const auto &DL = M->getDataLayout();

  auto Matches = [](const sandboxir::LegalityResult &Result,
                    const std::string &ExpectedStr) -> bool {
    std::string Buff;
    raw_string_ostream OS(Buff);
    Result.print(OS);
    return Buff == ExpectedStr;
  };

  sandboxir::Context Ctx(C);
  sandboxir::LegalityAnalysis Legality(*AA, *SE, DL, Ctx);
  EXPECT_TRUE(
      Matches(Legality.createLegalityResult<sandboxir::Widen>(), "Widen"));
  EXPECT_TRUE(Matches(Legality.createLegalityResult<sandboxir::Pack>(
                          sandboxir::ResultReason::NotInstructions),
                      "Pack Reason: NotInstructions"));
  EXPECT_TRUE(Matches(Legality.createLegalityResult<sandboxir::Pack>(
                          sandboxir::ResultReason::DiffOpcodes),
                      "Pack Reason: DiffOpcodes"));
  EXPECT_TRUE(Matches(Legality.createLegalityResult<sandboxir::Pack>(
                          sandboxir::ResultReason::DiffTypes),
                      "Pack Reason: DiffTypes"));
  EXPECT_TRUE(Matches(Legality.createLegalityResult<sandboxir::Pack>(
                          sandboxir::ResultReason::DiffMathFlags),
                      "Pack Reason: DiffMathFlags"));
  EXPECT_TRUE(Matches(Legality.createLegalityResult<sandboxir::Pack>(
                          sandboxir::ResultReason::DiffWrapFlags),
                      "Pack Reason: DiffWrapFlags"));
}
#endif // NDEBUG
