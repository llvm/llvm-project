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
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "gmock/gmock.h"
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

static sandboxir::BasicBlock *getBasicBlockByName(sandboxir::Function *F,
                                                  StringRef Name) {
  for (sandboxir::BasicBlock &BB : *F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST_F(LegalityTest, LegalitySkipSchedule) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x float> %vec2, <3 x float> %vec3, i8 %arg, float %farg0, float %farg1, i64 %v0, i64 %v1, i32 %v2, i1 %c0, i1 %c1) {
entry:
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  store float %farg0, ptr %gep1
  br label %bb

bb:
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
  %sel0 = select i1 %c0, <2 x float> %vec2, <2 x float> %vec2
  %sel1 = select i1 %c1, <2 x float> %vec2, <2 x float> %vec2
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  getAnalyses(*LLVMF);
  const auto &DL = M->getDataLayout();

  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *EntryBB = getBasicBlockByName(F, "entry");
  auto It = EntryBB->begin();
  [[maybe_unused]] auto *Gep0 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep1 = cast<sandboxir::GetElementPtrInst>(&*It++);
  auto *St1Entry = cast<sandboxir::StoreInst>(&*It++);

  auto *BB = getBasicBlockByName(F, "bb");
  It = BB->begin();
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
  auto *Sel0 = cast<sandboxir::SelectInst>(&*It++);
  auto *Sel1 = cast<sandboxir::SelectInst>(&*It++);

  llvm::sandboxir::InstrMaps IMaps(Ctx);
  sandboxir::LegalityAnalysis Legality(*AA, *SE, DL, Ctx, IMaps);
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
    // Check DiffBBs
    const auto &Result =
        Legality.canVectorize({St0, St1Entry}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffBBs);
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
  {
    // Check Repeated instructions (splat)
    const auto &Result =
        Legality.canVectorize({Ld0, Ld0}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::RepeatedInstrs);
  }
  {
    // Check Repeated instructions (not splat)
    const auto &Result =
        Legality.canVectorize({Ld0, Ld1, Ld0}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::RepeatedInstrs);
  }
  {
    // For now don't vectorize Selects when the number of elements of conditions
    // doesn't match the operands.
    const auto &Result =
        Legality.canVectorize({Sel0, Sel1}, /*SkipScheduling=*/true);
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::Unimplemented);
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

  llvm::sandboxir::InstrMaps IMaps(Ctx);
  sandboxir::LegalityAnalysis Legality(*AA, *SE, DL, Ctx, IMaps);
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
  llvm::sandboxir::InstrMaps IMaps(Ctx);
  sandboxir::LegalityAnalysis Legality(*AA, *SE, DL, Ctx, IMaps);
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

TEST_F(LegalityTest, CollectDescr) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %vld = load <4 x float>, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  getAnalyses(*LLVMF);
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  [[maybe_unused]] auto *Gep0 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep1 = cast<sandboxir::GetElementPtrInst>(&*It++);
  auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *Ld1 = cast<sandboxir::LoadInst>(&*It++);
  auto *VLd = cast<sandboxir::LoadInst>(&*It++);

  sandboxir::CollectDescr::DescrVecT Descrs;
  using EEDescr = sandboxir::CollectDescr::ExtractElementDescr;

  {
    // Check single input, no shuffle.
    Descrs.push_back(EEDescr(VLd, 0));
    Descrs.push_back(EEDescr(VLd, 1));
    sandboxir::CollectDescr CD(std::move(Descrs));
    EXPECT_TRUE(CD.getSingleInput());
    EXPECT_EQ(CD.getSingleInput()->first, VLd);
    EXPECT_THAT(CD.getSingleInput()->second, testing::ElementsAre(0, 1));
    EXPECT_TRUE(CD.hasVectorInputs());
  }
  {
    // Check single input, shuffle.
    Descrs.push_back(EEDescr(VLd, 1));
    Descrs.push_back(EEDescr(VLd, 0));
    sandboxir::CollectDescr CD(std::move(Descrs));
    EXPECT_TRUE(CD.getSingleInput());
    EXPECT_EQ(CD.getSingleInput()->first, VLd);
    EXPECT_THAT(CD.getSingleInput()->second, testing::ElementsAre(1, 0));
    EXPECT_TRUE(CD.hasVectorInputs());
  }
  {
    // Check multiple inputs.
    Descrs.push_back(EEDescr(Ld0));
    Descrs.push_back(EEDescr(VLd, 0));
    Descrs.push_back(EEDescr(VLd, 1));
    sandboxir::CollectDescr CD(std::move(Descrs));
    EXPECT_FALSE(CD.getSingleInput());
    EXPECT_TRUE(CD.hasVectorInputs());
  }
  {
    // Check multiple inputs only scalars.
    Descrs.push_back(EEDescr(Ld0));
    Descrs.push_back(EEDescr(Ld1));
    sandboxir::CollectDescr CD(std::move(Descrs));
    EXPECT_FALSE(CD.getSingleInput());
    EXPECT_FALSE(CD.hasVectorInputs());
  }
}

TEST_F(LegalityTest, ShuffleMask) {
  {
    // Check SmallVector constructor.
    SmallVector<int> Indices({0, 1, 2, 3});
    sandboxir::ShuffleMask Mask(std::move(Indices));
    EXPECT_THAT(Mask, testing::ElementsAre(0, 1, 2, 3));
  }
  {
    // Check initializer_list constructor.
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    EXPECT_THAT(Mask, testing::ElementsAre(0, 1, 2, 3));
  }
  {
    // Check ArrayRef constructor.
    sandboxir::ShuffleMask Mask(ArrayRef<int>({0, 1, 2, 3}));
    EXPECT_THAT(Mask, testing::ElementsAre(0, 1, 2, 3));
  }
  {
    // Check operator ArrayRef<int>().
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    ArrayRef<int> Array = Mask;
    EXPECT_THAT(Array, testing::ElementsAre(0, 1, 2, 3));
  }
  {
    // Check getIdentity().
    auto IdentityMask = sandboxir::ShuffleMask::getIdentity(4);
    EXPECT_THAT(IdentityMask, testing::ElementsAre(0, 1, 2, 3));
    EXPECT_TRUE(IdentityMask.isIdentity());
  }
  {
    // Check isIdentity().
    sandboxir::ShuffleMask Mask1({0, 1, 2, 3});
    EXPECT_TRUE(Mask1.isIdentity());
    sandboxir::ShuffleMask Mask2({1, 2, 3, 4});
    EXPECT_FALSE(Mask2.isIdentity());
  }
  {
    // Check operator==().
    sandboxir::ShuffleMask Mask1({0, 1, 2, 3});
    sandboxir::ShuffleMask Mask2({0, 1, 2, 3});
    EXPECT_TRUE(Mask1 == Mask2);
    EXPECT_FALSE(Mask1 != Mask2);
  }
  {
    // Check operator!=().
    sandboxir::ShuffleMask Mask1({0, 1, 2, 3});
    sandboxir::ShuffleMask Mask2({0, 1, 2, 4});
    EXPECT_TRUE(Mask1 != Mask2);
    EXPECT_FALSE(Mask1 == Mask2);
  }
  {
    // Check size().
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    EXPECT_EQ(Mask.size(), 4u);
  }
  {
    // Check operator[].
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    for (auto [Idx, Elm] : enumerate(Mask)) {
      EXPECT_EQ(Elm, Mask[Idx]);
    }
  }
  {
    // Check begin(), end().
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    sandboxir::ShuffleMask::const_iterator Begin = Mask.begin();
    sandboxir::ShuffleMask::const_iterator End = Mask.begin();
    int Idx = 0;
    for (auto It = Begin; It != End; ++It) {
      EXPECT_EQ(*It, Mask[Idx++]);
    }
  }
#ifndef NDEBUG
  {
    // Check print(OS).
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    std::string Str;
    raw_string_ostream OS(Str);
    Mask.print(OS);
    EXPECT_EQ(Str, "0,1,2,3");
  }
  {
    // Check operator<<().
    sandboxir::ShuffleMask Mask({0, 1, 2, 3});
    std::string Str;
    raw_string_ostream OS(Str);
    OS << Mask;
    EXPECT_EQ(Str, "0,1,2,3");
  }
#endif // NDEBUG
}
