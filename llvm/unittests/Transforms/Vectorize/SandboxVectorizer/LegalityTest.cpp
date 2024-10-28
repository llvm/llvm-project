//===- LegalityTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct LegalityTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("LegalityTest", errs());
  }
};

TEST_F(LegalityTest, Legality) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x float> %vec2, <3 x float> %vec3, i8 %arg, float %farg0, float %farg1) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %gep3 = getelementptr float, ptr %ptr, i32 3
  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep0
  store float %ld0, ptr %gep0
  store float %ld1, ptr %gep1
  store <2 x float> %vec2, ptr %gep1
  store <3 x float> %vec3, ptr %gep3
  store i8 %arg, ptr %gep1
  %fadd0 = fadd float %farg0, %farg0
  %fadd1 = fadd fast float %farg1, %farg1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  [[maybe_unused]] auto *Gep0 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep1 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Gep3 = cast<sandboxir::GetElementPtrInst>(&*It++);
  [[maybe_unused]] auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  [[maybe_unused]] auto *Ld1 = cast<sandboxir::LoadInst>(&*It++);
  auto *St0 = cast<sandboxir::StoreInst>(&*It++);
  auto *St1 = cast<sandboxir::StoreInst>(&*It++);
  auto *StVec2 = cast<sandboxir::StoreInst>(&*It++);
  auto *StVec3 = cast<sandboxir::StoreInst>(&*It++);
  auto *StI8 = cast<sandboxir::StoreInst>(&*It++);
  auto *FAdd0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *FAdd1 = cast<sandboxir::BinaryOperator>(&*It++);

  sandboxir::LegalityAnalysis Legality;
  const auto &Result = Legality.canVectorize({St0, St1});
  EXPECT_TRUE(isa<sandboxir::Widen>(Result));

  {
    // Check NotInstructions
    auto &Result = Legality.canVectorize({F, St0});
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::NotInstructions);
  }
  {
    // Check DiffOpcodes
    const auto &Result = Legality.canVectorize({St0, Ld0});
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffOpcodes);
  }
  {
    // Check DiffTypes
    EXPECT_TRUE(isa<sandboxir::Widen>(Legality.canVectorize({St0, StVec2})));
    EXPECT_TRUE(isa<sandboxir::Widen>(Legality.canVectorize({StVec2, StVec3})));

    const auto &Result = Legality.canVectorize({St0, StI8});
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffTypes);
  }
  {
    // Check DiffMathFlags
    const auto &Result = Legality.canVectorize({FAdd0, FAdd1});
    EXPECT_TRUE(isa<sandboxir::Pack>(Result));
    EXPECT_EQ(cast<sandboxir::Pack>(Result).getReason(),
              sandboxir::ResultReason::DiffMathFlags);
  }
}

#ifndef NDEBUG
TEST_F(LegalityTest, LegalityResultDump) {
  auto Matches = [](const sandboxir::LegalityResult &Result,
                    const std::string &ExpectedStr) -> bool {
    std::string Buff;
    raw_string_ostream OS(Buff);
    Result.print(OS);
    return Buff == ExpectedStr;
  };
  sandboxir::LegalityAnalysis Legality;
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
}
#endif // NDEBUG
