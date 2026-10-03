#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(IndirectCallPromotionAnalysisTest, MaxNumValueDataOverridesMaxNumPromotions) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n"
      "target triple = \"x86_64-unknown-linux-gnu\"\n"
      "define i32 @foo(ptr %p) {\n"
      "  %call = tail call i32 %p(), !prof !0\n"
      "  ret i32 0\n"
      "}\n"
      "!0 = !{!\"VP\", i32 0, i64 1750, i64 111, i64 1000, i64 222, i64 400, i64 333, i64 200, i64 444, i64 150}\n",
      Err, C);
  ASSERT_TRUE(M);

  Function *F = M->getFunction("foo");
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  ASSERT_TRUE(isa<CallInst>(Inst));

  ICallPromotionAnalysis ICallAnalysis;
  uint64_t TotalCount;
  uint32_t NumCandidates;

  // The default MaxNumPromotions is 3. We override MaxNumValueData to 4.
  // The VP metadata has 4 targets.
  auto Candidates = ICallAnalysis.getPromotionCandidatesForInstruction(
      Inst, TotalCount, NumCandidates, 4);

  EXPECT_EQ(TotalCount, 1750u);
  EXPECT_EQ(Candidates.size(), 4u);

  // NumCandidates should not be artificially truncated by the default
  // MaxNumPromotions (3), since it was overridden by MaxNumValueData (4).
  EXPECT_EQ(NumCandidates, 4u);
}

} // end anonymous namespace
