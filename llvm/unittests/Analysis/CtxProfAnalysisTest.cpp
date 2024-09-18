//===--- CtxProfAnalysisTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class CtxProfAnalysisTest : public testing::Test {
  static constexpr auto *IR = R"IR(
declare void @bar()

define private void @foo(i32 %a, ptr %fct) #0 !guid !0 {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no
yes:
  call void %fct(i32 %a)
  br label %exit
no:
  call void @bar()
  br label %exit
exit:
  ret void
}

define void @an_entrypoint(i32 %a) {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no

yes:
  call void @foo(i32 1, ptr null)
  ret void
no:
  ret void
}

define void @another_entrypoint_no_callees(i32 %a) {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no

yes:
  ret void
no:
  ret void
}

define void @inlineasm() {
  call void asm "nop", ""()
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64 11872291593386833696 }
)IR";

protected:
  LLVMContext C;
  PassBuilder PB;
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  LoopAnalysisManager LAM;
  std::unique_ptr<Module> M;

  void SetUp() override {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    ASSERT_TRUE(!!M);
  }

public:
  CtxProfAnalysisTest() {
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

TEST_F(CtxProfAnalysisTest, GetCallsiteIDTest) {
  ModulePassManager MPM;
  MPM.addPass(PGOInstrumentationGen(PGOInstrumentationType::CTXPROF));
  EXPECT_FALSE(MPM.run(*M, MAM).areAllPreserved());
  auto *F = M->getFunction("foo");
  ASSERT_NE(F, nullptr);
  std::vector<uint32_t> InsValues;

  for (auto &BB : *F)
    for (auto &I : BB)
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        // Skip instrumentation inserted intrinsics.
        if (CB->getCalledFunction() && CB->getCalledFunction()->isIntrinsic())
          continue;
        auto *Ins = CtxProfAnalysis::getCallsiteInstrumentation(*CB);
        ASSERT_NE(Ins, nullptr);
        InsValues.push_back(Ins->getIndex()->getZExtValue());
      }

  EXPECT_THAT(InsValues, testing::ElementsAre(0, 1));
}

TEST_F(CtxProfAnalysisTest, GetCallsiteIDInlineAsmTest) {
  ModulePassManager MPM;
  MPM.addPass(PGOInstrumentationGen(PGOInstrumentationType::CTXPROF));
  EXPECT_FALSE(MPM.run(*M, MAM).areAllPreserved());
  auto *F = M->getFunction("inlineasm");
  ASSERT_NE(F, nullptr);
  std::vector<const Instruction *> InsValues;

  for (auto &BB : *F)
    for (auto &I : BB)
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        // Skip instrumentation inserted intrinsics.
        if (CB->getCalledFunction() && CB->getCalledFunction()->isIntrinsic())
          continue;
        auto *Ins = CtxProfAnalysis::getCallsiteInstrumentation(*CB);
        InsValues.push_back(Ins);
      }

  EXPECT_THAT(InsValues, testing::ElementsAre(nullptr));
}

TEST_F(CtxProfAnalysisTest, GetCallsiteIDNegativeTest) {
  auto *F = M->getFunction("foo");
  ASSERT_NE(F, nullptr);
  CallBase *FirstCall = nullptr;
  for (auto &BB : *F)
    for (auto &I : BB)
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->isIndirectCall() || !CB->getCalledFunction()->isIntrinsic()) {
          FirstCall = CB;
          break;
        }
      }
  ASSERT_NE(FirstCall, nullptr);
  auto *IndIns = CtxProfAnalysis::getCallsiteInstrumentation(*FirstCall);
  EXPECT_EQ(IndIns, nullptr);
}

TEST_F(CtxProfAnalysisTest, GetBBIDTest) {
  ModulePassManager MPM;
  MPM.addPass(PGOInstrumentationGen(PGOInstrumentationType::CTXPROF));
  EXPECT_FALSE(MPM.run(*M, MAM).areAllPreserved());
  auto *F = M->getFunction("foo");
  ASSERT_NE(F, nullptr);
  std::map<std::string, int> BBNameAndID;

  for (auto &BB : *F) {
    auto *Ins = CtxProfAnalysis::getBBInstrumentation(BB);
    if (Ins)
      BBNameAndID[BB.getName().str()] =
          static_cast<int>(Ins->getIndex()->getZExtValue());
    else
      BBNameAndID[BB.getName().str()] = -1;
  }

  EXPECT_THAT(BBNameAndID,
              testing::UnorderedElementsAre(
                  testing::Pair("", 0), testing::Pair("yes", 1),
                  testing::Pair("no", -1), testing::Pair("exit", -1)));
}
} // namespace
