//===--- LastRunTrackingAnalysisTest.cpp - LastRunTrackingAnalysis tests---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LastRunTrackingAnalysis.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;

class LastRunTrackingAnalysisTest : public testing::Test {
protected:
  LLVMContext C;
  Module M;
  PassBuilder PB;

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;

  LastRunTrackingAnalysisTest() : M("LastRunTrackingAnalysisTest", C) {
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

struct PassOption final {
  uint32_t Threshold;

  /// Assume that this pass doesn't make changes with threshold A if we already
  /// know it doesn't make changes with a larger threshold B.
  bool isCompatibleWith(const PassOption &LastOpt) const {
    return Threshold <= LastOpt.Threshold;
  }
};

class ModuleNoopPass : public PassInfoMixin<ModuleNoopPass> {
  uint32_t &ExecutedBitMap;
  uint32_t RunID;
  void *PassID;
  bool ShouldChange;
  std::optional<PassOption> Option;

  bool shouldSkip(LastRunTrackingInfo &LRT) {
    if (Option.has_value())
      return LRT.shouldSkip(PassID, *Option);
    return LRT.shouldSkip(PassID);
  }

  void update(LastRunTrackingInfo &LRT) {
    if (Option.has_value())
      return LRT.update(PassID, ShouldChange, *Option);
    return LRT.update(PassID, ShouldChange);
  }

public:
  explicit ModuleNoopPass(uint32_t &ExecutedBitMapRef, uint32_t RunIDVal,
                          void *PassIDVal, bool ShouldChangeVal,
                          std::optional<PassOption> OptionVal = std::nullopt)
      : ExecutedBitMap(ExecutedBitMapRef), RunID(RunIDVal), PassID(PassIDVal),
        ShouldChange(ShouldChangeVal), Option(OptionVal) {}

  PreservedAnalyses run(Module &F, ModuleAnalysisManager &AM) {
    auto &LRT = AM.getResult<LastRunTrackingAnalysis>(F);
    if (shouldSkip(LRT)) {
      EXPECT_FALSE(ShouldChange) << "This pass is incorrectly skipped.";
      return PreservedAnalyses::all();
    }
    ExecutedBitMap |= 1U << RunID;
    update(LRT);
    PreservedAnalyses PA;
    PA.preserve<LastRunTrackingAnalysis>();
    return PA;
  }
};

static char PassA, PassB;

TEST_F(LastRunTrackingAnalysisTest, SkipTest) {
  uint32_t BitMap = 0;
  // Executed. This is first run of PassA.
  MPM.addPass(ModuleNoopPass(BitMap, 0, &PassA, true));
  // Skipped since PassA has just been executed.
  MPM.addPass(ModuleNoopPass(BitMap, 1, &PassA, false));
  // Skipped since PassA has just been executed.
  MPM.addPass(ModuleNoopPass(BitMap, 2, &PassA, false));
  // Executed. This is first run of PassB.
  MPM.addPass(ModuleNoopPass(BitMap, 3, &PassB, false, PassOption{2}));
  // Skipped. PassB doesn't make changes with lower threshold.
  MPM.addPass(ModuleNoopPass(BitMap, 4, &PassB, false, PassOption{1}));
  // Executed. PassB may make changes with higher threshold.
  MPM.addPass(ModuleNoopPass(BitMap, 5, &PassB, false, PassOption{3}));
  // Skipped. We don't make changes since last run of PassA.
  MPM.addPass(ModuleNoopPass(BitMap, 6, &PassA, false));
  // Executed. PassB may make changes with higher threshold.
  MPM.addPass(ModuleNoopPass(BitMap, 7, &PassB, true, PassOption{4}));
  // Executed. This module has been modified by PassB.
  MPM.addPass(ModuleNoopPass(BitMap, 8, &PassA, false));
  MPM.run(M, MAM);

  ASSERT_EQ(BitMap, 0b110101001U);
}

} // namespace
