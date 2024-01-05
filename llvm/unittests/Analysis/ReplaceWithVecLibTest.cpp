//===--- ReplaceWithVecLibTest.cpp - replace-with-veclib unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("ReplaceWithVecLibTest", errs());
  return Mod;
}

/// Runs ReplaceWithVecLib with different TLIIs that have custom VecDescs. This
/// allows checking that the pass won't crash when the function to replace (from
/// the input IR) does not match the replacement function (derived from the
/// VecDesc mapping).
class ReplaceWithVecLibTest : public ::testing::Test {
protected:
  LLVMContext Ctx;

  /// Creates TLII using the given \p VD, and then runs the ReplaceWithVeclib
  /// pass. The pass should not crash even when the replacement function
  /// (derived from the \p VD mapping) does not match the function to be
  /// replaced (from the input \p IR).
  bool run(const VecDesc &VD, const char *IR) {
    // Create TLII and register it with FAM so it's preserved when
    // ReplaceWithVeclib pass runs.
    TargetLibraryInfoImpl TLII = TargetLibraryInfoImpl(Triple());
    TLII.addVectorizableFunctions({VD});
    FunctionAnalysisManager FAM;
    FAM.registerPass([&TLII]() { return TargetLibraryAnalysis(TLII); });

    // Register and run the pass on the 'foo' function from the input IR.
    FunctionPassManager FPM;
    FPM.addPass(ReplaceWithVeclib());
    std::unique_ptr<Module> M = parseIR(Ctx, IR);
    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    FPM.run(*M->getFunction("foo"), FAM);

    return true;
  }
};

} // end anonymous namespace

static const char *IR = R"IR(
define <vscale x 4 x float> @foo(<vscale x 4 x float> %in){
  %call = call <vscale x 4 x float> @llvm.powi.f32.i32(<vscale x 4 x float> %in, i32 3)
  ret <vscale x 4 x float> %call
}

declare <vscale x 4 x float> @llvm.powi.f32.i32(<vscale x 4 x float>, i32) #0
)IR";

// LLVM intrinsic 'powi' (in IR) has the same signature with the VecDesc.
TEST_F(ReplaceWithVecLibTest, TestValidMapping) {
  VecDesc CorrectVD = {"llvm.powi.f32.i32", "_ZGVsMxvu_powi",
                       ElementCount::getScalable(4), true, "_ZGVsMxvu"};
  EXPECT_TRUE(run(CorrectVD, IR));
}

// LLVM intrinsic 'powi' (in IR) has different signature with the VecDesc.
TEST_F(ReplaceWithVecLibTest, TestInvalidMapping) {
  VecDesc IncorrectVD = {"llvm.powi.f32.i32", "_ZGVsMxvv_powi",
                         ElementCount::getScalable(4), true, "_ZGVsMxvv"};
  /// TODO: test should avoid and not crash.
  EXPECT_DEATH(run(IncorrectVD, IR), "");
}
