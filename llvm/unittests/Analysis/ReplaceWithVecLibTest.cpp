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

/// NOTE: Assertions must be enabled for these tests to run.
#ifndef NDEBUG

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
///
class ReplaceWithVecLibTest : public ::testing::Test {

  std::string getLastLine(std::string Out) {
    // remove any trailing '\n'
    if (!Out.empty() && *(Out.cend() - 1) == '\n')
      Out.pop_back();

    size_t LastNL = Out.find_last_of('\n');
    return (LastNL == std::string::npos) ? Out : Out.substr(LastNL + 1);
  }

protected:
  LLVMContext Ctx;

  /// Creates TLII using the given \p VD, and then runs the ReplaceWithVeclib
  /// pass. The pass should not crash even when the replacement function
  /// (derived from the \p VD mapping) does not match the function to be
  /// replaced (from the input \p IR).
  ///
  /// \returns the last line of the standard error to be compared for
  /// correctness.
  std::string run(const VecDesc &VD, const char *IR) {
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

    // Enable debugging and capture std error
    bool DebugFlagPrev = llvm::DebugFlag;
    llvm::DebugFlag = true;
    testing::internal::CaptureStderr();
    FPM.run(*M->getFunction("foo"), FAM);
    llvm::DebugFlag = DebugFlagPrev;
    return getLastLine(testing::internal::GetCapturedStderr());
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

// The VFABI prefix in TLI describes signature which is matching the powi
// intrinsic declaration.
TEST_F(ReplaceWithVecLibTest, TestValidMapping) {
  VecDesc CorrectVD = {"llvm.powi.f32.i32", "_ZGVsMxvu_powi",
                       ElementCount::getScalable(4), /*Masked*/ true,
                       "_ZGVsMxvu"};
  EXPECT_EQ(run(CorrectVD, IR),
            "Instructions replaced with vector libraries: 1");
}

// The VFABI prefix in TLI describes signature which is not matching the powi
// intrinsic declaration.
TEST_F(ReplaceWithVecLibTest, TestInvalidMapping) {
  VecDesc IncorrectVD = {"llvm.powi.f32.i32", "_ZGVsMxvv_powi",
                         ElementCount::getScalable(4), /*Masked*/ true,
                         "_ZGVsMxvv"};
  EXPECT_EQ(run(IncorrectVD, IR),
            "replace-with-veclib: Will not replace: llvm.powi.f32.i32. Wrong "
            "type at index 1: i32");
}
#endif