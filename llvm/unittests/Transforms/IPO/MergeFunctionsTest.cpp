//===- MergeFunctionsTest.cpp - Unit tests for MergeFunctionsPass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/MergeFunctions.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

class MergeFunctionsTest : public testing::Test {
protected:
  LLVMContext Ctx;
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;

  MergeFunctionsTest() {
    FAM.registerPass([&] { return TargetLibraryAnalysis(); });
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return PostDominatorTreeAnalysis(); });
    FAM.registerPass([&] { return LoopAnalysis(); });
    FAM.registerPass([&] { return BranchProbabilityAnalysis(); });
    FAM.registerPass([&] { return BlockFrequencyAnalysis(); });
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  }

  std::unique_ptr<Module> parseModule(StringRef IR) {
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
    EXPECT_TRUE(M);
    return M;
  }
};

TEST_F(MergeFunctionsTest, TrueOutputModuleTest) {
  std::unique_ptr<Module> M = parseModule(R"invalid(
        @.str = private unnamed_addr constant [10 x i8] c"On f: %d\0A\00", align 1
        @.str.1 = private unnamed_addr constant [13 x i8] c"On main: %d\0A\00", align 1

        define dso_local i32 @f(i32 noundef %arg) {
            entry:
                %add109 = call i32 @_slice_add10(i32 %arg)
                %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %add109)
                ret i32 %add109
        }

        declare i32 @printf(ptr noundef, ...)

        define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) {
            entry:
                %add99 = call i32 @_slice_add10(i32 %argc)
                %call = call i32 @f(i32 noundef 2)
                %sub = sub nsw i32 %call, 6
                %call10 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %add99)
                ret i32 %add99
        }

        define internal i32 @_slice_add10(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %4
        }

        define internal i32 @_slice_add10_alt(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %4
        }
    )invalid");

  // Expects true after merging _slice_add10 and _slice_add10_alt
  EXPECT_TRUE(MergeFunctionsPass::runOnModule(*M, MAM));
}

TEST_F(MergeFunctionsTest, TrueOutputFunctionsTest) {
  std::unique_ptr<Module> M = parseModule(R"invalid(
        @.str = private unnamed_addr constant [10 x i8] c"On f: %d\0A\00", align 1
        @.str.1 = private unnamed_addr constant [13 x i8] c"On main: %d\0A\00", align 1

        define dso_local i32 @f(i32 noundef %arg) {
            entry:
                %add109 = call i32 @_slice_add10(i32 %arg)
                %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %add109)
                ret i32 %add109
        }

        declare i32 @printf(ptr noundef, ...)

        define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) {
            entry:
                %add99 = call i32 @_slice_add10(i32 %argc)
                %call = call i32 @f(i32 noundef 2)
                %sub = sub nsw i32 %call, 6
                %call10 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %add99)
                ret i32 %add99
        }

        define internal i32 @_slice_add10(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %4
        }

        define internal i32 @_slice_add10_alt(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %4
        }
    )invalid");

  SetVector<Function *> FunctionsSet;
  for (Function &F : *M)
    FunctionsSet.insert(&F);

  DenseMap<Function *, Function *> MergeResult =
      MergeFunctionsPass::runOnFunctions(FunctionsSet.getArrayRef(), MAM);

  // Expects that both functions (_slice_add10 and _slice_add10_alt)
  // be mapped to the same new function
  EXPECT_TRUE(!MergeResult.empty());
  Function *NewFunction = M->getFunction("_slice_add10");
  for (auto P : MergeResult)
    if (P.second)
      EXPECT_EQ(P.second, NewFunction);
}

TEST_F(MergeFunctionsTest, FalseOutputModuleTest) {
  std::unique_ptr<Module> M = parseModule(R"invalid(
        @.str = private unnamed_addr constant [10 x i8] c"On f: %d\0A\00", align 1
        @.str.1 = private unnamed_addr constant [13 x i8] c"On main: %d\0A\00", align 1

        define dso_local i32 @f(i32 noundef %arg) {
            entry:
                %add109 = call i32 @_slice_add10(i32 %arg)
                %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %add109)
                ret i32 %add109
        }

        declare i32 @printf(ptr noundef, ...)

        define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) {
            entry:
                %add99 = call i32 @_slice_add10(i32 %argc)
                %call = call i32 @f(i32 noundef 2)
                %sub = sub nsw i32 %call, 6
                %call10 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %add99)
                ret i32 %add99
        }

        define internal i32 @_slice_add10(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %4
        }

        define internal i32 @_slice_add10_alt(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %0
        }
    )invalid");

  // Expects false after trying to merge _slice_add10 and _slice_add10_alt
  EXPECT_FALSE(MergeFunctionsPass::runOnModule(*M, MAM));
}

TEST_F(MergeFunctionsTest, FalseOutputFunctionsTest) {
  std::unique_ptr<Module> M = parseModule(R"invalid(
        @.str = private unnamed_addr constant [10 x i8] c"On f: %d\0A\00", align 1
        @.str.1 = private unnamed_addr constant [13 x i8] c"On main: %d\0A\00", align 1

        define dso_local i32 @f(i32 noundef %arg) {
            entry:
                %add109 = call i32 @_slice_add10(i32 %arg)
                %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %add109)
                ret i32 %add109
        }

        declare i32 @printf(ptr noundef, ...)

        define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) {
            entry:
                %add99 = call i32 @_slice_add10(i32 %argc)
                %call = call i32 @f(i32 noundef 2)
                %sub = sub nsw i32 %call, 6
                %call10 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %add99)
                ret i32 %add99
        }

        define internal i32 @_slice_add10(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %4
        }

        define internal i32 @_slice_add10_alt(i32 %arg) {
            sliceclone_entry:
                %0 = mul nsw i32 %arg, %arg
                %1 = mul nsw i32 %0, 2
                %2 = mul nsw i32 %1, 2
                %3 = mul nsw i32 %2, 2
                %4 = add nsw i32 %3, 2
                ret i32 %0
        }
    )invalid");

  SetVector<Function *> FunctionsSet;
  for (Function &F : *M)
    FunctionsSet.insert(&F);

  DenseMap<Function *, Function *> MergeResult =
      MergeFunctionsPass::runOnFunctions(FunctionsSet.getArrayRef(), MAM);

  // Expects empty map
  EXPECT_EQ(MergeResult.size(), 0u);
}

} // namespace
