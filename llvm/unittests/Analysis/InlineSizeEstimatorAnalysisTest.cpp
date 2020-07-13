//===- InlineSizeEstimatorAnalysisTest.cpp - test for ir2native -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InlineSizeEstimatorAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;

extern const char *TestMainArgv0;
extern cl::opt<std::string> TFIR2NativeModelPath;

#if LLVM_HAVE_TF_API
static std::string getModelPath() {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "ir2native_x86_64_model");
  return std::string(InputsDir);
}
#endif

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("MLAnalysisTests", errs());
  return Mod;
}

static FunctionAnalysisManager buildFAM() {
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  return FAM;
}

// Test model loading and evaluation.
TEST(InlineSizeEstimatorAnalysis, SizeIsValidTest) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i32 @f1(i32)
declare i32 @f2(i32)

define i32 @branches(i32) {
  %cond = icmp slt i32 %0, 3
  br i1 %cond, label %then, label %else

then:
  %ret.1 = call i32 @f1(i32 %0)
  br label %last.block

else:
  %ret.2 = call i32 @f2(i32 %0)
  br label %last.block

last.block:
  %ret = phi i32 [%ret.1, %then], [%ret.2, %else]
  ret i32 %ret
}

define internal i32 @top() {
  %1 = call i32 @branches(i32 2)
  %2 = call i32 @f1(i32 %1)
  ret i32 %2
}
)IR");

  FunctionAnalysisManager FAM = buildFAM();
#if LLVM_HAVE_TF_API
  TFIR2NativeModelPath = getModelPath();
#endif

  InlineSizeEstimatorAnalysis FA;
  auto SizeEstimate = FA.run(*M->getFunction("branches"), FAM);
#if LLVM_HAVE_TF_API
  EXPECT_GT(*SizeEstimate, 0);
#else
  EXPECT_FALSE(SizeEstimate.hasValue());
#endif
}
