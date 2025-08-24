//===- llvm/unittest/CodeGen/GCMetadata.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<Module> parseIR(LLVMContext &Context, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, Context);
}

class GCMetadataTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

public:
  GCMetadataTest()
      : M(parseIR(Context, R"(
%Env = type ptr

define void @.main(%Env) gc "shadow-stack" {
	%Root = alloca %Env
	call void @llvm.gcroot( ptr %Root, %Env null )
	unreachable
}

define void @g() gc "erlang" {
entry:
	ret void
}

declare void @llvm.gcroot(ptr, %Env)
)")) {}
};

TEST_F(GCMetadataTest, Basic) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  FunctionPassManager FPM;
  GCStrategyMap &StrategyMap = MAM.getResult<CollectorMetadataAnalysis>(*M);
  for (auto &[GCName, Strategy] : StrategyMap)
    EXPECT_EQ(GCName, Strategy->getName());
  for (auto &[GCName, Strategy] : llvm::reverse(StrategyMap))
    EXPECT_EQ(GCName, Strategy->getName());
}

} // namespace
