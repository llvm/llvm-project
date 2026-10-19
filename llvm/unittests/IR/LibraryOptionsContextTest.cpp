//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "gtest/gtest.h"

using namespace llvm;

// InstCombine sinks %a into its only user's block unless
// -instcombine-code-sinking=0.
const char IR[] = R"(
define i32 @f(i1 %c, i32 %x) {
entry:
  %a = add i32 %x, 1
  br i1 %c, label %t, label %e
t:
  ret i32 %a
e:
  ret i32 0
}
)";

// Runs InstCombine on IR in Ctx and returns the block that holds %a.
static std::string runInstCombine(LLVMContext &Ctx) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  if (!M)
    return Err.getMessage().str();
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
  Function &F = *M->getFunction("f");
  InstCombinePass().run(F, FAM);
  auto *A = cast<Instruction>(F.getValueSymbolTable()->lookup("a"));
  return A->getParent()->getName().str();
}

// parseCommandLineOptions sets InstCombine's options in the given context
// only, so two contexts in one process run InstCombine with different options.
TEST(LibraryOptionsContextTest, InstCombine) {
  LLVMContext C1, C2;
  const char *Args[] = {"prog", "-instcombine-code-sinking=0"};
  ASSERT_TRUE(
      cl::parseCommandLineOptions(C2.getLibraryOptions(), Args, "", &nulls()));
  EXPECT_EQ(runInstCombine(C1), "t");
  EXPECT_EQ(runInstCombine(C2), "entry");
}
