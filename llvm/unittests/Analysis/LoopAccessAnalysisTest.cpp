//===- LoopAccessAnalysisTest.cpp - LoopAccessAnalysis unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

class LoopAccessAnalysisTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

  void parseIR(StringRef Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);
    if (!M) {
      std::string Msg;
      raw_string_ostream OS(Msg);
      Error.print("", OS);
      report_fatal_error(Twine(Msg));
    }
  }
};

// A simple test to exercise the ExpensivePtrCheck argument. It should return
// the same value for simple pointer expressions like in this test case.
TEST_F(LoopAccessAnalysisTest, GetPointersDiff_ExpensivePtrCheck) {
  parseIR(R"(
    define void @foo(ptr %ptr0, ptr %ptr1) {
      %gep0 = getelementptr i8, ptr %ptr0, i64 0
      %gep1 = getelementptr i8, ptr %ptr0, i64 42
      ret void
    }
  )");

  Function &F = *M->getFunction("foo");
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(F);
  DominatorTree DT(F);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  const DataLayout &DL = M->getDataLayout();

  BasicBlock &BB = *F.begin();
  auto It = BB.begin();
  Instruction *PtrA = &*It++;
  Instruction *PtrB = &*It++;
  Type *ElemTy = IntegerType::getInt8Ty(M->getContext());
  auto Diff_F = getPointersDiff(ElemTy, PtrA, ElemTy, PtrB, DL, SE,
                                /*StrictCheck=*/false, /*CheckType=*/true,
                                /*ExpensivePtrCheck=*/false);
  auto Diff_T = getPointersDiff(ElemTy, PtrA, ElemTy, PtrB, DL, SE,
                                /*StrictCheck=*/false, /*CheckType=*/true,
                                /*ExpensivePtrCheck=*/true);
  EXPECT_EQ(Diff_F, Diff_T);
  EXPECT_EQ(Diff_F, 42);

  auto *Ptr0 = F.getArg(0);
  auto *Ptr1 = F.getArg(1);
  Diff_F = getPointersDiff(ElemTy, Ptr0, ElemTy, Ptr1, DL, SE,
                           /*StrictCheck=*/false, /*CheckType=*/true,
                           /*ExpensivePtrCheck=*/false);
  Diff_T = getPointersDiff(ElemTy, Ptr0, ElemTy, Ptr1, DL, SE,
                           /*StrictCheck=*/false, /*CheckType=*/true,
                           /*ExpensivePtrCheck=*/true);
  EXPECT_EQ(Diff_F, Diff_T);
  EXPECT_EQ(Diff_F, std::nullopt);
}
