//===- IVDescriptorsTest.cpp - IVDescriptors unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Build the loop info and scalar evolution for the function and run the Test.
static void runWithLoopInfoAndSE(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, LoopInfo &LI, ScalarEvolution &SE)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);

  Test(*F, LI, SE);
}

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("IVDescriptorsTests", errs());
  return Mod;
}

// This tests that IVDescriptors can obtain the induction binary operator for
// integer induction variables. And hasUnsafeAlgebra() and
// getUnsafeAlgebraInst() correctly return the expected behavior, i.e. no unsafe
// algebra.
TEST(IVDescriptorsTest, LoopWithSingleLatch) {
  // Parse the module.
  LLVMContext Context;

  std::unique_ptr<Module> M = parseIR(
    Context,
    R"(define void @foo(i32* %A, i32 %ub) {
entry:
  br label %for.body
for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  store i32 %i, i32* %arrayidx, align 4
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %ub
  br i1 %cmp, label %for.body, label %for.exit
for.exit:
  br label %for.end
for.end:
  ret void
})"
    );

  runWithLoopInfoAndSE(
      *M, "foo", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        // First basic block is entry - skip it.
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.body");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);
        PHINode *Inst_i = dyn_cast<PHINode>(&Header->front());
        assert(Inst_i->getName() == "i");
        InductionDescriptor IndDesc;
        bool IsInductionPHI =
            InductionDescriptor::isInductionPHI(Inst_i, L, &SE, IndDesc);
        EXPECT_TRUE(IsInductionPHI);
        Instruction *Inst_inc = nullptr;
        BasicBlock::iterator BBI = Header->begin();
        do {
          if ((&*BBI)->getName() == "inc")
            Inst_inc = &*BBI;
          ++BBI;
        } while (!Inst_inc);
        assert(Inst_inc->getName() == "inc");
        EXPECT_EQ(IndDesc.getInductionBinOp(), Inst_inc);
        EXPECT_EQ(IndDesc.getExactFPMathInst(), nullptr);
      });
}

// Depending on how SCEV deals with ptrtoint cast, the step of a phi could be
// a pointer, and InductionDescriptor used to fail with an assertion.
// So just check that it doesn't assert.
TEST(IVDescriptorsTest, LoopWithPtrToInt) {
  // Parse the module.
  LLVMContext Context;

  std::unique_ptr<Module> M = parseIR(Context, R"(
      target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
      target triple = "thumbv6m-arm-none-eabi"

      declare void @widget()
      declare void @wobble(i32)

      define void @barney(i8* %arg, i8* %arg18, i32 %arg19) {
      bb:
        %tmp = ptrtoint i8* %arg to i32
        %tmp20 = ptrtoint i8* %arg18 to i32
        %tmp21 = or i32 %tmp20, %tmp
        %tmp22 = and i32 %tmp21, 3
        %tmp23 = icmp eq i32 %tmp22, 0
        br i1 %tmp23, label %bb24, label %bb25

      bb24:
        tail call void @widget()
        br label %bb34

      bb25:
        %tmp26 = sub i32 %tmp, %tmp20
        %tmp27 = icmp ult i32 %tmp26, %arg19
        br i1 %tmp27, label %bb28, label %bb34

      bb28:
        br label %bb29

      bb29:
        %tmp30 = phi i32 [ %tmp31, %bb29 ], [ %arg19, %bb28 ]
        tail call void @wobble(i32 %tmp26)
        %tmp31 = sub i32 %tmp30, %tmp26
        %tmp32 = icmp ugt i32 %tmp31, %tmp26
        br i1 %tmp32, label %bb29, label %bb33

      bb33:
        br label %bb34

      bb34:
        ret void
      })");

  runWithLoopInfoAndSE(
      *M, "barney", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        Function::iterator FI = F.begin();
        // First basic block is entry - skip it.
        BasicBlock *Header = &*(++(++(++(++FI))));
        assert(Header->getName() == "bb29");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);
        PHINode *Inst_i = dyn_cast<PHINode>(&Header->front());
        assert(Inst_i->getName() == "tmp30");
        InductionDescriptor IndDesc;
        bool IsInductionPHI =
            InductionDescriptor::isInductionPHI(Inst_i, L, &SE, IndDesc);
        EXPECT_TRUE(IsInductionPHI);
      });
}
