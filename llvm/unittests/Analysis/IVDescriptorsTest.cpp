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
        EXPECT_FALSE(IndDesc.hasUnsafeAlgebra());
        EXPECT_EQ(IndDesc.getUnsafeAlgebraInst(), nullptr);
      });
}
