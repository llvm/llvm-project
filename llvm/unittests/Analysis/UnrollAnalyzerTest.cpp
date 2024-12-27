//===- UnrollAnalyzerTest.cpp - UnrollAnalyzer unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

typedef SmallVector<DenseMap<Value *, Value *>, 16> SimplifiedValuesVectorTy;

/// Build loop info and scalar evolution for the function and run the analysis.
static void
runUnrollAnalyzer(Module &M, StringRef FuncName,
                  SimplifiedValuesVectorTy &SimplifiedValuesVector) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);

  Function::iterator FI = F->begin();
  FI++; // First basic block is entry - skip it.
  BasicBlock *Header = &*FI++;
  Loop *L = LI.getLoopFor(Header);
  BasicBlock *Exiting = L->getExitingBlock();

  SimplifiedValuesVector.clear();
  unsigned TripCount = SE.getSmallConstantTripCount(L, Exiting);
  for (unsigned Iteration = 0; Iteration < TripCount; Iteration++) {
    DenseMap<Value *, Value *> SimplifiedValues;
    UnrolledInstAnalyzer Analyzer(Iteration, SimplifiedValues, SE, L);
    for (auto *BB : L->getBlocks())
      for (Instruction &I : *BB)
        Analyzer.visit(I);
    SimplifiedValuesVector.push_back(SimplifiedValues);
  }
}

std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                       const char *ModuleStr) {
  SMDiagnostic Err;
  return parseAssemblyString(ModuleStr, Err, Context);
}

TEST(UnrollAnalyzerTest, BasicSimplifications) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define i64 @propagate_loop_phis() {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]\n"
      "  %x0 = phi i64 [ 0, %entry ], [ %x2, %loop ]\n"
      "  %x1 = or i64 %x0, 1\n"
      "  %x2 = or i64 %x1, 2\n"
      "  %inc = add nuw nsw i64 %iv, 1\n"
      "  %cond = icmp sge i64 %inc, 8\n"
      "  br i1 %cond, label %loop.end, label %loop\n"
      "loop.end:\n"
      "  %x.lcssa = phi i64 [ %x2, %loop ]\n"
      "  ret i64 %x.lcssa\n"
      "}\n";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);
  SimplifiedValuesVectorTy SimplifiedValuesVector;
  runUnrollAnalyzer(*M, "propagate_loop_phis", SimplifiedValuesVector);
  unsigned TripCount = SimplifiedValuesVector.size();

  // Perform checks
  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++; // First basic block is entry - skip it.
  BasicBlock *Header = &*FI++;

  BasicBlock::iterator BBI = Header->begin();
  std::advance(BBI, 4);
  Instruction *Y1 = &*BBI++;
  Instruction *Y2 = &*BBI++;
  // Check simplification expected on the 1st iteration.
  // Check that "%inc = add nuw nsw i64 %iv, 1" is simplified to 1
  auto I1 = SimplifiedValuesVector[0].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[0].end());
  EXPECT_EQ(cast<ConstantInt>((*I1).second)->getZExtValue(), 1U);

  // Check that "%cond = icmp sge i64 %inc, 10" is simplified to false
  auto I2 = SimplifiedValuesVector[0].find(Y2);
  EXPECT_TRUE(I2 != SimplifiedValuesVector[0].end());
  EXPECT_FALSE(cast<ConstantInt>((*I2).second)->getZExtValue());

  // Check simplification expected on the last iteration.
  // Check that "%inc = add nuw nsw i64 %iv, 1" is simplified to 8
  I1 = SimplifiedValuesVector[TripCount - 1].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[TripCount - 1].end());
  EXPECT_EQ(cast<ConstantInt>((*I1).second)->getZExtValue(), TripCount);

  // Check that "%cond = icmp sge i64 %inc, 10" is simplified to false
  I2 = SimplifiedValuesVector[TripCount - 1].find(Y2);
  EXPECT_TRUE(I2 != SimplifiedValuesVector[TripCount - 1].end());
  EXPECT_TRUE(cast<ConstantInt>((*I2).second)->getZExtValue());
}

TEST(UnrollAnalyzerTest, OuterLoopSimplification) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @foo() {\n"
      "entry:\n"
      "  br label %outer.loop\n"
      "outer.loop:\n"
      "  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.loop.latch ]\n"
      "  %iv.outer.next = add nuw nsw i64 %iv.outer, 1\n"
      "  br label %inner.loop\n"
      "inner.loop:\n"
      "  %iv.inner = phi i64 [ 0, %outer.loop ], [ %iv.inner.next, %inner.loop ]\n"
      "  %iv.inner.next = add nuw nsw i64 %iv.inner, 1\n"
      "  %exitcond.inner = icmp eq i64 %iv.inner.next, 1000\n"
      "  br i1 %exitcond.inner, label %outer.loop.latch, label %inner.loop\n"
      "outer.loop.latch:\n"
      "  %exitcond.outer = icmp eq i64 %iv.outer.next, 40\n"
      "  br i1 %exitcond.outer, label %exit, label %outer.loop\n"
      "exit:\n"
      "  ret void\n"
      "}\n";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);
  SimplifiedValuesVectorTy SimplifiedValuesVector;
  runUnrollAnalyzer(*M, "foo", SimplifiedValuesVector);

  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++;
  BasicBlock *Header = &*FI++;
  BasicBlock *InnerBody = &*FI++;

  BasicBlock::iterator BBI = Header->begin();
  BBI++;
  Instruction *Y1 = &*BBI;
  BBI = InnerBody->begin();
  BBI++;
  Instruction *Y2 = &*BBI;
  // Check that we can simplify IV of the outer loop, but can't simplify the IV
  // of the inner loop if we only know the iteration number of the outer loop.
  //
  //  Y1 is %iv.outer.next, Y2 is %iv.inner.next
  auto I1 = SimplifiedValuesVector[0].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[0].end());
  auto I2 = SimplifiedValuesVector[0].find(Y2);
  EXPECT_TRUE(I2 == SimplifiedValuesVector[0].end());
}

TEST(UnrollAnalyzerTest, CmpSimplifications) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @branch_iv_trunc() {\n"
      "entry:\n"
      "  br label %for.body\n"
      "for.body:\n"
      "  %indvars.iv = phi i64 [ 0, %entry ], [ %tmp3, %for.body ]\n"
      "  %tmp2 = trunc i64 %indvars.iv to i32\n"
      "  %cmp3 = icmp eq i32 %tmp2, 5\n"
      "  %tmp3 = add nuw nsw i64 %indvars.iv, 1\n"
      "  %exitcond = icmp eq i64 %tmp3, 10\n"
      "  br i1 %exitcond, label %for.end, label %for.body\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);
  SimplifiedValuesVectorTy SimplifiedValuesVector;
  runUnrollAnalyzer(*M, "branch_iv_trunc", SimplifiedValuesVector);

  // Perform checks
  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++; // First basic block is entry - skip it.
  BasicBlock *Header = &*FI++;

  BasicBlock::iterator BBI = Header->begin();
  BBI++;
  Instruction *Y1 = &*BBI++;
  Instruction *Y2 = &*BBI++;
  // Check simplification expected on the 5th iteration.
  // Check that "%tmp2 = trunc i64 %indvars.iv to i32" is simplified to 5
  // and "%cmp3 = icmp eq i32 %tmp2, 5" is simplified to 1 (i.e. true).
  auto I1 = SimplifiedValuesVector[5].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I1).second)->getZExtValue(), 5U);
  auto I2 = SimplifiedValuesVector[5].find(Y2);
  EXPECT_TRUE(I2 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I2).second)->getZExtValue(), 1U);
}

TEST(UnrollAnalyzerTest, PtrCmpSimplifications) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "define void @ptr_cmp(i8 *%a) {\n"
      "entry:\n"
      "  %limit = getelementptr i8, i8* %a, i64 40\n"
      "  %start.iv2 = getelementptr i8, i8* %a, i64 7\n"
      "  br label %loop.body\n"
      "loop.body:\n"
      "  %iv.0 = phi i8* [ %a, %entry ], [ %iv.1, %loop.body ]\n"
      "  %iv2.0 = phi i8* [ %start.iv2, %entry ], [ %iv2.1, %loop.body ]\n"
      "  %cmp = icmp eq i8* %iv2.0, %iv.0\n"
      "  %cmp2 = icmp slt i8* %iv2.0, %iv.0\n"
      "  %cmp3 = icmp ult i8* %iv2.0, %iv.0\n"
      "  %iv.1 = getelementptr inbounds i8, i8* %iv.0, i64 1\n"
      "  %iv2.1 = getelementptr inbounds i8, i8* %iv2.0, i64 1\n"
      "  %exitcond = icmp ne i8* %iv.1, %limit\n"
      "  br i1 %exitcond, label %loop.body, label %loop.exit\n"
      "loop.exit:\n"
      "  ret void\n"
      "}\n";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);
  SimplifiedValuesVectorTy SimplifiedValuesVector;
  runUnrollAnalyzer(*M, "ptr_cmp", SimplifiedValuesVector);

  // Perform checks
  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++; // First basic block is entry - skip it.
  BasicBlock *Header = &*FI;

  BasicBlock::iterator BBI = Header->begin();
  std::advance(BBI, 2);
  Instruction *Cmp1 = &*BBI++;
  Instruction *Cmp2 = &*BBI++;
  Instruction *Cmp3 = &*BBI++;
  // Check simplification expected on the 5th iteration.
  // Check that "%cmp = icmp eq i8* %iv2.0, %iv.0" is simplified to 0.
  auto I1 = SimplifiedValuesVector[5].find(Cmp1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I1).second)->getZExtValue(), 0U);
  // Check that "%cmp2 = icmp slt i8* %iv2.0, %iv.0" does not simplify
  auto I2 = SimplifiedValuesVector[5].find(Cmp2);
  EXPECT_TRUE(I2 == SimplifiedValuesVector[5].end());
  // Check that "%cmp3 = icmp ult i8* %iv2.0, %iv.0" is simplified to 0.
  auto I3 = SimplifiedValuesVector[5].find(Cmp3);
  EXPECT_TRUE(I3 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I1).second)->getZExtValue(), 0U);
}

TEST(UnrollAnalyzerTest, CastSimplifications) {
  const char *ModuleStr =
      "target datalayout = \"e-m:o-i64:64-f80:128-n8:16:32:64-S128\"\n"
      "@known_constant = internal unnamed_addr constant [10 x i32] [i32 0, i32 1, i32 0, i32 1, i32 0, i32 259, i32 0, i32 1, i32 0, i32 1], align 16\n"
      "define void @const_load_cast() {\n"
      "entry:\n"
      "  br label %loop\n"
      "\n"
      "loop:\n"
      "  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]\n"
      "  %array_const_idx = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv\n"
      "  %const_array_element = load i32, i32* %array_const_idx, align 4\n"
      "  %se = sext i32 %const_array_element to i64\n"
      "  %ze = zext i32 %const_array_element to i64\n"
      "  %tr = trunc i32 %const_array_element to i8\n"
      "  %inc = add nuw nsw i64 %iv, 1\n"
      "  %exitcond86.i = icmp eq i64 %inc, 10\n"
      "  br i1 %exitcond86.i, label %loop.end, label %loop\n"
      "\n"
      "loop.end:\n"
      "  ret void\n"
      "}\n";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);
  SimplifiedValuesVectorTy SimplifiedValuesVector;
  runUnrollAnalyzer(*M, "const_load_cast", SimplifiedValuesVector);

  // Perform checks
  Module::iterator MI = M->begin();
  Function *F = &*MI++;
  Function::iterator FI = F->begin();
  FI++; // First basic block is entry - skip it.
  BasicBlock *Header = &*FI++;

  BasicBlock::iterator BBI = Header->begin();
  std::advance(BBI, 3);
  Instruction *Y1 = &*BBI++;
  Instruction *Y2 = &*BBI++;
  Instruction *Y3 = &*BBI++;
  // Check simplification expected on the 5th iteration.
  // "%se = sext i32 %const_array_element to i64" should be simplified to 259,
  // "%ze = zext i32 %const_array_element to i64" should be simplified to 259,
  // "%tr = trunc i32 %const_array_element to i8" should be simplified to 3.
  auto I1 = SimplifiedValuesVector[5].find(Y1);
  EXPECT_TRUE(I1 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I1).second)->getZExtValue(), 259U);
  auto I2 = SimplifiedValuesVector[5].find(Y2);
  EXPECT_TRUE(I2 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I2).second)->getZExtValue(), 259U);
  auto I3 = SimplifiedValuesVector[5].find(Y3);
  EXPECT_TRUE(I3 != SimplifiedValuesVector[5].end());
  EXPECT_EQ(cast<ConstantInt>((*I3).second)->getZExtValue(), 3U);
}
