//===- LoadsTest.cpp - local load analysis unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AnalysisTests", errs());
  return Mod;
}

TEST(LoadsTest, FindAvailableLoadedValueSameBasePtrConstantOffsetsNullAA) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
target datalayout = "p:64:64:64:32"
%class = type <{ i32, i32 }>

define i32 @f() {
entry:
  %o = alloca %class
  %f1 = getelementptr inbounds %class, %class* %o, i32 0, i32 0
  store i32 42, i32* %f1
  %f2 = getelementptr inbounds %class, %class* %o, i32 0, i32 1
  store i32 43, i32* %f2
  %v = load i32, i32* %f1
  ret i32 %v
}
)IR");
  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  auto *LI = dyn_cast<LoadInst>(Inst);
  ASSERT_TRUE(LI);
  BasicBlock::iterator BBI(LI);
  Value *Loaded = FindAvailableLoadedValue(
      LI, LI->getParent(), BBI, 0, nullptr, nullptr);
  ASSERT_TRUE(Loaded);
  auto *CI = dyn_cast<ConstantInt>(Loaded);
  ASSERT_TRUE(CI);
  ASSERT_TRUE(CI->equalsInt(42));
}

TEST(LoadsTest, CanReplacePointersIfEqual) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
@y = common global [1 x i32] zeroinitializer, align 4
@x = common global [1 x i32] zeroinitializer, align 4
declare void @use(i32*)

define void @f(i32* %p1, i32* %p2, i64 %i) {
  call void @use(i32* getelementptr inbounds ([1 x i32], [1 x i32]* @y, i64 0, i64 0))

  %p1_idx = getelementptr inbounds i32, i32* %p1, i64 %i
  call void @use(i32* %p1_idx)

  %icmp = icmp eq i32* %p1, getelementptr inbounds ([1 x i32], [1 x i32]* @y, i64 0, i64 0)
  %ptrInt = ptrtoint i32* %p1 to i64
  ret void
}
)IR");
  const DataLayout &DL = M->getDataLayout();
  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);

  Value *P1 = &*F->arg_begin();
  Value *P2 = F->getArg(1);
  Value *NullPtr = Constant::getNullValue(P1->getType());
  auto InstIter = F->front().begin();
  CallInst *UserOfY = cast<CallInst>(&*InstIter);
  Value *ConstDerefPtr = UserOfY->getArgOperand(0);
  // We cannot replace two pointers in arbitrary instructions unless we are
  // replacing with null, a constant dereferencable pointer or they have the
  // same underlying object.
  EXPECT_FALSE(canReplacePointersIfEqual(ConstDerefPtr, P1, DL));
  EXPECT_FALSE(canReplacePointersIfEqual(P1, P2, DL));
  EXPECT_TRUE(canReplacePointersIfEqual(P1, ConstDerefPtr, DL));
  EXPECT_TRUE(canReplacePointersIfEqual(P1, NullPtr, DL));

  GetElementPtrInst *BasedOnP1 = cast<GetElementPtrInst>(&*++InstIter);
  EXPECT_TRUE(canReplacePointersIfEqual(BasedOnP1, P1, DL));
  EXPECT_FALSE(canReplacePointersIfEqual(BasedOnP1, P2, DL));

  // We can replace two arbitrary pointers in icmp and ptrtoint instructions.
  auto P1UseIter = P1->use_begin();
  const Use &PtrToIntUse = *P1UseIter;
  const Use &IcmpUse = *++P1UseIter;
  const Use &GEPUse = *++P1UseIter;
  EXPECT_FALSE(canReplacePointersInUseIfEqual(GEPUse, P2, DL));
  EXPECT_TRUE(canReplacePointersInUseIfEqual(PtrToIntUse, P2, DL));
  EXPECT_TRUE(canReplacePointersInUseIfEqual(IcmpUse, P2, DL));
}

TEST(LoadsTest, IsDerefReadOnlyLoop) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
define i64 @f1() {
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %retval = phi i64 [ %index, %loop ], [ 67, %loop.inc ]
  ret i64 %retval
}

define i64 @f2(ptr %p1) {
entry:
  %p2 = alloca [1024 x i8]
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %retval = phi i64 [ %index, %loop ], [ 67, %loop.inc ]
  ret i64 %retval
}
)IR");
  auto *GV1 = M->getNamedValue("f1");
  auto *GV2 = M->getNamedValue("f2");
  ASSERT_TRUE(GV1 && GV2);
  auto *F1 = dyn_cast<Function>(GV1);
  auto *F2 = dyn_cast<Function>(GV2);
  ASSERT_TRUE(F1 && F2);

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);

  auto IsDerefReadOnlyLoop = [&TLI](Function *F) -> bool {
    AssumptionCache AC(*F);
    DominatorTree DT(*F);
    LoopInfo LI(DT);
    ScalarEvolution SE(*F, TLI, AC, DT, LI);

    Function::iterator FI = F->begin();
    // First basic block is entry - skip it.
    BasicBlock *Header = &*(++FI);
    assert(Header->getName() == "loop");
    Loop *L = LI.getLoopFor(Header);

    return isDereferenceableReadOnlyLoop(L, &SE, &DT, &AC);
  };

  ASSERT_TRUE(IsDerefReadOnlyLoop(F1));
  ASSERT_FALSE(IsDerefReadOnlyLoop(F2));
}
