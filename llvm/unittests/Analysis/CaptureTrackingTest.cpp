//=======- CaptureTrackingTest.cpp - Unit test for the Capture Tracking ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(CaptureTracking, MaxUsesToExplore) {
  StringRef Assembly = R"(
    ; Function Attrs: nounwind ssp uwtable
    declare void @doesnt_capture(i8* nocapture, i8* nocapture, i8* nocapture, 
                                 i8* nocapture, i8* nocapture)

    ; %arg has 5 uses
    define void @test_few_uses(i8* %arg) {
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      ret void
    }

    ; %arg has 50 uses
    define void @test_many_uses(i8* %arg) {
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      call void @doesnt_capture(i8* %arg, i8* %arg, i8* %arg, i8* %arg, i8* %arg)
      ret void
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  auto Test = [&M](const char *FName, unsigned FalseMaxUsesLimit,
                   unsigned TrueMaxUsesLimit) {
    Function *F = M->getFunction(FName);
    ASSERT_NE(F, nullptr);
    Value *Arg = &*F->arg_begin();
    ASSERT_NE(Arg, nullptr);
    ASSERT_FALSE(PointerMayBeCaptured(Arg, true, true, FalseMaxUsesLimit));
    ASSERT_TRUE(PointerMayBeCaptured(Arg, true, true, TrueMaxUsesLimit));

    BasicBlock *EntryBB = &F->getEntryBlock();
    DominatorTree DT(*F);

    Instruction *Ret = EntryBB->getTerminator();
    ASSERT_TRUE(isa<ReturnInst>(Ret));
    ASSERT_FALSE(PointerMayBeCapturedBefore(Arg, true, true, Ret, &DT, false,
                                            FalseMaxUsesLimit));
    ASSERT_TRUE(PointerMayBeCapturedBefore(Arg, true, true, Ret, &DT, false,
                                           TrueMaxUsesLimit));
  };

  Test("test_few_uses", 6, 4);
  Test("test_many_uses", 50, 30);
}
