//===- CodeExtractor.cpp - Unit tests for CodeExtractor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
BasicBlock *getBlockByName(Function *F, StringRef name) {
  for (auto &BB : *F)
    if (BB.getName() == name)
      return &BB;
  return nullptr;
}

TEST(CodeExtractor, ExitStub) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    define i32 @foo(i32 %x, i32 %y, i32 %z) {
    header:
      %0 = icmp ugt i32 %x, %y
      br i1 %0, label %body1, label %body2

    body1:
      %1 = add i32 %z, 2
      br label %notExtracted

    body2:
      %2 = mul i32 %z, 7
      br label %notExtracted

    notExtracted:
      %3 = phi i32 [ %1, %body1 ], [ %2, %body2 ]
      %4 = add i32 %3, %x
      ret i32 %4
    }
  )invalid",
                                                Err, Ctx));

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 3> Candidates{ getBlockByName(Func, "header"),
                                           getBlockByName(Func, "body1"),
                                           getBlockByName(Func, "body2") };

  CodeExtractor CE(Candidates);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);
  BasicBlock *Exit = getBlockByName(Func, "notExtracted");
  BasicBlock *ExitSplit = getBlockByName(Outlined, "notExtracted.split");
  // Ensure that PHI in exit block has only one incoming value (from code
  // replacer block).
  EXPECT_TRUE(Exit && cast<PHINode>(Exit->front()).getNumIncomingValues() == 1);
  // Ensure that there is a PHI in outlined function with 2 incoming values.
  EXPECT_TRUE(ExitSplit &&
              cast<PHINode>(ExitSplit->front()).getNumIncomingValues() == 2);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, InputOutputMonitoring) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    define i32 @foo(i32 %x, i32 %y, i32 %z) {
    header:
      %0 = icmp ugt i32 %x, %y
      br i1 %0, label %body1, label %body2

    body1:
      %1 = add i32 %z, 2
      br label %notExtracted

    body2:
      %2 = mul i32 %z, 7
      br label %notExtracted

    notExtracted:
      %3 = phi i32 [ %1, %body1 ], [ %2, %body2 ]
      %4 = add i32 %3, %x
      ret i32 %4
    }
  )invalid",
                                                Err, Ctx));

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 3> Candidates{getBlockByName(Func, "header"),
                                          getBlockByName(Func, "body1"),
                                          getBlockByName(Func, "body2")};

  CodeExtractor CE(Candidates);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  SetVector<Value *> Inputs, Outputs;
  Function *Outlined = CE.extractCodeRegion(CEAC, Inputs, Outputs);
  EXPECT_TRUE(Outlined);

  EXPECT_EQ(Inputs.size(), 3u);
  EXPECT_EQ(Inputs[0], Func->getArg(2));
  EXPECT_EQ(Inputs[1], Func->getArg(0));
  EXPECT_EQ(Inputs[2], Func->getArg(1));
  EXPECT_EQ(Outputs.size(), 1u);
  StoreInst *SI = cast<StoreInst>(Outlined->getArg(3)->user_back());
  Value *OutputVal = SI->getValueOperand();
  EXPECT_EQ(Outputs[0], OutputVal);
  BasicBlock *Exit = getBlockByName(Func, "notExtracted");
  BasicBlock *ExitSplit = getBlockByName(Outlined, "notExtracted.split");
  // Ensure that PHI in exit block has only one incoming value (from code
  // replacer block).
  EXPECT_TRUE(Exit && cast<PHINode>(Exit->front()).getNumIncomingValues() == 1);
  // Ensure that there is a PHI in outlined function with 2 incoming values.
  EXPECT_TRUE(ExitSplit &&
              cast<PHINode>(ExitSplit->front()).getNumIncomingValues() == 2);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, ExitBlockOrderingPhis) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    define void @foo(i32 %a, i32 %b) {
    entry:
      %0 = alloca i32, align 4
      br label %test0
    test0:
      %c = load i32, i32* %0, align 4
      br label %test1
    test1:
      %e = load i32, i32* %0, align 4
      br i1 true, label %first, label %test
    test:
      %d = load i32, i32* %0, align 4
      br i1 true, label %first, label %next
    first:
      %1 = phi i32 [ %c, %test ], [ %e, %test1 ]
      ret void
    next:
      %2 = add i32 %d, 1
      %3 = add i32 %e, 1
      ret void
    }
  )invalid",
                                                Err, Ctx));
  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 3> Candidates{ getBlockByName(Func, "test0"),
                                           getBlockByName(Func, "test1"),
                                           getBlockByName(Func, "test") };

  CodeExtractor CE(Candidates);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);

  BasicBlock *FirstExitStub = getBlockByName(Outlined, "first.exitStub");
  BasicBlock *NextExitStub = getBlockByName(Outlined, "next.exitStub");

  Instruction *FirstTerm = FirstExitStub->getTerminator();
  ReturnInst *FirstReturn = dyn_cast<ReturnInst>(FirstTerm);
  EXPECT_TRUE(FirstReturn);
  ConstantInt *CIFirst = dyn_cast<ConstantInt>(FirstReturn->getReturnValue());
  EXPECT_TRUE(CIFirst->getLimitedValue() == 1u);

  Instruction *NextTerm = NextExitStub->getTerminator();
  ReturnInst *NextReturn = dyn_cast<ReturnInst>(NextTerm);
  EXPECT_TRUE(NextReturn);
  ConstantInt *CINext = dyn_cast<ConstantInt>(NextReturn->getReturnValue());
  EXPECT_TRUE(CINext->getLimitedValue() == 0u);

  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, ExitBlockOrdering) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    define void @foo(i32 %a, i32 %b) {
    entry:
      %0 = alloca i32, align 4
      br label %test0
    test0:
      %c = load i32, i32* %0, align 4
      br label %test1
    test1:
      %e = load i32, i32* %0, align 4
      br i1 true, label %first, label %test
    test:
      %d = load i32, i32* %0, align 4
      br i1 true, label %first, label %next
    first:
      ret void
    next:
      %1 = add i32 %d, 1
      %2 = add i32 %e, 1
      ret void
    }
  )invalid",
                                                Err, Ctx));
  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 3> Candidates{ getBlockByName(Func, "test0"),
                                           getBlockByName(Func, "test1"),
                                           getBlockByName(Func, "test") };

  CodeExtractor CE(Candidates);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);

  BasicBlock *FirstExitStub = getBlockByName(Outlined, "first.exitStub");
  BasicBlock *NextExitStub = getBlockByName(Outlined, "next.exitStub");

  Instruction *FirstTerm = FirstExitStub->getTerminator();
  ReturnInst *FirstReturn = dyn_cast<ReturnInst>(FirstTerm);
  EXPECT_TRUE(FirstReturn);
  ConstantInt *CIFirst = dyn_cast<ConstantInt>(FirstReturn->getReturnValue());
  EXPECT_TRUE(CIFirst->getLimitedValue() == 1u);

  Instruction *NextTerm = NextExitStub->getTerminator();
  ReturnInst *NextReturn = dyn_cast<ReturnInst>(NextTerm);
  EXPECT_TRUE(NextReturn);
  ConstantInt *CINext = dyn_cast<ConstantInt>(NextReturn->getReturnValue());
  EXPECT_TRUE(CINext->getLimitedValue() == 0u);

  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, ExitPHIOnePredFromRegion) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    define i32 @foo() {
    header:
      br i1 undef, label %extracted1, label %pred

    pred:
      br i1 undef, label %exit1, label %exit2

    extracted1:
      br i1 undef, label %extracted2, label %exit1

    extracted2:
      br label %exit2

    exit1:
      %0 = phi i32 [ 1, %extracted1 ], [ 2, %pred ]
      ret i32 %0

    exit2:
      %1 = phi i32 [ 3, %extracted2 ], [ 4, %pred ]
      ret i32 %1
    }
  )invalid", Err, Ctx));

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 2> ExtractedBlocks{
    getBlockByName(Func, "extracted1"),
    getBlockByName(Func, "extracted2")
  };

  CodeExtractor CE(ExtractedBlocks);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);
  BasicBlock *Exit1 = getBlockByName(Func, "exit1");
  BasicBlock *Exit2 = getBlockByName(Func, "exit2");
  // Ensure that PHIs in exits are not splitted (since that they have only one
  // incoming value from extracted region).
  EXPECT_TRUE(Exit1 &&
          cast<PHINode>(Exit1->front()).getNumIncomingValues() == 2);
  EXPECT_TRUE(Exit2 &&
          cast<PHINode>(Exit2->front()).getNumIncomingValues() == 2);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, StoreOutputInvokeResultAfterEHPad) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    declare i8 @hoge()

    define i32 @foo() personality i8* null {
      entry:
        %call = invoke i8 @hoge()
                to label %invoke.cont unwind label %lpad

      invoke.cont:                                      ; preds = %entry
        unreachable

      lpad:                                             ; preds = %entry
        %0 = landingpad { i8*, i32 }
                catch i8* null
        br i1 undef, label %catch, label %finally.catchall

      catch:                                            ; preds = %lpad
        %call2 = invoke i8 @hoge()
                to label %invoke.cont2 unwind label %lpad2

      invoke.cont2:                                    ; preds = %catch
        %call3 = invoke i8 @hoge()
                to label %invoke.cont3 unwind label %lpad2

      invoke.cont3:                                    ; preds = %invoke.cont2
        unreachable

      lpad2:                                           ; preds = %invoke.cont2, %catch
        %ex.1 = phi i8* [ undef, %invoke.cont2 ], [ null, %catch ]
        %1 = landingpad { i8*, i32 }
                catch i8* null
        br label %finally.catchall

      finally.catchall:                                 ; preds = %lpad33, %lpad
        %ex.2 = phi i8* [ %ex.1, %lpad2 ], [ null, %lpad ]
        unreachable
    }
  )invalid", Err, Ctx));

	if (!M) {
    Err.print("unit", errs());
    exit(1);
  }

  Function *Func = M->getFunction("foo");
  EXPECT_FALSE(verifyFunction(*Func, &errs()));

  SmallVector<BasicBlock *, 2> ExtractedBlocks{
    getBlockByName(Func, "catch"),
    getBlockByName(Func, "invoke.cont2"),
    getBlockByName(Func, "invoke.cont3"),
    getBlockByName(Func, "lpad2")
  };

  CodeExtractor CE(ExtractedBlocks);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);
  EXPECT_FALSE(verifyFunction(*Outlined, &errs()));
  EXPECT_FALSE(verifyFunction(*Func, &errs()));
}

TEST(CodeExtractor, StoreOutputInvokeResultInExitStub) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    declare i32 @bar()

    define i32 @foo() personality i8* null {
    entry:
      %0 = invoke i32 @bar() to label %exit unwind label %lpad

    exit:
      ret i32 %0

    lpad:
      %1 = landingpad { i8*, i32 }
              cleanup
      resume { i8*, i32 } %1
    }
  )invalid",
                                                Err, Ctx));

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 1> Blocks{ getBlockByName(Func, "entry"),
                                       getBlockByName(Func, "lpad") };

  CodeExtractor CE(Blocks);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, ExtractAndInvalidateAssumptionCache) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"ir(
        target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
        target triple = "aarch64"

        %b = type { i64 }
        declare void @g(i8*)

        declare void @llvm.assume(i1) #0

        define void @test() {
        entry:
          br label %label

        label:
          %0 = load %b*, %b** inttoptr (i64 8 to %b**), align 8
          %1 = getelementptr inbounds %b, %b* %0, i64 undef, i32 0
          %2 = load i64, i64* %1, align 8
          %3 = icmp ugt i64 %2, 1
          br i1 %3, label %if.then, label %if.else

        if.then:
          unreachable

        if.else:
          call void @g(i8* undef)
          store i64 undef, i64* null, align 536870912
          %4 = icmp eq i64 %2, 0
          call void @llvm.assume(i1 %4)
          unreachable
        }

        attributes #0 = { nounwind willreturn }
  )ir",
                                                Err, Ctx));

  assert(M && "Could not parse module?");
  Function *Func = M->getFunction("test");
  SmallVector<BasicBlock *, 1> Blocks{ getBlockByName(Func, "if.else") };
  AssumptionCache AC(*Func);
  CodeExtractor CE(Blocks, nullptr, false, nullptr, nullptr, &AC);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
  EXPECT_FALSE(CE.verifyAssumptionCache(*Func, *Outlined, &AC));
}

TEST(CodeExtractor, RemoveBitcastUsesFromOuterLifetimeMarkers) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"ir(
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    declare void @use(i32*)
    declare void @llvm.lifetime.start.p0i8(i64, i8*)
    declare void @llvm.lifetime.end.p0i8(i64, i8*)

    define void @foo() {
    entry:
      %0 = alloca i32
      br label %extract

    extract:
      %1 = bitcast i32* %0 to i8*
      call void @llvm.lifetime.start.p0i8(i64 4, i8* %1)
      call void @use(i32* %0)
      br label %exit

    exit:
      call void @use(i32* %0)
      call void @llvm.lifetime.end.p0i8(i64 4, i8* %1)
      ret void
    }
  )ir",
                                                Err, Ctx));

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 1> Blocks{getBlockByName(Func, "extract")};

  CodeExtractor CE(Blocks);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  SetVector<Value *> Inputs, Outputs, SinkingCands, HoistingCands;
  BasicBlock *CommonExit = nullptr;
  CE.findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  CE.findInputsOutputs(Inputs, Outputs, SinkingCands);
  EXPECT_EQ(Outputs.size(), 0U);

  Function *Outlined = CE.extractCodeRegion(CEAC);
  EXPECT_TRUE(Outlined);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}

TEST(CodeExtractor, PartialAggregateArgs) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"ir(
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    declare void @use(i32)

    define void @foo(i32 %a, i32 %b, i32 %c) {
    entry:
      br label %extract

    extract:
      call void @use(i32 %a)
      call void @use(i32 %b)
      call void @use(i32 %c)
      br label %exit

    exit:
      ret void
    }
  )ir",
                                                Err, Ctx));

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 1> Blocks{getBlockByName(Func, "extract")};

  // Create the CodeExtractor with arguments aggregation enabled.
  CodeExtractor CE(Blocks, /* DominatorTree */ nullptr,
                   /* AggregateArgs */ true);
  EXPECT_TRUE(CE.isEligible());

  CodeExtractorAnalysisCache CEAC(*Func);
  SetVector<Value *> Inputs, Outputs, SinkingCands, HoistingCands;
  BasicBlock *CommonExit = nullptr;
  CE.findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  CE.findInputsOutputs(Inputs, Outputs, SinkingCands);
  // Exclude the first input from the argument aggregate.
  CE.excludeArgFromAggregate(Inputs[0]);

  Function *Outlined = CE.extractCodeRegion(CEAC, Inputs, Outputs);
  EXPECT_TRUE(Outlined);
  // Expect 2 arguments in the outlined function: the excluded input and the
  // struct aggregate for the remaining inputs.
  EXPECT_EQ(Outlined->arg_size(), 2U);
  EXPECT_FALSE(verifyFunction(*Outlined));
  EXPECT_FALSE(verifyFunction(*Func));
}
} // end anonymous namespace
