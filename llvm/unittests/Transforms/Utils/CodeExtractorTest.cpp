//===- CodeExtractor.cpp - Unit tests for CodeExtractor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
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

  DominatorTree DT(*Func);
  CodeExtractor CE(Candidates, &DT);
  EXPECT_TRUE(CE.isEligible());

  Function *Outlined = CE.extractCodeRegion();
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

  DominatorTree DT(*Func);
  CodeExtractor CE(ExtractedBlocks, &DT);
  EXPECT_TRUE(CE.isEligible());

  Function *Outlined = CE.extractCodeRegion();
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

  DominatorTree DT(*Func);
  CodeExtractor CE(ExtractedBlocks, &DT);
  EXPECT_TRUE(CE.isEligible());

  Function *Outlined = CE.extractCodeRegion();
  EXPECT_TRUE(Outlined);
  EXPECT_FALSE(verifyFunction(*Outlined, &errs()));
  EXPECT_FALSE(verifyFunction(*Func, &errs()));
}

} // end anonymous namespace
