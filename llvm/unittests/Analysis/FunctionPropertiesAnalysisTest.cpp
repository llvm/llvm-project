//===- FunctionPropertiesAnalysisTest.cpp - Function Properties Unit Tests-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace llvm;
namespace {

class FunctionPropertiesAnalysisTest : public testing::Test {
protected:
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  FunctionPropertiesInfo buildFPI(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    return FunctionPropertiesInfo::getFunctionPropertiesInfo(F, *LI);
  }

  std::unique_ptr<Module> makeLLVMModule(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
    if (!Mod)
      Err.print("MLAnalysisTests", errs());
    return Mod;
  }
  
  CallBase* findCall(Function& F, const char* Name = nullptr) {
    for (auto &BB : F)
      for (auto &I : BB )
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (!Name || CB->getName() == Name)
            return CB;
    return nullptr;
  }
};

TEST_F(FunctionPropertiesAnalysisTest, BasicTest) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
declare i32 @f1(i32)
declare i32 @f2(i32)
define i32 @branches(i32) {
  %cond = icmp slt i32 %0, 3
  br i1 %cond, label %then, label %else
then:
  %ret.1 = call i32 @f1(i32 %0)
  br label %last.block
else:
  %ret.2 = call i32 @f2(i32 %0)
  br label %last.block
last.block:
  %ret = phi i32 [%ret.1, %then], [%ret.2, %else]
  ret i32 %ret
}
define internal i32 @top() {
  %1 = call i32 @branches(i32 2)
  %2 = call i32 @f1(i32 %1)
  ret i32 %2
}
)IR");

  Function *BranchesFunction = M->getFunction("branches");
  FunctionPropertiesInfo BranchesFeatures = buildFPI(*BranchesFunction);
  EXPECT_EQ(BranchesFeatures.BasicBlockCount, 4);
  EXPECT_EQ(BranchesFeatures.BlocksReachedFromConditionalInstruction, 2);
  // 2 Users: top is one. The other is added because @branches is not internal,
  // so it may have external callers.
  EXPECT_EQ(BranchesFeatures.Uses, 2);
  EXPECT_EQ(BranchesFeatures.DirectCallsToDefinedFunctions, 0);
  EXPECT_EQ(BranchesFeatures.LoadInstCount, 0);
  EXPECT_EQ(BranchesFeatures.StoreInstCount, 0);
  EXPECT_EQ(BranchesFeatures.MaxLoopDepth, 0);
  EXPECT_EQ(BranchesFeatures.TopLevelLoopCount, 0);

  Function *TopFunction = M->getFunction("top");
  FunctionPropertiesInfo TopFeatures = buildFPI(*TopFunction);
  EXPECT_EQ(TopFeatures.BasicBlockCount, 1);
  EXPECT_EQ(TopFeatures.BlocksReachedFromConditionalInstruction, 0);
  EXPECT_EQ(TopFeatures.Uses, 0);
  EXPECT_EQ(TopFeatures.DirectCallsToDefinedFunctions, 1);
  EXPECT_EQ(BranchesFeatures.LoadInstCount, 0);
  EXPECT_EQ(BranchesFeatures.StoreInstCount, 0);
  EXPECT_EQ(BranchesFeatures.MaxLoopDepth, 0);
  EXPECT_EQ(BranchesFeatures.TopLevelLoopCount, 0);
}

TEST_F(FunctionPropertiesAnalysisTest, InlineSameBBSimple) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32 @f1(i32 %a) {
  %b = call i32 @f2(i32 %a)
  %c = add i32 %b, 2
  ret i32 %c
}

define i32 @f2(i32 %a) {
  %b = add i32 %a, 1
  ret i32 %b
}
)IR");

  Function *F1 = M->getFunction("f1");
  CallBase* CB = findCall(*F1, "b");
  EXPECT_NE(CB, nullptr);

  FunctionPropertiesInfo ExpectedInitial;
  ExpectedInitial.BasicBlockCount = 1;
  ExpectedInitial.TotalInstructionCount = 3;
  ExpectedInitial.Uses = 1;
  ExpectedInitial.DirectCallsToDefinedFunctions = 1;

  FunctionPropertiesInfo ExpectedFinal = ExpectedInitial;
  ExpectedFinal.DirectCallsToDefinedFunctions = 0;

  auto FPI = buildFPI(*F1);
  EXPECT_EQ(FPI, ExpectedInitial);

  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  FPU.finish(*LI);
  EXPECT_EQ(FPI, ExpectedFinal);
}

TEST_F(FunctionPropertiesAnalysisTest, InlineSameBBLargerCFG) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32 @f1(i32 %a) {
entry:
  %i = icmp slt i32 %a, 0
  br i1 %i, label %if.then, label %if.else
if.then:
  %b = call i32 @f2(i32 %a)
  %c1 = add i32 %b, 2
  br label %end
if.else:
  %c2 = add i32 %a, 1
  br label %end
end:
  %ret = phi i32 [%c1, %if.then],[%c2, %if.else]
  ret i32 %ret
}

define i32 @f2(i32 %a) {
  %b = add i32 %a, 1
  ret i32 %b
}
)IR");

  Function *F1 = M->getFunction("f1");
  CallBase* CB = findCall(*F1, "b");
  EXPECT_NE(CB, nullptr);

  FunctionPropertiesInfo ExpectedInitial;
  ExpectedInitial.BasicBlockCount = 4;
  ExpectedInitial.BlocksReachedFromConditionalInstruction = 2;
  ExpectedInitial.TotalInstructionCount = 9;
  ExpectedInitial.Uses = 1;
  ExpectedInitial.DirectCallsToDefinedFunctions = 1;

  FunctionPropertiesInfo ExpectedFinal = ExpectedInitial;
  ExpectedFinal.DirectCallsToDefinedFunctions = 0;

  auto FPI = buildFPI(*F1);
  EXPECT_EQ(FPI, ExpectedInitial);

  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  FPU.finish(*LI);
  EXPECT_EQ(FPI, ExpectedFinal);
}

TEST_F(FunctionPropertiesAnalysisTest, InlineSameBBLoops) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
define i32 @f1(i32 %a) {
entry:
  %i = icmp slt i32 %a, 0
  br i1 %i, label %if.then, label %if.else
if.then:
  %b = call i32 @f2(i32 %a)
  %c1 = add i32 %b, 2
  br label %end
if.else:
  %c2 = add i32 %a, 1
  br label %end
end:
  %ret = phi i32 [%c1, %if.then],[%c2, %if.else]
  ret i32 %ret
}

define i32 @f2(i32 %a) {
entry:
  br label %loop
loop:
  %indvar = phi i32 [%indvar.next, %loop], [0, %entry]
  %b = add i32 %a, %indvar
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %a
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %b
}
)IR");

  Function *F1 = M->getFunction("f1");
  CallBase* CB = findCall(*F1, "b");
  EXPECT_NE(CB, nullptr);

  FunctionPropertiesInfo ExpectedInitial;
  ExpectedInitial.BasicBlockCount = 4;
  ExpectedInitial.BlocksReachedFromConditionalInstruction = 2;
  ExpectedInitial.TotalInstructionCount = 9;
  ExpectedInitial.Uses = 1;
  ExpectedInitial.DirectCallsToDefinedFunctions = 1;

  FunctionPropertiesInfo ExpectedFinal;
  ExpectedFinal.BasicBlockCount = 6;
  ExpectedFinal.BlocksReachedFromConditionalInstruction = 4;
  ExpectedFinal.Uses = 1;
  ExpectedFinal.MaxLoopDepth = 1;
  ExpectedFinal.TopLevelLoopCount = 1;
  ExpectedFinal.TotalInstructionCount = 14;

  auto FPI = buildFPI(*F1);
  EXPECT_EQ(FPI, ExpectedInitial);
  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;

  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  DominatorTree DTNew(*F1);
  LoopInfo LINew(DTNew);
  FPU.finish(LINew);
  EXPECT_EQ(FPI, ExpectedFinal);
}

TEST_F(FunctionPropertiesAnalysisTest, InvokeSimple) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
declare void @might_throw()

define internal void @callee() {
entry:
  call void @might_throw()
  ret void
}

define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @callee()
      to label %cont unwind label %exc

cont:
  ret i32 0

exc:
  %exn = landingpad {i8*, i32}
         cleanup
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)
)IR");

  Function *F1 = M->getFunction("caller");
  CallBase* CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  auto FPI = buildFPI(*F1);
  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  DominatorTree DTNew(*F1);
  LoopInfo LINew(DTNew);
  FPU.finish(LINew);
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount),
            F1->getBasicBlockList().size());
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount());
}

TEST_F(FunctionPropertiesAnalysisTest, InvokeUnreachableHandler) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
declare void @might_throw()

define internal i32 @callee() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @might_throw()
      to label %cont unwind label %exc

cont:
  ret i32 0

exc:
  %exn = landingpad {i8*, i32}
         cleanup
  resume { i8*, i32 } %exn
}

define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
entry:
  %X = invoke i32 @callee()
           to label %cont unwind label %Handler

cont:
  ret i32 %X

Handler:
  %exn = landingpad {i8*, i32}
         cleanup
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)
)IR");

  Function *F1 = M->getFunction("caller");
  CallBase* CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  auto FPI = buildFPI(*F1);
  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  DominatorTree DTNew(*F1);
  LoopInfo LINew(DTNew);
  FPU.finish(LINew);
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount),
            F1->getBasicBlockList().size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, LINew));
}

TEST_F(FunctionPropertiesAnalysisTest, Rethrow) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
declare void @might_throw()

define internal i32 @callee() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @might_throw()
      to label %cont unwind label %exc

cont:
  ret i32 0

exc:
  %exn = landingpad {i8*, i32}
         cleanup
  resume { i8*, i32 } %exn
}

define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
entry:
  %X = invoke i32 @callee()
           to label %cont unwind label %Handler

cont:
  ret i32 %X

Handler:
  %exn = landingpad {i8*, i32}
         cleanup
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)
)IR");

  Function *F1 = M->getFunction("caller");
  CallBase* CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  auto FPI = buildFPI(*F1);
  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  DominatorTree DTNew(*F1);
  LoopInfo LINew(DTNew);
  FPU.finish(LINew);
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount),
            F1->getBasicBlockList().size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, LINew));
}

TEST_F(FunctionPropertiesAnalysisTest, LPadChanges) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
declare void @external_func()

@exception_type1 = external global i8
@exception_type2 = external global i8


define internal void @inner() personality i8* null {
  invoke void @external_func()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32
      catch i8* @exception_type1
  resume i32 %lp
}

define void @outer() personality i8* null {
  invoke void @inner()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32
      cleanup
      catch i8* @exception_type2
  resume i32 %lp
}

)IR");

  Function *F1 = M->getFunction("outer");
  CallBase* CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  auto FPI = buildFPI(*F1);
  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  DominatorTree DTNew(*F1);
  LoopInfo LINew(DTNew);
  FPU.finish(LINew);
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount),
            F1->getBasicBlockList().size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, LINew));
}

TEST_F(FunctionPropertiesAnalysisTest, LPadChangesConditional) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
declare void @external_func()

@exception_type1 = external global i8
@exception_type2 = external global i8


define internal void @inner() personality i8* null {
  invoke void @external_func()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32
      catch i8* @exception_type1
  resume i32 %lp
}

define void @outer(i32 %a) personality i8* null {
entry:
  %i = icmp slt i32 %a, 0
  br i1 %i, label %if.then, label %cont
if.then:
  invoke void @inner()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad i32
      cleanup
      catch i8* @exception_type2
  resume i32 %lp
}

)IR");

  Function *F1 = M->getFunction("outer");
  CallBase* CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  auto FPI = buildFPI(*F1);
  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  DominatorTree DTNew(*F1);
  LoopInfo LINew(DTNew);
  FPU.finish(LINew);
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount),
            F1->getBasicBlockList().size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, LINew));
}

} // end anonymous namespace
