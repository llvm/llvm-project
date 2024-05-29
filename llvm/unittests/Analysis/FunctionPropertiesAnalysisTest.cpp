//===- FunctionPropertiesAnalysisTest.cpp - Function Properties Unit Tests-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace llvm;

namespace llvm {
extern cl::opt<bool> EnableDetailedFunctionProperties;
extern cl::opt<bool> BigBasicBlockInstructionThreshold;
extern cl::opt<bool> MediumBasicBlockInstrutionThreshold;
} // namespace llvm

namespace {

class FunctionPropertiesAnalysisTest : public testing::Test {
public:
  FunctionPropertiesAnalysisTest() {
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return LoopAnalysis(); });
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  }

protected:
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;
  FunctionAnalysisManager FAM;

  FunctionPropertiesInfo buildFPI(Function &F) {
    return FunctionPropertiesInfo::getFunctionPropertiesInfo(F, FAM);
  }

  void invalidate(Function &F) {
    PreservedAnalyses PA = PreservedAnalyses::none();
    FAM.invalidate(F, PA);
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

  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedBranchesFeatures = buildFPI(*BranchesFunction);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlocksWithSingleSuccessor, 2);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlocksWithTwoSuccessors, 1);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlocksWithMoreThanTwoSuccessors, 0);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlocksWithSinglePredecessor, 2);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlocksWithTwoPredecessors, 1);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlocksWithMoreThanTwoPredecessors, 0);
  EXPECT_EQ(DetailedBranchesFeatures.BigBasicBlocks, 0);
  EXPECT_EQ(DetailedBranchesFeatures.MediumBasicBlocks, 0);
  EXPECT_EQ(DetailedBranchesFeatures.SmallBasicBlocks, 4);
  EXPECT_EQ(DetailedBranchesFeatures.CastInstructionCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.FloatingPointInstructionCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.IntegerInstructionCount, 4);
  EXPECT_EQ(DetailedBranchesFeatures.ConstantIntOperandCount, 1);
  EXPECT_EQ(DetailedBranchesFeatures.ConstantFPOperandCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.ConstantOperandCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.InstructionOperandCount, 4);
  EXPECT_EQ(DetailedBranchesFeatures.BasicBlockOperandCount, 4);
  EXPECT_EQ(DetailedBranchesFeatures.GlobalValueOperandCount, 2);
  EXPECT_EQ(DetailedBranchesFeatures.InlineAsmOperandCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.ArgumentOperandCount, 3);
  EXPECT_EQ(DetailedBranchesFeatures.UnknownOperandCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.CriticalEdgeCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.ControlFlowEdgeCount, 4);
  EXPECT_EQ(DetailedBranchesFeatures.UnconditionalBranchCount, 2);
  EXPECT_EQ(DetailedBranchesFeatures.IntrinsicCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.DirectCallCount, 2);
  EXPECT_EQ(DetailedBranchesFeatures.IndirectCallCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.CallReturnsIntegerCount, 2);
  EXPECT_EQ(DetailedBranchesFeatures.CallReturnsFloatCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.CallReturnsPointerCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.CallWithManyArgumentsCount, 0);
  EXPECT_EQ(DetailedBranchesFeatures.CallWithPointerArgumentCount, 0);
  EnableDetailedFunctionProperties.setValue(false);
}

TEST_F(FunctionPropertiesAnalysisTest, DifferentPredecessorSuccessorCounts) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
define i64 @f1() {
  br i1 0, label %br1, label %finally
br1:
  ret i64 0
finally:
  ret i64 3
}
)IR");

  Function *F1 = M->getFunction("f1");
  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedF1Properties = buildFPI(*F1);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithSingleSuccessor, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithTwoSuccessors, 1);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithMoreThanTwoSuccessors, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithSinglePredecessor, 2);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithTwoPredecessors, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithMoreThanTwoPredecessors, 0);
  EXPECT_EQ(DetailedF1Properties.BigBasicBlocks, 0);
  EXPECT_EQ(DetailedF1Properties.MediumBasicBlocks, 0);
  EXPECT_EQ(DetailedF1Properties.SmallBasicBlocks, 3);
  EXPECT_EQ(DetailedF1Properties.CastInstructionCount, 0);
  EXPECT_EQ(DetailedF1Properties.FloatingPointInstructionCount, 0);
  EXPECT_EQ(DetailedF1Properties.IntegerInstructionCount, 0);
  EXPECT_EQ(DetailedF1Properties.ConstantIntOperandCount, 3);
  EXPECT_EQ(DetailedF1Properties.ConstantFPOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.ConstantOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.InstructionOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlockOperandCount, 2);
  EXPECT_EQ(DetailedF1Properties.GlobalValueOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.InlineAsmOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.ArgumentOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.UnknownOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.CriticalEdgeCount, 0);
  EXPECT_EQ(DetailedF1Properties.ControlFlowEdgeCount, 2);
  EXPECT_EQ(DetailedF1Properties.UnconditionalBranchCount, 0);
  EXPECT_EQ(DetailedF1Properties.IntrinsicCount, 0);
  EXPECT_EQ(DetailedF1Properties.DirectCallCount, 0);
  EXPECT_EQ(DetailedF1Properties.IndirectCallCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsIntegerCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsFloatCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsPointerCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallWithManyArgumentsCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallWithPointerArgumentCount, 0);
  EnableDetailedFunctionProperties.setValue(false);
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount), F1->size());
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount), F1->size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, FAM));
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount), F1->size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, FAM));
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount), F1->size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, FAM));
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
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(static_cast<size_t>(FPI.BasicBlockCount), F1->size() - 1);
  EXPECT_EQ(static_cast<size_t>(FPI.TotalInstructionCount),
            F1->getInstructionCount() - 2);
  EXPECT_EQ(FPI, FunctionPropertiesInfo::getFunctionPropertiesInfo(*F1, FAM));
}

TEST_F(FunctionPropertiesAnalysisTest, InlineSameLoopBB) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i32 @a()
declare i32 @b()

define i32 @f1(i32 %a) {
entry:
  br label %loop
loop:
  %i = call i32 @f2(i32 %a)
  %c = icmp slt i32 %i, %a
  br i1 %c, label %loop, label %end
end:
  %r = phi i32 [%i, %loop], [%a, %entry]
  ret i32 %r
}

define i32 @f2(i32 %a) {
  %cnd = icmp slt i32 %a, 0
  br i1 %cnd, label %then, label %else
then:
  %r1 = call i32 @a()
  br label %end
else:
  %r2 = call i32 @b()
  br label %end
end:
  %r = phi i32 [%r1, %then], [%r2, %else]
  ret i32 %r
}
)IR");

  Function *F1 = M->getFunction("f1");
  CallBase *CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  FunctionPropertiesInfo ExpectedInitial;
  ExpectedInitial.BasicBlockCount = 3;
  ExpectedInitial.TotalInstructionCount = 6;
  ExpectedInitial.BlocksReachedFromConditionalInstruction = 2;
  ExpectedInitial.Uses = 1;
  ExpectedInitial.DirectCallsToDefinedFunctions = 1;
  ExpectedInitial.MaxLoopDepth = 1;
  ExpectedInitial.TopLevelLoopCount = 1;

  FunctionPropertiesInfo ExpectedFinal = ExpectedInitial;
  ExpectedFinal.BasicBlockCount = 6;
  ExpectedFinal.DirectCallsToDefinedFunctions = 0;
  ExpectedFinal.BlocksReachedFromConditionalInstruction = 4;
  ExpectedFinal.TotalInstructionCount = 12;

  auto FPI = buildFPI(*F1);
  EXPECT_EQ(FPI, ExpectedInitial);

  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(FPI, ExpectedFinal);
}

TEST_F(FunctionPropertiesAnalysisTest, Unreachable) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define i64 @f1(i32 noundef %value) {
entry:
  br i1 true, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %conv2 = sext i32 %value to i64
  br label %cond.end

cond.false:                                       ; preds = %entry
  %call3 = call noundef i64 @f2()
  br label %extra

extra:
  br label %extra2

extra2:
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i64 [ %conv2, %cond.true ], [ %call3, %extra ]
  ret i64 %cond
}

define i64 @f2() {
entry:
  tail call void @llvm.trap()
  unreachable
}

declare void @llvm.trap()
)IR");

  Function *F1 = M->getFunction("f1");
  CallBase *CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  FunctionPropertiesInfo ExpectedInitial;
  ExpectedInitial.BasicBlockCount = 6;
  ExpectedInitial.TotalInstructionCount = 9;
  ExpectedInitial.BlocksReachedFromConditionalInstruction = 2;
  ExpectedInitial.Uses = 1;
  ExpectedInitial.DirectCallsToDefinedFunctions = 1;
  
  FunctionPropertiesInfo ExpectedFinal = ExpectedInitial;
  ExpectedFinal.BasicBlockCount = 4;
  ExpectedFinal.DirectCallsToDefinedFunctions = 0;
  ExpectedFinal.TotalInstructionCount = 7;

  auto FPI = buildFPI(*F1);
  EXPECT_EQ(FPI, ExpectedInitial);

  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(FPI, ExpectedFinal);
}

TEST_F(FunctionPropertiesAnalysisTest, InvokeSkipLP) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define i64 @f1(i32 noundef %value) {
entry:
  invoke fastcc void @f2() to label %cont unwind label %lpad
cont:
  ret i64 1
lpad:
  %lp = landingpad i32 cleanup
  br label %ehcleanup
ehcleanup:
  resume i32 0
}
define void @f2() {
  invoke noundef void @f3() to label %exit unwind label %lpad
exit:
  ret void
lpad:
  %lp = landingpad i32 cleanup
  resume i32 %lp
}
declare void @f3()
)IR");

  // The outcome of inlining will be that lpad becomes unreachable. The landing
  // pad of the invoke inherited from f2 will land on a new bb which will branch
  // to a bb containing the body of lpad.
  Function *F1 = M->getFunction("f1");
  CallBase *CB = findCall(*F1);
  EXPECT_NE(CB, nullptr);

  FunctionPropertiesInfo ExpectedInitial;
  ExpectedInitial.BasicBlockCount = 4;
  ExpectedInitial.TotalInstructionCount = 5;
  ExpectedInitial.BlocksReachedFromConditionalInstruction = 0;
  ExpectedInitial.Uses = 1;
  ExpectedInitial.DirectCallsToDefinedFunctions = 1;

  FunctionPropertiesInfo ExpectedFinal = ExpectedInitial;
  ExpectedFinal.BasicBlockCount = 6;
  ExpectedFinal.DirectCallsToDefinedFunctions = 0;
  ExpectedFinal.TotalInstructionCount = 8;

  auto FPI = buildFPI(*F1);
  EXPECT_EQ(FPI, ExpectedInitial);

  FunctionPropertiesUpdater FPU(FPI, *CB);
  InlineFunctionInfo IFI;
  auto IR = llvm::InlineFunction(*CB, IFI);
  EXPECT_TRUE(IR.isSuccess());
  invalidate(*F1);
  EXPECT_TRUE(FPU.finishAndTest(FAM));
  EXPECT_EQ(FPI, ExpectedFinal);
}

TEST_F(FunctionPropertiesAnalysisTest, DetailedOperandCount) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
@a = global i64 1

define i64 @f1(i64 %e) {
	%b = load i64, i64* @a
  %c = add i64 %b, 2
  %d = call i64 asm "mov $1,$0", "=r,r" (i64 %c)																						
	%f = add i64 %d, %e
	ret i64 %f
}
)IR");

  Function *F1 = M->getFunction("f1");
  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedF1Properties = buildFPI(*F1);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithSingleSuccessor, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithTwoSuccessors, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithMoreThanTwoSuccessors, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithSinglePredecessor, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithTwoPredecessors, 0);
  EXPECT_EQ(DetailedF1Properties.BasicBlocksWithMoreThanTwoPredecessors, 0);
  EXPECT_EQ(DetailedF1Properties.BigBasicBlocks, 0);
  EXPECT_EQ(DetailedF1Properties.MediumBasicBlocks, 0);
  EXPECT_EQ(DetailedF1Properties.SmallBasicBlocks, 1);
  EXPECT_EQ(DetailedF1Properties.CastInstructionCount, 0);
  EXPECT_EQ(DetailedF1Properties.FloatingPointInstructionCount, 0);
  EXPECT_EQ(DetailedF1Properties.IntegerInstructionCount, 4);
  EXPECT_EQ(DetailedF1Properties.ConstantIntOperandCount, 1);
  EXPECT_EQ(DetailedF1Properties.ConstantFPOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.ConstantOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.InstructionOperandCount, 4);
  EXPECT_EQ(DetailedF1Properties.BasicBlockOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.GlobalValueOperandCount, 1);
  EXPECT_EQ(DetailedF1Properties.InlineAsmOperandCount, 1);
  EXPECT_EQ(DetailedF1Properties.ArgumentOperandCount, 1);
  EXPECT_EQ(DetailedF1Properties.UnknownOperandCount, 0);
  EXPECT_EQ(DetailedF1Properties.CriticalEdgeCount, 0);
  EXPECT_EQ(DetailedF1Properties.ControlFlowEdgeCount, 0);
  EXPECT_EQ(DetailedF1Properties.UnconditionalBranchCount, 0);
  EXPECT_EQ(DetailedF1Properties.IntrinsicCount, 0);
  EXPECT_EQ(DetailedF1Properties.DirectCallCount, 1);
  EXPECT_EQ(DetailedF1Properties.IndirectCallCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsIntegerCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsFloatCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsPointerCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallWithManyArgumentsCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallWithPointerArgumentCount, 0);
  EnableDetailedFunctionProperties.setValue(false);
}

TEST_F(FunctionPropertiesAnalysisTest, IntrinsicCount) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
define float @f1(float %a) {
  %b = call float @llvm.cos.f32(float %a)
  ret float %b
}
declare float @llvm.cos.f32(float)
)IR");

  Function *F1 = M->getFunction("f1");
  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedF1Properties = buildFPI(*F1);
  EXPECT_EQ(DetailedF1Properties.IntrinsicCount, 1);
  EXPECT_EQ(DetailedF1Properties.DirectCallCount, 1);
  EXPECT_EQ(DetailedF1Properties.IndirectCallCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsIntegerCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallReturnsFloatCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsPointerCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallWithManyArgumentsCount, 0);
  EXPECT_EQ(DetailedF1Properties.CallWithPointerArgumentCount, 0);
  EnableDetailedFunctionProperties.setValue(false);
}

TEST_F(FunctionPropertiesAnalysisTest, FunctionCallMetrics) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
define i64 @f1(i64 %a) {
  %b = call i64 @f2(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a)
  %c = call ptr @f3()
  call void @f4(ptr %c)
  %d = call float @f5()
  %e = call i64 %c(i64 %b)
  ret i64 %b
}

declare i64 @f2(i64,i64,i64,i64,i64)
declare ptr @f3()
declare void @f4(ptr)
declare float @f5()
)IR");

  Function *F1 = M->getFunction("f1");
  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedF1Properties = buildFPI(*F1);
  EXPECT_EQ(DetailedF1Properties.IntrinsicCount, 0);
  EXPECT_EQ(DetailedF1Properties.DirectCallCount, 4);
  EXPECT_EQ(DetailedF1Properties.IndirectCallCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsIntegerCount, 2);
  EXPECT_EQ(DetailedF1Properties.CallReturnsFloatCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsPointerCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallWithManyArgumentsCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallWithPointerArgumentCount, 1);
  EnableDetailedFunctionProperties.setValue(false);
}

TEST_F(FunctionPropertiesAnalysisTest, CriticalEdge) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
define i64 @f1(i64 %a) {
  %b = icmp eq i64 %a, 1
  br i1 %b, label %TopBlock1, label %TopBlock2
TopBlock1:
  %c = add i64 %a, 1
  %e = icmp eq i64 %c, 2
  br i1 %e, label %BottomBlock1, label %BottomBlock2
TopBlock2:
  %d = add i64 %a, 2
  br label %BottomBlock2
BottomBlock1:
  ret i64 0
BottomBlock2:
  %f = phi i64 [ %c, %TopBlock1 ], [ %d, %TopBlock2 ]
  ret i64 %f
}
)IR");

  Function *F1 = M->getFunction("f1");
  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedF1Properties = buildFPI(*F1);
  EXPECT_EQ(DetailedF1Properties.CriticalEdgeCount, 1);
  EnableDetailedFunctionProperties.setValue(false);
}


TEST_F(FunctionPropertiesAnalysisTest, FunctionReturnVectors) {
  LLVMContext C;
  std::unique_ptr<Module> M = makeLLVMModule(C,
                                             R"IR(
define <4 x i64> @f1(<4 x i64> %a) {
  %b = call <4 x i64> @f2()
  %c = call <4 x float> @f3()
  %d = call <4 x ptr> @f4()
  ret <4 x i64> %b
}

declare <4 x i64> @f2()
declare <4 x float> @f3()
declare <4 x ptr> @f4()
)IR");

  Function *F1 = M->getFunction("f1");
  EnableDetailedFunctionProperties.setValue(true);
  FunctionPropertiesInfo DetailedF1Properties = buildFPI(*F1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsVectorIntCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsVectorFloatCount, 1);
  EXPECT_EQ(DetailedF1Properties.CallReturnsVectorPointerCount, 1);
  EnableDetailedFunctionProperties.setValue(false);
}
} // end anonymous namespace
