//===- FunctionSpecializationTest.cpp - Cost model unit tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/FunctionSpecialization.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {

static void removeSSACopy(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &Inst : llvm::make_early_inc_range(BB)) {
      auto *BC = dyn_cast<BitCastInst>(&Inst);
      if (!BC || BC->getType() != BC->getOperand(0)->getType())
        continue;
      Inst.replaceAllUsesWith(BC->getOperand(0));
      Inst.eraseFromParent();
    }
  }
}

class FunctionSpecializationTest : public testing::Test {
protected:
  LLVMContext Ctx;
  FunctionAnalysisManager FAM;
  std::unique_ptr<Module> M;
  std::unique_ptr<SCCPSolver> Solver;
  SmallVector<Instruction *, 8> KnownConstants;

  FunctionSpecializationTest() {
    FAM.registerPass([&] { return TargetLibraryAnalysis(); });
    FAM.registerPass([&] { return TargetIRAnalysis(); });
    FAM.registerPass([&] { return BlockFrequencyAnalysis(); });
    FAM.registerPass([&] { return BranchProbabilityAnalysis(); });
    FAM.registerPass([&] { return LoopAnalysis(); });
    FAM.registerPass([&] { return AssumptionAnalysis(); });
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return PostDominatorTreeAnalysis(); });
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  }

  Module &parseModule(const char *ModuleString) {
    SMDiagnostic Err;
    M = parseAssemblyString(ModuleString, Err, Ctx);
    EXPECT_TRUE(M);
    return *M;
  }

  FunctionSpecializer getSpecializerFor(Function *F) {
    auto GetTLI = [this](Function &F) -> const TargetLibraryInfo & {
      return FAM.getResult<TargetLibraryAnalysis>(F);
    };
    auto GetTTI = [this](Function &F) -> TargetTransformInfo & {
      return FAM.getResult<TargetIRAnalysis>(F);
    };
    auto GetAC = [this](Function &F) -> AssumptionCache & {
      return FAM.getResult<AssumptionAnalysis>(F);
    };
    auto GetDT = [this](Function &F) -> DominatorTree & {
      return FAM.getResult<DominatorTreeAnalysis>(F);
    };
    auto GetBFI = [this](Function &F) -> BlockFrequencyInfo & {
      return FAM.getResult<BlockFrequencyAnalysis>(F);
    };

    Solver = std::make_unique<SCCPSolver>(M->getDataLayout(), GetTLI, Ctx);

    DominatorTree &DT = GetDT(*F);
    AssumptionCache &AC = GetAC(*F);
    Solver->addPredicateInfo(*F, DT, AC);

    Solver->markBlockExecutable(&F->front());
    for (Argument &Arg : F->args())
      Solver->markOverdefined(&Arg);
    Solver->solveWhileResolvedUndefsIn(*M);

    removeSSACopy(*F);

    return FunctionSpecializer(*Solver, *M, &FAM, GetBFI, GetTLI, GetTTI,
                               GetAC);
  }

  Cost getCodeSizeSavings(Instruction &I, bool HasLatencySavings = true) {
    auto &TTI = FAM.getResult<TargetIRAnalysis>(*I.getFunction());

    Cost CodeSize =
        TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);

    if (HasLatencySavings)
      KnownConstants.push_back(&I);

    return CodeSize;
  }

  Cost getLatencySavings(Function *F) {
    auto &TTI = FAM.getResult<TargetIRAnalysis>(*F);
    auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(*F);

    Cost Latency = 0;
    for (const Instruction *I : KnownConstants)
      Latency += BFI.getBlockFreq(I->getParent()).getFrequency() /
                 BFI.getEntryFreq().getFrequency() *
                 TTI.getInstructionCost(I, TargetTransformInfo::TCK_Latency);

    return Latency;
  }
};

} // namespace llvm

using namespace llvm;

TEST_F(FunctionSpecializationTest, SwitchInst) {
  const char *ModuleString = R"(
    define void @foo(i32 %a, i32 %b, i32 %i) {
    entry:
      br label %loop
    loop:
      switch i32 %i, label %default
      [ i32 1, label %case1
        i32 2, label %case2 ]
    case1:
      %0 = mul i32 %a, 2
      %1 = sub i32 6, 5
      br label %bb1
    case2:
      %2 = and i32 %b, 3
      %3 = sdiv i32 8, 2
      br label %bb2
    bb1:
      %4 = add i32 %0, %b
      br label %loop
    bb2:
      %5 = or i32 %2, %a
      br label %loop
    default:
      ret void
    }
  )";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  FunctionSpecializer Specializer = getSpecializerFor(F);
  InstCostVisitor Visitor = Specializer.getInstCostVisitorFor(F);

  Constant *One = ConstantInt::get(IntegerType::getInt32Ty(M.getContext()), 1);

  auto FuncIter = F->begin();
  BasicBlock &Loop = *++FuncIter;
  BasicBlock &Case1 = *++FuncIter;
  BasicBlock &Case2 = *++FuncIter;
  BasicBlock &BB1 = *++FuncIter;
  BasicBlock &BB2 = *++FuncIter;

  Instruction &Switch = Loop.front();
  Instruction &Mul = Case1.front();
  Instruction &And = Case2.front();
  Instruction &Sdiv = *++Case2.begin();
  Instruction &BrBB2 = Case2.back();
  Instruction &Add = BB1.front();
  Instruction &Or = BB2.front();
  Instruction &BrLoop = BB2.back();

  // mul
  Cost Ref = getCodeSizeSavings(Mul);
  Cost Test = Visitor.getCodeSizeSavingsForArg(F->getArg(0), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // and + or + add
  Ref = getCodeSizeSavings(And) + getCodeSizeSavings(Or) +
        getCodeSizeSavings(Add);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(1), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // switch + sdiv + br + br
  Ref = getCodeSizeSavings(Switch) +
        getCodeSizeSavings(Sdiv, /*HasLatencySavings=*/false) +
        getCodeSizeSavings(BrBB2, /*HasLatencySavings=*/false) +
        getCodeSizeSavings(BrLoop, /*HasLatencySavings=*/false);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(2), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // Latency.
  Ref = getLatencySavings(F);
  Test = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);
}

TEST_F(FunctionSpecializationTest, BranchInst) {
  const char *ModuleString = R"(
    define void @foo(i32 %a, i32 %b, i1 %cond) {
    entry:
      br label %loop
    loop:
      br i1 %cond, label %bb0, label %bb3
    bb0:
      %0 = mul i32 %a, 2
      %1 = sub i32 6, 5
      br i1 %cond, label %bb1, label %bb2
    bb1:
      %2 = add i32 %0, %b
      %3 = sdiv i32 8, 2
      br label %bb2
    bb2:
      br label %loop
    bb3:
      ret void
    }
  )";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  FunctionSpecializer Specializer = getSpecializerFor(F);
  InstCostVisitor Visitor = Specializer.getInstCostVisitorFor(F);

  Constant *One = ConstantInt::get(IntegerType::getInt32Ty(M.getContext()), 1);
  Constant *False = ConstantInt::getFalse(M.getContext());

  auto FuncIter = F->begin();
  BasicBlock &Loop = *++FuncIter;
  BasicBlock &BB0 = *++FuncIter;
  BasicBlock &BB1 = *++FuncIter;
  BasicBlock &BB2 = *++FuncIter;

  Instruction &Branch = Loop.front();
  Instruction &Mul = BB0.front();
  Instruction &Sub = *++BB0.begin();
  Instruction &BrBB1BB2 = BB0.back();
  Instruction &Add = BB1.front();
  Instruction &Sdiv = *++BB1.begin();
  Instruction &BrBB2 = BB1.back();
  Instruction &BrLoop = BB2.front();

  // mul
  Cost Ref = getCodeSizeSavings(Mul);
  Cost Test = Visitor.getCodeSizeSavingsForArg(F->getArg(0), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // add
  Ref = getCodeSizeSavings(Add);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(1), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // branch + sub + br + sdiv + br
  Ref = getCodeSizeSavings(Branch) +
        getCodeSizeSavings(Sub, /*HasLatencySavings=*/false) +
        getCodeSizeSavings(BrBB1BB2) +
        getCodeSizeSavings(Sdiv, /*HasLatencySavings=*/false) +
        getCodeSizeSavings(BrBB2, /*HasLatencySavings=*/false) +
        getCodeSizeSavings(BrLoop, /*HasLatencySavings=*/false);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(2), False);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // Latency.
  Ref = getLatencySavings(F);
  Test = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);
}

TEST_F(FunctionSpecializationTest, SelectInst) {
  const char *ModuleString = R"(
    define i32 @foo(i1 %cond, i32 %a, i32 %b) {
      %sel = select i1 %cond, i32 %a, i32 %b
      ret i32 %sel
    }
  )";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  FunctionSpecializer Specializer = getSpecializerFor(F);
  InstCostVisitor Visitor = Specializer.getInstCostVisitorFor(F);

  Constant *One = ConstantInt::get(IntegerType::getInt32Ty(M.getContext()), 1);
  Constant *Zero = ConstantInt::get(IntegerType::getInt32Ty(M.getContext()), 0);
  Constant *False = ConstantInt::getFalse(M.getContext());
  Instruction &Select = *F->front().begin();

  Cost RefCodeSize = getCodeSizeSavings(Select);
  Cost RefLatency = getLatencySavings(F);

  Cost TestCodeSize = Visitor.getCodeSizeSavingsForArg(F->getArg(0), False);
  EXPECT_TRUE(TestCodeSize == 0);
  TestCodeSize = Visitor.getCodeSizeSavingsForArg(F->getArg(1), One);
  EXPECT_TRUE(TestCodeSize == 0);
  Cost TestLatency = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_TRUE(TestLatency == 0);

  TestCodeSize = Visitor.getCodeSizeSavingsForArg(F->getArg(2), Zero);
  EXPECT_EQ(TestCodeSize, RefCodeSize);
  EXPECT_TRUE(TestCodeSize > 0);
  TestLatency = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_EQ(TestLatency, RefLatency);
  EXPECT_TRUE(TestLatency > 0);
}

TEST_F(FunctionSpecializationTest, Misc) {
  const char *ModuleString = R"(
    %struct_t = type { [8 x i16], [8 x i16], i32, i32, i32, ptr, [8 x i8] }
    @g = constant %struct_t zeroinitializer, align 16

    declare i32 @llvm.smax.i32(i32, i32)
    declare i32 @bar(i32)

    define i32 @foo(i8 %a, i1 %cond, ptr %b, i32 %c) {
      %cmp = icmp eq i8 %a, 10
      %ext = zext i1 %cmp to i64
      %sel = select i1 %cond, i64 %ext, i64 1
      %gep = getelementptr inbounds %struct_t, ptr %b, i64 %sel, i32 4
      %ld = load i32, ptr %gep
      %fr = freeze i32 %ld
      %smax = call i32 @llvm.smax.i32(i32 %fr, i32 1)
      %call = call i32 @bar(i32 %smax)
      %fr2 = freeze i32 %c
      %add = add i32 %call, %fr2
      ret i32 %add
    }
  )";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  FunctionSpecializer Specializer = getSpecializerFor(F);
  InstCostVisitor Visitor = Specializer.getInstCostVisitorFor(F);

  GlobalVariable *GV = M.getGlobalVariable("g");
  Constant *One = ConstantInt::get(IntegerType::getInt8Ty(M.getContext()), 1);
  Constant *True = ConstantInt::getTrue(M.getContext());
  Constant *Undef = UndefValue::get(IntegerType::getInt32Ty(M.getContext()));

  auto BlockIter = F->front().begin();
  Instruction &Icmp = *BlockIter++;
  Instruction &Zext = *BlockIter++;
  Instruction &Select = *BlockIter++;
  Instruction &Gep = *BlockIter++;
  Instruction &Load = *BlockIter++;
  Instruction &Freeze = *BlockIter++;
  Instruction &Smax = *BlockIter++;

  // icmp + zext
  Cost Ref = getCodeSizeSavings(Icmp) + getCodeSizeSavings(Zext);
  Cost Test = Visitor.getCodeSizeSavingsForArg(F->getArg(0), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // select
  Ref = getCodeSizeSavings(Select);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(1), True);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // gep + load + freeze + smax
  Ref = getCodeSizeSavings(Gep) + getCodeSizeSavings(Load) +
        getCodeSizeSavings(Freeze) + getCodeSizeSavings(Smax);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(2), GV);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(3), Undef);
  EXPECT_TRUE(Test == 0);

  // Latency.
  Ref = getLatencySavings(F);
  Test = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);
}

TEST_F(FunctionSpecializationTest, PhiNode) {
  const char *ModuleString = R"(
    define void @foo(i32 %a, i32 %b, i32 %i) {
    entry:
      br label %loop
    loop:
      %0 = phi i32 [ %a, %entry ], [ %3, %bb ]
      switch i32 %i, label %default
      [ i32 1, label %case1
        i32 2, label %case2 ]
    case1:
      %1 = add i32 %0, 1
      br label %bb
    case2:
      %2 = phi i32 [ %a, %entry ], [ %0, %loop ]
      br label %bb
    bb:
      %3 = phi i32 [ %b, %case1 ], [ %2, %case2 ], [ %3, %bb ]
      %4 = icmp eq i32 %3, 1
      br i1 %4, label %bb, label %loop
    default:
      ret void
    }
  )";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  FunctionSpecializer Specializer = getSpecializerFor(F);
  InstCostVisitor Visitor = Specializer.getInstCostVisitorFor(F);

  Constant *One = ConstantInt::get(IntegerType::getInt32Ty(M.getContext()), 1);

  auto FuncIter = F->begin();
  BasicBlock &Loop = *++FuncIter;
  BasicBlock &Case1 = *++FuncIter;
  BasicBlock &Case2 = *++FuncIter;
  BasicBlock &BB = *++FuncIter;

  Instruction &PhiLoop = Loop.front();
  Instruction &Switch = Loop.back();
  Instruction &Add = Case1.front();
  Instruction &PhiCase2 = Case2.front();
  Instruction &BrBB = Case2.back();
  Instruction &PhiBB = BB.front();
  Instruction &Icmp = *++BB.begin();
  Instruction &Branch = BB.back();

  Cost Test = Visitor.getCodeSizeSavingsForArg(F->getArg(0), One);
  EXPECT_TRUE(Test == 0);

  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(1), One);
  EXPECT_TRUE(Test == 0);

  Test = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_TRUE(Test == 0);

  // switch + phi + br
  Cost Ref = getCodeSizeSavings(Switch) +
             getCodeSizeSavings(PhiCase2, /*HasLatencySavings=*/false) +
             getCodeSizeSavings(BrBB, /*HasLatencySavings=*/false);
  Test = Visitor.getCodeSizeSavingsForArg(F->getArg(2), One);
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0 && Test > 0);

  // phi + phi + add + icmp + branch
  Ref = getCodeSizeSavings(PhiBB) + getCodeSizeSavings(PhiLoop) +
        getCodeSizeSavings(Add) + getCodeSizeSavings(Icmp) +
        getCodeSizeSavings(Branch);
  Test = Visitor.getCodeSizeSavingsFromPendingPHIs();
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);

  // Latency.
  Ref = getLatencySavings(F);
  Test = Visitor.getLatencySavingsForKnownConstants();
  EXPECT_EQ(Test, Ref);
  EXPECT_TRUE(Test > 0);
}

TEST_F(FunctionSpecializationTest, BinOp) {
  // Verify that we can handle binary operators even when only one operand is
  // constant.
  const char *ModuleString = R"(
    define i32 @foo(i1 %a, i1 %b) {
      %and1 = and i1 %a, %b
      %and2 = and i1 %b, %and1
      %sel = select i1 %and2, i32 1, i32 0
      ret i32 %sel
    }
  )";

  Module &M = parseModule(ModuleString);
  Function *F = M.getFunction("foo");
  FunctionSpecializer Specializer = getSpecializerFor(F);
  InstCostVisitor Visitor = Specializer.getInstCostVisitorFor(F);

  Constant *False = ConstantInt::getFalse(M.getContext());
  BasicBlock &BB = F->front();
  Instruction &And1 = BB.front();
  Instruction &And2 = *++BB.begin();
  Instruction &Select = *++BB.begin();

  Cost RefCodeSize = getCodeSizeSavings(And1) + getCodeSizeSavings(And2) +
                     getCodeSizeSavings(Select);
  Cost RefLatency = getLatencySavings(F);

  Cost TestCodeSize = Visitor.getCodeSizeSavingsForArg(F->getArg(0), False);
  Cost TestLatency = Visitor.getLatencySavingsForKnownConstants();

  EXPECT_EQ(TestCodeSize, RefCodeSize);
  EXPECT_TRUE(TestCodeSize > 0);
  EXPECT_EQ(TestLatency, RefLatency);
  EXPECT_TRUE(TestLatency > 0);
}
