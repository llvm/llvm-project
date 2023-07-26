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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/FunctionSpecialization.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {

class FunctionSpecializationTest : public testing::Test {
protected:
  LLVMContext Ctx;
  FunctionAnalysisManager FAM;
  std::unique_ptr<Module> M;
  std::unique_ptr<SCCPSolver> Solver;

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

    return FunctionSpecializer(*Solver, *M, &FAM, GetBFI, GetTLI, GetTTI,
                               GetAC);
  }

  Cost getInstCost(Instruction &I) {
    auto &TTI = FAM.getResult<TargetIRAnalysis>(*I.getFunction());
    auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(*I.getFunction());

    return BFI.getBlockFreq(I.getParent()).getFrequency() / BFI.getEntryFreq() *
         TTI.getInstructionCost(&I, TargetTransformInfo::TCK_SizeAndLatency);
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
  ++FuncIter;
  BasicBlock &Case1 = *++FuncIter;
  BasicBlock &Case2 = *++FuncIter;
  BasicBlock &BB1 = *++FuncIter;
  BasicBlock &BB2 = *++FuncIter;

  Instruction &Mul = Case1.front();
  Instruction &And = Case2.front();
  Instruction &Sdiv = *++Case2.begin();
  Instruction &BrBB2 = Case2.back();
  Instruction &Add = BB1.front();
  Instruction &Or = BB2.front();
  Instruction &BrLoop = BB2.back();

  // mul
  Cost Ref = getInstCost(Mul);
  Cost Bonus = Specializer.getSpecializationBonus(F->getArg(0), One, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  // and + or + add
  Ref = getInstCost(And) + getInstCost(Or) + getInstCost(Add);
  Bonus = Specializer.getSpecializationBonus(F->getArg(1), One, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  // sdiv + br + br
  Ref = getInstCost(Sdiv) + getInstCost(BrBB2) + getInstCost(BrLoop);
  Bonus = Specializer.getSpecializationBonus(F->getArg(2), One, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);
}

TEST_F(FunctionSpecializationTest, BranchInst) {
  const char *ModuleString = R"(
    define void @foo(i32 %a, i32 %b, i1 %cond) {
    entry:
      br label %loop
    loop:
      br i1 %cond, label %bb0, label %bb2
    bb0:
      %0 = mul i32 %a, 2
      %1 = sub i32 6, 5
      br label %bb1
    bb1:
      %2 = add i32 %0, %b
      %3 = sdiv i32 8, 2
      br label %loop
    bb2:
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
  ++FuncIter;
  BasicBlock &BB0 = *++FuncIter;
  BasicBlock &BB1 = *++FuncIter;

  Instruction &Mul = BB0.front();
  Instruction &Sub = *++BB0.begin();
  Instruction &BrBB1 = BB0.back();
  Instruction &Add = BB1.front();
  Instruction &Sdiv = *++BB1.begin();
  Instruction &BrLoop = BB1.back();

  // mul
  Cost Ref = getInstCost(Mul);
  Cost Bonus = Specializer.getSpecializationBonus(F->getArg(0), One, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  // add
  Ref = getInstCost(Add);
  Bonus = Specializer.getSpecializationBonus(F->getArg(1), One, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  // sub + br + sdiv + br
  Ref = getInstCost(Sub) + getInstCost(BrBB1) + getInstCost(Sdiv) +
        getInstCost(BrLoop);
  Bonus = Specializer.getSpecializationBonus(F->getArg(2), False, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);
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
  Cost Ref = getInstCost(Icmp) + getInstCost(Zext);
  Cost Bonus = Specializer.getSpecializationBonus(F->getArg(0), One, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  // select
  Ref = getInstCost(Select);
  Bonus = Specializer.getSpecializationBonus(F->getArg(1), True, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  // gep + load + freeze + smax
  Ref = getInstCost(Gep) + getInstCost(Load) + getInstCost(Freeze) +
        getInstCost(Smax);
  Bonus = Specializer.getSpecializationBonus(F->getArg(2), GV, Visitor);
  EXPECT_EQ(Bonus, Ref);
  EXPECT_TRUE(Bonus > 0);

  Bonus = Specializer.getSpecializationBonus(F->getArg(3), Undef, Visitor);
  EXPECT_TRUE(Bonus == 0);
}
