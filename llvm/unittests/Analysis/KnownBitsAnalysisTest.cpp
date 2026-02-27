//===- KnownBitsAnalysisTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/KnownBitsAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
static Instruction *findInstructionByName(Function *F, StringRef Name) {
  for (Instruction &I : instructions(F))
    if (I.getName() == Name)
      return &I;
  return nullptr;
}

class KnownBitsAnalysisTest : public ::testing::Test {
protected:
  void parseAssembly(StringRef Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);
    ASSERT_TRUE(M) << "Test has invalid IR";
    if (!M)
      return;
    F = M->getFunction("test");
    ASSERT_TRUE(F) << "Test must have a function named 'test'";
    if (!F)
      return;
    DL = F->getDataLayout();
    Counter = findInstructionByName(F, "counter");
    NextCounter = findInstructionByName(F, "next_counter");
    Result = findInstructionByName(F, "result");
    PhiCounter = findInstructionByName(F, "phi_counter");
    PhiResult = findInstructionByName(F, "phi_result");
    Cond = findInstructionByName(F, "cond");
    BranchVal = findInstructionByName(F, "branch_val");
    MergeVal = findInstructionByName(F, "merge_val");
  }
  void SetUp() override { M = std::make_unique<Module>("test", Context); }
  LLVMContext Context;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  DataLayout DL;
  Instruction *Counter = nullptr, *NextCounter = nullptr, *Result = nullptr,
              *PhiCounter = nullptr, *PhiResult = nullptr, *Cond = nullptr,
              *BranchVal = nullptr, *MergeVal = nullptr;
};

/// KnownBitsDataflow tests follow.
struct KnownBitsDataflowForTest : public KnownBitsDataflow {
  KnownBitsDataflowForTest(Function &F) : KnownBitsDataflow(F) {}
  ArrayRef<const Value *> getRoots() const {
    return KnownBitsDataflow::getRoots();
  }
  SmallVector<const Value *> getLeaves() const {
    return KnownBitsDataflow::getLeaves();
  }
  void intersectWith(const Value *V, KnownBits Known) {
    return KnownBitsDataflow::intersectWith(V, Known);
  }
  void setAllZero(const Value *V) { return KnownBitsDataflow::setAllZero(V); }
  void setAllOnes(const Value *V) { return KnownBitsDataflow::setAllOnes(V); }
  bool isAllConflict(const Value *V) const {
    return KnownBitsDataflow::isAllConflict(V);
  }
  bool isAllOnes(const Value *V) const {
    KnownBits Known = at(V);
    return Known.Zero.isZero() && Known.One.isAllOnes();
  }
};

TEST_F(KnownBitsAnalysisTest, BasicConstruction) {
  parseAssembly(R"(
define void @test(i32 %n) {
entry:
  br label %loop
loop:
  %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]
  %phi_result = phi i32 [ 1, %entry ], [ %result, %loop ]
  %counter = add i32 %phi_counter, 1
  %result = mul i32 %phi_result, 2
  %next_counter = add i32 %counter, 1
  %cond = icmp slt i32 %next_counter, %n
  br i1 %cond, label %loop, label %exit
exit:
  store i32 %result, ptr poison
  ret void
})");
  KnownBitsDataflowForTest Lat(*F);
  Argument *ArgN = &*F->arg_begin();
  EXPECT_TRUE(Lat.isAllConflict(ArgN));
  EXPECT_TRUE(Lat.isAllConflict(PhiCounter));
  EXPECT_TRUE(Lat.isAllConflict(PhiResult));
  EXPECT_TRUE(Lat.isAllConflict(Counter));
  EXPECT_TRUE(Lat.isAllConflict(Result));
  EXPECT_TRUE(Lat.isAllConflict(NextCounter));
  EXPECT_TRUE(Lat.isAllConflict(Cond));
  EXPECT_EQ(Lat.size(), 7u);
}

TEST_F(KnownBitsAnalysisTest, ConstructionWithIntAndPtr) {
  parseAssembly(R"(
define void @test(i32 %int_arg, float %float_arg, ptr %ptr_arg, <2 x i32> %vec_int_arg, <2 x ptr> %vec_ptr_arg) {
entry:
  br i1 poison, label %then, label %else
then:
  %int_val = add i32 %int_arg, 1
  %float_val = fadd float %float_arg, 1.0
  %vec_val = add <2 x i32> %vec_int_arg, <i32 1, i32 2>
  br label %merge
else:
  %fpconv = fptoui float %float_arg to i32
  %int_val2 = mul i32 %int_arg, %fpconv
  %ptr_val = getelementptr i8, ptr %ptr_arg, i32 4
  %vec_val2 = mul <2 x i32> %vec_int_arg, <i32 3, i32 4>
  br label %merge
merge:
  %phi_int = phi i32 [ %int_val, %then ], [ %int_val2, %else ]
  %phi_float = phi float [ %float_val, %then ], [ %float_arg, %else ]
  %phi_ptr = phi ptr [ %ptr_arg, %then ], [ %ptr_val, %else ]
  %phi_vec = phi <2 x i32> [ %vec_val, %then ], [ %vec_val2, %else ]
  %final_int = add i32 %phi_int, 5
  %vec_ptr_conv = ptrtoint <2 x ptr> %vec_ptr_arg to <2 x i32>
  %final_vec = add <2 x i32> %phi_vec, %vec_ptr_conv
  store float %phi_float, ptr %phi_ptr
  ret void
})");
  KnownBitsDataflowForTest Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *IntArg = &*ArgIt++;
  Argument *FloatArg = &*ArgIt++;
  Argument *PtrArg = &*ArgIt++;
  Argument *VecIntArg = &*ArgIt++;
  Argument *VecPtrArg = &*ArgIt++;
  Instruction *IntVal = findInstructionByName(F, "int_val");
  Instruction *FloatVal = findInstructionByName(F, "float_val");
  Instruction *VecVal = findInstructionByName(F, "vec_val");
  Instruction *IntVal2 = findInstructionByName(F, "int_val2");
  Instruction *PtrVal = findInstructionByName(F, "ptr_val");
  Instruction *VecVal2 = findInstructionByName(F, "vec_val2");
  Instruction *PhiInt = findInstructionByName(F, "phi_int");
  Instruction *PhiFloat = findInstructionByName(F, "phi_float");
  Instruction *PhiPtr = findInstructionByName(F, "phi_ptr");
  Instruction *PhiVec = findInstructionByName(F, "phi_vec");
  Instruction *FinalInt = findInstructionByName(F, "final_int");
  Instruction *FPConv = findInstructionByName(F, "fpconv");
  Instruction *VecPtrConv = findInstructionByName(F, "vec_ptr_conv");
  Instruction *FinalVec = findInstructionByName(F, "final_vec");

  EXPECT_THAT(Lat.getRoots(), ::testing::ElementsAre(IntArg, PtrArg, VecIntArg,
                                                     VecPtrArg, FPConv));
  EXPECT_THAT(Lat.getLeaves(),
              ::testing::ElementsAre(FinalInt, PhiPtr, FinalVec));

  EXPECT_TRUE(Lat.isAllConflict(IntArg));
  EXPECT_FALSE(Lat.contains(FloatArg));
  EXPECT_TRUE(Lat.isAllConflict(PtrArg));
  EXPECT_TRUE(Lat.isAllConflict(VecIntArg));
  EXPECT_TRUE(Lat.isAllConflict(VecPtrArg));

  EXPECT_TRUE(Lat.isAllConflict(IntVal));
  EXPECT_TRUE(Lat.isAllConflict(IntVal2));
  EXPECT_TRUE(Lat.isAllConflict(PhiInt));
  EXPECT_TRUE(Lat.isAllConflict(FinalInt));
  EXPECT_TRUE(Lat.isAllConflict(VecVal));
  EXPECT_TRUE(Lat.isAllConflict(VecVal2));
  EXPECT_TRUE(Lat.isAllConflict(PhiVec));
  EXPECT_TRUE(Lat.isAllConflict(FPConv));
  EXPECT_TRUE(Lat.isAllConflict(VecPtrConv));
  EXPECT_TRUE(Lat.isAllConflict(FinalVec));
  EXPECT_FALSE(Lat.contains(FloatVal));
  EXPECT_TRUE(Lat.isAllConflict(PtrVal));
  EXPECT_FALSE(Lat.contains(PhiFloat));
  EXPECT_TRUE(Lat.isAllConflict(PhiPtr));

  EXPECT_EQ(Lat.size(), 16u);
}

TEST_F(KnownBitsAnalysisTest, ConstructionWithNestedLoop) {
  parseAssembly(R"(
define void @test(i32 %n, i32 %m) {
entry:
  br label %outer_loop
outer_loop:
  %outer_phi = phi i32 [ 0, %entry ], [ %outer_next, %outer_latch ]
  br label %inner_loop
inner_loop:
  %inner_phi = phi i32 [ 0, %outer_loop ], [ %inner_next, %inner_loop ]
  %inner_next = add i32 %inner_phi, 1
  %inner_cond = icmp slt i32 %inner_next, %m
  br i1 %inner_cond, label %inner_loop, label %outer_latch
outer_latch:
  %outer_next = add i32 %outer_phi, 1
  %outer_cond = icmp slt i32 %outer_next, %n
  br i1 %outer_cond, label %outer_loop, label %exit
exit:
  ret void
})");
  KnownBitsDataflowForTest Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgM = &*ArgIt;
  Instruction *OuterPHI = findInstructionByName(F, "outer_phi");
  Instruction *InnerPHI = findInstructionByName(F, "inner_phi");
  Instruction *InnerNext = findInstructionByName(F, "inner_next");
  Instruction *OuterNext = findInstructionByName(F, "outer_next");
  Instruction *InnerCond = findInstructionByName(F, "inner_cond");
  Instruction *OuterCond = findInstructionByName(F, "outer_cond");

  EXPECT_THAT(Lat.getRoots(),
              ::testing::ElementsAre(ArgN, ArgM, OuterPHI, InnerPHI));
  EXPECT_THAT(Lat.getLeaves(), ::testing::ElementsAre(OuterCond, InnerCond));

  EXPECT_TRUE(Lat.isAllConflict(ArgN));
  EXPECT_TRUE(Lat.isAllConflict(ArgM));
  EXPECT_TRUE(Lat.isAllConflict(OuterPHI));
  EXPECT_TRUE(Lat.isAllConflict(InnerPHI));
  EXPECT_TRUE(Lat.isAllConflict(InnerNext));
  EXPECT_TRUE(Lat.isAllConflict(OuterNext));
  EXPECT_TRUE(Lat.isAllConflict(InnerCond));
  EXPECT_TRUE(Lat.isAllConflict(OuterCond));
  EXPECT_EQ(Lat.size(), 8u);
}

TEST_F(KnownBitsAnalysisTest, InvalidateKnownBitsSingleBB) {
  parseAssembly(R"(
define void @test(i32 %arg, <2 x i32> %vec_arg) {
  %counter = add i32 %arg, 1
  %result = mul i32 %counter, 2
  %next_counter = add i32 %result, 3
  %branch_val = sub i32 %next_counter, 1
  %merge_val = add i32 %branch_val, 5
  store i32 %merge_val, ptr poison
  ret void
})");
  KnownBitsDataflowForTest Lat(*F);
  Argument *Arg = &*F->arg_begin();
  Argument *VecArg = &*F->arg_begin();
  KnownBits Known32(32);
  Known32.setAllOnes();

  Lat.intersectWith(Arg, Known32);
  Lat.intersectWith(Counter, Known32);
  Lat.intersectWith(Result, Known32);
  Lat.intersectWith(NextCounter, Known32);
  Lat.intersectWith(BranchVal, Known32);
  Lat.intersectWith(MergeVal, Known32);
  Lat.setAllOnes(VecArg);

  EXPECT_TRUE(Lat.isAllOnes(Arg));
  EXPECT_TRUE(Lat.isAllOnes(Counter));
  EXPECT_TRUE(Lat.isAllOnes(Result));
  EXPECT_TRUE(Lat.isAllOnes(NextCounter));
  EXPECT_TRUE(Lat.isAllOnes(BranchVal));
  EXPECT_TRUE(Lat.isAllOnes(MergeVal));
  EXPECT_TRUE(Lat.isAllOnes(VecArg));

  auto InvalidatedLeaves = Lat.invalidate(Counter);
  EXPECT_THAT(InvalidatedLeaves, ::testing::ElementsAre(MergeVal));

  EXPECT_TRUE(Lat.isAllOnes(Arg));
  EXPECT_TRUE(Lat.isAllOnes(VecArg));
  EXPECT_TRUE(Lat.isAllConflict(Counter));
  EXPECT_TRUE(Lat.isAllConflict(Result));
  EXPECT_TRUE(Lat.isAllConflict(NextCounter));
  EXPECT_TRUE(Lat.isAllConflict(BranchVal));
  EXPECT_TRUE(Lat.isAllConflict(MergeVal));
}

TEST_F(KnownBitsAnalysisTest, InvalidateKnownBitsMultipleBBs) {
  parseAssembly(R"(
define void @test(i32 %n, i1 %cond) {
entry:
  %counter = add i32 %n, 1
  br i1 %cond, label %then, label %else
then:
  %branch_val = mul i32 %counter, 2
  br label %merge
else:
  %result = add i32 %counter, 3
  br label %merge
merge:
  %merge_val = phi i32 [ %branch_val, %then ], [ %result, %else ]
  %next_counter = add i32 %merge_val, 1
  store i32 %next_counter, ptr poison
  ret void
})");
  KnownBitsDataflowForTest Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgCond = &*ArgIt;
  KnownBits Known32(32);
  Known32.setAllOnes();

  Lat.intersectWith(ArgN, Known32);
  Lat.intersectWith(Counter, Known32);
  Lat.intersectWith(BranchVal, Known32);
  Lat.intersectWith(Result, Known32);
  Lat.intersectWith(MergeVal, Known32);
  Lat.intersectWith(NextCounter, Known32);
  Lat.setAllOnes(ArgCond);

  EXPECT_TRUE(Lat.isAllOnes(Counter));
  EXPECT_TRUE(Lat.isAllOnes(BranchVal));
  EXPECT_TRUE(Lat.isAllOnes(Result));
  EXPECT_TRUE(Lat.isAllOnes(MergeVal));
  EXPECT_TRUE(Lat.isAllOnes(NextCounter));

  auto InvalidatedLeaves = Lat.invalidate(Result);
  EXPECT_THAT(InvalidatedLeaves, ::testing::ElementsAre(NextCounter));

  EXPECT_TRUE(Lat.isAllOnes(ArgN));
  EXPECT_TRUE(Lat.isAllOnes(ArgCond));
  EXPECT_TRUE(Lat.isAllOnes(Counter));
  EXPECT_TRUE(Lat.isAllOnes(BranchVal));
  EXPECT_TRUE(Lat.isAllConflict(Result));
  EXPECT_TRUE(Lat.isAllConflict(MergeVal));
  EXPECT_TRUE(Lat.isAllConflict(NextCounter));
}

TEST_F(KnownBitsAnalysisTest, Print) {
  parseAssembly(R"(
define void @test(i32 %n) {
entry:
  br label %loop
loop:
  %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]
  %counter = add i32 %phi_counter, 1
  %result = mul i32 %counter, 2
  %next_counter = add i32 %result, 1
  %cond = icmp slt i32 %next_counter, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret void
})");
  KnownBitsDataflowForTest Lat(*F);
  Instruction *Result = findInstructionByName(F, "result");
  Lat.setAllZero(Result);
  std::string ActualOutput;
  raw_string_ostream OS(ActualOutput);
  Lat.print(OS);
  std::string ExpectedOutput =
      R"(^ i32 %n | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
$   %cond = icmp slt i32 %next_counter, %n | !
^   %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ] | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    %counter = add i32 %phi_counter, 1 | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    %result = mul i32 %counter, 2 | 00000000000000000000000000000000
    %next_counter = add i32 %result, 1 | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
$   %cond = icmp slt i32 %next_counter, %n | !
)";
  EXPECT_EQ(ActualOutput, ExpectedOutput);
}

/// KnownBitsCache tests follow.
struct KnownBitsCacheForTest : public KnownBitsCache {
  KnownBitsCacheForTest(Function &F) : KnownBitsCache(F) {}
  KnownBits at(const Value *V) const { return KnownBitsCache::at(V); }
};

TEST_F(KnownBitsAnalysisTest, KnownBitsCacheLeavesInitialization) {
  parseAssembly(R"(
define void @test(i32 %int_arg, float %float_arg, ptr %ptr_arg, <2 x i32> %vec_int_arg, <2 x ptr> %vec_ptr_arg) {
entry:
  br i1 poison, label %then, label %else
then:
  %int_val = add i32 %int_arg, 1
  %float_val = fadd float %float_arg, 1.0
  %vec_val = add <2 x i32> %vec_int_arg, <i32 1, i32 2>
  br label %merge
else:
  %fpconv = fptoui float %float_arg to i32
  %int_val2 = mul i32 %int_arg, %fpconv
  %ptr_val = getelementptr i8, ptr %ptr_arg, i32 4
  %vec_val2 = mul <2 x i32> %vec_int_arg, <i32 3, i32 4>
  br label %merge
merge:
  %phi_int = phi i32 [ %int_val, %then ], [ %int_val2, %else ]
  %phi_float = phi float [ %float_val, %then ], [ %float_arg, %else ]
  %phi_ptr = phi ptr [ %ptr_arg, %then ], [ %ptr_val, %else ]
  %phi_vec = phi <2 x i32> [ %vec_val, %then ], [ %vec_val2, %else ]
  %final_int = add i32 %phi_int, 5
  %vec_ptr_conv = ptrtoint <2 x ptr> %vec_ptr_arg to <2 x i32>
  %final_vec = add <2 x i32> %phi_vec, %vec_ptr_conv
  store float %phi_float, ptr %phi_ptr
  ret void
})");
  KnownBitsCacheForTest Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *IntArg = &*ArgIt++;
  ArgIt++;
  Argument *PtrArg = &*ArgIt++;
  Argument *VecIntArg = &*ArgIt++;
  Argument *VecPtrArg = &*ArgIt++;
  Instruction *IntVal = findInstructionByName(F, "int_val");
  Instruction *VecVal = findInstructionByName(F, "vec_val");
  Instruction *IntVal2 = findInstructionByName(F, "int_val2");
  Instruction *PtrVal = findInstructionByName(F, "ptr_val");
  Instruction *VecVal2 = findInstructionByName(F, "vec_val2");
  Instruction *PhiInt = findInstructionByName(F, "phi_int");
  Instruction *PhiPtr = findInstructionByName(F, "phi_ptr");
  Instruction *PhiVec = findInstructionByName(F, "phi_vec");
  Instruction *FinalInt = findInstructionByName(F, "final_int");
  Instruction *FPConv = findInstructionByName(F, "fpconv");
  Instruction *VecPtrConv = findInstructionByName(F, "vec_ptr_conv");
  Instruction *FinalVec = findInstructionByName(F, "final_vec");

  EXPECT_EQ(Lat.at(IntArg), computeKnownBits(IntArg, DL));
  EXPECT_EQ(Lat.at(PtrArg), computeKnownBits(PtrArg, DL));
  EXPECT_EQ(Lat.at(VecIntArg), computeKnownBits(VecIntArg, DL));
  EXPECT_EQ(Lat.at(VecPtrArg), computeKnownBits(VecPtrArg, DL));
  EXPECT_EQ(Lat.at(IntVal), computeKnownBits(IntVal, DL));
  EXPECT_EQ(Lat.at(VecVal), computeKnownBits(VecVal, DL));
  EXPECT_EQ(Lat.at(IntVal2), computeKnownBits(IntVal2, DL));
  EXPECT_EQ(Lat.at(PtrVal), computeKnownBits(PtrVal, DL));
  EXPECT_EQ(Lat.at(VecVal2), computeKnownBits(VecVal2, DL));
  EXPECT_EQ(Lat.at(PhiInt), computeKnownBits(PhiInt, DL));
  EXPECT_EQ(Lat.at(PhiPtr), computeKnownBits(PhiPtr, DL));
  EXPECT_EQ(Lat.at(PhiVec), computeKnownBits(PhiVec, DL));
  EXPECT_EQ(Lat.at(FinalInt), computeKnownBits(FinalInt, DL));
  EXPECT_EQ(Lat.at(FPConv), computeKnownBits(FPConv, DL));
  EXPECT_EQ(Lat.at(VecPtrConv), computeKnownBits(VecPtrConv, DL));
  EXPECT_EQ(Lat.at(FinalVec), computeKnownBits(FinalVec, DL));
}
} // namespace
