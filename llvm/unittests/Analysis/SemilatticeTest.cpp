//===- SemilatticeTest.cpp - Semilattice tests ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Semilattice.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
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

class SemilatticeTest : public ::testing::Test {
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

TEST_F(SemilatticeTest, BasicConstruction) {
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
  Semilattice Lat(*F);
  Argument *ArgN = &*F->arg_begin();
  EXPECT_TRUE(Lat.contains(ArgN));
  EXPECT_TRUE(Lat.contains(PhiCounter));
  EXPECT_TRUE(Lat.contains(PhiResult));
  EXPECT_TRUE(Lat.contains(Counter));
  EXPECT_TRUE(Lat.contains(Result));
  EXPECT_TRUE(Lat.contains(NextCounter));
  EXPECT_TRUE(Lat.contains(Cond));
  EXPECT_EQ(Lat.size(), 7u);
  EXPECT_FALSE(Lat.lookup(PhiCounter)->isLeaf());
  EXPECT_FALSE(Lat.lookup(PhiResult)->isLeaf());
  EXPECT_FALSE(Lat.lookup(Counter)->isLeaf());
  EXPECT_FALSE(Lat.lookup(Result)->isLeaf());
  EXPECT_FALSE(Lat.lookup(NextCounter)->isLeaf());
  EXPECT_TRUE(Lat.lookup(Cond)->isLeaf());
}

TEST_F(SemilatticeTest, ConstructionWithIntAndPtr) {
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
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *IntArg = &*ArgIt++;
  Argument *FloatArg = &*ArgIt++;
  Argument *PtrArg = &*ArgIt++;
  Argument *VecIntArg = &*ArgIt;
  Argument *VecPtrArg = &*ArgIt;
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

  EXPECT_TRUE(Lat.contains(IntArg));
  EXPECT_FALSE(Lat.contains(FloatArg));
  EXPECT_TRUE(Lat.contains(PtrArg));
  EXPECT_TRUE(Lat.contains(VecIntArg));
  EXPECT_TRUE(Lat.contains(VecPtrArg));

  EXPECT_TRUE(Lat.contains(IntVal));
  EXPECT_TRUE(Lat.contains(IntVal2));
  EXPECT_TRUE(Lat.contains(PhiInt));
  EXPECT_TRUE(Lat.contains(FinalInt));
  EXPECT_TRUE(Lat.contains(VecVal));
  EXPECT_TRUE(Lat.contains(VecVal2));
  EXPECT_TRUE(Lat.contains(PhiVec));
  EXPECT_TRUE(Lat.contains(FPConv));
  EXPECT_TRUE(Lat.contains(VecPtrConv));
  EXPECT_TRUE(Lat.contains(FinalVec));
  EXPECT_FALSE(Lat.contains(FloatVal));
  EXPECT_TRUE(Lat.contains(PtrVal));
  EXPECT_FALSE(Lat.contains(PhiFloat));
  EXPECT_TRUE(Lat.contains(PhiPtr));

  EXPECT_EQ(Lat.size(), 16u);
}

TEST_F(SemilatticeTest, ConstructionWithNestedLoop) {
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
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgM = &*ArgIt;
  Instruction *OuterPHI = findInstructionByName(F, "outer_phi");
  Instruction *InnerPHI = findInstructionByName(F, "inner_phi");
  Instruction *InnerNext = findInstructionByName(F, "inner_next");
  Instruction *OuterNext = findInstructionByName(F, "outer_next");
  Instruction *InnerCond = findInstructionByName(F, "inner_cond");
  Instruction *OuterCond = findInstructionByName(F, "outer_cond");

  EXPECT_TRUE(Lat.contains(ArgN));
  EXPECT_TRUE(Lat.contains(ArgM));
  EXPECT_TRUE(Lat.contains(OuterPHI));
  EXPECT_TRUE(Lat.contains(InnerPHI));
  EXPECT_TRUE(Lat.contains(InnerNext));
  EXPECT_TRUE(Lat.contains(OuterNext));
  EXPECT_TRUE(Lat.contains(InnerCond));
  EXPECT_TRUE(Lat.contains(OuterCond));
  EXPECT_EQ(Lat.size(), 8u);
}

TEST_F(SemilatticeTest, Iteration) {
  parseAssembly(R"(
define void @test(i32 %arg1, i32 %arg2) {
  %add1 = add i32 %arg1, 1
  %add2 = add i32 %arg2, 2
  %mul1 = mul i32 %add1, 3
  %mul2 = mul i32 %add2, 4
  %final = add i32 %mul1, %mul2
  ret void
})");
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *Arg1 = &*ArgIt++;
  Argument *Arg2 = &*ArgIt;
  Instruction *Add1 = findInstructionByName(F, "add1");
  Instruction *Mul1 = findInstructionByName(F, "mul1");
  Instruction *Final = findInstructionByName(F, "final");

  SemilatticeNode *RootNode = Lat.getSentinelRoot();
  EXPECT_FALSE(RootNode->isLeaf());
  SmallVector<SemilatticeNode *> RootChildren(RootNode->children());
  EXPECT_EQ(RootChildren.size(), 2u);
  EXPECT_EQ(RootChildren[0]->getValue(), Arg1);
  EXPECT_EQ(RootChildren[1]->getValue(), Arg2);

  SemilatticeNode *Arg1Node = Lat.lookup(Arg1);
  EXPECT_FALSE(Arg1Node->isLeaf());
  SmallVector<SemilatticeNode *> Arg1Children(Arg1Node->children());
  EXPECT_EQ(Arg1Children.size(), 1u);
  EXPECT_EQ(Arg1Children[0]->getValue(), Add1);

  SemilatticeNode *Add1Node = Lat.lookup(Add1);
  EXPECT_FALSE(Add1Node->isLeaf());
  SmallVector<SemilatticeNode *> Add1Children(Add1Node->children());
  EXPECT_EQ(Add1Children.size(), 1u);
  EXPECT_EQ(Add1Children[0]->getValue(), Mul1);

  SemilatticeNode *Mul1Node = Lat.lookup(Mul1);
  EXPECT_FALSE(Mul1Node->isLeaf());
  SmallVector<SemilatticeNode *> Mul1Children(Mul1Node->children());
  EXPECT_EQ(Mul1Children.size(), 1u);
  EXPECT_EQ(Mul1Children[0]->getValue(), Final);

  SemilatticeNode *FinalNode = Lat.lookup(Final);
  EXPECT_TRUE(FinalNode->isLeaf());
  SmallVector<SemilatticeNode *> FinalChildren(FinalNode->children());
  EXPECT_EQ(FinalChildren.size(), 0u);

  SmallVector<SemilatticeNode *> DepthFirstOrder(depth_first(RootNode));
  EXPECT_GE(DepthFirstOrder.size(), 7u);
  EXPECT_EQ(DepthFirstOrder[0], RootNode);

  auto *RootPos = find(DepthFirstOrder, RootNode);
  auto *Arg1NodePos = find(DepthFirstOrder, Arg1Node);
  auto *Add1NodePos = find(DepthFirstOrder, Add1Node);
  auto *Mul1NodePos = find(DepthFirstOrder, Mul1Node);
  auto *FinalNodePos = find(DepthFirstOrder, FinalNode);

  EXPECT_NE(RootPos, DepthFirstOrder.end());
  EXPECT_NE(Arg1NodePos, DepthFirstOrder.end());
  EXPECT_NE(Add1NodePos, DepthFirstOrder.end());
  EXPECT_NE(Mul1NodePos, DepthFirstOrder.end());
  EXPECT_NE(FinalNodePos, DepthFirstOrder.end());
  EXPECT_LT(RootPos, Arg1NodePos);
  EXPECT_LT(Arg1NodePos, Add1NodePos);
  EXPECT_LT(Add1NodePos, Mul1NodePos);
  EXPECT_LT(Mul1NodePos, FinalNodePos);
}

TEST_F(SemilatticeTest, InvalidateKnownBitsSingleBB) {
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
  Semilattice Lat(*F);
  Argument *Arg = &*F->arg_begin();
  Argument *VecArg = &*F->arg_begin();
  KnownBits Known32(32);
  KnownBits KnownVec(DL.getTypeSizeInBits(VecArg->getType()));
  Known32.setAllOnes();
  KnownVec.setAllOnes();

  Lat.updateKnownBits(Arg, Known32);
  Lat.updateKnownBits(Counter, Known32);
  Lat.updateKnownBits(Result, Known32);
  Lat.updateKnownBits(NextCounter, Known32);
  Lat.updateKnownBits(BranchVal, Known32);
  Lat.updateKnownBits(MergeVal, Known32);
  Lat.updateKnownBits(VecArg, KnownVec);

  EXPECT_TRUE(Lat.getKnownBits(Arg).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(VecArg).isAllOnes());

  SmallVector<SemilatticeNode *> InvalidatedNodes =
      Lat.invalidateKnownBits(Counter);
  EXPECT_THAT(InvalidatedNodes, ::testing::ElementsAre(
                                    Lat.lookup(MergeVal), Lat.lookup(BranchVal),
                                    Lat.lookup(NextCounter), Lat.lookup(Result),
                                    Lat.lookup(Counter)));

  EXPECT_TRUE(Lat.getKnownBits(Arg).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(VecArg).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isUnknown());
}

TEST_F(SemilatticeTest, InvalidateKnownBitsMultipleBBs) {
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
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgCond = &*ArgIt;
  KnownBits Known32(32);
  Known32.setAllOnes();
  KnownBits Known1(1);
  Known1.setAllOnes();

  Lat.updateKnownBits(ArgN, Known32);
  Lat.updateKnownBits(ArgCond, Known1);
  Lat.updateKnownBits(Counter, Known32);
  Lat.updateKnownBits(BranchVal, Known32);
  Lat.updateKnownBits(Result, Known32);
  Lat.updateKnownBits(MergeVal, Known32);
  Lat.updateKnownBits(NextCounter, Known32);

  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isAllOnes());

  SmallVector<SemilatticeNode *> InvalidatedNodes =
      Lat.invalidateKnownBits(Result);
  EXPECT_THAT(InvalidatedNodes,
              ::testing::ElementsAre(Lat.lookup(NextCounter),
                                     Lat.lookup(MergeVal), Lat.lookup(Result)));

  EXPECT_TRUE(Lat.getKnownBits(ArgN).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(ArgCond).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
}

TEST_F(SemilatticeTest, RauwWithConstant) {
  parseAssembly(R"(
define void @test(i32 %arg1, i32 %arg2) {
  %counter = add i32 %arg1, 1
  %result = mul i32 %counter, 2
  %next_counter = add i32 %result, %arg2
  store i32 %next_counter, ptr poison
  ret void
})");
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *Arg1 = &*ArgIt++;
  Argument *Arg2 = &*ArgIt;
  KnownBits Known32(32);
  Known32.setAllOnes();

  Lat.updateKnownBits(Arg1, Known32);
  Lat.updateKnownBits(Arg2, Known32);
  Lat.updateKnownBits(Counter, Known32);
  Lat.updateKnownBits(Result, Known32);
  Lat.updateKnownBits(NextCounter, Known32);

  EXPECT_TRUE(Lat.getKnownBits(Arg1).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Arg2).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isAllOnes());

  EXPECT_EQ(Lat.lookup(Counter)->getNumParents(), 1u);
  EXPECT_EQ(Lat.lookup(Counter)->getNumChildren(), 1u);

  SmallVector<SemilatticeNode *> InvalidatedNodes =
      Lat.rauw(Arg1, ConstantInt::get(Context, APInt::getZero(32)));
  EXPECT_THAT(InvalidatedNodes,
              ::testing::ElementsAre(Lat.lookup(NextCounter),
                                     Lat.lookup(Result), Lat.lookup(Counter)));

  EXPECT_EQ(Lat.lookup(Counter)->getNumParents(), 0u);
  EXPECT_EQ(Lat.lookup(Counter)->getNumChildren(), 1u);

  EXPECT_FALSE(Lat.contains(Arg1));
  EXPECT_TRUE(Lat.getKnownBits(Arg2).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
}

TEST_F(SemilatticeTest, RauwWithVariable) {
  parseAssembly(R"(
define void @test(i32 %n, i1 %cond) {
entry:
  br i1 %cond, label %then, label %else
then:
  %branch_val = add i32 %n, 5
  br label %merge
else:
  %result = mul i32 %n, 3
  br label %merge
merge:
  %merge_val = phi i32 [ %branch_val, %then ], [ %result, %else ]
  %counter = add i32 %merge_val, 1
  %next_counter = mul i32 %counter, 2
  store i32 %next_counter, ptr poison
  ret void
})");
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *ArgN = &*ArgIt++;
  Argument *ArgCond = &*ArgIt;
  KnownBits Known32(32);
  Known32.setAllOnes();
  KnownBits Known1(1);
  Known1.setAllOnes();

  Lat.updateKnownBits(ArgN, Known32);
  Lat.updateKnownBits(ArgCond, Known1);
  Lat.updateKnownBits(BranchVal, Known32);
  Lat.updateKnownBits(Result, Known32);
  Lat.updateKnownBits(MergeVal, Known32);
  Lat.updateKnownBits(Counter, Known32);
  Lat.updateKnownBits(NextCounter, Known32);

  EXPECT_TRUE(Lat.getKnownBits(ArgN).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(ArgCond).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isAllOnes());

  EXPECT_EQ(Lat.lookup(MergeVal)->getNumParents(), 2u);
  EXPECT_EQ(Lat.lookup(MergeVal)->getNumChildren(), 1u);

  SmallVector<SemilatticeNode *> InvalidatedNodes =
      Lat.rauw(Result, Result->getOperand(0));
  EXPECT_THAT(InvalidatedNodes,
              ::testing::ElementsAre(Lat.lookup(NextCounter),
                                     Lat.lookup(Counter), Lat.lookup(MergeVal),
                                     Lat.lookup(ArgN)));

  EXPECT_EQ(Lat.lookup(MergeVal)->getNumParents(), 2u);
  EXPECT_EQ(Lat.lookup(MergeVal)->getNumChildren(), 1u);

  EXPECT_FALSE(Lat.contains(Result));
  EXPECT_TRUE(Lat.getKnownBits(ArgN).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(ArgCond).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
}

TEST_F(SemilatticeTest, Print) {
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
  Semilattice Lat(*F);
  KnownBits Known32(32);
  Known32.setAllZero();
  Instruction *Result = findInstructionByName(F, "result");
  Lat.updateKnownBits(Result, Known32);
  std::string ActualOutput;
  raw_string_ostream OS(ActualOutput);
  Lat.print(OS);
  std::string ExpectedOutput =
      R"(^ i32 %n
$   %cond = icmp slt i32 %next_counter, %n
^   %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]
    %counter = add i32 %phi_counter, 1
    %result = mul i32 %counter, 2 | 00000000000000000000000000000000
    %next_counter = add i32 %result, 1
)";
  EXPECT_EQ(ActualOutput, ExpectedOutput);
}
} // namespace
