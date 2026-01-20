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
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
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
    ASSERT_TRUE(M);
    F = M->getFunction("test");
    ASSERT_TRUE(F) << "Test must have a function @test";
    if (!F)
      return;
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
  Function *createSimpleFunction(StringRef Name,
                                 ArrayRef<Type *> ArgTypes = {}) {
    std::vector<Type *> Types(ArgTypes.begin(), ArgTypes.end());
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Context), Types, false);
    return Function::Create(FTy, Function::ExternalLinkage, Name, M.get());
  }
  LLVMContext Context;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  Instruction *Counter = nullptr, *NextCounter = nullptr, *Result = nullptr,
              *PhiCounter = nullptr, *PhiResult = nullptr, *Cond = nullptr,
              *BranchVal = nullptr, *MergeVal = nullptr;
};

TEST_F(SemilatticeTest, BasicConstruction) {
  parseAssembly(
      "define void @test(i32 %n) {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]\n"
      "  %phi_result = phi i32 [ 1, %entry ], [ %result, %loop ]\n"
      "  %counter = add i32 %phi_counter, 1\n"
      "  %result = mul i32 %phi_result, 2\n"
      "  %next_counter = add i32 %counter, 1\n"
      "  %cond = icmp slt i32 %next_counter, %n\n"
      "  br i1 %cond, label %loop, label %exit\n"
      "exit:\n"
      "  store i32 %result, ptr poison\n"
      "  ret void\n"
      "}\n");
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

TEST_F(SemilatticeTest, ConstructionNonIntegralExcluded) {
  parseAssembly(
      "define void @test(i32 %int_arg, float %float_arg, ptr %ptr_arg, <2 x "
      "i32> %vec_arg) {\n"
      "entry:\n"
      "  br i1 poison, label %then, label %else\n"
      "then:\n"
      "  %int_val = add i32 %int_arg, 1\n"
      "  %float_val = fadd float %float_arg, 1.0\n"
      "  %vec_val = add <2 x i32> %vec_arg, <i32 1, i32 2>\n"
      "  br label %merge\n"
      "else:\n"
      "  %int_val2 = mul i32 %int_arg, 2\n"
      "  %ptr_val = getelementptr i8, ptr %ptr_arg, i32 4\n"
      "  %vec_val2 = mul <2 x i32> %vec_arg, <i32 3, i32 4>\n"
      "  br label %merge\n"
      "merge:\n"
      "  %phi_int = phi i32 [ %int_val, %then ], [ %int_val2, %else ]\n"
      "  %phi_float = phi float [ %float_val, %then ], [ %float_arg, %else ]\n"
      "  %phi_ptr = phi ptr [ %ptr_arg, %then ], [ %ptr_val, %else ]\n"
      "  %phi_vec = phi <2 x i32> [ %vec_val, %then ], [ %vec_val2, %else ]\n"
      "  %final_int = add i32 %phi_int, 5\n"
      "  %final_vec = add <2 x i32> %phi_vec, <i32 5, i32 6>\n"
      "  store float %phi_float, ptr %phi_ptr\n"
      "  ret void\n"
      "}\n");
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *IntArg = &*ArgIt++;
  Argument *FloatArg = &*ArgIt++;
  Argument *PtrArg = &*ArgIt++;
  Argument *VecArg = &*ArgIt;
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
  Instruction *FinalVec = findInstructionByName(F, "final_vec");

  EXPECT_TRUE(Lat.contains(IntArg));
  EXPECT_TRUE(Lat.contains(IntVal));
  EXPECT_TRUE(Lat.contains(IntVal2));
  EXPECT_TRUE(Lat.contains(PhiInt));
  EXPECT_TRUE(Lat.contains(FinalInt));

  EXPECT_TRUE(Lat.contains(VecArg));
  EXPECT_TRUE(Lat.contains(VecVal));
  EXPECT_TRUE(Lat.contains(VecVal2));
  EXPECT_TRUE(Lat.contains(PhiVec));
  EXPECT_TRUE(Lat.contains(FinalVec));

  EXPECT_FALSE(Lat.contains(FloatArg));
  EXPECT_FALSE(Lat.contains(PtrArg));
  EXPECT_FALSE(Lat.contains(FloatVal));
  EXPECT_FALSE(Lat.contains(PtrVal));
  EXPECT_FALSE(Lat.contains(PhiFloat));
  EXPECT_FALSE(Lat.contains(PhiPtr));

  EXPECT_EQ(Lat.size(), 10u);
}

TEST_F(SemilatticeTest, Iteration) {
  parseAssembly("define void @test(i32 %arg1, i32 %arg2) {\n"
                "  %add1 = add i32 %arg1, 1\n"
                "  %add2 = add i32 %arg2, 2\n"
                "  %mul1 = mul i32 %add1, 3\n"
                "  %mul2 = mul i32 %add2, 4\n"
                "  %final = add i32 %mul1, %mul2\n"
                "  ret void\n"
                "}\n");
  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *Arg1 = &*ArgIt++;
  Argument *Arg2 = &*ArgIt;
  Instruction *Add1 = findInstructionByName(F, "add1");
  Instruction *Mul1 = findInstructionByName(F, "mul1");
  Instruction *Final = findInstructionByName(F, "final");

  SemilatticeNode *RootNode = Lat.getRootNode();
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

TEST_F(SemilatticeTest, NestedLoop) {
  parseAssembly(
      "define void @test(i32 %n, i32 %m) {\n"
      "entry:\n"
      "  br label %outer_loop\n"
      "outer_loop:\n"
      "  %outer_phi = phi i32 [ 0, %entry ], [ %outer_next, %outer_latch ]\n"
      "  br label %inner_loop\n"
      "inner_loop:\n"
      "  %inner_phi = phi i32 [ 0, %outer_loop ], [ %inner_next, %inner_loop "
      "]\n"
      "  %inner_next = add i32 %inner_phi, 1\n"
      "  %inner_cond = icmp slt i32 %inner_next, %m\n"
      "  br i1 %inner_cond, label %inner_loop, label %outer_latch\n"
      "outer_latch:\n"
      "  %outer_next = add i32 %outer_phi, 1\n"
      "  %outer_cond = icmp slt i32 %outer_next, %n\n"
      "  br i1 %outer_cond, label %outer_loop, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}\n");
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

TEST_F(SemilatticeTest, InvalidateKnownBitsSubgraph) {
  parseAssembly("define void @test(i32 %arg) {\n"
                "  %counter = add i32 %arg, 1\n"
                "  %result = mul i32 %counter, 2\n"
                "  %next_counter = add i32 %result, 3\n"
                "  %branch_val = sub i32 %next_counter, 1\n"
                "  %merge_val = add i32 %branch_val, 5\n"
                "  store i32 %merge_val, ptr poison\n"
                "  ret void\n"
                "}\n");
  Semilattice Lat(*F);
  Argument *Arg = &*F->arg_begin();
  KnownBits Known32(32);
  Known32.setAllOnes();

  Lat.updateKnownBits(Arg, Known32);
  Lat.updateKnownBits(Counter, Known32);
  Lat.updateKnownBits(Result, Known32);
  Lat.updateKnownBits(NextCounter, Known32);
  Lat.updateKnownBits(BranchVal, Known32);
  Lat.updateKnownBits(MergeVal, Known32);

  EXPECT_TRUE(Lat.getKnownBits(Arg).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isAllOnes());

  SmallVector<SemilatticeNode *> InvalidatedNodes =
      Lat.invalidateKnownBits(Counter);
  EXPECT_TRUE(Lat.getKnownBits(Arg).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isUnknown());
  EXPECT_EQ(InvalidatedNodes.size(), 5u);
}

TEST_F(SemilatticeTest, InvalidateKnownBitsPhiSubgraph) {
  parseAssembly(
      "define void @test(i32 %n, i1 %cond) {\n"
      "entry:\n"
      "  %counter = add i32 %n, 1\n"
      "  br i1 %cond, label %then, label %else\n"
      "then:\n"
      "  %branch_val = mul i32 %counter, 2\n"
      "  br label %merge\n"
      "else:\n"
      "  %result = add i32 %counter, 3\n"
      "  br label %merge\n"
      "merge:\n"
      "  %merge_val = phi i32 [ %branch_val, %then ], [ %result, %else ]\n"
      "  %next_counter = add i32 %merge_val, 1\n"
      "  store i32 %next_counter, ptr poison\n"
      "  ret void\n"
      "}\n");
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
      Lat.invalidateKnownBits(Counter);
  EXPECT_TRUE(Lat.getKnownBits(ArgN).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(ArgCond).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
  EXPECT_EQ(InvalidatedNodes.size(), 5u);
}

TEST_F(SemilatticeTest, RauwSubgraphInvalidation) {
  parseAssembly("define void @test(i32 %arg1, i32 %arg2) {\n"
                "  %counter = add i32 %arg1, 1\n"
                "  %result = mul i32 %counter, 2\n"
                "  %next_counter = add i32 %result, %arg2\n"
                "  store i32 %next_counter, ptr poison\n"
                "  ret void\n"
                "}\n");
  Semilattice Lat(*F);
  Function *F2 =
      createSimpleFunction("other_func", {Type::getInt32Ty(Context)});
  Argument *NewArg = &*F2->arg_begin();
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
  EXPECT_TRUE(Lat.getKnownBits(Counter).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(Result).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isAllOnes());
  EXPECT_TRUE(Lat.contains(Arg1));
  EXPECT_FALSE(Lat.contains(NewArg));

  SmallVector<SemilatticeNode *> InvalidatedNodes = Lat.rauw(Arg1, NewArg);
  EXPECT_FALSE(Lat.contains(Arg1));
  EXPECT_TRUE(Lat.contains(NewArg));
  EXPECT_TRUE(Lat.getKnownBits(NewArg).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
  EXPECT_EQ(InvalidatedNodes.size(), 4u);
}

TEST_F(SemilatticeTest, RauwPhiSubgraphInvalidation) {
  parseAssembly(
      "define void @test(i32 %n, i1 %cond) {\n"
      "entry:\n"
      "  br i1 %cond, label %then, label %else\n"
      "then:\n"
      "  %branch_val = add i32 %n, 5\n"
      "  br label %merge\n"
      "else:\n"
      "  %result = mul i32 %n, 3\n"
      "  br label %merge\n"
      "merge:\n"
      "  %merge_val = phi i32 [ %branch_val, %then ], [ %result, %else ]\n"
      "  %counter = add i32 %merge_val, 1\n"
      "  %next_counter = mul i32 %counter, 2\n"
      "  store i32 %next_counter, ptr poison\n"
      "  ret void\n"
      "}\n");
  Semilattice Lat(*F);
  Function *F2 =
      createSimpleFunction("other_func", {Type::getInt32Ty(Context)});
  Argument *NewArg = &*F2->arg_begin();
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
  EXPECT_TRUE(Lat.contains(ArgN));
  EXPECT_FALSE(Lat.contains(NewArg));

  SmallVector<SemilatticeNode *> InvalidatedNodes = Lat.rauw(ArgN, NewArg);
  EXPECT_FALSE(Lat.contains(ArgN));
  EXPECT_TRUE(Lat.contains(NewArg));
  EXPECT_TRUE(Lat.getKnownBits(ArgCond).isAllOnes());
  EXPECT_TRUE(Lat.getKnownBits(NewArg).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(BranchVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Result).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(MergeVal).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(Counter).isUnknown());
  EXPECT_TRUE(Lat.getKnownBits(NextCounter).isUnknown());
  EXPECT_EQ(InvalidatedNodes.size(), 6u);
}

TEST_F(SemilatticeTest, Print) {
  parseAssembly(
      "define void @test(i32 %n) {\n"
      "entry:\n"
      "  br label %loop\n"
      "loop:\n"
      "  %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]\n"
      "  %counter = add i32 %phi_counter, 1\n"
      "  %result = mul i32 %counter, 2\n"
      "  %next_counter = add i32 %result, 1\n"
      "  %cond = icmp slt i32 %next_counter, %n\n"
      "  br i1 %cond, label %loop, label %exit\n"
      "exit:\n"
      "  ret void\n"
      "}\n");
  Semilattice Lat(*F);
  KnownBits Known32(32);
  Known32.setAllZero();
  Instruction *Result = findInstructionByName(F, "result");
  Lat.updateKnownBits(Result, Known32);
  std::string ActualOutput;
  raw_string_ostream OS(ActualOutput);
  Lat.print(OS);
  std::string ExpectedOutput =
      "^ i32 %n\n"
      "$   %cond = icmp slt i32 %next_counter, %n\n"
      "^   %phi_counter = phi i32 [ 0, %entry ], [ %next_counter, %loop ]\n"
      "    %counter = add i32 %phi_counter, 1\n"
      "    %result = mul i32 %counter, 2 | 00000000000000000000000000000000\n"
      "    %next_counter = add i32 %result, 1\n";
  EXPECT_EQ(ActualOutput, ExpectedOutput);
}
} // namespace
