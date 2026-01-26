//===- SemilatticeTest.cpp - Semilattice tests ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Semilattice.h"
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

    A = findInstructionByName(F, "A");
    A2 = findInstructionByName(F, "A2");
    A3 = findInstructionByName(F, "A3");
    A4 = findInstructionByName(F, "A4");
    A5 = findInstructionByName(F, "A5");
    A6 = findInstructionByName(F, "A6");
    A7 = findInstructionByName(F, "A7");
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
  Instruction *A = nullptr, *A2 = nullptr, *A3 = nullptr, *A4 = nullptr,
              *A5 = nullptr, *A6 = nullptr, *A7 = nullptr;
};

TEST_F(SemilatticeTest, SimplePrototype) {
  parseAssembly("define void @test() { ret void }");
  Semilattice EmptyLat(*F);
  EXPECT_TRUE(EmptyLat.empty());
  EXPECT_TRUE(EmptyLat.getRootNode()->isLeaf());
  EXPECT_TRUE(EmptyLat.getRootNode()->isRoot());
  EXPECT_EQ(EmptyLat.size(), 0u);

  parseAssembly("define void @test(i32 %arg) { ret void }");
  Semilattice SingleLat(*F);
  EXPECT_FALSE(SingleLat.empty());
  EXPECT_FALSE(SingleLat.getRootNode()->isLeaf());
  EXPECT_EQ(SingleLat.getKnownBits(&*F->arg_begin()).getBitWidth(), 32u);
}

TEST_F(SemilatticeTest, IntegerArguments) {
  parseAssembly("define void @test(i32 %arg1, i64 %arg2) { ret void }");
  Semilattice TwoArgLat(*F);
  EXPECT_FALSE(TwoArgLat.empty());
  auto *ArgIt = F->arg_begin();
  EXPECT_EQ(TwoArgLat.getKnownBits(&*ArgIt++).getBitWidth(), 32u);
  EXPECT_EQ(TwoArgLat.getKnownBits(&*ArgIt).getBitWidth(), 64u);

  parseAssembly("define void @test(i32 %intarg, float %floatarg) { ret void }");
  Semilattice MixedLat(*F);
  EXPECT_EQ(MixedLat.getKnownBits(&*F->arg_begin()).getBitWidth(), 32u);
}

TEST_F(SemilatticeTest, Iterators) {
  parseAssembly("define void @test() { ret void }");
  Semilattice EmptyLat(*F);
  SemilatticeNode *EmptyRoot = EmptyLat.getRootNode();
  EXPECT_TRUE(EmptyRoot->isLeaf());
  EXPECT_EQ(EmptyRoot->child_begin(), EmptyRoot->child_end());
  EXPECT_EQ(
      std::distance(EmptyRoot->children().begin(), EmptyRoot->children().end()),
      0u);

  parseAssembly("define void @test(i32 %arg) { ret void }");
  Semilattice SingleLat(*F);
  SemilatticeNode *Root = SingleLat.getRootNode();
  EXPECT_TRUE(Root->isRoot());
  EXPECT_FALSE(Root->isLeaf());

  auto *It = Root->child_begin();
  EXPECT_NE(It, Root->child_end());
  SemilatticeNode *Child = *It++;
  EXPECT_EQ(It, Root->child_end());
  EXPECT_FALSE(Child->isRoot());
  EXPECT_TRUE(Child->isLeaf());

  size_t ChildCount = 0;
  for (auto *C : Root->children()) {
    EXPECT_EQ(C, Child);
    ++ChildCount;
  }
  EXPECT_EQ(ChildCount, 1u);
}

TEST_F(SemilatticeTest, LookupAndGetKnownBits) {
  parseAssembly("define void @test(i32 %arg) { ret void }");

  Function *F2 =
      createSimpleFunction("other_func", {Type::getInt32Ty(Context)});
  Argument *OtherArg = &*F2->arg_begin();

  Semilattice Lat(*F);
  EXPECT_FALSE(Lat.contains(OtherArg));
  EXPECT_EQ(Lat.lookup(OtherArg), Lat.getRootNode());

  Argument *Arg = &*F->arg_begin();
  EXPECT_TRUE(Lat.getKnownBits(Arg).isUnknown());
  KnownBits NewKnown(Arg->getType()->getIntegerBitWidth());
  NewKnown.setAllOnes();
  Lat.updateKnownBits(Arg, NewKnown);
  EXPECT_TRUE(Lat.getKnownBits(Arg).isAllOnes());

  Lat.invalidateKnownBits(Arg);
  EXPECT_TRUE(Lat.getKnownBits(Arg).isUnknown());
}

TEST_F(SemilatticeTest, RuawAndReset) {
  parseAssembly("define void @test(i32 %arg1, i32 %arg2) { ret void }");

  Function *F1 = F;
  Semilattice Lat(*F1);
  EXPECT_FALSE(Lat.empty());
  EXPECT_EQ(Lat.size(), 2u);

  Function *F2 =
      createSimpleFunction("other_func", {Type::getInt32Ty(Context)});
  Argument *NewArg = &*F2->arg_begin();
  Argument *OldArg = &*F1->arg_begin();

  EXPECT_TRUE(Lat.contains(OldArg));

  Lat.rauw(OldArg, NewArg);
  EXPECT_FALSE(Lat.contains(OldArg));
  EXPECT_TRUE(Lat.contains(NewArg));

  Lat.reset(*F2);
  EXPECT_EQ(Lat.size(), 1u);
  EXPECT_FALSE(Lat.contains(&*F1->arg_begin()));
  EXPECT_TRUE(Lat.contains(&*F2->arg_begin()));
}

TEST_F(SemilatticeTest, TypeFilteringAndSizing) {
  parseAssembly("define void @test(i8 %arg1, float %arg2, i16 %arg3, double "
                "%arg4, i32 %arg5) {\n"
                "  ret void\n"
                "}\n");

  Semilattice MixedLat(*F);
  EXPECT_FALSE(MixedLat.empty());
  EXPECT_EQ(MixedLat.size(), 3u);

  auto *ArgIt = F->arg_begin();
  EXPECT_TRUE(MixedLat.contains(&*ArgIt++));
  EXPECT_FALSE(MixedLat.contains(&*ArgIt++));
  EXPECT_TRUE(MixedLat.contains(&*ArgIt++));
  EXPECT_FALSE(MixedLat.contains(&*ArgIt++));
  EXPECT_TRUE(MixedLat.contains(&*ArgIt));

  parseAssembly("define void @test(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, "
                "i32 %arg4) {\n"
                "  ret void\n"
                "}\n");

  Semilattice ManyArgLat(*F);
  EXPECT_EQ(ManyArgLat.size(), 5u);
  for (auto &Arg : F->args()) {
    EXPECT_TRUE(ManyArgLat.contains(&Arg));
    EXPECT_EQ(ManyArgLat.getKnownBits(&Arg).getBitWidth(), 32u);
  }
}

TEST_F(SemilatticeTest, LeafNodeAnalysis) {
  parseAssembly("define void @test(i32 %arg) {\n"
                "  %A = add i32 %arg, 42\n"
                "  store i32 %A, ptr undef\n"
                "  ret void\n"
                "}\n");

  Semilattice Lat(*F);
  EXPECT_TRUE(Lat.contains(A));
  EXPECT_TRUE(Lat.lookup(A)->isLeaf());
  EXPECT_EQ(Lat.getKnownBits(A).getBitWidth(), 32u);
}

TEST_F(SemilatticeTest, InstructionChain) {
  parseAssembly("define void @test(i32 %arg) {\n"
                "  %A = add i32 %arg, 42\n"
                "  %A2 = mul i32 %A, 2\n"
                "  %A3 = sub i32 %A2, 10\n"
                "  store i32 %A3, ptr undef\n"
                "  ret void\n"
                "}\n");

  Semilattice Lat(*F);
  EXPECT_TRUE(Lat.contains(A));
  EXPECT_TRUE(Lat.contains(A2));
  EXPECT_TRUE(Lat.contains(A3));

  EXPECT_FALSE(Lat.lookup(A)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A2)->isLeaf());
  EXPECT_TRUE(Lat.lookup(A3)->isLeaf());

  EXPECT_EQ(Lat.getKnownBits(A).getBitWidth(), 32u);
  EXPECT_EQ(Lat.getKnownBits(A2).getBitWidth(), 32u);
  EXPECT_EQ(Lat.getKnownBits(A3).getBitWidth(), 32u);
}

TEST_F(SemilatticeTest, LongInstructionChain) {
  parseAssembly("define void @test(i32 %arg) {\n"
                "  %A = add i32 %arg, 1\n"
                "  %A2 = add i32 %A, 2\n"
                "  %A3 = add i32 %A2, 3\n"
                "  %A4 = add i32 %A3, 4\n"
                "  %A5 = add i32 %A4, 5\n"
                "  %A6 = add i32 %A5, 6\n"
                "  %A7 = add i32 %A6, 7\n"
                "  store i32 %A7, ptr undef\n"
                "  ret void\n"
                "}\n");

  Semilattice Lat(*F);

  EXPECT_TRUE(Lat.contains(A));
  EXPECT_TRUE(Lat.contains(A2));
  EXPECT_TRUE(Lat.contains(A3));
  EXPECT_TRUE(Lat.contains(A4));
  EXPECT_TRUE(Lat.contains(A5));
  EXPECT_TRUE(Lat.contains(A6));
  EXPECT_TRUE(Lat.contains(A7));

  EXPECT_FALSE(Lat.lookup(A)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A2)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A3)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A4)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A5)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A6)->isLeaf());
  EXPECT_TRUE(Lat.lookup(A7)->isLeaf());

  EXPECT_EQ(Lat.getKnownBits(A7).getBitWidth(), 32u);
}

TEST_F(SemilatticeTest, ComplexDataflow) {
  parseAssembly("define void @test(i32 %arg1, i32 %arg2, i32 %arg3) {\n"
                "  %A = add i32 %arg1, %arg2\n"
                "  %A2 = mul i32 %arg1, %arg3\n"
                "  %A3 = add i32 %A, %A2\n"
                "  %A4 = sub i32 %A3, %arg2\n"
                "  store i32 %A4, ptr undef\n"
                "  ret void\n"
                "}\n");

  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *Arg1 = &*ArgIt++;
  Argument *Arg2 = &*ArgIt++;
  Argument *Arg3 = &*ArgIt;

  EXPECT_TRUE(Lat.contains(Arg1));
  EXPECT_TRUE(Lat.contains(Arg2));
  EXPECT_TRUE(Lat.contains(Arg3));
  EXPECT_TRUE(Lat.contains(A));
  EXPECT_TRUE(Lat.contains(A2));
  EXPECT_TRUE(Lat.contains(A3));
  EXPECT_TRUE(Lat.contains(A4));

  EXPECT_FALSE(Lat.lookup(Arg1)->isLeaf());
  EXPECT_FALSE(Lat.lookup(Arg2)->isLeaf());
  EXPECT_FALSE(Lat.lookup(Arg3)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A2)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A3)->isLeaf());
  EXPECT_TRUE(Lat.lookup(A4)->isLeaf());
}

TEST_F(SemilatticeTest, BranchAndPhiNodes) {
  parseAssembly("define void @test(i32 %arg, i1 %cond) {\n"
                "entry:\n"
                "  %A = add i32 %arg, 10\n"
                "  br i1 %cond, label %then, label %else\n"
                "then:\n"
                "  %A2 = mul i32 %A, 2\n"
                "  br label %merge\n"
                "else:\n"
                "  %A3 = add i32 %A, 5\n"
                "  br label %merge\n"
                "merge:\n"
                "  %A4 = phi i32 [ %A2, %then ], [ %A3, %else ]\n"
                "  store i32 %A4, ptr undef\n"
                "  ret void\n"
                "}\n");

  Semilattice Lat(*F);
  auto *ArgIt = F->arg_begin();
  Argument *IntArg = &*ArgIt++;
  Argument *CondArg = &*ArgIt;

  EXPECT_TRUE(Lat.contains(IntArg));
  EXPECT_TRUE(Lat.contains(CondArg));
  EXPECT_TRUE(Lat.contains(A));
  EXPECT_TRUE(Lat.contains(A2));
  EXPECT_TRUE(Lat.contains(A3));
  EXPECT_TRUE(Lat.contains(A4));

  EXPECT_FALSE(Lat.lookup(IntArg)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A2)->isLeaf());
  EXPECT_FALSE(Lat.lookup(A3)->isLeaf());
  EXPECT_TRUE(Lat.lookup(A4)->isLeaf());

  EXPECT_EQ(Lat.getKnownBits(A4).getBitWidth(), 32u);
}
} // namespace
