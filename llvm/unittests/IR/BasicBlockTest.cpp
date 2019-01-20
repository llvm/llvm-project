//===- llvm/unittest/IR/BasicBlockTest.cpp - BasicBlock unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {
namespace {

TEST(BasicBlockTest, PhiRange) {
  LLVMContext Context;

  // Create the main block.
  std::unique_ptr<BasicBlock> BB(BasicBlock::Create(Context));

  // Create some predecessors of it.
  std::unique_ptr<BasicBlock> BB1(BasicBlock::Create(Context));
  BranchInst::Create(BB.get(), BB1.get());
  std::unique_ptr<BasicBlock> BB2(BasicBlock::Create(Context));
  BranchInst::Create(BB.get(), BB2.get());

  // Make sure this doesn't crash if there are no phis.
  for (auto &PN : BB->phis()) {
    (void)PN;
    EXPECT_TRUE(false) << "empty block should have no phis";
  }

  // Make it a cycle.
  auto *BI = BranchInst::Create(BB.get(), BB.get());

  // Now insert some PHI nodes.
  auto *Int32Ty = Type::getInt32Ty(Context);
  auto *P1 = PHINode::Create(Int32Ty, /*NumReservedValues*/ 3, "phi.1", BI);
  auto *P2 = PHINode::Create(Int32Ty, /*NumReservedValues*/ 3, "phi.2", BI);
  auto *P3 = PHINode::Create(Int32Ty, /*NumReservedValues*/ 3, "phi.3", BI);

  // Some non-PHI nodes.
  auto *Sum = BinaryOperator::CreateAdd(P1, P2, "sum", BI);

  // Now wire up the incoming values that are interesting.
  P1->addIncoming(P2, BB.get());
  P2->addIncoming(P1, BB.get());
  P3->addIncoming(Sum, BB.get());

  // Finally, let's iterate them, which is the thing we're trying to test.
  // We'll use this to wire up the rest of the incoming values.
  for (auto &PN : BB->phis()) {
    PN.addIncoming(UndefValue::get(Int32Ty), BB1.get());
    PN.addIncoming(UndefValue::get(Int32Ty), BB2.get());
  }

  // Test that we can use const iterators and generally that the iterators
  // behave like iterators.
  BasicBlock::const_phi_iterator CI;
  CI = BB->phis().begin();
  EXPECT_NE(CI, BB->phis().end());

  // Test that filtering iterators work with basic blocks.
  auto isPhi = [](Instruction &I) { return isa<PHINode>(&I); };
  auto Phis = make_filter_range(*BB, isPhi);
  auto ReversedPhis = reverse(make_filter_range(*BB, isPhi));
  EXPECT_EQ(std::distance(Phis.begin(), Phis.end()), 3);
  EXPECT_EQ(&*Phis.begin(), P1);
  EXPECT_EQ(std::distance(ReversedPhis.begin(), ReversedPhis.end()), 3);
  EXPECT_EQ(&*ReversedPhis.begin(), P3);

  // And iterate a const range.
  for (const auto &PN : const_cast<const BasicBlock *>(BB.get())->phis()) {
    EXPECT_EQ(BB.get(), PN.getIncomingBlock(0));
    EXPECT_EQ(BB1.get(), PN.getIncomingBlock(1));
    EXPECT_EQ(BB2.get(), PN.getIncomingBlock(2));
  }
}

#define CHECK_ITERATORS(Range1, Range2)                                        \
  EXPECT_EQ(std::distance(Range1.begin(), Range1.end()),                       \
            std::distance(Range2.begin(), Range2.end()));                      \
  for (auto Pair : zip(Range1, Range2))                                        \
    EXPECT_EQ(&std::get<0>(Pair), std::get<1>(Pair));

TEST(BasicBlockTest, TestInstructionsWithoutDebug) {
  LLVMContext Ctx;

  Module *M = new Module("MyModule", Ctx);
  Type *ArgTy1[] = {Type::getInt32PtrTy(Ctx)};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), ArgTy1, false);
  Argument *V = new Argument(Type::getInt32Ty(Ctx));
  Function *F = Function::Create(FT, Function::ExternalLinkage, "", M);

  Value *DbgAddr = Intrinsic::getDeclaration(M, Intrinsic::dbg_addr);
  Value *DbgDeclare =
      Intrinsic::getDeclaration(M, Intrinsic::dbg_declare);
  Value *DbgValue = Intrinsic::getDeclaration(M, Intrinsic::dbg_value);
  Value *DIV = MetadataAsValue::get(Ctx, (Metadata *)nullptr);
  SmallVector<Value *, 3> Args = {DIV, DIV, DIV};

  BasicBlock *BB1 = BasicBlock::Create(Ctx, "", F);
  const BasicBlock *BBConst = BB1;
  IRBuilder<> Builder1(BB1);

  AllocaInst *Var = Builder1.CreateAlloca(Builder1.getInt8Ty());
  Builder1.CreateCall(DbgValue, Args);
  Instruction *AddInst = cast<Instruction>(Builder1.CreateAdd(V, V));
  Instruction *MulInst = cast<Instruction>(Builder1.CreateMul(AddInst, V));
  Builder1.CreateCall(DbgDeclare, Args);
  Instruction *SubInst = cast<Instruction>(Builder1.CreateSub(MulInst, V));
  Builder1.CreateCall(DbgAddr, Args);

  SmallVector<Instruction *, 4> Exp = {Var, AddInst, MulInst, SubInst};
  CHECK_ITERATORS(BB1->instructionsWithoutDebug(), Exp);
  CHECK_ITERATORS(BBConst->instructionsWithoutDebug(), Exp);

  delete M;
  delete V;
}

} // End anonymous namespace.
} // End llvm namespace.
