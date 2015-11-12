//===- Local.cpp - Unit tests for Local -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(Local, RecursivelyDeleteDeadPHINodes) {
  LLVMContext &C(getGlobalContext());

  IRBuilder<> builder(C);

  // Make blocks
  BasicBlock *bb0 = BasicBlock::Create(C);
  BasicBlock *bb1 = BasicBlock::Create(C);

  builder.SetInsertPoint(bb0);
  PHINode    *phi = builder.CreatePHI(Type::getInt32Ty(C), 2);
  BranchInst *br0 = builder.CreateCondBr(builder.getTrue(), bb0, bb1);

  builder.SetInsertPoint(bb1);
  BranchInst *br1 = builder.CreateBr(bb0);

  phi->addIncoming(phi, bb0);
  phi->addIncoming(phi, bb1);

  // The PHI will be removed
  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  // Make sure the blocks only contain the branches
  EXPECT_EQ(&bb0->front(), br0);
  EXPECT_EQ(&bb1->front(), br1);

  builder.SetInsertPoint(bb0);
  phi = builder.CreatePHI(Type::getInt32Ty(C), 0);

  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  builder.SetInsertPoint(bb0);
  phi = builder.CreatePHI(Type::getInt32Ty(C), 0);
  builder.CreateAdd(phi, phi);

  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  bb0->dropAllReferences();
  bb1->dropAllReferences();
  delete bb0;
  delete bb1;
}

TEST(Local, RemoveDuplicatePHINodes) {
  LLVMContext &C(getGlobalContext());
  IRBuilder<> B(C);

  std::unique_ptr<Function> F(
      Function::Create(FunctionType::get(B.getVoidTy(), false),
                       GlobalValue::ExternalLinkage, "F"));
  BasicBlock *Entry(BasicBlock::Create(C, "", F.get()));
  BasicBlock *BB(BasicBlock::Create(C, "", F.get()));
  BranchInst::Create(BB, Entry);

  B.SetInsertPoint(BB);

  AssertingVH<PHINode> P1 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P1->addIncoming(B.getInt32(42), Entry);

  PHINode *P2 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P2->addIncoming(B.getInt32(42), Entry);

  AssertingVH<PHINode> P3 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P3->addIncoming(B.getInt32(42), Entry);
  P3->addIncoming(B.getInt32(23), BB);

  PHINode *P4 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P4->addIncoming(B.getInt32(42), Entry);
  P4->addIncoming(B.getInt32(23), BB);

  P1->addIncoming(P3, BB);
  P2->addIncoming(P4, BB);
  BranchInst::Create(BB, BB);

  // Verify that we can eliminate PHIs that become duplicates after chaning PHIs
  // downstream.
  EXPECT_TRUE(EliminateDuplicatePHINodes(BB));
  EXPECT_EQ(3U, BB->size());
}
