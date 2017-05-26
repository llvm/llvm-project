//===- llvm/unittest/IR/BasicBlockTest.cpp - BasicBlock unit tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  // And iterate a const range.
  for (const auto &PN : const_cast<const BasicBlock *>(BB.get())->phis()) {
    EXPECT_EQ(BB.get(), PN.getIncomingBlock(0));
    EXPECT_EQ(BB1.get(), PN.getIncomingBlock(1));
    EXPECT_EQ(BB2.get(), PN.getIncomingBlock(2));
  }
}

} // End anonymous namespace.
} // End llvm namespace.
