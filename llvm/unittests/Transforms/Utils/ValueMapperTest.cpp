//===- ValueMapper.cpp - Unit tests for ValueMapper -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ValueMapperTest, MapMetadata) {
  LLVMContext Context;
  auto *U = MDTuple::get(Context, None);

  // The node should be unchanged.
  ValueToValueMapTy VM;
  EXPECT_EQ(U, MapMetadata(U, VM, RF_None));
}

TEST(ValueMapperTest, MapMetadataCycle) {
  LLVMContext Context;
  MDNode *U0;
  MDNode *U1;
  {
    Metadata *Ops[] = {nullptr};
    auto T = MDTuple::getTemporary(Context, Ops);
    Ops[0] = T.get();
    U0 = MDTuple::get(Context, Ops);
    T->replaceOperandWith(0, U0);
    U1 = MDNode::replaceWithUniqued(std::move(T));
    U0->resolveCycles();
  }

  EXPECT_TRUE(U0->isResolved());
  EXPECT_TRUE(U0->isUniqued());
  EXPECT_TRUE(U1->isResolved());
  EXPECT_TRUE(U1->isUniqued());
  EXPECT_EQ(U1, U0->getOperand(0));
  EXPECT_EQ(U0, U1->getOperand(0));

  // Cycles shouldn't be duplicated.
  {
    ValueToValueMapTy VM;
    EXPECT_EQ(U0, MapMetadata(U0, VM, RF_None));
    EXPECT_EQ(U1, MapMetadata(U1, VM, RF_None));
  }

  // Check the other order.
  {
    ValueToValueMapTy VM;
    EXPECT_EQ(U1, MapMetadata(U1, VM, RF_None));
    EXPECT_EQ(U0, MapMetadata(U0, VM, RF_None));
  }
}

TEST(ValueMapperTest, MapMetadataUnresolved) {
  LLVMContext Context;
  TempMDTuple T = MDTuple::getTemporary(Context, None);

  ValueToValueMapTy VM;
  EXPECT_EQ(T.get(), MapMetadata(T.get(), VM, RF_NoModuleLevelChanges));
}

TEST(ValueMapperTest, MapMetadataDistinct) {
  LLVMContext Context;
  auto *D = MDTuple::getDistinct(Context, None);

  {
    // The node should be cloned.
    ValueToValueMapTy VM;
    EXPECT_NE(D, MapMetadata(D, VM, RF_None));
  }
  {
    // The node should be moved.
    ValueToValueMapTy VM;
    EXPECT_EQ(D, MapMetadata(D, VM, RF_MoveDistinctMDs));
  }
}

TEST(ValueMapperTest, MapMetadataDistinctOperands) {
  LLVMContext Context;
  Metadata *Old = MDTuple::getDistinct(Context, None);
  auto *D = MDTuple::getDistinct(Context, Old);
  ASSERT_EQ(Old, D->getOperand(0));

  Metadata *New = MDTuple::getDistinct(Context, None);
  ValueToValueMapTy VM;
  VM.MD()[Old].reset(New);

  // Make sure operands are updated.
  EXPECT_EQ(D, MapMetadata(D, VM, RF_MoveDistinctMDs));
  EXPECT_EQ(New, D->getOperand(0));
}

TEST(ValueMapperTest, MapMetadataSeeded) {
  LLVMContext Context;
  auto *D = MDTuple::getDistinct(Context, None);

  // The node should be moved.
  ValueToValueMapTy VM;
  EXPECT_EQ(None, VM.getMappedMD(D));

  VM.MD().insert(std::make_pair(D, TrackingMDRef(D)));
  EXPECT_EQ(D, *VM.getMappedMD(D));
  EXPECT_EQ(D, MapMetadata(D, VM, RF_None));
}

TEST(ValueMapperTest, MapMetadataSeededWithNull) {
  LLVMContext Context;
  auto *D = MDTuple::getDistinct(Context, None);

  // The node should be moved.
  ValueToValueMapTy VM;
  EXPECT_EQ(None, VM.getMappedMD(D));

  VM.MD().insert(std::make_pair(D, TrackingMDRef()));
  EXPECT_EQ(nullptr, *VM.getMappedMD(D));
  EXPECT_EQ(nullptr, MapMetadata(D, VM, RF_None));
}

} // end namespace
