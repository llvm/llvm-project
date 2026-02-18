//===- PtrUseVisitorTest.cpp - PtrUseVisitor unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct TestVisitor : public PtrUseVisitor<TestVisitor> {
  using Base = PtrUseVisitor<TestVisitor>;

  struct Visit {
    Instruction *I;
    bool OffsetKnown;
    APInt Offset;
    DenseMap<Instruction *, APInt> InstsInPath;
  };

  std::vector<Visit> Visits;

  TestVisitor(const DataLayout &DL) : Base(DL) {}

  void visitPHINode(PHINode &PN) { enqueueUsers(PN); }

  void visitSelectInst(SelectInst &SI) { enqueueUsers(SI); }

  void visitStoreInst(StoreInst &SI) {
    Visits.push_back({&SI, IsOffsetKnown, Offset, InstsInPath});
    if (SI.getValueOperand() == U->get())
      PI.setEscaped(&SI);
  }

  void visitLoadInst(LoadInst &LI) {
    Visits.push_back({&LI, IsOffsetKnown, Offset, InstsInPath});
  }
};

TEST(PtrUseVisitorTest, PHIWithConflictingOffsets) {
  LLVMContext C;
  Module M("PtrUseVisitorTest", C);

  Type *I64Ty = Type::getInt64Ty(C);
  StructType *StructTy = StructType::get(C, {I64Ty, I64Ty, I64Ty});
  Type *PtrTy = PointerType::get(C, 0);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *VoidTy = Type::getVoidTy(C);

  // Create a function with a PHI that merges GEPs with different offsets
  Function *F = Function::Create(FunctionType::get(VoidTy, {I1Ty}, false),
                                 Function::ExternalLinkage, "f", M);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Then = BasicBlock::Create(C, "then", F);
  BasicBlock *Else = BasicBlock::Create(C, "else", F);
  BasicBlock *Merge = BasicBlock::Create(C, "merge", F);

  AllocaInst *Alloca = new AllocaInst(StructTy, 0, "alloca", Entry);
  BranchInst::Create(Then, Else, F->getArg(0), Entry);

  Value *GEP0 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)}, "gep0", Then);
  BranchInst::Create(Merge, Then);

  Value *GEP1 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 1)}, "gep1", Else);
  BranchInst::Create(Merge, Else);

  PHINode *Phi = PHINode::Create(PtrTy, 2, "phi", Merge);
  Phi->addIncoming(GEP0, Then);
  Phi->addIncoming(GEP1, Else);

  LoadInst *Load = new LoadInst(I64Ty, Phi, "load", Merge);
  ReturnInst::Create(C, Merge);

  TestVisitor TV(M.getDataLayout());
  TV.visitPtr(*Alloca);

  // Check that load is visited twice with different offsets
  ASSERT_EQ(TV.Visits.size(), 2u);
  EXPECT_EQ(TV.Visits[0].I, Load);
  EXPECT_EQ(TV.Visits[1].I, Load);
  EXPECT_TRUE(TV.Visits[0].OffsetKnown);
  EXPECT_TRUE(TV.Visits[1].OffsetKnown);
  EXPECT_NE(TV.Visits[0].Offset, TV.Visits[1].Offset);
  EXPECT_TRUE((TV.Visits[0].Offset == 0 && TV.Visits[1].Offset == 8) ||
              (TV.Visits[0].Offset == 8 && TV.Visits[1].Offset == 0));
  // InstsInPath tracks all instructions in the path (Alloca, GEP, PHI)
  EXPECT_GE(TV.Visits[0].InstsInPath.size(), 1u);
  EXPECT_TRUE(TV.Visits[0].InstsInPath.contains(Phi));
  EXPECT_GE(TV.Visits[1].InstsInPath.size(), 1u);
  EXPECT_TRUE(TV.Visits[1].InstsInPath.contains(Phi));
}

TEST(PtrUseVisitorTest, PHIWithSameOffset) {
  LLVMContext C;
  Module M("PtrUseVisitorTest", C);

  Type *I64Ty = Type::getInt64Ty(C);
  StructType *StructTy = StructType::get(C, {I64Ty, I64Ty, I64Ty});
  Type *PtrTy = PointerType::get(C, 0);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *VoidTy = Type::getVoidTy(C);

  // Create a function with a PHI that merges GEPs with the same offset
  Function *F = Function::Create(FunctionType::get(VoidTy, {I1Ty}, false),
                                 Function::ExternalLinkage, "f", M);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Then = BasicBlock::Create(C, "then", F);
  BasicBlock *Else = BasicBlock::Create(C, "else", F);
  BasicBlock *Merge = BasicBlock::Create(C, "merge", F);

  AllocaInst *Alloca = new AllocaInst(StructTy, 0, "alloca", Entry);
  BranchInst::Create(Then, Else, F->getArg(0), Entry);

  Value *GEP0 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 1)}, "gep0", Then);
  BranchInst::Create(Merge, Then);

  Value *GEP1 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 1)}, "gep1", Else);
  BranchInst::Create(Merge, Else);

  PHINode *Phi = PHINode::Create(PtrTy, 2, "phi", Merge);
  Phi->addIncoming(GEP0, Then);
  Phi->addIncoming(GEP1, Else);

  LoadInst *Load = new LoadInst(I64Ty, Phi, "load", Merge);
  ReturnInst::Create(C, Merge);

  TestVisitor TV(M.getDataLayout());
  TV.visitPtr(*Alloca);

  // Check that load is visited once when offsets are the same
  ASSERT_EQ(TV.Visits.size(), 1u);
  EXPECT_EQ(TV.Visits[0].I, Load);
  EXPECT_TRUE(TV.Visits[0].OffsetKnown);
  EXPECT_EQ(TV.Visits[0].Offset, 8u);
}

TEST(PtrUseVisitorTest, UnknownAndKnownOffsets) {
  LLVMContext C;
  Module M("PtrUseVisitorTest", C);

  Type *I64Ty = Type::getInt64Ty(C);
  StructType *StructTy = StructType::get(C, {I64Ty, I64Ty, I64Ty});
  Type *PtrTy = PointerType::get(C, 0);
  Type *VoidTy = Type::getVoidTy(C);

  // Create a function with both unknown and known offset GEPs
  Function *F = Function::Create(FunctionType::get(VoidTy, {PtrTy}, false),
                                 Function::ExternalLinkage, "f", M);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);

  AllocaInst *Alloca = new AllocaInst(StructTy, 0, "alloca", Entry);

  Value *VarGEP = GetElementPtrInst::Create(I64Ty, Alloca, {F->getArg(0)},
                                            "var_gep", Entry);
  LoadInst *Load1 = new LoadInst(I64Ty, VarGEP, "load1", Entry);

  Value *ConstGEP = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 1)}, "const_gep",
      Entry);
  LoadInst *Load2 = new LoadInst(I64Ty, ConstGEP, "load2", Entry);

  ReturnInst::Create(C, Entry);

  TestVisitor TV(M.getDataLayout());
  TV.visitPtr(*Alloca);

  // Check that both unknown and known offset loads are visited
  ASSERT_EQ(TV.Visits.size(), 2u);
  EXPECT_EQ(TV.Visits[0].I, Load1);
  EXPECT_FALSE(TV.Visits[0].OffsetKnown);
  EXPECT_EQ(TV.Visits[1].I, Load2);
  EXPECT_TRUE(TV.Visits[1].OffsetKnown);
  EXPECT_EQ(TV.Visits[1].Offset, 8u);
}

TEST(PtrUseVisitorTest, NoOffsetTracking) {
  LLVMContext C;
  Module M("PtrUseVisitorTest", C);

  Type *I64Ty = Type::getInt64Ty(C);
  StructType *StructTy = StructType::get(C, {I64Ty, I64Ty, I64Ty});
  Type *PtrTy = PointerType::get(C, 0);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *VoidTy = Type::getVoidTy(C);

  // Create a function with a PHI merging GEPs, visited without offset tracking
  Function *F = Function::Create(FunctionType::get(VoidTy, {I1Ty}, false),
                                 Function::ExternalLinkage, "f", M);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Then = BasicBlock::Create(C, "then", F);
  BasicBlock *Else = BasicBlock::Create(C, "else", F);
  BasicBlock *Merge = BasicBlock::Create(C, "merge", F);

  AllocaInst *Alloca = new AllocaInst(StructTy, 0, "alloca", Entry);
  BranchInst::Create(Then, Else, F->getArg(0), Entry);

  Value *GEP0 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)}, "gep0", Then);
  BranchInst::Create(Merge, Then);

  Value *GEP1 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 1)}, "gep1", Else);
  BranchInst::Create(Merge, Else);

  PHINode *Phi = PHINode::Create(PtrTy, 2, "phi", Merge);
  Phi->addIncoming(GEP0, Then);
  Phi->addIncoming(GEP1, Else);

  LoadInst *Load = new LoadInst(I64Ty, Phi, "load", Merge);
  ReturnInst::Create(C, Merge);

  TestVisitor TV(M.getDataLayout());
  TV.visitPtr(*Alloca, /*TrackOffsets=*/false);

  // Check that load is visited once when offset tracking is disabled
  ASSERT_EQ(TV.Visits.size(), 1u);
  EXPECT_EQ(TV.Visits[0].I, Load);
  EXPECT_FALSE(TV.Visits[0].OffsetKnown);
}

TEST(PtrUseVisitorTest, PHITracking) {
  LLVMContext C;
  Module M("PtrUseVisitorTest", C);

  Type *I64Ty = Type::getInt64Ty(C);
  StructType *StructTy = StructType::get(C, {I64Ty, I64Ty, I64Ty});
  Type *PtrTy = PointerType::get(C, 0);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *VoidTy = Type::getVoidTy(C);

  // Create a function with a PHI merging GEPs with different offsets
  Function *F = Function::Create(FunctionType::get(VoidTy, {I1Ty}, false),
                                 Function::ExternalLinkage, "f", M);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Then = BasicBlock::Create(C, "then", F);
  BasicBlock *Else = BasicBlock::Create(C, "else", F);
  BasicBlock *Merge = BasicBlock::Create(C, "merge", F);

  AllocaInst *Alloca = new AllocaInst(StructTy, 0, "alloca", Entry);
  BranchInst::Create(Then, Else, F->getArg(0), Entry);

  Value *GEP0 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)}, "gep0", Then);
  BranchInst::Create(Merge, Then);

  Value *GEP1 = GetElementPtrInst::CreateInBounds(
      StructTy, Alloca,
      {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 1)}, "gep1", Else);
  BranchInst::Create(Merge, Else);

  PHINode *Phi = PHINode::Create(PtrTy, 2, "phi", Merge);
  Phi->addIncoming(GEP0, Then);
  Phi->addIncoming(GEP1, Else);

  LoadInst *Load = new LoadInst(I64Ty, Phi, "load", Merge);
  ReturnInst::Create(C, Merge);

  TestVisitor TV(M.getDataLayout());
  TV.visitPtr(*Alloca);

  // Check that both visits tracked the instructions in the path correctly
  ASSERT_EQ(TV.Visits.size(), 2u);
  EXPECT_GE(TV.Visits[0].InstsInPath.size(), 1u);
  EXPECT_GE(TV.Visits[1].InstsInPath.size(), 1u);
  EXPECT_EQ(TV.Visits[0].InstsInPath.count(Phi), 1u);
  EXPECT_EQ(TV.Visits[1].InstsInPath.count(Phi), 1u);
  EXPECT_EQ(TV.Visits[0].I, Load);
  EXPECT_EQ(TV.Visits[1].I, Load);
  EXPECT_TRUE((TV.Visits[0].InstsInPath[Phi] == 0 &&
               TV.Visits[1].InstsInPath[Phi] == 8) ||
              (TV.Visits[0].InstsInPath[Phi] == 8 &&
               TV.Visits[1].InstsInPath[Phi] == 0));
}

TEST(PtrUseVisitorTest, PHICycle) {
  LLVMContext C;
  Module M("PtrUseVisitorTest", C);

  Type *I64Ty = Type::getInt64Ty(C);
  Type *PtrTy = PointerType::get(C, 0);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *VoidTy = Type::getVoidTy(C);

  // Create a function with a PHI cycle
  Function *F =
      Function::Create(FunctionType::get(VoidTy, {PtrTy, I1Ty}, false),
                       Function::ExternalLinkage, "f", M);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Loop = BasicBlock::Create(C, "loop", F);
  BasicBlock *Exit = BasicBlock::Create(C, "exit", F);

  BranchInst::Create(Loop, Entry);

  PHINode *Phi = PHINode::Create(PtrTy, 2, "phi", Loop);
  Phi->addIncoming(F->getArg(0), Entry);

  Value *GEP = GetElementPtrInst::Create(
      I64Ty, Phi, {ConstantInt::get(I64Ty, 1)}, "gep", Loop);
  Phi->addIncoming(GEP, Loop);

  BranchInst::Create(Loop, Exit, F->getArg(1), Loop);

  LoadInst *Load = new LoadInst(I64Ty, Phi, "load", Exit);
  ReturnInst::Create(C, Exit);

  TestVisitor TV(M.getDataLayout());
  TV.visitPtr(*F->getArg(0));

  // The PHI creates a cycle, so when we revisit it, offset should become
  // unknown
  bool FoundUnknownOffset = false;
  for (const auto &Visit : TV.Visits) {
    EXPECT_EQ(Visit.I, Load);
    if (!Visit.OffsetKnown) {
      FoundUnknownOffset = true;
      break;
    }
  }
  EXPECT_TRUE(FoundUnknownOffset);
}

} // namespace
