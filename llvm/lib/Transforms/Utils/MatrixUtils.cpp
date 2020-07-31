//===- MatrixUtils.cpp - Utilities to lower matrix intrinsics ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for generating tiled loops for matrix operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/MatrixUtils.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

using namespace llvm;

BasicBlock *TileInfo::CreateLoop(BasicBlock *Preheader, BasicBlock *Exit,
                                 Value *Bound, Value *Step, StringRef Name,
                                 IRBuilderBase &B, DomTreeUpdater &DTU, Loop *L,
                                 LoopInfo &LI) {
  LLVMContext &Ctx = Preheader->getContext();
  BasicBlock *Header = BasicBlock::Create(
      Preheader->getContext(), Name + ".header", Preheader->getParent(), Exit);
  BasicBlock *Body = BasicBlock::Create(Header->getContext(), Name + ".body",
                                        Header->getParent(), Exit);
  BasicBlock *Latch = BasicBlock::Create(Header->getContext(), Name + ".latch",
                                         Header->getParent(), Exit);

  Type *I32Ty = Type::getInt64Ty(Ctx);
  BranchInst::Create(Body, Header);
  BranchInst::Create(Latch, Body);
  PHINode *IV =
      PHINode::Create(I32Ty, 2, Name + ".iv", Header->getTerminator());
  IV->addIncoming(ConstantInt::get(I32Ty, 0), Preheader);

  B.SetInsertPoint(Latch);
  Value *Inc = B.CreateAdd(IV, Step, Name + ".step");
  Value *Cond = B.CreateICmpNE(Inc, Bound, Name + ".cond");
  BranchInst::Create(Header, Exit, Cond, Latch);
  IV->addIncoming(Inc, Latch);

  BranchInst *PreheaderBr = cast<BranchInst>(Preheader->getTerminator());
  BasicBlock *Tmp = PreheaderBr->getSuccessor(0);
  PreheaderBr->setSuccessor(0, Header);
  DTU.applyUpdatesPermissive({
      {DominatorTree::Delete, Preheader, Tmp},
      {DominatorTree::Insert, Header, Body},
      {DominatorTree::Insert, Body, Latch},
      {DominatorTree::Insert, Latch, Header},
      {DominatorTree::Insert, Latch, Exit},
      {DominatorTree::Insert, Preheader, Header},
  });

  L->addBasicBlockToLoop(Header, LI);
  L->addBasicBlockToLoop(Body, LI);
  L->addBasicBlockToLoop(Latch, LI);
  return Body;
}

// Creates the following loop nest skeleton:
//  for C = 0; C < NumColumns; C += TileSize
//    for R = 0; R < NumRows; R += TileSize
//      for K = 0; K < Inner ; K += TileSize
BasicBlock *TileInfo::CreateTiledLoops(BasicBlock *Start, BasicBlock *End,
                                       IRBuilderBase &B, DomTreeUpdater &DTU,
                                       LoopInfo &LI) {
  Loop *ColLoop = LI.AllocateLoop();
  Loop *RowLoop = LI.AllocateLoop();
  Loop *InnerLoop = LI.AllocateLoop();
  RowLoop->addChildLoop(InnerLoop);
  ColLoop->addChildLoop(RowLoop);
  if (Loop *ParentL = LI.getLoopFor(Start))
    ParentL->addChildLoop(ColLoop);
  else
    LI.addTopLevelLoop(ColLoop);

  BasicBlock *ColBody =
      CreateLoop(Start, End, B.getInt64(NumColumns), B.getInt64(TileSize),
                 "cols", B, DTU, ColLoop, LI);
  BasicBlock *ColLatch = ColBody->getSingleSuccessor();
  BasicBlock *RowBody =
      CreateLoop(ColBody, ColLatch, B.getInt64(NumRows), B.getInt64(TileSize),
                 "rows", B, DTU, RowLoop, LI);
  RowLoopLatch = RowBody->getSingleSuccessor();

  BasicBlock *InnerBody =
      CreateLoop(RowBody, RowLoopLatch, B.getInt64(NumInner),
                 B.getInt64(TileSize), "inner", B, DTU, InnerLoop, LI);
  InnerLoopLatch = InnerBody->getSingleSuccessor();
  ColumnLoopHeader = ColBody->getSinglePredecessor();
  RowLoopHeader = RowBody->getSinglePredecessor();
  InnerLoopHeader = InnerBody->getSinglePredecessor();
  CurrentRow = &*RowLoopHeader->begin();
  CurrentCol = &*ColumnLoopHeader->begin();
  CurrentK = &*InnerLoopHeader->begin();

  return InnerBody;
}
