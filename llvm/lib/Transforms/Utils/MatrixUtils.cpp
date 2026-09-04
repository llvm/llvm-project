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
#include "llvm/IR/Constant.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace llvm {
Type *indexType(LLVMContext &Ctx, const DataLayout &DL) {
  auto *PtrTy = PointerType::get(Ctx, /*AddrSpace=*/0);
  return DL.getIndexType(PtrTy);
}

Constant *asIndex(IRBuilderBase &Builder, unsigned Val) {
  const DataLayout &DL = Builder.GetInsertBlock()->getModule()->getDataLayout();
  return ConstantInt::get(indexType(Builder.getContext(), DL), Val);
}

extern cl::opt<bool> ProfcheckDisableMetadataFixes;
} // namespace llvm

BasicBlock *TileInfo::CreateLoop(BasicBlock *Preheader, BasicBlock *Exit,
                                 Value *Start, Value *Bound, Value *Step,
                                 StringRef Name, IRBuilderBase &B,
                                 DomTreeUpdater &DTU, Loop *L, LoopInfo &LI,
                                 unsigned Successor) {
  LLVMContext &Ctx = Preheader->getContext();
  BasicBlock *Header = BasicBlock::Create(
      Preheader->getContext(), Name + ".header", Preheader->getParent(), Exit);
  BasicBlock *Body = BasicBlock::Create(Header->getContext(), Name + ".body",
                                        Header->getParent(), Exit);
  BasicBlock *Latch = BasicBlock::Create(Header->getContext(), Name + ".latch",
                                         Header->getParent(), Exit);

  Type *I32Ty = Type::getInt64Ty(Ctx);
  UncondBrInst::Create(Body, Header);
  UncondBrInst::Create(Latch, Body);
  PHINode *IV =
      PHINode::Create(I32Ty, 2, Name + ".iv", Header->getTerminator()->getIterator());
  IV->addIncoming(Start, Preheader);

  B.SetInsertPoint(Latch);
  Value *Inc = B.CreateAdd(IV, Step, Name + ".step");
  Value *Cond = B.CreateICmpNE(Inc, Bound, Name + ".cond");
  auto *BR = B.CreateCondBr(Cond, Header, Exit);
  if (!ProfcheckDisableMetadataFixes) {
    auto *CIStep = cast<ConstantInt>(Step);
    auto *CIBound = cast<ConstantInt>(Bound);
    assert(CIStep->getZExtValue() != 0 &&
           "Expected a non-zero step size. This is chosen by the pass and "
           "should always be non-zero to imply a finite loop.");
    MDBuilder MDB(Preheader->getContext());
    setFittedBranchWeights(
        *BR, {CIBound->getZExtValue() / CIStep->getZExtValue(), 1}, false);
  }
  IV->addIncoming(Inc, Latch);

  UncondBrInst *PreheaderBr = cast<UncondBrInst>(Preheader->getTerminator());
  BasicBlock *Tmp = PreheaderBr->getSuccessor(Successor);
  PreheaderBr->setSuccessor(Successor, Header);
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

BasicBlock *TileInfo::CreateLoop(BasicBlock *Preheader, BasicBlock *Exit,
                                 Value *Bound, Value *Step, StringRef Name,
                                 IRBuilderBase &B, DomTreeUpdater &DTU, Loop *L,
                                 LoopInfo &LI, unsigned Successor) {
  return CreateLoop(Preheader, Exit, asIndex(B, 0), Bound, Step, Name, B, DTU,
                    L, LI, Successor);
}

// Creates the following loop nest skeleton:
//  for C = 0; C < NumColumns; C += TileNumColumns
//    for R = 0; R < NumRows; R += TileNumRows
//      for K = 0; K < NumInner; K += TileNumInner
BasicBlock *TileInfo::CreateTiledLoops(BasicBlock *Start, BasicBlock *End,
                                       IRBuilderBase &B, DomTreeUpdater &DTU,
                                       LoopInfo &LI) {
  Loop *ColumnLoopInfo = LI.AllocateLoop();
  Loop *RowLoopInfo = LI.AllocateLoop();
  Loop *KLoopInfo = LI.AllocateLoop();
  RowLoopInfo->addChildLoop(KLoopInfo);
  ColumnLoopInfo->addChildLoop(RowLoopInfo);
  if (Loop *ParentL = LI.getLoopFor(Start))
    ParentL->addChildLoop(ColumnLoopInfo);
  else
    LI.addTopLevelLoop(ColumnLoopInfo);

  BasicBlock *ColBody =
      CreateLoop(Start, End, asIndex(B, NumColumns), asIndex(B, TileNumColumns),
                 "cols", B, DTU, ColumnLoopInfo, LI);
  ColumnLoop.Latch = ColBody->getSingleSuccessor();
  BasicBlock *RowBody =
      CreateLoop(ColBody, ColumnLoop.Latch, asIndex(B, NumRows),
                 asIndex(B, TileNumRows), "rows", B, DTU, RowLoopInfo, LI);
  RowLoop.Latch = RowBody->getSingleSuccessor();

  BasicBlock *KBody =
      CreateLoop(RowBody, RowLoop.Latch, asIndex(B, NumInner),
                 asIndex(B, TileNumInner), "k", B, DTU, KLoopInfo, LI);
  KLoop.Latch = KBody->getSingleSuccessor();
  ColumnLoop.Header = ColBody->getSinglePredecessor();
  RowLoop.Header = RowBody->getSinglePredecessor();
  KLoop.Header = KBody->getSinglePredecessor();
  RowLoop.Index = &*RowLoop.Header->begin();
  ColumnLoop.Index = &*ColumnLoop.Header->begin();
  KLoop.Index = &*KLoop.Header->begin();

  return KBody;
}

void TileInfo::CreateTiledLoopsWithRemainder(BasicBlock *Start, BasicBlock *End,
                                             IRBuilderBase &B,
                                             DomTreeUpdater &DTU,
                                             LoopInfo &LI) {
  Loop *ColumnLoopInfo = LI.AllocateLoop();
  Loop *RowLoopInfo = LI.AllocateLoop();
  Loop *KLoopInfo = LI.AllocateLoop();

  unsigned RemainderRows = NumRows % TileNumRows;
  unsigned RemainderColumns = NumColumns % TileNumColumns;
  unsigned RemainderInner = NumInner % TileNumInner;

  Loop *RemainderColumnsKLoopInfo =
      RemainderColumns ? LI.AllocateLoop() : nullptr;
  Loop *RemainderRowsColumnLoopInfo =
      RemainderRows ? LI.AllocateLoop() : nullptr;
  Loop *RemainderRowsKLoopInfo = RemainderRows ? LI.AllocateLoop() : nullptr;
  Loop *RemainderRowsRemainderColumnsLoopInfo =
      RemainderRows && RemainderColumns ? LI.AllocateLoop() : nullptr;

  ColumnLoopInfo->addChildLoop(KLoopInfo);
  RowLoopInfo->addChildLoop(ColumnLoopInfo);
  if (RemainderColumnsKLoopInfo)
    RowLoopInfo->addChildLoop(RemainderColumnsKLoopInfo);

  auto AddTopLevelLoop = [&](Loop *L) {
    if (Loop *ParentL = LI.getLoopFor(Start))
      ParentL->addChildLoop(L);
    else
      LI.addTopLevelLoop(L);
  };

  AddTopLevelLoop(RowLoopInfo);

  if (RemainderRowsColumnLoopInfo) {
    RemainderRowsColumnLoopInfo->addChildLoop(RemainderRowsKLoopInfo);
    AddTopLevelLoop(RemainderRowsColumnLoopInfo);
  }
  if (RemainderRowsRemainderColumnsLoopInfo)
    AddTopLevelLoop(RemainderRowsRemainderColumnsLoopInfo);

  BasicBlock *RowBody =
      CreateLoop(Start, End, asIndex(B, NumRows - RemainderRows),
                 asIndex(B, TileNumRows), "rows", B, DTU, RowLoopInfo, LI);
  RowLoop.Latch = RowBody->getSingleSuccessor();

  if (RemainderRowsColumnLoopInfo) {
    BasicBlock *RemainderRowsColumnBody = CreateLoop(
        RowLoop.Latch, End, asIndex(B, NumColumns - RemainderColumns),
        asIndex(B, TileNumColumns), "remainder.rows.cols", B, DTU,
        RemainderRowsColumnLoopInfo, LI, 1);
    RemainderRowsColumnLoop.Latch =
        RemainderRowsColumnBody->getSingleSuccessor();

    BasicBlock *RemainderRowsKBody = CreateLoop(
        RemainderRowsColumnBody, RemainderRowsColumnLoop.Latch,
        asIndex(B, NumInner - RemainderInner), asIndex(B, TileNumInner),
        "remainder.rows.k", B, DTU, RemainderRowsKLoopInfo, LI);
    RemainderRowsKLoop.Latch = RemainderRowsKBody->getSingleSuccessor();

    RemainderRowsColumnLoop.Header =
        RemainderRowsColumnBody->getSinglePredecessor();
    RemainderRowsKLoop.Header = RemainderRowsKBody->getSinglePredecessor();

    RemainderRowsColumnLoop.Index = &*RemainderRowsColumnLoop.Header->begin();
    RemainderRowsKLoop.Index = &*RemainderRowsKLoop.Header->begin();
  }

  if (RemainderRowsRemainderColumnsLoopInfo) {
    BasicBlock *RemainderRowsRemainderColumnsBody =
        CreateLoop(RemainderRowsColumnLoop.Latch, End,
                   asIndex(B, NumInner - RemainderInner),
                   asIndex(B, TileNumInner), "remainder.rows.remainder.cols", B,
                   DTU, RemainderRowsRemainderColumnsLoopInfo, LI, 1);
    RemainderRowsRemainderColumnsKLoop.Latch =
        RemainderRowsRemainderColumnsBody->getSingleSuccessor();
    RemainderRowsRemainderColumnsKLoop.Header =
        RemainderRowsRemainderColumnsBody->getSinglePredecessor();

    RemainderRowsRemainderColumnsKLoop.Index =
        &*RemainderRowsRemainderColumnsKLoop.Header->begin();
  }

  BasicBlock *ColBody = CreateLoop(
      RowBody, RowLoop.Latch, asIndex(B, NumColumns - RemainderColumns),
      asIndex(B, TileNumColumns), "cols", B, DTU, ColumnLoopInfo, LI);
  ColumnLoop.Latch = ColBody->getSingleSuccessor();

  if (RemainderColumnsKLoopInfo) {
    BasicBlock *RemainderColumnsKBody = CreateLoop(
        ColumnLoop.Latch, RowLoop.Latch, asIndex(B, NumInner - RemainderInner),
        asIndex(B, TileNumInner), "remainder.cols.k", B, DTU,
        RemainderColumnsKLoopInfo, LI, 1);
    RemainderColumnsKLoop.Latch = RemainderColumnsKBody->getSingleSuccessor();
    RemainderColumnsKLoop.Header =
        RemainderColumnsKBody->getSinglePredecessor();
    RemainderColumnsKLoop.Index = &*RemainderColumnsKLoop.Header->begin();
  }

  BasicBlock *KBody = CreateLoop(
      ColBody, ColumnLoop.Latch, asIndex(B, NumInner - RemainderInner),
      asIndex(B, TileNumInner), "k", B, DTU, KLoopInfo, LI);

  KLoop.Latch = KBody->getSingleSuccessor();
  ColumnLoop.Header = ColBody->getSinglePredecessor();
  RowLoop.Header = RowBody->getSinglePredecessor();
  KLoop.Header = KBody->getSinglePredecessor();

  RowLoop.Index = &*RowLoop.Header->begin();
  ColumnLoop.Index = &*ColumnLoop.Header->begin();
  KLoop.Index = &*KLoop.Header->begin();
}
