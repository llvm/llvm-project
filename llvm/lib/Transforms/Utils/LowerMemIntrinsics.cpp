//===- LowerMemIntrinsics.cpp ----------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

void llvm::createMemCpyLoop(Instruction *InsertBefore,
                            Value *SrcAddr, Value *DstAddr, Value *CopyLen,
                            unsigned SrcAlign, unsigned DestAlign,
                            bool SrcIsVolatile, bool DstIsVolatile) {
  Type *TypeOfCopyLen = CopyLen->getType();

  BasicBlock *OrigBB = InsertBefore->getParent();
  Function *F = OrigBB->getParent();
  BasicBlock *NewBB =
    InsertBefore->getParent()->splitBasicBlock(InsertBefore, "split");
  BasicBlock *LoopBB = BasicBlock::Create(F->getContext(), "loadstoreloop",
                                          F, NewBB);

  IRBuilder<> Builder(OrigBB->getTerminator());

  // SrcAddr and DstAddr are expected to be pointer types,
  // so no check is made here.
  unsigned SrcAS = cast<PointerType>(SrcAddr->getType())->getAddressSpace();
  unsigned DstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();

  // Cast pointers to (char *)
  SrcAddr = Builder.CreateBitCast(SrcAddr, Builder.getInt8PtrTy(SrcAS));
  DstAddr = Builder.CreateBitCast(DstAddr, Builder.getInt8PtrTy(DstAS));

  Builder.CreateCondBr(
      Builder.CreateICmpEQ(ConstantInt::get(TypeOfCopyLen, 0), CopyLen), NewBB,
      LoopBB);
  OrigBB->getTerminator()->eraseFromParent();

  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopIndex = LoopBuilder.CreatePHI(TypeOfCopyLen, 0);
  LoopIndex->addIncoming(ConstantInt::get(TypeOfCopyLen, 0), OrigBB);

  // load from SrcAddr+LoopIndex
  // TODO: we can leverage the align parameter of llvm.memcpy for more efficient
  // word-sized loads and stores.
  Value *Element =
    LoopBuilder.CreateLoad(LoopBuilder.CreateInBoundsGEP(
                             LoopBuilder.getInt8Ty(), SrcAddr, LoopIndex),
                           SrcIsVolatile);
  // store at DstAddr+LoopIndex
  LoopBuilder.CreateStore(Element,
                          LoopBuilder.CreateInBoundsGEP(LoopBuilder.getInt8Ty(),
                                                        DstAddr, LoopIndex),
                          DstIsVolatile);

  // The value for LoopIndex coming from backedge is (LoopIndex + 1)
  Value *NewIndex =
    LoopBuilder.CreateAdd(LoopIndex, ConstantInt::get(TypeOfCopyLen, 1));
  LoopIndex->addIncoming(NewIndex, LoopBB);

  LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(NewIndex, CopyLen), LoopBB,
                           NewBB);
}

// Lower memmove to IR. memmove is required to correctly copy overlapping memory
// regions; therefore, it has to check the relative positions of the source and
// destination pointers and choose the copy direction accordingly.
//
// The code below is an IR rendition of this C function:
//
// void* memmove(void* dst, const void* src, size_t n) {
//   unsigned char* d = dst;
//   const unsigned char* s = src;
//   if (s < d) {
//     // copy backwards
//     while (n--) {
//       d[n] = s[n];
//     }
//   } else {
//     // copy forward
//     for (size_t i = 0; i < n; ++i) {
//       d[i] = s[i];
//     }
//   }
//   return dst;
// }
static void createMemMoveLoop(Instruction *InsertBefore,
                              Value *SrcAddr, Value *DstAddr, Value *CopyLen,
                              unsigned SrcAlign, unsigned DestAlign,
                              bool SrcIsVolatile, bool DstIsVolatile) {
  Type *TypeOfCopyLen = CopyLen->getType();
  BasicBlock *OrigBB = InsertBefore->getParent();
  Function *F = OrigBB->getParent();

  // Create the a comparison of src and dst, based on which we jump to either
  // the forward-copy part of the function (if src >= dst) or the backwards-copy
  // part (if src < dst).
  // SplitBlockAndInsertIfThenElse conveniently creates the basic if-then-else
  // structure. Its block terminators (unconditional branches) are replaced by
  // the appropriate conditional branches when the loop is built.
  ICmpInst *PtrCompare = new ICmpInst(InsertBefore, ICmpInst::ICMP_ULT,
                                      SrcAddr, DstAddr, "compare_src_dst");
  TerminatorInst *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(PtrCompare, InsertBefore, &ThenTerm,
                                &ElseTerm);

  // Each part of the function consists of two blocks:
  //   copy_backwards:        used to skip the loop when n == 0
  //   copy_backwards_loop:   the actual backwards loop BB
  //   copy_forward:          used to skip the loop when n == 0
  //   copy_forward_loop:     the actual forward loop BB
  BasicBlock *CopyBackwardsBB = ThenTerm->getParent();
  CopyBackwardsBB->setName("copy_backwards");
  BasicBlock *CopyForwardBB = ElseTerm->getParent();
  CopyForwardBB->setName("copy_forward");
  BasicBlock *ExitBB = InsertBefore->getParent();
  ExitBB->setName("memmove_done");

  // Initial comparison of n == 0 that lets us skip the loops altogether. Shared
  // between both backwards and forward copy clauses.
  ICmpInst *CompareN =
      new ICmpInst(OrigBB->getTerminator(), ICmpInst::ICMP_EQ, CopyLen,
                   ConstantInt::get(TypeOfCopyLen, 0), "compare_n_to_0");

  // Copying backwards.
  BasicBlock *LoopBB =
    BasicBlock::Create(F->getContext(), "copy_backwards_loop", F, CopyForwardBB);
  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopPhi = LoopBuilder.CreatePHI(TypeOfCopyLen, 0);
  Value *IndexPtr = LoopBuilder.CreateSub(
      LoopPhi, ConstantInt::get(TypeOfCopyLen, 1), "index_ptr");
  Value *Element = LoopBuilder.CreateLoad(
      LoopBuilder.CreateInBoundsGEP(SrcAddr, IndexPtr), "element");
  LoopBuilder.CreateStore(Element,
                          LoopBuilder.CreateInBoundsGEP(DstAddr, IndexPtr));
  LoopBuilder.CreateCondBr(
      LoopBuilder.CreateICmpEQ(IndexPtr, ConstantInt::get(TypeOfCopyLen, 0)),
      ExitBB, LoopBB);
  LoopPhi->addIncoming(IndexPtr, LoopBB);
  LoopPhi->addIncoming(CopyLen, CopyBackwardsBB);
  BranchInst::Create(ExitBB, LoopBB, CompareN, ThenTerm);
  ThenTerm->eraseFromParent();

  // Copying forward.
  BasicBlock *FwdLoopBB =
    BasicBlock::Create(F->getContext(), "copy_forward_loop", F, ExitBB);
  IRBuilder<> FwdLoopBuilder(FwdLoopBB);
  PHINode *FwdCopyPhi = FwdLoopBuilder.CreatePHI(TypeOfCopyLen, 0, "index_ptr");
  Value *FwdElement = FwdLoopBuilder.CreateLoad(
      FwdLoopBuilder.CreateInBoundsGEP(SrcAddr, FwdCopyPhi), "element");
  FwdLoopBuilder.CreateStore(
      FwdElement, FwdLoopBuilder.CreateInBoundsGEP(DstAddr, FwdCopyPhi));
  Value *FwdIndexPtr = FwdLoopBuilder.CreateAdd(
      FwdCopyPhi, ConstantInt::get(TypeOfCopyLen, 1), "index_increment");
  FwdLoopBuilder.CreateCondBr(FwdLoopBuilder.CreateICmpEQ(FwdIndexPtr, CopyLen),
                              ExitBB, FwdLoopBB);
  FwdCopyPhi->addIncoming(FwdIndexPtr, FwdLoopBB);
  FwdCopyPhi->addIncoming(ConstantInt::get(TypeOfCopyLen, 0), CopyForwardBB);

  BranchInst::Create(ExitBB, FwdLoopBB, CompareN, ElseTerm);
  ElseTerm->eraseFromParent();
}

static void createMemSetLoop(Instruction *InsertBefore,
                             Value *DstAddr, Value *CopyLen, Value *SetValue,
                             unsigned Align, bool IsVolatile) {
  Type *TypeOfCopyLen = CopyLen->getType();
  BasicBlock *OrigBB = InsertBefore->getParent();
  Function *F = OrigBB->getParent();
  BasicBlock *NewBB =
      OrigBB->splitBasicBlock(InsertBefore, "split");
  BasicBlock *LoopBB
    = BasicBlock::Create(F->getContext(), "loadstoreloop", F, NewBB);

  IRBuilder<> Builder(OrigBB->getTerminator());

  // Cast pointer to the type of value getting stored
  unsigned dstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();
  DstAddr = Builder.CreateBitCast(DstAddr,
                                  PointerType::get(SetValue->getType(), dstAS));

  Builder.CreateCondBr(
      Builder.CreateICmpEQ(ConstantInt::get(TypeOfCopyLen, 0), CopyLen), NewBB,
      LoopBB);
  OrigBB->getTerminator()->eraseFromParent();

  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopIndex = LoopBuilder.CreatePHI(TypeOfCopyLen, 0);
  LoopIndex->addIncoming(ConstantInt::get(TypeOfCopyLen, 0), OrigBB);

  LoopBuilder.CreateStore(
      SetValue,
      LoopBuilder.CreateInBoundsGEP(SetValue->getType(), DstAddr, LoopIndex),
      IsVolatile);

  Value *NewIndex =
      LoopBuilder.CreateAdd(LoopIndex, ConstantInt::get(TypeOfCopyLen, 1));
  LoopIndex->addIncoming(NewIndex, LoopBB);

  LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(NewIndex, CopyLen), LoopBB,
                           NewBB);
}

void llvm::expandMemCpyAsLoop(MemCpyInst *Memcpy) {
  createMemCpyLoop(/* InsertBefore */ Memcpy,
                   /* SrcAddr */ Memcpy->getRawSource(),
                   /* DstAddr */ Memcpy->getRawDest(),
                   /* CopyLen */ Memcpy->getLength(),
                   /* SrcAlign */ Memcpy->getAlignment(),
                   /* DestAlign */ Memcpy->getAlignment(),
                   /* SrcIsVolatile */ Memcpy->isVolatile(),
                   /* DstIsVolatile */ Memcpy->isVolatile());
}

void llvm::expandMemMoveAsLoop(MemMoveInst *Memmove) {
  createMemMoveLoop(/* InsertBefore */ Memmove,
                    /* SrcAddr */ Memmove->getRawSource(),
                    /* DstAddr */ Memmove->getRawDest(),
                    /* CopyLen */ Memmove->getLength(),
                    /* SrcAlign */ Memmove->getAlignment(),
                    /* DestAlign */ Memmove->getAlignment(),
                    /* SrcIsVolatile */ Memmove->isVolatile(),
                    /* DstIsVolatile */ Memmove->isVolatile());
}

void llvm::expandMemSetAsLoop(MemSetInst *Memset) {
  createMemSetLoop(/* InsertBefore */ Memset,
                   /* DstAddr */ Memset->getRawDest(),
                   /* CopyLen */ Memset->getLength(),
                   /* SetValue */ Memset->getValue(),
                   /* Alignment */ Memset->getAlignment(),
                   Memset->isVolatile());
}
