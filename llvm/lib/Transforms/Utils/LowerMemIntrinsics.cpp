//===- LowerMemIntrinsics.cpp ----------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>

#define DEBUG_TYPE "lower-mem-intrinsics"

using namespace llvm;

void llvm::createMemCpyLoopKnownSize(
    Instruction *InsertBefore, Value *SrcAddr, Value *DstAddr,
    ConstantInt *CopyLen, Align SrcAlign, Align DstAlign, bool SrcIsVolatile,
    bool DstIsVolatile, bool CanOverlap, const TargetTransformInfo &TTI,
    std::optional<uint32_t> AtomicElementSize) {
  // No need to expand zero length copies.
  if (CopyLen->isZero())
    return;

  BasicBlock *PreLoopBB = InsertBefore->getParent();
  BasicBlock *PostLoopBB = nullptr;
  Function *ParentFunc = PreLoopBB->getParent();
  LLVMContext &Ctx = PreLoopBB->getContext();
  const DataLayout &DL = ParentFunc->getDataLayout();
  MDBuilder MDB(Ctx);
  MDNode *NewDomain = MDB.createAnonymousAliasScopeDomain("MemCopyDomain");
  StringRef Name = "MemCopyAliasScope";
  MDNode *NewScope = MDB.createAnonymousAliasScope(NewDomain, Name);

  unsigned SrcAS = cast<PointerType>(SrcAddr->getType())->getAddressSpace();
  unsigned DstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();

  Type *TypeOfCopyLen = CopyLen->getType();
  Type *LoopOpType = TTI.getMemcpyLoopLoweringType(
      Ctx, CopyLen, SrcAS, DstAS, SrcAlign, DstAlign, AtomicElementSize);
  assert((!AtomicElementSize || !LoopOpType->isVectorTy()) &&
         "Atomic memcpy lowering is not supported for vector operand type");

  Type *Int8Type = Type::getInt8Ty(Ctx);
  unsigned LoopOpSize = DL.getTypeStoreSize(LoopOpType);
  assert((!AtomicElementSize || LoopOpSize % *AtomicElementSize == 0) &&
         "Atomic memcpy lowering is not supported for selected operand size");

  uint64_t LoopEndCount = alignDown(CopyLen->getZExtValue(), LoopOpSize);

  if (LoopEndCount != 0) {
    // Split
    PostLoopBB = PreLoopBB->splitBasicBlock(InsertBefore, "memcpy-split");
    BasicBlock *LoopBB =
        BasicBlock::Create(Ctx, "load-store-loop", ParentFunc, PostLoopBB);
    PreLoopBB->getTerminator()->setSuccessor(0, LoopBB);

    IRBuilder<> PLBuilder(PreLoopBB->getTerminator());

    Align PartDstAlign(commonAlignment(DstAlign, LoopOpSize));
    Align PartSrcAlign(commonAlignment(SrcAlign, LoopOpSize));

    IRBuilder<> LoopBuilder(LoopBB);
    PHINode *LoopIndex = LoopBuilder.CreatePHI(TypeOfCopyLen, 2, "loop-index");
    LoopIndex->addIncoming(ConstantInt::get(TypeOfCopyLen, 0U), PreLoopBB);
    // Loop Body

    // If we used LoopOpType as GEP element type, we would iterate over the
    // buffers in TypeStoreSize strides while copying TypeAllocSize bytes, i.e.,
    // we would miss bytes if TypeStoreSize != TypeAllocSize. Therefore, use
    // byte offsets computed from the TypeStoreSize.
    Value *SrcGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, LoopIndex);
    LoadInst *Load = LoopBuilder.CreateAlignedLoad(LoopOpType, SrcGEP,
                                                   PartSrcAlign, SrcIsVolatile);
    if (!CanOverlap) {
      // Set alias scope for loads.
      Load->setMetadata(LLVMContext::MD_alias_scope,
                        MDNode::get(Ctx, NewScope));
    }
    Value *DstGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr, LoopIndex);
    StoreInst *Store = LoopBuilder.CreateAlignedStore(
        Load, DstGEP, PartDstAlign, DstIsVolatile);
    if (!CanOverlap) {
      // Indicate that stores don't overlap loads.
      Store->setMetadata(LLVMContext::MD_noalias, MDNode::get(Ctx, NewScope));
    }
    if (AtomicElementSize) {
      Load->setAtomic(AtomicOrdering::Unordered);
      Store->setAtomic(AtomicOrdering::Unordered);
    }
    Value *NewIndex = LoopBuilder.CreateAdd(
        LoopIndex, ConstantInt::get(TypeOfCopyLen, LoopOpSize));
    LoopIndex->addIncoming(NewIndex, LoopBB);

    // Create the loop branch condition.
    Constant *LoopEndCI = ConstantInt::get(TypeOfCopyLen, LoopEndCount);
    LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(NewIndex, LoopEndCI),
                             LoopBB, PostLoopBB);
  }

  uint64_t BytesCopied = LoopEndCount;
  uint64_t RemainingBytes = CopyLen->getZExtValue() - BytesCopied;
  if (RemainingBytes) {
    IRBuilder<> RBuilder(PostLoopBB ? PostLoopBB->getFirstNonPHI()
                                    : InsertBefore);

    SmallVector<Type *, 5> RemainingOps;
    TTI.getMemcpyLoopResidualLoweringType(RemainingOps, Ctx, RemainingBytes,
                                          SrcAS, DstAS, SrcAlign, DstAlign,
                                          AtomicElementSize);

    for (auto *OpTy : RemainingOps) {
      Align PartSrcAlign(commonAlignment(SrcAlign, BytesCopied));
      Align PartDstAlign(commonAlignment(DstAlign, BytesCopied));

      unsigned OperandSize = DL.getTypeStoreSize(OpTy);
      assert(
          (!AtomicElementSize || OperandSize % *AtomicElementSize == 0) &&
          "Atomic memcpy lowering is not supported for selected operand size");

      Value *SrcGEP = RBuilder.CreateInBoundsGEP(
          Int8Type, SrcAddr, ConstantInt::get(TypeOfCopyLen, BytesCopied));
      LoadInst *Load =
          RBuilder.CreateAlignedLoad(OpTy, SrcGEP, PartSrcAlign, SrcIsVolatile);
      if (!CanOverlap) {
        // Set alias scope for loads.
        Load->setMetadata(LLVMContext::MD_alias_scope,
                          MDNode::get(Ctx, NewScope));
      }
      Value *DstGEP = RBuilder.CreateInBoundsGEP(
          Int8Type, DstAddr, ConstantInt::get(TypeOfCopyLen, BytesCopied));
      StoreInst *Store = RBuilder.CreateAlignedStore(Load, DstGEP, PartDstAlign,
                                                     DstIsVolatile);
      if (!CanOverlap) {
        // Indicate that stores don't overlap loads.
        Store->setMetadata(LLVMContext::MD_noalias, MDNode::get(Ctx, NewScope));
      }
      if (AtomicElementSize) {
        Load->setAtomic(AtomicOrdering::Unordered);
        Store->setAtomic(AtomicOrdering::Unordered);
      }
      BytesCopied += OperandSize;
    }
  }
  assert(BytesCopied == CopyLen->getZExtValue() &&
         "Bytes copied should match size in the call!");
}

// \returns \p Len urem \p OpSize, checking for optimization opportunities.
static Value *getRuntimeLoopRemainder(const DataLayout &DL, IRBuilderBase &B,
                                      Value *Len, Value *OpSize,
                                      unsigned OpSizeVal) {
  // For powers of 2, we can and by (OpSizeVal - 1) instead of using urem.
  if (isPowerOf2_32(OpSizeVal))
    return B.CreateAnd(Len, OpSizeVal - 1);
  return B.CreateURem(Len, OpSize);
}

// \returns (\p Len udiv \p OpSize) mul \p OpSize, checking for optimization
// opportunities.
// If RTLoopRemainder is provided, it must be the result of
// getRuntimeLoopRemainder() with the same arguments.
static Value *getRuntimeLoopBytes(const DataLayout &DL, IRBuilderBase &B,
                                  Value *Len, Value *OpSize, unsigned OpSizeVal,
                                  Value *RTLoopRemainder = nullptr) {
  if (!RTLoopRemainder)
    RTLoopRemainder = getRuntimeLoopRemainder(DL, B, Len, OpSize, OpSizeVal);
  return B.CreateSub(Len, RTLoopRemainder);
}

void llvm::createMemCpyLoopUnknownSize(
    Instruction *InsertBefore, Value *SrcAddr, Value *DstAddr, Value *CopyLen,
    Align SrcAlign, Align DstAlign, bool SrcIsVolatile, bool DstIsVolatile,
    bool CanOverlap, const TargetTransformInfo &TTI,
    std::optional<uint32_t> AtomicElementSize) {
  BasicBlock *PreLoopBB = InsertBefore->getParent();
  BasicBlock *PostLoopBB =
      PreLoopBB->splitBasicBlock(InsertBefore, "post-loop-memcpy-expansion");

  Function *ParentFunc = PreLoopBB->getParent();
  const DataLayout &DL = ParentFunc->getDataLayout();
  LLVMContext &Ctx = PreLoopBB->getContext();
  MDBuilder MDB(Ctx);
  MDNode *NewDomain = MDB.createAnonymousAliasScopeDomain("MemCopyDomain");
  StringRef Name = "MemCopyAliasScope";
  MDNode *NewScope = MDB.createAnonymousAliasScope(NewDomain, Name);

  unsigned SrcAS = cast<PointerType>(SrcAddr->getType())->getAddressSpace();
  unsigned DstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();

  Type *LoopOpType = TTI.getMemcpyLoopLoweringType(
      Ctx, CopyLen, SrcAS, DstAS, SrcAlign, DstAlign, AtomicElementSize);
  assert((!AtomicElementSize || !LoopOpType->isVectorTy()) &&
         "Atomic memcpy lowering is not supported for vector operand type");
  unsigned LoopOpSize = DL.getTypeStoreSize(LoopOpType);
  assert((!AtomicElementSize || LoopOpSize % *AtomicElementSize == 0) &&
         "Atomic memcpy lowering is not supported for selected operand size");

  IRBuilder<> PLBuilder(PreLoopBB->getTerminator());

  // Calculate the loop trip count, and remaining bytes to copy after the loop.
  Type *CopyLenType = CopyLen->getType();
  IntegerType *ILengthType = dyn_cast<IntegerType>(CopyLenType);
  assert(ILengthType &&
         "expected size argument to memcpy to be an integer type!");
  Type *Int8Type = Type::getInt8Ty(Ctx);
  bool LoopOpIsInt8 = LoopOpType == Int8Type;
  ConstantInt *CILoopOpSize = ConstantInt::get(ILengthType, LoopOpSize);

  Value *RuntimeLoopBytes = CopyLen;
  Value *RuntimeResidualBytes = nullptr;
  if (!LoopOpIsInt8) {
    RuntimeResidualBytes = getRuntimeLoopRemainder(DL, PLBuilder, CopyLen,
                                                   CILoopOpSize, LoopOpSize);
    RuntimeLoopBytes = getRuntimeLoopBytes(DL, PLBuilder, CopyLen, CILoopOpSize,
                                           LoopOpSize, RuntimeResidualBytes);
  }

  BasicBlock *LoopBB =
      BasicBlock::Create(Ctx, "loop-memcpy-expansion", ParentFunc, PostLoopBB);
  IRBuilder<> LoopBuilder(LoopBB);

  Align PartSrcAlign(commonAlignment(SrcAlign, LoopOpSize));
  Align PartDstAlign(commonAlignment(DstAlign, LoopOpSize));

  PHINode *LoopIndex = LoopBuilder.CreatePHI(CopyLenType, 2, "loop-index");
  LoopIndex->addIncoming(ConstantInt::get(CopyLenType, 0U), PreLoopBB);

  // If we used LoopOpType as GEP element type, we would iterate over the
  // buffers in TypeStoreSize strides while copying TypeAllocSize bytes, i.e.,
  // we would miss bytes if TypeStoreSize != TypeAllocSize. Therefore, use byte
  // offsets computed from the TypeStoreSize.
  Value *SrcGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, LoopIndex);
  LoadInst *Load = LoopBuilder.CreateAlignedLoad(LoopOpType, SrcGEP,
                                                 PartSrcAlign, SrcIsVolatile);
  if (!CanOverlap) {
    // Set alias scope for loads.
    Load->setMetadata(LLVMContext::MD_alias_scope, MDNode::get(Ctx, NewScope));
  }
  Value *DstGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr, LoopIndex);
  StoreInst *Store =
      LoopBuilder.CreateAlignedStore(Load, DstGEP, PartDstAlign, DstIsVolatile);
  if (!CanOverlap) {
    // Indicate that stores don't overlap loads.
    Store->setMetadata(LLVMContext::MD_noalias, MDNode::get(Ctx, NewScope));
  }
  if (AtomicElementSize) {
    Load->setAtomic(AtomicOrdering::Unordered);
    Store->setAtomic(AtomicOrdering::Unordered);
  }
  Value *NewIndex = LoopBuilder.CreateAdd(
      LoopIndex, ConstantInt::get(CopyLenType, LoopOpSize));
  LoopIndex->addIncoming(NewIndex, LoopBB);

  bool RequiresResidual =
      !LoopOpIsInt8 && !(AtomicElementSize && LoopOpSize == AtomicElementSize);
  if (RequiresResidual) {
    Type *ResLoopOpType = AtomicElementSize
                              ? Type::getIntNTy(Ctx, *AtomicElementSize * 8)
                              : Int8Type;
    unsigned ResLoopOpSize = DL.getTypeStoreSize(ResLoopOpType);
    assert((ResLoopOpSize == AtomicElementSize ? *AtomicElementSize : 1) &&
           "Store size is expected to match type size");

    Align ResSrcAlign(commonAlignment(PartSrcAlign, ResLoopOpSize));
    Align ResDstAlign(commonAlignment(PartDstAlign, ResLoopOpSize));

    // Loop body for the residual copy.
    BasicBlock *ResLoopBB = BasicBlock::Create(
        Ctx, "loop-memcpy-residual", PreLoopBB->getParent(), PostLoopBB);
    // Residual loop header.
    BasicBlock *ResHeaderBB = BasicBlock::Create(
        Ctx, "loop-memcpy-residual-header", PreLoopBB->getParent(), nullptr);

    // Need to update the pre-loop basic block to branch to the correct place.
    // branch to the main loop if the count is non-zero, branch to the residual
    // loop if the copy size is smaller then 1 iteration of the main loop but
    // non-zero and finally branch to after the residual loop if the memcpy
    //  size is zero.
    ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);
    PLBuilder.CreateCondBr(PLBuilder.CreateICmpNE(RuntimeLoopBytes, Zero),
                           LoopBB, ResHeaderBB);
    PreLoopBB->getTerminator()->eraseFromParent();

    LoopBuilder.CreateCondBr(
        LoopBuilder.CreateICmpULT(NewIndex, RuntimeLoopBytes), LoopBB,
        ResHeaderBB);

    // Determine if we need to branch to the residual loop or bypass it.
    IRBuilder<> RHBuilder(ResHeaderBB);
    RHBuilder.CreateCondBr(RHBuilder.CreateICmpNE(RuntimeResidualBytes, Zero),
                           ResLoopBB, PostLoopBB);

    // Copy the residual with single byte load/store loop.
    IRBuilder<> ResBuilder(ResLoopBB);
    PHINode *ResidualIndex =
        ResBuilder.CreatePHI(CopyLenType, 2, "residual-loop-index");
    ResidualIndex->addIncoming(Zero, ResHeaderBB);

    Value *FullOffset = ResBuilder.CreateAdd(RuntimeLoopBytes, ResidualIndex);
    Value *SrcGEP = ResBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, FullOffset);
    LoadInst *Load = ResBuilder.CreateAlignedLoad(ResLoopOpType, SrcGEP,
                                                  ResSrcAlign, SrcIsVolatile);
    if (!CanOverlap) {
      // Set alias scope for loads.
      Load->setMetadata(LLVMContext::MD_alias_scope,
                        MDNode::get(Ctx, NewScope));
    }
    Value *DstGEP = ResBuilder.CreateInBoundsGEP(Int8Type, DstAddr, FullOffset);
    StoreInst *Store =
        ResBuilder.CreateAlignedStore(Load, DstGEP, ResDstAlign, DstIsVolatile);
    if (!CanOverlap) {
      // Indicate that stores don't overlap loads.
      Store->setMetadata(LLVMContext::MD_noalias, MDNode::get(Ctx, NewScope));
    }
    if (AtomicElementSize) {
      Load->setAtomic(AtomicOrdering::Unordered);
      Store->setAtomic(AtomicOrdering::Unordered);
    }
    Value *ResNewIndex = ResBuilder.CreateAdd(
        ResidualIndex, ConstantInt::get(CopyLenType, ResLoopOpSize));
    ResidualIndex->addIncoming(ResNewIndex, ResLoopBB);

    // Create the loop branch condition.
    ResBuilder.CreateCondBr(
        ResBuilder.CreateICmpULT(ResNewIndex, RuntimeResidualBytes), ResLoopBB,
        PostLoopBB);
  } else {
    // In this case the loop operand type was a byte, and there is no need for a
    // residual loop to copy the remaining memory after the main loop.
    // We do however need to patch up the control flow by creating the
    // terminators for the preloop block and the memcpy loop.
    ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);
    PLBuilder.CreateCondBr(PLBuilder.CreateICmpNE(RuntimeLoopBytes, Zero),
                           LoopBB, PostLoopBB);
    PreLoopBB->getTerminator()->eraseFromParent();
    LoopBuilder.CreateCondBr(
        LoopBuilder.CreateICmpULT(NewIndex, RuntimeLoopBytes), LoopBB,
        PostLoopBB);
  }
}

// If \p Addr1 and \p Addr2 are pointers to different address spaces, create an
// addresspacecast to obtain a pair of pointers in the same addressspace. The
// caller needs to ensure that addrspacecasting is possible.
// No-op if the pointers are in the same address space.
static std::pair<Value *, Value *>
tryInsertCastToCommonAddrSpace(IRBuilderBase &B, Value *Addr1, Value *Addr2,
                               const TargetTransformInfo &TTI) {
  Value *ResAddr1 = Addr1;
  Value *ResAddr2 = Addr2;

  unsigned AS1 = cast<PointerType>(Addr1->getType())->getAddressSpace();
  unsigned AS2 = cast<PointerType>(Addr2->getType())->getAddressSpace();
  if (AS1 != AS2) {
    if (TTI.isValidAddrSpaceCast(AS2, AS1))
      ResAddr2 = B.CreateAddrSpaceCast(Addr2, Addr1->getType());
    else if (TTI.isValidAddrSpaceCast(AS1, AS2))
      ResAddr1 = B.CreateAddrSpaceCast(Addr1, Addr2->getType());
    else
      llvm_unreachable("Can only lower memmove between address spaces if they "
                       "support addrspacecast");
  }
  return {ResAddr1, ResAddr2};
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
//
// If the TargetTransformInfo specifies a wider MemcpyLoopLoweringType, it is
// used for the memory accesses in the loops. Then, additional loops with
// byte-wise accesses are added for the remaining bytes.
static void createMemMoveLoopUnknownSize(Instruction *InsertBefore,
                                         Value *SrcAddr, Value *DstAddr,
                                         Value *CopyLen, Align SrcAlign,
                                         Align DstAlign, bool SrcIsVolatile,
                                         bool DstIsVolatile,
                                         const TargetTransformInfo &TTI) {
  Type *TypeOfCopyLen = CopyLen->getType();
  BasicBlock *OrigBB = InsertBefore->getParent();
  Function *F = OrigBB->getParent();
  const DataLayout &DL = F->getDataLayout();
  LLVMContext &Ctx = OrigBB->getContext();
  unsigned SrcAS = cast<PointerType>(SrcAddr->getType())->getAddressSpace();
  unsigned DstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();

  Type *LoopOpType = TTI.getMemcpyLoopLoweringType(Ctx, CopyLen, SrcAS, DstAS,
                                                   SrcAlign, DstAlign);
  unsigned LoopOpSize = DL.getTypeStoreSize(LoopOpType);
  Type *Int8Type = Type::getInt8Ty(Ctx);
  bool LoopOpIsInt8 = LoopOpType == Int8Type;

  // If the memory accesses are wider than one byte, residual loops with
  // i8-accesses are required to move remaining bytes.
  bool RequiresResidual = !LoopOpIsInt8;

  Type *ResidualLoopOpType = Int8Type;
  unsigned ResidualLoopOpSize = DL.getTypeStoreSize(ResidualLoopOpType);

  // Calculate the loop trip count and remaining bytes to copy after the loop.
  IntegerType *ILengthType = cast<IntegerType>(TypeOfCopyLen);
  ConstantInt *CILoopOpSize = ConstantInt::get(ILengthType, LoopOpSize);
  ConstantInt *CIResidualLoopOpSize =
      ConstantInt::get(ILengthType, ResidualLoopOpSize);
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0);

  IRBuilder<> PLBuilder(InsertBefore);

  Value *RuntimeLoopBytes = CopyLen;
  Value *RuntimeLoopRemainder = nullptr;
  Value *SkipResidualCondition = nullptr;
  if (RequiresResidual) {
    RuntimeLoopRemainder = getRuntimeLoopRemainder(DL, PLBuilder, CopyLen,
                                                   CILoopOpSize, LoopOpSize);
    RuntimeLoopBytes = getRuntimeLoopBytes(DL, PLBuilder, CopyLen, CILoopOpSize,
                                           LoopOpSize, RuntimeLoopRemainder);
    SkipResidualCondition =
        PLBuilder.CreateICmpEQ(RuntimeLoopRemainder, Zero, "skip_residual");
  }
  Value *SkipMainCondition =
      PLBuilder.CreateICmpEQ(RuntimeLoopBytes, Zero, "skip_main");

  // Create the a comparison of src and dst, based on which we jump to either
  // the forward-copy part of the function (if src >= dst) or the backwards-copy
  // part (if src < dst).
  // SplitBlockAndInsertIfThenElse conveniently creates the basic if-then-else
  // structure. Its block terminators (unconditional branches) are replaced by
  // the appropriate conditional branches when the loop is built.
  // If the pointers are in different address spaces, they need to be converted
  // to a compatible one. Cases where memory ranges in the different address
  // spaces cannot overlap are lowered as memcpy and not handled here.
  auto [CmpSrcAddr, CmpDstAddr] =
      tryInsertCastToCommonAddrSpace(PLBuilder, SrcAddr, DstAddr, TTI);
  Value *PtrCompare =
      PLBuilder.CreateICmpULT(CmpSrcAddr, CmpDstAddr, "compare_src_dst");
  Instruction *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(PtrCompare, InsertBefore->getIterator(),
                                &ThenTerm, &ElseTerm);

  // If the LoopOpSize is greater than 1, each part of the function consists of
  // four blocks:
  //   memmove_copy_backwards:
  //       skip the residual loop when 0 iterations are required
  //   memmove_bwd_residual_loop:
  //       copy the last few bytes individually so that the remaining length is
  //       a multiple of the LoopOpSize
  //   memmove_bwd_middle: skip the main loop when 0 iterations are required
  //   memmove_bwd_main_loop: the actual backwards loop BB with wide accesses
  //   memmove_copy_forward: skip the main loop when 0 iterations are required
  //   memmove_fwd_main_loop: the actual forward loop BB with wide accesses
  //   memmove_fwd_middle: skip the residual loop when 0 iterations are required
  //   memmove_fwd_residual_loop: copy the last few bytes individually
  //
  // The main and residual loop are switched between copying forward and
  // backward so that the residual loop always operates on the end of the moved
  // range. This is based on the assumption that buffers whose start is aligned
  // with the LoopOpSize are more common than buffers whose end is.
  //
  // If the LoopOpSize is 1, each part of the function consists of two blocks:
  //   memmove_copy_backwards: skip the loop when 0 iterations are required
  //   memmove_bwd_main_loop: the actual backwards loop BB
  //   memmove_copy_forward: skip the loop when 0 iterations are required
  //   memmove_fwd_main_loop: the actual forward loop BB
  BasicBlock *CopyBackwardsBB = ThenTerm->getParent();
  CopyBackwardsBB->setName("memmove_copy_backwards");
  BasicBlock *CopyForwardBB = ElseTerm->getParent();
  CopyForwardBB->setName("memmove_copy_forward");
  BasicBlock *ExitBB = InsertBefore->getParent();
  ExitBB->setName("memmove_done");

  Align PartSrcAlign(commonAlignment(SrcAlign, LoopOpSize));
  Align PartDstAlign(commonAlignment(DstAlign, LoopOpSize));

  // Accesses in the residual loops do not share the same alignment as those in
  // the main loops.
  Align ResidualSrcAlign(commonAlignment(PartSrcAlign, ResidualLoopOpSize));
  Align ResidualDstAlign(commonAlignment(PartDstAlign, ResidualLoopOpSize));

  // Copying backwards.
  {
    BasicBlock *MainLoopBB = BasicBlock::Create(
        F->getContext(), "memmove_bwd_main_loop", F, CopyForwardBB);

    // The predecessor of the memmove_bwd_main_loop. Updated in the
    // following if a residual loop is emitted first.
    BasicBlock *PredBB = CopyBackwardsBB;

    if (RequiresResidual) {
      // backwards residual loop
      BasicBlock *ResidualLoopBB = BasicBlock::Create(
          F->getContext(), "memmove_bwd_residual_loop", F, MainLoopBB);
      IRBuilder<> ResidualLoopBuilder(ResidualLoopBB);
      PHINode *ResidualLoopPhi = ResidualLoopBuilder.CreatePHI(ILengthType, 0);
      Value *ResidualIndex = ResidualLoopBuilder.CreateSub(
          ResidualLoopPhi, CIResidualLoopOpSize, "bwd_residual_index");
      // If we used LoopOpType as GEP element type, we would iterate over the
      // buffers in TypeStoreSize strides while copying TypeAllocSize bytes,
      // i.e., we would miss bytes if TypeStoreSize != TypeAllocSize. Therefore,
      // use byte offsets computed from the TypeStoreSize.
      Value *LoadGEP = ResidualLoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr,
                                                             ResidualIndex);
      Value *Element = ResidualLoopBuilder.CreateAlignedLoad(
          ResidualLoopOpType, LoadGEP, ResidualSrcAlign, SrcIsVolatile,
          "element");
      Value *StoreGEP = ResidualLoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr,
                                                              ResidualIndex);
      ResidualLoopBuilder.CreateAlignedStore(Element, StoreGEP,
                                             ResidualDstAlign, DstIsVolatile);

      // After the residual loop, go to an intermediate block.
      BasicBlock *IntermediateBB = BasicBlock::Create(
          F->getContext(), "memmove_bwd_middle", F, MainLoopBB);
      // Later code expects a terminator in the PredBB.
      IRBuilder<> IntermediateBuilder(IntermediateBB);
      IntermediateBuilder.CreateUnreachable();
      ResidualLoopBuilder.CreateCondBr(
          ResidualLoopBuilder.CreateICmpEQ(ResidualIndex, RuntimeLoopBytes),
          IntermediateBB, ResidualLoopBB);

      ResidualLoopPhi->addIncoming(ResidualIndex, ResidualLoopBB);
      ResidualLoopPhi->addIncoming(CopyLen, CopyBackwardsBB);

      // How to get to the residual:
      BranchInst::Create(IntermediateBB, ResidualLoopBB, SkipResidualCondition,
                         ThenTerm->getIterator());
      ThenTerm->eraseFromParent();

      PredBB = IntermediateBB;
    }

    // main loop
    IRBuilder<> MainLoopBuilder(MainLoopBB);
    PHINode *MainLoopPhi = MainLoopBuilder.CreatePHI(ILengthType, 0);
    Value *MainIndex =
        MainLoopBuilder.CreateSub(MainLoopPhi, CILoopOpSize, "bwd_main_index");
    Value *LoadGEP =
        MainLoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, MainIndex);
    Value *Element = MainLoopBuilder.CreateAlignedLoad(
        LoopOpType, LoadGEP, PartSrcAlign, SrcIsVolatile, "element");
    Value *StoreGEP =
        MainLoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr, MainIndex);
    MainLoopBuilder.CreateAlignedStore(Element, StoreGEP, PartDstAlign,
                                       DstIsVolatile);
    MainLoopBuilder.CreateCondBr(MainLoopBuilder.CreateICmpEQ(MainIndex, Zero),
                                 ExitBB, MainLoopBB);
    MainLoopPhi->addIncoming(MainIndex, MainLoopBB);
    MainLoopPhi->addIncoming(RuntimeLoopBytes, PredBB);

    // How to get to the main loop:
    Instruction *PredBBTerm = PredBB->getTerminator();
    BranchInst::Create(ExitBB, MainLoopBB, SkipMainCondition,
                       PredBBTerm->getIterator());
    PredBBTerm->eraseFromParent();
  }

  // Copying forward.
  // main loop
  {
    BasicBlock *MainLoopBB =
        BasicBlock::Create(F->getContext(), "memmove_fwd_main_loop", F, ExitBB);
    IRBuilder<> MainLoopBuilder(MainLoopBB);
    PHINode *MainLoopPhi =
        MainLoopBuilder.CreatePHI(ILengthType, 0, "fwd_main_index");
    Value *LoadGEP =
        MainLoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, MainLoopPhi);
    Value *Element = MainLoopBuilder.CreateAlignedLoad(
        LoopOpType, LoadGEP, PartSrcAlign, SrcIsVolatile, "element");
    Value *StoreGEP =
        MainLoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr, MainLoopPhi);
    MainLoopBuilder.CreateAlignedStore(Element, StoreGEP, PartDstAlign,
                                       DstIsVolatile);
    Value *MainIndex = MainLoopBuilder.CreateAdd(MainLoopPhi, CILoopOpSize);
    MainLoopPhi->addIncoming(MainIndex, MainLoopBB);
    MainLoopPhi->addIncoming(Zero, CopyForwardBB);

    Instruction *CopyFwdBBTerm = CopyForwardBB->getTerminator();
    BasicBlock *SuccessorBB = ExitBB;
    if (RequiresResidual)
      SuccessorBB =
          BasicBlock::Create(F->getContext(), "memmove_fwd_middle", F, ExitBB);

    // leaving or staying in the main loop
    MainLoopBuilder.CreateCondBr(
        MainLoopBuilder.CreateICmpEQ(MainIndex, RuntimeLoopBytes), SuccessorBB,
        MainLoopBB);

    // getting in or skipping the main loop
    BranchInst::Create(SuccessorBB, MainLoopBB, SkipMainCondition,
                       CopyFwdBBTerm->getIterator());
    CopyFwdBBTerm->eraseFromParent();

    if (RequiresResidual) {
      BasicBlock *IntermediateBB = SuccessorBB;
      IRBuilder<> IntermediateBuilder(IntermediateBB);
      BasicBlock *ResidualLoopBB = BasicBlock::Create(
          F->getContext(), "memmove_fwd_residual_loop", F, ExitBB);
      IntermediateBuilder.CreateCondBr(SkipResidualCondition, ExitBB,
                                       ResidualLoopBB);

      // Residual loop
      IRBuilder<> ResidualLoopBuilder(ResidualLoopBB);
      PHINode *ResidualLoopPhi =
          ResidualLoopBuilder.CreatePHI(ILengthType, 0, "fwd_residual_index");
      Value *LoadGEP = ResidualLoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr,
                                                             ResidualLoopPhi);
      Value *Element = ResidualLoopBuilder.CreateAlignedLoad(
          ResidualLoopOpType, LoadGEP, ResidualSrcAlign, SrcIsVolatile,
          "element");
      Value *StoreGEP = ResidualLoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr,
                                                              ResidualLoopPhi);
      ResidualLoopBuilder.CreateAlignedStore(Element, StoreGEP,
                                             ResidualDstAlign, DstIsVolatile);
      Value *ResidualIndex =
          ResidualLoopBuilder.CreateAdd(ResidualLoopPhi, CIResidualLoopOpSize);
      ResidualLoopBuilder.CreateCondBr(
          ResidualLoopBuilder.CreateICmpEQ(ResidualIndex, CopyLen), ExitBB,
          ResidualLoopBB);
      ResidualLoopPhi->addIncoming(ResidualIndex, ResidualLoopBB);
      ResidualLoopPhi->addIncoming(RuntimeLoopBytes, IntermediateBB);
    }
  }
}

// Similar to createMemMoveLoopUnknownSize, only the trip counts are computed at
// compile time, obsolete loops and branches are omitted, and the residual code
// is straight-line code instead of a loop.
static void createMemMoveLoopKnownSize(Instruction *InsertBefore,
                                       Value *SrcAddr, Value *DstAddr,
                                       ConstantInt *CopyLen, Align SrcAlign,
                                       Align DstAlign, bool SrcIsVolatile,
                                       bool DstIsVolatile,
                                       const TargetTransformInfo &TTI) {
  // No need to expand zero length moves.
  if (CopyLen->isZero())
    return;

  Type *TypeOfCopyLen = CopyLen->getType();
  BasicBlock *OrigBB = InsertBefore->getParent();
  Function *F = OrigBB->getParent();
  const DataLayout &DL = F->getDataLayout();
  LLVMContext &Ctx = OrigBB->getContext();
  unsigned SrcAS = cast<PointerType>(SrcAddr->getType())->getAddressSpace();
  unsigned DstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();

  Type *LoopOpType = TTI.getMemcpyLoopLoweringType(Ctx, CopyLen, SrcAS, DstAS,
                                                   SrcAlign, DstAlign);
  unsigned LoopOpSize = DL.getTypeStoreSize(LoopOpType);
  Type *Int8Type = Type::getInt8Ty(Ctx);

  // Calculate the loop trip count and remaining bytes to copy after the loop.
  uint64_t BytesCopiedInLoop = alignDown(CopyLen->getZExtValue(), LoopOpSize);
  uint64_t RemainingBytes = CopyLen->getZExtValue() - BytesCopiedInLoop;

  IntegerType *ILengthType = cast<IntegerType>(TypeOfCopyLen);
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0);
  ConstantInt *LoopBound = ConstantInt::get(ILengthType, BytesCopiedInLoop);
  ConstantInt *CILoopOpSize = ConstantInt::get(ILengthType, LoopOpSize);

  IRBuilder<> PLBuilder(InsertBefore);

  auto [CmpSrcAddr, CmpDstAddr] =
      tryInsertCastToCommonAddrSpace(PLBuilder, SrcAddr, DstAddr, TTI);
  Value *PtrCompare =
      PLBuilder.CreateICmpULT(CmpSrcAddr, CmpDstAddr, "compare_src_dst");
  Instruction *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(PtrCompare, InsertBefore->getIterator(),
                                &ThenTerm, &ElseTerm);

  BasicBlock *CopyBackwardsBB = ThenTerm->getParent();
  BasicBlock *CopyForwardBB = ElseTerm->getParent();
  BasicBlock *ExitBB = InsertBefore->getParent();
  ExitBB->setName("memmove_done");

  Align PartSrcAlign(commonAlignment(SrcAlign, LoopOpSize));
  Align PartDstAlign(commonAlignment(DstAlign, LoopOpSize));

  // Helper function to generate a load/store pair of a given type in the
  // residual. Used in the forward and backward branches.
  auto GenerateResidualLdStPair = [&](Type *OpTy, IRBuilderBase &Builder,
                                      uint64_t &BytesCopied) {
    Align ResSrcAlign(commonAlignment(SrcAlign, BytesCopied));
    Align ResDstAlign(commonAlignment(DstAlign, BytesCopied));

    unsigned OperandSize = DL.getTypeStoreSize(OpTy);

    // If we used LoopOpType as GEP element type, we would iterate over the
    // buffers in TypeStoreSize strides while copying TypeAllocSize bytes, i.e.,
    // we would miss bytes if TypeStoreSize != TypeAllocSize. Therefore, use
    // byte offsets computed from the TypeStoreSize.
    Value *SrcGEP = Builder.CreateInBoundsGEP(
        Int8Type, SrcAddr, ConstantInt::get(TypeOfCopyLen, BytesCopied));
    LoadInst *Load =
        Builder.CreateAlignedLoad(OpTy, SrcGEP, ResSrcAlign, SrcIsVolatile);
    Value *DstGEP = Builder.CreateInBoundsGEP(
        Int8Type, DstAddr, ConstantInt::get(TypeOfCopyLen, BytesCopied));
    Builder.CreateAlignedStore(Load, DstGEP, ResDstAlign, DstIsVolatile);
    BytesCopied += OperandSize;
  };

  // Copying backwards.
  if (RemainingBytes != 0) {
    CopyBackwardsBB->setName("memmove_bwd_residual");
    uint64_t BytesCopied = BytesCopiedInLoop;

    // Residual code is required to move the remaining bytes. We need the same
    // instructions as in the forward case, only in reverse. So we generate code
    // the same way, except that we change the IRBuilder insert point for each
    // load/store pair so that each one is inserted before the previous one
    // instead of after it.
    IRBuilder<> BwdResBuilder(CopyBackwardsBB->getFirstNonPHI());
    SmallVector<Type *, 5> RemainingOps;
    TTI.getMemcpyLoopResidualLoweringType(RemainingOps, Ctx, RemainingBytes,
                                          SrcAS, DstAS, PartSrcAlign,
                                          PartDstAlign);
    for (auto *OpTy : RemainingOps) {
      // reverse the order of the emitted operations
      BwdResBuilder.SetInsertPoint(CopyBackwardsBB->getFirstNonPHI());
      GenerateResidualLdStPair(OpTy, BwdResBuilder, BytesCopied);
    }
  }
  if (BytesCopiedInLoop != 0) {
    BasicBlock *LoopBB = CopyBackwardsBB;
    BasicBlock *PredBB = OrigBB;
    if (RemainingBytes != 0) {
      // if we introduce residual code, it needs its separate BB
      LoopBB = CopyBackwardsBB->splitBasicBlock(
          CopyBackwardsBB->getTerminator(), "memmove_bwd_loop");
      PredBB = CopyBackwardsBB;
    } else {
      CopyBackwardsBB->setName("memmove_bwd_loop");
    }
    IRBuilder<> LoopBuilder(LoopBB->getTerminator());
    PHINode *LoopPhi = LoopBuilder.CreatePHI(ILengthType, 0);
    Value *Index = LoopBuilder.CreateSub(LoopPhi, CILoopOpSize, "bwd_index");
    Value *LoadGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, Index);
    Value *Element = LoopBuilder.CreateAlignedLoad(
        LoopOpType, LoadGEP, PartSrcAlign, SrcIsVolatile, "element");
    Value *StoreGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr, Index);
    LoopBuilder.CreateAlignedStore(Element, StoreGEP, PartDstAlign,
                                   DstIsVolatile);

    // Replace the unconditional branch introduced by
    // SplitBlockAndInsertIfThenElse to turn LoopBB into a loop.
    Instruction *UncondTerm = LoopBB->getTerminator();
    LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpEQ(Index, Zero), ExitBB,
                             LoopBB);
    UncondTerm->eraseFromParent();

    LoopPhi->addIncoming(Index, LoopBB);
    LoopPhi->addIncoming(LoopBound, PredBB);
  }

  // Copying forward.
  BasicBlock *FwdResidualBB = CopyForwardBB;
  if (BytesCopiedInLoop != 0) {
    CopyForwardBB->setName("memmove_fwd_loop");
    BasicBlock *LoopBB = CopyForwardBB;
    BasicBlock *SuccBB = ExitBB;
    if (RemainingBytes != 0) {
      // if we introduce residual code, it needs its separate BB
      SuccBB = CopyForwardBB->splitBasicBlock(CopyForwardBB->getTerminator(),
                                              "memmove_fwd_residual");
      FwdResidualBB = SuccBB;
    }
    IRBuilder<> LoopBuilder(LoopBB->getTerminator());
    PHINode *LoopPhi = LoopBuilder.CreatePHI(ILengthType, 0, "fwd_index");
    Value *LoadGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, SrcAddr, LoopPhi);
    Value *Element = LoopBuilder.CreateAlignedLoad(
        LoopOpType, LoadGEP, PartSrcAlign, SrcIsVolatile, "element");
    Value *StoreGEP = LoopBuilder.CreateInBoundsGEP(Int8Type, DstAddr, LoopPhi);
    LoopBuilder.CreateAlignedStore(Element, StoreGEP, PartDstAlign,
                                   DstIsVolatile);
    Value *Index = LoopBuilder.CreateAdd(LoopPhi, CILoopOpSize);
    LoopPhi->addIncoming(Index, LoopBB);
    LoopPhi->addIncoming(Zero, OrigBB);

    // Replace the unconditional branch to turn LoopBB into a loop.
    Instruction *UncondTerm = LoopBB->getTerminator();
    LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpEQ(Index, LoopBound), SuccBB,
                             LoopBB);
    UncondTerm->eraseFromParent();
  }

  if (RemainingBytes != 0) {
    uint64_t BytesCopied = BytesCopiedInLoop;

    // Residual code is required to move the remaining bytes. In the forward
    // case, we emit it in the normal order.
    IRBuilder<> FwdResBuilder(FwdResidualBB->getTerminator());
    SmallVector<Type *, 5> RemainingOps;
    TTI.getMemcpyLoopResidualLoweringType(RemainingOps, Ctx, RemainingBytes,
                                          SrcAS, DstAS, PartSrcAlign,
                                          PartDstAlign);
    for (auto *OpTy : RemainingOps)
      GenerateResidualLdStPair(OpTy, FwdResBuilder, BytesCopied);
  }
}

static void createMemSetLoop(Instruction *InsertBefore, Value *DstAddr,
                             Value *CopyLen, Value *SetValue, Align DstAlign,
                             bool IsVolatile) {
  Type *TypeOfCopyLen = CopyLen->getType();
  BasicBlock *OrigBB = InsertBefore->getParent();
  Function *F = OrigBB->getParent();
  const DataLayout &DL = F->getDataLayout();
  BasicBlock *NewBB =
      OrigBB->splitBasicBlock(InsertBefore, "split");
  BasicBlock *LoopBB
    = BasicBlock::Create(F->getContext(), "loadstoreloop", F, NewBB);

  IRBuilder<> Builder(OrigBB->getTerminator());

  Builder.CreateCondBr(
      Builder.CreateICmpEQ(ConstantInt::get(TypeOfCopyLen, 0), CopyLen), NewBB,
      LoopBB);
  OrigBB->getTerminator()->eraseFromParent();

  unsigned PartSize = DL.getTypeStoreSize(SetValue->getType());
  Align PartAlign(commonAlignment(DstAlign, PartSize));

  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopIndex = LoopBuilder.CreatePHI(TypeOfCopyLen, 0);
  LoopIndex->addIncoming(ConstantInt::get(TypeOfCopyLen, 0), OrigBB);

  LoopBuilder.CreateAlignedStore(
      SetValue,
      LoopBuilder.CreateInBoundsGEP(SetValue->getType(), DstAddr, LoopIndex),
      PartAlign, IsVolatile);

  Value *NewIndex =
      LoopBuilder.CreateAdd(LoopIndex, ConstantInt::get(TypeOfCopyLen, 1));
  LoopIndex->addIncoming(NewIndex, LoopBB);

  LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(NewIndex, CopyLen), LoopBB,
                           NewBB);
}

template <typename T>
static bool canOverlap(MemTransferBase<T> *Memcpy, ScalarEvolution *SE) {
  if (SE) {
    const SCEV *SrcSCEV = SE->getSCEV(Memcpy->getRawSource());
    const SCEV *DestSCEV = SE->getSCEV(Memcpy->getRawDest());
    if (SE->isKnownPredicateAt(CmpInst::ICMP_NE, SrcSCEV, DestSCEV, Memcpy))
      return false;
  }
  return true;
}

void llvm::expandMemCpyAsLoop(MemCpyInst *Memcpy,
                              const TargetTransformInfo &TTI,
                              ScalarEvolution *SE) {
  bool CanOverlap = canOverlap(Memcpy, SE);
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Memcpy->getLength())) {
    createMemCpyLoopKnownSize(
        /* InsertBefore */ Memcpy,
        /* SrcAddr */ Memcpy->getRawSource(),
        /* DstAddr */ Memcpy->getRawDest(),
        /* CopyLen */ CI,
        /* SrcAlign */ Memcpy->getSourceAlign().valueOrOne(),
        /* DestAlign */ Memcpy->getDestAlign().valueOrOne(),
        /* SrcIsVolatile */ Memcpy->isVolatile(),
        /* DstIsVolatile */ Memcpy->isVolatile(),
        /* CanOverlap */ CanOverlap,
        /* TargetTransformInfo */ TTI);
  } else {
    createMemCpyLoopUnknownSize(
        /* InsertBefore */ Memcpy,
        /* SrcAddr */ Memcpy->getRawSource(),
        /* DstAddr */ Memcpy->getRawDest(),
        /* CopyLen */ Memcpy->getLength(),
        /* SrcAlign */ Memcpy->getSourceAlign().valueOrOne(),
        /* DestAlign */ Memcpy->getDestAlign().valueOrOne(),
        /* SrcIsVolatile */ Memcpy->isVolatile(),
        /* DstIsVolatile */ Memcpy->isVolatile(),
        /* CanOverlap */ CanOverlap,
        /* TargetTransformInfo */ TTI);
  }
}

bool llvm::expandMemMoveAsLoop(MemMoveInst *Memmove,
                               const TargetTransformInfo &TTI) {
  Value *CopyLen = Memmove->getLength();
  Value *SrcAddr = Memmove->getRawSource();
  Value *DstAddr = Memmove->getRawDest();
  Align SrcAlign = Memmove->getSourceAlign().valueOrOne();
  Align DstAlign = Memmove->getDestAlign().valueOrOne();
  bool SrcIsVolatile = Memmove->isVolatile();
  bool DstIsVolatile = SrcIsVolatile;
  IRBuilder<> CastBuilder(Memmove);

  unsigned SrcAS = SrcAddr->getType()->getPointerAddressSpace();
  unsigned DstAS = DstAddr->getType()->getPointerAddressSpace();
  if (SrcAS != DstAS) {
    if (!TTI.addrspacesMayAlias(SrcAS, DstAS)) {
      // We may not be able to emit a pointer comparison, but we don't have
      // to. Expand as memcpy.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(CopyLen)) {
        createMemCpyLoopKnownSize(/*InsertBefore=*/Memmove, SrcAddr, DstAddr,
                                  CI, SrcAlign, DstAlign, SrcIsVolatile,
                                  DstIsVolatile,
                                  /*CanOverlap=*/false, TTI);
      } else {
        createMemCpyLoopUnknownSize(/*InsertBefore=*/Memmove, SrcAddr, DstAddr,
                                    CopyLen, SrcAlign, DstAlign, SrcIsVolatile,
                                    DstIsVolatile,
                                    /*CanOverlap=*/false, TTI);
      }

      return true;
    }

    if (!(TTI.isValidAddrSpaceCast(DstAS, SrcAS) ||
          TTI.isValidAddrSpaceCast(SrcAS, DstAS))) {
      // We don't know generically if it's legal to introduce an
      // addrspacecast. We need to know either if it's legal to insert an
      // addrspacecast, or if the address spaces cannot alias.
      LLVM_DEBUG(
          dbgs() << "Do not know how to expand memmove between different "
                    "address spaces\n");
      return false;
    }
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(CopyLen)) {
    createMemMoveLoopKnownSize(
        /*InsertBefore=*/Memmove, SrcAddr, DstAddr, CI, SrcAlign, DstAlign,
        SrcIsVolatile, DstIsVolatile, TTI);
  } else {
    createMemMoveLoopUnknownSize(
        /*InsertBefore=*/Memmove, SrcAddr, DstAddr, CopyLen, SrcAlign, DstAlign,
        SrcIsVolatile, DstIsVolatile, TTI);
  }
  return true;
}

void llvm::expandMemSetAsLoop(MemSetInst *Memset) {
  createMemSetLoop(/* InsertBefore */ Memset,
                   /* DstAddr */ Memset->getRawDest(),
                   /* CopyLen */ Memset->getLength(),
                   /* SetValue */ Memset->getValue(),
                   /* Alignment */ Memset->getDestAlign().valueOrOne(),
                   Memset->isVolatile());
}

void llvm::expandMemSetPatternAsLoop(MemSetPatternInst *Memset) {
  createMemSetLoop(/* InsertBefore=*/Memset,
                   /* DstAddr=*/Memset->getRawDest(),
                   /* CopyLen=*/Memset->getLength(),
                   /* SetValue=*/Memset->getValue(),
                   /* Alignment=*/Memset->getDestAlign().valueOrOne(),
                   Memset->isVolatile());
}

void llvm::expandAtomicMemCpyAsLoop(AtomicMemCpyInst *AtomicMemcpy,
                                    const TargetTransformInfo &TTI,
                                    ScalarEvolution *SE) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(AtomicMemcpy->getLength())) {
    createMemCpyLoopKnownSize(
        /* InsertBefore */ AtomicMemcpy,
        /* SrcAddr */ AtomicMemcpy->getRawSource(),
        /* DstAddr */ AtomicMemcpy->getRawDest(),
        /* CopyLen */ CI,
        /* SrcAlign */ AtomicMemcpy->getSourceAlign().valueOrOne(),
        /* DestAlign */ AtomicMemcpy->getDestAlign().valueOrOne(),
        /* SrcIsVolatile */ AtomicMemcpy->isVolatile(),
        /* DstIsVolatile */ AtomicMemcpy->isVolatile(),
        /* CanOverlap */ false, // SrcAddr & DstAddr may not overlap by spec.
        /* TargetTransformInfo */ TTI,
        /* AtomicCpySize */ AtomicMemcpy->getElementSizeInBytes());
  } else {
    createMemCpyLoopUnknownSize(
        /* InsertBefore */ AtomicMemcpy,
        /* SrcAddr */ AtomicMemcpy->getRawSource(),
        /* DstAddr */ AtomicMemcpy->getRawDest(),
        /* CopyLen */ AtomicMemcpy->getLength(),
        /* SrcAlign */ AtomicMemcpy->getSourceAlign().valueOrOne(),
        /* DestAlign */ AtomicMemcpy->getDestAlign().valueOrOne(),
        /* SrcIsVolatile */ AtomicMemcpy->isVolatile(),
        /* DstIsVolatile */ AtomicMemcpy->isVolatile(),
        /* CanOverlap */ false, // SrcAddr & DstAddr may not overlap by spec.
        /* TargetTransformInfo */ TTI,
        /* AtomicCpySize */ AtomicMemcpy->getElementSizeInBytes());
  }
}
