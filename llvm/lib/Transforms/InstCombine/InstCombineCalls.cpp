//===- InstCombineCalls.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitCall, visitInvoke, and visitCallBr functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

STATISTIC(NumSimplified, "Number of library calls simplified");

static cl::opt<unsigned> GuardWideningWindow(
    "instcombine-guard-widening-window",
    cl::init(3),
    cl::desc("How wide an instruction window to bypass looking for "
             "another guard"));

/// Return the specified type promoted as it would be to pass though a va_arg
/// area.
static Type *getPromotedType(Type *Ty) {
  if (IntegerType* ITy = dyn_cast<IntegerType>(Ty)) {
    if (ITy->getBitWidth() < 32)
      return Type::getInt32Ty(Ty->getContext());
  }
  return Ty;
}

/// Return a constant boolean vector that has true elements in all positions
/// where the input constant data vector has an element with the sign bit set.
static Constant *getNegativeIsTrueBoolVec(ConstantDataVector *V) {
  SmallVector<Constant *, 32> BoolVec;
  IntegerType *BoolTy = Type::getInt1Ty(V->getContext());
  for (unsigned I = 0, E = V->getNumElements(); I != E; ++I) {
    Constant *Elt = V->getElementAsConstant(I);
    assert((isa<ConstantInt>(Elt) || isa<ConstantFP>(Elt)) &&
           "Unexpected constant data vector element type");
    bool Sign = V->getElementType()->isIntegerTy()
                    ? cast<ConstantInt>(Elt)->isNegative()
                    : cast<ConstantFP>(Elt)->isNegative();
    BoolVec.push_back(ConstantInt::get(BoolTy, Sign));
  }
  return ConstantVector::get(BoolVec);
}

Instruction *InstCombiner::SimplifyAnyMemTransfer(AnyMemTransferInst *MI) {
  Align DstAlign = getKnownAlignment(MI->getRawDest(), DL, MI, &AC, &DT);
  MaybeAlign CopyDstAlign = MI->getDestAlign();
  if (!CopyDstAlign || *CopyDstAlign < DstAlign) {
    MI->setDestAlignment(DstAlign);
    return MI;
  }

  Align SrcAlign = getKnownAlignment(MI->getRawSource(), DL, MI, &AC, &DT);
  MaybeAlign CopySrcAlign = MI->getSourceAlign();
  if (!CopySrcAlign || *CopySrcAlign < SrcAlign) {
    MI->setSourceAlignment(SrcAlign);
    return MI;
  }

  // If we have a store to a location which is known constant, we can conclude
  // that the store must be storing the constant value (else the memory
  // wouldn't be constant), and this must be a noop.
  if (AA->pointsToConstantMemory(MI->getDest())) {
    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(MI->getLength()->getType()));
    return MI;
  }

  // If MemCpyInst length is 1/2/4/8 bytes then replace memcpy with
  // load/store.
  ConstantInt *MemOpLength = dyn_cast<ConstantInt>(MI->getLength());
  if (!MemOpLength) return nullptr;

  // Source and destination pointer types are always "i8*" for intrinsic.  See
  // if the size is something we can handle with a single primitive load/store.
  // A single load+store correctly handles overlapping memory in the memmove
  // case.
  uint64_t Size = MemOpLength->getLimitedValue();
  assert(Size && "0-sized memory transferring should be removed already.");

  if (Size > 8 || (Size&(Size-1)))
    return nullptr;  // If not 1/2/4/8 bytes, exit.

  // If it is an atomic and alignment is less than the size then we will
  // introduce the unaligned memory access which will be later transformed
  // into libcall in CodeGen. This is not evident performance gain so disable
  // it now.
  if (isa<AtomicMemTransferInst>(MI))
    if (CopyDstAlign < Size || CopySrcAlign < Size)
      return nullptr;

  // Use an integer load+store unless we can find something better.
  unsigned SrcAddrSp =
    cast<PointerType>(MI->getArgOperand(1)->getType())->getAddressSpace();
  unsigned DstAddrSp =
    cast<PointerType>(MI->getArgOperand(0)->getType())->getAddressSpace();

  IntegerType* IntType = IntegerType::get(MI->getContext(), Size<<3);
  Type *NewSrcPtrTy = PointerType::get(IntType, SrcAddrSp);
  Type *NewDstPtrTy = PointerType::get(IntType, DstAddrSp);

  // If the memcpy has metadata describing the members, see if we can get the
  // TBAA tag describing our copy.
  MDNode *CopyMD = nullptr;
  if (MDNode *M = MI->getMetadata(LLVMContext::MD_tbaa)) {
    CopyMD = M;
  } else if (MDNode *M = MI->getMetadata(LLVMContext::MD_tbaa_struct)) {
    if (M->getNumOperands() == 3 && M->getOperand(0) &&
        mdconst::hasa<ConstantInt>(M->getOperand(0)) &&
        mdconst::extract<ConstantInt>(M->getOperand(0))->isZero() &&
        M->getOperand(1) &&
        mdconst::hasa<ConstantInt>(M->getOperand(1)) &&
        mdconst::extract<ConstantInt>(M->getOperand(1))->getValue() ==
        Size &&
        M->getOperand(2) && isa<MDNode>(M->getOperand(2)))
      CopyMD = cast<MDNode>(M->getOperand(2));
  }

  Value *Src = Builder.CreateBitCast(MI->getArgOperand(1), NewSrcPtrTy);
  Value *Dest = Builder.CreateBitCast(MI->getArgOperand(0), NewDstPtrTy);
  LoadInst *L = Builder.CreateLoad(IntType, Src);
  // Alignment from the mem intrinsic will be better, so use it.
  L->setAlignment(*CopySrcAlign);
  if (CopyMD)
    L->setMetadata(LLVMContext::MD_tbaa, CopyMD);
  MDNode *LoopMemParallelMD =
    MI->getMetadata(LLVMContext::MD_mem_parallel_loop_access);
  if (LoopMemParallelMD)
    L->setMetadata(LLVMContext::MD_mem_parallel_loop_access, LoopMemParallelMD);
  MDNode *AccessGroupMD = MI->getMetadata(LLVMContext::MD_access_group);
  if (AccessGroupMD)
    L->setMetadata(LLVMContext::MD_access_group, AccessGroupMD);

  StoreInst *S = Builder.CreateStore(L, Dest);
  // Alignment from the mem intrinsic will be better, so use it.
  S->setAlignment(
      MaybeAlign(CopyDstAlign)); // FIXME: Check if we can use Align instead.
  if (CopyMD)
    S->setMetadata(LLVMContext::MD_tbaa, CopyMD);
  if (LoopMemParallelMD)
    S->setMetadata(LLVMContext::MD_mem_parallel_loop_access, LoopMemParallelMD);
  if (AccessGroupMD)
    S->setMetadata(LLVMContext::MD_access_group, AccessGroupMD);

  if (auto *MT = dyn_cast<MemTransferInst>(MI)) {
    // non-atomics can be volatile
    L->setVolatile(MT->isVolatile());
    S->setVolatile(MT->isVolatile());
  }
  if (isa<AtomicMemTransferInst>(MI)) {
    // atomics have to be unordered
    L->setOrdering(AtomicOrdering::Unordered);
    S->setOrdering(AtomicOrdering::Unordered);
  }

  // Set the size of the copy to 0, it will be deleted on the next iteration.
  MI->setLength(Constant::getNullValue(MemOpLength->getType()));
  return MI;
}

Instruction *InstCombiner::SimplifyAnyMemSet(AnyMemSetInst *MI) {
  const Align KnownAlignment =
      getKnownAlignment(MI->getDest(), DL, MI, &AC, &DT);
  MaybeAlign MemSetAlign = MI->getDestAlign();
  if (!MemSetAlign || *MemSetAlign < KnownAlignment) {
    MI->setDestAlignment(KnownAlignment);
    return MI;
  }

  // If we have a store to a location which is known constant, we can conclude
  // that the store must be storing the constant value (else the memory
  // wouldn't be constant), and this must be a noop.
  if (AA->pointsToConstantMemory(MI->getDest())) {
    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(MI->getLength()->getType()));
    return MI;
  }

  // Extract the length and alignment and fill if they are constant.
  ConstantInt *LenC = dyn_cast<ConstantInt>(MI->getLength());
  ConstantInt *FillC = dyn_cast<ConstantInt>(MI->getValue());
  if (!LenC || !FillC || !FillC->getType()->isIntegerTy(8))
    return nullptr;
  const uint64_t Len = LenC->getLimitedValue();
  assert(Len && "0-sized memory setting should be removed already.");
  const Align Alignment = assumeAligned(MI->getDestAlignment());

  // If it is an atomic and alignment is less than the size then we will
  // introduce the unaligned memory access which will be later transformed
  // into libcall in CodeGen. This is not evident performance gain so disable
  // it now.
  if (isa<AtomicMemSetInst>(MI))
    if (Alignment < Len)
      return nullptr;

  // memset(s,c,n) -> store s, c (for n=1,2,4,8)
  if (Len <= 8 && isPowerOf2_32((uint32_t)Len)) {
    Type *ITy = IntegerType::get(MI->getContext(), Len*8);  // n=1 -> i8.

    Value *Dest = MI->getDest();
    unsigned DstAddrSp = cast<PointerType>(Dest->getType())->getAddressSpace();
    Type *NewDstPtrTy = PointerType::get(ITy, DstAddrSp);
    Dest = Builder.CreateBitCast(Dest, NewDstPtrTy);

    // Extract the fill value and store.
    uint64_t Fill = FillC->getZExtValue()*0x0101010101010101ULL;
    StoreInst *S = Builder.CreateStore(ConstantInt::get(ITy, Fill), Dest,
                                       MI->isVolatile());
    S->setAlignment(Alignment);
    if (isa<AtomicMemSetInst>(MI))
      S->setOrdering(AtomicOrdering::Unordered);

    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(LenC->getType()));
    return MI;
  }

  return nullptr;
}

static Value *simplifyX86immShift(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  bool LogicalShift = false;
  bool ShiftLeft = false;
  bool IsImm = false;

  switch (II.getIntrinsicID()) {
  default: llvm_unreachable("Unexpected intrinsic!");
  case Intrinsic::x86_sse2_psrai_d:
  case Intrinsic::x86_sse2_psrai_w:
  case Intrinsic::x86_avx2_psrai_d:
  case Intrinsic::x86_avx2_psrai_w:
  case Intrinsic::x86_avx512_psrai_q_128:
  case Intrinsic::x86_avx512_psrai_q_256:
  case Intrinsic::x86_avx512_psrai_d_512:
  case Intrinsic::x86_avx512_psrai_q_512:
  case Intrinsic::x86_avx512_psrai_w_512:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case Intrinsic::x86_sse2_psra_d:
  case Intrinsic::x86_sse2_psra_w:
  case Intrinsic::x86_avx2_psra_d:
  case Intrinsic::x86_avx2_psra_w:
  case Intrinsic::x86_avx512_psra_q_128:
  case Intrinsic::x86_avx512_psra_q_256:
  case Intrinsic::x86_avx512_psra_d_512:
  case Intrinsic::x86_avx512_psra_q_512:
  case Intrinsic::x86_avx512_psra_w_512:
    LogicalShift = false;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_avx2_psrli_d:
  case Intrinsic::x86_avx2_psrli_q:
  case Intrinsic::x86_avx2_psrli_w:
  case Intrinsic::x86_avx512_psrli_d_512:
  case Intrinsic::x86_avx512_psrli_q_512:
  case Intrinsic::x86_avx512_psrli_w_512:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case Intrinsic::x86_sse2_psrl_d:
  case Intrinsic::x86_sse2_psrl_q:
  case Intrinsic::x86_sse2_psrl_w:
  case Intrinsic::x86_avx2_psrl_d:
  case Intrinsic::x86_avx2_psrl_q:
  case Intrinsic::x86_avx2_psrl_w:
  case Intrinsic::x86_avx512_psrl_d_512:
  case Intrinsic::x86_avx512_psrl_q_512:
  case Intrinsic::x86_avx512_psrl_w_512:
    LogicalShift = true;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_avx2_pslli_d:
  case Intrinsic::x86_avx2_pslli_q:
  case Intrinsic::x86_avx2_pslli_w:
  case Intrinsic::x86_avx512_pslli_d_512:
  case Intrinsic::x86_avx512_pslli_q_512:
  case Intrinsic::x86_avx512_pslli_w_512:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case Intrinsic::x86_sse2_psll_d:
  case Intrinsic::x86_sse2_psll_q:
  case Intrinsic::x86_sse2_psll_w:
  case Intrinsic::x86_avx2_psll_d:
  case Intrinsic::x86_avx2_psll_q:
  case Intrinsic::x86_avx2_psll_w:
  case Intrinsic::x86_avx512_psll_d_512:
  case Intrinsic::x86_avx512_psll_q_512:
  case Intrinsic::x86_avx512_psll_w_512:
    LogicalShift = true;
    ShiftLeft = true;
    break;
  }
  assert((LogicalShift || !ShiftLeft) && "Only logical shifts can shift left");

  auto Vec = II.getArgOperand(0);
  auto Amt = II.getArgOperand(1);
  auto VT = cast<VectorType>(Vec->getType());
  auto SVT = VT->getElementType();
  auto AmtVT = Amt->getType();
  unsigned VWidth = VT->getNumElements();
  unsigned BitWidth = SVT->getPrimitiveSizeInBits();

  // If the shift amount is guaranteed to be in-range we can replace it with a
  // generic shift. If its guaranteed to be out of range, logical shifts combine to
  // zero and arithmetic shifts are clamped to (BitWidth - 1).
  if (IsImm) {
    assert(AmtVT ->isIntegerTy(32) &&
           "Unexpected shift-by-immediate type");
    KnownBits KnownAmtBits =
        llvm::computeKnownBits(Amt, II.getModule()->getDataLayout());
    if (KnownAmtBits.getMaxValue().ult(BitWidth)) {
      Amt = Builder.CreateZExtOrTrunc(Amt, SVT);
      Amt = Builder.CreateVectorSplat(VWidth, Amt);
      return (LogicalShift ? (ShiftLeft ? Builder.CreateShl(Vec, Amt)
                                        : Builder.CreateLShr(Vec, Amt))
                           : Builder.CreateAShr(Vec, Amt));
    }
    if (KnownAmtBits.getMinValue().uge(BitWidth)) {
      if (LogicalShift)
        return ConstantAggregateZero::get(VT);
      Amt = ConstantInt::get(SVT, BitWidth - 1);
      return Builder.CreateAShr(Vec, Builder.CreateVectorSplat(VWidth, Amt));
    }
  } else {
    // Ensure the first element has an in-range value and the rest of the
    // elements in the bottom 64 bits are zero.
    assert(AmtVT->isVectorTy() && AmtVT->getPrimitiveSizeInBits() == 128 &&
           cast<VectorType>(AmtVT)->getElementType() == SVT &&
           "Unexpected shift-by-scalar type");
    unsigned NumAmtElts = cast<VectorType>(AmtVT)->getNumElements();
    APInt DemandedLower = APInt::getOneBitSet(NumAmtElts, 0);
    APInt DemandedUpper = APInt::getBitsSet(NumAmtElts, 1, NumAmtElts / 2);
    KnownBits KnownLowerBits = llvm::computeKnownBits(
        Amt, DemandedLower, II.getModule()->getDataLayout());
    KnownBits KnownUpperBits = llvm::computeKnownBits(
        Amt, DemandedUpper, II.getModule()->getDataLayout());
    if (KnownLowerBits.getMaxValue().ult(BitWidth) &&
        (DemandedUpper.isNullValue() || KnownUpperBits.isZero())) {
      SmallVector<int, 16> ZeroSplat(VWidth, 0);
      Amt = Builder.CreateShuffleVector(Amt, Amt, ZeroSplat);
      return (LogicalShift ? (ShiftLeft ? Builder.CreateShl(Vec, Amt)
                                        : Builder.CreateLShr(Vec, Amt))
                           : Builder.CreateAShr(Vec, Amt));
    }
  }

  // Simplify if count is constant vector.
  auto CDV = dyn_cast<ConstantDataVector>(Amt);
  if (!CDV)
    return nullptr;

  // SSE2/AVX2 uses all the first 64-bits of the 128-bit vector
  // operand to compute the shift amount.
  assert(AmtVT->isVectorTy() && AmtVT->getPrimitiveSizeInBits() == 128 &&
         cast<VectorType>(AmtVT)->getElementType() == SVT &&
         "Unexpected shift-by-scalar type");

  // Concatenate the sub-elements to create the 64-bit value.
  APInt Count(64, 0);
  for (unsigned i = 0, NumSubElts = 64 / BitWidth; i != NumSubElts; ++i) {
    unsigned SubEltIdx = (NumSubElts - 1) - i;
    auto SubElt = cast<ConstantInt>(CDV->getElementAsConstant(SubEltIdx));
    Count <<= BitWidth;
    Count |= SubElt->getValue().zextOrTrunc(64);
  }

  // If shift-by-zero then just return the original value.
  if (Count.isNullValue())
    return Vec;

  // Handle cases when Shift >= BitWidth.
  if (Count.uge(BitWidth)) {
    // If LogicalShift - just return zero.
    if (LogicalShift)
      return ConstantAggregateZero::get(VT);

    // If ArithmeticShift - clamp Shift to (BitWidth - 1).
    Count = APInt(64, BitWidth - 1);
  }

  // Get a constant vector of the same type as the first operand.
  auto ShiftAmt = ConstantInt::get(SVT, Count.zextOrTrunc(BitWidth));
  auto ShiftVec = Builder.CreateVectorSplat(VWidth, ShiftAmt);

  if (ShiftLeft)
    return Builder.CreateShl(Vec, ShiftVec);

  if (LogicalShift)
    return Builder.CreateLShr(Vec, ShiftVec);

  return Builder.CreateAShr(Vec, ShiftVec);
}

// Attempt to simplify AVX2 per-element shift intrinsics to a generic IR shift.
// Unlike the generic IR shifts, the intrinsics have defined behaviour for out
// of range shift amounts (logical - set to zero, arithmetic - splat sign bit).
static Value *simplifyX86varShift(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  bool LogicalShift = false;
  bool ShiftLeft = false;

  switch (II.getIntrinsicID()) {
  default: llvm_unreachable("Unexpected intrinsic!");
  case Intrinsic::x86_avx2_psrav_d:
  case Intrinsic::x86_avx2_psrav_d_256:
  case Intrinsic::x86_avx512_psrav_q_128:
  case Intrinsic::x86_avx512_psrav_q_256:
  case Intrinsic::x86_avx512_psrav_d_512:
  case Intrinsic::x86_avx512_psrav_q_512:
  case Intrinsic::x86_avx512_psrav_w_128:
  case Intrinsic::x86_avx512_psrav_w_256:
  case Intrinsic::x86_avx512_psrav_w_512:
    LogicalShift = false;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_avx2_psrlv_d:
  case Intrinsic::x86_avx2_psrlv_d_256:
  case Intrinsic::x86_avx2_psrlv_q:
  case Intrinsic::x86_avx2_psrlv_q_256:
  case Intrinsic::x86_avx512_psrlv_d_512:
  case Intrinsic::x86_avx512_psrlv_q_512:
  case Intrinsic::x86_avx512_psrlv_w_128:
  case Intrinsic::x86_avx512_psrlv_w_256:
  case Intrinsic::x86_avx512_psrlv_w_512:
    LogicalShift = true;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_avx2_psllv_d:
  case Intrinsic::x86_avx2_psllv_d_256:
  case Intrinsic::x86_avx2_psllv_q:
  case Intrinsic::x86_avx2_psllv_q_256:
  case Intrinsic::x86_avx512_psllv_d_512:
  case Intrinsic::x86_avx512_psllv_q_512:
  case Intrinsic::x86_avx512_psllv_w_128:
  case Intrinsic::x86_avx512_psllv_w_256:
  case Intrinsic::x86_avx512_psllv_w_512:
    LogicalShift = true;
    ShiftLeft = true;
    break;
  }
  assert((LogicalShift || !ShiftLeft) && "Only logical shifts can shift left");

  auto Vec = II.getArgOperand(0);
  auto Amt = II.getArgOperand(1);
  auto VT = cast<VectorType>(II.getType());
  auto SVT = VT->getElementType();
  int NumElts = VT->getNumElements();
  int BitWidth = SVT->getIntegerBitWidth();

  // If the shift amount is guaranteed to be in-range we can replace it with a
  // generic shift.
  APInt UpperBits =
      APInt::getHighBitsSet(BitWidth, BitWidth - Log2_32(BitWidth));
  if (llvm::MaskedValueIsZero(Amt, UpperBits,
                              II.getModule()->getDataLayout())) {
    return (LogicalShift ? (ShiftLeft ? Builder.CreateShl(Vec, Amt)
                                      : Builder.CreateLShr(Vec, Amt))
                         : Builder.CreateAShr(Vec, Amt));
  }

  // Simplify if all shift amounts are constant/undef.
  auto *CShift = dyn_cast<Constant>(Amt);
  if (!CShift)
    return nullptr;

  // Collect each element's shift amount.
  // We also collect special cases: UNDEF = -1, OUT-OF-RANGE = BitWidth.
  bool AnyOutOfRange = false;
  SmallVector<int, 8> ShiftAmts;
  for (int I = 0; I < NumElts; ++I) {
    auto *CElt = CShift->getAggregateElement(I);
    if (CElt && isa<UndefValue>(CElt)) {
      ShiftAmts.push_back(-1);
      continue;
    }

    auto *COp = dyn_cast_or_null<ConstantInt>(CElt);
    if (!COp)
      return nullptr;

    // Handle out of range shifts.
    // If LogicalShift - set to BitWidth (special case).
    // If ArithmeticShift - set to (BitWidth - 1) (sign splat).
    APInt ShiftVal = COp->getValue();
    if (ShiftVal.uge(BitWidth)) {
      AnyOutOfRange = LogicalShift;
      ShiftAmts.push_back(LogicalShift ? BitWidth : BitWidth - 1);
      continue;
    }

    ShiftAmts.push_back((int)ShiftVal.getZExtValue());
  }

  // If all elements out of range or UNDEF, return vector of zeros/undefs.
  // ArithmeticShift should only hit this if they are all UNDEF.
  auto OutOfRange = [&](int Idx) { return (Idx < 0) || (BitWidth <= Idx); };
  if (llvm::all_of(ShiftAmts, OutOfRange)) {
    SmallVector<Constant *, 8> ConstantVec;
    for (int Idx : ShiftAmts) {
      if (Idx < 0) {
        ConstantVec.push_back(UndefValue::get(SVT));
      } else {
        assert(LogicalShift && "Logical shift expected");
        ConstantVec.push_back(ConstantInt::getNullValue(SVT));
      }
    }
    return ConstantVector::get(ConstantVec);
  }

  // We can't handle only some out of range values with generic logical shifts.
  if (AnyOutOfRange)
    return nullptr;

  // Build the shift amount constant vector.
  SmallVector<Constant *, 8> ShiftVecAmts;
  for (int Idx : ShiftAmts) {
    if (Idx < 0)
      ShiftVecAmts.push_back(UndefValue::get(SVT));
    else
      ShiftVecAmts.push_back(ConstantInt::get(SVT, Idx));
  }
  auto ShiftVec = ConstantVector::get(ShiftVecAmts);

  if (ShiftLeft)
    return Builder.CreateShl(Vec, ShiftVec);

  if (LogicalShift)
    return Builder.CreateLShr(Vec, ShiftVec);

  return Builder.CreateAShr(Vec, ShiftVec);
}

static Value *simplifyX86pack(IntrinsicInst &II,
                              InstCombiner::BuilderTy &Builder, bool IsSigned) {
  Value *Arg0 = II.getArgOperand(0);
  Value *Arg1 = II.getArgOperand(1);
  Type *ResTy = II.getType();

  // Fast all undef handling.
  if (isa<UndefValue>(Arg0) && isa<UndefValue>(Arg1))
    return UndefValue::get(ResTy);

  auto *ArgTy = cast<VectorType>(Arg0->getType());
  unsigned NumLanes = ResTy->getPrimitiveSizeInBits() / 128;
  unsigned NumSrcElts = ArgTy->getNumElements();
  assert(cast<VectorType>(ResTy)->getNumElements() == (2 * NumSrcElts) &&
         "Unexpected packing types");

  unsigned NumSrcEltsPerLane = NumSrcElts / NumLanes;
  unsigned DstScalarSizeInBits = ResTy->getScalarSizeInBits();
  unsigned SrcScalarSizeInBits = ArgTy->getScalarSizeInBits();
  assert(SrcScalarSizeInBits == (2 * DstScalarSizeInBits) &&
         "Unexpected packing types");

  // Constant folding.
  if (!isa<Constant>(Arg0) || !isa<Constant>(Arg1))
    return nullptr;

  // Clamp Values - signed/unsigned both use signed clamp values, but they
  // differ on the min/max values.
  APInt MinValue, MaxValue;
  if (IsSigned) {
    // PACKSS: Truncate signed value with signed saturation.
    // Source values less than dst minint are saturated to minint.
    // Source values greater than dst maxint are saturated to maxint.
    MinValue =
        APInt::getSignedMinValue(DstScalarSizeInBits).sext(SrcScalarSizeInBits);
    MaxValue =
        APInt::getSignedMaxValue(DstScalarSizeInBits).sext(SrcScalarSizeInBits);
  } else {
    // PACKUS: Truncate signed value with unsigned saturation.
    // Source values less than zero are saturated to zero.
    // Source values greater than dst maxuint are saturated to maxuint.
    MinValue = APInt::getNullValue(SrcScalarSizeInBits);
    MaxValue = APInt::getLowBitsSet(SrcScalarSizeInBits, DstScalarSizeInBits);
  }

  auto *MinC = Constant::getIntegerValue(ArgTy, MinValue);
  auto *MaxC = Constant::getIntegerValue(ArgTy, MaxValue);
  Arg0 = Builder.CreateSelect(Builder.CreateICmpSLT(Arg0, MinC), MinC, Arg0);
  Arg1 = Builder.CreateSelect(Builder.CreateICmpSLT(Arg1, MinC), MinC, Arg1);
  Arg0 = Builder.CreateSelect(Builder.CreateICmpSGT(Arg0, MaxC), MaxC, Arg0);
  Arg1 = Builder.CreateSelect(Builder.CreateICmpSGT(Arg1, MaxC), MaxC, Arg1);

  // Shuffle clamped args together at the lane level.
  SmallVector<int, 32> PackMask;
  for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
    for (unsigned Elt = 0; Elt != NumSrcEltsPerLane; ++Elt)
      PackMask.push_back(Elt + (Lane * NumSrcEltsPerLane));
    for (unsigned Elt = 0; Elt != NumSrcEltsPerLane; ++Elt)
      PackMask.push_back(Elt + (Lane * NumSrcEltsPerLane) + NumSrcElts);
  }
  auto *Shuffle = Builder.CreateShuffleVector(Arg0, Arg1, PackMask);

  // Truncate to dst size.
  return Builder.CreateTrunc(Shuffle, ResTy);
}

static Value *simplifyX86movmsk(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  Value *Arg = II.getArgOperand(0);
  Type *ResTy = II.getType();

  // movmsk(undef) -> zero as we must ensure the upper bits are zero.
  if (isa<UndefValue>(Arg))
    return Constant::getNullValue(ResTy);

  auto *ArgTy = dyn_cast<VectorType>(Arg->getType());
  // We can't easily peek through x86_mmx types.
  if (!ArgTy)
    return nullptr;

  // Expand MOVMSK to compare/bitcast/zext:
  // e.g. PMOVMSKB(v16i8 x):
  // %cmp = icmp slt <16 x i8> %x, zeroinitializer
  // %int = bitcast <16 x i1> %cmp to i16
  // %res = zext i16 %int to i32
  unsigned NumElts = ArgTy->getNumElements();
  Type *IntegerVecTy = VectorType::getInteger(ArgTy);
  Type *IntegerTy = Builder.getIntNTy(NumElts);

  Value *Res = Builder.CreateBitCast(Arg, IntegerVecTy);
  Res = Builder.CreateICmpSLT(Res, Constant::getNullValue(IntegerVecTy));
  Res = Builder.CreateBitCast(Res, IntegerTy);
  Res = Builder.CreateZExtOrTrunc(Res, ResTy);
  return Res;
}

static Value *simplifyX86addcarry(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  Value *CarryIn = II.getArgOperand(0);
  Value *Op1 = II.getArgOperand(1);
  Value *Op2 = II.getArgOperand(2);
  Type *RetTy = II.getType();
  Type *OpTy = Op1->getType();
  assert(RetTy->getStructElementType(0)->isIntegerTy(8) &&
         RetTy->getStructElementType(1) == OpTy && OpTy == Op2->getType() &&
         "Unexpected types for x86 addcarry");

  // If carry-in is zero, this is just an unsigned add with overflow.
  if (match(CarryIn, m_ZeroInt())) {
    Value *UAdd = Builder.CreateIntrinsic(Intrinsic::uadd_with_overflow, OpTy,
                                          { Op1, Op2 });
    // The types have to be adjusted to match the x86 call types.
    Value *UAddResult = Builder.CreateExtractValue(UAdd, 0);
    Value *UAddOV = Builder.CreateZExt(Builder.CreateExtractValue(UAdd, 1),
                                       Builder.getInt8Ty());
    Value *Res = UndefValue::get(RetTy);
    Res = Builder.CreateInsertValue(Res, UAddOV, 0);
    return Builder.CreateInsertValue(Res, UAddResult, 1);
  }

  return nullptr;
}

static Value *simplifyX86insertps(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  auto *CInt = dyn_cast<ConstantInt>(II.getArgOperand(2));
  if (!CInt)
    return nullptr;

  VectorType *VecTy = cast<VectorType>(II.getType());
  assert(VecTy->getNumElements() == 4 && "insertps with wrong vector type");

  // The immediate permute control byte looks like this:
  //    [3:0] - zero mask for each 32-bit lane
  //    [5:4] - select one 32-bit destination lane
  //    [7:6] - select one 32-bit source lane

  uint8_t Imm = CInt->getZExtValue();
  uint8_t ZMask = Imm & 0xf;
  uint8_t DestLane = (Imm >> 4) & 0x3;
  uint8_t SourceLane = (Imm >> 6) & 0x3;

  ConstantAggregateZero *ZeroVector = ConstantAggregateZero::get(VecTy);

  // If all zero mask bits are set, this was just a weird way to
  // generate a zero vector.
  if (ZMask == 0xf)
    return ZeroVector;

  // Initialize by passing all of the first source bits through.
  int ShuffleMask[4] = {0, 1, 2, 3};

  // We may replace the second operand with the zero vector.
  Value *V1 = II.getArgOperand(1);

  if (ZMask) {
    // If the zero mask is being used with a single input or the zero mask
    // overrides the destination lane, this is a shuffle with the zero vector.
    if ((II.getArgOperand(0) == II.getArgOperand(1)) ||
        (ZMask & (1 << DestLane))) {
      V1 = ZeroVector;
      // We may still move 32-bits of the first source vector from one lane
      // to another.
      ShuffleMask[DestLane] = SourceLane;
      // The zero mask may override the previous insert operation.
      for (unsigned i = 0; i < 4; ++i)
        if ((ZMask >> i) & 0x1)
          ShuffleMask[i] = i + 4;
    } else {
      // TODO: Model this case as 2 shuffles or a 'logical and' plus shuffle?
      return nullptr;
    }
  } else {
    // Replace the selected destination lane with the selected source lane.
    ShuffleMask[DestLane] = SourceLane + 4;
  }

  return Builder.CreateShuffleVector(II.getArgOperand(0), V1, ShuffleMask);
}

/// Attempt to simplify SSE4A EXTRQ/EXTRQI instructions using constant folding
/// or conversion to a shuffle vector.
static Value *simplifyX86extrq(IntrinsicInst &II, Value *Op0,
                               ConstantInt *CILength, ConstantInt *CIIndex,
                               InstCombiner::BuilderTy &Builder) {
  auto LowConstantHighUndef = [&](uint64_t Val) {
    Type *IntTy64 = Type::getInt64Ty(II.getContext());
    Constant *Args[] = {ConstantInt::get(IntTy64, Val),
                        UndefValue::get(IntTy64)};
    return ConstantVector::get(Args);
  };

  // See if we're dealing with constant values.
  Constant *C0 = dyn_cast<Constant>(Op0);
  ConstantInt *CI0 =
      C0 ? dyn_cast_or_null<ConstantInt>(C0->getAggregateElement((unsigned)0))
         : nullptr;

  // Attempt to constant fold.
  if (CILength && CIIndex) {
    // From AMD documentation: "The bit index and field length are each six
    // bits in length other bits of the field are ignored."
    APInt APIndex = CIIndex->getValue().zextOrTrunc(6);
    APInt APLength = CILength->getValue().zextOrTrunc(6);

    unsigned Index = APIndex.getZExtValue();

    // From AMD documentation: "a value of zero in the field length is
    // defined as length of 64".
    unsigned Length = APLength == 0 ? 64 : APLength.getZExtValue();

    // From AMD documentation: "If the sum of the bit index + length field
    // is greater than 64, the results are undefined".
    unsigned End = Index + Length;

    // Note that both field index and field length are 8-bit quantities.
    // Since variables 'Index' and 'Length' are unsigned values
    // obtained from zero-extending field index and field length
    // respectively, their sum should never wrap around.
    if (End > 64)
      return UndefValue::get(II.getType());

    // If we are inserting whole bytes, we can convert this to a shuffle.
    // Lowering can recognize EXTRQI shuffle masks.
    if ((Length % 8) == 0 && (Index % 8) == 0) {
      // Convert bit indices to byte indices.
      Length /= 8;
      Index /= 8;

      Type *IntTy8 = Type::getInt8Ty(II.getContext());
      VectorType *ShufTy = VectorType::get(IntTy8, 16);

      SmallVector<int, 16> ShuffleMask;
      for (int i = 0; i != (int)Length; ++i)
        ShuffleMask.push_back(i + Index);
      for (int i = Length; i != 8; ++i)
        ShuffleMask.push_back(i + 16);
      for (int i = 8; i != 16; ++i)
        ShuffleMask.push_back(-1);

      Value *SV = Builder.CreateShuffleVector(
          Builder.CreateBitCast(Op0, ShufTy),
          ConstantAggregateZero::get(ShufTy), ShuffleMask);
      return Builder.CreateBitCast(SV, II.getType());
    }

    // Constant Fold - shift Index'th bit to lowest position and mask off
    // Length bits.
    if (CI0) {
      APInt Elt = CI0->getValue();
      Elt.lshrInPlace(Index);
      Elt = Elt.zextOrTrunc(Length);
      return LowConstantHighUndef(Elt.getZExtValue());
    }

    // If we were an EXTRQ call, we'll save registers if we convert to EXTRQI.
    if (II.getIntrinsicID() == Intrinsic::x86_sse4a_extrq) {
      Value *Args[] = {Op0, CILength, CIIndex};
      Module *M = II.getModule();
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::x86_sse4a_extrqi);
      return Builder.CreateCall(F, Args);
    }
  }

  // Constant Fold - extraction from zero is always {zero, undef}.
  if (CI0 && CI0->isZero())
    return LowConstantHighUndef(0);

  return nullptr;
}

/// Attempt to simplify SSE4A INSERTQ/INSERTQI instructions using constant
/// folding or conversion to a shuffle vector.
static Value *simplifyX86insertq(IntrinsicInst &II, Value *Op0, Value *Op1,
                                 APInt APLength, APInt APIndex,
                                 InstCombiner::BuilderTy &Builder) {
  // From AMD documentation: "The bit index and field length are each six bits
  // in length other bits of the field are ignored."
  APIndex = APIndex.zextOrTrunc(6);
  APLength = APLength.zextOrTrunc(6);

  // Attempt to constant fold.
  unsigned Index = APIndex.getZExtValue();

  // From AMD documentation: "a value of zero in the field length is
  // defined as length of 64".
  unsigned Length = APLength == 0 ? 64 : APLength.getZExtValue();

  // From AMD documentation: "If the sum of the bit index + length field
  // is greater than 64, the results are undefined".
  unsigned End = Index + Length;

  // Note that both field index and field length are 8-bit quantities.
  // Since variables 'Index' and 'Length' are unsigned values
  // obtained from zero-extending field index and field length
  // respectively, their sum should never wrap around.
  if (End > 64)
    return UndefValue::get(II.getType());

  // If we are inserting whole bytes, we can convert this to a shuffle.
  // Lowering can recognize INSERTQI shuffle masks.
  if ((Length % 8) == 0 && (Index % 8) == 0) {
    // Convert bit indices to byte indices.
    Length /= 8;
    Index /= 8;

    Type *IntTy8 = Type::getInt8Ty(II.getContext());
    VectorType *ShufTy = VectorType::get(IntTy8, 16);

    SmallVector<int, 16> ShuffleMask;
    for (int i = 0; i != (int)Index; ++i)
      ShuffleMask.push_back(i);
    for (int i = 0; i != (int)Length; ++i)
      ShuffleMask.push_back(i + 16);
    for (int i = Index + Length; i != 8; ++i)
      ShuffleMask.push_back(i);
    for (int i = 8; i != 16; ++i)
      ShuffleMask.push_back(-1);

    Value *SV = Builder.CreateShuffleVector(Builder.CreateBitCast(Op0, ShufTy),
                                            Builder.CreateBitCast(Op1, ShufTy),
                                            ShuffleMask);
    return Builder.CreateBitCast(SV, II.getType());
  }

  // See if we're dealing with constant values.
  Constant *C0 = dyn_cast<Constant>(Op0);
  Constant *C1 = dyn_cast<Constant>(Op1);
  ConstantInt *CI00 =
      C0 ? dyn_cast_or_null<ConstantInt>(C0->getAggregateElement((unsigned)0))
         : nullptr;
  ConstantInt *CI10 =
      C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)0))
         : nullptr;

  // Constant Fold - insert bottom Length bits starting at the Index'th bit.
  if (CI00 && CI10) {
    APInt V00 = CI00->getValue();
    APInt V10 = CI10->getValue();
    APInt Mask = APInt::getLowBitsSet(64, Length).shl(Index);
    V00 = V00 & ~Mask;
    V10 = V10.zextOrTrunc(Length).zextOrTrunc(64).shl(Index);
    APInt Val = V00 | V10;
    Type *IntTy64 = Type::getInt64Ty(II.getContext());
    Constant *Args[] = {ConstantInt::get(IntTy64, Val.getZExtValue()),
                        UndefValue::get(IntTy64)};
    return ConstantVector::get(Args);
  }

  // If we were an INSERTQ call, we'll save demanded elements if we convert to
  // INSERTQI.
  if (II.getIntrinsicID() == Intrinsic::x86_sse4a_insertq) {
    Type *IntTy8 = Type::getInt8Ty(II.getContext());
    Constant *CILength = ConstantInt::get(IntTy8, Length, false);
    Constant *CIIndex = ConstantInt::get(IntTy8, Index, false);

    Value *Args[] = {Op0, Op1, CILength, CIIndex};
    Module *M = II.getModule();
    Function *F = Intrinsic::getDeclaration(M, Intrinsic::x86_sse4a_insertqi);
    return Builder.CreateCall(F, Args);
  }

  return nullptr;
}

/// Attempt to convert pshufb* to shufflevector if the mask is constant.
static Value *simplifyX86pshufb(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  Constant *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();
  assert((NumElts == 16 || NumElts == 32 || NumElts == 64) &&
         "Unexpected number of elements in shuffle mask!");

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[64];

  // Each byte in the shuffle control mask forms an index to permute the
  // corresponding byte in the destination operand.
  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    int8_t Index = cast<ConstantInt>(COp)->getValue().getZExtValue();

    // If the most significant bit (bit[7]) of each byte of the shuffle
    // control mask is set, then zero is written in the result byte.
    // The zero vector is in the right-hand side of the resulting
    // shufflevector.

    // The value of each index for the high 128-bit lane is the least
    // significant 4 bits of the respective shuffle control byte.
    Index = ((Index < 0) ? NumElts : Index & 0x0F) + (I & 0xF0);
    Indexes[I] = Index;
  }

  auto V1 = II.getArgOperand(0);
  auto V2 = Constant::getNullValue(VecTy);
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes, NumElts));
}

/// Attempt to convert vpermilvar* to shufflevector if the mask is constant.
static Value *simplifyX86vpermilvar(const IntrinsicInst &II,
                                    InstCombiner::BuilderTy &Builder) {
  Constant *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();
  bool IsPD = VecTy->getScalarType()->isDoubleTy();
  unsigned NumLaneElts = IsPD ? 2 : 4;
  assert(NumElts == 16 || NumElts == 8 || NumElts == 4 || NumElts == 2);

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[16];

  // The intrinsics only read one or two bits, clear the rest.
  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    APInt Index = cast<ConstantInt>(COp)->getValue();
    Index = Index.zextOrTrunc(32).getLoBits(2);

    // The PD variants uses bit 1 to select per-lane element index, so
    // shift down to convert to generic shuffle mask index.
    if (IsPD)
      Index.lshrInPlace(1);

    // The _256 variants are a bit trickier since the mask bits always index
    // into the corresponding 128 half. In order to convert to a generic
    // shuffle, we have to make that explicit.
    Index += APInt(32, (I / NumLaneElts) * NumLaneElts);

    Indexes[I] = Index.getZExtValue();
  }

  auto V1 = II.getArgOperand(0);
  auto V2 = UndefValue::get(V1->getType());
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes, NumElts));
}

/// Attempt to convert vpermd/vpermps to shufflevector if the mask is constant.
static Value *simplifyX86vpermv(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  auto *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned Size = VecTy->getNumElements();
  assert((Size == 4 || Size == 8 || Size == 16 || Size == 32 || Size == 64) &&
         "Unexpected shuffle mask size");

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[64];

  for (unsigned I = 0; I < Size; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    uint32_t Index = cast<ConstantInt>(COp)->getZExtValue();
    Index &= Size - 1;
    Indexes[I] = Index;
  }

  auto V1 = II.getArgOperand(0);
  auto V2 = UndefValue::get(VecTy);
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes, Size));
}

// TODO, Obvious Missing Transforms:
// * Narrow width by halfs excluding zero/undef lanes
Value *InstCombiner::simplifyMaskedLoad(IntrinsicInst &II) {
  Value *LoadPtr = II.getArgOperand(0);
  const Align Alignment =
      cast<ConstantInt>(II.getArgOperand(1))->getAlignValue();

  // If the mask is all ones or undefs, this is a plain vector load of the 1st
  // argument.
  if (maskIsAllOneOrUndef(II.getArgOperand(2)))
    return Builder.CreateAlignedLoad(II.getType(), LoadPtr, Alignment,
                                     "unmaskedload");

  // If we can unconditionally load from this address, replace with a
  // load/select idiom. TODO: use DT for context sensitive query
  if (isDereferenceableAndAlignedPointer(LoadPtr, II.getType(), Alignment,
                                         II.getModule()->getDataLayout(), &II,
                                         nullptr)) {
    Value *LI = Builder.CreateAlignedLoad(II.getType(), LoadPtr, Alignment,
                                         "unmaskedload");
    return Builder.CreateSelect(II.getArgOperand(2), LI, II.getArgOperand(3));
  }

  return nullptr;
}

// TODO, Obvious Missing Transforms:
// * Single constant active lane -> store
// * Narrow width by halfs excluding zero/undef lanes
Instruction *InstCombiner::simplifyMaskedStore(IntrinsicInst &II) {
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(3));
  if (!ConstMask)
    return nullptr;

  // If the mask is all zeros, this instruction does nothing.
  if (ConstMask->isNullValue())
    return eraseInstFromFunction(II);

  // If the mask is all ones, this is a plain vector store of the 1st argument.
  if (ConstMask->isAllOnesValue()) {
    Value *StorePtr = II.getArgOperand(1);
    MaybeAlign Alignment(
        cast<ConstantInt>(II.getArgOperand(2))->getZExtValue());
    return new StoreInst(II.getArgOperand(0), StorePtr, false, Alignment);
  }

  // Use masked off lanes to simplify operands via SimplifyDemandedVectorElts
  APInt DemandedElts = possiblyDemandedEltsInMask(ConstMask);
  APInt UndefElts(DemandedElts.getBitWidth(), 0);
  if (Value *V = SimplifyDemandedVectorElts(II.getOperand(0),
                                            DemandedElts, UndefElts))
    return replaceOperand(II, 0, V);

  return nullptr;
}

// TODO, Obvious Missing Transforms:
// * Single constant active lane load -> load
// * Dereferenceable address & few lanes -> scalarize speculative load/selects
// * Adjacent vector addresses -> masked.load
// * Narrow width by halfs excluding zero/undef lanes
// * Vector splat address w/known mask -> scalar load
// * Vector incrementing address -> vector masked load
Instruction *InstCombiner::simplifyMaskedGather(IntrinsicInst &II) {
  return nullptr;
}

// TODO, Obvious Missing Transforms:
// * Single constant active lane -> store
// * Adjacent vector addresses -> masked.store
// * Narrow store width by halfs excluding zero/undef lanes
// * Vector splat address w/known mask -> scalar store
// * Vector incrementing address -> vector masked store
Instruction *InstCombiner::simplifyMaskedScatter(IntrinsicInst &II) {
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(3));
  if (!ConstMask)
    return nullptr;

  // If the mask is all zeros, a scatter does nothing.
  if (ConstMask->isNullValue())
    return eraseInstFromFunction(II);

  // Use masked off lanes to simplify operands via SimplifyDemandedVectorElts
  APInt DemandedElts = possiblyDemandedEltsInMask(ConstMask);
  APInt UndefElts(DemandedElts.getBitWidth(), 0);
  if (Value *V = SimplifyDemandedVectorElts(II.getOperand(0),
                                            DemandedElts, UndefElts))
    return replaceOperand(II, 0, V);
  if (Value *V = SimplifyDemandedVectorElts(II.getOperand(1),
                                            DemandedElts, UndefElts))
    return replaceOperand(II, 1, V);

  return nullptr;
}

/// This function transforms launder.invariant.group and strip.invariant.group
/// like:
/// launder(launder(%x)) -> launder(%x)       (the result is not the argument)
/// launder(strip(%x)) -> launder(%x)
/// strip(strip(%x)) -> strip(%x)             (the result is not the argument)
/// strip(launder(%x)) -> strip(%x)
/// This is legal because it preserves the most recent information about
/// the presence or absence of invariant.group.
static Instruction *simplifyInvariantGroupIntrinsic(IntrinsicInst &II,
                                                    InstCombiner &IC) {
  auto *Arg = II.getArgOperand(0);
  auto *StrippedArg = Arg->stripPointerCasts();
  auto *StrippedInvariantGroupsArg = Arg->stripPointerCastsAndInvariantGroups();
  if (StrippedArg == StrippedInvariantGroupsArg)
    return nullptr; // No launders/strips to remove.

  Value *Result = nullptr;

  if (II.getIntrinsicID() == Intrinsic::launder_invariant_group)
    Result = IC.Builder.CreateLaunderInvariantGroup(StrippedInvariantGroupsArg);
  else if (II.getIntrinsicID() == Intrinsic::strip_invariant_group)
    Result = IC.Builder.CreateStripInvariantGroup(StrippedInvariantGroupsArg);
  else
    llvm_unreachable(
        "simplifyInvariantGroupIntrinsic only handles launder and strip");
  if (Result->getType()->getPointerAddressSpace() !=
      II.getType()->getPointerAddressSpace())
    Result = IC.Builder.CreateAddrSpaceCast(Result, II.getType());
  if (Result->getType() != II.getType())
    Result = IC.Builder.CreateBitCast(Result, II.getType());

  return cast<Instruction>(Result);
}

static Instruction *foldCttzCtlz(IntrinsicInst &II, InstCombiner &IC) {
  assert((II.getIntrinsicID() == Intrinsic::cttz ||
          II.getIntrinsicID() == Intrinsic::ctlz) &&
         "Expected cttz or ctlz intrinsic");
  bool IsTZ = II.getIntrinsicID() == Intrinsic::cttz;
  Value *Op0 = II.getArgOperand(0);
  Value *X;
  // ctlz(bitreverse(x)) -> cttz(x)
  // cttz(bitreverse(x)) -> ctlz(x)
  if (match(Op0, m_BitReverse(m_Value(X)))) {
    Intrinsic::ID ID = IsTZ ? Intrinsic::ctlz : Intrinsic::cttz;
    Function *F = Intrinsic::getDeclaration(II.getModule(), ID, II.getType());
    return CallInst::Create(F, {X, II.getArgOperand(1)});
  }

  if (IsTZ) {
    // cttz(-x) -> cttz(x)
    if (match(Op0, m_Neg(m_Value(X))))
      return IC.replaceOperand(II, 0, X);

    // cttz(abs(x)) -> cttz(x)
    // cttz(nabs(x)) -> cttz(x)
    Value *Y;
    SelectPatternFlavor SPF = matchSelectPattern(Op0, X, Y).Flavor;
    if (SPF == SPF_ABS || SPF == SPF_NABS)
      return IC.replaceOperand(II, 0, X);
  }

  KnownBits Known = IC.computeKnownBits(Op0, 0, &II);

  // Create a mask for bits above (ctlz) or below (cttz) the first known one.
  unsigned PossibleZeros = IsTZ ? Known.countMaxTrailingZeros()
                                : Known.countMaxLeadingZeros();
  unsigned DefiniteZeros = IsTZ ? Known.countMinTrailingZeros()
                                : Known.countMinLeadingZeros();

  // If all bits above (ctlz) or below (cttz) the first known one are known
  // zero, this value is constant.
  // FIXME: This should be in InstSimplify because we're replacing an
  // instruction with a constant.
  if (PossibleZeros == DefiniteZeros) {
    auto *C = ConstantInt::get(Op0->getType(), DefiniteZeros);
    return IC.replaceInstUsesWith(II, C);
  }

  // If the input to cttz/ctlz is known to be non-zero,
  // then change the 'ZeroIsUndef' parameter to 'true'
  // because we know the zero behavior can't affect the result.
  if (!Known.One.isNullValue() ||
      isKnownNonZero(Op0, IC.getDataLayout(), 0, &IC.getAssumptionCache(), &II,
                     &IC.getDominatorTree())) {
    if (!match(II.getArgOperand(1), m_One()))
      return IC.replaceOperand(II, 1, IC.Builder.getTrue());
  }

  // Add range metadata since known bits can't completely reflect what we know.
  // TODO: Handle splat vectors.
  auto *IT = dyn_cast<IntegerType>(Op0->getType());
  if (IT && IT->getBitWidth() != 1 && !II.getMetadata(LLVMContext::MD_range)) {
    Metadata *LowAndHigh[] = {
        ConstantAsMetadata::get(ConstantInt::get(IT, DefiniteZeros)),
        ConstantAsMetadata::get(ConstantInt::get(IT, PossibleZeros + 1))};
    II.setMetadata(LLVMContext::MD_range,
                   MDNode::get(II.getContext(), LowAndHigh));
    return &II;
  }

  return nullptr;
}

static Instruction *foldCtpop(IntrinsicInst &II, InstCombiner &IC) {
  assert(II.getIntrinsicID() == Intrinsic::ctpop &&
         "Expected ctpop intrinsic");
  Type *Ty = II.getType();
  unsigned BitWidth = Ty->getScalarSizeInBits();
  Value *Op0 = II.getArgOperand(0);
  Value *X;

  // ctpop(bitreverse(x)) -> ctpop(x)
  // ctpop(bswap(x)) -> ctpop(x)
  if (match(Op0, m_BitReverse(m_Value(X))) || match(Op0, m_BSwap(m_Value(X))))
    return IC.replaceOperand(II, 0, X);

  // ctpop(x | -x) -> bitwidth - cttz(x, false)
  if (Op0->hasOneUse() &&
      match(Op0, m_c_Or(m_Value(X), m_Neg(m_Deferred(X))))) {
    Function *F =
        Intrinsic::getDeclaration(II.getModule(), Intrinsic::cttz, Ty);
    auto *Cttz = IC.Builder.CreateCall(F, {X, IC.Builder.getFalse()});
    auto *Bw = ConstantInt::get(Ty, APInt(BitWidth, BitWidth));
    return IC.replaceInstUsesWith(II, IC.Builder.CreateSub(Bw, Cttz));
  }

  // ctpop(~x & (x - 1)) -> cttz(x, false)
  if (match(Op0,
            m_c_And(m_Not(m_Value(X)), m_Add(m_Deferred(X), m_AllOnes())))) {
    Function *F =
        Intrinsic::getDeclaration(II.getModule(), Intrinsic::cttz, Ty);
    return CallInst::Create(F, {X, IC.Builder.getFalse()});
  }

  // FIXME: Try to simplify vectors of integers.
  auto *IT = dyn_cast<IntegerType>(Ty);
  if (!IT)
    return nullptr;

  KnownBits Known(BitWidth);
  IC.computeKnownBits(Op0, Known, 0, &II);

  unsigned MinCount = Known.countMinPopulation();
  unsigned MaxCount = Known.countMaxPopulation();

  // Add range metadata since known bits can't completely reflect what we know.
  if (IT->getBitWidth() != 1 && !II.getMetadata(LLVMContext::MD_range)) {
    Metadata *LowAndHigh[] = {
        ConstantAsMetadata::get(ConstantInt::get(IT, MinCount)),
        ConstantAsMetadata::get(ConstantInt::get(IT, MaxCount + 1))};
    II.setMetadata(LLVMContext::MD_range,
                   MDNode::get(II.getContext(), LowAndHigh));
    return &II;
  }

  return nullptr;
}

// TODO: If the x86 backend knew how to convert a bool vector mask back to an
// XMM register mask efficiently, we could transform all x86 masked intrinsics
// to LLVM masked intrinsics and remove the x86 masked intrinsic defs.
static Instruction *simplifyX86MaskedLoad(IntrinsicInst &II, InstCombiner &IC) {
  Value *Ptr = II.getOperand(0);
  Value *Mask = II.getOperand(1);
  Constant *ZeroVec = Constant::getNullValue(II.getType());

  // Special case a zero mask since that's not a ConstantDataVector.
  // This masked load instruction creates a zero vector.
  if (isa<ConstantAggregateZero>(Mask))
    return IC.replaceInstUsesWith(II, ZeroVec);

  auto *ConstMask = dyn_cast<ConstantDataVector>(Mask);
  if (!ConstMask)
    return nullptr;

  // The mask is constant. Convert this x86 intrinsic to the LLVM instrinsic
  // to allow target-independent optimizations.

  // First, cast the x86 intrinsic scalar pointer to a vector pointer to match
  // the LLVM intrinsic definition for the pointer argument.
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  PointerType *VecPtrTy = PointerType::get(II.getType(), AddrSpace);
  Value *PtrCast = IC.Builder.CreateBitCast(Ptr, VecPtrTy, "castvec");

  // Second, convert the x86 XMM integer vector mask to a vector of bools based
  // on each element's most significant bit (the sign bit).
  Constant *BoolMask = getNegativeIsTrueBoolVec(ConstMask);

  // The pass-through vector for an x86 masked load is a zero vector.
  CallInst *NewMaskedLoad =
      IC.Builder.CreateMaskedLoad(PtrCast, Align(1), BoolMask, ZeroVec);
  return IC.replaceInstUsesWith(II, NewMaskedLoad);
}

// TODO: If the x86 backend knew how to convert a bool vector mask back to an
// XMM register mask efficiently, we could transform all x86 masked intrinsics
// to LLVM masked intrinsics and remove the x86 masked intrinsic defs.
static bool simplifyX86MaskedStore(IntrinsicInst &II, InstCombiner &IC) {
  Value *Ptr = II.getOperand(0);
  Value *Mask = II.getOperand(1);
  Value *Vec = II.getOperand(2);

  // Special case a zero mask since that's not a ConstantDataVector:
  // this masked store instruction does nothing.
  if (isa<ConstantAggregateZero>(Mask)) {
    IC.eraseInstFromFunction(II);
    return true;
  }

  // The SSE2 version is too weird (eg, unaligned but non-temporal) to do
  // anything else at this level.
  if (II.getIntrinsicID() == Intrinsic::x86_sse2_maskmov_dqu)
    return false;

  auto *ConstMask = dyn_cast<ConstantDataVector>(Mask);
  if (!ConstMask)
    return false;

  // The mask is constant. Convert this x86 intrinsic to the LLVM instrinsic
  // to allow target-independent optimizations.

  // First, cast the x86 intrinsic scalar pointer to a vector pointer to match
  // the LLVM intrinsic definition for the pointer argument.
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  PointerType *VecPtrTy = PointerType::get(Vec->getType(), AddrSpace);
  Value *PtrCast = IC.Builder.CreateBitCast(Ptr, VecPtrTy, "castvec");

  // Second, convert the x86 XMM integer vector mask to a vector of bools based
  // on each element's most significant bit (the sign bit).
  Constant *BoolMask = getNegativeIsTrueBoolVec(ConstMask);

  IC.Builder.CreateMaskedStore(Vec, PtrCast, Align(1), BoolMask);

  // 'Replace uses' doesn't work for stores. Erase the original masked store.
  IC.eraseInstFromFunction(II);
  return true;
}

// Constant fold llvm.amdgcn.fmed3 intrinsics for standard inputs.
//
// A single NaN input is folded to minnum, so we rely on that folding for
// handling NaNs.
static APFloat fmed3AMDGCN(const APFloat &Src0, const APFloat &Src1,
                           const APFloat &Src2) {
  APFloat Max3 = maxnum(maxnum(Src0, Src1), Src2);

  APFloat::cmpResult Cmp0 = Max3.compare(Src0);
  assert(Cmp0 != APFloat::cmpUnordered && "nans handled separately");
  if (Cmp0 == APFloat::cmpEqual)
    return maxnum(Src1, Src2);

  APFloat::cmpResult Cmp1 = Max3.compare(Src1);
  assert(Cmp1 != APFloat::cmpUnordered && "nans handled separately");
  if (Cmp1 == APFloat::cmpEqual)
    return maxnum(Src0, Src2);

  return maxnum(Src0, Src1);
}

/// Convert a table lookup to shufflevector if the mask is constant.
/// This could benefit tbl1 if the mask is { 7,6,5,4,3,2,1,0 }, in
/// which case we could lower the shufflevector with rev64 instructions
/// as it's actually a byte reverse.
static Value *simplifyNeonTbl1(const IntrinsicInst &II,
                               InstCombiner::BuilderTy &Builder) {
  // Bail out if the mask is not a constant.
  auto *C = dyn_cast<Constant>(II.getArgOperand(1));
  if (!C)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();

  // Only perform this transformation for <8 x i8> vector types.
  if (!VecTy->getElementType()->isIntegerTy(8) || NumElts != 8)
    return nullptr;

  int Indexes[8];

  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = C->getAggregateElement(I);

    if (!COp || !isa<ConstantInt>(COp))
      return nullptr;

    Indexes[I] = cast<ConstantInt>(COp)->getLimitedValue();

    // Make sure the mask indices are in range.
    if ((unsigned)Indexes[I] >= NumElts)
      return nullptr;
  }

  auto *V1 = II.getArgOperand(0);
  auto *V2 = Constant::getNullValue(V1->getType());
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes));
}

/// Convert a vector load intrinsic into a simple llvm load instruction.
/// This is beneficial when the underlying object being addressed comes
/// from a constant, since we get constant-folding for free.
static Value *simplifyNeonVld1(const IntrinsicInst &II,
                               unsigned MemAlign,
                               InstCombiner::BuilderTy &Builder) {
  auto *IntrAlign = dyn_cast<ConstantInt>(II.getArgOperand(1));

  if (!IntrAlign)
    return nullptr;

  unsigned Alignment = IntrAlign->getLimitedValue() < MemAlign ?
                       MemAlign : IntrAlign->getLimitedValue();

  if (!isPowerOf2_32(Alignment))
    return nullptr;

  auto *BCastInst = Builder.CreateBitCast(II.getArgOperand(0),
                                          PointerType::get(II.getType(), 0));
  return Builder.CreateAlignedLoad(II.getType(), BCastInst, Align(Alignment));
}

// Returns true iff the 2 intrinsics have the same operands, limiting the
// comparison to the first NumOperands.
static bool haveSameOperands(const IntrinsicInst &I, const IntrinsicInst &E,
                             unsigned NumOperands) {
  assert(I.getNumArgOperands() >= NumOperands && "Not enough operands");
  assert(E.getNumArgOperands() >= NumOperands && "Not enough operands");
  for (unsigned i = 0; i < NumOperands; i++)
    if (I.getArgOperand(i) != E.getArgOperand(i))
      return false;
  return true;
}

// Remove trivially empty start/end intrinsic ranges, i.e. a start
// immediately followed by an end (ignoring debuginfo or other
// start/end intrinsics in between). As this handles only the most trivial
// cases, tracking the nesting level is not needed:
//
//   call @llvm.foo.start(i1 0)
//   call @llvm.foo.start(i1 0) ; This one won't be skipped: it will be removed
//   call @llvm.foo.end(i1 0)
//   call @llvm.foo.end(i1 0) ; &I
static bool removeTriviallyEmptyRange(
    IntrinsicInst &EndI, InstCombiner &IC,
    std::function<bool(const IntrinsicInst &)> IsStart) {
  // We start from the end intrinsic and scan backwards, so that InstCombine
  // has already processed (and potentially removed) all the instructions
  // before the end intrinsic.
  BasicBlock::reverse_iterator BI(EndI), BE(EndI.getParent()->rend());
  for (; BI != BE; ++BI) {
    if (auto *I = dyn_cast<IntrinsicInst>(&*BI)) {
      if (isa<DbgInfoIntrinsic>(I) ||
          I->getIntrinsicID() == EndI.getIntrinsicID())
        continue;
      if (IsStart(*I)) {
        if (haveSameOperands(EndI, *I, EndI.getNumArgOperands())) {
          IC.eraseInstFromFunction(*I);
          IC.eraseInstFromFunction(EndI);
          return true;
        }
        // Skip start intrinsics that don't pair with this end intrinsic.
        continue;
      }
    }
    break;
  }

  return false;
}

// Convert NVVM intrinsics to target-generic LLVM code where possible.
static Instruction *SimplifyNVVMIntrinsic(IntrinsicInst *II, InstCombiner &IC) {
  // Each NVVM intrinsic we can simplify can be replaced with one of:
  //
  //  * an LLVM intrinsic,
  //  * an LLVM cast operation,
  //  * an LLVM binary operation, or
  //  * ad-hoc LLVM IR for the particular operation.

  // Some transformations are only valid when the module's
  // flush-denormals-to-zero (ftz) setting is true/false, whereas other
  // transformations are valid regardless of the module's ftz setting.
  enum FtzRequirementTy {
    FTZ_Any,       // Any ftz setting is ok.
    FTZ_MustBeOn,  // Transformation is valid only if ftz is on.
    FTZ_MustBeOff, // Transformation is valid only if ftz is off.
  };
  // Classes of NVVM intrinsics that can't be replaced one-to-one with a
  // target-generic intrinsic, cast op, or binary op but that we can nonetheless
  // simplify.
  enum SpecialCase {
    SPC_Reciprocal,
  };

  // SimplifyAction is a poor-man's variant (plus an additional flag) that
  // represents how to replace an NVVM intrinsic with target-generic LLVM IR.
  struct SimplifyAction {
    // Invariant: At most one of these Optionals has a value.
    Optional<Intrinsic::ID> IID;
    Optional<Instruction::CastOps> CastOp;
    Optional<Instruction::BinaryOps> BinaryOp;
    Optional<SpecialCase> Special;

    FtzRequirementTy FtzRequirement = FTZ_Any;

    SimplifyAction() = default;

    SimplifyAction(Intrinsic::ID IID, FtzRequirementTy FtzReq)
        : IID(IID), FtzRequirement(FtzReq) {}

    // Cast operations don't have anything to do with FTZ, so we skip that
    // argument.
    SimplifyAction(Instruction::CastOps CastOp) : CastOp(CastOp) {}

    SimplifyAction(Instruction::BinaryOps BinaryOp, FtzRequirementTy FtzReq)
        : BinaryOp(BinaryOp), FtzRequirement(FtzReq) {}

    SimplifyAction(SpecialCase Special, FtzRequirementTy FtzReq)
        : Special(Special), FtzRequirement(FtzReq) {}
  };

  // Try to generate a SimplifyAction describing how to replace our
  // IntrinsicInstr with target-generic LLVM IR.
  const SimplifyAction Action = [II]() -> SimplifyAction {
    switch (II->getIntrinsicID()) {
    // NVVM intrinsics that map directly to LLVM intrinsics.
    case Intrinsic::nvvm_ceil_d:
      return {Intrinsic::ceil, FTZ_Any};
    case Intrinsic::nvvm_ceil_f:
      return {Intrinsic::ceil, FTZ_MustBeOff};
    case Intrinsic::nvvm_ceil_ftz_f:
      return {Intrinsic::ceil, FTZ_MustBeOn};
    case Intrinsic::nvvm_fabs_d:
      return {Intrinsic::fabs, FTZ_Any};
    case Intrinsic::nvvm_fabs_f:
      return {Intrinsic::fabs, FTZ_MustBeOff};
    case Intrinsic::nvvm_fabs_ftz_f:
      return {Intrinsic::fabs, FTZ_MustBeOn};
    case Intrinsic::nvvm_floor_d:
      return {Intrinsic::floor, FTZ_Any};
    case Intrinsic::nvvm_floor_f:
      return {Intrinsic::floor, FTZ_MustBeOff};
    case Intrinsic::nvvm_floor_ftz_f:
      return {Intrinsic::floor, FTZ_MustBeOn};
    case Intrinsic::nvvm_fma_rn_d:
      return {Intrinsic::fma, FTZ_Any};
    case Intrinsic::nvvm_fma_rn_f:
      return {Intrinsic::fma, FTZ_MustBeOff};
    case Intrinsic::nvvm_fma_rn_ftz_f:
      return {Intrinsic::fma, FTZ_MustBeOn};
    case Intrinsic::nvvm_fmax_d:
      return {Intrinsic::maxnum, FTZ_Any};
    case Intrinsic::nvvm_fmax_f:
      return {Intrinsic::maxnum, FTZ_MustBeOff};
    case Intrinsic::nvvm_fmax_ftz_f:
      return {Intrinsic::maxnum, FTZ_MustBeOn};
    case Intrinsic::nvvm_fmin_d:
      return {Intrinsic::minnum, FTZ_Any};
    case Intrinsic::nvvm_fmin_f:
      return {Intrinsic::minnum, FTZ_MustBeOff};
    case Intrinsic::nvvm_fmin_ftz_f:
      return {Intrinsic::minnum, FTZ_MustBeOn};
    case Intrinsic::nvvm_round_d:
      return {Intrinsic::round, FTZ_Any};
    case Intrinsic::nvvm_round_f:
      return {Intrinsic::round, FTZ_MustBeOff};
    case Intrinsic::nvvm_round_ftz_f:
      return {Intrinsic::round, FTZ_MustBeOn};
    case Intrinsic::nvvm_sqrt_rn_d:
      return {Intrinsic::sqrt, FTZ_Any};
    case Intrinsic::nvvm_sqrt_f:
      // nvvm_sqrt_f is a special case.  For  most intrinsics, foo_ftz_f is the
      // ftz version, and foo_f is the non-ftz version.  But nvvm_sqrt_f adopts
      // the ftz-ness of the surrounding code.  sqrt_rn_f and sqrt_rn_ftz_f are
      // the versions with explicit ftz-ness.
      return {Intrinsic::sqrt, FTZ_Any};
    case Intrinsic::nvvm_sqrt_rn_f:
      return {Intrinsic::sqrt, FTZ_MustBeOff};
    case Intrinsic::nvvm_sqrt_rn_ftz_f:
      return {Intrinsic::sqrt, FTZ_MustBeOn};
    case Intrinsic::nvvm_trunc_d:
      return {Intrinsic::trunc, FTZ_Any};
    case Intrinsic::nvvm_trunc_f:
      return {Intrinsic::trunc, FTZ_MustBeOff};
    case Intrinsic::nvvm_trunc_ftz_f:
      return {Intrinsic::trunc, FTZ_MustBeOn};

    // NVVM intrinsics that map to LLVM cast operations.
    //
    // Note that llvm's target-generic conversion operators correspond to the rz
    // (round to zero) versions of the nvvm conversion intrinsics, even though
    // most everything else here uses the rn (round to nearest even) nvvm ops.
    case Intrinsic::nvvm_d2i_rz:
    case Intrinsic::nvvm_f2i_rz:
    case Intrinsic::nvvm_d2ll_rz:
    case Intrinsic::nvvm_f2ll_rz:
      return {Instruction::FPToSI};
    case Intrinsic::nvvm_d2ui_rz:
    case Intrinsic::nvvm_f2ui_rz:
    case Intrinsic::nvvm_d2ull_rz:
    case Intrinsic::nvvm_f2ull_rz:
      return {Instruction::FPToUI};
    case Intrinsic::nvvm_i2d_rz:
    case Intrinsic::nvvm_i2f_rz:
    case Intrinsic::nvvm_ll2d_rz:
    case Intrinsic::nvvm_ll2f_rz:
      return {Instruction::SIToFP};
    case Intrinsic::nvvm_ui2d_rz:
    case Intrinsic::nvvm_ui2f_rz:
    case Intrinsic::nvvm_ull2d_rz:
    case Intrinsic::nvvm_ull2f_rz:
      return {Instruction::UIToFP};

    // NVVM intrinsics that map to LLVM binary ops.
    case Intrinsic::nvvm_add_rn_d:
      return {Instruction::FAdd, FTZ_Any};
    case Intrinsic::nvvm_add_rn_f:
      return {Instruction::FAdd, FTZ_MustBeOff};
    case Intrinsic::nvvm_add_rn_ftz_f:
      return {Instruction::FAdd, FTZ_MustBeOn};
    case Intrinsic::nvvm_mul_rn_d:
      return {Instruction::FMul, FTZ_Any};
    case Intrinsic::nvvm_mul_rn_f:
      return {Instruction::FMul, FTZ_MustBeOff};
    case Intrinsic::nvvm_mul_rn_ftz_f:
      return {Instruction::FMul, FTZ_MustBeOn};
    case Intrinsic::nvvm_div_rn_d:
      return {Instruction::FDiv, FTZ_Any};
    case Intrinsic::nvvm_div_rn_f:
      return {Instruction::FDiv, FTZ_MustBeOff};
    case Intrinsic::nvvm_div_rn_ftz_f:
      return {Instruction::FDiv, FTZ_MustBeOn};

    // The remainder of cases are NVVM intrinsics that map to LLVM idioms, but
    // need special handling.
    //
    // We seem to be missing intrinsics for rcp.approx.{ftz.}f32, which is just
    // as well.
    case Intrinsic::nvvm_rcp_rn_d:
      return {SPC_Reciprocal, FTZ_Any};
    case Intrinsic::nvvm_rcp_rn_f:
      return {SPC_Reciprocal, FTZ_MustBeOff};
    case Intrinsic::nvvm_rcp_rn_ftz_f:
      return {SPC_Reciprocal, FTZ_MustBeOn};

    // We do not currently simplify intrinsics that give an approximate answer.
    // These include:
    //
    //   - nvvm_cos_approx_{f,ftz_f}
    //   - nvvm_ex2_approx_{d,f,ftz_f}
    //   - nvvm_lg2_approx_{d,f,ftz_f}
    //   - nvvm_sin_approx_{f,ftz_f}
    //   - nvvm_sqrt_approx_{f,ftz_f}
    //   - nvvm_rsqrt_approx_{d,f,ftz_f}
    //   - nvvm_div_approx_{ftz_d,ftz_f,f}
    //   - nvvm_rcp_approx_ftz_d
    //
    // Ideally we'd encode them as e.g. "fast call @llvm.cos", where "fast"
    // means that fastmath is enabled in the intrinsic.  Unfortunately only
    // binary operators (currently) have a fastmath bit in SelectionDAG, so this
    // information gets lost and we can't select on it.
    //
    // TODO: div and rcp are lowered to a binary op, so these we could in theory
    // lower them to "fast fdiv".

    default:
      return {};
    }
  }();

  // If Action.FtzRequirementTy is not satisfied by the module's ftz state, we
  // can bail out now.  (Notice that in the case that IID is not an NVVM
  // intrinsic, we don't have to look up any module metadata, as
  // FtzRequirementTy will be FTZ_Any.)
  if (Action.FtzRequirement != FTZ_Any) {
    StringRef Attr = II->getFunction()
                         ->getFnAttribute("denormal-fp-math-f32")
                         .getValueAsString();
    DenormalMode Mode = parseDenormalFPAttribute(Attr);
    bool FtzEnabled = Mode.Output != DenormalMode::IEEE;

    if (FtzEnabled != (Action.FtzRequirement == FTZ_MustBeOn))
      return nullptr;
  }

  // Simplify to target-generic intrinsic.
  if (Action.IID) {
    SmallVector<Value *, 4> Args(II->arg_operands());
    // All the target-generic intrinsics currently of interest to us have one
    // type argument, equal to that of the nvvm intrinsic's argument.
    Type *Tys[] = {II->getArgOperand(0)->getType()};
    return CallInst::Create(
        Intrinsic::getDeclaration(II->getModule(), *Action.IID, Tys), Args);
  }

  // Simplify to target-generic binary op.
  if (Action.BinaryOp)
    return BinaryOperator::Create(*Action.BinaryOp, II->getArgOperand(0),
                                  II->getArgOperand(1), II->getName());

  // Simplify to target-generic cast op.
  if (Action.CastOp)
    return CastInst::Create(*Action.CastOp, II->getArgOperand(0), II->getType(),
                            II->getName());

  // All that's left are the special cases.
  if (!Action.Special)
    return nullptr;

  switch (*Action.Special) {
  case SPC_Reciprocal:
    // Simplify reciprocal.
    return BinaryOperator::Create(
        Instruction::FDiv, ConstantFP::get(II->getArgOperand(0)->getType(), 1),
        II->getArgOperand(0), II->getName());
  }
  llvm_unreachable("All SpecialCase enumerators should be handled in switch.");
}

Instruction *InstCombiner::visitVAEndInst(VAEndInst &I) {
  removeTriviallyEmptyRange(I, *this, [](const IntrinsicInst &I) {
    return I.getIntrinsicID() == Intrinsic::vastart ||
           I.getIntrinsicID() == Intrinsic::vacopy;
  });
  return nullptr;
}

static Instruction *canonicalizeConstantArg0ToArg1(CallInst &Call) {
  assert(Call.getNumArgOperands() > 1 && "Need at least 2 args to swap");
  Value *Arg0 = Call.getArgOperand(0), *Arg1 = Call.getArgOperand(1);
  if (isa<Constant>(Arg0) && !isa<Constant>(Arg1)) {
    Call.setArgOperand(0, Arg1);
    Call.setArgOperand(1, Arg0);
    return &Call;
  }
  return nullptr;
}

Instruction *InstCombiner::foldIntrinsicWithOverflowCommon(IntrinsicInst *II) {
  WithOverflowInst *WO = cast<WithOverflowInst>(II);
  Value *OperationResult = nullptr;
  Constant *OverflowResult = nullptr;
  if (OptimizeOverflowCheck(WO->getBinaryOp(), WO->isSigned(), WO->getLHS(),
                            WO->getRHS(), *WO, OperationResult, OverflowResult))
    return CreateOverflowTuple(WO, OperationResult, OverflowResult);
  return nullptr;
}

/// CallInst simplification. This mostly only handles folding of intrinsic
/// instructions. For normal calls, it allows visitCallBase to do the heavy
/// lifting.
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  // Don't try to simplify calls without uses. It will not do anything useful,
  // but will result in the following folds being skipped.
  if (!CI.use_empty())
    if (Value *V = SimplifyCall(&CI, SQ.getWithInstruction(&CI)))
      return replaceInstUsesWith(CI, V);

  if (isFreeCall(&CI, &TLI))
    return visitFree(CI);

  // If the caller function is nounwind, mark the call as nounwind, even if the
  // callee isn't.
  if (CI.getFunction()->doesNotThrow() && !CI.doesNotThrow()) {
    CI.setDoesNotThrow();
    return &CI;
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CI);
  if (!II) return visitCallBase(CI);

  // For atomic unordered mem intrinsics if len is not a positive or
  // not a multiple of element size then behavior is undefined.
  if (auto *AMI = dyn_cast<AtomicMemIntrinsic>(II))
    if (ConstantInt *NumBytes = dyn_cast<ConstantInt>(AMI->getLength()))
      if (NumBytes->getSExtValue() < 0 ||
          (NumBytes->getZExtValue() % AMI->getElementSizeInBytes() != 0)) {
        CreateNonTerminatorUnreachable(AMI);
        assert(AMI->getType()->isVoidTy() &&
               "non void atomic unordered mem intrinsic");
        return eraseInstFromFunction(*AMI);
      }

  // Intrinsics cannot occur in an invoke or a callbr, so handle them here
  // instead of in visitCallBase.
  if (auto *MI = dyn_cast<AnyMemIntrinsic>(II)) {
    bool Changed = false;

    // memmove/cpy/set of zero bytes is a noop.
    if (Constant *NumBytes = dyn_cast<Constant>(MI->getLength())) {
      if (NumBytes->isNullValue())
        return eraseInstFromFunction(CI);

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getZExtValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }

    // No other transformations apply to volatile transfers.
    if (auto *M = dyn_cast<MemIntrinsic>(MI))
      if (M->isVolatile())
        return nullptr;

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (auto *MMI = dyn_cast<AnyMemMoveInst>(MI)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getModule();
          Intrinsic::ID MemCpyID =
              isa<AtomicMemMoveInst>(MMI)
                  ? Intrinsic::memcpy_element_unordered_atomic
                  : Intrinsic::memcpy;
          Type *Tys[3] = { CI.getArgOperand(0)->getType(),
                           CI.getArgOperand(1)->getType(),
                           CI.getArgOperand(2)->getType() };
          CI.setCalledFunction(Intrinsic::getDeclaration(M, MemCpyID, Tys));
          Changed = true;
        }
    }

    if (AnyMemTransferInst *MTI = dyn_cast<AnyMemTransferInst>(MI)) {
      // memmove(x,x,size) -> noop.
      if (MTI->getSource() == MTI->getDest())
        return eraseInstFromFunction(CI);
    }

    // If we can determine a pointer alignment that is bigger than currently
    // set, update the alignment.
    if (auto *MTI = dyn_cast<AnyMemTransferInst>(MI)) {
      if (Instruction *I = SimplifyAnyMemTransfer(MTI))
        return I;
    } else if (auto *MSI = dyn_cast<AnyMemSetInst>(MI)) {
      if (Instruction *I = SimplifyAnyMemSet(MSI))
        return I;
    }

    if (Changed) return II;
  }

  // For fixed width vector result intrinsics, use the generic demanded vector
  // support.
  if (auto *IIFVTy = dyn_cast<FixedVectorType>(II->getType())) {
    auto VWidth = IIFVTy->getNumElements();
    APInt UndefElts(VWidth, 0);
    APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
    if (Value *V = SimplifyDemandedVectorElts(II, AllOnesEltMask, UndefElts)) {
      if (V != II)
        return replaceInstUsesWith(*II, V);
      return II;
    }
  }

  if (Instruction *I = SimplifyNVVMIntrinsic(II, *this))
    return I;

  auto SimplifyDemandedVectorEltsLow = [this](Value *Op, unsigned Width,
                                              unsigned DemandedWidth) {
    APInt UndefElts(Width, 0);
    APInt DemandedElts = APInt::getLowBitsSet(Width, DemandedWidth);
    return SimplifyDemandedVectorElts(Op, DemandedElts, UndefElts);
  };

  Intrinsic::ID IID = II->getIntrinsicID();
  switch (IID) {
  default: break;
  case Intrinsic::objectsize:
    if (Value *V = lowerObjectSizeCall(II, DL, &TLI, /*MustSucceed=*/false))
      return replaceInstUsesWith(CI, V);
    return nullptr;
  case Intrinsic::bswap: {
    Value *IIOperand = II->getArgOperand(0);
    Value *X = nullptr;

    // bswap(trunc(bswap(x))) -> trunc(lshr(x, c))
    if (match(IIOperand, m_Trunc(m_BSwap(m_Value(X))))) {
      unsigned C = X->getType()->getPrimitiveSizeInBits() -
        IIOperand->getType()->getPrimitiveSizeInBits();
      Value *CV = ConstantInt::get(X->getType(), C);
      Value *V = Builder.CreateLShr(X, CV);
      return new TruncInst(V, IIOperand->getType());
    }
    break;
  }
  case Intrinsic::masked_load:
    if (Value *SimplifiedMaskedOp = simplifyMaskedLoad(*II))
      return replaceInstUsesWith(CI, SimplifiedMaskedOp);
    break;
  case Intrinsic::masked_store:
    return simplifyMaskedStore(*II);
  case Intrinsic::masked_gather:
    return simplifyMaskedGather(*II);
  case Intrinsic::masked_scatter:
    return simplifyMaskedScatter(*II);
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group:
    if (auto *SkippedBarrier = simplifyInvariantGroupIntrinsic(*II, *this))
      return replaceInstUsesWith(*II, SkippedBarrier);
    break;
  case Intrinsic::powi:
    if (ConstantInt *Power = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      // 0 and 1 are handled in instsimplify

      // powi(x, -1) -> 1/x
      if (Power->isMinusOne())
        return BinaryOperator::CreateFDiv(ConstantFP::get(CI.getType(), 1.0),
                                          II->getArgOperand(0));
      // powi(x, 2) -> x*x
      if (Power->equalsInt(2))
        return BinaryOperator::CreateFMul(II->getArgOperand(0),
                                          II->getArgOperand(0));
    }
    break;

  case Intrinsic::cttz:
  case Intrinsic::ctlz:
    if (auto *I = foldCttzCtlz(*II, *this))
      return I;
    break;

  case Intrinsic::ctpop:
    if (auto *I = foldCtpop(*II, *this))
      return I;
    break;

  case Intrinsic::fshl:
  case Intrinsic::fshr: {
    Value *Op0 = II->getArgOperand(0), *Op1 = II->getArgOperand(1);
    Type *Ty = II->getType();
    unsigned BitWidth = Ty->getScalarSizeInBits();
    Constant *ShAmtC;
    if (match(II->getArgOperand(2), m_Constant(ShAmtC)) &&
        !isa<ConstantExpr>(ShAmtC) && !ShAmtC->containsConstantExpression()) {
      // Canonicalize a shift amount constant operand to modulo the bit-width.
      Constant *WidthC = ConstantInt::get(Ty, BitWidth);
      Constant *ModuloC = ConstantExpr::getURem(ShAmtC, WidthC);
      if (ModuloC != ShAmtC)
        return replaceOperand(*II, 2, ModuloC);

      assert(ConstantExpr::getICmp(ICmpInst::ICMP_UGT, WidthC, ShAmtC) ==
                 ConstantInt::getTrue(CmpInst::makeCmpResultType(Ty)) &&
             "Shift amount expected to be modulo bitwidth");

      // Canonicalize funnel shift right by constant to funnel shift left. This
      // is not entirely arbitrary. For historical reasons, the backend may
      // recognize rotate left patterns but miss rotate right patterns.
      if (IID == Intrinsic::fshr) {
        // fshr X, Y, C --> fshl X, Y, (BitWidth - C)
        Constant *LeftShiftC = ConstantExpr::getSub(WidthC, ShAmtC);
        Module *Mod = II->getModule();
        Function *Fshl = Intrinsic::getDeclaration(Mod, Intrinsic::fshl, Ty);
        return CallInst::Create(Fshl, { Op0, Op1, LeftShiftC });
      }
      assert(IID == Intrinsic::fshl &&
             "All funnel shifts by simple constants should go left");

      // fshl(X, 0, C) --> shl X, C
      // fshl(X, undef, C) --> shl X, C
      if (match(Op1, m_ZeroInt()) || match(Op1, m_Undef()))
        return BinaryOperator::CreateShl(Op0, ShAmtC);

      // fshl(0, X, C) --> lshr X, (BW-C)
      // fshl(undef, X, C) --> lshr X, (BW-C)
      if (match(Op0, m_ZeroInt()) || match(Op0, m_Undef()))
        return BinaryOperator::CreateLShr(Op1,
                                          ConstantExpr::getSub(WidthC, ShAmtC));

      // fshl i16 X, X, 8 --> bswap i16 X (reduce to more-specific form)
      if (Op0 == Op1 && BitWidth == 16 && match(ShAmtC, m_SpecificInt(8))) {
        Module *Mod = II->getModule();
        Function *Bswap = Intrinsic::getDeclaration(Mod, Intrinsic::bswap, Ty);
        return CallInst::Create(Bswap, { Op0 });
      }
    }

    // Left or right might be masked.
    if (SimplifyDemandedInstructionBits(*II))
      return &CI;

    // The shift amount (operand 2) of a funnel shift is modulo the bitwidth,
    // so only the low bits of the shift amount are demanded if the bitwidth is
    // a power-of-2.
    if (!isPowerOf2_32(BitWidth))
      break;
    APInt Op2Demanded = APInt::getLowBitsSet(BitWidth, Log2_32_Ceil(BitWidth));
    KnownBits Op2Known(BitWidth);
    if (SimplifyDemandedBits(II, 2, Op2Demanded, Op2Known))
      return &CI;
    break;
  }
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow: {
    if (Instruction *I = canonicalizeConstantArg0ToArg1(CI))
      return I;
    if (Instruction *I = foldIntrinsicWithOverflowCommon(II))
      return I;

    // Given 2 constant operands whose sum does not overflow:
    // uaddo (X +nuw C0), C1 -> uaddo X, C0 + C1
    // saddo (X +nsw C0), C1 -> saddo X, C0 + C1
    Value *X;
    const APInt *C0, *C1;
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);
    bool IsSigned = IID == Intrinsic::sadd_with_overflow;
    bool HasNWAdd = IsSigned ? match(Arg0, m_NSWAdd(m_Value(X), m_APInt(C0)))
                             : match(Arg0, m_NUWAdd(m_Value(X), m_APInt(C0)));
    if (HasNWAdd && match(Arg1, m_APInt(C1))) {
      bool Overflow;
      APInt NewC =
          IsSigned ? C1->sadd_ov(*C0, Overflow) : C1->uadd_ov(*C0, Overflow);
      if (!Overflow)
        return replaceInstUsesWith(
            *II, Builder.CreateBinaryIntrinsic(
                     IID, X, ConstantInt::get(Arg1->getType(), NewC)));
    }
    break;
  }

  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    if (Instruction *I = canonicalizeConstantArg0ToArg1(CI))
      return I;
    LLVM_FALLTHROUGH;

  case Intrinsic::usub_with_overflow:
    if (Instruction *I = foldIntrinsicWithOverflowCommon(II))
      return I;
    break;

  case Intrinsic::ssub_with_overflow: {
    if (Instruction *I = foldIntrinsicWithOverflowCommon(II))
      return I;

    Constant *C;
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);
    // Given a constant C that is not the minimum signed value
    // for an integer of a given bit width:
    //
    // ssubo X, C -> saddo X, -C
    if (match(Arg1, m_Constant(C)) && C->isNotMinSignedValue()) {
      Value *NegVal = ConstantExpr::getNeg(C);
      // Build a saddo call that is equivalent to the discovered
      // ssubo call.
      return replaceInstUsesWith(
          *II, Builder.CreateBinaryIntrinsic(Intrinsic::sadd_with_overflow,
                                             Arg0, NegVal));
    }

    break;
  }

  case Intrinsic::uadd_sat:
  case Intrinsic::sadd_sat:
    if (Instruction *I = canonicalizeConstantArg0ToArg1(CI))
      return I;
    LLVM_FALLTHROUGH;
  case Intrinsic::usub_sat:
  case Intrinsic::ssub_sat: {
    SaturatingInst *SI = cast<SaturatingInst>(II);
    Type *Ty = SI->getType();
    Value *Arg0 = SI->getLHS();
    Value *Arg1 = SI->getRHS();

    // Make use of known overflow information.
    OverflowResult OR = computeOverflow(SI->getBinaryOp(), SI->isSigned(),
                                        Arg0, Arg1, SI);
    switch (OR) {
      case OverflowResult::MayOverflow:
        break;
      case OverflowResult::NeverOverflows:
        if (SI->isSigned())
          return BinaryOperator::CreateNSW(SI->getBinaryOp(), Arg0, Arg1);
        else
          return BinaryOperator::CreateNUW(SI->getBinaryOp(), Arg0, Arg1);
      case OverflowResult::AlwaysOverflowsLow: {
        unsigned BitWidth = Ty->getScalarSizeInBits();
        APInt Min = APSInt::getMinValue(BitWidth, !SI->isSigned());
        return replaceInstUsesWith(*SI, ConstantInt::get(Ty, Min));
      }
      case OverflowResult::AlwaysOverflowsHigh: {
        unsigned BitWidth = Ty->getScalarSizeInBits();
        APInt Max = APSInt::getMaxValue(BitWidth, !SI->isSigned());
        return replaceInstUsesWith(*SI, ConstantInt::get(Ty, Max));
      }
    }

    // ssub.sat(X, C) -> sadd.sat(X, -C) if C != MIN
    Constant *C;
    if (IID == Intrinsic::ssub_sat && match(Arg1, m_Constant(C)) &&
        C->isNotMinSignedValue()) {
      Value *NegVal = ConstantExpr::getNeg(C);
      return replaceInstUsesWith(
          *II, Builder.CreateBinaryIntrinsic(
              Intrinsic::sadd_sat, Arg0, NegVal));
    }

    // sat(sat(X + Val2) + Val) -> sat(X + (Val+Val2))
    // sat(sat(X - Val2) - Val) -> sat(X - (Val+Val2))
    // if Val and Val2 have the same sign
    if (auto *Other = dyn_cast<IntrinsicInst>(Arg0)) {
      Value *X;
      const APInt *Val, *Val2;
      APInt NewVal;
      bool IsUnsigned =
          IID == Intrinsic::uadd_sat || IID == Intrinsic::usub_sat;
      if (Other->getIntrinsicID() == IID &&
          match(Arg1, m_APInt(Val)) &&
          match(Other->getArgOperand(0), m_Value(X)) &&
          match(Other->getArgOperand(1), m_APInt(Val2))) {
        if (IsUnsigned)
          NewVal = Val->uadd_sat(*Val2);
        else if (Val->isNonNegative() == Val2->isNonNegative()) {
          bool Overflow;
          NewVal = Val->sadd_ov(*Val2, Overflow);
          if (Overflow) {
            // Both adds together may add more than SignedMaxValue
            // without saturating the final result.
            break;
          }
        } else {
          // Cannot fold saturated addition with different signs.
          break;
        }

        return replaceInstUsesWith(
            *II, Builder.CreateBinaryIntrinsic(
                     IID, X, ConstantInt::get(II->getType(), NewVal)));
      }
    }
    break;
  }

  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum: {
    if (Instruction *I = canonicalizeConstantArg0ToArg1(CI))
      return I;
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);
    Value *X, *Y;
    if (match(Arg0, m_FNeg(m_Value(X))) && match(Arg1, m_FNeg(m_Value(Y))) &&
        (Arg0->hasOneUse() || Arg1->hasOneUse())) {
      // If both operands are negated, invert the call and negate the result:
      // min(-X, -Y) --> -(max(X, Y))
      // max(-X, -Y) --> -(min(X, Y))
      Intrinsic::ID NewIID;
      switch (IID) {
      case Intrinsic::maxnum:
        NewIID = Intrinsic::minnum;
        break;
      case Intrinsic::minnum:
        NewIID = Intrinsic::maxnum;
        break;
      case Intrinsic::maximum:
        NewIID = Intrinsic::minimum;
        break;
      case Intrinsic::minimum:
        NewIID = Intrinsic::maximum;
        break;
      default:
        llvm_unreachable("unexpected intrinsic ID");
      }
      Value *NewCall = Builder.CreateBinaryIntrinsic(NewIID, X, Y, II);
      Instruction *FNeg = UnaryOperator::CreateFNeg(NewCall);
      FNeg->copyIRFlags(II);
      return FNeg;
    }

    // m(m(X, C2), C1) -> m(X, C)
    const APFloat *C1, *C2;
    if (auto *M = dyn_cast<IntrinsicInst>(Arg0)) {
      if (M->getIntrinsicID() == IID && match(Arg1, m_APFloat(C1)) &&
          ((match(M->getArgOperand(0), m_Value(X)) &&
            match(M->getArgOperand(1), m_APFloat(C2))) ||
           (match(M->getArgOperand(1), m_Value(X)) &&
            match(M->getArgOperand(0), m_APFloat(C2))))) {
        APFloat Res(0.0);
        switch (IID) {
        case Intrinsic::maxnum:
          Res = maxnum(*C1, *C2);
          break;
        case Intrinsic::minnum:
          Res = minnum(*C1, *C2);
          break;
        case Intrinsic::maximum:
          Res = maximum(*C1, *C2);
          break;
        case Intrinsic::minimum:
          Res = minimum(*C1, *C2);
          break;
        default:
          llvm_unreachable("unexpected intrinsic ID");
        }
        Instruction *NewCall = Builder.CreateBinaryIntrinsic(
            IID, X, ConstantFP::get(Arg0->getType(), Res), II);
        // TODO: Conservatively intersecting FMF. If Res == C2, the transform
        //       was a simplification (so Arg0 and its original flags could
        //       propagate?)
        NewCall->andIRFlags(M);
        return replaceInstUsesWith(*II, NewCall);
      }
    }

    Value *ExtSrc0;
    Value *ExtSrc1;

    // minnum (fpext x), (fpext y) -> minnum x, y
    // maxnum (fpext x), (fpext y) -> maxnum x, y
    if (match(II->getArgOperand(0), m_OneUse(m_FPExt(m_Value(ExtSrc0)))) &&
        match(II->getArgOperand(1), m_OneUse(m_FPExt(m_Value(ExtSrc1)))) &&
        ExtSrc0->getType() == ExtSrc1->getType()) {
      Function *F = Intrinsic::getDeclaration(
          II->getModule(), II->getIntrinsicID(), {ExtSrc0->getType()});
      CallInst *NewCall = Builder.CreateCall(F, { ExtSrc0, ExtSrc1 });
      NewCall->copyFastMathFlags(II);
      NewCall->takeName(II);
      return new FPExtInst(NewCall, II->getType());
    }

    break;
  }
  case Intrinsic::fmuladd: {
    // Canonicalize fast fmuladd to the separate fmul + fadd.
    if (II->isFast()) {
      BuilderTy::FastMathFlagGuard Guard(Builder);
      Builder.setFastMathFlags(II->getFastMathFlags());
      Value *Mul = Builder.CreateFMul(II->getArgOperand(0),
                                      II->getArgOperand(1));
      Value *Add = Builder.CreateFAdd(Mul, II->getArgOperand(2));
      Add->takeName(II);
      return replaceInstUsesWith(*II, Add);
    }

    // Try to simplify the underlying FMul.
    if (Value *V = SimplifyFMulInst(II->getArgOperand(0), II->getArgOperand(1),
                                    II->getFastMathFlags(),
                                    SQ.getWithInstruction(II))) {
      auto *FAdd = BinaryOperator::CreateFAdd(V, II->getArgOperand(2));
      FAdd->copyFastMathFlags(II);
      return FAdd;
    }

    LLVM_FALLTHROUGH;
  }
  case Intrinsic::fma: {
    if (Instruction *I = canonicalizeConstantArg0ToArg1(CI))
      return I;

    // fma fneg(x), fneg(y), z -> fma x, y, z
    Value *Src0 = II->getArgOperand(0);
    Value *Src1 = II->getArgOperand(1);
    Value *X, *Y;
    if (match(Src0, m_FNeg(m_Value(X))) && match(Src1, m_FNeg(m_Value(Y)))) {
      replaceOperand(*II, 0, X);
      replaceOperand(*II, 1, Y);
      return II;
    }

    // fma fabs(x), fabs(x), z -> fma x, x, z
    if (match(Src0, m_FAbs(m_Value(X))) &&
        match(Src1, m_FAbs(m_Specific(X)))) {
      replaceOperand(*II, 0, X);
      replaceOperand(*II, 1, X);
      return II;
    }

    // Try to simplify the underlying FMul. We can only apply simplifications
    // that do not require rounding.
    if (Value *V = SimplifyFMAFMul(II->getArgOperand(0), II->getArgOperand(1),
                                   II->getFastMathFlags(),
                                   SQ.getWithInstruction(II))) {
      auto *FAdd = BinaryOperator::CreateFAdd(V, II->getArgOperand(2));
      FAdd->copyFastMathFlags(II);
      return FAdd;
    }

    break;
  }
  case Intrinsic::copysign: {
    if (SignBitMustBeZero(II->getArgOperand(1), &TLI)) {
      // If we know that the sign argument is positive, reduce to FABS:
      // copysign X, Pos --> fabs X
      Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs,
                                                 II->getArgOperand(0), II);
      return replaceInstUsesWith(*II, Fabs);
    }
    // TODO: There should be a ValueTracking sibling like SignBitMustBeOne.
    const APFloat *C;
    if (match(II->getArgOperand(1), m_APFloat(C)) && C->isNegative()) {
      // If we know that the sign argument is negative, reduce to FNABS:
      // copysign X, Neg --> fneg (fabs X)
      Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs,
                                                 II->getArgOperand(0), II);
      return replaceInstUsesWith(*II, Builder.CreateFNegFMF(Fabs, II));
    }

    // Propagate sign argument through nested calls:
    // copysign X, (copysign ?, SignArg) --> copysign X, SignArg
    Value *SignArg;
    if (match(II->getArgOperand(1),
              m_Intrinsic<Intrinsic::copysign>(m_Value(), m_Value(SignArg))))
      return replaceOperand(*II, 1, SignArg);

    break;
  }
  case Intrinsic::fabs: {
    Value *Cond;
    Constant *LHS, *RHS;
    if (match(II->getArgOperand(0),
              m_Select(m_Value(Cond), m_Constant(LHS), m_Constant(RHS)))) {
      CallInst *Call0 = Builder.CreateCall(II->getCalledFunction(), {LHS});
      CallInst *Call1 = Builder.CreateCall(II->getCalledFunction(), {RHS});
      return SelectInst::Create(Cond, Call0, Call1);
    }

    LLVM_FALLTHROUGH;
  }
  case Intrinsic::ceil:
  case Intrinsic::floor:
  case Intrinsic::round:
  case Intrinsic::nearbyint:
  case Intrinsic::rint:
  case Intrinsic::trunc: {
    Value *ExtSrc;
    if (match(II->getArgOperand(0), m_OneUse(m_FPExt(m_Value(ExtSrc))))) {
      // Narrow the call: intrinsic (fpext x) -> fpext (intrinsic x)
      Value *NarrowII = Builder.CreateUnaryIntrinsic(IID, ExtSrc, II);
      return new FPExtInst(NarrowII, II->getType());
    }
    break;
  }
  case Intrinsic::cos:
  case Intrinsic::amdgcn_cos: {
    Value *X;
    Value *Src = II->getArgOperand(0);
    if (match(Src, m_FNeg(m_Value(X))) || match(Src, m_FAbs(m_Value(X)))) {
      // cos(-x) -> cos(x)
      // cos(fabs(x)) -> cos(x)
      return replaceOperand(*II, 0, X);
    }
    break;
  }
  case Intrinsic::sin: {
    Value *X;
    if (match(II->getArgOperand(0), m_OneUse(m_FNeg(m_Value(X))))) {
      // sin(-x) --> -sin(x)
      Value *NewSin = Builder.CreateUnaryIntrinsic(Intrinsic::sin, X, II);
      Instruction *FNeg = UnaryOperator::CreateFNeg(NewSin);
      FNeg->copyFastMathFlags(II);
      return FNeg;
    }
    break;
  }
  case Intrinsic::ppc_altivec_lvx:
  case Intrinsic::ppc_altivec_lvxl:
    // Turn PPC lvx -> load if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), Align(16), DL, II, &AC,
                                   &DT) >= 16) {
      Value *Ptr = Builder.CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(II->getType(), Ptr, "", false, Align(16));
    }
    break;
  case Intrinsic::ppc_vsx_lxvw4x:
  case Intrinsic::ppc_vsx_lxvd2x: {
    // Turn PPC VSX loads into normal loads.
    Value *Ptr = Builder.CreateBitCast(II->getArgOperand(0),
                                       PointerType::getUnqual(II->getType()));
    return new LoadInst(II->getType(), Ptr, Twine(""), false, Align(1));
  }
  case Intrinsic::ppc_altivec_stvx:
  case Intrinsic::ppc_altivec_stvxl:
    // Turn stvx -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), Align(16), DL, II, &AC,
                                   &DT) >= 16) {
      Type *OpPtrTy =
        PointerType::getUnqual(II->getArgOperand(0)->getType());
      Value *Ptr = Builder.CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(II->getArgOperand(0), Ptr);
    }
    break;
  case Intrinsic::ppc_vsx_stxvw4x:
  case Intrinsic::ppc_vsx_stxvd2x: {
    // Turn PPC VSX stores into normal stores.
    Type *OpPtrTy = PointerType::getUnqual(II->getArgOperand(0)->getType());
    Value *Ptr = Builder.CreateBitCast(II->getArgOperand(1), OpPtrTy);
    return new StoreInst(II->getArgOperand(0), Ptr, false, Align(1));
  }
  case Intrinsic::ppc_qpx_qvlfs:
    // Turn PPC QPX qvlfs -> load if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), Align(16), DL, II, &AC,
                                   &DT) >= 16) {
      Type *VTy =
          VectorType::get(Builder.getFloatTy(),
                          cast<VectorType>(II->getType())->getElementCount());
      Value *Ptr = Builder.CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(VTy));
      Value *Load = Builder.CreateLoad(VTy, Ptr);
      return new FPExtInst(Load, II->getType());
    }
    break;
  case Intrinsic::ppc_qpx_qvlfd:
    // Turn PPC QPX qvlfd -> load if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), Align(32), DL, II, &AC,
                                   &DT) >= 32) {
      Value *Ptr = Builder.CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(II->getType(), Ptr, "", false, Align(32));
    }
    break;
  case Intrinsic::ppc_qpx_qvstfs:
    // Turn PPC QPX qvstfs -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), Align(16), DL, II, &AC,
                                   &DT) >= 16) {
      Type *VTy = VectorType::get(
          Builder.getFloatTy(),
          cast<VectorType>(II->getArgOperand(0)->getType())->getElementCount());
      Value *TOp = Builder.CreateFPTrunc(II->getArgOperand(0), VTy);
      Type *OpPtrTy = PointerType::getUnqual(VTy);
      Value *Ptr = Builder.CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(TOp, Ptr);
    }
    break;
  case Intrinsic::ppc_qpx_qvstfd:
    // Turn PPC QPX qvstfd -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), Align(32), DL, II, &AC,
                                   &DT) >= 32) {
      Type *OpPtrTy =
        PointerType::getUnqual(II->getArgOperand(0)->getType());
      Value *Ptr = Builder.CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(II->getArgOperand(0), Ptr);
    }
    break;

  case Intrinsic::x86_bmi_bextr_32:
  case Intrinsic::x86_bmi_bextr_64:
  case Intrinsic::x86_tbm_bextri_u32:
  case Intrinsic::x86_tbm_bextri_u64:
    // If the RHS is a constant we can try some simplifications.
    if (auto *C = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      uint64_t Shift = C->getZExtValue();
      uint64_t Length = (Shift >> 8) & 0xff;
      Shift &= 0xff;
      unsigned BitWidth = II->getType()->getIntegerBitWidth();
      // If the length is 0 or the shift is out of range, replace with zero.
      if (Length == 0 || Shift >= BitWidth)
        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), 0));
      // If the LHS is also a constant, we can completely constant fold this.
      if (auto *InC = dyn_cast<ConstantInt>(II->getArgOperand(0))) {
        uint64_t Result = InC->getZExtValue() >> Shift;
        if (Length > BitWidth)
          Length = BitWidth;
        Result &= maskTrailingOnes<uint64_t>(Length);
        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), Result));
      }
      // TODO should we turn this into 'and' if shift is 0? Or 'shl' if we
      // are only masking bits that a shift already cleared?
    }
    break;

  case Intrinsic::x86_bmi_bzhi_32:
  case Intrinsic::x86_bmi_bzhi_64:
    // If the RHS is a constant we can try some simplifications.
    if (auto *C = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      uint64_t Index = C->getZExtValue() & 0xff;
      unsigned BitWidth = II->getType()->getIntegerBitWidth();
      if (Index >= BitWidth)
        return replaceInstUsesWith(CI, II->getArgOperand(0));
      if (Index == 0)
        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), 0));
      // If the LHS is also a constant, we can completely constant fold this.
      if (auto *InC = dyn_cast<ConstantInt>(II->getArgOperand(0))) {
        uint64_t Result = InC->getZExtValue();
        Result &= maskTrailingOnes<uint64_t>(Index);
        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), Result));
      }
      // TODO should we convert this to an AND if the RHS is constant?
    }
    break;
  case Intrinsic::x86_bmi_pext_32:
  case Intrinsic::x86_bmi_pext_64:
    if (auto *MaskC = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      if (MaskC->isNullValue())
        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), 0));
      if (MaskC->isAllOnesValue())
        return replaceInstUsesWith(CI, II->getArgOperand(0));

      if (auto *SrcC = dyn_cast<ConstantInt>(II->getArgOperand(0))) {
        uint64_t Src = SrcC->getZExtValue();
        uint64_t Mask = MaskC->getZExtValue();
        uint64_t Result = 0;
        uint64_t BitToSet = 1;

        while (Mask) {
          // Isolate lowest set bit.
          uint64_t BitToTest = Mask & -Mask;
          if (BitToTest & Src)
            Result |= BitToSet;

          BitToSet <<= 1;
          // Clear lowest set bit.
          Mask &= Mask - 1;
        }

        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), Result));
      }
    }
    break;
  case Intrinsic::x86_bmi_pdep_32:
  case Intrinsic::x86_bmi_pdep_64:
    if (auto *MaskC = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      if (MaskC->isNullValue())
        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), 0));
      if (MaskC->isAllOnesValue())
        return replaceInstUsesWith(CI, II->getArgOperand(0));

      if (auto *SrcC = dyn_cast<ConstantInt>(II->getArgOperand(0))) {
        uint64_t Src = SrcC->getZExtValue();
        uint64_t Mask = MaskC->getZExtValue();
        uint64_t Result = 0;
        uint64_t BitToTest = 1;

        while (Mask) {
          // Isolate lowest set bit.
          uint64_t BitToSet = Mask & -Mask;
          if (BitToTest & Src)
            Result |= BitToSet;

          BitToTest <<= 1;
          // Clear lowest set bit;
          Mask &= Mask - 1;
        }

        return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), Result));
      }
    }
    break;

  case Intrinsic::x86_sse_cvtss2si:
  case Intrinsic::x86_sse_cvtss2si64:
  case Intrinsic::x86_sse_cvttss2si:
  case Intrinsic::x86_sse_cvttss2si64:
  case Intrinsic::x86_sse2_cvtsd2si:
  case Intrinsic::x86_sse2_cvtsd2si64:
  case Intrinsic::x86_sse2_cvttsd2si:
  case Intrinsic::x86_sse2_cvttsd2si64:
  case Intrinsic::x86_avx512_vcvtss2si32:
  case Intrinsic::x86_avx512_vcvtss2si64:
  case Intrinsic::x86_avx512_vcvtss2usi32:
  case Intrinsic::x86_avx512_vcvtss2usi64:
  case Intrinsic::x86_avx512_vcvtsd2si32:
  case Intrinsic::x86_avx512_vcvtsd2si64:
  case Intrinsic::x86_avx512_vcvtsd2usi32:
  case Intrinsic::x86_avx512_vcvtsd2usi64:
  case Intrinsic::x86_avx512_cvttss2si:
  case Intrinsic::x86_avx512_cvttss2si64:
  case Intrinsic::x86_avx512_cvttss2usi:
  case Intrinsic::x86_avx512_cvttss2usi64:
  case Intrinsic::x86_avx512_cvttsd2si:
  case Intrinsic::x86_avx512_cvttsd2si64:
  case Intrinsic::x86_avx512_cvttsd2usi:
  case Intrinsic::x86_avx512_cvttsd2usi64: {
    // These intrinsics only demand the 0th element of their input vectors. If
    // we can simplify the input based on that, do so now.
    Value *Arg = II->getArgOperand(0);
    unsigned VWidth = cast<VectorType>(Arg->getType())->getNumElements();
    if (Value *V = SimplifyDemandedVectorEltsLow(Arg, VWidth, 1))
      return replaceOperand(*II, 0, V);
    break;
  }

  case Intrinsic::x86_mmx_pmovmskb:
  case Intrinsic::x86_sse_movmsk_ps:
  case Intrinsic::x86_sse2_movmsk_pd:
  case Intrinsic::x86_sse2_pmovmskb_128:
  case Intrinsic::x86_avx_movmsk_pd_256:
  case Intrinsic::x86_avx_movmsk_ps_256:
  case Intrinsic::x86_avx2_pmovmskb:
    if (Value *V = simplifyX86movmsk(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_sse_comieq_ss:
  case Intrinsic::x86_sse_comige_ss:
  case Intrinsic::x86_sse_comigt_ss:
  case Intrinsic::x86_sse_comile_ss:
  case Intrinsic::x86_sse_comilt_ss:
  case Intrinsic::x86_sse_comineq_ss:
  case Intrinsic::x86_sse_ucomieq_ss:
  case Intrinsic::x86_sse_ucomige_ss:
  case Intrinsic::x86_sse_ucomigt_ss:
  case Intrinsic::x86_sse_ucomile_ss:
  case Intrinsic::x86_sse_ucomilt_ss:
  case Intrinsic::x86_sse_ucomineq_ss:
  case Intrinsic::x86_sse2_comieq_sd:
  case Intrinsic::x86_sse2_comige_sd:
  case Intrinsic::x86_sse2_comigt_sd:
  case Intrinsic::x86_sse2_comile_sd:
  case Intrinsic::x86_sse2_comilt_sd:
  case Intrinsic::x86_sse2_comineq_sd:
  case Intrinsic::x86_sse2_ucomieq_sd:
  case Intrinsic::x86_sse2_ucomige_sd:
  case Intrinsic::x86_sse2_ucomigt_sd:
  case Intrinsic::x86_sse2_ucomile_sd:
  case Intrinsic::x86_sse2_ucomilt_sd:
  case Intrinsic::x86_sse2_ucomineq_sd:
  case Intrinsic::x86_avx512_vcomi_ss:
  case Intrinsic::x86_avx512_vcomi_sd:
  case Intrinsic::x86_avx512_mask_cmp_ss:
  case Intrinsic::x86_avx512_mask_cmp_sd: {
    // These intrinsics only demand the 0th element of their input vectors. If
    // we can simplify the input based on that, do so now.
    bool MadeChange = false;
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);
    unsigned VWidth = cast<VectorType>(Arg0->getType())->getNumElements();
    if (Value *V = SimplifyDemandedVectorEltsLow(Arg0, VWidth, 1)) {
      replaceOperand(*II, 0, V);
      MadeChange = true;
    }
    if (Value *V = SimplifyDemandedVectorEltsLow(Arg1, VWidth, 1)) {
      replaceOperand(*II, 1, V);
      MadeChange = true;
    }
    if (MadeChange)
      return II;
    break;
  }
  case Intrinsic::x86_avx512_cmp_pd_128:
  case Intrinsic::x86_avx512_cmp_pd_256:
  case Intrinsic::x86_avx512_cmp_pd_512:
  case Intrinsic::x86_avx512_cmp_ps_128:
  case Intrinsic::x86_avx512_cmp_ps_256:
  case Intrinsic::x86_avx512_cmp_ps_512: {
    // Folding cmp(sub(a,b),0) -> cmp(a,b) and cmp(0,sub(a,b)) -> cmp(b,a)
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);
    bool Arg0IsZero = match(Arg0, m_PosZeroFP());
    if (Arg0IsZero)
      std::swap(Arg0, Arg1);
    Value *A, *B;
    // This fold requires only the NINF(not +/- inf) since inf minus
    // inf is nan.
    // NSZ(No Signed Zeros) is not needed because zeros of any sign are
    // equal for both compares.
    // NNAN is not needed because nans compare the same for both compares.
    // The compare intrinsic uses the above assumptions and therefore
    // doesn't require additional flags.
    if ((match(Arg0, m_OneUse(m_FSub(m_Value(A), m_Value(B)))) &&
         match(Arg1, m_PosZeroFP()) && isa<Instruction>(Arg0) &&
         cast<Instruction>(Arg0)->getFastMathFlags().noInfs())) {
      if (Arg0IsZero)
        std::swap(A, B);
      replaceOperand(*II, 0, A);
      replaceOperand(*II, 1, B);
      return II;
    }
    break;
  }

  case Intrinsic::x86_avx512_add_ps_512:
  case Intrinsic::x86_avx512_div_ps_512:
  case Intrinsic::x86_avx512_mul_ps_512:
  case Intrinsic::x86_avx512_sub_ps_512:
  case Intrinsic::x86_avx512_add_pd_512:
  case Intrinsic::x86_avx512_div_pd_512:
  case Intrinsic::x86_avx512_mul_pd_512:
  case Intrinsic::x86_avx512_sub_pd_512:
    // If the rounding mode is CUR_DIRECTION(4) we can turn these into regular
    // IR operations.
    if (auto *R = dyn_cast<ConstantInt>(II->getArgOperand(2))) {
      if (R->getValue() == 4) {
        Value *Arg0 = II->getArgOperand(0);
        Value *Arg1 = II->getArgOperand(1);

        Value *V;
        switch (IID) {
        default: llvm_unreachable("Case stmts out of sync!");
        case Intrinsic::x86_avx512_add_ps_512:
        case Intrinsic::x86_avx512_add_pd_512:
          V = Builder.CreateFAdd(Arg0, Arg1);
          break;
        case Intrinsic::x86_avx512_sub_ps_512:
        case Intrinsic::x86_avx512_sub_pd_512:
          V = Builder.CreateFSub(Arg0, Arg1);
          break;
        case Intrinsic::x86_avx512_mul_ps_512:
        case Intrinsic::x86_avx512_mul_pd_512:
          V = Builder.CreateFMul(Arg0, Arg1);
          break;
        case Intrinsic::x86_avx512_div_ps_512:
        case Intrinsic::x86_avx512_div_pd_512:
          V = Builder.CreateFDiv(Arg0, Arg1);
          break;
        }

        return replaceInstUsesWith(*II, V);
      }
    }
    break;

  case Intrinsic::x86_avx512_mask_add_ss_round:
  case Intrinsic::x86_avx512_mask_div_ss_round:
  case Intrinsic::x86_avx512_mask_mul_ss_round:
  case Intrinsic::x86_avx512_mask_sub_ss_round:
  case Intrinsic::x86_avx512_mask_add_sd_round:
  case Intrinsic::x86_avx512_mask_div_sd_round:
  case Intrinsic::x86_avx512_mask_mul_sd_round:
  case Intrinsic::x86_avx512_mask_sub_sd_round:
    // If the rounding mode is CUR_DIRECTION(4) we can turn these into regular
    // IR operations.
    if (auto *R = dyn_cast<ConstantInt>(II->getArgOperand(4))) {
      if (R->getValue() == 4) {
        // Extract the element as scalars.
        Value *Arg0 = II->getArgOperand(0);
        Value *Arg1 = II->getArgOperand(1);
        Value *LHS = Builder.CreateExtractElement(Arg0, (uint64_t)0);
        Value *RHS = Builder.CreateExtractElement(Arg1, (uint64_t)0);

        Value *V;
        switch (IID) {
        default: llvm_unreachable("Case stmts out of sync!");
        case Intrinsic::x86_avx512_mask_add_ss_round:
        case Intrinsic::x86_avx512_mask_add_sd_round:
          V = Builder.CreateFAdd(LHS, RHS);
          break;
        case Intrinsic::x86_avx512_mask_sub_ss_round:
        case Intrinsic::x86_avx512_mask_sub_sd_round:
          V = Builder.CreateFSub(LHS, RHS);
          break;
        case Intrinsic::x86_avx512_mask_mul_ss_round:
        case Intrinsic::x86_avx512_mask_mul_sd_round:
          V = Builder.CreateFMul(LHS, RHS);
          break;
        case Intrinsic::x86_avx512_mask_div_ss_round:
        case Intrinsic::x86_avx512_mask_div_sd_round:
          V = Builder.CreateFDiv(LHS, RHS);
          break;
        }

        // Handle the masking aspect of the intrinsic.
        Value *Mask = II->getArgOperand(3);
        auto *C = dyn_cast<ConstantInt>(Mask);
        // We don't need a select if we know the mask bit is a 1.
        if (!C || !C->getValue()[0]) {
          // Cast the mask to an i1 vector and then extract the lowest element.
          auto *MaskTy = VectorType::get(Builder.getInt1Ty(),
                             cast<IntegerType>(Mask->getType())->getBitWidth());
          Mask = Builder.CreateBitCast(Mask, MaskTy);
          Mask = Builder.CreateExtractElement(Mask, (uint64_t)0);
          // Extract the lowest element from the passthru operand.
          Value *Passthru = Builder.CreateExtractElement(II->getArgOperand(2),
                                                          (uint64_t)0);
          V = Builder.CreateSelect(Mask, V, Passthru);
        }

        // Insert the result back into the original argument 0.
        V = Builder.CreateInsertElement(Arg0, V, (uint64_t)0);

        return replaceInstUsesWith(*II, V);
      }
    }
    break;

  // Constant fold ashr( <A x Bi>, Ci ).
  // Constant fold lshr( <A x Bi>, Ci ).
  // Constant fold shl( <A x Bi>, Ci ).
  case Intrinsic::x86_sse2_psrai_d:
  case Intrinsic::x86_sse2_psrai_w:
  case Intrinsic::x86_avx2_psrai_d:
  case Intrinsic::x86_avx2_psrai_w:
  case Intrinsic::x86_avx512_psrai_q_128:
  case Intrinsic::x86_avx512_psrai_q_256:
  case Intrinsic::x86_avx512_psrai_d_512:
  case Intrinsic::x86_avx512_psrai_q_512:
  case Intrinsic::x86_avx512_psrai_w_512:
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_avx2_psrli_d:
  case Intrinsic::x86_avx2_psrli_q:
  case Intrinsic::x86_avx2_psrli_w:
  case Intrinsic::x86_avx512_psrli_d_512:
  case Intrinsic::x86_avx512_psrli_q_512:
  case Intrinsic::x86_avx512_psrli_w_512:
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_avx2_pslli_d:
  case Intrinsic::x86_avx2_pslli_q:
  case Intrinsic::x86_avx2_pslli_w:
  case Intrinsic::x86_avx512_pslli_d_512:
  case Intrinsic::x86_avx512_pslli_q_512:
  case Intrinsic::x86_avx512_pslli_w_512:
    if (Value *V = simplifyX86immShift(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_sse2_psra_d:
  case Intrinsic::x86_sse2_psra_w:
  case Intrinsic::x86_avx2_psra_d:
  case Intrinsic::x86_avx2_psra_w:
  case Intrinsic::x86_avx512_psra_q_128:
  case Intrinsic::x86_avx512_psra_q_256:
  case Intrinsic::x86_avx512_psra_d_512:
  case Intrinsic::x86_avx512_psra_q_512:
  case Intrinsic::x86_avx512_psra_w_512:
  case Intrinsic::x86_sse2_psrl_d:
  case Intrinsic::x86_sse2_psrl_q:
  case Intrinsic::x86_sse2_psrl_w:
  case Intrinsic::x86_avx2_psrl_d:
  case Intrinsic::x86_avx2_psrl_q:
  case Intrinsic::x86_avx2_psrl_w:
  case Intrinsic::x86_avx512_psrl_d_512:
  case Intrinsic::x86_avx512_psrl_q_512:
  case Intrinsic::x86_avx512_psrl_w_512:
  case Intrinsic::x86_sse2_psll_d:
  case Intrinsic::x86_sse2_psll_q:
  case Intrinsic::x86_sse2_psll_w:
  case Intrinsic::x86_avx2_psll_d:
  case Intrinsic::x86_avx2_psll_q:
  case Intrinsic::x86_avx2_psll_w:
  case Intrinsic::x86_avx512_psll_d_512:
  case Intrinsic::x86_avx512_psll_q_512:
  case Intrinsic::x86_avx512_psll_w_512: {
    if (Value *V = simplifyX86immShift(*II, Builder))
      return replaceInstUsesWith(*II, V);

    // SSE2/AVX2 uses only the first 64-bits of the 128-bit vector
    // operand to compute the shift amount.
    Value *Arg1 = II->getArgOperand(1);
    assert(Arg1->getType()->getPrimitiveSizeInBits() == 128 &&
           "Unexpected packed shift size");
    unsigned VWidth = cast<VectorType>(Arg1->getType())->getNumElements();

    if (Value *V = SimplifyDemandedVectorEltsLow(Arg1, VWidth, VWidth / 2))
      return replaceOperand(*II, 1, V);
    break;
  }

  case Intrinsic::x86_avx2_psllv_d:
  case Intrinsic::x86_avx2_psllv_d_256:
  case Intrinsic::x86_avx2_psllv_q:
  case Intrinsic::x86_avx2_psllv_q_256:
  case Intrinsic::x86_avx512_psllv_d_512:
  case Intrinsic::x86_avx512_psllv_q_512:
  case Intrinsic::x86_avx512_psllv_w_128:
  case Intrinsic::x86_avx512_psllv_w_256:
  case Intrinsic::x86_avx512_psllv_w_512:
  case Intrinsic::x86_avx2_psrav_d:
  case Intrinsic::x86_avx2_psrav_d_256:
  case Intrinsic::x86_avx512_psrav_q_128:
  case Intrinsic::x86_avx512_psrav_q_256:
  case Intrinsic::x86_avx512_psrav_d_512:
  case Intrinsic::x86_avx512_psrav_q_512:
  case Intrinsic::x86_avx512_psrav_w_128:
  case Intrinsic::x86_avx512_psrav_w_256:
  case Intrinsic::x86_avx512_psrav_w_512:
  case Intrinsic::x86_avx2_psrlv_d:
  case Intrinsic::x86_avx2_psrlv_d_256:
  case Intrinsic::x86_avx2_psrlv_q:
  case Intrinsic::x86_avx2_psrlv_q_256:
  case Intrinsic::x86_avx512_psrlv_d_512:
  case Intrinsic::x86_avx512_psrlv_q_512:
  case Intrinsic::x86_avx512_psrlv_w_128:
  case Intrinsic::x86_avx512_psrlv_w_256:
  case Intrinsic::x86_avx512_psrlv_w_512:
    if (Value *V = simplifyX86varShift(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_sse2_packssdw_128:
  case Intrinsic::x86_sse2_packsswb_128:
  case Intrinsic::x86_avx2_packssdw:
  case Intrinsic::x86_avx2_packsswb:
  case Intrinsic::x86_avx512_packssdw_512:
  case Intrinsic::x86_avx512_packsswb_512:
    if (Value *V = simplifyX86pack(*II, Builder, true))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_sse2_packuswb_128:
  case Intrinsic::x86_sse41_packusdw:
  case Intrinsic::x86_avx2_packusdw:
  case Intrinsic::x86_avx2_packuswb:
  case Intrinsic::x86_avx512_packusdw_512:
  case Intrinsic::x86_avx512_packuswb_512:
    if (Value *V = simplifyX86pack(*II, Builder, false))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_pclmulqdq:
  case Intrinsic::x86_pclmulqdq_256:
  case Intrinsic::x86_pclmulqdq_512: {
    if (auto *C = dyn_cast<ConstantInt>(II->getArgOperand(2))) {
      unsigned Imm = C->getZExtValue();

      bool MadeChange = false;
      Value *Arg0 = II->getArgOperand(0);
      Value *Arg1 = II->getArgOperand(1);
      unsigned VWidth = cast<VectorType>(Arg0->getType())->getNumElements();

      APInt UndefElts1(VWidth, 0);
      APInt DemandedElts1 = APInt::getSplat(VWidth,
                                            APInt(2, (Imm & 0x01) ? 2 : 1));
      if (Value *V = SimplifyDemandedVectorElts(Arg0, DemandedElts1,
                                                UndefElts1)) {
        replaceOperand(*II, 0, V);
        MadeChange = true;
      }

      APInt UndefElts2(VWidth, 0);
      APInt DemandedElts2 = APInt::getSplat(VWidth,
                                            APInt(2, (Imm & 0x10) ? 2 : 1));
      if (Value *V = SimplifyDemandedVectorElts(Arg1, DemandedElts2,
                                                UndefElts2)) {
        replaceOperand(*II, 1, V);
        MadeChange = true;
      }

      // If either input elements are undef, the result is zero.
      if (DemandedElts1.isSubsetOf(UndefElts1) ||
          DemandedElts2.isSubsetOf(UndefElts2))
        return replaceInstUsesWith(*II,
                                   ConstantAggregateZero::get(II->getType()));

      if (MadeChange)
        return II;
    }
    break;
  }

  case Intrinsic::x86_sse41_insertps:
    if (Value *V = simplifyX86insertps(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_sse4a_extrq: {
    Value *Op0 = II->getArgOperand(0);
    Value *Op1 = II->getArgOperand(1);
    unsigned VWidth0 = cast<VectorType>(Op0->getType())->getNumElements();
    unsigned VWidth1 = cast<VectorType>(Op1->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 &&
           Op1->getType()->getPrimitiveSizeInBits() == 128 && VWidth0 == 2 &&
           VWidth1 == 16 && "Unexpected operand sizes");

    // See if we're dealing with constant values.
    Constant *C1 = dyn_cast<Constant>(Op1);
    ConstantInt *CILength =
        C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)0))
           : nullptr;
    ConstantInt *CIIndex =
        C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)1))
           : nullptr;

    // Attempt to simplify to a constant, shuffle vector or EXTRQI call.
    if (Value *V = simplifyX86extrq(*II, Op0, CILength, CIIndex, Builder))
      return replaceInstUsesWith(*II, V);

    // EXTRQ only uses the lowest 64-bits of the first 128-bit vector
    // operands and the lowest 16-bits of the second.
    bool MadeChange = false;
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth0, 1)) {
      replaceOperand(*II, 0, V);
      MadeChange = true;
    }
    if (Value *V = SimplifyDemandedVectorEltsLow(Op1, VWidth1, 2)) {
      replaceOperand(*II, 1, V);
      MadeChange = true;
    }
    if (MadeChange)
      return II;
    break;
  }

  case Intrinsic::x86_sse4a_extrqi: {
    // EXTRQI: Extract Length bits starting from Index. Zero pad the remaining
    // bits of the lower 64-bits. The upper 64-bits are undefined.
    Value *Op0 = II->getArgOperand(0);
    unsigned VWidth = cast<VectorType>(Op0->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 && VWidth == 2 &&
           "Unexpected operand size");

    // See if we're dealing with constant values.
    ConstantInt *CILength = dyn_cast<ConstantInt>(II->getArgOperand(1));
    ConstantInt *CIIndex = dyn_cast<ConstantInt>(II->getArgOperand(2));

    // Attempt to simplify to a constant or shuffle vector.
    if (Value *V = simplifyX86extrq(*II, Op0, CILength, CIIndex, Builder))
      return replaceInstUsesWith(*II, V);

    // EXTRQI only uses the lowest 64-bits of the first 128-bit vector
    // operand.
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth, 1))
      return replaceOperand(*II, 0, V);
    break;
  }

  case Intrinsic::x86_sse4a_insertq: {
    Value *Op0 = II->getArgOperand(0);
    Value *Op1 = II->getArgOperand(1);
    unsigned VWidth = cast<VectorType>(Op0->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 &&
           Op1->getType()->getPrimitiveSizeInBits() == 128 && VWidth == 2 &&
           cast<VectorType>(Op1->getType())->getNumElements() == 2 &&
           "Unexpected operand size");

    // See if we're dealing with constant values.
    Constant *C1 = dyn_cast<Constant>(Op1);
    ConstantInt *CI11 =
        C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)1))
           : nullptr;

    // Attempt to simplify to a constant, shuffle vector or INSERTQI call.
    if (CI11) {
      const APInt &V11 = CI11->getValue();
      APInt Len = V11.zextOrTrunc(6);
      APInt Idx = V11.lshr(8).zextOrTrunc(6);
      if (Value *V = simplifyX86insertq(*II, Op0, Op1, Len, Idx, Builder))
        return replaceInstUsesWith(*II, V);
    }

    // INSERTQ only uses the lowest 64-bits of the first 128-bit vector
    // operand.
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth, 1))
      return replaceOperand(*II, 0, V);
    break;
  }

  case Intrinsic::x86_sse4a_insertqi: {
    // INSERTQI: Extract lowest Length bits from lower half of second source and
    // insert over first source starting at Index bit. The upper 64-bits are
    // undefined.
    Value *Op0 = II->getArgOperand(0);
    Value *Op1 = II->getArgOperand(1);
    unsigned VWidth0 = cast<VectorType>(Op0->getType())->getNumElements();
    unsigned VWidth1 = cast<VectorType>(Op1->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 &&
           Op1->getType()->getPrimitiveSizeInBits() == 128 && VWidth0 == 2 &&
           VWidth1 == 2 && "Unexpected operand sizes");

    // See if we're dealing with constant values.
    ConstantInt *CILength = dyn_cast<ConstantInt>(II->getArgOperand(2));
    ConstantInt *CIIndex = dyn_cast<ConstantInt>(II->getArgOperand(3));

    // Attempt to simplify to a constant or shuffle vector.
    if (CILength && CIIndex) {
      APInt Len = CILength->getValue().zextOrTrunc(6);
      APInt Idx = CIIndex->getValue().zextOrTrunc(6);
      if (Value *V = simplifyX86insertq(*II, Op0, Op1, Len, Idx, Builder))
        return replaceInstUsesWith(*II, V);
    }

    // INSERTQI only uses the lowest 64-bits of the first two 128-bit vector
    // operands.
    bool MadeChange = false;
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth0, 1)) {
      replaceOperand(*II, 0, V);
      MadeChange = true;
    }
    if (Value *V = SimplifyDemandedVectorEltsLow(Op1, VWidth1, 1)) {
      replaceOperand(*II, 1, V);
      MadeChange = true;
    }
    if (MadeChange)
      return II;
    break;
  }

  case Intrinsic::x86_sse41_pblendvb:
  case Intrinsic::x86_sse41_blendvps:
  case Intrinsic::x86_sse41_blendvpd:
  case Intrinsic::x86_avx_blendv_ps_256:
  case Intrinsic::x86_avx_blendv_pd_256:
  case Intrinsic::x86_avx2_pblendvb: {
    // fold (blend A, A, Mask) -> A
    Value *Op0 = II->getArgOperand(0);
    Value *Op1 = II->getArgOperand(1);
    Value *Mask = II->getArgOperand(2);
    if (Op0 == Op1)
      return replaceInstUsesWith(CI, Op0);

    // Zero Mask - select 1st argument.
    if (isa<ConstantAggregateZero>(Mask))
      return replaceInstUsesWith(CI, Op0);

    // Constant Mask - select 1st/2nd argument lane based on top bit of mask.
    if (auto *ConstantMask = dyn_cast<ConstantDataVector>(Mask)) {
      Constant *NewSelector = getNegativeIsTrueBoolVec(ConstantMask);
      return SelectInst::Create(NewSelector, Op1, Op0, "blendv");
    }

    // Convert to a vector select if we can bypass casts and find a boolean
    // vector condition value.
    Value *BoolVec;
    Mask = peekThroughBitcast(Mask);
    if (match(Mask, m_SExt(m_Value(BoolVec))) &&
        BoolVec->getType()->isVectorTy() &&
        BoolVec->getType()->getScalarSizeInBits() == 1) {
      assert(Mask->getType()->getPrimitiveSizeInBits() ==
             II->getType()->getPrimitiveSizeInBits() &&
             "Not expecting mask and operands with different sizes");

      unsigned NumMaskElts =
          cast<VectorType>(Mask->getType())->getNumElements();
      unsigned NumOperandElts =
          cast<VectorType>(II->getType())->getNumElements();
      if (NumMaskElts == NumOperandElts)
        return SelectInst::Create(BoolVec, Op1, Op0);

      // If the mask has less elements than the operands, each mask bit maps to
      // multiple elements of the operands. Bitcast back and forth.
      if (NumMaskElts < NumOperandElts) {
        Value *CastOp0 = Builder.CreateBitCast(Op0, Mask->getType());
        Value *CastOp1 = Builder.CreateBitCast(Op1, Mask->getType());
        Value *Sel = Builder.CreateSelect(BoolVec, CastOp1, CastOp0);
        return new BitCastInst(Sel, II->getType());
      }
    }

    break;
  }

  case Intrinsic::x86_ssse3_pshuf_b_128:
  case Intrinsic::x86_avx2_pshuf_b:
  case Intrinsic::x86_avx512_pshuf_b_512:
    if (Value *V = simplifyX86pshufb(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_avx_vpermilvar_ps:
  case Intrinsic::x86_avx_vpermilvar_ps_256:
  case Intrinsic::x86_avx512_vpermilvar_ps_512:
  case Intrinsic::x86_avx_vpermilvar_pd:
  case Intrinsic::x86_avx_vpermilvar_pd_256:
  case Intrinsic::x86_avx512_vpermilvar_pd_512:
    if (Value *V = simplifyX86vpermilvar(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_avx2_permd:
  case Intrinsic::x86_avx2_permps:
  case Intrinsic::x86_avx512_permvar_df_256:
  case Intrinsic::x86_avx512_permvar_df_512:
  case Intrinsic::x86_avx512_permvar_di_256:
  case Intrinsic::x86_avx512_permvar_di_512:
  case Intrinsic::x86_avx512_permvar_hi_128:
  case Intrinsic::x86_avx512_permvar_hi_256:
  case Intrinsic::x86_avx512_permvar_hi_512:
  case Intrinsic::x86_avx512_permvar_qi_128:
  case Intrinsic::x86_avx512_permvar_qi_256:
  case Intrinsic::x86_avx512_permvar_qi_512:
  case Intrinsic::x86_avx512_permvar_sf_512:
  case Intrinsic::x86_avx512_permvar_si_512:
    if (Value *V = simplifyX86vpermv(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::x86_avx_maskload_ps:
  case Intrinsic::x86_avx_maskload_pd:
  case Intrinsic::x86_avx_maskload_ps_256:
  case Intrinsic::x86_avx_maskload_pd_256:
  case Intrinsic::x86_avx2_maskload_d:
  case Intrinsic::x86_avx2_maskload_q:
  case Intrinsic::x86_avx2_maskload_d_256:
  case Intrinsic::x86_avx2_maskload_q_256:
    if (Instruction *I = simplifyX86MaskedLoad(*II, *this))
      return I;
    break;

  case Intrinsic::x86_sse2_maskmov_dqu:
  case Intrinsic::x86_avx_maskstore_ps:
  case Intrinsic::x86_avx_maskstore_pd:
  case Intrinsic::x86_avx_maskstore_ps_256:
  case Intrinsic::x86_avx_maskstore_pd_256:
  case Intrinsic::x86_avx2_maskstore_d:
  case Intrinsic::x86_avx2_maskstore_q:
  case Intrinsic::x86_avx2_maskstore_d_256:
  case Intrinsic::x86_avx2_maskstore_q_256:
    if (simplifyX86MaskedStore(*II, *this))
      return nullptr;
    break;

  case Intrinsic::x86_addcarry_32:
  case Intrinsic::x86_addcarry_64:
    if (Value *V = simplifyX86addcarry(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::ppc_altivec_vperm:
    // Turn vperm(V1,V2,mask) -> shuffle(V1,V2,mask) if mask is a constant.
    // Note that ppc_altivec_vperm has a big-endian bias, so when creating
    // a vectorshuffle for little endian, we must undo the transformation
    // performed on vec_perm in altivec.h.  That is, we must complement
    // the permutation mask with respect to 31 and reverse the order of
    // V1 and V2.
    if (Constant *Mask = dyn_cast<Constant>(II->getArgOperand(2))) {
      assert(cast<VectorType>(Mask->getType())->getNumElements() == 16 &&
             "Bad type for intrinsic!");

      // Check that all of the elements are integer constants or undefs.
      bool AllEltsOk = true;
      for (unsigned i = 0; i != 16; ++i) {
        Constant *Elt = Mask->getAggregateElement(i);
        if (!Elt || !(isa<ConstantInt>(Elt) || isa<UndefValue>(Elt))) {
          AllEltsOk = false;
          break;
        }
      }

      if (AllEltsOk) {
        // Cast the input vectors to byte vectors.
        Value *Op0 = Builder.CreateBitCast(II->getArgOperand(0),
                                           Mask->getType());
        Value *Op1 = Builder.CreateBitCast(II->getArgOperand(1),
                                           Mask->getType());
        Value *Result = UndefValue::get(Op0->getType());

        // Only extract each element once.
        Value *ExtractedElts[32];
        memset(ExtractedElts, 0, sizeof(ExtractedElts));

        for (unsigned i = 0; i != 16; ++i) {
          if (isa<UndefValue>(Mask->getAggregateElement(i)))
            continue;
          unsigned Idx =
            cast<ConstantInt>(Mask->getAggregateElement(i))->getZExtValue();
          Idx &= 31;  // Match the hardware behavior.
          if (DL.isLittleEndian())
            Idx = 31 - Idx;

          if (!ExtractedElts[Idx]) {
            Value *Op0ToUse = (DL.isLittleEndian()) ? Op1 : Op0;
            Value *Op1ToUse = (DL.isLittleEndian()) ? Op0 : Op1;
            ExtractedElts[Idx] =
              Builder.CreateExtractElement(Idx < 16 ? Op0ToUse : Op1ToUse,
                                           Builder.getInt32(Idx&15));
          }

          // Insert this value into the result vector.
          Result = Builder.CreateInsertElement(Result, ExtractedElts[Idx],
                                               Builder.getInt32(i));
        }
        return CastInst::Create(Instruction::BitCast, Result, CI.getType());
      }
    }
    break;

  case Intrinsic::arm_neon_vld1: {
    Align MemAlign = getKnownAlignment(II->getArgOperand(0), DL, II, &AC, &DT);
    if (Value *V = simplifyNeonVld1(*II, MemAlign.value(), Builder))
      return replaceInstUsesWith(*II, V);
    break;
  }

  case Intrinsic::arm_neon_vld2:
  case Intrinsic::arm_neon_vld3:
  case Intrinsic::arm_neon_vld4:
  case Intrinsic::arm_neon_vld2lane:
  case Intrinsic::arm_neon_vld3lane:
  case Intrinsic::arm_neon_vld4lane:
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    Align MemAlign = getKnownAlignment(II->getArgOperand(0), DL, II, &AC, &DT);
    unsigned AlignArg = II->getNumArgOperands() - 1;
    ConstantInt *IntrAlign = dyn_cast<ConstantInt>(II->getArgOperand(AlignArg));
    if (IntrAlign && IntrAlign->getZExtValue() < MemAlign.value())
      return replaceOperand(*II, AlignArg,
                            ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                             MemAlign.value(), false));
    break;
  }

  case Intrinsic::arm_neon_vtbl1:
  case Intrinsic::aarch64_neon_tbl1:
    if (Value *V = simplifyNeonTbl1(*II, Builder))
      return replaceInstUsesWith(*II, V);
    break;

  case Intrinsic::arm_neon_vmulls:
  case Intrinsic::arm_neon_vmullu:
  case Intrinsic::aarch64_neon_smull:
  case Intrinsic::aarch64_neon_umull: {
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);

    // Handle mul by zero first:
    if (isa<ConstantAggregateZero>(Arg0) || isa<ConstantAggregateZero>(Arg1)) {
      return replaceInstUsesWith(CI, ConstantAggregateZero::get(II->getType()));
    }

    // Check for constant LHS & RHS - in this case we just simplify.
    bool Zext = (IID == Intrinsic::arm_neon_vmullu ||
                 IID == Intrinsic::aarch64_neon_umull);
    VectorType *NewVT = cast<VectorType>(II->getType());
    if (Constant *CV0 = dyn_cast<Constant>(Arg0)) {
      if (Constant *CV1 = dyn_cast<Constant>(Arg1)) {
        CV0 = ConstantExpr::getIntegerCast(CV0, NewVT, /*isSigned=*/!Zext);
        CV1 = ConstantExpr::getIntegerCast(CV1, NewVT, /*isSigned=*/!Zext);

        return replaceInstUsesWith(CI, ConstantExpr::getMul(CV0, CV1));
      }

      // Couldn't simplify - canonicalize constant to the RHS.
      std::swap(Arg0, Arg1);
    }

    // Handle mul by one:
    if (Constant *CV1 = dyn_cast<Constant>(Arg1))
      if (ConstantInt *Splat =
              dyn_cast_or_null<ConstantInt>(CV1->getSplatValue()))
        if (Splat->isOne())
          return CastInst::CreateIntegerCast(Arg0, II->getType(),
                                             /*isSigned=*/!Zext);

    break;
  }
  case Intrinsic::arm_neon_aesd:
  case Intrinsic::arm_neon_aese:
  case Intrinsic::aarch64_crypto_aesd:
  case Intrinsic::aarch64_crypto_aese: {
    Value *DataArg = II->getArgOperand(0);
    Value *KeyArg  = II->getArgOperand(1);

    // Try to use the builtin XOR in AESE and AESD to eliminate a prior XOR
    Value *Data, *Key;
    if (match(KeyArg, m_ZeroInt()) &&
        match(DataArg, m_Xor(m_Value(Data), m_Value(Key)))) {
      replaceOperand(*II, 0, Data);
      replaceOperand(*II, 1, Key);
      return II;
    }
    break;
  }
  case Intrinsic::arm_mve_pred_i2v: {
    Value *Arg = II->getArgOperand(0);
    Value *ArgArg;
    if (match(Arg, m_Intrinsic<Intrinsic::arm_mve_pred_v2i>(m_Value(ArgArg))) &&
        II->getType() == ArgArg->getType())
      return replaceInstUsesWith(*II, ArgArg);
    Constant *XorMask;
    if (match(Arg,
              m_Xor(m_Intrinsic<Intrinsic::arm_mve_pred_v2i>(m_Value(ArgArg)),
                    m_Constant(XorMask))) &&
        II->getType() == ArgArg->getType()) {
      if (auto *CI = dyn_cast<ConstantInt>(XorMask)) {
        if (CI->getValue().trunc(16).isAllOnesValue()) {
          auto TrueVector = Builder.CreateVectorSplat(
              cast<VectorType>(II->getType())->getNumElements(),
              Builder.getTrue());
          return BinaryOperator::Create(Instruction::Xor, ArgArg, TrueVector);
        }
      }
    }
    KnownBits ScalarKnown(32);
    if (SimplifyDemandedBits(II, 0, APInt::getLowBitsSet(32, 16),
                             ScalarKnown, 0))
      return II;
    break;
  }
  case Intrinsic::arm_mve_pred_v2i: {
    Value *Arg = II->getArgOperand(0);
    Value *ArgArg;
    if (match(Arg, m_Intrinsic<Intrinsic::arm_mve_pred_i2v>(m_Value(ArgArg))))
      return replaceInstUsesWith(*II, ArgArg);
    if (!II->getMetadata(LLVMContext::MD_range)) {
      Type *IntTy32 = Type::getInt32Ty(II->getContext());
      Metadata *M[] = {
        ConstantAsMetadata::get(ConstantInt::get(IntTy32, 0)),
        ConstantAsMetadata::get(ConstantInt::get(IntTy32, 0xFFFF))
      };
      II->setMetadata(LLVMContext::MD_range, MDNode::get(II->getContext(), M));
      return II;
    }
    break;
  }
  case Intrinsic::arm_mve_vadc:
  case Intrinsic::arm_mve_vadc_predicated: {
    unsigned CarryOp =
        (II->getIntrinsicID() == Intrinsic::arm_mve_vadc_predicated) ? 3 : 2;
    assert(II->getArgOperand(CarryOp)->getType()->getScalarSizeInBits() == 32 &&
           "Bad type for intrinsic!");

    KnownBits CarryKnown(32);
    if (SimplifyDemandedBits(II, CarryOp, APInt::getOneBitSet(32, 29),
                             CarryKnown))
      return II;
    break;
  }
  case Intrinsic::amdgcn_rcp: {
    Value *Src = II->getArgOperand(0);

    // TODO: Move to ConstantFolding/InstSimplify?
    if (isa<UndefValue>(Src))
      return replaceInstUsesWith(CI, Src);

    if (const ConstantFP *C = dyn_cast<ConstantFP>(Src)) {
      const APFloat &ArgVal = C->getValueAPF();
      APFloat Val(ArgVal.getSemantics(), 1);
      APFloat::opStatus Status = Val.divide(ArgVal,
                                            APFloat::rmNearestTiesToEven);
      // Only do this if it was exact and therefore not dependent on the
      // rounding mode.
      if (Status == APFloat::opOK)
        return replaceInstUsesWith(CI, ConstantFP::get(II->getContext(), Val));
    }

    break;
  }
  case Intrinsic::amdgcn_rsq: {
    Value *Src = II->getArgOperand(0);

    // TODO: Move to ConstantFolding/InstSimplify?
    if (isa<UndefValue>(Src))
      return replaceInstUsesWith(CI, Src);
    break;
  }
  case Intrinsic::amdgcn_frexp_mant:
  case Intrinsic::amdgcn_frexp_exp: {
    Value *Src = II->getArgOperand(0);
    if (const ConstantFP *C = dyn_cast<ConstantFP>(Src)) {
      int Exp;
      APFloat Significand = frexp(C->getValueAPF(), Exp,
                                  APFloat::rmNearestTiesToEven);

      if (IID == Intrinsic::amdgcn_frexp_mant) {
        return replaceInstUsesWith(CI, ConstantFP::get(II->getContext(),
                                                       Significand));
      }

      // Match instruction special case behavior.
      if (Exp == APFloat::IEK_NaN || Exp == APFloat::IEK_Inf)
        Exp = 0;

      return replaceInstUsesWith(CI, ConstantInt::get(II->getType(), Exp));
    }

    if (isa<UndefValue>(Src))
      return replaceInstUsesWith(CI, UndefValue::get(II->getType()));

    break;
  }
  case Intrinsic::amdgcn_class: {
    enum  {
      S_NAN = 1 << 0,        // Signaling NaN
      Q_NAN = 1 << 1,        // Quiet NaN
      N_INFINITY = 1 << 2,   // Negative infinity
      N_NORMAL = 1 << 3,     // Negative normal
      N_SUBNORMAL = 1 << 4,  // Negative subnormal
      N_ZERO = 1 << 5,       // Negative zero
      P_ZERO = 1 << 6,       // Positive zero
      P_SUBNORMAL = 1 << 7,  // Positive subnormal
      P_NORMAL = 1 << 8,     // Positive normal
      P_INFINITY = 1 << 9    // Positive infinity
    };

    const uint32_t FullMask = S_NAN | Q_NAN | N_INFINITY | N_NORMAL |
      N_SUBNORMAL | N_ZERO | P_ZERO | P_SUBNORMAL | P_NORMAL | P_INFINITY;

    Value *Src0 = II->getArgOperand(0);
    Value *Src1 = II->getArgOperand(1);
    const ConstantInt *CMask = dyn_cast<ConstantInt>(Src1);
    if (!CMask) {
      if (isa<UndefValue>(Src0))
        return replaceInstUsesWith(*II, UndefValue::get(II->getType()));

      if (isa<UndefValue>(Src1))
        return replaceInstUsesWith(*II, ConstantInt::get(II->getType(), false));
      break;
    }

    uint32_t Mask = CMask->getZExtValue();

    // If all tests are made, it doesn't matter what the value is.
    if ((Mask & FullMask) == FullMask)
      return replaceInstUsesWith(*II, ConstantInt::get(II->getType(), true));

    if ((Mask & FullMask) == 0)
      return replaceInstUsesWith(*II, ConstantInt::get(II->getType(), false));

    if (Mask == (S_NAN | Q_NAN)) {
      // Equivalent of isnan. Replace with standard fcmp.
      Value *FCmp = Builder.CreateFCmpUNO(Src0, Src0);
      FCmp->takeName(II);
      return replaceInstUsesWith(*II, FCmp);
    }

    if (Mask == (N_ZERO | P_ZERO)) {
      // Equivalent of == 0.
      Value *FCmp = Builder.CreateFCmpOEQ(
        Src0, ConstantFP::get(Src0->getType(), 0.0));

      FCmp->takeName(II);
      return replaceInstUsesWith(*II, FCmp);
    }

    // fp_class (nnan x), qnan|snan|other -> fp_class (nnan x), other
    if (((Mask & S_NAN) || (Mask & Q_NAN)) && isKnownNeverNaN(Src0, &TLI))
      return replaceOperand(*II, 1, ConstantInt::get(Src1->getType(),
                                                     Mask & ~(S_NAN | Q_NAN)));

    const ConstantFP *CVal = dyn_cast<ConstantFP>(Src0);
    if (!CVal) {
      if (isa<UndefValue>(Src0))
        return replaceInstUsesWith(*II, UndefValue::get(II->getType()));

      // Clamp mask to used bits
      if ((Mask & FullMask) != Mask) {
        CallInst *NewCall = Builder.CreateCall(II->getCalledFunction(),
          { Src0, ConstantInt::get(Src1->getType(), Mask & FullMask) }
        );

        NewCall->takeName(II);
        return replaceInstUsesWith(*II, NewCall);
      }

      break;
    }

    const APFloat &Val = CVal->getValueAPF();

    bool Result =
      ((Mask & S_NAN) && Val.isNaN() && Val.isSignaling()) ||
      ((Mask & Q_NAN) && Val.isNaN() && !Val.isSignaling()) ||
      ((Mask & N_INFINITY) && Val.isInfinity() && Val.isNegative()) ||
      ((Mask & N_NORMAL) && Val.isNormal() && Val.isNegative()) ||
      ((Mask & N_SUBNORMAL) && Val.isDenormal() && Val.isNegative()) ||
      ((Mask & N_ZERO) && Val.isZero() && Val.isNegative()) ||
      ((Mask & P_ZERO) && Val.isZero() && !Val.isNegative()) ||
      ((Mask & P_SUBNORMAL) && Val.isDenormal() && !Val.isNegative()) ||
      ((Mask & P_NORMAL) && Val.isNormal() && !Val.isNegative()) ||
      ((Mask & P_INFINITY) && Val.isInfinity() && !Val.isNegative());

    return replaceInstUsesWith(*II, ConstantInt::get(II->getType(), Result));
  }
  case Intrinsic::amdgcn_cvt_pkrtz: {
    Value *Src0 = II->getArgOperand(0);
    Value *Src1 = II->getArgOperand(1);
    if (const ConstantFP *C0 = dyn_cast<ConstantFP>(Src0)) {
      if (const ConstantFP *C1 = dyn_cast<ConstantFP>(Src1)) {
        const fltSemantics &HalfSem
          = II->getType()->getScalarType()->getFltSemantics();
        bool LosesInfo;
        APFloat Val0 = C0->getValueAPF();
        APFloat Val1 = C1->getValueAPF();
        Val0.convert(HalfSem, APFloat::rmTowardZero, &LosesInfo);
        Val1.convert(HalfSem, APFloat::rmTowardZero, &LosesInfo);

        Constant *Folded = ConstantVector::get({
            ConstantFP::get(II->getContext(), Val0),
            ConstantFP::get(II->getContext(), Val1) });
        return replaceInstUsesWith(*II, Folded);
      }
    }

    if (isa<UndefValue>(Src0) && isa<UndefValue>(Src1))
      return replaceInstUsesWith(*II, UndefValue::get(II->getType()));

    break;
  }
  case Intrinsic::amdgcn_cvt_pknorm_i16:
  case Intrinsic::amdgcn_cvt_pknorm_u16:
  case Intrinsic::amdgcn_cvt_pk_i16:
  case Intrinsic::amdgcn_cvt_pk_u16: {
    Value *Src0 = II->getArgOperand(0);
    Value *Src1 = II->getArgOperand(1);

    if (isa<UndefValue>(Src0) && isa<UndefValue>(Src1))
      return replaceInstUsesWith(*II, UndefValue::get(II->getType()));

    break;
  }
  case Intrinsic::amdgcn_ubfe:
  case Intrinsic::amdgcn_sbfe: {
    // Decompose simple cases into standard shifts.
    Value *Src = II->getArgOperand(0);
    if (isa<UndefValue>(Src))
      return replaceInstUsesWith(*II, Src);

    unsigned Width;
    Type *Ty = II->getType();
    unsigned IntSize = Ty->getIntegerBitWidth();

    ConstantInt *CWidth = dyn_cast<ConstantInt>(II->getArgOperand(2));
    if (CWidth) {
      Width = CWidth->getZExtValue();
      if ((Width & (IntSize - 1)) == 0)
        return replaceInstUsesWith(*II, ConstantInt::getNullValue(Ty));

      // Hardware ignores high bits, so remove those.
      if (Width >= IntSize)
        return replaceOperand(*II, 2, ConstantInt::get(CWidth->getType(),
                                                       Width & (IntSize - 1)));
    }

    unsigned Offset;
    ConstantInt *COffset = dyn_cast<ConstantInt>(II->getArgOperand(1));
    if (COffset) {
      Offset = COffset->getZExtValue();
      if (Offset >= IntSize)
        return replaceOperand(*II, 1, ConstantInt::get(COffset->getType(),
                                                       Offset & (IntSize - 1)));
    }

    bool Signed = IID == Intrinsic::amdgcn_sbfe;

    if (!CWidth || !COffset)
      break;

    // The case of Width == 0 is handled above, which makes this tranformation
    // safe.  If Width == 0, then the ashr and lshr instructions become poison
    // value since the shift amount would be equal to the bit size.
    assert(Width != 0);

    // TODO: This allows folding to undef when the hardware has specific
    // behavior?
    if (Offset + Width < IntSize) {
      Value *Shl = Builder.CreateShl(Src, IntSize - Offset - Width);
      Value *RightShift = Signed ? Builder.CreateAShr(Shl, IntSize - Width)
                                 : Builder.CreateLShr(Shl, IntSize - Width);
      RightShift->takeName(II);
      return replaceInstUsesWith(*II, RightShift);
    }

    Value *RightShift = Signed ? Builder.CreateAShr(Src, Offset)
                               : Builder.CreateLShr(Src, Offset);

    RightShift->takeName(II);
    return replaceInstUsesWith(*II, RightShift);
  }
  case Intrinsic::amdgcn_exp:
  case Intrinsic::amdgcn_exp_compr: {
    ConstantInt *En = cast<ConstantInt>(II->getArgOperand(1));
    unsigned EnBits = En->getZExtValue();
    if (EnBits == 0xf)
      break; // All inputs enabled.

    bool IsCompr = IID == Intrinsic::amdgcn_exp_compr;
    bool Changed = false;
    for (int I = 0; I < (IsCompr ? 2 : 4); ++I) {
      if ((!IsCompr && (EnBits & (1 << I)) == 0) ||
          (IsCompr && ((EnBits & (0x3 << (2 * I))) == 0))) {
        Value *Src = II->getArgOperand(I + 2);
        if (!isa<UndefValue>(Src)) {
          replaceOperand(*II, I + 2, UndefValue::get(Src->getType()));
          Changed = true;
        }
      }
    }

    if (Changed)
      return II;

    break;
  }
  case Intrinsic::amdgcn_fmed3: {
    // Note this does not preserve proper sNaN behavior if IEEE-mode is enabled
    // for the shader.

    Value *Src0 = II->getArgOperand(0);
    Value *Src1 = II->getArgOperand(1);
    Value *Src2 = II->getArgOperand(2);

    // Checking for NaN before canonicalization provides better fidelity when
    // mapping other operations onto fmed3 since the order of operands is
    // unchanged.
    CallInst *NewCall = nullptr;
    if (match(Src0, m_NaN()) || isa<UndefValue>(Src0)) {
      NewCall = Builder.CreateMinNum(Src1, Src2);
    } else if (match(Src1, m_NaN()) || isa<UndefValue>(Src1)) {
      NewCall = Builder.CreateMinNum(Src0, Src2);
    } else if (match(Src2, m_NaN()) || isa<UndefValue>(Src2)) {
      NewCall = Builder.CreateMaxNum(Src0, Src1);
    }

    if (NewCall) {
      NewCall->copyFastMathFlags(II);
      NewCall->takeName(II);
      return replaceInstUsesWith(*II, NewCall);
    }

    bool Swap = false;
    // Canonicalize constants to RHS operands.
    //
    // fmed3(c0, x, c1) -> fmed3(x, c0, c1)
    if (isa<Constant>(Src0) && !isa<Constant>(Src1)) {
      std::swap(Src0, Src1);
      Swap = true;
    }

    if (isa<Constant>(Src1) && !isa<Constant>(Src2)) {
      std::swap(Src1, Src2);
      Swap = true;
    }

    if (isa<Constant>(Src0) && !isa<Constant>(Src1)) {
      std::swap(Src0, Src1);
      Swap = true;
    }

    if (Swap) {
      II->setArgOperand(0, Src0);
      II->setArgOperand(1, Src1);
      II->setArgOperand(2, Src2);
      return II;
    }

    if (const ConstantFP *C0 = dyn_cast<ConstantFP>(Src0)) {
      if (const ConstantFP *C1 = dyn_cast<ConstantFP>(Src1)) {
        if (const ConstantFP *C2 = dyn_cast<ConstantFP>(Src2)) {
          APFloat Result = fmed3AMDGCN(C0->getValueAPF(), C1->getValueAPF(),
                                       C2->getValueAPF());
          return replaceInstUsesWith(*II,
            ConstantFP::get(Builder.getContext(), Result));
        }
      }
    }

    break;
  }
  case Intrinsic::amdgcn_icmp:
  case Intrinsic::amdgcn_fcmp: {
    const ConstantInt *CC = cast<ConstantInt>(II->getArgOperand(2));
    // Guard against invalid arguments.
    int64_t CCVal = CC->getZExtValue();
    bool IsInteger = IID == Intrinsic::amdgcn_icmp;
    if ((IsInteger && (CCVal < CmpInst::FIRST_ICMP_PREDICATE ||
                       CCVal > CmpInst::LAST_ICMP_PREDICATE)) ||
        (!IsInteger && (CCVal < CmpInst::FIRST_FCMP_PREDICATE ||
                        CCVal > CmpInst::LAST_FCMP_PREDICATE)))
      break;

    Value *Src0 = II->getArgOperand(0);
    Value *Src1 = II->getArgOperand(1);

    if (auto *CSrc0 = dyn_cast<Constant>(Src0)) {
      if (auto *CSrc1 = dyn_cast<Constant>(Src1)) {
        Constant *CCmp = ConstantExpr::getCompare(CCVal, CSrc0, CSrc1);
        if (CCmp->isNullValue()) {
          return replaceInstUsesWith(
              *II, ConstantExpr::getSExt(CCmp, II->getType()));
        }

        // The result of V_ICMP/V_FCMP assembly instructions (which this
        // intrinsic exposes) is one bit per thread, masked with the EXEC
        // register (which contains the bitmask of live threads). So a
        // comparison that always returns true is the same as a read of the
        // EXEC register.
        Function *NewF = Intrinsic::getDeclaration(
            II->getModule(), Intrinsic::read_register, II->getType());
        Metadata *MDArgs[] = {MDString::get(II->getContext(), "exec")};
        MDNode *MD = MDNode::get(II->getContext(), MDArgs);
        Value *Args[] = {MetadataAsValue::get(II->getContext(), MD)};
        CallInst *NewCall = Builder.CreateCall(NewF, Args);
        NewCall->addAttribute(AttributeList::FunctionIndex,
                              Attribute::Convergent);
        NewCall->takeName(II);
        return replaceInstUsesWith(*II, NewCall);
      }

      // Canonicalize constants to RHS.
      CmpInst::Predicate SwapPred
        = CmpInst::getSwappedPredicate(static_cast<CmpInst::Predicate>(CCVal));
      II->setArgOperand(0, Src1);
      II->setArgOperand(1, Src0);
      II->setArgOperand(2, ConstantInt::get(CC->getType(),
                                            static_cast<int>(SwapPred)));
      return II;
    }

    if (CCVal != CmpInst::ICMP_EQ && CCVal != CmpInst::ICMP_NE)
      break;

    // Canonicalize compare eq with true value to compare != 0
    // llvm.amdgcn.icmp(zext (i1 x), 1, eq)
    //   -> llvm.amdgcn.icmp(zext (i1 x), 0, ne)
    // llvm.amdgcn.icmp(sext (i1 x), -1, eq)
    //   -> llvm.amdgcn.icmp(sext (i1 x), 0, ne)
    Value *ExtSrc;
    if (CCVal == CmpInst::ICMP_EQ &&
        ((match(Src1, m_One()) && match(Src0, m_ZExt(m_Value(ExtSrc)))) ||
         (match(Src1, m_AllOnes()) && match(Src0, m_SExt(m_Value(ExtSrc))))) &&
        ExtSrc->getType()->isIntegerTy(1)) {
      replaceOperand(*II, 1, ConstantInt::getNullValue(Src1->getType()));
      replaceOperand(*II, 2, ConstantInt::get(CC->getType(), CmpInst::ICMP_NE));
      return II;
    }

    CmpInst::Predicate SrcPred;
    Value *SrcLHS;
    Value *SrcRHS;

    // Fold compare eq/ne with 0 from a compare result as the predicate to the
    // intrinsic. The typical use is a wave vote function in the library, which
    // will be fed from a user code condition compared with 0. Fold in the
    // redundant compare.

    // llvm.amdgcn.icmp([sz]ext ([if]cmp pred a, b), 0, ne)
    //   -> llvm.amdgcn.[if]cmp(a, b, pred)
    //
    // llvm.amdgcn.icmp([sz]ext ([if]cmp pred a, b), 0, eq)
    //   -> llvm.amdgcn.[if]cmp(a, b, inv pred)
    if (match(Src1, m_Zero()) &&
        match(Src0,
              m_ZExtOrSExt(m_Cmp(SrcPred, m_Value(SrcLHS), m_Value(SrcRHS))))) {
      if (CCVal == CmpInst::ICMP_EQ)
        SrcPred = CmpInst::getInversePredicate(SrcPred);

      Intrinsic::ID NewIID = CmpInst::isFPPredicate(SrcPred) ?
        Intrinsic::amdgcn_fcmp : Intrinsic::amdgcn_icmp;

      Type *Ty = SrcLHS->getType();
      if (auto *CmpType = dyn_cast<IntegerType>(Ty)) {
        // Promote to next legal integer type.
        unsigned Width = CmpType->getBitWidth();
        unsigned NewWidth = Width;

        // Don't do anything for i1 comparisons.
        if (Width == 1)
          break;

        if (Width <= 16)
          NewWidth = 16;
        else if (Width <= 32)
          NewWidth = 32;
        else if (Width <= 64)
          NewWidth = 64;
        else if (Width > 64)
          break; // Can't handle this.

        if (Width != NewWidth) {
          IntegerType *CmpTy = Builder.getIntNTy(NewWidth);
          if (CmpInst::isSigned(SrcPred)) {
            SrcLHS = Builder.CreateSExt(SrcLHS, CmpTy);
            SrcRHS = Builder.CreateSExt(SrcRHS, CmpTy);
          } else {
            SrcLHS = Builder.CreateZExt(SrcLHS, CmpTy);
            SrcRHS = Builder.CreateZExt(SrcRHS, CmpTy);
          }
        }
      } else if (!Ty->isFloatTy() && !Ty->isDoubleTy() && !Ty->isHalfTy())
        break;

      Function *NewF =
          Intrinsic::getDeclaration(II->getModule(), NewIID,
                                    { II->getType(),
                                      SrcLHS->getType() });
      Value *Args[] = { SrcLHS, SrcRHS,
                        ConstantInt::get(CC->getType(), SrcPred) };
      CallInst *NewCall = Builder.CreateCall(NewF, Args);
      NewCall->takeName(II);
      return replaceInstUsesWith(*II, NewCall);
    }

    break;
  }
  case Intrinsic::amdgcn_ballot: {
    if (auto *Src = dyn_cast<ConstantInt>(II->getArgOperand(0))) {
      if (Src->isZero()) {
        // amdgcn.ballot(i1 0) is zero.
        return replaceInstUsesWith(*II, Constant::getNullValue(II->getType()));
      }

      if (Src->isOne()) {
        // amdgcn.ballot(i1 1) is exec.
        const char *RegName = "exec";
        if (II->getType()->isIntegerTy(32))
          RegName = "exec_lo";
        else if (!II->getType()->isIntegerTy(64))
          break;

        Function *NewF = Intrinsic::getDeclaration(
            II->getModule(), Intrinsic::read_register, II->getType());
        Metadata *MDArgs[] = {MDString::get(II->getContext(), RegName)};
        MDNode *MD = MDNode::get(II->getContext(), MDArgs);
        Value *Args[] = {MetadataAsValue::get(II->getContext(), MD)};
        CallInst *NewCall = Builder.CreateCall(NewF, Args);
        NewCall->addAttribute(AttributeList::FunctionIndex,
                              Attribute::Convergent);
        NewCall->takeName(II);
        return replaceInstUsesWith(*II, NewCall);
      }
    }
    break;
  }
  case Intrinsic::amdgcn_wqm_vote: {
    // wqm_vote is identity when the argument is constant.
    if (!isa<Constant>(II->getArgOperand(0)))
      break;

    return replaceInstUsesWith(*II, II->getArgOperand(0));
  }
  case Intrinsic::amdgcn_kill: {
    const ConstantInt *C = dyn_cast<ConstantInt>(II->getArgOperand(0));
    if (!C || !C->getZExtValue())
      break;

    // amdgcn.kill(i1 1) is a no-op
    return eraseInstFromFunction(CI);
  }
  case Intrinsic::amdgcn_update_dpp: {
    Value *Old = II->getArgOperand(0);

    auto BC = cast<ConstantInt>(II->getArgOperand(5));
    auto RM = cast<ConstantInt>(II->getArgOperand(3));
    auto BM = cast<ConstantInt>(II->getArgOperand(4));
    if (BC->isZeroValue() ||
        RM->getZExtValue() != 0xF ||
        BM->getZExtValue() != 0xF ||
        isa<UndefValue>(Old))
      break;

    // If bound_ctrl = 1, row mask = bank mask = 0xf we can omit old value.
    return replaceOperand(*II, 0, UndefValue::get(Old->getType()));
  }
  case Intrinsic::amdgcn_permlane16:
  case Intrinsic::amdgcn_permlanex16: {
    // Discard vdst_in if it's not going to be read.
    Value *VDstIn = II->getArgOperand(0);
   if (isa<UndefValue>(VDstIn))
     break;

    ConstantInt *FetchInvalid = cast<ConstantInt>(II->getArgOperand(4));
    ConstantInt *BoundCtrl = cast<ConstantInt>(II->getArgOperand(5));
    if (!FetchInvalid->getZExtValue() && !BoundCtrl->getZExtValue())
      break;

    return replaceOperand(*II, 0, UndefValue::get(VDstIn->getType()));
  }
  case Intrinsic::amdgcn_readfirstlane:
  case Intrinsic::amdgcn_readlane: {
    // A constant value is trivially uniform.
    if (Constant *C = dyn_cast<Constant>(II->getArgOperand(0)))
      return replaceInstUsesWith(*II, C);

    // The rest of these may not be safe if the exec may not be the same between
    // the def and use.
    Value *Src = II->getArgOperand(0);
    Instruction *SrcInst = dyn_cast<Instruction>(Src);
    if (SrcInst && SrcInst->getParent() != II->getParent())
      break;

    // readfirstlane (readfirstlane x) -> readfirstlane x
    // readlane (readfirstlane x), y -> readfirstlane x
    if (match(Src, m_Intrinsic<Intrinsic::amdgcn_readfirstlane>()))
      return replaceInstUsesWith(*II, Src);

    if (IID == Intrinsic::amdgcn_readfirstlane) {
      // readfirstlane (readlane x, y) -> readlane x, y
      if (match(Src, m_Intrinsic<Intrinsic::amdgcn_readlane>()))
        return replaceInstUsesWith(*II, Src);
    } else {
      // readlane (readlane x, y), y -> readlane x, y
      if (match(Src, m_Intrinsic<Intrinsic::amdgcn_readlane>(
                  m_Value(), m_Specific(II->getArgOperand(1)))))
        return replaceInstUsesWith(*II, Src);
    }

    break;
  }
  case Intrinsic::hexagon_V6_vandvrt:
  case Intrinsic::hexagon_V6_vandvrt_128B: {
    // Simplify Q -> V -> Q conversion.
    if (auto Op0 = dyn_cast<IntrinsicInst>(II->getArgOperand(0))) {
      Intrinsic::ID ID0 = Op0->getIntrinsicID();
      if (ID0 != Intrinsic::hexagon_V6_vandqrt &&
          ID0 != Intrinsic::hexagon_V6_vandqrt_128B)
        break;
      Value *Bytes = Op0->getArgOperand(1), *Mask = II->getArgOperand(1);
      uint64_t Bytes1 = computeKnownBits(Bytes, 0, Op0).One.getZExtValue();
      uint64_t Mask1 = computeKnownBits(Mask, 0, II).One.getZExtValue();
      // Check if every byte has common bits in Bytes and Mask.
      uint64_t C = Bytes1 & Mask1;
      if ((C & 0xFF) && (C & 0xFF00) && (C & 0xFF0000) && (C & 0xFF000000))
        return replaceInstUsesWith(*II, Op0->getArgOperand(0));
    }
    break;
  }
  case Intrinsic::stackrestore: {
    // If the save is right next to the restore, remove the restore.  This can
    // happen when variable allocas are DCE'd.
    if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getArgOperand(0))) {
      if (SS->getIntrinsicID() == Intrinsic::stacksave) {
        // Skip over debug info.
        if (SS->getNextNonDebugInstruction() == II) {
          return eraseInstFromFunction(CI);
        }
      }
    }

    // Scan down this block to see if there is another stack restore in the
    // same block without an intervening call/alloca.
    BasicBlock::iterator BI(II);
    Instruction *TI = II->getParent()->getTerminator();
    bool CannotRemove = false;
    for (++BI; &*BI != TI; ++BI) {
      if (isa<AllocaInst>(BI)) {
        CannotRemove = true;
        break;
      }
      if (CallInst *BCI = dyn_cast<CallInst>(BI)) {
        if (auto *II2 = dyn_cast<IntrinsicInst>(BCI)) {
          // If there is a stackrestore below this one, remove this one.
          if (II2->getIntrinsicID() == Intrinsic::stackrestore)
            return eraseInstFromFunction(CI);

          // Bail if we cross over an intrinsic with side effects, such as
          // llvm.stacksave, or llvm.read_register.
          if (II2->mayHaveSideEffects()) {
            CannotRemove = true;
            break;
          }
        } else {
          // If we found a non-intrinsic call, we can't remove the stack
          // restore.
          CannotRemove = true;
          break;
        }
      }
    }

    // If the stack restore is in a return, resume, or unwind block and if there
    // are no allocas or calls between the restore and the return, nuke the
    // restore.
    if (!CannotRemove && (isa<ReturnInst>(TI) || isa<ResumeInst>(TI)))
      return eraseInstFromFunction(CI);
    break;
  }
  case Intrinsic::lifetime_end:
    // Asan needs to poison memory to detect invalid access which is possible
    // even for empty lifetime range.
    if (II->getFunction()->hasFnAttribute(Attribute::SanitizeAddress) ||
        II->getFunction()->hasFnAttribute(Attribute::SanitizeMemory) ||
        II->getFunction()->hasFnAttribute(Attribute::SanitizeHWAddress))
      break;

    if (removeTriviallyEmptyRange(*II, *this, [](const IntrinsicInst &I) {
          return I.getIntrinsicID() == Intrinsic::lifetime_start;
        }))
      return nullptr;
    break;
  case Intrinsic::assume: {
    Value *IIOperand = II->getArgOperand(0);
    // Remove an assume if it is followed by an identical assume.
    // TODO: Do we need this? Unless there are conflicting assumptions, the
    // computeKnownBits(IIOperand) below here eliminates redundant assumes.
    Instruction *Next = II->getNextNonDebugInstruction();
    if (match(Next, m_Intrinsic<Intrinsic::assume>(m_Specific(IIOperand))))
      return eraseInstFromFunction(CI);

    // Canonicalize assume(a && b) -> assume(a); assume(b);
    // Note: New assumption intrinsics created here are registered by
    // the InstCombineIRInserter object.
    FunctionType *AssumeIntrinsicTy = II->getFunctionType();
    Value *AssumeIntrinsic = II->getCalledOperand();
    Value *A, *B;
    if (match(IIOperand, m_And(m_Value(A), m_Value(B)))) {
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic, A, II->getName());
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic, B, II->getName());
      return eraseInstFromFunction(*II);
    }
    // assume(!(a || b)) -> assume(!a); assume(!b);
    if (match(IIOperand, m_Not(m_Or(m_Value(A), m_Value(B))))) {
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic,
                         Builder.CreateNot(A), II->getName());
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic,
                         Builder.CreateNot(B), II->getName());
      return eraseInstFromFunction(*II);
    }

    // assume( (load addr) != null ) -> add 'nonnull' metadata to load
    // (if assume is valid at the load)
    CmpInst::Predicate Pred;
    Instruction *LHS;
    if (match(IIOperand, m_ICmp(Pred, m_Instruction(LHS), m_Zero())) &&
        Pred == ICmpInst::ICMP_NE && LHS->getOpcode() == Instruction::Load &&
        LHS->getType()->isPointerTy() &&
        isValidAssumeForContext(II, LHS, &DT)) {
      MDNode *MD = MDNode::get(II->getContext(), None);
      LHS->setMetadata(LLVMContext::MD_nonnull, MD);
      return eraseInstFromFunction(*II);

      // TODO: apply nonnull return attributes to calls and invokes
      // TODO: apply range metadata for range check patterns?
    }

    // If there is a dominating assume with the same condition as this one,
    // then this one is redundant, and should be removed.
    KnownBits Known(1);
    computeKnownBits(IIOperand, Known, 0, II);
    if (Known.isAllOnes() && isAssumeWithEmptyBundle(*II))
      return eraseInstFromFunction(*II);

    // Update the cache of affected values for this assumption (we might be
    // here because we just simplified the condition).
    AC.updateAffectedValues(II);
    break;
  }
  case Intrinsic::experimental_gc_relocate: {
    auto &GCR = *cast<GCRelocateInst>(II);

    // If we have two copies of the same pointer in the statepoint argument
    // list, canonicalize to one.  This may let us common gc.relocates.
    if (GCR.getBasePtr() == GCR.getDerivedPtr() &&
        GCR.getBasePtrIndex() != GCR.getDerivedPtrIndex()) {
      auto *OpIntTy = GCR.getOperand(2)->getType();
      return replaceOperand(*II, 2,
          ConstantInt::get(OpIntTy, GCR.getBasePtrIndex()));
    }

    // Translate facts known about a pointer before relocating into
    // facts about the relocate value, while being careful to
    // preserve relocation semantics.
    Value *DerivedPtr = GCR.getDerivedPtr();

    // Remove the relocation if unused, note that this check is required
    // to prevent the cases below from looping forever.
    if (II->use_empty())
      return eraseInstFromFunction(*II);

    // Undef is undef, even after relocation.
    // TODO: provide a hook for this in GCStrategy.  This is clearly legal for
    // most practical collectors, but there was discussion in the review thread
    // about whether it was legal for all possible collectors.
    if (isa<UndefValue>(DerivedPtr))
      // Use undef of gc_relocate's type to replace it.
      return replaceInstUsesWith(*II, UndefValue::get(II->getType()));

    if (auto *PT = dyn_cast<PointerType>(II->getType())) {
      // The relocation of null will be null for most any collector.
      // TODO: provide a hook for this in GCStrategy.  There might be some
      // weird collector this property does not hold for.
      if (isa<ConstantPointerNull>(DerivedPtr))
        // Use null-pointer of gc_relocate's type to replace it.
        return replaceInstUsesWith(*II, ConstantPointerNull::get(PT));

      // isKnownNonNull -> nonnull attribute
      if (!II->hasRetAttr(Attribute::NonNull) &&
          isKnownNonZero(DerivedPtr, DL, 0, &AC, II, &DT)) {
        II->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
        return II;
      }
    }

    // TODO: bitcast(relocate(p)) -> relocate(bitcast(p))
    // Canonicalize on the type from the uses to the defs

    // TODO: relocate((gep p, C, C2, ...)) -> gep(relocate(p), C, C2, ...)
    break;
  }

  case Intrinsic::experimental_guard: {
    // Is this guard followed by another guard?  We scan forward over a small
    // fixed window of instructions to handle common cases with conditions
    // computed between guards.
    Instruction *NextInst = II->getNextNonDebugInstruction();
    for (unsigned i = 0; i < GuardWideningWindow; i++) {
      // Note: Using context-free form to avoid compile time blow up
      if (!isSafeToSpeculativelyExecute(NextInst))
        break;
      NextInst = NextInst->getNextNonDebugInstruction();
    }
    Value *NextCond = nullptr;
    if (match(NextInst,
              m_Intrinsic<Intrinsic::experimental_guard>(m_Value(NextCond)))) {
      Value *CurrCond = II->getArgOperand(0);

      // Remove a guard that it is immediately preceded by an identical guard.
      // Otherwise canonicalize guard(a); guard(b) -> guard(a & b).
      if (CurrCond != NextCond) {
        Instruction *MoveI = II->getNextNonDebugInstruction();
        while (MoveI != NextInst) {
          auto *Temp = MoveI;
          MoveI = MoveI->getNextNonDebugInstruction();
          Temp->moveBefore(II);
        }
        replaceOperand(*II, 0, Builder.CreateAnd(CurrCond, NextCond));
      }
      eraseInstFromFunction(*NextInst);
      return II;
    }
    break;
  }
  }
  return visitCallBase(*II);
}

// Fence instruction simplification
Instruction *InstCombiner::visitFenceInst(FenceInst &FI) {
  // Remove identical consecutive fences.
  Instruction *Next = FI.getNextNonDebugInstruction();
  if (auto *NFI = dyn_cast<FenceInst>(Next))
    if (FI.isIdenticalTo(NFI))
      return eraseInstFromFunction(FI);
  return nullptr;
}

// InvokeInst simplification
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallBase(II);
}

// CallBrInst simplification
Instruction *InstCombiner::visitCallBrInst(CallBrInst &CBI) {
  return visitCallBase(CBI);
}

/// If this cast does not affect the value passed through the varargs area, we
/// can eliminate the use of the cast.
static bool isSafeToEliminateVarargsCast(const CallBase &Call,
                                         const DataLayout &DL,
                                         const CastInst *const CI,
                                         const int ix) {
  if (!CI->isLosslessCast())
    return false;

  // If this is a GC intrinsic, avoid munging types.  We need types for
  // statepoint reconstruction in SelectionDAG.
  // TODO: This is probably something which should be expanded to all
  // intrinsics since the entire point of intrinsics is that
  // they are understandable by the optimizer.
  if (isStatepoint(&Call) || isGCRelocate(&Call) || isGCResult(&Call))
    return false;

  // The size of ByVal or InAlloca arguments is derived from the type, so we
  // can't change to a type with a different size.  If the size were
  // passed explicitly we could avoid this check.
  if (!Call.isPassPointeeByValueArgument(ix))
    return true;

  Type* SrcTy =
            cast<PointerType>(CI->getOperand(0)->getType())->getElementType();
  Type *DstTy = Call.isByValArgument(ix)
                    ? Call.getParamByValType(ix)
                    : cast<PointerType>(CI->getType())->getElementType();
  if (!SrcTy->isSized() || !DstTy->isSized())
    return false;
  if (DL.getTypeAllocSize(SrcTy) != DL.getTypeAllocSize(DstTy))
    return false;
  return true;
}

Instruction *InstCombiner::tryOptimizeCall(CallInst *CI) {
  if (!CI->getCalledFunction()) return nullptr;

  auto InstCombineRAUW = [this](Instruction *From, Value *With) {
    replaceInstUsesWith(*From, With);
  };
  auto InstCombineErase = [this](Instruction *I) {
    eraseInstFromFunction(*I);
  };
  LibCallSimplifier Simplifier(DL, &TLI, ORE, BFI, PSI, InstCombineRAUW,
                               InstCombineErase);
  if (Value *With = Simplifier.optimizeCall(CI, Builder)) {
    ++NumSimplified;
    return CI->use_empty() ? CI : replaceInstUsesWith(*CI, With);
  }

  return nullptr;
}

static IntrinsicInst *findInitTrampolineFromAlloca(Value *TrampMem) {
  // Strip off at most one level of pointer casts, looking for an alloca.  This
  // is good enough in practice and simpler than handling any number of casts.
  Value *Underlying = TrampMem->stripPointerCasts();
  if (Underlying != TrampMem &&
      (!Underlying->hasOneUse() || Underlying->user_back() != TrampMem))
    return nullptr;
  if (!isa<AllocaInst>(Underlying))
    return nullptr;

  IntrinsicInst *InitTrampoline = nullptr;
  for (User *U : TrampMem->users()) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
    if (!II)
      return nullptr;
    if (II->getIntrinsicID() == Intrinsic::init_trampoline) {
      if (InitTrampoline)
        // More than one init_trampoline writes to this value.  Give up.
        return nullptr;
      InitTrampoline = II;
      continue;
    }
    if (II->getIntrinsicID() == Intrinsic::adjust_trampoline)
      // Allow any number of calls to adjust.trampoline.
      continue;
    return nullptr;
  }

  // No call to init.trampoline found.
  if (!InitTrampoline)
    return nullptr;

  // Check that the alloca is being used in the expected way.
  if (InitTrampoline->getOperand(0) != TrampMem)
    return nullptr;

  return InitTrampoline;
}

static IntrinsicInst *findInitTrampolineFromBB(IntrinsicInst *AdjustTramp,
                                               Value *TrampMem) {
  // Visit all the previous instructions in the basic block, and try to find a
  // init.trampoline which has a direct path to the adjust.trampoline.
  for (BasicBlock::iterator I = AdjustTramp->getIterator(),
                            E = AdjustTramp->getParent()->begin();
       I != E;) {
    Instruction *Inst = &*--I;
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
      if (II->getIntrinsicID() == Intrinsic::init_trampoline &&
          II->getOperand(0) == TrampMem)
        return II;
    if (Inst->mayWriteToMemory())
      return nullptr;
  }
  return nullptr;
}

// Given a call to llvm.adjust.trampoline, find and return the corresponding
// call to llvm.init.trampoline if the call to the trampoline can be optimized
// to a direct call to a function.  Otherwise return NULL.
static IntrinsicInst *findInitTrampoline(Value *Callee) {
  Callee = Callee->stripPointerCasts();
  IntrinsicInst *AdjustTramp = dyn_cast<IntrinsicInst>(Callee);
  if (!AdjustTramp ||
      AdjustTramp->getIntrinsicID() != Intrinsic::adjust_trampoline)
    return nullptr;

  Value *TrampMem = AdjustTramp->getOperand(0);

  if (IntrinsicInst *IT = findInitTrampolineFromAlloca(TrampMem))
    return IT;
  if (IntrinsicInst *IT = findInitTrampolineFromBB(AdjustTramp, TrampMem))
    return IT;
  return nullptr;
}

static void annotateAnyAllocSite(CallBase &Call, const TargetLibraryInfo *TLI) {
  unsigned NumArgs = Call.getNumArgOperands();
  ConstantInt *Op0C = dyn_cast<ConstantInt>(Call.getOperand(0));
  ConstantInt *Op1C =
      (NumArgs == 1) ? nullptr : dyn_cast<ConstantInt>(Call.getOperand(1));
  // Bail out if the allocation size is zero (or an invalid alignment of zero
  // with aligned_alloc).
  if ((Op0C && Op0C->isNullValue()) || (Op1C && Op1C->isNullValue()))
    return;

  if (isMallocLikeFn(&Call, TLI) && Op0C) {
    if (isOpNewLikeFn(&Call, TLI))
      Call.addAttribute(AttributeList::ReturnIndex,
                        Attribute::getWithDereferenceableBytes(
                            Call.getContext(), Op0C->getZExtValue()));
    else
      Call.addAttribute(AttributeList::ReturnIndex,
                        Attribute::getWithDereferenceableOrNullBytes(
                            Call.getContext(), Op0C->getZExtValue()));
  } else if (isAlignedAllocLikeFn(&Call, TLI) && Op1C) {
    Call.addAttribute(AttributeList::ReturnIndex,
                      Attribute::getWithDereferenceableOrNullBytes(
                          Call.getContext(), Op1C->getZExtValue()));
    // Add alignment attribute if alignment is a power of two constant.
    if (Op0C) {
      uint64_t AlignmentVal = Op0C->getZExtValue();
      if (llvm::isPowerOf2_64(AlignmentVal))
        Call.addAttribute(AttributeList::ReturnIndex,
                          Attribute::getWithAlignment(Call.getContext(),
                                                      Align(AlignmentVal)));
    }
  } else if (isReallocLikeFn(&Call, TLI) && Op1C) {
    Call.addAttribute(AttributeList::ReturnIndex,
                      Attribute::getWithDereferenceableOrNullBytes(
                          Call.getContext(), Op1C->getZExtValue()));
  } else if (isCallocLikeFn(&Call, TLI) && Op0C && Op1C) {
    bool Overflow;
    const APInt &N = Op0C->getValue();
    APInt Size = N.umul_ov(Op1C->getValue(), Overflow);
    if (!Overflow)
      Call.addAttribute(AttributeList::ReturnIndex,
                        Attribute::getWithDereferenceableOrNullBytes(
                            Call.getContext(), Size.getZExtValue()));
  } else if (isStrdupLikeFn(&Call, TLI)) {
    uint64_t Len = GetStringLength(Call.getOperand(0));
    if (Len) {
      // strdup
      if (NumArgs == 1)
        Call.addAttribute(AttributeList::ReturnIndex,
                          Attribute::getWithDereferenceableOrNullBytes(
                              Call.getContext(), Len));
      // strndup
      else if (NumArgs == 2 && Op1C)
        Call.addAttribute(
            AttributeList::ReturnIndex,
            Attribute::getWithDereferenceableOrNullBytes(
                Call.getContext(), std::min(Len, Op1C->getZExtValue() + 1)));
    }
  }
}

/// Improvements for call, callbr and invoke instructions.
Instruction *InstCombiner::visitCallBase(CallBase &Call) {
  if (isAllocationFn(&Call, &TLI))
    annotateAnyAllocSite(Call, &TLI);

  bool Changed = false;

  // Mark any parameters that are known to be non-null with the nonnull
  // attribute.  This is helpful for inlining calls to functions with null
  // checks on their arguments.
  SmallVector<unsigned, 4> ArgNos;
  unsigned ArgNo = 0;

  for (Value *V : Call.args()) {
    if (V->getType()->isPointerTy() &&
        !Call.paramHasAttr(ArgNo, Attribute::NonNull) &&
        isKnownNonZero(V, DL, 0, &AC, &Call, &DT))
      ArgNos.push_back(ArgNo);
    ArgNo++;
  }

  assert(ArgNo == Call.arg_size() && "sanity check");

  if (!ArgNos.empty()) {
    AttributeList AS = Call.getAttributes();
    LLVMContext &Ctx = Call.getContext();
    AS = AS.addParamAttribute(Ctx, ArgNos,
                              Attribute::get(Ctx, Attribute::NonNull));
    Call.setAttributes(AS);
    Changed = true;
  }

  // If the callee is a pointer to a function, attempt to move any casts to the
  // arguments of the call/callbr/invoke.
  Value *Callee = Call.getCalledOperand();
  if (!isa<Function>(Callee) && transformConstExprCastCall(Call))
    return nullptr;

  if (Function *CalleeF = dyn_cast<Function>(Callee)) {
    // Remove the convergent attr on calls when the callee is not convergent.
    if (Call.isConvergent() && !CalleeF->isConvergent() &&
        !CalleeF->isIntrinsic()) {
      LLVM_DEBUG(dbgs() << "Removing convergent attr from instr " << Call
                        << "\n");
      Call.setNotConvergent();
      return &Call;
    }

    // If the call and callee calling conventions don't match, this call must
    // be unreachable, as the call is undefined.
    if (CalleeF->getCallingConv() != Call.getCallingConv() &&
        // Only do this for calls to a function with a body.  A prototype may
        // not actually end up matching the implementation's calling conv for a
        // variety of reasons (e.g. it may be written in assembly).
        !CalleeF->isDeclaration()) {
      Instruction *OldCall = &Call;
      CreateNonTerminatorUnreachable(OldCall);
      // If OldCall does not return void then replaceAllUsesWith undef.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!OldCall->getType()->isVoidTy())
        replaceInstUsesWith(*OldCall, UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))
        return eraseInstFromFunction(*OldCall);

      // We cannot remove an invoke or a callbr, because it would change thexi
      // CFG, just change the callee to a null pointer.
      cast<CallBase>(OldCall)->setCalledFunction(
          CalleeF->getFunctionType(),
          Constant::getNullValue(CalleeF->getType()));
      return nullptr;
    }
  }

  if ((isa<ConstantPointerNull>(Callee) &&
       !NullPointerIsDefined(Call.getFunction())) ||
      isa<UndefValue>(Callee)) {
    // If Call does not return void then replaceAllUsesWith undef.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!Call.getType()->isVoidTy())
      replaceInstUsesWith(Call, UndefValue::get(Call.getType()));

    if (Call.isTerminator()) {
      // Can't remove an invoke or callbr because we cannot change the CFG.
      return nullptr;
    }

    // This instruction is not reachable, just remove it.
    CreateNonTerminatorUnreachable(&Call);
    return eraseInstFromFunction(Call);
  }

  if (IntrinsicInst *II = findInitTrampoline(Callee))
    return transformCallThroughTrampoline(Call, *II);

  PointerType *PTy = cast<PointerType>(Callee->getType());
  FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    int ix = FTy->getNumParams();
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (auto I = Call.arg_begin() + FTy->getNumParams(), E = Call.arg_end();
         I != E; ++I, ++ix) {
      CastInst *CI = dyn_cast<CastInst>(*I);
      if (CI && isSafeToEliminateVarargsCast(Call, DL, CI, ix)) {
        replaceUse(*I, CI->getOperand(0));

        // Update the byval type to match the argument type.
        if (Call.isByValArgument(ix)) {
          Call.removeParamAttr(ix, Attribute::ByVal);
          Call.addParamAttr(
              ix, Attribute::getWithByValType(
                      Call.getContext(),
                      CI->getOperand(0)->getType()->getPointerElementType()));
        }
        Changed = true;
      }
    }
  }

  if (isa<InlineAsm>(Callee) && !Call.doesNotThrow()) {
    // Inline asm calls cannot throw - mark them 'nounwind'.
    Call.setDoesNotThrow();
    Changed = true;
  }

  // Try to optimize the call if possible, we require DataLayout for most of
  // this.  None of these calls are seen as possibly dead so go ahead and
  // delete the instruction now.
  if (CallInst *CI = dyn_cast<CallInst>(&Call)) {
    Instruction *I = tryOptimizeCall(CI);
    // If we changed something return the result, etc. Otherwise let
    // the fallthrough check.
    if (I) return eraseInstFromFunction(*I);
  }

  if (!Call.use_empty() && !Call.isMustTailCall())
    if (Value *ReturnedArg = Call.getReturnedArgOperand()) {
      Type *CallTy = Call.getType();
      Type *RetArgTy = ReturnedArg->getType();
      if (RetArgTy->canLosslesslyBitCastTo(CallTy))
        return replaceInstUsesWith(
            Call, Builder.CreateBitOrPointerCast(ReturnedArg, CallTy));
    }

  if (isAllocLikeFn(&Call, &TLI))
    return visitAllocSite(Call);

  return Changed ? &Call : nullptr;
}

/// If the callee is a constexpr cast of a function, attempt to move the cast to
/// the arguments of the call/callbr/invoke.
bool InstCombiner::transformConstExprCastCall(CallBase &Call) {
  auto *Callee =
      dyn_cast<Function>(Call.getCalledOperand()->stripPointerCasts());
  if (!Callee)
    return false;

  // If this is a call to a thunk function, don't remove the cast. Thunks are
  // used to transparently forward all incoming parameters and outgoing return
  // values, so it's important to leave the cast in place.
  if (Callee->hasFnAttribute("thunk"))
    return false;

  // If this is a musttail call, the callee's prototype must match the caller's
  // prototype with the exception of pointee types. The code below doesn't
  // implement that, so we can't do this transform.
  // TODO: Do the transform if it only requires adding pointer casts.
  if (Call.isMustTailCall())
    return false;

  Instruction *Caller = &Call;
  const AttributeList &CallerPAL = Call.getAttributes();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  FunctionType *FT = Callee->getFunctionType();
  Type *OldRetTy = Caller->getType();
  Type *NewRetTy = FT->getReturnType();

  // Check to see if we are changing the return type...
  if (OldRetTy != NewRetTy) {

    if (NewRetTy->isStructTy())
      return false; // TODO: Handle multiple return values.

    if (!CastInst::isBitOrNoopPointerCastable(NewRetTy, OldRetTy, DL)) {
      if (Callee->isDeclaration())
        return false;   // Cannot transform this return value.

      if (!Caller->use_empty() &&
          // void -> non-void is handled specially
          !NewRetTy->isVoidTy())
        return false;   // Cannot transform this return value.
    }

    if (!CallerPAL.isEmpty() && !Caller->use_empty()) {
      AttrBuilder RAttrs(CallerPAL, AttributeList::ReturnIndex);
      if (RAttrs.overlaps(AttributeFuncs::typeIncompatible(NewRetTy)))
        return false;   // Attribute not compatible with transformed value.
    }

    // If the callbase is an invoke/callbr instruction, and the return value is
    // used by a PHI node in a successor, we cannot change the return type of
    // the call because there is no place to put the cast instruction (without
    // breaking the critical edge).  Bail out in this case.
    if (!Caller->use_empty()) {
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller))
        for (User *U : II->users())
          if (PHINode *PN = dyn_cast<PHINode>(U))
            if (PN->getParent() == II->getNormalDest() ||
                PN->getParent() == II->getUnwindDest())
              return false;
      // FIXME: Be conservative for callbr to avoid a quadratic search.
      if (isa<CallBrInst>(Caller))
        return false;
    }
  }

  unsigned NumActualArgs = Call.arg_size();
  unsigned NumCommonArgs = std::min(FT->getNumParams(), NumActualArgs);

  // Prevent us turning:
  // declare void @takes_i32_inalloca(i32* inalloca)
  //  call void bitcast (void (i32*)* @takes_i32_inalloca to void (i32)*)(i32 0)
  //
  // into:
  //  call void @takes_i32_inalloca(i32* null)
  //
  //  Similarly, avoid folding away bitcasts of byval calls.
  if (Callee->getAttributes().hasAttrSomewhere(Attribute::InAlloca) ||
      Callee->getAttributes().hasAttrSomewhere(Attribute::ByVal))
    return false;

  auto AI = Call.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    Type *ParamTy = FT->getParamType(i);
    Type *ActTy = (*AI)->getType();

    if (!CastInst::isBitOrNoopPointerCastable(ActTy, ParamTy, DL))
      return false;   // Cannot transform this parameter value.

    if (AttrBuilder(CallerPAL.getParamAttributes(i))
            .overlaps(AttributeFuncs::typeIncompatible(ParamTy)))
      return false;   // Attribute not compatible with transformed value.

    if (Call.isInAllocaArgument(i))
      return false;   // Cannot transform to and from inalloca.

    // If the parameter is passed as a byval argument, then we have to have a
    // sized type and the sized type has to have the same size as the old type.
    if (ParamTy != ActTy && CallerPAL.hasParamAttribute(i, Attribute::ByVal)) {
      PointerType *ParamPTy = dyn_cast<PointerType>(ParamTy);
      if (!ParamPTy || !ParamPTy->getElementType()->isSized())
        return false;

      Type *CurElTy = Call.getParamByValType(i);
      if (DL.getTypeAllocSize(CurElTy) !=
          DL.getTypeAllocSize(ParamPTy->getElementType()))
        return false;
    }
  }

  if (Callee->isDeclaration()) {
    // Do not delete arguments unless we have a function body.
    if (FT->getNumParams() < NumActualArgs && !FT->isVarArg())
      return false;

    // If the callee is just a declaration, don't change the varargsness of the
    // call.  We don't want to introduce a varargs call where one doesn't
    // already exist.
    PointerType *APTy = cast<PointerType>(Call.getCalledOperand()->getType());
    if (FT->isVarArg()!=cast<FunctionType>(APTy->getElementType())->isVarArg())
      return false;

    // If both the callee and the cast type are varargs, we still have to make
    // sure the number of fixed parameters are the same or we have the same
    // ABI issues as if we introduce a varargs call.
    if (FT->isVarArg() &&
        cast<FunctionType>(APTy->getElementType())->isVarArg() &&
        FT->getNumParams() !=
        cast<FunctionType>(APTy->getElementType())->getNumParams())
      return false;
  }

  if (FT->getNumParams() < NumActualArgs && FT->isVarArg() &&
      !CallerPAL.isEmpty()) {
    // In this case we have more arguments than the new function type, but we
    // won't be dropping them.  Check that these extra arguments have attributes
    // that are compatible with being a vararg call argument.
    unsigned SRetIdx;
    if (CallerPAL.hasAttrSomewhere(Attribute::StructRet, &SRetIdx) &&
        SRetIdx > FT->getNumParams())
      return false;
  }

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary.
  SmallVector<Value *, 8> Args;
  SmallVector<AttributeSet, 8> ArgAttrs;
  Args.reserve(NumActualArgs);
  ArgAttrs.reserve(NumActualArgs);

  // Get any return attributes.
  AttrBuilder RAttrs(CallerPAL, AttributeList::ReturnIndex);

  // If the return value is not being used, the type may not be compatible
  // with the existing attributes.  Wipe out any problematic attributes.
  RAttrs.remove(AttributeFuncs::typeIncompatible(NewRetTy));

  LLVMContext &Ctx = Call.getContext();
  AI = Call.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    Type *ParamTy = FT->getParamType(i);

    Value *NewArg = *AI;
    if ((*AI)->getType() != ParamTy)
      NewArg = Builder.CreateBitOrPointerCast(*AI, ParamTy);
    Args.push_back(NewArg);

    // Add any parameter attributes.
    if (CallerPAL.hasParamAttribute(i, Attribute::ByVal)) {
      AttrBuilder AB(CallerPAL.getParamAttributes(i));
      AB.addByValAttr(NewArg->getType()->getPointerElementType());
      ArgAttrs.push_back(AttributeSet::get(Ctx, AB));
    } else
      ArgAttrs.push_back(CallerPAL.getParamAttributes(i));
  }

  // If the function takes more arguments than the call was taking, add them
  // now.
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i) {
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));
    ArgAttrs.push_back(AttributeSet());
  }

  // If we are removing arguments to the function, emit an obnoxious warning.
  if (FT->getNumParams() < NumActualArgs) {
    // TODO: if (!FT->isVarArg()) this call may be unreachable. PR14722
    if (FT->isVarArg()) {
      // Add all of the arguments in their promoted form to the arg list.
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        Type *PTy = getPromotedType((*AI)->getType());
        Value *NewArg = *AI;
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction::CastOps opcode =
            CastInst::getCastOpcode(*AI, false, PTy, false);
          NewArg = Builder.CreateCast(opcode, *AI, PTy);
        }
        Args.push_back(NewArg);

        // Add any parameter attributes.
        ArgAttrs.push_back(CallerPAL.getParamAttributes(i));
      }
    }
  }

  AttributeSet FnAttrs = CallerPAL.getFnAttributes();

  if (NewRetTy->isVoidTy())
    Caller->setName("");   // Void type should not have a name.

  assert((ArgAttrs.size() == FT->getNumParams() || FT->isVarArg()) &&
         "missing argument attributes");
  AttributeList NewCallerPAL = AttributeList::get(
      Ctx, FnAttrs, AttributeSet::get(Ctx, RAttrs), ArgAttrs);

  SmallVector<OperandBundleDef, 1> OpBundles;
  Call.getOperandBundlesAsDefs(OpBundles);

  CallBase *NewCall;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NewCall = Builder.CreateInvoke(Callee, II->getNormalDest(),
                                   II->getUnwindDest(), Args, OpBundles);
  } else if (CallBrInst *CBI = dyn_cast<CallBrInst>(Caller)) {
    NewCall = Builder.CreateCallBr(Callee, CBI->getDefaultDest(),
                                   CBI->getIndirectDests(), Args, OpBundles);
  } else {
    NewCall = Builder.CreateCall(Callee, Args, OpBundles);
    cast<CallInst>(NewCall)->setTailCallKind(
        cast<CallInst>(Caller)->getTailCallKind());
  }
  NewCall->takeName(Caller);
  NewCall->setCallingConv(Call.getCallingConv());
  NewCall->setAttributes(NewCallerPAL);

  // Preserve the weight metadata for the new call instruction. The metadata
  // is used by SamplePGO to check callsite's hotness.
  uint64_t W;
  if (Caller->extractProfTotalWeight(W))
    NewCall->setProfWeight(W);

  // Insert a cast of the return type as necessary.
  Instruction *NC = NewCall;
  Value *NV = NC;
  if (OldRetTy != NV->getType() && !Caller->use_empty()) {
    if (!NV->getType()->isVoidTy()) {
      NV = NC = CastInst::CreateBitOrPointerCast(NC, OldRetTy);
      NC->setDebugLoc(Caller->getDebugLoc());

      // If this is an invoke/callbr instruction, we should insert it after the
      // first non-phi instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->getFirstInsertionPt();
        InsertNewInstBefore(NC, *I);
      } else if (CallBrInst *CBI = dyn_cast<CallBrInst>(Caller)) {
        BasicBlock::iterator I = CBI->getDefaultDest()->getFirstInsertionPt();
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call.
        InsertNewInstBefore(NC, *Caller);
      }
      Worklist.pushUsersToWorkList(*Caller);
    } else {
      NV = UndefValue::get(Caller->getType());
    }
  }

  if (!Caller->use_empty())
    replaceInstUsesWith(*Caller, NV);
  else if (Caller->hasValueHandle()) {
    if (OldRetTy == NV->getType())
      ValueHandleBase::ValueIsRAUWd(Caller, NV);
    else
      // We cannot call ValueIsRAUWd with a different type, and the
      // actual tracked value will disappear.
      ValueHandleBase::ValueIsDeleted(Caller);
  }

  eraseInstFromFunction(*Caller);
  return true;
}

/// Turn a call to a function created by init_trampoline / adjust_trampoline
/// intrinsic pair into a direct call to the underlying function.
Instruction *
InstCombiner::transformCallThroughTrampoline(CallBase &Call,
                                             IntrinsicInst &Tramp) {
  Value *Callee = Call.getCalledOperand();
  Type *CalleeTy = Callee->getType();
  FunctionType *FTy = Call.getFunctionType();
  AttributeList Attrs = Call.getAttributes();

  // If the call already has the 'nest' attribute somewhere then give up -
  // otherwise 'nest' would occur twice after splicing in the chain.
  if (Attrs.hasAttrSomewhere(Attribute::Nest))
    return nullptr;

  Function *NestF = cast<Function>(Tramp.getArgOperand(1)->stripPointerCasts());
  FunctionType *NestFTy = NestF->getFunctionType();

  AttributeList NestAttrs = NestF->getAttributes();
  if (!NestAttrs.isEmpty()) {
    unsigned NestArgNo = 0;
    Type *NestTy = nullptr;
    AttributeSet NestAttr;

    // Look for a parameter marked with the 'nest' attribute.
    for (FunctionType::param_iterator I = NestFTy->param_begin(),
                                      E = NestFTy->param_end();
         I != E; ++NestArgNo, ++I) {
      AttributeSet AS = NestAttrs.getParamAttributes(NestArgNo);
      if (AS.hasAttribute(Attribute::Nest)) {
        // Record the parameter type and any other attributes.
        NestTy = *I;
        NestAttr = AS;
        break;
      }
    }

    if (NestTy) {
      std::vector<Value*> NewArgs;
      std::vector<AttributeSet> NewArgAttrs;
      NewArgs.reserve(Call.arg_size() + 1);
      NewArgAttrs.reserve(Call.arg_size());

      // Insert the nest argument into the call argument list, which may
      // mean appending it.  Likewise for attributes.

      {
        unsigned ArgNo = 0;
        auto I = Call.arg_begin(), E = Call.arg_end();
        do {
          if (ArgNo == NestArgNo) {
            // Add the chain argument and attributes.
            Value *NestVal = Tramp.getArgOperand(2);
            if (NestVal->getType() != NestTy)
              NestVal = Builder.CreateBitCast(NestVal, NestTy, "nest");
            NewArgs.push_back(NestVal);
            NewArgAttrs.push_back(NestAttr);
          }

          if (I == E)
            break;

          // Add the original argument and attributes.
          NewArgs.push_back(*I);
          NewArgAttrs.push_back(Attrs.getParamAttributes(ArgNo));

          ++ArgNo;
          ++I;
        } while (true);
      }

      // The trampoline may have been bitcast to a bogus type (FTy).
      // Handle this by synthesizing a new function type, equal to FTy
      // with the chain parameter inserted.

      std::vector<Type*> NewTypes;
      NewTypes.reserve(FTy->getNumParams()+1);

      // Insert the chain's type into the list of parameter types, which may
      // mean appending it.
      {
        unsigned ArgNo = 0;
        FunctionType::param_iterator I = FTy->param_begin(),
          E = FTy->param_end();

        do {
          if (ArgNo == NestArgNo)
            // Add the chain's type.
            NewTypes.push_back(NestTy);

          if (I == E)
            break;

          // Add the original type.
          NewTypes.push_back(*I);

          ++ArgNo;
          ++I;
        } while (true);
      }

      // Replace the trampoline call with a direct call.  Let the generic
      // code sort out any function type mismatches.
      FunctionType *NewFTy = FunctionType::get(FTy->getReturnType(), NewTypes,
                                                FTy->isVarArg());
      Constant *NewCallee =
        NestF->getType() == PointerType::getUnqual(NewFTy) ?
        NestF : ConstantExpr::getBitCast(NestF,
                                         PointerType::getUnqual(NewFTy));
      AttributeList NewPAL =
          AttributeList::get(FTy->getContext(), Attrs.getFnAttributes(),
                             Attrs.getRetAttributes(), NewArgAttrs);

      SmallVector<OperandBundleDef, 1> OpBundles;
      Call.getOperandBundlesAsDefs(OpBundles);

      Instruction *NewCaller;
      if (InvokeInst *II = dyn_cast<InvokeInst>(&Call)) {
        NewCaller = InvokeInst::Create(NewFTy, NewCallee,
                                       II->getNormalDest(), II->getUnwindDest(),
                                       NewArgs, OpBundles);
        cast<InvokeInst>(NewCaller)->setCallingConv(II->getCallingConv());
        cast<InvokeInst>(NewCaller)->setAttributes(NewPAL);
      } else if (CallBrInst *CBI = dyn_cast<CallBrInst>(&Call)) {
        NewCaller =
            CallBrInst::Create(NewFTy, NewCallee, CBI->getDefaultDest(),
                               CBI->getIndirectDests(), NewArgs, OpBundles);
        cast<CallBrInst>(NewCaller)->setCallingConv(CBI->getCallingConv());
        cast<CallBrInst>(NewCaller)->setAttributes(NewPAL);
      } else {
        NewCaller = CallInst::Create(NewFTy, NewCallee, NewArgs, OpBundles);
        cast<CallInst>(NewCaller)->setTailCallKind(
            cast<CallInst>(Call).getTailCallKind());
        cast<CallInst>(NewCaller)->setCallingConv(
            cast<CallInst>(Call).getCallingConv());
        cast<CallInst>(NewCaller)->setAttributes(NewPAL);
      }
      NewCaller->setDebugLoc(Call.getDebugLoc());

      return NewCaller;
    }
  }

  // Replace the trampoline call with a direct call.  Since there is no 'nest'
  // parameter, there is no need to adjust the argument list.  Let the generic
  // code sort out any function type mismatches.
  Constant *NewCallee = ConstantExpr::getBitCast(NestF, CalleeTy);
  Call.setCalledFunction(FTy, NewCallee);
  return &Call;
}
