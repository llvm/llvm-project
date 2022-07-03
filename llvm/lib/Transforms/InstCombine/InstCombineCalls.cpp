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
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
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
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IntrinsicsHexagon.h"
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
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Utils/InstructionWorklist.h"

using namespace llvm;
using namespace PatternMatch;

STATISTIC(NumSimplified, "Number of library calls simplified");

static cl::opt<unsigned> GuardWideningWindow(
    "instcombine-guard-widening-window",
    cl::init(3),
    cl::desc("How wide an instruction window to bypass looking for "
             "another guard"));

namespace llvm {
/// enable preservation of attributes in assume like:
/// call void @llvm.assume(i1 true) [ "nonnull"(i32* %PTR) ]
extern cl::opt<bool> EnableKnowledgeRetention;
} // namespace llvm

/// Return the specified type promoted as it would be to pass though a va_arg
/// area.
static Type *getPromotedType(Type *Ty) {
  if (IntegerType* ITy = dyn_cast<IntegerType>(Ty)) {
    if (ITy->getBitWidth() < 32)
      return Type::getInt32Ty(Ty->getContext());
  }
  return Ty;
}

/// Recognize a memcpy/memmove from a trivially otherwise unused alloca.
/// TODO: This should probably be integrated with visitAllocSites, but that
/// requires a deeper change to allow either unread or unwritten objects.
static bool hasUndefSource(AnyMemTransferInst *MI) {
  auto *Src = MI->getRawSource();
  while (isa<GetElementPtrInst>(Src) || isa<BitCastInst>(Src)) {
    if (!Src->hasOneUse())
      return false;
    Src = cast<Instruction>(Src)->getOperand(0);
  }
  return isa<AllocaInst>(Src) && Src->hasOneUse();
}

Instruction *InstCombinerImpl::SimplifyAnyMemTransfer(AnyMemTransferInst *MI) {
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

  // If the source is provably undef, the memcpy/memmove doesn't do anything
  // (unless the transfer is volatile).
  if (hasUndefSource(MI) && !MI->isVolatile()) {
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
    if (*CopyDstAlign < Size || *CopySrcAlign < Size)
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
  S->setAlignment(*CopyDstAlign);
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

Instruction *InstCombinerImpl::SimplifyAnyMemSet(AnyMemSetInst *MI) {
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

  // Remove memset with an undef value.
  // FIXME: This is technically incorrect because it might overwrite a poison
  // value. Change to PoisonValue once #52930 is resolved.
  if (isa<UndefValue>(MI->getValue())) {
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
  const Align Alignment = MI->getDestAlign().valueOrOne();

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

// TODO, Obvious Missing Transforms:
// * Narrow width by halfs excluding zero/undef lanes
Value *InstCombinerImpl::simplifyMaskedLoad(IntrinsicInst &II) {
  Value *LoadPtr = II.getArgOperand(0);
  const Align Alignment =
      cast<ConstantInt>(II.getArgOperand(1))->getAlignValue();

  // If the mask is all ones or undefs, this is a plain vector load of the 1st
  // argument.
  if (maskIsAllOneOrUndef(II.getArgOperand(2))) {
    LoadInst *L = Builder.CreateAlignedLoad(II.getType(), LoadPtr, Alignment,
                                            "unmaskedload");
    L->copyMetadata(II);
    return L;
  }

  // If we can unconditionally load from this address, replace with a
  // load/select idiom. TODO: use DT for context sensitive query
  if (isDereferenceablePointer(LoadPtr, II.getType(),
                               II.getModule()->getDataLayout(), &II, nullptr)) {
    LoadInst *LI = Builder.CreateAlignedLoad(II.getType(), LoadPtr, Alignment,
                                             "unmaskedload");
    LI->copyMetadata(II);
    return Builder.CreateSelect(II.getArgOperand(2), LI, II.getArgOperand(3));
  }

  return nullptr;
}

// TODO, Obvious Missing Transforms:
// * Single constant active lane -> store
// * Narrow width by halfs excluding zero/undef lanes
Instruction *InstCombinerImpl::simplifyMaskedStore(IntrinsicInst &II) {
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(3));
  if (!ConstMask)
    return nullptr;

  // If the mask is all zeros, this instruction does nothing.
  if (ConstMask->isNullValue())
    return eraseInstFromFunction(II);

  // If the mask is all ones, this is a plain vector store of the 1st argument.
  if (ConstMask->isAllOnesValue()) {
    Value *StorePtr = II.getArgOperand(1);
    Align Alignment = cast<ConstantInt>(II.getArgOperand(2))->getAlignValue();
    StoreInst *S =
        new StoreInst(II.getArgOperand(0), StorePtr, false, Alignment);
    S->copyMetadata(II);
    return S;
  }

  if (isa<ScalableVectorType>(ConstMask->getType()))
    return nullptr;

  // Use masked off lanes to simplify operands via SimplifyDemandedVectorElts
  APInt DemandedElts = possiblyDemandedEltsInMask(ConstMask);
  APInt UndefElts(DemandedElts.getBitWidth(), 0);
  if (Value *V =
          SimplifyDemandedVectorElts(II.getOperand(0), DemandedElts, UndefElts))
    return replaceOperand(II, 0, V);

  return nullptr;
}

// TODO, Obvious Missing Transforms:
// * Single constant active lane load -> load
// * Dereferenceable address & few lanes -> scalarize speculative load/selects
// * Adjacent vector addresses -> masked.load
// * Narrow width by halfs excluding zero/undef lanes
// * Vector incrementing address -> vector masked load
Instruction *InstCombinerImpl::simplifyMaskedGather(IntrinsicInst &II) {
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(2));
  if (!ConstMask)
    return nullptr;

  // Vector splat address w/known mask -> scalar load
  // Fold the gather to load the source vector first lane
  // because it is reloading the same value each time
  if (ConstMask->isAllOnesValue())
    if (auto *SplatPtr = getSplatValue(II.getArgOperand(0))) {
      auto *VecTy = cast<VectorType>(II.getType());
      const Align Alignment =
          cast<ConstantInt>(II.getArgOperand(1))->getAlignValue();
      LoadInst *L = Builder.CreateAlignedLoad(VecTy->getElementType(), SplatPtr,
                                              Alignment, "load.scalar");
      Value *Shuf =
          Builder.CreateVectorSplat(VecTy->getElementCount(), L, "broadcast");
      return replaceInstUsesWith(II, cast<Instruction>(Shuf));
    }

  return nullptr;
}

// TODO, Obvious Missing Transforms:
// * Single constant active lane -> store
// * Adjacent vector addresses -> masked.store
// * Narrow store width by halfs excluding zero/undef lanes
// * Vector incrementing address -> vector masked store
Instruction *InstCombinerImpl::simplifyMaskedScatter(IntrinsicInst &II) {
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(3));
  if (!ConstMask)
    return nullptr;

  // If the mask is all zeros, a scatter does nothing.
  if (ConstMask->isNullValue())
    return eraseInstFromFunction(II);

  // Vector splat address -> scalar store
  if (auto *SplatPtr = getSplatValue(II.getArgOperand(1))) {
    // scatter(splat(value), splat(ptr), non-zero-mask) -> store value, ptr
    if (auto *SplatValue = getSplatValue(II.getArgOperand(0))) {
      Align Alignment = cast<ConstantInt>(II.getArgOperand(2))->getAlignValue();
      StoreInst *S =
          new StoreInst(SplatValue, SplatPtr, /*IsVolatile=*/false, Alignment);
      S->copyMetadata(II);
      return S;
    }
    // scatter(vector, splat(ptr), splat(true)) -> store extract(vector,
    // lastlane), ptr
    if (ConstMask->isAllOnesValue()) {
      Align Alignment = cast<ConstantInt>(II.getArgOperand(2))->getAlignValue();
      VectorType *WideLoadTy = cast<VectorType>(II.getArgOperand(1)->getType());
      ElementCount VF = WideLoadTy->getElementCount();
      Constant *EC =
          ConstantInt::get(Builder.getInt32Ty(), VF.getKnownMinValue());
      Value *RunTimeVF = VF.isScalable() ? Builder.CreateVScale(EC) : EC;
      Value *LastLane = Builder.CreateSub(RunTimeVF, Builder.getInt32(1));
      Value *Extract =
          Builder.CreateExtractElement(II.getArgOperand(0), LastLane);
      StoreInst *S =
          new StoreInst(Extract, SplatPtr, /*IsVolatile=*/false, Alignment);
      S->copyMetadata(II);
      return S;
    }
  }
  if (isa<ScalableVectorType>(ConstMask->getType()))
    return nullptr;

  // Use masked off lanes to simplify operands via SimplifyDemandedVectorElts
  APInt DemandedElts = possiblyDemandedEltsInMask(ConstMask);
  APInt UndefElts(DemandedElts.getBitWidth(), 0);
  if (Value *V =
          SimplifyDemandedVectorElts(II.getOperand(0), DemandedElts, UndefElts))
    return replaceOperand(II, 0, V);
  if (Value *V =
          SimplifyDemandedVectorElts(II.getOperand(1), DemandedElts, UndefElts))
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
                                                    InstCombinerImpl &IC) {
  auto *Arg = II.getArgOperand(0);
  auto *StrippedArg = Arg->stripPointerCasts();
  auto *StrippedInvariantGroupsArg = StrippedArg;
  while (auto *Intr = dyn_cast<IntrinsicInst>(StrippedInvariantGroupsArg)) {
    if (Intr->getIntrinsicID() != Intrinsic::launder_invariant_group &&
        Intr->getIntrinsicID() != Intrinsic::strip_invariant_group)
      break;
    StrippedInvariantGroupsArg = Intr->getArgOperand(0)->stripPointerCasts();
  }
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

static Instruction *foldCttzCtlz(IntrinsicInst &II, InstCombinerImpl &IC) {
  assert((II.getIntrinsicID() == Intrinsic::cttz ||
          II.getIntrinsicID() == Intrinsic::ctlz) &&
         "Expected cttz or ctlz intrinsic");
  bool IsTZ = II.getIntrinsicID() == Intrinsic::cttz;
  Value *Op0 = II.getArgOperand(0);
  Value *Op1 = II.getArgOperand(1);
  Value *X;
  // ctlz(bitreverse(x)) -> cttz(x)
  // cttz(bitreverse(x)) -> ctlz(x)
  if (match(Op0, m_BitReverse(m_Value(X)))) {
    Intrinsic::ID ID = IsTZ ? Intrinsic::ctlz : Intrinsic::cttz;
    Function *F = Intrinsic::getDeclaration(II.getModule(), ID, II.getType());
    return CallInst::Create(F, {X, II.getArgOperand(1)});
  }

  if (II.getType()->isIntOrIntVectorTy(1)) {
    // ctlz/cttz i1 Op0 --> not Op0
    if (match(Op1, m_Zero()))
      return BinaryOperator::CreateNot(Op0);
    // If zero is poison, then the input can be assumed to be "true", so the
    // instruction simplifies to "false".
    assert(match(Op1, m_One()) && "Expected ctlz/cttz operand to be 0 or 1");
    return IC.replaceInstUsesWith(II, ConstantInt::getNullValue(II.getType()));
  }

  // If the operand is a select with constant arm(s), try to hoist ctlz/cttz.
  if (auto *Sel = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = IC.FoldOpIntoSelect(II, Sel))
      return R;

  if (IsTZ) {
    // cttz(-x) -> cttz(x)
    if (match(Op0, m_Neg(m_Value(X))))
      return IC.replaceOperand(II, 0, X);

    // cttz(sext(x)) -> cttz(zext(x))
    if (match(Op0, m_OneUse(m_SExt(m_Value(X))))) {
      auto *Zext = IC.Builder.CreateZExt(X, II.getType());
      auto *CttzZext =
          IC.Builder.CreateBinaryIntrinsic(Intrinsic::cttz, Zext, Op1);
      return IC.replaceInstUsesWith(II, CttzZext);
    }

    // Zext doesn't change the number of trailing zeros, so narrow:
    // cttz(zext(x)) -> zext(cttz(x)) if the 'ZeroIsPoison' parameter is 'true'.
    if (match(Op0, m_OneUse(m_ZExt(m_Value(X)))) && match(Op1, m_One())) {
      auto *Cttz = IC.Builder.CreateBinaryIntrinsic(Intrinsic::cttz, X,
                                                    IC.Builder.getTrue());
      auto *ZextCttz = IC.Builder.CreateZExt(Cttz, II.getType());
      return IC.replaceInstUsesWith(II, ZextCttz);
    }

    // cttz(abs(x)) -> cttz(x)
    // cttz(nabs(x)) -> cttz(x)
    Value *Y;
    SelectPatternFlavor SPF = matchSelectPattern(Op0, X, Y).Flavor;
    if (SPF == SPF_ABS || SPF == SPF_NABS)
      return IC.replaceOperand(II, 0, X);

    if (match(Op0, m_Intrinsic<Intrinsic::abs>(m_Value(X))))
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
  // then change the 'ZeroIsPoison' parameter to 'true'
  // because we know the zero behavior can't affect the result.
  if (!Known.One.isZero() ||
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

static Instruction *foldCtpop(IntrinsicInst &II, InstCombinerImpl &IC) {
  assert(II.getIntrinsicID() == Intrinsic::ctpop &&
         "Expected ctpop intrinsic");
  Type *Ty = II.getType();
  unsigned BitWidth = Ty->getScalarSizeInBits();
  Value *Op0 = II.getArgOperand(0);
  Value *X, *Y;

  // ctpop(bitreverse(x)) -> ctpop(x)
  // ctpop(bswap(x)) -> ctpop(x)
  if (match(Op0, m_BitReverse(m_Value(X))) || match(Op0, m_BSwap(m_Value(X))))
    return IC.replaceOperand(II, 0, X);

  // ctpop(rot(x)) -> ctpop(x)
  if ((match(Op0, m_FShl(m_Value(X), m_Value(Y), m_Value())) ||
       match(Op0, m_FShr(m_Value(X), m_Value(Y), m_Value()))) &&
      X == Y)
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

  // Zext doesn't change the number of set bits, so narrow:
  // ctpop (zext X) --> zext (ctpop X)
  if (match(Op0, m_OneUse(m_ZExt(m_Value(X))))) {
    Value *NarrowPop = IC.Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, X);
    return CastInst::Create(Instruction::ZExt, NarrowPop, Ty);
  }

  // If the operand is a select with constant arm(s), try to hoist ctpop.
  if (auto *Sel = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = IC.FoldOpIntoSelect(II, Sel))
      return R;

  KnownBits Known(BitWidth);
  IC.computeKnownBits(Op0, Known, 0, &II);

  // If all bits are zero except for exactly one fixed bit, then the result
  // must be 0 or 1, and we can get that answer by shifting to LSB:
  // ctpop (X & 32) --> (X & 32) >> 5
  if ((~Known.Zero).isPowerOf2())
    return BinaryOperator::CreateLShr(
        Op0, ConstantInt::get(Ty, (~Known.Zero).exactLogBase2()));

  // FIXME: Try to simplify vectors of integers.
  auto *IT = dyn_cast<IntegerType>(Ty);
  if (!IT)
    return nullptr;

  // Add range metadata since known bits can't completely reflect what we know.
  unsigned MinCount = Known.countMinPopulation();
  unsigned MaxCount = Known.countMaxPopulation();
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

  auto *VecTy = cast<FixedVectorType>(II.getType());
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

// Returns true iff the 2 intrinsics have the same operands, limiting the
// comparison to the first NumOperands.
static bool haveSameOperands(const IntrinsicInst &I, const IntrinsicInst &E,
                             unsigned NumOperands) {
  assert(I.arg_size() >= NumOperands && "Not enough operands");
  assert(E.arg_size() >= NumOperands && "Not enough operands");
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
static bool
removeTriviallyEmptyRange(IntrinsicInst &EndI, InstCombinerImpl &IC,
                          std::function<bool(const IntrinsicInst &)> IsStart) {
  // We start from the end intrinsic and scan backwards, so that InstCombine
  // has already processed (and potentially removed) all the instructions
  // before the end intrinsic.
  BasicBlock::reverse_iterator BI(EndI), BE(EndI.getParent()->rend());
  for (; BI != BE; ++BI) {
    if (auto *I = dyn_cast<IntrinsicInst>(&*BI)) {
      if (I->isDebugOrPseudoInst() ||
          I->getIntrinsicID() == EndI.getIntrinsicID())
        continue;
      if (IsStart(*I)) {
        if (haveSameOperands(EndI, *I, EndI.arg_size())) {
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

Instruction *InstCombinerImpl::visitVAEndInst(VAEndInst &I) {
  removeTriviallyEmptyRange(I, *this, [](const IntrinsicInst &I) {
    return I.getIntrinsicID() == Intrinsic::vastart ||
           I.getIntrinsicID() == Intrinsic::vacopy;
  });
  return nullptr;
}

static CallInst *canonicalizeConstantArg0ToArg1(CallInst &Call) {
  assert(Call.arg_size() > 1 && "Need at least 2 args to swap");
  Value *Arg0 = Call.getArgOperand(0), *Arg1 = Call.getArgOperand(1);
  if (isa<Constant>(Arg0) && !isa<Constant>(Arg1)) {
    Call.setArgOperand(0, Arg1);
    Call.setArgOperand(1, Arg0);
    return &Call;
  }
  return nullptr;
}

/// Creates a result tuple for an overflow intrinsic \p II with a given
/// \p Result and a constant \p Overflow value.
static Instruction *createOverflowTuple(IntrinsicInst *II, Value *Result,
                                        Constant *Overflow) {
  Constant *V[] = {PoisonValue::get(Result->getType()), Overflow};
  StructType *ST = cast<StructType>(II->getType());
  Constant *Struct = ConstantStruct::get(ST, V);
  return InsertValueInst::Create(Struct, Result, 0);
}

Instruction *
InstCombinerImpl::foldIntrinsicWithOverflowCommon(IntrinsicInst *II) {
  WithOverflowInst *WO = cast<WithOverflowInst>(II);
  Value *OperationResult = nullptr;
  Constant *OverflowResult = nullptr;
  if (OptimizeOverflowCheck(WO->getBinaryOp(), WO->isSigned(), WO->getLHS(),
                            WO->getRHS(), *WO, OperationResult, OverflowResult))
    return createOverflowTuple(WO, OperationResult, OverflowResult);
  return nullptr;
}

static Optional<bool> getKnownSign(Value *Op, Instruction *CxtI,
                                   const DataLayout &DL, AssumptionCache *AC,
                                   DominatorTree *DT) {
  KnownBits Known = computeKnownBits(Op, DL, 0, AC, CxtI, DT);
  if (Known.isNonNegative())
    return false;
  if (Known.isNegative())
    return true;

  Value *X, *Y;
  if (match(Op, m_NSWSub(m_Value(X), m_Value(Y))))
    return isImpliedByDomCondition(ICmpInst::ICMP_SLT, X, Y, CxtI, DL);

  return isImpliedByDomCondition(
      ICmpInst::ICMP_SLT, Op, Constant::getNullValue(Op->getType()), CxtI, DL);
}

/// Try to canonicalize min/max(X + C0, C1) as min/max(X, C1 - C0) + C0. This
/// can trigger other combines.
static Instruction *moveAddAfterMinMax(IntrinsicInst *II,
                                       InstCombiner::BuilderTy &Builder) {
  Intrinsic::ID MinMaxID = II->getIntrinsicID();
  assert((MinMaxID == Intrinsic::smax || MinMaxID == Intrinsic::smin ||
          MinMaxID == Intrinsic::umax || MinMaxID == Intrinsic::umin) &&
         "Expected a min or max intrinsic");

  // TODO: Match vectors with undef elements, but undef may not propagate.
  Value *Op0 = II->getArgOperand(0), *Op1 = II->getArgOperand(1);
  Value *X;
  const APInt *C0, *C1;
  if (!match(Op0, m_OneUse(m_Add(m_Value(X), m_APInt(C0)))) ||
      !match(Op1, m_APInt(C1)))
    return nullptr;

  // Check for necessary no-wrap and overflow constraints.
  bool IsSigned = MinMaxID == Intrinsic::smax || MinMaxID == Intrinsic::smin;
  auto *Add = cast<BinaryOperator>(Op0);
  if ((IsSigned && !Add->hasNoSignedWrap()) ||
      (!IsSigned && !Add->hasNoUnsignedWrap()))
    return nullptr;

  // If the constant difference overflows, then instsimplify should reduce the
  // min/max to the add or C1.
  bool Overflow;
  APInt CDiff =
      IsSigned ? C1->ssub_ov(*C0, Overflow) : C1->usub_ov(*C0, Overflow);
  assert(!Overflow && "Expected simplify of min/max");

  // min/max (add X, C0), C1 --> add (min/max X, C1 - C0), C0
  // Note: the "mismatched" no-overflow setting does not propagate.
  Constant *NewMinMaxC = ConstantInt::get(II->getType(), CDiff);
  Value *NewMinMax = Builder.CreateBinaryIntrinsic(MinMaxID, X, NewMinMaxC);
  return IsSigned ? BinaryOperator::CreateNSWAdd(NewMinMax, Add->getOperand(1))
                  : BinaryOperator::CreateNUWAdd(NewMinMax, Add->getOperand(1));
}
/// Match a sadd_sat or ssub_sat which is using min/max to clamp the value.
Instruction *InstCombinerImpl::matchSAddSubSat(IntrinsicInst &MinMax1) {
  Type *Ty = MinMax1.getType();

  // We are looking for a tree of:
  // max(INT_MIN, min(INT_MAX, add(sext(A), sext(B))))
  // Where the min and max could be reversed
  Instruction *MinMax2;
  BinaryOperator *AddSub;
  const APInt *MinValue, *MaxValue;
  if (match(&MinMax1, m_SMin(m_Instruction(MinMax2), m_APInt(MaxValue)))) {
    if (!match(MinMax2, m_SMax(m_BinOp(AddSub), m_APInt(MinValue))))
      return nullptr;
  } else if (match(&MinMax1,
                   m_SMax(m_Instruction(MinMax2), m_APInt(MinValue)))) {
    if (!match(MinMax2, m_SMin(m_BinOp(AddSub), m_APInt(MaxValue))))
      return nullptr;
  } else
    return nullptr;

  // Check that the constants clamp a saturate, and that the new type would be
  // sensible to convert to.
  if (!(*MaxValue + 1).isPowerOf2() || -*MinValue != *MaxValue + 1)
    return nullptr;
  // In what bitwidth can this be treated as saturating arithmetics?
  unsigned NewBitWidth = (*MaxValue + 1).logBase2() + 1;
  // FIXME: This isn't quite right for vectors, but using the scalar type is a
  // good first approximation for what should be done there.
  if (!shouldChangeType(Ty->getScalarType()->getIntegerBitWidth(), NewBitWidth))
    return nullptr;

  // Also make sure that the inner min/max and the add/sub have one use.
  if (!MinMax2->hasOneUse() || !AddSub->hasOneUse())
    return nullptr;

  // Create the new type (which can be a vector type)
  Type *NewTy = Ty->getWithNewBitWidth(NewBitWidth);

  Intrinsic::ID IntrinsicID;
  if (AddSub->getOpcode() == Instruction::Add)
    IntrinsicID = Intrinsic::sadd_sat;
  else if (AddSub->getOpcode() == Instruction::Sub)
    IntrinsicID = Intrinsic::ssub_sat;
  else
    return nullptr;

  // The two operands of the add/sub must be nsw-truncatable to the NewTy. This
  // is usually achieved via a sext from a smaller type.
  if (ComputeMaxSignificantBits(AddSub->getOperand(0), 0, AddSub) >
          NewBitWidth ||
      ComputeMaxSignificantBits(AddSub->getOperand(1), 0, AddSub) > NewBitWidth)
    return nullptr;

  // Finally create and return the sat intrinsic, truncated to the new type
  Function *F = Intrinsic::getDeclaration(MinMax1.getModule(), IntrinsicID, NewTy);
  Value *AT = Builder.CreateTrunc(AddSub->getOperand(0), NewTy);
  Value *BT = Builder.CreateTrunc(AddSub->getOperand(1), NewTy);
  Value *Sat = Builder.CreateCall(F, {AT, BT});
  return CastInst::Create(Instruction::SExt, Sat, Ty);
}


/// If we have a clamp pattern like max (min X, 42), 41 -- where the output
/// can only be one of two possible constant values -- turn that into a select
/// of constants.
static Instruction *foldClampRangeOfTwo(IntrinsicInst *II,
                                        InstCombiner::BuilderTy &Builder) {
  Value *I0 = II->getArgOperand(0), *I1 = II->getArgOperand(1);
  Value *X;
  const APInt *C0, *C1;
  if (!match(I1, m_APInt(C1)) || !I0->hasOneUse())
    return nullptr;

  CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
  switch (II->getIntrinsicID()) {
  case Intrinsic::smax:
    if (match(I0, m_SMin(m_Value(X), m_APInt(C0))) && *C0 == *C1 + 1)
      Pred = ICmpInst::ICMP_SGT;
    break;
  case Intrinsic::smin:
    if (match(I0, m_SMax(m_Value(X), m_APInt(C0))) && *C1 == *C0 + 1)
      Pred = ICmpInst::ICMP_SLT;
    break;
  case Intrinsic::umax:
    if (match(I0, m_UMin(m_Value(X), m_APInt(C0))) && *C0 == *C1 + 1)
      Pred = ICmpInst::ICMP_UGT;
    break;
  case Intrinsic::umin:
    if (match(I0, m_UMax(m_Value(X), m_APInt(C0))) && *C1 == *C0 + 1)
      Pred = ICmpInst::ICMP_ULT;
    break;
  default:
    llvm_unreachable("Expected min/max intrinsic");
  }
  if (Pred == CmpInst::BAD_ICMP_PREDICATE)
    return nullptr;

  // max (min X, 42), 41 --> X > 41 ? 42 : 41
  // min (max X, 42), 43 --> X < 43 ? 42 : 43
  Value *Cmp = Builder.CreateICmp(Pred, X, I1);
  return SelectInst::Create(Cmp, ConstantInt::get(II->getType(), *C0), I1);
}

/// If this min/max has a constant operand and an operand that is a matching
/// min/max with a constant operand, constant-fold the 2 constant operands.
static Instruction *reassociateMinMaxWithConstants(IntrinsicInst *II) {
  Intrinsic::ID MinMaxID = II->getIntrinsicID();
  auto *LHS = dyn_cast<IntrinsicInst>(II->getArgOperand(0));
  if (!LHS || LHS->getIntrinsicID() != MinMaxID)
    return nullptr;

  Constant *C0, *C1;
  if (!match(LHS->getArgOperand(1), m_ImmConstant(C0)) ||
      !match(II->getArgOperand(1), m_ImmConstant(C1)))
    return nullptr;

  // max (max X, C0), C1 --> max X, (max C0, C1) --> max X, NewC
  ICmpInst::Predicate Pred = MinMaxIntrinsic::getPredicate(MinMaxID);
  Constant *CondC = ConstantExpr::getICmp(Pred, C0, C1);
  Constant *NewC = ConstantExpr::getSelect(CondC, C0, C1);

  Module *Mod = II->getModule();
  Function *MinMax = Intrinsic::getDeclaration(Mod, MinMaxID, II->getType());
  return CallInst::Create(MinMax, {LHS->getArgOperand(0), NewC});
}

/// If this min/max has a matching min/max operand with a constant, try to push
/// the constant operand into this instruction. This can enable more folds.
static Instruction *
reassociateMinMaxWithConstantInOperand(IntrinsicInst *II,
                                       InstCombiner::BuilderTy &Builder) {
  // Match and capture a min/max operand candidate.
  Value *X, *Y;
  Constant *C;
  Instruction *Inner;
  if (!match(II, m_c_MaxOrMin(m_OneUse(m_CombineAnd(
                                  m_Instruction(Inner),
                                  m_MaxOrMin(m_Value(X), m_ImmConstant(C)))),
                              m_Value(Y))))
    return nullptr;

  // The inner op must match. Check for constants to avoid infinite loops.
  Intrinsic::ID MinMaxID = II->getIntrinsicID();
  auto *InnerMM = dyn_cast<IntrinsicInst>(Inner);
  if (!InnerMM || InnerMM->getIntrinsicID() != MinMaxID ||
      match(X, m_ImmConstant()) || match(Y, m_ImmConstant()))
    return nullptr;

  // max (max X, C), Y --> max (max X, Y), C
  Function *MinMax =
      Intrinsic::getDeclaration(II->getModule(), MinMaxID, II->getType());
  Value *NewInner = Builder.CreateBinaryIntrinsic(MinMaxID, X, Y);
  NewInner->takeName(Inner);
  return CallInst::Create(MinMax, {NewInner, C});
}

/// Reduce a sequence of min/max intrinsics with a common operand.
static Instruction *factorizeMinMaxTree(IntrinsicInst *II) {
  // Match 3 of the same min/max ops. Example: umin(umin(), umin()).
  auto *LHS = dyn_cast<IntrinsicInst>(II->getArgOperand(0));
  auto *RHS = dyn_cast<IntrinsicInst>(II->getArgOperand(1));
  Intrinsic::ID MinMaxID = II->getIntrinsicID();
  if (!LHS || !RHS || LHS->getIntrinsicID() != MinMaxID ||
      RHS->getIntrinsicID() != MinMaxID ||
      (!LHS->hasOneUse() && !RHS->hasOneUse()))
    return nullptr;

  Value *A = LHS->getArgOperand(0);
  Value *B = LHS->getArgOperand(1);
  Value *C = RHS->getArgOperand(0);
  Value *D = RHS->getArgOperand(1);

  // Look for a common operand.
  Value *MinMaxOp = nullptr;
  Value *ThirdOp = nullptr;
  if (LHS->hasOneUse()) {
    // If the LHS is only used in this chain and the RHS is used outside of it,
    // reuse the RHS min/max because that will eliminate the LHS.
    if (D == A || C == A) {
      // min(min(a, b), min(c, a)) --> min(min(c, a), b)
      // min(min(a, b), min(a, d)) --> min(min(a, d), b)
      MinMaxOp = RHS;
      ThirdOp = B;
    } else if (D == B || C == B) {
      // min(min(a, b), min(c, b)) --> min(min(c, b), a)
      // min(min(a, b), min(b, d)) --> min(min(b, d), a)
      MinMaxOp = RHS;
      ThirdOp = A;
    }
  } else {
    assert(RHS->hasOneUse() && "Expected one-use operand");
    // Reuse the LHS. This will eliminate the RHS.
    if (D == A || D == B) {
      // min(min(a, b), min(c, a)) --> min(min(a, b), c)
      // min(min(a, b), min(c, b)) --> min(min(a, b), c)
      MinMaxOp = LHS;
      ThirdOp = C;
    } else if (C == A || C == B) {
      // min(min(a, b), min(b, d)) --> min(min(a, b), d)
      // min(min(a, b), min(c, b)) --> min(min(a, b), d)
      MinMaxOp = LHS;
      ThirdOp = D;
    }
  }

  if (!MinMaxOp || !ThirdOp)
    return nullptr;

  Module *Mod = II->getModule();
  Function *MinMax = Intrinsic::getDeclaration(Mod, MinMaxID, II->getType());
  return CallInst::Create(MinMax, { MinMaxOp, ThirdOp });
}

/// If all arguments of the intrinsic are unary shuffles with the same mask,
/// try to shuffle after the intrinsic.
static Instruction *
foldShuffledIntrinsicOperands(IntrinsicInst *II,
                              InstCombiner::BuilderTy &Builder) {
  // TODO: This should be extended to handle other intrinsics like fshl, ctpop,
  //       etc. Use llvm::isTriviallyVectorizable() and related to determine
  //       which intrinsics are safe to shuffle?
  switch (II->getIntrinsicID()) {
  case Intrinsic::smax:
  case Intrinsic::smin:
  case Intrinsic::umax:
  case Intrinsic::umin:
  case Intrinsic::fma:
  case Intrinsic::fshl:
  case Intrinsic::fshr:
    break;
  default:
    return nullptr;
  }

  Value *X;
  ArrayRef<int> Mask;
  if (!match(II->getArgOperand(0),
             m_Shuffle(m_Value(X), m_Undef(), m_Mask(Mask))))
    return nullptr;

  // At least 1 operand must have 1 use because we are creating 2 instructions.
  if (none_of(II->args(), [](Value *V) { return V->hasOneUse(); }))
    return nullptr;

  // See if all arguments are shuffled with the same mask.
  SmallVector<Value *, 4> NewArgs(II->arg_size());
  NewArgs[0] = X;
  Type *SrcTy = X->getType();
  for (unsigned i = 1, e = II->arg_size(); i != e; ++i) {
    if (!match(II->getArgOperand(i),
               m_Shuffle(m_Value(X), m_Undef(), m_SpecificMask(Mask))) ||
        X->getType() != SrcTy)
      return nullptr;
    NewArgs[i] = X;
  }

  // intrinsic (shuf X, M), (shuf Y, M), ... --> shuf (intrinsic X, Y, ...), M
  Instruction *FPI = isa<FPMathOperator>(II) ? II : nullptr;
  Value *NewIntrinsic =
      Builder.CreateIntrinsic(II->getIntrinsicID(), SrcTy, NewArgs, FPI);
  return new ShuffleVectorInst(NewIntrinsic, Mask);
}

/// CallInst simplification. This mostly only handles folding of intrinsic
/// instructions. For normal calls, it allows visitCallBase to do the heavy
/// lifting.
Instruction *InstCombinerImpl::visitCallInst(CallInst &CI) {
  // Don't try to simplify calls without uses. It will not do anything useful,
  // but will result in the following folds being skipped.
  if (!CI.use_empty())
    if (Value *V = simplifyCall(&CI, SQ.getWithInstruction(&CI)))
      return replaceInstUsesWith(CI, V);

  if (isFreeCall(&CI, &TLI))
    return visitFree(CI);

  // If the caller function (i.e. us, the function that contains this CallInst)
  // is nounwind, mark the call as nounwind, even if the callee isn't.
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
    APInt AllOnesEltMask(APInt::getAllOnes(VWidth));
    if (Value *V = SimplifyDemandedVectorElts(II, AllOnesEltMask, UndefElts)) {
      if (V != II)
        return replaceInstUsesWith(*II, V);
      return II;
    }
  }

  if (II->isCommutative()) {
    if (CallInst *NewCall = canonicalizeConstantArg0ToArg1(CI))
      return NewCall;
  }

  // Unused constrained FP intrinsic calls may have declared side effect, which
  // prevents it from being removed. In some cases however the side effect is
  // actually absent. To detect this case, call SimplifyConstrainedFPCall. If it
  // returns a replacement, the call may be removed.
  if (CI.use_empty() && isa<ConstrainedFPIntrinsic>(CI)) {
    if (simplifyConstrainedFPCall(&CI, SQ.getWithInstruction(&CI)))
      return eraseInstFromFunction(CI);
  }

  Intrinsic::ID IID = II->getIntrinsicID();
  switch (IID) {
  case Intrinsic::objectsize:
    if (Value *V = lowerObjectSizeCall(II, DL, &TLI, AA, /*MustSucceed=*/false))
      return replaceInstUsesWith(CI, V);
    return nullptr;
  case Intrinsic::abs: {
    Value *IIOperand = II->getArgOperand(0);
    bool IntMinIsPoison = cast<Constant>(II->getArgOperand(1))->isOneValue();

    // abs(-x) -> abs(x)
    // TODO: Copy nsw if it was present on the neg?
    Value *X;
    if (match(IIOperand, m_Neg(m_Value(X))))
      return replaceOperand(*II, 0, X);
    if (match(IIOperand, m_Select(m_Value(), m_Value(X), m_Neg(m_Deferred(X)))))
      return replaceOperand(*II, 0, X);
    if (match(IIOperand, m_Select(m_Value(), m_Neg(m_Value(X)), m_Deferred(X))))
      return replaceOperand(*II, 0, X);

    if (Optional<bool> Sign = getKnownSign(IIOperand, II, DL, &AC, &DT)) {
      // abs(x) -> x if x >= 0
      if (!*Sign)
        return replaceInstUsesWith(*II, IIOperand);

      // abs(x) -> -x if x < 0
      if (IntMinIsPoison)
        return BinaryOperator::CreateNSWNeg(IIOperand);
      return BinaryOperator::CreateNeg(IIOperand);
    }

    // abs (sext X) --> zext (abs X*)
    // Clear the IsIntMin (nsw) bit on the abs to allow narrowing.
    if (match(IIOperand, m_OneUse(m_SExt(m_Value(X))))) {
      Value *NarrowAbs =
          Builder.CreateBinaryIntrinsic(Intrinsic::abs, X, Builder.getFalse());
      return CastInst::Create(Instruction::ZExt, NarrowAbs, II->getType());
    }

    // Match a complicated way to check if a number is odd/even:
    // abs (srem X, 2) --> and X, 1
    const APInt *C;
    if (match(IIOperand, m_SRem(m_Value(X), m_APInt(C))) && *C == 2)
      return BinaryOperator::CreateAnd(X, ConstantInt::get(II->getType(), 1));

    break;
  }
  case Intrinsic::umin: {
    Value *I0 = II->getArgOperand(0), *I1 = II->getArgOperand(1);
    // umin(x, 1) == zext(x != 0)
    if (match(I1, m_One())) {
      Value *Zero = Constant::getNullValue(I0->getType());
      Value *Cmp = Builder.CreateICmpNE(I0, Zero);
      return CastInst::Create(Instruction::ZExt, Cmp, II->getType());
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::umax: {
    Value *I0 = II->getArgOperand(0), *I1 = II->getArgOperand(1);
    Value *X, *Y;
    if (match(I0, m_ZExt(m_Value(X))) && match(I1, m_ZExt(m_Value(Y))) &&
        (I0->hasOneUse() || I1->hasOneUse()) && X->getType() == Y->getType()) {
      Value *NarrowMaxMin = Builder.CreateBinaryIntrinsic(IID, X, Y);
      return CastInst::Create(Instruction::ZExt, NarrowMaxMin, II->getType());
    }
    Constant *C;
    if (match(I0, m_ZExt(m_Value(X))) && match(I1, m_Constant(C)) &&
        I0->hasOneUse()) {
      Constant *NarrowC = ConstantExpr::getTrunc(C, X->getType());
      if (ConstantExpr::getZExt(NarrowC, II->getType()) == C) {
        Value *NarrowMaxMin = Builder.CreateBinaryIntrinsic(IID, X, NarrowC);
        return CastInst::Create(Instruction::ZExt, NarrowMaxMin, II->getType());
      }
    }
    // If both operands of unsigned min/max are sign-extended, it is still ok
    // to narrow the operation.
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::smax:
  case Intrinsic::smin: {
    Value *I0 = II->getArgOperand(0), *I1 = II->getArgOperand(1);
    Value *X, *Y;
    if (match(I0, m_SExt(m_Value(X))) && match(I1, m_SExt(m_Value(Y))) &&
        (I0->hasOneUse() || I1->hasOneUse()) && X->getType() == Y->getType()) {
      Value *NarrowMaxMin = Builder.CreateBinaryIntrinsic(IID, X, Y);
      return CastInst::Create(Instruction::SExt, NarrowMaxMin, II->getType());
    }

    Constant *C;
    if (match(I0, m_SExt(m_Value(X))) && match(I1, m_Constant(C)) &&
        I0->hasOneUse()) {
      Constant *NarrowC = ConstantExpr::getTrunc(C, X->getType());
      if (ConstantExpr::getSExt(NarrowC, II->getType()) == C) {
        Value *NarrowMaxMin = Builder.CreateBinaryIntrinsic(IID, X, NarrowC);
        return CastInst::Create(Instruction::SExt, NarrowMaxMin, II->getType());
      }
    }

    if (IID == Intrinsic::smax || IID == Intrinsic::smin) {
      // smax (neg nsw X), (neg nsw Y) --> neg nsw (smin X, Y)
      // smin (neg nsw X), (neg nsw Y) --> neg nsw (smax X, Y)
      // TODO: Canonicalize neg after min/max if I1 is constant.
      if (match(I0, m_NSWNeg(m_Value(X))) && match(I1, m_NSWNeg(m_Value(Y))) &&
          (I0->hasOneUse() || I1->hasOneUse())) {
        Intrinsic::ID InvID = getInverseMinMaxIntrinsic(IID);
        Value *InvMaxMin = Builder.CreateBinaryIntrinsic(InvID, X, Y);
        return BinaryOperator::CreateNSWNeg(InvMaxMin);
      }
    }

    // If we can eliminate ~A and Y is free to invert:
    // max ~A, Y --> ~(min A, ~Y)
    //
    // Examples:
    // max ~A, ~Y --> ~(min A, Y)
    // max ~A, C --> ~(min A, ~C)
    // max ~A, (max ~Y, ~Z) --> ~min( A, (min Y, Z))
    auto moveNotAfterMinMax = [&](Value *X, Value *Y) -> Instruction * {
      Value *A;
      if (match(X, m_OneUse(m_Not(m_Value(A)))) &&
          !isFreeToInvert(A, A->hasOneUse()) &&
          isFreeToInvert(Y, Y->hasOneUse())) {
        Value *NotY = Builder.CreateNot(Y);
        Intrinsic::ID InvID = getInverseMinMaxIntrinsic(IID);
        Value *InvMaxMin = Builder.CreateBinaryIntrinsic(InvID, A, NotY);
        return BinaryOperator::CreateNot(InvMaxMin);
      }
      return nullptr;
    };

    if (Instruction *I = moveNotAfterMinMax(I0, I1))
      return I;
    if (Instruction *I = moveNotAfterMinMax(I1, I0))
      return I;

    if (Instruction *I = moveAddAfterMinMax(II, Builder))
      return I;

    // smax(X, -X) --> abs(X)
    // smin(X, -X) --> -abs(X)
    // umax(X, -X) --> -abs(X)
    // umin(X, -X) --> abs(X)
    if (isKnownNegation(I0, I1)) {
      // We can choose either operand as the input to abs(), but if we can
      // eliminate the only use of a value, that's better for subsequent
      // transforms/analysis.
      if (I0->hasOneUse() && !I1->hasOneUse())
        std::swap(I0, I1);

      // This is some variant of abs(). See if we can propagate 'nsw' to the abs
      // operation and potentially its negation.
      bool IntMinIsPoison = isKnownNegation(I0, I1, /* NeedNSW */ true);
      Value *Abs = Builder.CreateBinaryIntrinsic(
          Intrinsic::abs, I0,
          ConstantInt::getBool(II->getContext(), IntMinIsPoison));

      // We don't have a "nabs" intrinsic, so negate if needed based on the
      // max/min operation.
      if (IID == Intrinsic::smin || IID == Intrinsic::umax)
        Abs = Builder.CreateNeg(Abs, "nabs", /* NUW */ false, IntMinIsPoison);
      return replaceInstUsesWith(CI, Abs);
    }

    if (Instruction *Sel = foldClampRangeOfTwo(II, Builder))
      return Sel;

    if (Instruction *SAdd = matchSAddSubSat(*II))
      return SAdd;

    if (match(I1, m_ImmConstant()))
      if (auto *Sel = dyn_cast<SelectInst>(I0))
        if (Instruction *R = FoldOpIntoSelect(*II, Sel))
          return R;

    if (Instruction *NewMinMax = reassociateMinMaxWithConstants(II))
      return NewMinMax;

    if (Instruction *R = reassociateMinMaxWithConstantInOperand(II, Builder))
      return R;

    if (Instruction *NewMinMax = factorizeMinMaxTree(II))
       return NewMinMax;

    break;
  }
  case Intrinsic::bswap: {
    Value *IIOperand = II->getArgOperand(0);

    // Try to canonicalize bswap-of-logical-shift-by-8-bit-multiple as
    // inverse-shift-of-bswap:
    // bswap (shl X, Y) --> lshr (bswap X), Y
    // bswap (lshr X, Y) --> shl (bswap X), Y
    Value *X, *Y;
    if (match(IIOperand, m_OneUse(m_LogicalShift(m_Value(X), m_Value(Y))))) {
      // The transform allows undef vector elements, so try a constant match
      // first. If knownbits can handle that case, that clause could be removed.
      unsigned BitWidth = IIOperand->getType()->getScalarSizeInBits();
      const APInt *C;
      if ((match(Y, m_APIntAllowUndef(C)) && (*C & 7) == 0) ||
          MaskedValueIsZero(Y, APInt::getLowBitsSet(BitWidth, 3))) {
        Value *NewSwap = Builder.CreateUnaryIntrinsic(Intrinsic::bswap, X);
        BinaryOperator::BinaryOps InverseShift =
            cast<BinaryOperator>(IIOperand)->getOpcode() == Instruction::Shl
                ? Instruction::LShr
                : Instruction::Shl;
        return BinaryOperator::Create(InverseShift, NewSwap, Y);
      }
    }

    KnownBits Known = computeKnownBits(IIOperand, 0, II);
    uint64_t LZ = alignDown(Known.countMinLeadingZeros(), 8);
    uint64_t TZ = alignDown(Known.countMinTrailingZeros(), 8);
    unsigned BW = Known.getBitWidth();

    // bswap(x) -> shift(x) if x has exactly one "active byte"
    if (BW - LZ - TZ == 8) {
      assert(LZ != TZ && "active byte cannot be in the middle");
      if (LZ > TZ)  // -> shl(x) if the "active byte" is in the low part of x
        return BinaryOperator::CreateNUWShl(
            IIOperand, ConstantInt::get(IIOperand->getType(), LZ - TZ));
      // -> lshr(x) if the "active byte" is in the high part of x
      return BinaryOperator::CreateExactLShr(
            IIOperand, ConstantInt::get(IIOperand->getType(), TZ - LZ));
    }

    // bswap(trunc(bswap(x))) -> trunc(lshr(x, c))
    if (match(IIOperand, m_Trunc(m_BSwap(m_Value(X))))) {
      unsigned C = X->getType()->getScalarSizeInBits() - BW;
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
        return BinaryOperator::CreateFDivFMF(ConstantFP::get(CI.getType(), 1.0),
                                             II->getArgOperand(0), II);
      // powi(x, 2) -> x*x
      if (Power->equalsInt(2))
        return BinaryOperator::CreateFMulFMF(II->getArgOperand(0),
                                             II->getArgOperand(0), II);

      if (!Power->getValue()[0]) {
        Value *X;
        // If power is even:
        // powi(-x, p) -> powi(x, p)
        // powi(fabs(x), p) -> powi(x, p)
        // powi(copysign(x, y), p) -> powi(x, p)
        if (match(II->getArgOperand(0), m_FNeg(m_Value(X))) ||
            match(II->getArgOperand(0), m_FAbs(m_Value(X))) ||
            match(II->getArgOperand(0),
                  m_Intrinsic<Intrinsic::copysign>(m_Value(X), m_Value())))
          return replaceOperand(*II, 0, X);
      }
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
    if (match(II->getArgOperand(2), m_ImmConstant(ShAmtC)) &&
        !ShAmtC->containsConstantExpression()) {
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

    // m((fpext X), (fpext Y)) -> fpext (m(X, Y))
    if (match(Arg0, m_OneUse(m_FPExt(m_Value(X)))) &&
        match(Arg1, m_OneUse(m_FPExt(m_Value(Y)))) &&
        X->getType() == Y->getType()) {
      Value *NewCall =
          Builder.CreateBinaryIntrinsic(IID, X, Y, II, II->getName());
      return new FPExtInst(NewCall, II->getType());
    }

    // max X, -X --> fabs X
    // min X, -X --> -(fabs X)
    // TODO: Remove one-use limitation? That is obviously better for max.
    //       It would be an extra instruction for min (fnabs), but that is
    //       still likely better for analysis and codegen.
    if ((match(Arg0, m_OneUse(m_FNeg(m_Value(X)))) && Arg1 == X) ||
        (match(Arg1, m_OneUse(m_FNeg(m_Value(X)))) && Arg0 == X)) {
      Value *R = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, X, II);
      if (IID == Intrinsic::minimum || IID == Intrinsic::minnum)
        R = Builder.CreateFNegFMF(R, II);
      return replaceInstUsesWith(*II, R);
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
    if (Value *V = simplifyFMulInst(II->getArgOperand(0), II->getArgOperand(1),
                                    II->getFastMathFlags(),
                                    SQ.getWithInstruction(II))) {
      auto *FAdd = BinaryOperator::CreateFAdd(V, II->getArgOperand(2));
      FAdd->copyFastMathFlags(II);
      return FAdd;
    }

    LLVM_FALLTHROUGH;
  }
  case Intrinsic::fma: {
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
    if (Value *V = simplifyFMAFMul(II->getArgOperand(0), II->getArgOperand(1),
                                   II->getFastMathFlags(),
                                   SQ.getWithInstruction(II))) {
      auto *FAdd = BinaryOperator::CreateFAdd(V, II->getArgOperand(2));
      FAdd->copyFastMathFlags(II);
      return FAdd;
    }

    // fma x, y, 0 -> fmul x, y
    // This is always valid for -0.0, but requires nsz for +0.0 as
    // -0.0 + 0.0 = 0.0, which would not be the same as the fmul on its own.
    if (match(II->getArgOperand(2), m_NegZeroFP()) ||
        (match(II->getArgOperand(2), m_PosZeroFP()) &&
         II->getFastMathFlags().noSignedZeros()))
      return BinaryOperator::CreateFMulFMF(Src0, Src1, II);

    break;
  }
  case Intrinsic::copysign: {
    Value *Mag = II->getArgOperand(0), *Sign = II->getArgOperand(1);
    if (SignBitMustBeZero(Sign, &TLI)) {
      // If we know that the sign argument is positive, reduce to FABS:
      // copysign Mag, +Sign --> fabs Mag
      Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, Mag, II);
      return replaceInstUsesWith(*II, Fabs);
    }
    // TODO: There should be a ValueTracking sibling like SignBitMustBeOne.
    const APFloat *C;
    if (match(Sign, m_APFloat(C)) && C->isNegative()) {
      // If we know that the sign argument is negative, reduce to FNABS:
      // copysign Mag, -Sign --> fneg (fabs Mag)
      Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, Mag, II);
      return replaceInstUsesWith(*II, Builder.CreateFNegFMF(Fabs, II));
    }

    // Propagate sign argument through nested calls:
    // copysign Mag, (copysign ?, X) --> copysign Mag, X
    Value *X;
    if (match(Sign, m_Intrinsic<Intrinsic::copysign>(m_Value(), m_Value(X))))
      return replaceOperand(*II, 1, X);

    // Peek through changes of magnitude's sign-bit. This call rewrites those:
    // copysign (fabs X), Sign --> copysign X, Sign
    // copysign (fneg X), Sign --> copysign X, Sign
    if (match(Mag, m_FAbs(m_Value(X))) || match(Mag, m_FNeg(m_Value(X))))
      return replaceOperand(*II, 0, X);

    break;
  }
  case Intrinsic::fabs: {
    Value *Cond, *TVal, *FVal;
    if (match(II->getArgOperand(0),
              m_Select(m_Value(Cond), m_Value(TVal), m_Value(FVal)))) {
      // fabs (select Cond, TrueC, FalseC) --> select Cond, AbsT, AbsF
      if (isa<Constant>(TVal) && isa<Constant>(FVal)) {
        CallInst *AbsT = Builder.CreateCall(II->getCalledFunction(), {TVal});
        CallInst *AbsF = Builder.CreateCall(II->getCalledFunction(), {FVal});
        return SelectInst::Create(Cond, AbsT, AbsF);
      }
      // fabs (select Cond, -FVal, FVal) --> fabs FVal
      if (match(TVal, m_FNeg(m_Specific(FVal))))
        return replaceOperand(*II, 0, FVal);
      // fabs (select Cond, TVal, -TVal) --> fabs TVal
      if (match(FVal, m_FNeg(m_Specific(TVal))))
        return replaceOperand(*II, 0, TVal);
    }

    LLVM_FALLTHROUGH;
  }
  case Intrinsic::ceil:
  case Intrinsic::floor:
  case Intrinsic::round:
  case Intrinsic::roundeven:
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
    enum class ClassifyResult {
      None,
      Alloca,
      StackRestore,
      CallWithSideEffects,
    };
    auto Classify = [](const Instruction *I) {
      if (isa<AllocaInst>(I))
        return ClassifyResult::Alloca;

      if (auto *CI = dyn_cast<CallInst>(I)) {
        if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
          if (II->getIntrinsicID() == Intrinsic::stackrestore)
            return ClassifyResult::StackRestore;

          if (II->mayHaveSideEffects())
            return ClassifyResult::CallWithSideEffects;
        } else {
          // Consider all non-intrinsic calls to be side effects
          return ClassifyResult::CallWithSideEffects;
        }
      }

      return ClassifyResult::None;
    };

    // If the stacksave and the stackrestore are in the same BB, and there is
    // no intervening call, alloca, or stackrestore of a different stacksave,
    // remove the restore. This can happen when variable allocas are DCE'd.
    if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getArgOperand(0))) {
      if (SS->getIntrinsicID() == Intrinsic::stacksave &&
          SS->getParent() == II->getParent()) {
        BasicBlock::iterator BI(SS);
        bool CannotRemove = false;
        for (++BI; &*BI != II; ++BI) {
          switch (Classify(&*BI)) {
          case ClassifyResult::None:
            // So far so good, look at next instructions.
            break;

          case ClassifyResult::StackRestore:
            // If we found an intervening stackrestore for a different
            // stacksave, we can't remove the stackrestore. Otherwise, continue.
            if (cast<IntrinsicInst>(*BI).getArgOperand(0) != SS)
              CannotRemove = true;
            break;

          case ClassifyResult::Alloca:
          case ClassifyResult::CallWithSideEffects:
            // If we found an alloca, a non-intrinsic call, or an intrinsic
            // call with side effects, we can't remove the stackrestore.
            CannotRemove = true;
            break;
          }
          if (CannotRemove)
            break;
        }

        if (!CannotRemove)
          return eraseInstFromFunction(CI);
      }
    }

    // Scan down this block to see if there is another stack restore in the
    // same block without an intervening call/alloca.
    BasicBlock::iterator BI(II);
    Instruction *TI = II->getParent()->getTerminator();
    bool CannotRemove = false;
    for (++BI; &*BI != TI; ++BI) {
      switch (Classify(&*BI)) {
      case ClassifyResult::None:
        // So far so good, look at next instructions.
        break;

      case ClassifyResult::StackRestore:
        // If there is a stackrestore below this one, remove this one.
        return eraseInstFromFunction(CI);

      case ClassifyResult::Alloca:
      case ClassifyResult::CallWithSideEffects:
        // If we found an alloca, a non-intrinsic call, or an intrinsic call
        // with side effects (such as llvm.stacksave and llvm.read_register),
        // we can't remove the stack restore.
        CannotRemove = true;
        break;
      }
      if (CannotRemove)
        break;
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
    SmallVector<OperandBundleDef, 4> OpBundles;
    II->getOperandBundlesAsDefs(OpBundles);

    /// This will remove the boolean Condition from the assume given as
    /// argument and remove the assume if it becomes useless.
    /// always returns nullptr for use as a return values.
    auto RemoveConditionFromAssume = [&](Instruction *Assume) -> Instruction * {
      assert(isa<AssumeInst>(Assume));
      if (isAssumeWithEmptyBundle(*cast<AssumeInst>(II)))
        return eraseInstFromFunction(CI);
      replaceUse(II->getOperandUse(0), ConstantInt::getTrue(II->getContext()));
      return nullptr;
    };
    // Remove an assume if it is followed by an identical assume.
    // TODO: Do we need this? Unless there are conflicting assumptions, the
    // computeKnownBits(IIOperand) below here eliminates redundant assumes.
    Instruction *Next = II->getNextNonDebugInstruction();
    if (match(Next, m_Intrinsic<Intrinsic::assume>(m_Specific(IIOperand))))
      return RemoveConditionFromAssume(Next);

    // Canonicalize assume(a && b) -> assume(a); assume(b);
    // Note: New assumption intrinsics created here are registered by
    // the InstCombineIRInserter object.
    FunctionType *AssumeIntrinsicTy = II->getFunctionType();
    Value *AssumeIntrinsic = II->getCalledOperand();
    Value *A, *B;
    if (match(IIOperand, m_LogicalAnd(m_Value(A), m_Value(B)))) {
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic, A, OpBundles,
                         II->getName());
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic, B, II->getName());
      return eraseInstFromFunction(*II);
    }
    // assume(!(a || b)) -> assume(!a); assume(!b);
    if (match(IIOperand, m_Not(m_LogicalOr(m_Value(A), m_Value(B))))) {
      Builder.CreateCall(AssumeIntrinsicTy, AssumeIntrinsic,
                         Builder.CreateNot(A), OpBundles, II->getName());
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
      return RemoveConditionFromAssume(II);

      // TODO: apply nonnull return attributes to calls and invokes
      // TODO: apply range metadata for range check patterns?
    }

    // Convert nonnull assume like:
    // %A = icmp ne i32* %PTR, null
    // call void @llvm.assume(i1 %A)
    // into
    // call void @llvm.assume(i1 true) [ "nonnull"(i32* %PTR) ]
    if (EnableKnowledgeRetention &&
        match(IIOperand, m_Cmp(Pred, m_Value(A), m_Zero())) &&
        Pred == CmpInst::ICMP_NE && A->getType()->isPointerTy()) {
      if (auto *Replacement = buildAssumeFromKnowledge(
              {RetainedKnowledge{Attribute::NonNull, 0, A}}, Next, &AC, &DT)) {

        Replacement->insertBefore(Next);
        AC.registerAssumption(Replacement);
        return RemoveConditionFromAssume(II);
      }
    }

    // Convert alignment assume like:
    // %B = ptrtoint i32* %A to i64
    // %C = and i64 %B, Constant
    // %D = icmp eq i64 %C, 0
    // call void @llvm.assume(i1 %D)
    // into
    // call void @llvm.assume(i1 true) [ "align"(i32* [[A]], i64  Constant + 1)]
    uint64_t AlignMask;
    if (EnableKnowledgeRetention &&
        match(IIOperand,
              m_Cmp(Pred, m_And(m_Value(A), m_ConstantInt(AlignMask)),
                    m_Zero())) &&
        Pred == CmpInst::ICMP_EQ) {
      if (isPowerOf2_64(AlignMask + 1)) {
        uint64_t Offset = 0;
        match(A, m_Add(m_Value(A), m_ConstantInt(Offset)));
        if (match(A, m_PtrToInt(m_Value(A)))) {
          /// Note: this doesn't preserve the offset information but merges
          /// offset and alignment.
          /// TODO: we can generate a GEP instead of merging the alignment with
          /// the offset.
          RetainedKnowledge RK{Attribute::Alignment,
                               (unsigned)MinAlign(Offset, AlignMask + 1), A};
          if (auto *Replacement =
                  buildAssumeFromKnowledge(RK, Next, &AC, &DT)) {

            Replacement->insertAfter(II);
            AC.registerAssumption(Replacement);
          }
          return RemoveConditionFromAssume(II);
        }
      }
    }

    /// Canonicalize Knowledge in operand bundles.
    if (EnableKnowledgeRetention && II->hasOperandBundles()) {
      for (unsigned Idx = 0; Idx < II->getNumOperandBundles(); Idx++) {
        auto &BOI = II->bundle_op_info_begin()[Idx];
        RetainedKnowledge RK =
          llvm::getKnowledgeFromBundle(cast<AssumeInst>(*II), BOI);
        if (BOI.End - BOI.Begin > 2)
          continue; // Prevent reducing knowledge in an align with offset since
                    // extracting a RetainedKnowledge form them looses offset
                    // information
        RetainedKnowledge CanonRK =
          llvm::simplifyRetainedKnowledge(cast<AssumeInst>(II), RK,
                                          &getAssumptionCache(),
                                          &getDominatorTree());
        if (CanonRK == RK)
          continue;
        if (!CanonRK) {
          if (BOI.End - BOI.Begin > 0) {
            Worklist.pushValue(II->op_begin()[BOI.Begin]);
            Value::dropDroppableUse(II->op_begin()[BOI.Begin]);
          }
          continue;
        }
        assert(RK.AttrKind == CanonRK.AttrKind);
        if (BOI.End - BOI.Begin > 0)
          II->op_begin()[BOI.Begin].set(CanonRK.WasOn);
        if (BOI.End - BOI.Begin > 1)
          II->op_begin()[BOI.Begin + 1].set(ConstantInt::get(
              Type::getInt64Ty(II->getContext()), CanonRK.ArgValue));
        if (RK.WasOn)
          Worklist.pushValue(RK.WasOn);
        return II;
      }
    }

    // If there is a dominating assume with the same condition as this one,
    // then this one is redundant, and should be removed.
    KnownBits Known(1);
    computeKnownBits(IIOperand, Known, 0, II);
    if (Known.isAllOnes() && isAssumeWithEmptyBundle(cast<AssumeInst>(*II)))
      return eraseInstFromFunction(*II);

    // Update the cache of affected values for this assumption (we might be
    // here because we just simplified the condition).
    AC.updateAffectedValues(cast<AssumeInst>(II));
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
  case Intrinsic::vector_insert: {
    Value *Vec = II->getArgOperand(0);
    Value *SubVec = II->getArgOperand(1);
    Value *Idx = II->getArgOperand(2);
    auto *DstTy = dyn_cast<FixedVectorType>(II->getType());
    auto *VecTy = dyn_cast<FixedVectorType>(Vec->getType());
    auto *SubVecTy = dyn_cast<FixedVectorType>(SubVec->getType());

    // Only canonicalize if the destination vector, Vec, and SubVec are all
    // fixed vectors.
    if (DstTy && VecTy && SubVecTy) {
      unsigned DstNumElts = DstTy->getNumElements();
      unsigned VecNumElts = VecTy->getNumElements();
      unsigned SubVecNumElts = SubVecTy->getNumElements();
      unsigned IdxN = cast<ConstantInt>(Idx)->getZExtValue();

      // An insert that entirely overwrites Vec with SubVec is a nop.
      if (VecNumElts == SubVecNumElts)
        return replaceInstUsesWith(CI, SubVec);

      // Widen SubVec into a vector of the same width as Vec, since
      // shufflevector requires the two input vectors to be the same width.
      // Elements beyond the bounds of SubVec within the widened vector are
      // undefined.
      SmallVector<int, 8> WidenMask;
      unsigned i;
      for (i = 0; i != SubVecNumElts; ++i)
        WidenMask.push_back(i);
      for (; i != VecNumElts; ++i)
        WidenMask.push_back(UndefMaskElem);

      Value *WidenShuffle = Builder.CreateShuffleVector(SubVec, WidenMask);

      SmallVector<int, 8> Mask;
      for (unsigned i = 0; i != IdxN; ++i)
        Mask.push_back(i);
      for (unsigned i = DstNumElts; i != DstNumElts + SubVecNumElts; ++i)
        Mask.push_back(i);
      for (unsigned i = IdxN + SubVecNumElts; i != DstNumElts; ++i)
        Mask.push_back(i);

      Value *Shuffle = Builder.CreateShuffleVector(Vec, WidenShuffle, Mask);
      return replaceInstUsesWith(CI, Shuffle);
    }
    break;
  }
  case Intrinsic::vector_extract: {
    Value *Vec = II->getArgOperand(0);
    Value *Idx = II->getArgOperand(1);

    auto *DstTy = dyn_cast<FixedVectorType>(II->getType());
    auto *VecTy = dyn_cast<FixedVectorType>(Vec->getType());

    // Only canonicalize if the the destination vector and Vec are fixed
    // vectors.
    if (DstTy && VecTy) {
      unsigned DstNumElts = DstTy->getNumElements();
      unsigned VecNumElts = VecTy->getNumElements();
      unsigned IdxN = cast<ConstantInt>(Idx)->getZExtValue();

      // Extracting the entirety of Vec is a nop.
      if (VecNumElts == DstNumElts) {
        replaceInstUsesWith(CI, Vec);
        return eraseInstFromFunction(CI);
      }

      SmallVector<int, 8> Mask;
      for (unsigned i = 0; i != DstNumElts; ++i)
        Mask.push_back(IdxN + i);

      Value *Shuffle = Builder.CreateShuffleVector(Vec, Mask);
      return replaceInstUsesWith(CI, Shuffle);
    }
    break;
  }
  case Intrinsic::experimental_vector_reverse: {
    Value *BO0, *BO1, *X, *Y;
    Value *Vec = II->getArgOperand(0);
    if (match(Vec, m_OneUse(m_BinOp(m_Value(BO0), m_Value(BO1))))) {
      auto *OldBinOp = cast<BinaryOperator>(Vec);
      if (match(BO0, m_Intrinsic<Intrinsic::experimental_vector_reverse>(
                         m_Value(X)))) {
        // rev(binop rev(X), rev(Y)) --> binop X, Y
        if (match(BO1, m_Intrinsic<Intrinsic::experimental_vector_reverse>(
                           m_Value(Y))))
          return replaceInstUsesWith(CI,
                                     BinaryOperator::CreateWithCopiedFlags(
                                         OldBinOp->getOpcode(), X, Y, OldBinOp,
                                         OldBinOp->getName(), II));
        // rev(binop rev(X), BO1Splat) --> binop X, BO1Splat
        if (isSplatValue(BO1))
          return replaceInstUsesWith(CI,
                                     BinaryOperator::CreateWithCopiedFlags(
                                         OldBinOp->getOpcode(), X, BO1,
                                         OldBinOp, OldBinOp->getName(), II));
      }
      // rev(binop BO0Splat, rev(Y)) --> binop BO0Splat, Y
      if (match(BO1, m_Intrinsic<Intrinsic::experimental_vector_reverse>(
                         m_Value(Y))) &&
          isSplatValue(BO0))
        return replaceInstUsesWith(CI, BinaryOperator::CreateWithCopiedFlags(
                                           OldBinOp->getOpcode(), BO0, Y,
                                           OldBinOp, OldBinOp->getName(), II));
    }
    // rev(unop rev(X)) --> unop X
    if (match(Vec, m_OneUse(m_UnOp(
                       m_Intrinsic<Intrinsic::experimental_vector_reverse>(
                           m_Value(X)))))) {
      auto *OldUnOp = cast<UnaryOperator>(Vec);
      auto *NewUnOp = UnaryOperator::CreateWithCopiedFlags(
          OldUnOp->getOpcode(), X, OldUnOp, OldUnOp->getName(), II);
      return replaceInstUsesWith(CI, NewUnOp);
    }
    break;
  }
  case Intrinsic::vector_reduce_or:
  case Intrinsic::vector_reduce_and: {
    // Canonicalize logical or/and reductions:
    // Or reduction for i1 is represented as:
    // %val = bitcast <ReduxWidth x i1> to iReduxWidth
    // %res = cmp ne iReduxWidth %val, 0
    // And reduction for i1 is represented as:
    // %val = bitcast <ReduxWidth x i1> to iReduxWidth
    // %res = cmp eq iReduxWidth %val, 11111
    Value *Arg = II->getArgOperand(0);
    Value *Vect;
    if (match(Arg, m_ZExtOrSExtOrSelf(m_Value(Vect)))) {
      if (auto *FTy = dyn_cast<FixedVectorType>(Vect->getType()))
        if (FTy->getElementType() == Builder.getInt1Ty()) {
          Value *Res = Builder.CreateBitCast(
              Vect, Builder.getIntNTy(FTy->getNumElements()));
          if (IID == Intrinsic::vector_reduce_and) {
            Res = Builder.CreateICmpEQ(
                Res, ConstantInt::getAllOnesValue(Res->getType()));
          } else {
            assert(IID == Intrinsic::vector_reduce_or &&
                   "Expected or reduction.");
            Res = Builder.CreateIsNotNull(Res);
          }
          if (Arg != Vect)
            Res = Builder.CreateCast(cast<CastInst>(Arg)->getOpcode(), Res,
                                     II->getType());
          return replaceInstUsesWith(CI, Res);
        }
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::vector_reduce_add: {
    if (IID == Intrinsic::vector_reduce_add) {
      // Convert vector_reduce_add(ZExt(<n x i1>)) to
      // ZExtOrTrunc(ctpop(bitcast <n x i1> to in)).
      // Convert vector_reduce_add(SExt(<n x i1>)) to
      // -ZExtOrTrunc(ctpop(bitcast <n x i1> to in)).
      // Convert vector_reduce_add(<n x i1>) to
      // Trunc(ctpop(bitcast <n x i1> to in)).
      Value *Arg = II->getArgOperand(0);
      Value *Vect;
      if (match(Arg, m_ZExtOrSExtOrSelf(m_Value(Vect)))) {
        if (auto *FTy = dyn_cast<FixedVectorType>(Vect->getType()))
          if (FTy->getElementType() == Builder.getInt1Ty()) {
            Value *V = Builder.CreateBitCast(
                Vect, Builder.getIntNTy(FTy->getNumElements()));
            Value *Res = Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, V);
            if (Res->getType() != II->getType())
              Res = Builder.CreateZExtOrTrunc(Res, II->getType());
            if (Arg != Vect &&
                cast<Instruction>(Arg)->getOpcode() == Instruction::SExt)
              Res = Builder.CreateNeg(Res);
            return replaceInstUsesWith(CI, Res);
          }
      }
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::vector_reduce_xor: {
    if (IID == Intrinsic::vector_reduce_xor) {
      // Exclusive disjunction reduction over the vector with
      // (potentially-extended) i1 element type is actually a
      // (potentially-extended) arithmetic `add` reduction over the original
      // non-extended value:
      //   vector_reduce_xor(?ext(<n x i1>))
      //     -->
      //   ?ext(vector_reduce_add(<n x i1>))
      Value *Arg = II->getArgOperand(0);
      Value *Vect;
      if (match(Arg, m_ZExtOrSExtOrSelf(m_Value(Vect)))) {
        if (auto *FTy = dyn_cast<FixedVectorType>(Vect->getType()))
          if (FTy->getElementType() == Builder.getInt1Ty()) {
            Value *Res = Builder.CreateAddReduce(Vect);
            if (Arg != Vect)
              Res = Builder.CreateCast(cast<CastInst>(Arg)->getOpcode(), Res,
                                       II->getType());
            return replaceInstUsesWith(CI, Res);
          }
      }
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::vector_reduce_mul: {
    if (IID == Intrinsic::vector_reduce_mul) {
      // Multiplicative reduction over the vector with (potentially-extended)
      // i1 element type is actually a (potentially zero-extended)
      // logical `and` reduction over the original non-extended value:
      //   vector_reduce_mul(?ext(<n x i1>))
      //     -->
      //   zext(vector_reduce_and(<n x i1>))
      Value *Arg = II->getArgOperand(0);
      Value *Vect;
      if (match(Arg, m_ZExtOrSExtOrSelf(m_Value(Vect)))) {
        if (auto *FTy = dyn_cast<FixedVectorType>(Vect->getType()))
          if (FTy->getElementType() == Builder.getInt1Ty()) {
            Value *Res = Builder.CreateAndReduce(Vect);
            if (Res->getType() != II->getType())
              Res = Builder.CreateZExt(Res, II->getType());
            return replaceInstUsesWith(CI, Res);
          }
      }
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::vector_reduce_umin:
  case Intrinsic::vector_reduce_umax: {
    if (IID == Intrinsic::vector_reduce_umin ||
        IID == Intrinsic::vector_reduce_umax) {
      // UMin/UMax reduction over the vector with (potentially-extended)
      // i1 element type is actually a (potentially-extended)
      // logical `and`/`or` reduction over the original non-extended value:
      //   vector_reduce_u{min,max}(?ext(<n x i1>))
      //     -->
      //   ?ext(vector_reduce_{and,or}(<n x i1>))
      Value *Arg = II->getArgOperand(0);
      Value *Vect;
      if (match(Arg, m_ZExtOrSExtOrSelf(m_Value(Vect)))) {
        if (auto *FTy = dyn_cast<FixedVectorType>(Vect->getType()))
          if (FTy->getElementType() == Builder.getInt1Ty()) {
            Value *Res = IID == Intrinsic::vector_reduce_umin
                             ? Builder.CreateAndReduce(Vect)
                             : Builder.CreateOrReduce(Vect);
            if (Arg != Vect)
              Res = Builder.CreateCast(cast<CastInst>(Arg)->getOpcode(), Res,
                                       II->getType());
            return replaceInstUsesWith(CI, Res);
          }
      }
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_smax: {
    if (IID == Intrinsic::vector_reduce_smin ||
        IID == Intrinsic::vector_reduce_smax) {
      // SMin/SMax reduction over the vector with (potentially-extended)
      // i1 element type is actually a (potentially-extended)
      // logical `and`/`or` reduction over the original non-extended value:
      //   vector_reduce_s{min,max}(<n x i1>)
      //     -->
      //   vector_reduce_{or,and}(<n x i1>)
      // and
      //   vector_reduce_s{min,max}(sext(<n x i1>))
      //     -->
      //   sext(vector_reduce_{or,and}(<n x i1>))
      // and
      //   vector_reduce_s{min,max}(zext(<n x i1>))
      //     -->
      //   zext(vector_reduce_{and,or}(<n x i1>))
      Value *Arg = II->getArgOperand(0);
      Value *Vect;
      if (match(Arg, m_ZExtOrSExtOrSelf(m_Value(Vect)))) {
        if (auto *FTy = dyn_cast<FixedVectorType>(Vect->getType()))
          if (FTy->getElementType() == Builder.getInt1Ty()) {
            Instruction::CastOps ExtOpc = Instruction::CastOps::CastOpsEnd;
            if (Arg != Vect)
              ExtOpc = cast<CastInst>(Arg)->getOpcode();
            Value *Res = ((IID == Intrinsic::vector_reduce_smin) ==
                          (ExtOpc == Instruction::CastOps::ZExt))
                             ? Builder.CreateAndReduce(Vect)
                             : Builder.CreateOrReduce(Vect);
            if (Arg != Vect)
              Res = Builder.CreateCast(ExtOpc, Res, II->getType());
            return replaceInstUsesWith(CI, Res);
          }
      }
    }
    LLVM_FALLTHROUGH;
  }
  case Intrinsic::vector_reduce_fmax:
  case Intrinsic::vector_reduce_fmin:
  case Intrinsic::vector_reduce_fadd:
  case Intrinsic::vector_reduce_fmul: {
    bool CanBeReassociated = (IID != Intrinsic::vector_reduce_fadd &&
                              IID != Intrinsic::vector_reduce_fmul) ||
                             II->hasAllowReassoc();
    const unsigned ArgIdx = (IID == Intrinsic::vector_reduce_fadd ||
                             IID == Intrinsic::vector_reduce_fmul)
                                ? 1
                                : 0;
    Value *Arg = II->getArgOperand(ArgIdx);
    Value *V;
    ArrayRef<int> Mask;
    if (!isa<FixedVectorType>(Arg->getType()) || !CanBeReassociated ||
        !match(Arg, m_Shuffle(m_Value(V), m_Undef(), m_Mask(Mask))) ||
        !cast<ShuffleVectorInst>(Arg)->isSingleSource())
      break;
    int Sz = Mask.size();
    SmallBitVector UsedIndices(Sz);
    for (int Idx : Mask) {
      if (Idx == UndefMaskElem || UsedIndices.test(Idx))
        break;
      UsedIndices.set(Idx);
    }
    // Can remove shuffle iff just shuffled elements, no repeats, undefs, or
    // other changes.
    if (UsedIndices.all()) {
      replaceUse(II->getOperandUse(ArgIdx), V);
      return nullptr;
    }
    break;
  }
  default: {
    // Handle target specific intrinsics
    Optional<Instruction *> V = targetInstCombineIntrinsic(*II);
    if (V)
      return V.getValue();
    break;
  }
  }

  if (Instruction *Shuf = foldShuffledIntrinsicOperands(II, Builder))
    return Shuf;

  // Some intrinsics (like experimental_gc_statepoint) can be used in invoke
  // context, so it is handled in visitCallBase and we should trigger it.
  return visitCallBase(*II);
}

// Fence instruction simplification
Instruction *InstCombinerImpl::visitFenceInst(FenceInst &FI) {
  auto *NFI = dyn_cast<FenceInst>(FI.getNextNonDebugInstruction());
  // This check is solely here to handle arbitrary target-dependent syncscopes.
  // TODO: Can remove if does not matter in practice.
  if (NFI && FI.isIdenticalTo(NFI))
    return eraseInstFromFunction(FI);

  // Returns true if FI1 is identical or stronger fence than FI2.
  auto isIdenticalOrStrongerFence = [](FenceInst *FI1, FenceInst *FI2) {
    auto FI1SyncScope = FI1->getSyncScopeID();
    // Consider same scope, where scope is global or single-thread.
    if (FI1SyncScope != FI2->getSyncScopeID() ||
        (FI1SyncScope != SyncScope::System &&
         FI1SyncScope != SyncScope::SingleThread))
      return false;

    return isAtLeastOrStrongerThan(FI1->getOrdering(), FI2->getOrdering());
  };
  if (NFI && isIdenticalOrStrongerFence(NFI, &FI))
    return eraseInstFromFunction(FI);

  if (auto *PFI = dyn_cast_or_null<FenceInst>(FI.getPrevNonDebugInstruction()))
    if (isIdenticalOrStrongerFence(PFI, &FI))
      return eraseInstFromFunction(FI);
  return nullptr;
}

// InvokeInst simplification
Instruction *InstCombinerImpl::visitInvokeInst(InvokeInst &II) {
  return visitCallBase(II);
}

// CallBrInst simplification
Instruction *InstCombinerImpl::visitCallBrInst(CallBrInst &CBI) {
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
  if (isa<GCStatepointInst>(Call) || isa<GCRelocateInst>(Call) ||
      isa<GCResultInst>(Call))
    return false;

  // Opaque pointers are compatible with any byval types.
  PointerType *SrcTy = cast<PointerType>(CI->getOperand(0)->getType());
  if (SrcTy->isOpaque())
    return true;

  // The size of ByVal or InAlloca arguments is derived from the type, so we
  // can't change to a type with a different size.  If the size were
  // passed explicitly we could avoid this check.
  if (!Call.isPassPointeeByValueArgument(ix))
    return true;

  // The transform currently only handles type replacement for byval, not other
  // type-carrying attributes.
  if (!Call.isByValArgument(ix))
    return false;

  Type *SrcElemTy = SrcTy->getNonOpaquePointerElementType();
  Type *DstElemTy = Call.getParamByValType(ix);
  if (!SrcElemTy->isSized() || !DstElemTy->isSized())
    return false;
  if (DL.getTypeAllocSize(SrcElemTy) != DL.getTypeAllocSize(DstElemTy))
    return false;
  return true;
}

Instruction *InstCombinerImpl::tryOptimizeCall(CallInst *CI) {
  if (!CI->getCalledFunction()) return nullptr;

  // Skip optimizing notail and musttail calls so
  // LibCallSimplifier::optimizeCall doesn't have to preserve those invariants.
  // LibCallSimplifier::optimizeCall should try to preseve tail calls though.
  if (CI->isMustTailCall() || CI->isNoTailCall())
    return nullptr;

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

bool InstCombinerImpl::annotateAnyAllocSite(CallBase &Call,
                                            const TargetLibraryInfo *TLI) {
  // Note: We only handle cases which can't be driven from generic attributes
  // here.  So, for example, nonnull and noalias (which are common properties
  // of some allocation functions) are expected to be handled via annotation
  // of the respective allocator declaration with generic attributes.
  bool Changed = false;

  if (isAllocationFn(&Call, TLI)) {
    uint64_t Size;
    ObjectSizeOpts Opts;
    if (getObjectSize(&Call, Size, DL, TLI, Opts) && Size > 0) {
      // TODO: We really should just emit deref_or_null here and then
      // let the generic inference code combine that with nonnull.
      if (Call.hasRetAttr(Attribute::NonNull)) {
        Changed = !Call.hasRetAttr(Attribute::Dereferenceable);
        Call.addRetAttr(
            Attribute::getWithDereferenceableBytes(Call.getContext(), Size));
      } else {
        Changed = !Call.hasRetAttr(Attribute::DereferenceableOrNull);
        Call.addRetAttr(Attribute::getWithDereferenceableOrNullBytes(
            Call.getContext(), Size));
      }
    }
  }

  // Add alignment attribute if alignment is a power of two constant.
  Value *Alignment = getAllocAlignment(&Call, TLI);
  if (!Alignment)
    return Changed;

  ConstantInt *AlignOpC = dyn_cast<ConstantInt>(Alignment);
  if (AlignOpC && AlignOpC->getValue().ult(llvm::Value::MaximumAlignment)) {
    uint64_t AlignmentVal = AlignOpC->getZExtValue();
    if (llvm::isPowerOf2_64(AlignmentVal)) {
      Align ExistingAlign = Call.getRetAlign().valueOrOne();
      Align NewAlign = Align(AlignmentVal);
      if (NewAlign > ExistingAlign) {
        Call.addRetAttr(
            Attribute::getWithAlignment(Call.getContext(), NewAlign));
        Changed = true;
      }
    }
  }
  return Changed;
}

/// Improvements for call, callbr and invoke instructions.
Instruction *InstCombinerImpl::visitCallBase(CallBase &Call) {
  bool Changed = annotateAnyAllocSite(Call, &TLI);

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

  assert(ArgNo == Call.arg_size() && "Call arguments not processed correctly.");

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
  Function *CalleeF = dyn_cast<Function>(Callee);
  if ((!CalleeF || CalleeF->getFunctionType() != Call.getFunctionType()) &&
      transformConstExprCastCall(Call))
    return nullptr;

  if (CalleeF) {
    // Remove the convergent attr on calls when the callee is not convergent.
    if (Call.isConvergent() && !CalleeF->isConvergent() &&
        !CalleeF->isIntrinsic()) {
      LLVM_DEBUG(dbgs() << "Removing convergent attr from instr " << Call
                        << "\n");
      Call.setNotConvergent();
      return &Call;
    }

    // If the call and callee calling conventions don't match, and neither one
    // of the calling conventions is compatible with C calling convention
    // this call must be unreachable, as the call is undefined.
    if ((CalleeF->getCallingConv() != Call.getCallingConv() &&
         !(CalleeF->getCallingConv() == llvm::CallingConv::C &&
           TargetLibraryInfoImpl::isCallingConvCCompatible(&Call)) &&
         !(Call.getCallingConv() == llvm::CallingConv::C &&
           TargetLibraryInfoImpl::isCallingConvCCompatible(CalleeF))) &&
        // Only do this for calls to a function with a body.  A prototype may
        // not actually end up matching the implementation's calling conv for a
        // variety of reasons (e.g. it may be written in assembly).
        !CalleeF->isDeclaration()) {
      Instruction *OldCall = &Call;
      CreateNonTerminatorUnreachable(OldCall);
      // If OldCall does not return void then replaceInstUsesWith poison.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!OldCall->getType()->isVoidTy())
        replaceInstUsesWith(*OldCall, PoisonValue::get(OldCall->getType()));
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

  // Calling a null function pointer is undefined if a null address isn't
  // dereferenceable.
  if ((isa<ConstantPointerNull>(Callee) &&
       !NullPointerIsDefined(Call.getFunction())) ||
      isa<UndefValue>(Callee)) {
    // If Call does not return void then replaceInstUsesWith poison.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!Call.getType()->isVoidTy())
      replaceInstUsesWith(Call, PoisonValue::get(Call.getType()));

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

  // TODO: Drop this transform once opaque pointer transition is done.
  FunctionType *FTy = Call.getFunctionType();
  if (FTy->isVarArg()) {
    int ix = FTy->getNumParams();
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (auto I = Call.arg_begin() + FTy->getNumParams(), E = Call.arg_end();
         I != E; ++I, ++ix) {
      CastInst *CI = dyn_cast<CastInst>(*I);
      if (CI && isSafeToEliminateVarargsCast(Call, DL, CI, ix)) {
        replaceUse(*I, CI->getOperand(0));

        // Update the byval type to match the pointer type.
        // Not necessary for opaque pointers.
        PointerType *NewTy = cast<PointerType>(CI->getOperand(0)->getType());
        if (!NewTy->isOpaque() && Call.isByValArgument(ix)) {
          Call.removeParamAttr(ix, Attribute::ByVal);
          Call.addParamAttr(ix, Attribute::getWithByValType(
                                    Call.getContext(),
                                    NewTy->getNonOpaquePointerElementType()));
        }
        Changed = true;
      }
    }
  }

  if (isa<InlineAsm>(Callee) && !Call.doesNotThrow()) {
    InlineAsm *IA = cast<InlineAsm>(Callee);
    if (!IA->canThrow()) {
      // Normal inline asm calls cannot throw - mark them
      // 'nounwind'.
      Call.setDoesNotThrow();
      Changed = true;
    }
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

  if (isAllocationFn(&Call, &TLI) &&
      isAllocRemovable(&cast<CallBase>(Call), &TLI))
    return visitAllocSite(Call);

  // Handle intrinsics which can be used in both call and invoke context.
  switch (Call.getIntrinsicID()) {
  case Intrinsic::experimental_gc_statepoint: {
    GCStatepointInst &GCSP = *cast<GCStatepointInst>(&Call);
    SmallPtrSet<Value *, 32> LiveGcValues;
    for (const GCRelocateInst *Reloc : GCSP.getGCRelocates()) {
      GCRelocateInst &GCR = *const_cast<GCRelocateInst *>(Reloc);

      // Remove the relocation if unused.
      if (GCR.use_empty()) {
        eraseInstFromFunction(GCR);
        continue;
      }

      Value *DerivedPtr = GCR.getDerivedPtr();
      Value *BasePtr = GCR.getBasePtr();

      // Undef is undef, even after relocation.
      if (isa<UndefValue>(DerivedPtr) || isa<UndefValue>(BasePtr)) {
        replaceInstUsesWith(GCR, UndefValue::get(GCR.getType()));
        eraseInstFromFunction(GCR);
        continue;
      }

      if (auto *PT = dyn_cast<PointerType>(GCR.getType())) {
        // The relocation of null will be null for most any collector.
        // TODO: provide a hook for this in GCStrategy.  There might be some
        // weird collector this property does not hold for.
        if (isa<ConstantPointerNull>(DerivedPtr)) {
          // Use null-pointer of gc_relocate's type to replace it.
          replaceInstUsesWith(GCR, ConstantPointerNull::get(PT));
          eraseInstFromFunction(GCR);
          continue;
        }

        // isKnownNonNull -> nonnull attribute
        if (!GCR.hasRetAttr(Attribute::NonNull) &&
            isKnownNonZero(DerivedPtr, DL, 0, &AC, &Call, &DT)) {
          GCR.addRetAttr(Attribute::NonNull);
          // We discovered new fact, re-check users.
          Worklist.pushUsersToWorkList(GCR);
        }
      }

      // If we have two copies of the same pointer in the statepoint argument
      // list, canonicalize to one.  This may let us common gc.relocates.
      if (GCR.getBasePtr() == GCR.getDerivedPtr() &&
          GCR.getBasePtrIndex() != GCR.getDerivedPtrIndex()) {
        auto *OpIntTy = GCR.getOperand(2)->getType();
        GCR.setOperand(2, ConstantInt::get(OpIntTy, GCR.getBasePtrIndex()));
      }

      // TODO: bitcast(relocate(p)) -> relocate(bitcast(p))
      // Canonicalize on the type from the uses to the defs

      // TODO: relocate((gep p, C, C2, ...)) -> gep(relocate(p), C, C2, ...)
      LiveGcValues.insert(BasePtr);
      LiveGcValues.insert(DerivedPtr);
    }
    Optional<OperandBundleUse> Bundle =
        GCSP.getOperandBundle(LLVMContext::OB_gc_live);
    unsigned NumOfGCLives = LiveGcValues.size();
    if (!Bundle || NumOfGCLives == Bundle->Inputs.size())
      break;
    // We can reduce the size of gc live bundle.
    DenseMap<Value *, unsigned> Val2Idx;
    std::vector<Value *> NewLiveGc;
    for (unsigned I = 0, E = Bundle->Inputs.size(); I < E; ++I) {
      Value *V = Bundle->Inputs[I];
      if (Val2Idx.count(V))
        continue;
      if (LiveGcValues.count(V)) {
        Val2Idx[V] = NewLiveGc.size();
        NewLiveGc.push_back(V);
      } else
        Val2Idx[V] = NumOfGCLives;
    }
    // Update all gc.relocates
    for (const GCRelocateInst *Reloc : GCSP.getGCRelocates()) {
      GCRelocateInst &GCR = *const_cast<GCRelocateInst *>(Reloc);
      Value *BasePtr = GCR.getBasePtr();
      assert(Val2Idx.count(BasePtr) && Val2Idx[BasePtr] != NumOfGCLives &&
             "Missed live gc for base pointer");
      auto *OpIntTy1 = GCR.getOperand(1)->getType();
      GCR.setOperand(1, ConstantInt::get(OpIntTy1, Val2Idx[BasePtr]));
      Value *DerivedPtr = GCR.getDerivedPtr();
      assert(Val2Idx.count(DerivedPtr) && Val2Idx[DerivedPtr] != NumOfGCLives &&
             "Missed live gc for derived pointer");
      auto *OpIntTy2 = GCR.getOperand(2)->getType();
      GCR.setOperand(2, ConstantInt::get(OpIntTy2, Val2Idx[DerivedPtr]));
    }
    // Create new statepoint instruction.
    OperandBundleDef NewBundle("gc-live", NewLiveGc);
    return CallBase::Create(&Call, NewBundle);
  }
  default: { break; }
  }

  return Changed ? &Call : nullptr;
}

/// If the callee is a constexpr cast of a function, attempt to move the cast to
/// the arguments of the call/callbr/invoke.
bool InstCombinerImpl::transformConstExprCastCall(CallBase &Call) {
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
      AttrBuilder RAttrs(FT->getContext(), CallerPAL.getRetAttrs());
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
      Callee->getAttributes().hasAttrSomewhere(Attribute::Preallocated))
    return false;

  auto AI = Call.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    Type *ParamTy = FT->getParamType(i);
    Type *ActTy = (*AI)->getType();

    if (!CastInst::isBitOrNoopPointerCastable(ActTy, ParamTy, DL))
      return false;   // Cannot transform this parameter value.

    // Check if there are any incompatible attributes we cannot drop safely.
    if (AttrBuilder(FT->getContext(), CallerPAL.getParamAttrs(i))
            .overlaps(AttributeFuncs::typeIncompatible(
                ParamTy, AttributeFuncs::ASK_UNSAFE_TO_DROP)))
      return false;   // Attribute not compatible with transformed value.

    if (Call.isInAllocaArgument(i) ||
        CallerPAL.hasParamAttr(i, Attribute::Preallocated))
      return false; // Cannot transform to and from inalloca/preallocated.

    if (CallerPAL.hasParamAttr(i, Attribute::SwiftError))
      return false;

    // If the parameter is passed as a byval argument, then we have to have a
    // sized type and the sized type has to have the same size as the old type.
    if (ParamTy != ActTy && CallerPAL.hasParamAttr(i, Attribute::ByVal)) {
      PointerType *ParamPTy = dyn_cast<PointerType>(ParamTy);
      if (!ParamPTy)
        return false;

      if (!ParamPTy->isOpaque()) {
        Type *ParamElTy = ParamPTy->getNonOpaquePointerElementType();
        if (!ParamElTy->isSized())
          return false;

        Type *CurElTy = Call.getParamByValType(i);
        if (DL.getTypeAllocSize(CurElTy) != DL.getTypeAllocSize(ParamElTy))
          return false;
      }
    }
  }

  if (Callee->isDeclaration()) {
    // Do not delete arguments unless we have a function body.
    if (FT->getNumParams() < NumActualArgs && !FT->isVarArg())
      return false;

    // If the callee is just a declaration, don't change the varargsness of the
    // call.  We don't want to introduce a varargs call where one doesn't
    // already exist.
    if (FT->isVarArg() != Call.getFunctionType()->isVarArg())
      return false;

    // If both the callee and the cast type are varargs, we still have to make
    // sure the number of fixed parameters are the same or we have the same
    // ABI issues as if we introduce a varargs call.
    if (FT->isVarArg() && Call.getFunctionType()->isVarArg() &&
        FT->getNumParams() != Call.getFunctionType()->getNumParams())
      return false;
  }

  if (FT->getNumParams() < NumActualArgs && FT->isVarArg() &&
      !CallerPAL.isEmpty()) {
    // In this case we have more arguments than the new function type, but we
    // won't be dropping them.  Check that these extra arguments have attributes
    // that are compatible with being a vararg call argument.
    unsigned SRetIdx;
    if (CallerPAL.hasAttrSomewhere(Attribute::StructRet, &SRetIdx) &&
        SRetIdx - AttributeList::FirstArgIndex >= FT->getNumParams())
      return false;
  }

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary.
  SmallVector<Value *, 8> Args;
  SmallVector<AttributeSet, 8> ArgAttrs;
  Args.reserve(NumActualArgs);
  ArgAttrs.reserve(NumActualArgs);

  // Get any return attributes.
  AttrBuilder RAttrs(FT->getContext(), CallerPAL.getRetAttrs());

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

    // Add any parameter attributes except the ones incompatible with the new
    // type. Note that we made sure all incompatible ones are safe to drop.
    AttributeMask IncompatibleAttrs = AttributeFuncs::typeIncompatible(
        ParamTy, AttributeFuncs::ASK_SAFE_TO_DROP);
    if (CallerPAL.hasParamAttr(i, Attribute::ByVal) &&
        !ParamTy->isOpaquePointerTy()) {
      AttrBuilder AB(Ctx, CallerPAL.getParamAttrs(i).removeAttributes(
                              Ctx, IncompatibleAttrs));
      AB.addByValAttr(ParamTy->getNonOpaquePointerElementType());
      ArgAttrs.push_back(AttributeSet::get(Ctx, AB));
    } else {
      ArgAttrs.push_back(
          CallerPAL.getParamAttrs(i).removeAttributes(Ctx, IncompatibleAttrs));
    }
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
        ArgAttrs.push_back(CallerPAL.getParamAttrs(i));
      }
    }
  }

  AttributeSet FnAttrs = CallerPAL.getFnAttrs();

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

  // Preserve prof metadata if any.
  NewCall->copyMetadata(*Caller, {LLVMContext::MD_prof});

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
InstCombinerImpl::transformCallThroughTrampoline(CallBase &Call,
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
      AttributeSet AS = NestAttrs.getParamAttrs(NestArgNo);
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
          NewArgAttrs.push_back(Attrs.getParamAttrs(ArgNo));

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
          AttributeList::get(FTy->getContext(), Attrs.getFnAttrs(),
                             Attrs.getRetAttrs(), NewArgAttrs);

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
