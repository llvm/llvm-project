//===- IRBuilder.cpp - Builder for LLVM Instrs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

using namespace llvm;

/// CreateGlobalString - Make a new global variable with an initializer that
/// has array of i8 type filled in with the nul terminated string value
/// specified.  If Name is specified, it is the name of the global variable
/// created.
GlobalVariable *IRBuilderBase::CreateGlobalString(StringRef Str,
                                                  const Twine &Name,
                                                  unsigned AddressSpace,
                                                  Module *M, bool AddNull) {
  Constant *StrConstant = ConstantDataArray::getString(Context, Str, AddNull);
  if (!M)
    M = BB->getParent()->getParent();
  auto *GV = new GlobalVariable(
      *M, StrConstant->getType(), true, GlobalValue::PrivateLinkage,
      StrConstant, Name, nullptr, GlobalVariable::NotThreadLocal, AddressSpace);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  GV->setAlignment(M->getDataLayout().getPrefTypeAlign(getInt8Ty()));
  return GV;
}

Type *IRBuilderBase::getCurrentFunctionReturnType() const {
  assert(BB && BB->getParent() && "No current function!");
  return BB->getParent()->getReturnType();
}

DebugLoc IRBuilderBase::getCurrentDebugLocation() const { return StoredDL; }
void IRBuilderBase::SetInstDebugLocation(Instruction *I) const {
  // We prefer to set our current debug location if any has been set, but if
  // our debug location is empty and I has a valid location, we shouldn't
  // overwrite it.
  I->setDebugLoc(StoredDL.orElse(I->getDebugLoc()));
}

Value *IRBuilderBase::CreateAggregateCast(Value *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy == DestTy)
    return V;

  if (SrcTy->isAggregateType()) {
    unsigned NumElements;
    if (SrcTy->isStructTy()) {
      assert(DestTy->isStructTy() && "Expected StructType");
      assert(SrcTy->getStructNumElements() == DestTy->getStructNumElements() &&
             "Expected StructTypes with equal number of elements");
      NumElements = SrcTy->getStructNumElements();
    } else {
      assert(SrcTy->isArrayTy() && DestTy->isArrayTy() && "Expected ArrayType");
      assert(SrcTy->getArrayNumElements() == DestTy->getArrayNumElements() &&
             "Expected ArrayTypes with equal number of elements");
      NumElements = SrcTy->getArrayNumElements();
    }

    Value *Result = PoisonValue::get(DestTy);
    for (unsigned I = 0; I < NumElements; ++I) {
      Type *ElementTy = SrcTy->isStructTy() ? DestTy->getStructElementType(I)
                                            : DestTy->getArrayElementType();
      Value *Element =
          CreateAggregateCast(CreateExtractValue(V, ArrayRef(I)), ElementTy);

      Result = CreateInsertValue(Result, Element, ArrayRef(I));
    }
    return Result;
  }

  return CreateBitOrPointerCast(V, DestTy);
}

Value *IRBuilderBase::CreateBitPreservingCastChain(const DataLayout &DL,
                                                   Value *V, Type *NewTy) {
  Type *OldTy = V->getType();

  if (OldTy == NewTy)
    return V;

  assert(!(isa<IntegerType>(OldTy) && isa<IntegerType>(NewTy)) &&
         "Integer types must be the exact same to convert.");

  // A variant of bitcast that supports a mixture of fixed and scalable types
  // that are know to have the same size.
  auto CreateBitCastLike = [this](Value *In, Type *Ty) -> Value * {
    Type *InTy = In->getType();
    if (InTy == Ty)
      return In;

    if (isa<FixedVectorType>(InTy) && isa<ScalableVectorType>(Ty)) {
      // For vscale_range(2) expand <4 x i32> to <vscale x 4 x i16> -->
      //   <4 x i32> to <vscale x 2 x i32> to <vscale x 4 x i16>
      auto *VTy = VectorType::getWithSizeAndScalar(cast<VectorType>(Ty), InTy);
      return CreateBitCast(
          CreateInsertVector(VTy, PoisonValue::get(VTy), In, getInt64(0)), Ty);
    }

    if (isa<ScalableVectorType>(InTy) && isa<FixedVectorType>(Ty)) {
      // For vscale_range(2) expand <vscale x 4 x i16> to <4 x i32> -->
      //   <vscale x 4 x i16> to <vscale x 2 x i32> to <4 x i32>
      auto *VTy = VectorType::getWithSizeAndScalar(cast<VectorType>(InTy), Ty);
      return CreateExtractVector(Ty, CreateBitCast(In, VTy), getInt64(0));
    }

    return CreateBitCast(In, Ty);
  };

  // See if we need inttoptr for this type pair. May require additional bitcast.
  if (!OldTy->isPtrOrPtrVectorTy() && NewTy->isPtrOrPtrVectorTy()) {
    // Expand <2 x i32> to i8* --> <2 x i32> to i64 to i8*
    // Expand i128 to <2 x i8*> --> i128 to <2 x i64> to <2 x i8*>
    // Expand <4 x i32> to <2 x i8*> --> <4 x i32> to <2 x i64> to <2 x i8*>
    // Directly handle i64 to i8*
    return CreateIntToPtr(CreateBitCastLike(V, DL.getIntPtrType(NewTy)), NewTy);
  }

  // See if we need ptrtoint for this type pair. May require additional bitcast.
  if (OldTy->isPtrOrPtrVectorTy() && !NewTy->isPtrOrPtrVectorTy()) {
    // Expand <2 x i8*> to i128 --> <2 x i8*> to <2 x i64> to i128
    // Expand i8* to <2 x i32> --> i8* to i64 to <2 x i32>
    // Expand <2 x i8*> to <4 x i32> --> <2 x i8*> to <2 x i64> to <4 x i32>
    // Expand i8* to i64 --> i8* to i64 to i64
    return CreateBitCastLike(CreatePtrToInt(V, DL.getIntPtrType(OldTy)), NewTy);
  }

  if (OldTy->isPtrOrPtrVectorTy() && NewTy->isPtrOrPtrVectorTy()) {
    unsigned OldAS = OldTy->getPointerAddressSpace();
    unsigned NewAS = NewTy->getPointerAddressSpace();
    // To convert pointers with different address spaces (they are already
    // checked convertible, i.e. they have the same pointer size), so far we
    // cannot use `bitcast` (which has restrict on the same address space) or
    // `addrspacecast` (which is not always no-op casting). Instead, use a pair
    // of no-op `ptrtoint`/`inttoptr` casts through an integer with the same bit
    // size.
    if (OldAS != NewAS) {
      return CreateIntToPtr(
          CreateBitCastLike(CreatePtrToInt(V, DL.getIntPtrType(OldTy)),
                            DL.getIntPtrType(NewTy)),
          NewTy);
    }
  }

  return CreateBitCastLike(V, NewTy);
}

//===----------------------------------------------------------------------===//
// CreateLayoutReinterpretCast helpers
//===----------------------------------------------------------------------===//

namespace {

// Lazy DFS iterator over the leaves of a type tree.
struct TypeLeafIterator {
  // A fully structured GEP representation over a specific type
  struct PathGEP {
    // Type of parent
    Type *ParentTy;
    // Index of leaf element in Parent
    unsigned CurrentIdx;
    // Count of elements in Parent
    unsigned EndIdx;
    // Absolute byte offset, from start of outermost type.
    uint64_t ParentOffset;
  };

  struct Leaf {
    // Absolute byte offset, from start of outermost type.
    uint64_t Offset;
    // Size of Ty
    uint64_t BitWidth;
    // `advance` sets this to an integer, float, pointer, or vector thereof.
    // `pop` sets it to the parent of that.
    Type *Ty;
  };

  const DataLayout &DL;
  SmallVector<PathGEP, 8> Path;
  Leaf Element;
  bool Done;

  static unsigned numChildren(Type *Ty) {
    if (auto *STy = dyn_cast<StructType>(Ty))
      return STy->getNumElements();
    if (auto *ATy = dyn_cast<ArrayType>(Ty))
      return ATy->getNumElements();
    return 0;
  }

  TypeLeafIterator(Type *RootTy, const DataLayout &DL) : DL(DL), Done(false) {
    unsigned N = numChildren(RootTy);
    if (N) {
      Path.push_back({RootTy, -1U, N, 0});
      advance();
    } else
      Element = Leaf{0, DL.getTypeStoreSizeInBits(RootTy), RootTy};
  }

  bool done() { return Done; }

  void advance() {
    while (!Path.empty()) {
      PathGEP &W = Path.back();

      unsigned Idx = ++W.CurrentIdx;
      if (Idx == W.EndIdx) {
        Path.pop_back();
        continue;
      }

      Type *ChildTy;
      uint64_t ChildBitOffset;
      if (auto *STy = dyn_cast<StructType>(W.ParentTy)) {
        ChildTy = STy->getElementType(Idx);
        ChildBitOffset =
            W.ParentOffset + DL.getStructLayout(STy)->getElementOffset(Idx);
      } else if (auto *ATy = dyn_cast<ArrayType>(W.ParentTy)) {
        // Arrays: stride is alloc size (includes tail padding on element)
        ChildTy = ATy->getElementType();
        ChildBitOffset = W.ParentOffset + Idx * DL.getTypeAllocSize(ChildTy);
      }

      unsigned N = numChildren(ChildTy);
      if (N)
        Path.push_back({ChildTy, -1U, N, ChildBitOffset});
      else {
        Element = {ChildBitOffset, DL.getTypeStoreSizeInBits(ChildTy), ChildTy};
        return;
      }
    }
    Done = true;
  }

  bool parent_equal(const TypeLeafIterator &O) const {
    if (Path.empty() || O.Path.empty())
      return false;
    const PathGEP &PT = Path.back();
    const PathGEP &PO = O.Path.back();
    return PT.ParentTy == PO.ParentTy && PT.CurrentIdx == 0 &&
           PO.CurrentIdx == 0;
  }

  void pop() {
    PathGEP &W = Path.back();
    assert(W.CurrentIdx == 0);
    Element = {W.ParentOffset, DL.getTypeStoreSizeInBits(W.ParentTy),
               W.ParentTy};
    Path.pop_back();
  }
};

// Build parameter for testing nested vs flat FCA: use incremental
// insert/extract or full
static const bool USE_FULL_IDX = false;
struct AggChild {
  Value *V = nullptr;
  unsigned AtIdx = 0;
};
using IncompleteAgg = SmallVector<AggChild>;

// Given a GEP path in SrcIt, extract Src, caching in Srcs.
static Value *buildExtractSrc(IRBuilderBase &B, Value *Src,
                              TypeLeafIterator SrcIt, IncompleteAgg &Srcs) {
  auto &Path = SrcIt.Path;
  size_t NumElem = Path.size();
  if (NumElem == 0)
    return Src;
  if (USE_FULL_IDX) {
    // TODO: implement caching in Srcs
    SmallVector<unsigned> ChildIdxs;
    ChildIdxs.reserve(NumElem);
    for (auto Child : Path)
      ChildIdxs.push_back(Child.CurrentIdx);
    return B.CreateExtractValue(Src, ChildIdxs);
  }
  // Truncate old Srcs at first mismatching index
  size_t Idx = 0;
  for (; Idx < Srcs.size(); ++Idx) {
    if (Idx >= NumElem || Srcs[Idx].AtIdx != Path[Idx].CurrentIdx) {
      Srcs.truncate(Idx);
      break;
    }
  }
  // Then extract new index until completed
  for (; Idx < NumElem; ++Idx) {
    unsigned ChildIdx = Path[Idx].CurrentIdx;
    Value *Base = Idx == 0 ? Src : Srcs[Idx - 1].V;
    Base = B.CreateExtractValue(Base, ChildIdx);
    Srcs.push_back({Base, ChildIdx});
  }
  return Srcs.back().V;
}

// Given a GEP path in DstIt, insert Fragment, caching in Dsts.
static void buildInsertDst(IRBuilderBase &B, const TypeLeafIterator &DstIt,
                           Value *Fragment, IncompleteAgg &Dsts) {
  auto &Path = DstIt.Path;
  size_t NumElem = Path.size();
  if (NumElem == 0 && Fragment) {
    assert(Dsts.empty());
    Dsts.push_back({Fragment, -1U});
    return;
  }
  if (USE_FULL_IDX) {
    if (Dsts.empty())
      Dsts.push_back({PoisonValue::get(Path[0].ParentTy), -1U});
    if (!Fragment)
      return;
    SmallVector<unsigned> ChildIdxs;
    ChildIdxs.reserve(NumElem);
    for (auto Child : Path)
      ChildIdxs.push_back(Child.CurrentIdx);
    Dsts[0].V = B.CreateInsertValue(Dsts[0].V, Fragment, ChildIdxs);
    return;
  }
  volatile size_t Idx = 0;
  if (!Dsts.empty()) {
    // Find first mismatching index in Dsts
    for (; Idx < NumElem && Idx < Dsts.size(); ++Idx) {
      unsigned ChildIdx = Path[Idx].CurrentIdx;
      if (Dsts[Idx].AtIdx != ChildIdx)
        break;
    }
    // Truncate until that is the last item
    while (Idx + 1 < Dsts.size()) {
      volatile AggChild Last = Dsts.pop_back_val();
      AggChild &Parent = Dsts.back();
      assert(Parent.AtIdx != -1U);
      Parent.V = B.CreateInsertValue(Parent.V, Last.V, Parent.AtIdx);
      Parent.AtIdx = -1;
    }
    if (NumElem == 0)
      return;
    // Update just child index of last item
    Dsts[Idx].AtIdx = Path[Idx].CurrentIdx;
    ++Idx;
  }
  assert(NumElem);
  // Now initialize new elements in this path with poison
  for (; Idx < NumElem; ++Idx) {
    unsigned ChildIdx = Path[Idx].CurrentIdx;
    Value *Base = PoisonValue::get(Path[Idx].ParentTy);
    Dsts.push_back({Base, ChildIdx});
  }
  AggChild &Tail = Dsts.back();
  assert(Tail.AtIdx == Path.back().CurrentIdx);
  Tail.AtIdx = -1U; // Unused.
  Tail.V = B.CreateInsertValue(Tail.V, Fragment, Path.back().CurrentIdx);
}

// Build the IR value for one destination leaf, given its source contributions.
static Value *
buildDstLeaf(IRBuilderBase &B, const DataLayout &DL,
             Value *Fragment,      // Current parts of Dst, or nullptr
             Type *Ty,             // Target Type
             Value *SrcVal,        // Source Value
             uint64_t SrcBitWidth, // Size of Src
             uint64_t DstBitWidth, // Size of Dst
             uint64_t SrcShift,    // Bytes to shift within src leaf before use.
             uint64_t DstShift,    // Bytes to shift before placing in dst leaf.
             uint64_t BitWidth) {  // Bits to be transferred.

  // Type-preserving path: single full-width, zero-shift contribution where the
  // source leaf is exactly the same width as the destination leaf.
  if (Fragment == nullptr && SrcShift == 0 && DstShift == 0 &&
      BitWidth == DstBitWidth && BitWidth == SrcBitWidth) {
    return B.CreateBitPreservingCastChain(DL, SrcVal, Ty);
  }

  // Try to keep vectors as vectors, instead of using large integers.
  // This is required, at minimum, when both types contain ptrs,
  // since those need to get moved without inttoptr / ptrtoint pairs whenever
  // applicable and possible.
  if (BitWidth > 0 && BitWidth % 8 == 0) {
    // If the source is a vector, try to extract the relevant element directly
    // with extractelement, rather than converting the whole vector to an
    // integer.
    if ((SrcShift * 8) % BitWidth == 0) {
      if (auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcVal->getType())) {
        Type *EltTy = SrcVecTy->getElementType();
        bool SameWidthAsSrc = BitWidth == DL.getTypeSizeInBits(EltTy);
        if (!SameWidthAsSrc && BitWidth == DstBitWidth && !Ty->isVectorTy() &&
            SrcBitWidth % BitWidth == 0) {
          // Bitcast SrcVal to a vector of Ty so that extractelement below can
          // be used
          unsigned NewNumElts = SrcBitWidth / BitWidth;
          SrcVecTy = FixedVectorType::get(Ty, NewNumElts);
          SrcVal = B.CreateBitPreservingCastChain(DL, SrcVal, SrcVecTy);
          SameWidthAsSrc = true;
        }
        if (SameWidthAsSrc) {
          // SrcShift is the byte offset from the start (LE) or end (BE) of the
          // source vector register to the overlap region. Convert to element
          // index.
          uint64_t EltIdx = SrcShift * 8 / BitWidth;
          if (DL.isBigEndian())
            EltIdx = SrcVecTy->getNumElements() - 1 - EltIdx;
          SrcVal = B.CreateExtractElement(SrcVal, EltIdx);
          SrcBitWidth = BitWidth;
          SrcShift = 0;
          // After extractelement the type-preserving conditions may now hold;
          // short-circuit to avoid the integer round-trip in the general path.
          if (Fragment == nullptr && DstShift == 0 && BitWidth == DstBitWidth)
            return B.CreateBitPreservingCastChain(DL, SrcVal, Ty);
        }
      }
    }

    // If the destination is a vector, try to place the
    // value with insertelement rather than building a large integer and
    // bitcasting at the end.
    if ((DstShift * 8) % BitWidth == 0) {
      if (auto *DstVecTy = dyn_cast<FixedVectorType>(Ty)) {
        Type *EltTy = DstVecTy->getElementType();
        bool SameWidthAsDst = BitWidth == DL.getTypeSizeInBits(EltTy);
        if (!SameWidthAsDst && BitWidth == SrcBitWidth &&
            !SrcVal->getType()->isVectorTy() && DstBitWidth % BitWidth == 0) {
          // If possible, replace DstVecTy with a vector of SrcVal's type so
          // that insertelement below can be used.
          Type *SrcTy = SrcVal->getType();
          unsigned NewNumElts = DstBitWidth / BitWidth;
          DstVecTy = FixedVectorType::get(SrcTy, NewNumElts);
          EltTy = SrcTy;
          SameWidthAsDst = true;
        }
        if (SameWidthAsDst) {
          // DstShift is the byte offset from the start (LE) or end (BE) of the
          // destination vector register to the overlap region.
          uint64_t EltIdx = DstShift * 8 / BitWidth;
          if (DL.isBigEndian())
            EltIdx = DstVecTy->getNumElements() - 1 - EltIdx;
          Value *EltVal;
          if (SrcShift == 0 && SrcBitWidth == BitWidth) {
            // Source is already the right size; just bitcast to the element
            // type.
            EltVal = B.CreateBitPreservingCastChain(DL, SrcVal, EltTy);
          } else {
            // Extract the relevant bits from a larger source via integer ops.
            if (!isa<IntegerType>(SrcVal->getType()))
              SrcVal = B.CreateBitPreservingCastChain(DL, SrcVal,
                                                      B.getIntNTy(SrcBitWidth));
            if (SrcShift)
              SrcVal = B.CreateLShr(SrcVal, SrcShift * 8);
            SrcVal = B.CreateZExtOrTrunc(SrcVal, B.getIntNTy(BitWidth));
            EltVal = B.CreateBitPreservingCastChain(DL, SrcVal, EltTy);
          }
          if (Fragment)
            Fragment = B.CreateBitPreservingCastChain(DL, Fragment, DstVecTy);
          else
            Fragment = PoisonValue::get(DstVecTy);
          return B.CreateInsertElement(Fragment, EltVal, EltIdx);
        }
      }
    }

    // If both source and destination are vectors, use shufflevector to transfer
    // multiple elements at once, avoiding integer intermediates.
    // Requires element-aligned shifts and transfer size.
    if (isa<FixedVectorType>(SrcVal->getType())) {
      if (auto *DVTy = dyn_cast<FixedVectorType>(Ty)) {
        auto *SVTy = cast<FixedVectorType>(SrcVal->getType());
        uint64_t DstEltBitSize = DL.getTypeSizeInBits(DVTy->getElementType());
        uint64_t SrcEltBitSize = DL.getTypeSizeInBits(SVTy->getElementType());
        // Prefer DstEltTy as the common element type (SrcVal gets
        // reinterpreted). Fall back to SrcEltTy (result gets reinterpreted)
        // when SrcBitWidth is not divisible by DstEltBitSize but DstBitWidth is
        // by SrcEltBitSize.
        Type *EltTy = nullptr;
        uint64_t EltBitSize = 0;
        if (DstEltBitSize > 0 && SrcBitWidth % DstEltBitSize == 0 &&
            BitWidth % DstEltBitSize == 0 &&
            (DstShift * 8) % DstEltBitSize == 0) {
          EltTy = DVTy->getElementType();
          EltBitSize = DstEltBitSize;
        } else if (SrcEltBitSize > 0 && DstBitWidth % SrcEltBitSize == 0 &&
                   BitWidth % SrcEltBitSize == 0 &&
                   (DstShift * 8) % SrcEltBitSize == 0) {
          EltTy = SVTy->getElementType();
          EltBitSize = SrcEltBitSize;
        }
        if (EltTy) {
          // Check whether the source extraction can be done directly in
          // EltBitSize units, or whether we need to shuffle in SrcEltBitSize
          // units first (when SrcShift is only aligned to SrcEltBitSize, not
          // EltBitSize).
          bool SrcShiftAligned = (SrcShift * 8) % EltBitSize == 0;
          // Shuffle-before-cast: SrcShift aligns to SrcEltBitSize (not
          // EltBitSize). A single shufflevector in SrcEltTy space directly into
          // a NumResultElts_src-sized result, then bitcast to ResultVecTy.
          // Requires both Src/DstShift aligned to SrcEltBitSize.
          bool ShuffleBeforeCast = !SrcShiftAligned && SrcEltBitSize > 0 &&
                                   BitWidth % SrcEltBitSize == 0 &&
                                   (SrcShift * 8) % SrcEltBitSize == 0 &&
                                   DstBitWidth % SrcEltBitSize == 0 &&
                                   (DstShift * 8) % SrcEltBitSize == 0;
          if (SrcShiftAligned || ShuffleBeforeCast) {

            uint64_t NumResultElts = DstBitWidth / EltBitSize;
            uint64_t NumTransferElts = BitWidth / EltBitSize;
            uint64_t DstStartElt = (DstShift * 8) / EltBitSize;
            if (DL.isBigEndian())
              DstStartElt = NumResultElts - DstStartElt - NumTransferElts;

            auto *ResultVecTy = FixedVectorType::get(EltTy, NumResultElts);
            // Step 1: scatter the transferred elements into a ResultVecTy-sized
            // vector at their destination positions (poison elsewhere).
            // SrcShiftAligned: cast SrcVal to EltTy view, then shufflevector.
            // ShuffleBeforeCast: shufflevector in SrcEltTy space, then bitcast.
            Value *SrcExtracted;
            if (SrcShiftAligned) {
              uint64_t NumCommonElts = SrcBitWidth / EltBitSize;
              uint64_t SrcStartElt = (SrcShift * 8) / EltBitSize;
              if (DL.isBigEndian())
                SrcStartElt = NumCommonElts - SrcStartElt - NumTransferElts;
              auto *CommonVecTy = FixedVectorType::get(EltTy, NumCommonElts);
              Value *SrcAsCommon =
                  B.CreateBitPreservingCastChain(DL, SrcVal, CommonVecTy);
              SmallVector<int, 16> ScatterMask(NumResultElts, -1);
              for (unsigned k = 0; k < NumTransferElts; ++k)
                ScatterMask[DstStartElt + k] = SrcStartElt + k;
              Value *FragAsCommon = Fragment && NumCommonElts == NumResultElts
                                        ? B.CreateBitPreservingCastChain(
                                              DL, Fragment, CommonVecTy)
                                        : PoisonValue::get(CommonVecTy);
              if (Fragment && NumCommonElts == NumResultElts) {
                for (unsigned i = 0; i < NumResultElts; ++i)
                  if (ScatterMask[i] == -1)
                    ScatterMask[i] = NumCommonElts + i;
                Fragment = nullptr;
              }
              SrcExtracted =
                  B.CreateShuffleVector(SrcAsCommon, FragAsCommon, ScatterMask);
            } else {
              // SrcShift aligns to SrcEltBitSize but not EltBitSize: scatter in
              // SrcEltTy space directly to the result-sized vector, then
              // bitcast.
              uint64_t NumResultElts_src = DstBitWidth / SrcEltBitSize;
              uint64_t NumTransferElts_src = BitWidth / SrcEltBitSize;
              uint64_t NumSrcElts = SrcBitWidth / SrcEltBitSize;
              uint64_t SrcStartElt_src = (SrcShift * 8) / SrcEltBitSize;
              uint64_t DstStartElt_src = (DstShift * 8) / SrcEltBitSize;
              if (DL.isBigEndian()) {
                SrcStartElt_src =
                    NumSrcElts - SrcStartElt_src - NumTransferElts_src;
                DstStartElt_src =
                    NumResultElts_src - DstStartElt_src - NumTransferElts_src;
              }
              SmallVector<int, 16> ScatterMask(NumResultElts_src, -1);
              for (unsigned k = 0; k < NumTransferElts_src; ++k)
                ScatterMask[DstStartElt_src + k] = SrcStartElt_src + k;
              Value *Shuffled = B.CreateShuffleVector(
                  SrcVal, PoisonValue::get(SVTy), ScatterMask);
              SrcExtracted =
                  B.CreateBitPreservingCastChain(DL, Shuffled, ResultVecTy);
            }

            // Step 2: blend SrcExtracted with Fragment if needed.
            if (!Fragment)
              return SrcExtracted;
            Value *FragAsDst =
                Fragment
                    ? B.CreateBitPreservingCastChain(DL, Fragment, ResultVecTy)
                    : PoisonValue::get(ResultVecTy);
            SmallVector<int, 16> Mask(NumResultElts);
            for (unsigned i = 0; i < NumResultElts; ++i)
              Mask[i] = (i >= DstStartElt && i < DstStartElt + NumTransferElts)
                            ? (int)(NumResultElts + i)
                            : (Fragment ? (int)i : -1);
            return B.CreateShuffleVector(FragAsDst, SrcExtracted, Mask);
          } // SrcShiftAligned || ShuffleBeforeCast
        }
      }
    }
  }

  // General path: accumulate into an integer of DstLeaf.BitWidth
  // for integers, floats, and pointers.
  if (!isa<IntegerType>(SrcVal->getType()))
    SrcVal =
        B.CreateBitPreservingCastChain(DL, SrcVal, B.getIntNTy(SrcBitWidth));
  if (SrcShift)
    SrcVal = B.CreateLShr(SrcVal, SrcShift * 8);
  if (BitWidth < DstBitWidth && BitWidth + SrcShift * 8 < SrcBitWidth)
    SrcVal = B.CreateAnd(SrcVal, APInt::getLowBitsSet(SrcBitWidth, BitWidth));
  SrcVal = B.CreateZExtOrTrunc(SrcVal, B.getIntNTy(DstBitWidth));
  if (DstShift)
    SrcVal = B.CreateShl(SrcVal, DstShift * 8);
  if (Fragment) {
    if (Fragment->getType() != SrcVal->getType()) {
      assert(isa<FixedVectorType>(Ty));
      Fragment = B.CreateBitPreservingCastChain(DL, Fragment,
                                                B.getIntNTy(DstBitWidth));
    }
    SrcVal = B.CreateOr(Fragment, SrcVal);
  }
  return SrcVal;
}

} // anonymous namespace

Value *IRBuilderBase::CreateLayoutReinterpretCast(Value *Src, Type *DestTy,
                                                  uint64_t SrcOffset,
                                                  uint64_t DstOffset,
                                                  const Twine &Name) {
  const DataLayout &DL = BB->getDataLayout();
  TypeLeafIterator SrcIt(Src->getType(), DL);
  TypeLeafIterator DstIt(DestTy, DL);
  IncompleteAgg ExtractedSrc;
  IncompleteAgg InsertedDst;

  // Build destination aggregate leaf by leaf.
  for (; !DstIt.done(); DstIt.advance()) {
    TypeLeafIterator::Leaf &DstLeaf = DstIt.Element;
    if (DstLeaf.BitWidth == 0)
      continue;
    uint64_t DStart = DstLeaf.Offset + DstOffset;
    uint64_t DBitEnd = DStart * 8 + DstLeaf.BitWidth;
    Value *Fragment = nullptr;
    for (; !SrcIt.done(); SrcIt.advance()) {
      TypeLeafIterator::Leaf &SrcLeaf = SrcIt.Element;
      if (SrcLeaf.BitWidth == 0)
        continue;
      uint64_t SStart = SrcLeaf.Offset + SrcOffset;
      uint64_t SBitEnd = SStart * 8 + SrcLeaf.BitWidth;
      if (SBitEnd <= DStart * 8)
        continue;
      if (SStart * 8 >= DBitEnd)
        break;
      // Have the first element with any overlap.
      if (SStart == DStart && SrcLeaf.Ty == DstLeaf.Ty) {
        // In this case, all the math and casts below are zeros and no-ops.
        // But that also means we might be able to extract more at once.
        while (DstIt.parent_equal(SrcIt)) {
          DstIt.pop();
          SrcIt.pop();
        }
        Fragment = buildExtractSrc(*this, Src, SrcIt, ExtractedSrc);
        assert(Fragment->getType() == DstLeaf.Ty);
        break;
      }
      // Compute the range of bits that overlap in memory.
      Value *Extract = buildExtractSrc(*this, Src, SrcIt, ExtractedSrc);
      uint64_t OvStart = std::max(SStart, DStart);
      uint64_t OvBitEnd = std::min(SBitEnd, DBitEnd);
      uint64_t OvBitWidth = OvBitEnd - OvStart * 8;
      // Convert that byte index to a register bit shift, from either the start
      // (LE) or the end (BE).
      //
      // LangRef: a non-byte-sized store behaves like a zext followed by a
      // byte-sized store.
      bool LE = DL.isLittleEndian();
      uint64_t SrcShift =
          LE ? OvStart - SStart
             : (alignToPowerOf2(SBitEnd, 8) - alignToPowerOf2(OvBitEnd, 8)) / 8;
      uint64_t DstShift =
          LE ? OvStart - DStart
             : (alignToPowerOf2(DBitEnd, 8) - alignToPowerOf2(OvBitEnd, 8)) / 8;
      Fragment = buildDstLeaf(*this, DL, Fragment, DstLeaf.Ty, Extract,
                              SrcLeaf.BitWidth, DstLeaf.BitWidth, SrcShift,
                              DstShift, OvBitWidth);
      if (SBitEnd >= DBitEnd)
        break;
    }
    if (Fragment) {
      Fragment = CreateBitPreservingCastChain(DL, Fragment, DstLeaf.Ty);
      buildInsertDst(*this, DstIt, Fragment, InsertedDst);
    }
    if (SrcIt.done()) {
      DstIt.Path.clear();
      break;
    }
  }

  assert(DstIt.Path.empty());
  Value *Result;
  if (InsertedDst.empty()) {
    // No overlapping bits got extracted.
    Result = PoisonValue::get(DestTy);
  } else {
    // Create final CreateInsertValue chain.
    buildInsertDst(*this, DstIt, nullptr, InsertedDst);
    assert(InsertedDst.size() == 1);
    Result = InsertedDst.back().V;
  }

  if (auto *I = dyn_cast<Instruction>(Result))
    I->setName(Name);
  return Result;
}

CallInst *
IRBuilderBase::createCallHelper(Function *Callee, ArrayRef<Value *> Ops,
                                const Twine &Name, FMFSource FMFSource,
                                ArrayRef<OperandBundleDef> OpBundles) {
  CallInst *CI = CreateCall(Callee, Ops, OpBundles, Name);
  if (isa<FPMathOperator>(CI))
    CI->setFastMathFlags(FMFSource.get(FMF));
  return CI;
}

static Value *CreateVScaleMultiple(IRBuilderBase &B, Type *Ty, uint64_t Scale) {
  Value *VScale = B.CreateVScale(Ty);
  if (Scale == 1)
    return VScale;

  return B.CreateNUWMul(VScale, ConstantInt::get(Ty, Scale));
}

Value *IRBuilderBase::CreateElementCount(Type *Ty, ElementCount EC) {
  if (EC.isFixed() || EC.isZero())
    return ConstantInt::get(Ty, EC.getKnownMinValue());

  return CreateVScaleMultiple(*this, Ty, EC.getKnownMinValue());
}

Value *IRBuilderBase::CreateTypeSize(Type *Ty, TypeSize Size) {
  if (Size.isFixed() || Size.isZero())
    return ConstantInt::get(Ty, Size.getKnownMinValue());

  return CreateVScaleMultiple(*this, Ty, Size.getKnownMinValue());
}

Value *IRBuilderBase::CreateAllocationSize(Type *DestTy, AllocaInst *AI) {
  const DataLayout &DL = BB->getDataLayout();
  TypeSize ElemSize = DL.getTypeAllocSize(AI->getAllocatedType());
  Value *Size = CreateTypeSize(DestTy, ElemSize);
  if (AI->isArrayAllocation())
    Size = CreateMul(CreateZExtOrTrunc(AI->getArraySize(), DestTy), Size);
  return Size;
}

Value *IRBuilderBase::CreateStepVector(Type *DstType, const Twine &Name) {
  Type *STy = DstType->getScalarType();
  if (isa<ScalableVectorType>(DstType)) {
    Type *StepVecType = DstType;
    // TODO: We expect this special case (element type < 8 bits) to be
    // temporary - once the intrinsic properly supports < 8 bits this code
    // can be removed.
    if (STy->getScalarSizeInBits() < 8)
      StepVecType =
          VectorType::get(getInt8Ty(), cast<ScalableVectorType>(DstType));
    Value *Res = CreateIntrinsic(Intrinsic::stepvector, {StepVecType}, {},
                                 nullptr, Name);
    if (StepVecType != DstType)
      Res = CreateTrunc(Res, DstType);
    return Res;
  }

  unsigned NumEls = cast<FixedVectorType>(DstType)->getNumElements();

  // Create a vector of consecutive numbers from zero to VF.
  // It's okay if the values wrap around.
  SmallVector<Constant *, 8> Indices;
  for (unsigned i = 0; i < NumEls; ++i)
    Indices.push_back(
        ConstantInt::get(STy, i, /*IsSigned=*/false, /*ImplicitTrunc=*/true));

  // Add the consecutive indices to the vector value.
  return ConstantVector::get(Indices);
}

CallInst *IRBuilderBase::CreateMemSet(Value *Ptr, Value *Val, Value *Size,
                                      MaybeAlign Align, bool isVolatile,
                                      const AAMDNodes &AAInfo) {
  Value *Ops[] = {Ptr, Val, Size, getInt1(isVolatile)};
  Type *Tys[] = {Ptr->getType(), Size->getType()};

  auto *CI = cast<MemSetInst>(
      CreateIntrinsicWithoutFolding(Intrinsic::memset, Tys, Ops));

  if (Align)
    CI->setDestAlignment(*Align);
  CI->setAAMetadata(AAInfo);
  return CI;
}

CallInst *IRBuilderBase::CreateMemSetInline(Value *Dst, MaybeAlign DstAlign,
                                            Value *Val, Value *Size,
                                            bool IsVolatile,
                                            const AAMDNodes &AAInfo) {
  Value *Ops[] = {Dst, Val, Size, getInt1(IsVolatile)};
  Type *Tys[] = {Dst->getType(), Size->getType()};

  auto *CI = cast<MemSetInst>(
      CreateIntrinsicWithoutFolding(Intrinsic::memset_inline, Tys, Ops));

  if (DstAlign)
    CI->setDestAlignment(*DstAlign);
  CI->setAAMetadata(AAInfo);
  return CI;
}

CallInst *IRBuilderBase::CreateElementUnorderedAtomicMemSet(
    Value *Ptr, Value *Val, Value *Size, Align Alignment, uint32_t ElementSize,
    const AAMDNodes &AAInfo) {

  Value *Ops[] = {Ptr, Val, Size, getInt32(ElementSize)};
  Type *Tys[] = {Ptr->getType(), Size->getType()};

  auto *CI = cast<AnyMemSetInst>(CreateIntrinsicWithoutFolding(
      Intrinsic::memset_element_unordered_atomic, Tys, Ops));
  CI->setDestAlignment(Alignment);
  CI->setAAMetadata(AAInfo);
  return CI;
}

CallInst *IRBuilderBase::CreateMemTransferInst(Intrinsic::ID IntrID, Value *Dst,
                                               MaybeAlign DstAlign, Value *Src,
                                               MaybeAlign SrcAlign, Value *Size,
                                               bool isVolatile,
                                               const AAMDNodes &AAInfo) {
  assert((IntrID == Intrinsic::memcpy || IntrID == Intrinsic::memcpy_inline ||
          IntrID == Intrinsic::memmove) &&
         "Unexpected intrinsic ID");
  Value *Ops[] = {Dst, Src, Size, getInt1(isVolatile)};
  Type *Tys[] = {Dst->getType(), Src->getType(), Size->getType()};

  auto *MCI =
      cast<MemTransferInst>(CreateIntrinsicWithoutFolding(IntrID, Tys, Ops));

  if (DstAlign)
    MCI->setDestAlignment(*DstAlign);
  if (SrcAlign)
    MCI->setSourceAlignment(*SrcAlign);
  MCI->setAAMetadata(AAInfo);
  return MCI;
}

CallInst *IRBuilderBase::CreateElementUnorderedAtomicMemCpy(
    Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, Value *Size,
    uint32_t ElementSize, const AAMDNodes &AAInfo) {
  assert(DstAlign >= ElementSize &&
         "Pointer alignment must be at least element size");
  assert(SrcAlign >= ElementSize &&
         "Pointer alignment must be at least element size");
  Value *Ops[] = {Dst, Src, Size, getInt32(ElementSize)};
  Type *Tys[] = {Dst->getType(), Src->getType(), Size->getType()};

  auto *AMCI = cast<AnyMemCpyInst>(CreateIntrinsicWithoutFolding(
      Intrinsic::memcpy_element_unordered_atomic, Tys, Ops));

  // Set the alignment of the pointer args.
  AMCI->setDestAlignment(DstAlign);
  AMCI->setSourceAlignment(SrcAlign);
  AMCI->setAAMetadata(AAInfo);
  return AMCI;
}

/// isConstantOne - Return true only if val is constant int 1
static bool isConstantOne(const Value *Val) {
  assert(Val && "isConstantOne does not work with nullptr Val");
  const ConstantInt *CVal = dyn_cast<ConstantInt>(Val);
  return CVal && CVal->isOne();
}

CallInst *IRBuilderBase::CreateMalloc(Type *IntPtrTy, Type *AllocTy,
                                      Value *AllocSize, Value *ArraySize,
                                      ArrayRef<OperandBundleDef> OpB,
                                      Function *MallocF, const Twine &Name) {
  // malloc(type) becomes:
  //       i8* malloc(typeSize)
  // malloc(type, arraySize) becomes:
  //       i8* malloc(typeSize*arraySize)
  if (!ArraySize)
    ArraySize = ConstantInt::get(IntPtrTy, 1);
  else if (ArraySize->getType() != IntPtrTy)
    ArraySize = CreateIntCast(ArraySize, IntPtrTy, false);

  if (!isConstantOne(ArraySize)) {
    if (isConstantOne(AllocSize)) {
      AllocSize = ArraySize; // Operand * 1 = Operand
    } else {
      // Multiply type size by the array size...
      AllocSize = CreateMul(ArraySize, AllocSize, "mallocsize");
    }
  }

  assert(AllocSize->getType() == IntPtrTy && "malloc arg is wrong size");
  // Create the call to Malloc.
  Module *M = BB->getParent()->getParent();
  Type *BPTy = PointerType::getUnqual(Context);
  FunctionCallee MallocFunc = MallocF;
  if (!MallocFunc)
    // prototype malloc as "void *malloc(size_t)"
    MallocFunc = M->getOrInsertFunction("malloc", BPTy, IntPtrTy);
  CallInst *MCall = CreateCall(MallocFunc, AllocSize, OpB, Name);

  MCall->setTailCall();
  if (Function *F = dyn_cast<Function>(MallocFunc.getCallee())) {
    MCall->setCallingConv(F->getCallingConv());
    F->setReturnDoesNotAlias();
  }

  assert(!MCall->getType()->isVoidTy() && "Malloc has void return type");

  return MCall;
}

CallInst *IRBuilderBase::CreateMalloc(Type *IntPtrTy, Type *AllocTy,
                                      Value *AllocSize, Value *ArraySize,
                                      Function *MallocF, const Twine &Name) {

  return CreateMalloc(IntPtrTy, AllocTy, AllocSize, ArraySize, {}, MallocF,
                      Name);
}

/// CreateFree - Generate the IR for a call to the builtin free function.
CallInst *IRBuilderBase::CreateFree(Value *Source,
                                    ArrayRef<OperandBundleDef> Bundles) {
  assert(Source->getType()->isPointerTy() &&
         "Can not free something of nonpointer type!");

  Module *M = BB->getParent()->getParent();

  Type *VoidTy = Type::getVoidTy(M->getContext());
  Type *VoidPtrTy = PointerType::getUnqual(M->getContext());
  // prototype free as "void free(void*)"
  FunctionCallee FreeFunc = M->getOrInsertFunction("free", VoidTy, VoidPtrTy);
  CallInst *Result = CreateCall(FreeFunc, Source, Bundles, "");
  Result->setTailCall();
  if (Function *F = dyn_cast<Function>(FreeFunc.getCallee()))
    Result->setCallingConv(F->getCallingConv());

  return Result;
}

CallInst *IRBuilderBase::CreateElementUnorderedAtomicMemMove(
    Value *Dst, Align DstAlign, Value *Src, Align SrcAlign, Value *Size,
    uint32_t ElementSize, const AAMDNodes &AAInfo) {
  assert(DstAlign >= ElementSize &&
         "Pointer alignment must be at least element size");
  assert(SrcAlign >= ElementSize &&
         "Pointer alignment must be at least element size");
  Value *Ops[] = {Dst, Src, Size, getInt32(ElementSize)};
  Type *Tys[] = {Dst->getType(), Src->getType(), Size->getType()};

  CallInst *CI = CreateIntrinsicWithoutFolding(
      Intrinsic::memmove_element_unordered_atomic, Tys, Ops);

  // Set the alignment of the pointer args.
  CI->addParamAttr(0, Attribute::getWithAlignment(CI->getContext(), DstAlign));
  CI->addParamAttr(1, Attribute::getWithAlignment(CI->getContext(), SrcAlign));
  CI->setAAMetadata(AAInfo);
  return CI;
}

Value *IRBuilderBase::getReductionIntrinsic(Intrinsic::ID ID, Value *Src) {
  Value *Ops[] = {Src};
  Type *Tys[] = { Src->getType() };
  return CreateIntrinsic(ID, Tys, Ops);
}

Value *IRBuilderBase::CreateFAddReduce(Value *Acc, Value *Src) {
  Value *Ops[] = {Acc, Src};
  return CreateIntrinsic(Intrinsic::vector_reduce_fadd, {Src->getType()}, Ops);
}

Value *IRBuilderBase::CreateFMulReduce(Value *Acc, Value *Src) {
  Value *Ops[] = {Acc, Src};
  return CreateIntrinsic(Intrinsic::vector_reduce_fmul, {Src->getType()}, Ops);
}

Value *IRBuilderBase::CreateAddReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_add, Src);
}

Value *IRBuilderBase::CreateMulReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_mul, Src);
}

Value *IRBuilderBase::CreateAndReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_and, Src);
}

Value *IRBuilderBase::CreateOrReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_or, Src);
}

Value *IRBuilderBase::CreateXorReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_xor, Src);
}

Value *IRBuilderBase::CreateIntMaxReduce(Value *Src, bool IsSigned) {
  auto ID =
      IsSigned ? Intrinsic::vector_reduce_smax : Intrinsic::vector_reduce_umax;
  return getReductionIntrinsic(ID, Src);
}

Value *IRBuilderBase::CreateIntMinReduce(Value *Src, bool IsSigned) {
  auto ID =
      IsSigned ? Intrinsic::vector_reduce_smin : Intrinsic::vector_reduce_umin;
  return getReductionIntrinsic(ID, Src);
}

Value *IRBuilderBase::CreateFPMaxReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_fmax, Src);
}

Value *IRBuilderBase::CreateFPMinReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_fmin, Src);
}

Value *IRBuilderBase::CreateFPMaximumReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_fmaximum, Src);
}

Value *IRBuilderBase::CreateFPMinimumReduce(Value *Src) {
  return getReductionIntrinsic(Intrinsic::vector_reduce_fminimum, Src);
}

CallInst *IRBuilderBase::CreateLifetimeStart(Value *Ptr) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "lifetime.start only applies to pointers.");
  return CreateIntrinsicWithoutFolding(Intrinsic::lifetime_start,
                                       {Ptr->getType()}, {Ptr});
}

CallInst *IRBuilderBase::CreateLifetimeEnd(Value *Ptr) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "lifetime.end only applies to pointers.");
  return CreateIntrinsicWithoutFolding(Intrinsic::lifetime_end,
                                       {Ptr->getType()}, {Ptr});
}

CallInst *IRBuilderBase::CreateInvariantStart(Value *Ptr, ConstantInt *Size) {

  assert(isa<PointerType>(Ptr->getType()) &&
         "invariant.start only applies to pointers.");
  if (!Size)
    Size = getInt64(-1);
  else
    assert(Size->getType() == getInt64Ty() &&
           "invariant.start requires the size to be an i64");

  Value *Ops[] = {Size, Ptr};
  // Fill in the single overloaded type: memory object type.
  Type *ObjectPtr[1] = {Ptr->getType()};
  return CreateIntrinsicWithoutFolding(Intrinsic::invariant_start, ObjectPtr,
                                       Ops);
}

static MaybeAlign getAlign(Value *Ptr) {
  if (auto *V = dyn_cast<GlobalVariable>(Ptr))
    return V->getAlign();
  if (auto *A = dyn_cast<GlobalAlias>(Ptr))
    return getAlign(A->getAliaseeObject());
  return {};
}

CallInst *IRBuilderBase::CreateThreadLocalAddress(Value *Ptr) {
  assert(isa<GlobalValue>(Ptr) && cast<GlobalValue>(Ptr)->isThreadLocal() &&
         "threadlocal_address only applies to thread local variables.");
  CallInst *CI = CreateIntrinsicWithoutFolding(
      llvm::Intrinsic::threadlocal_address, {Ptr->getType()}, {Ptr});
  if (MaybeAlign A = getAlign(Ptr)) {
    CI->addParamAttr(0, Attribute::getWithAlignment(CI->getContext(), *A));
    CI->addRetAttr(Attribute::getWithAlignment(CI->getContext(), *A));
  }
  return CI;
}

CallInst *IRBuilderBase::CreateAssumption(Value *Cond) {
  assert(Cond->getType() == getInt1Ty() &&
         "an assumption condition must be of type i1");
  return CreateIntrinsicWithoutFolding(Intrinsic::assume, /*OverloadTypes=*/{},
                                       {Cond});
}

CallInst *
IRBuilderBase::CreateAssumption(ArrayRef<OperandBundleDef> OpBundles) {
  Value *Args[] = {ConstantInt::getTrue(getContext())};
  return CreateIntrinsicWithoutFolding(
      Intrinsic::assume, /*OverloadTypes=*/{}, Args,
      /*FMFSource=*/nullptr, /*Name=*/"", OpBundles);
}

Instruction *IRBuilderBase::CreateNoAliasScopeDeclaration(Value *Scope) {
  return CreateIntrinsicWithoutFolding(
      Intrinsic::experimental_noalias_scope_decl, {}, {Scope});
}

/// Create a call to a Masked Load intrinsic.
/// \p Ty        - vector type to load
/// \p Ptr       - base pointer for the load
/// \p Alignment - alignment of the source location
/// \p Mask      - vector of booleans which indicates what vector lanes should
///                be accessed in memory
/// \p PassThru  - pass-through value that is used to fill the masked-off lanes
///                of the result
/// \p Name      - name of the result variable
CallInst *IRBuilderBase::CreateMaskedLoad(Type *Ty, Value *Ptr, Align Alignment,
                                          Value *Mask, Value *PassThru,
                                          const Twine &Name) {
  auto *PtrTy = cast<PointerType>(Ptr->getType());
  assert(Ty->isVectorTy() && "Type should be vector");
  assert(Mask && "Mask should not be all-ones (null)");
  if (!PassThru)
    PassThru = PoisonValue::get(Ty);
  Type *OverloadedTypes[] = { Ty, PtrTy };
  Value *Ops[] = {Ptr, Mask, PassThru};
  CallInst *CI =
      CreateMaskedIntrinsic(Intrinsic::masked_load, Ops, OverloadedTypes, Name);
  CI->addParamAttr(0, Attribute::getWithAlignment(CI->getContext(), Alignment));
  return CI;
}

/// Create a call to a Masked Store intrinsic.
/// \p Val       - data to be stored,
/// \p Ptr       - base pointer for the store
/// \p Alignment - alignment of the destination location
/// \p Mask      - vector of booleans which indicates what vector lanes should
///                be accessed in memory
CallInst *IRBuilderBase::CreateMaskedStore(Value *Val, Value *Ptr,
                                           Align Alignment, Value *Mask) {
  auto *PtrTy = cast<PointerType>(Ptr->getType());
  Type *DataTy = Val->getType();
  assert(DataTy->isVectorTy() && "Val should be a vector");
  assert(Mask && "Mask should not be all-ones (null)");
  Type *OverloadedTypes[] = { DataTy, PtrTy };
  Value *Ops[] = {Val, Ptr, Mask};
  CallInst *CI =
      CreateMaskedIntrinsic(Intrinsic::masked_store, Ops, OverloadedTypes);
  CI->addParamAttr(1, Attribute::getWithAlignment(CI->getContext(), Alignment));
  return CI;
}

/// Create a call to a Masked intrinsic, with given intrinsic Id,
/// an array of operands - Ops, and an array of overloaded types -
/// OverloadedTypes.
CallInst *IRBuilderBase::CreateMaskedIntrinsic(Intrinsic::ID Id,
                                               ArrayRef<Value *> Ops,
                                               ArrayRef<Type *> OverloadedTypes,
                                               const Twine &Name) {
  return CreateIntrinsicWithoutFolding(Id, OverloadedTypes, Ops, {}, Name);
}

/// Create a call to a Masked Gather intrinsic.
/// \p Ty       - vector type to gather
/// \p Ptrs     - vector of pointers for loading
/// \p Align    - alignment for one element
/// \p Mask     - vector of booleans which indicates what vector lanes should
///               be accessed in memory
/// \p PassThru - pass-through value that is used to fill the masked-off lanes
///               of the result
/// \p Name     - name of the result variable
CallInst *IRBuilderBase::CreateMaskedGather(Type *Ty, Value *Ptrs,
                                            Align Alignment, Value *Mask,
                                            Value *PassThru,
                                            const Twine &Name) {
  auto *VecTy = cast<VectorType>(Ty);
  ElementCount NumElts = VecTy->getElementCount();
  auto *PtrsTy = cast<VectorType>(Ptrs->getType());
  assert(NumElts == PtrsTy->getElementCount() && "Element count mismatch");

  if (!Mask)
    Mask = getAllOnesMask(NumElts);

  if (!PassThru)
    PassThru = PoisonValue::get(Ty);

  Type *OverloadedTypes[] = {Ty, PtrsTy};
  Value *Ops[] = {Ptrs, Mask, PassThru};

  // We specify only one type when we create this intrinsic. Types of other
  // arguments are derived from this type.
  CallInst *CI = CreateMaskedIntrinsic(Intrinsic::masked_gather, Ops,
                                       OverloadedTypes, Name);
  CI->addParamAttr(0, Attribute::getWithAlignment(CI->getContext(), Alignment));
  return CI;
}

/// Create a call to a Masked Scatter intrinsic.
/// \p Data  - data to be stored,
/// \p Ptrs  - the vector of pointers, where the \p Data elements should be
///            stored
/// \p Align - alignment for one element
/// \p Mask  - vector of booleans which indicates what vector lanes should
///            be accessed in memory
CallInst *IRBuilderBase::CreateMaskedScatter(Value *Data, Value *Ptrs,
                                             Align Alignment, Value *Mask) {
  auto *PtrsTy = cast<VectorType>(Ptrs->getType());
  auto *DataTy = cast<VectorType>(Data->getType());
  ElementCount NumElts = PtrsTy->getElementCount();

  if (!Mask)
    Mask = getAllOnesMask(NumElts);

  Type *OverloadedTypes[] = {DataTy, PtrsTy};
  Value *Ops[] = {Data, Ptrs, Mask};

  // We specify only one type when we create this intrinsic. Types of other
  // arguments are derived from this type.
  CallInst *CI =
      CreateMaskedIntrinsic(Intrinsic::masked_scatter, Ops, OverloadedTypes);
  CI->addParamAttr(1, Attribute::getWithAlignment(CI->getContext(), Alignment));
  return CI;
}

/// Create a call to Masked Expand Load intrinsic
/// \p Ty        - vector type to load
/// \p Ptr       - base pointer for the load
/// \p Align     - alignment of \p Ptr
/// \p Mask      - vector of booleans which indicates what vector lanes should
///                be accessed in memory
/// \p PassThru  - pass-through value that is used to fill the masked-off lanes
///                of the result
/// \p Name      - name of the result variable
CallInst *IRBuilderBase::CreateMaskedExpandLoad(Type *Ty, Value *Ptr,
                                                MaybeAlign Align, Value *Mask,
                                                Value *PassThru,
                                                const Twine &Name) {
  assert(Ty->isVectorTy() && "Type should be vector");
  assert(Mask && "Mask should not be all-ones (null)");
  if (!PassThru)
    PassThru = PoisonValue::get(Ty);
  Type *PtrTy = Ptr->getType();
  Type *OverloadedTypes[] = {Ty, PtrTy};
  Value *Ops[] = {Ptr, Mask, PassThru};
  CallInst *CI = CreateMaskedIntrinsic(Intrinsic::masked_expandload, Ops,
                                       OverloadedTypes, Name);
  if (Align)
    CI->addParamAttr(0, Attribute::getWithAlignment(CI->getContext(), *Align));
  return CI;
}

/// Create a call to Masked Compress Store intrinsic
/// \p Val       - data to be stored,
/// \p Ptr       - base pointer for the store
/// \p Align     - alignment of \p Ptr
/// \p Mask      - vector of booleans which indicates what vector lanes should
///                be accessed in memory
CallInst *IRBuilderBase::CreateMaskedCompressStore(Value *Val, Value *Ptr,
                                                   MaybeAlign Align,
                                                   Value *Mask) {
  Type *DataTy = Val->getType();
  assert(DataTy->isVectorTy() && "Val should be a vector");
  assert(Mask && "Mask should not be all-ones (null)");
  Type *PtrTy = Ptr->getType();
  Type *OverloadedTypes[] = {DataTy, PtrTy};
  Value *Ops[] = {Val, Ptr, Mask};
  CallInst *CI = CreateMaskedIntrinsic(Intrinsic::masked_compressstore, Ops,
                                       OverloadedTypes);
  if (Align)
    CI->addParamAttr(1, Attribute::getWithAlignment(CI->getContext(), *Align));
  return CI;
}

template <typename T0>
static std::vector<Value *>
getStatepointArgs(IRBuilderBase &B, uint64_t ID, uint32_t NumPatchBytes,
                  Value *ActualCallee, uint32_t Flags, ArrayRef<T0> CallArgs) {
  std::vector<Value *> Args;
  Args.push_back(B.getInt64(ID));
  Args.push_back(B.getInt32(NumPatchBytes));
  Args.push_back(ActualCallee);
  Args.push_back(B.getInt32(CallArgs.size()));
  Args.push_back(B.getInt32(Flags));
  llvm::append_range(Args, CallArgs);
  // GC Transition and Deopt args are now always handled via operand bundle.
  // They will be removed from the signature of gc.statepoint shortly.
  Args.push_back(B.getInt32(0));
  Args.push_back(B.getInt32(0));
  // GC args are now encoded in the gc-live operand bundle
  return Args;
}

template<typename T1, typename T2, typename T3>
static std::vector<OperandBundleDef>
getStatepointBundles(std::optional<ArrayRef<T1>> TransitionArgs,
                     std::optional<ArrayRef<T2>> DeoptArgs,
                     ArrayRef<T3> GCArgs) {
  std::vector<OperandBundleDef> Rval;
  if (DeoptArgs)
    Rval.emplace_back("deopt", SmallVector<Value *, 16>(*DeoptArgs));
  if (TransitionArgs)
    Rval.emplace_back("gc-transition",
                      SmallVector<Value *, 16>(*TransitionArgs));
  if (GCArgs.size())
    Rval.emplace_back("gc-live", SmallVector<Value *, 16>(GCArgs));
  return Rval;
}

template <typename T0, typename T1, typename T2, typename T3>
static CallInst *CreateGCStatepointCallCommon(
    IRBuilderBase *Builder, uint64_t ID, uint32_t NumPatchBytes,
    FunctionCallee ActualCallee, uint32_t Flags, ArrayRef<T0> CallArgs,
    std::optional<ArrayRef<T1>> TransitionArgs,
    std::optional<ArrayRef<T2>> DeoptArgs, ArrayRef<T3> GCArgs,
    const Twine &Name) {
  Module *M = Builder->GetInsertBlock()->getParent()->getParent();
  // Fill in the one generic type'd argument (the function is also vararg)
  Function *FnStatepoint = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::experimental_gc_statepoint,
      {ActualCallee.getCallee()->getType()});

  std::vector<Value *> Args = getStatepointArgs(
      *Builder, ID, NumPatchBytes, ActualCallee.getCallee(), Flags, CallArgs);

  CallInst *CI = Builder->CreateCall(
      FnStatepoint, Args,
      getStatepointBundles(TransitionArgs, DeoptArgs, GCArgs), Name);
  CI->addParamAttr(2,
                   Attribute::get(Builder->getContext(), Attribute::ElementType,
                                  ActualCallee.getFunctionType()));
  return CI;
}

CallInst *IRBuilderBase::CreateGCStatepointCall(
    uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualCallee,
    ArrayRef<Value *> CallArgs, std::optional<ArrayRef<Value *>> DeoptArgs,
    ArrayRef<Value *> GCArgs, const Twine &Name) {
  return CreateGCStatepointCallCommon<Value *, Value *, Value *, Value *>(
      this, ID, NumPatchBytes, ActualCallee, uint32_t(StatepointFlags::None),
      CallArgs, std::nullopt /* No Transition Args */, DeoptArgs, GCArgs, Name);
}

CallInst *IRBuilderBase::CreateGCStatepointCall(
    uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualCallee,
    uint32_t Flags, ArrayRef<Value *> CallArgs,
    std::optional<ArrayRef<Use>> TransitionArgs,
    std::optional<ArrayRef<Use>> DeoptArgs, ArrayRef<Value *> GCArgs,
    const Twine &Name) {
  return CreateGCStatepointCallCommon<Value *, Use, Use, Value *>(
      this, ID, NumPatchBytes, ActualCallee, Flags, CallArgs, TransitionArgs,
      DeoptArgs, GCArgs, Name);
}

CallInst *IRBuilderBase::CreateGCStatepointCall(
    uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualCallee,
    ArrayRef<Use> CallArgs, std::optional<ArrayRef<Value *>> DeoptArgs,
    ArrayRef<Value *> GCArgs, const Twine &Name) {
  return CreateGCStatepointCallCommon<Use, Value *, Value *, Value *>(
      this, ID, NumPatchBytes, ActualCallee, uint32_t(StatepointFlags::None),
      CallArgs, std::nullopt, DeoptArgs, GCArgs, Name);
}

template <typename T0, typename T1, typename T2, typename T3>
static InvokeInst *CreateGCStatepointInvokeCommon(
    IRBuilderBase *Builder, uint64_t ID, uint32_t NumPatchBytes,
    FunctionCallee ActualInvokee, BasicBlock *NormalDest,
    BasicBlock *UnwindDest, uint32_t Flags, ArrayRef<T0> InvokeArgs,
    std::optional<ArrayRef<T1>> TransitionArgs,
    std::optional<ArrayRef<T2>> DeoptArgs, ArrayRef<T3> GCArgs,
    const Twine &Name) {
  Module *M = Builder->GetInsertBlock()->getParent()->getParent();
  // Fill in the one generic type'd argument (the function is also vararg)
  Function *FnStatepoint = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::experimental_gc_statepoint,
      {ActualInvokee.getCallee()->getType()});

  std::vector<Value *> Args =
      getStatepointArgs(*Builder, ID, NumPatchBytes, ActualInvokee.getCallee(),
                        Flags, InvokeArgs);

  InvokeInst *II = Builder->CreateInvoke(
      FnStatepoint, NormalDest, UnwindDest, Args,
      getStatepointBundles(TransitionArgs, DeoptArgs, GCArgs), Name);
  II->addParamAttr(2,
                   Attribute::get(Builder->getContext(), Attribute::ElementType,
                                  ActualInvokee.getFunctionType()));
  return II;
}

InvokeInst *IRBuilderBase::CreateGCStatepointInvoke(
    uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualInvokee,
    BasicBlock *NormalDest, BasicBlock *UnwindDest,
    ArrayRef<Value *> InvokeArgs, std::optional<ArrayRef<Value *>> DeoptArgs,
    ArrayRef<Value *> GCArgs, const Twine &Name) {
  return CreateGCStatepointInvokeCommon<Value *, Value *, Value *, Value *>(
      this, ID, NumPatchBytes, ActualInvokee, NormalDest, UnwindDest,
      uint32_t(StatepointFlags::None), InvokeArgs,
      std::nullopt /* No Transition Args*/, DeoptArgs, GCArgs, Name);
}

InvokeInst *IRBuilderBase::CreateGCStatepointInvoke(
    uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualInvokee,
    BasicBlock *NormalDest, BasicBlock *UnwindDest, uint32_t Flags,
    ArrayRef<Value *> InvokeArgs, std::optional<ArrayRef<Use>> TransitionArgs,
    std::optional<ArrayRef<Use>> DeoptArgs, ArrayRef<Value *> GCArgs,
    const Twine &Name) {
  return CreateGCStatepointInvokeCommon<Value *, Use, Use, Value *>(
      this, ID, NumPatchBytes, ActualInvokee, NormalDest, UnwindDest, Flags,
      InvokeArgs, TransitionArgs, DeoptArgs, GCArgs, Name);
}

InvokeInst *IRBuilderBase::CreateGCStatepointInvoke(
    uint64_t ID, uint32_t NumPatchBytes, FunctionCallee ActualInvokee,
    BasicBlock *NormalDest, BasicBlock *UnwindDest, ArrayRef<Use> InvokeArgs,
    std::optional<ArrayRef<Value *>> DeoptArgs, ArrayRef<Value *> GCArgs,
    const Twine &Name) {
  return CreateGCStatepointInvokeCommon<Use, Value *, Value *, Value *>(
      this, ID, NumPatchBytes, ActualInvokee, NormalDest, UnwindDest,
      uint32_t(StatepointFlags::None), InvokeArgs, std::nullopt, DeoptArgs,
      GCArgs, Name);
}

CallInst *IRBuilderBase::CreateGCResult(Instruction *Statepoint,
                                        Type *ResultType, const Twine &Name) {
  Intrinsic::ID ID = Intrinsic::experimental_gc_result;
  Type *Types[] = {ResultType};

  Value *Args[] = {Statepoint};
  return CreateIntrinsicWithoutFolding(ID, Types, Args, {}, Name);
}

CallInst *IRBuilderBase::CreateGCRelocate(Instruction *Statepoint,
                                          int BaseOffset, int DerivedOffset,
                                          Type *ResultType, const Twine &Name) {
  Type *Types[] = {ResultType};

  Value *Args[] = {Statepoint, getInt32(BaseOffset), getInt32(DerivedOffset)};
  return CreateIntrinsicWithoutFolding(Intrinsic::experimental_gc_relocate,
                                       Types, Args, {}, Name);
}

CallInst *IRBuilderBase::CreateGCGetPointerBase(Value *DerivedPtr,
                                                const Twine &Name) {
  Type *PtrTy = DerivedPtr->getType();
  return CreateIntrinsicWithoutFolding(
      Intrinsic::experimental_gc_get_pointer_base, {PtrTy, PtrTy}, {DerivedPtr},
      {}, Name);
}

CallInst *IRBuilderBase::CreateGCGetPointerOffset(Value *DerivedPtr,
                                                  const Twine &Name) {
  Type *PtrTy = DerivedPtr->getType();
  return CreateIntrinsicWithoutFolding(
      Intrinsic::experimental_gc_get_pointer_offset, {PtrTy}, {DerivedPtr}, {},
      Name);
}

Value *IRBuilderBase::CreateUnaryIntrinsic(Intrinsic::ID ID, Value *Op,
                                           FMFSource FMFSource,
                                           const Twine &Name) {
  Module *M = BB->getModule();
  Function *Fn = Intrinsic::getOrInsertDeclaration(M, ID, Op->getType());
  if (Value *V =
          Folder.FoldIntrinsic(ID, Op, Fn->getReturnType(), FMFSource.get(FMF),
                               GetInsertBlock()->getParent()))
    return V;
  return createCallHelper(Fn, Op, Name, FMFSource);
}

Value *IRBuilderBase::CreateBinaryIntrinsic(Intrinsic::ID ID, Value *LHS,
                                            Value *RHS, FMFSource FMFSource,
                                            const Twine &Name) {
  Module *M = BB->getModule();
  Function *Fn = Intrinsic::getOrInsertDeclaration(M, ID, {LHS->getType()});
  if (Value *V = Folder.FoldIntrinsic(ID, {LHS, RHS}, Fn->getReturnType(),
                                      FMFSource.get(FMF),
                                      GetInsertBlock()->getParent()))
    return V;
  return createCallHelper(Fn, {LHS, RHS}, Name, FMFSource);
}

CallInst *IRBuilderBase::CreateIntrinsicWithoutFolding(
    Intrinsic::ID ID, ArrayRef<Type *> OverloadTypes, ArrayRef<Value *> Args,
    FMFSource FMFSource, const Twine &Name,
    ArrayRef<OperandBundleDef> OpBundles) {
  Module *M = BB->getModule();
  Function *Fn = Intrinsic::getOrInsertDeclaration(M, ID, OverloadTypes);
  return createCallHelper(Fn, Args, Name, FMFSource, OpBundles);
}

CallInst *IRBuilderBase::CreateIntrinsicWithoutFolding(Type *RetTy,
                                                       Intrinsic::ID ID,
                                                       ArrayRef<Value *> Args,
                                                       FMFSource FMFSource,
                                                       const Twine &Name) {
  Module *M = BB->getModule();
  SmallVector<Type *> ArgTys = llvm::map_to_vector(Args, &Value::getType);
  Function *Fn = Intrinsic::getOrInsertDeclaration(M, ID, RetTy, ArgTys);
  return createCallHelper(Fn, Args, Name, FMFSource);
}

Value *IRBuilderBase::CreateIntrinsic(Intrinsic::ID ID,
                                      ArrayRef<Type *> OverloadTypes,
                                      ArrayRef<Value *> Args,
                                      FMFSource FMFSource, const Twine &Name,
                                      ArrayRef<OperandBundleDef> OpBundles,
                                      function_ref<void(CallInst *)> SetFn) {
  Type *RetTy = Intrinsic::getType(Context, ID, OverloadTypes)->getReturnType();
  if (Value *V = Folder.FoldIntrinsic(ID, Args, RetTy, FMFSource.get(FMF),
                                      GetInsertBlock()->getParent()))
    return V;
  CallInst *CI = CreateIntrinsicWithoutFolding(ID, OverloadTypes, Args,
                                               FMFSource, Name, OpBundles);
  SetFn(CI);
  return CI;
}

Value *IRBuilderBase::CreateIntrinsic(Type *RetTy, Intrinsic::ID ID,
                                      ArrayRef<Value *> Args,
                                      FMFSource FMFSource, const Twine &Name,
                                      function_ref<void(CallInst *)> SetFn) {
  if (Value *V = Folder.FoldIntrinsic(ID, Args, RetTy, FMFSource.get(FMF),
                                      GetInsertBlock()->getParent()))
    return V;
  CallInst *CI =
      CreateIntrinsicWithoutFolding(RetTy, ID, Args, FMFSource, Name);
  SetFn(CI);
  return CI;
}

CallInst *IRBuilderBase::CreateConstrainedFPBinOp(
    Intrinsic::ID ID, Value *L, Value *R, FMFSource FMFSource,
    const Twine &Name, MDNode *FPMathTag, std::optional<RoundingMode> Rounding,
    std::optional<fp::ExceptionBehavior> Except) {
  Value *RoundingV = getConstrainedFPRounding(Rounding);
  Value *ExceptV = getConstrainedFPExcept(Except);

  FastMathFlags UseFMF = FMFSource.get(FMF);
  CallInst *C = CreateIntrinsicWithoutFolding(
      ID, {L->getType()}, {L, R, RoundingV, ExceptV}, nullptr, Name, {});
  setConstrainedFPCallAttr(C);
  setFPAttrs(C, FPMathTag, UseFMF);
  return C;
}

CallInst *IRBuilderBase::CreateConstrainedFPIntrinsic(
    Intrinsic::ID ID, ArrayRef<Type *> Types, ArrayRef<Value *> Args,
    FMFSource FMFSource, const Twine &Name, MDNode *FPMathTag,
    std::optional<RoundingMode> Rounding,
    std::optional<fp::ExceptionBehavior> Except) {
  Value *RoundingV = getConstrainedFPRounding(Rounding);
  Value *ExceptV = getConstrainedFPExcept(Except);

  FastMathFlags UseFMF = FMFSource.get(FMF);

  llvm::SmallVector<Value *, 5> ExtArgs(Args);
  ExtArgs.push_back(RoundingV);
  ExtArgs.push_back(ExceptV);
  CallInst *C =
      CreateIntrinsicWithoutFolding(ID, Types, ExtArgs, nullptr, Name, {});
  setConstrainedFPCallAttr(C);
  setFPAttrs(C, FPMathTag, UseFMF);
  return C;
}

CallInst *IRBuilderBase::CreateConstrainedFPUnroundedBinOp(
    Intrinsic::ID ID, Value *L, Value *R, FMFSource FMFSource,
    const Twine &Name, MDNode *FPMathTag,
    std::optional<fp::ExceptionBehavior> Except) {
  Value *ExceptV = getConstrainedFPExcept(Except);

  FastMathFlags UseFMF = FMFSource.get(FMF);
  CallInst *C = CreateIntrinsicWithoutFolding(
      ID, {L->getType()}, {L, R, ExceptV}, nullptr, Name, {});
  setConstrainedFPCallAttr(C);
  setFPAttrs(C, FPMathTag, UseFMF);
  return C;
}

Value *IRBuilderBase::CreateNAryOp(unsigned Opc, ArrayRef<Value *> Ops,
                                   const Twine &Name, MDNode *FPMathTag) {
  if (Instruction::isBinaryOp(Opc)) {
    assert(Ops.size() == 2 && "Invalid number of operands!");
    return CreateBinOp(static_cast<Instruction::BinaryOps>(Opc),
                       Ops[0], Ops[1], Name, FPMathTag);
  }
  if (Instruction::isUnaryOp(Opc)) {
    assert(Ops.size() == 1 && "Invalid number of operands!");
    return CreateUnOp(static_cast<Instruction::UnaryOps>(Opc),
                      Ops[0], Name, FPMathTag);
  }
  llvm_unreachable("Unexpected opcode!");
}

CallInst *IRBuilderBase::CreateConstrainedFPCast(
    Intrinsic::ID ID, Value *V, Type *DestTy, FMFSource FMFSource,
    const Twine &Name, MDNode *FPMathTag, std::optional<RoundingMode> Rounding,
    std::optional<fp::ExceptionBehavior> Except) {
  Value *ExceptV = getConstrainedFPExcept(Except);

  FastMathFlags UseFMF = FMFSource.get(FMF);

  CallInst *C;
  if (Intrinsic::hasConstrainedFPRoundingModeOperand(ID)) {
    Value *RoundingV = getConstrainedFPRounding(Rounding);
    C = CreateIntrinsicWithoutFolding(
        ID, {DestTy, V->getType()}, {V, RoundingV, ExceptV}, nullptr, Name, {});
  } else
    C = CreateIntrinsicWithoutFolding(ID, {DestTy, V->getType()}, {V, ExceptV},
                                      nullptr, Name, {});
  setConstrainedFPCallAttr(C);

  if (isa<FPMathOperator>(C))
    setFPAttrs(C, FPMathTag, UseFMF);
  return C;
}

Value *IRBuilderBase::CreateFCmpHelper(CmpInst::Predicate P, Value *LHS,
                                       Value *RHS, const Twine &Name,
                                       MDNode *FPMathTag, FMFSource FMFSource,
                                       bool IsSignaling) {
  if (IsFPConstrained) {
    auto ID = IsSignaling ? Intrinsic::experimental_constrained_fcmps
                          : Intrinsic::experimental_constrained_fcmp;
    return CreateConstrainedFPCmp(ID, P, LHS, RHS, Name);
  }

  if (auto *V = Folder.FoldCmp(P, LHS, RHS))
    return V;
  return Insert(
      setFPAttrs(new FCmpInst(P, LHS, RHS), FPMathTag, FMFSource.get(FMF)),
      Name);
}

CallInst *IRBuilderBase::CreateConstrainedFPCmp(
    Intrinsic::ID ID, CmpInst::Predicate P, Value *L, Value *R,
    const Twine &Name, std::optional<fp::ExceptionBehavior> Except) {
  Value *PredicateV = getConstrainedFPPredicate(P);
  Value *ExceptV = getConstrainedFPExcept(Except);

  CallInst *C = CreateIntrinsicWithoutFolding(
      ID, {L->getType()}, {L, R, PredicateV, ExceptV}, nullptr, Name, {});
  setConstrainedFPCallAttr(C);
  return C;
}

CallInst *IRBuilderBase::CreateConstrainedFPCall(
    Function *Callee, ArrayRef<Value *> Args, const Twine &Name,
    std::optional<RoundingMode> Rounding,
    std::optional<fp::ExceptionBehavior> Except) {
  llvm::SmallVector<Value *, 6> UseArgs(Args);

  if (Intrinsic::hasConstrainedFPRoundingModeOperand(Callee->getIntrinsicID()))
    UseArgs.push_back(getConstrainedFPRounding(Rounding));
  UseArgs.push_back(getConstrainedFPExcept(Except));

  CallInst *C = CreateCall(Callee, UseArgs, Name);
  setConstrainedFPCallAttr(C);
  return C;
}

Value *IRBuilderBase::CreateSelectWithUnknownProfile(Value *C, Value *True,
                                                     Value *False,
                                                     StringRef PassName,
                                                     const Twine &Name) {
  Value *Ret = CreateSelectFMF(C, True, False, {}, Name);
  if (auto *SI = dyn_cast<SelectInst>(Ret)) {
    setExplicitlyUnknownBranchWeightsIfProfiled(*SI, PassName);
  }
  return Ret;
}

Value *IRBuilderBase::CreateSelectFMFWithUnknownProfile(Value *C, Value *True,
                                                        Value *False,
                                                        FMFSource FMFSource,
                                                        StringRef PassName,
                                                        const Twine &Name) {
  Value *Ret = CreateSelectFMF(C, True, False, FMFSource, Name);
  if (auto *SI = dyn_cast<SelectInst>(Ret))
    setExplicitlyUnknownBranchWeightsIfProfiled(*SI, PassName);
  return Ret;
}

Value *IRBuilderBase::CreateSelect(Value *C, Value *True, Value *False,
                                   const Twine &Name, Instruction *MDFrom) {
  return CreateSelectFMF(C, True, False, {}, Name, MDFrom);
}

Value *IRBuilderBase::CreateSelectFMF(Value *C, Value *True, Value *False,
                                      FMFSource FMFSource, const Twine &Name,
                                      Instruction *MDFrom) {
  if (auto *V = Folder.FoldSelect(C, True, False, FMFSource.get(FMF)))
    return V;

  SelectInst *Sel = SelectInst::Create(C, True, False);
  if (MDFrom) {
    MDNode *Prof = MDFrom->getMetadata(LLVMContext::MD_prof);
    MDNode *Unpred = MDFrom->getMetadata(LLVMContext::MD_unpredictable);
    Sel = addBranchMetadata(Sel, Prof, Unpred);
  }
  if (isa<FPMathOperator>(Sel))
    setFPAttrs(Sel, /*MDNode=*/nullptr, FMFSource.get(FMF));
  return Insert(Sel, Name);
}

Value *IRBuilderBase::CreatePtrDiff(Value *LHS, Value *RHS, const Twine &Name,
                                    bool IsNUW) {
  assert(LHS->getType() == RHS->getType() &&
         "Pointer subtraction operand types must match!");
  Value *LHSAddr = CreatePtrToAddr(LHS);
  Value *RHSAddr = CreatePtrToAddr(RHS);
  return CreateSub(LHSAddr, RHSAddr, Name, IsNUW);
}
Value *IRBuilderBase::CreatePtrDiff(Type *ElemTy, Value *LHS, Value *RHS,
                                    const Twine &Name) {
  const DataLayout &DL = BB->getDataLayout();
  TypeSize ElemSize = DL.getTypeAllocSize(ElemTy);
  if (ElemSize == TypeSize::getFixed(1))
    return CreatePtrDiff(LHS, RHS, Name);

  Value *Diff = CreatePtrDiff(LHS, RHS);
  return CreateExactSDiv(Diff, CreateTypeSize(Diff->getType(), ElemSize), Name);
}

Value *IRBuilderBase::CreateLaunderInvariantGroup(Value *Ptr) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "launder.invariant.group only applies to pointers.");
  auto *PtrType = Ptr->getType();
  Module *M = BB->getParent()->getParent();
  Function *FnLaunderInvariantGroup = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::launder_invariant_group, {PtrType});

  assert(FnLaunderInvariantGroup->getReturnType() == PtrType &&
         FnLaunderInvariantGroup->getFunctionType()->getParamType(0) ==
             PtrType &&
         "LaunderInvariantGroup should take and return the same type");

  return CreateCall(FnLaunderInvariantGroup, {Ptr});
}

Value *IRBuilderBase::CreateStripInvariantGroup(Value *Ptr) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "strip.invariant.group only applies to pointers.");

  auto *PtrType = Ptr->getType();
  Module *M = BB->getParent()->getParent();
  Function *FnStripInvariantGroup = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::strip_invariant_group, {PtrType});

  assert(FnStripInvariantGroup->getReturnType() == PtrType &&
         FnStripInvariantGroup->getFunctionType()->getParamType(0) ==
             PtrType &&
         "StripInvariantGroup should take and return the same type");

  return CreateCall(FnStripInvariantGroup, {Ptr});
}

Value *IRBuilderBase::CreateVectorReverse(Value *V, const Twine &Name) {
  auto *Ty = cast<VectorType>(V->getType());
  if (isa<ScalableVectorType>(Ty)) {
    Module *M = BB->getParent()->getParent();
    Function *F =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::vector_reverse, Ty);
    return Insert(CallInst::Create(F, V), Name);
  }
  // Keep the original behaviour for fixed vector
  SmallVector<int, 8> ShuffleMask;
  int NumElts = Ty->getElementCount().getKnownMinValue();
  for (int i = 0; i < NumElts; ++i)
    ShuffleMask.push_back(NumElts - i - 1);
  return CreateShuffleVector(V, ShuffleMask, Name);
}

static SmallVector<int, 8> getSpliceMask(int64_t Imm, unsigned NumElts) {
  unsigned Idx = (NumElts + Imm) % NumElts;
  SmallVector<int, 8> Mask;
  for (unsigned I = 0; I < NumElts; ++I)
    Mask.push_back(Idx + I);
  return Mask;
}

Value *IRBuilderBase::CreateVectorSpliceLeft(Value *V1, Value *V2,
                                             Value *Offset, const Twine &Name) {
  assert(isa<VectorType>(V1->getType()) && "Unexpected type");
  assert(V1->getType() == V2->getType() &&
         "Splice expects matching operand types!");

  // Emit a shufflevector for fixed vectors with a constant offset
  if (auto *COffset = dyn_cast<ConstantInt>(Offset))
    if (auto *FVTy = dyn_cast<FixedVectorType>(V1->getType()))
      return CreateShuffleVector(
          V1, V2,
          getSpliceMask(COffset->getZExtValue(), FVTy->getNumElements()));

  return CreateIntrinsic(Intrinsic::vector_splice_left, V1->getType(),
                         {V1, V2, Offset}, {}, Name);
}

Value *IRBuilderBase::CreateVectorSpliceRight(Value *V1, Value *V2,
                                              Value *Offset,
                                              const Twine &Name) {
  assert(isa<VectorType>(V1->getType()) && "Unexpected type");
  assert(V1->getType() == V2->getType() &&
         "Splice expects matching operand types!");

  // Emit a shufflevector for fixed vectors with a constant offset
  if (auto *COffset = dyn_cast<ConstantInt>(Offset))
    if (auto *FVTy = dyn_cast<FixedVectorType>(V1->getType()))
      return CreateShuffleVector(
          V1, V2,
          getSpliceMask(-COffset->getZExtValue(), FVTy->getNumElements()));

  return CreateIntrinsic(Intrinsic::vector_splice_right, V1->getType(),
                         {V1, V2, Offset}, {}, Name);
}

Value *IRBuilderBase::CreateVectorSplat(unsigned NumElts, Value *V,
                                        const Twine &Name) {
  auto EC = ElementCount::getFixed(NumElts);
  return CreateVectorSplat(EC, V, Name);
}

Value *IRBuilderBase::CreateVectorSplat(ElementCount EC, Value *V,
                                        const Twine &Name) {
  assert(EC.isNonZero() && "Cannot splat to an empty vector!");

  // First insert it into a poison vector so we can shuffle it.
  Value *Poison = PoisonValue::get(VectorType::get(V->getType(), EC));
  V = CreateInsertElement(Poison, V, getInt64(0), Name + ".splatinsert");

  // Shuffle the value across the desired number of elements.
  SmallVector<int, 16> Zeros;
  Zeros.resize(EC.getKnownMinValue());
  return CreateShuffleVector(V, Zeros, Name + ".splat");
}

Value *IRBuilderBase::CreateVectorInterleave(ArrayRef<Value *> Ops,
                                             const Twine &Name) {
  assert(Ops.size() >= 2 && Ops.size() <= 8 &&
         "Unexpected number of operands to interleave");

  // Make sure all operands are the same type.
  assert(isa<VectorType>(Ops[0]->getType()) && "Unexpected type");

#ifndef NDEBUG
  for (unsigned I = 1; I < Ops.size(); I++) {
    assert(Ops[I]->getType() == Ops[0]->getType() &&
           "Vector interleave expects matching operand types!");
  }
#endif

  unsigned IID = Intrinsic::getInterleaveIntrinsicID(Ops.size());
  auto *SubvecTy = cast<VectorType>(Ops[0]->getType());
  Type *DestTy = VectorType::get(SubvecTy->getElementType(),
                                 SubvecTy->getElementCount() * Ops.size());
  return CreateIntrinsic(IID, {DestTy}, Ops, {}, Name);
}

Value *IRBuilderBase::CreatePreserveArrayAccessIndex(Type *ElTy, Value *Base,
                                                     unsigned Dimension,
                                                     unsigned LastIndex,
                                                     MDNode *DbgInfo) {
  auto *BaseType = Base->getType();
  assert(isa<PointerType>(BaseType) &&
         "Invalid Base ptr type for preserve.array.access.index.");

  Value *LastIndexV = getInt32(LastIndex);
  Constant *Zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
  SmallVector<Value *, 4> IdxList(Dimension, Zero);
  IdxList.push_back(LastIndexV);

  Type *ResultType = GetElementPtrInst::getGEPReturnType(Base, IdxList);

  Value *DimV = getInt32(Dimension);
  CallInst *Fn = CreateIntrinsicWithoutFolding(
      Intrinsic::preserve_array_access_index, {ResultType, BaseType},
      {Base, DimV, LastIndexV});
  Fn->addParamAttr(
      0, Attribute::get(Fn->getContext(), Attribute::ElementType, ElTy));
  if (DbgInfo)
    Fn->setMetadata(LLVMContext::MD_preserve_access_index, DbgInfo);

  return Fn;
}

Value *IRBuilderBase::CreatePreserveUnionAccessIndex(
    Value *Base, unsigned FieldIndex, MDNode *DbgInfo) {
  assert(isa<PointerType>(Base->getType()) &&
         "Invalid Base ptr type for preserve.union.access.index.");
  auto *BaseType = Base->getType();

  Value *DIIndex = getInt32(FieldIndex);
  CallInst *Fn =
      CreateIntrinsicWithoutFolding(Intrinsic::preserve_union_access_index,
                                    {BaseType, BaseType}, {Base, DIIndex});
  if (DbgInfo)
    Fn->setMetadata(LLVMContext::MD_preserve_access_index, DbgInfo);

  return Fn;
}

Value *IRBuilderBase::CreatePreserveStructAccessIndex(
    Type *ElTy, Value *Base, unsigned Index, unsigned FieldIndex,
    MDNode *DbgInfo) {
  auto *BaseType = Base->getType();
  assert(isa<PointerType>(BaseType) &&
         "Invalid Base ptr type for preserve.struct.access.index.");

  Value *GEPIndex = getInt32(Index);
  Constant *Zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
  Type *ResultType =
      GetElementPtrInst::getGEPReturnType(Base, {Zero, GEPIndex});

  Value *DIIndex = getInt32(FieldIndex);
  CallInst *Fn = CreateIntrinsicWithoutFolding(
      Intrinsic::preserve_struct_access_index, {ResultType, BaseType},
      {Base, GEPIndex, DIIndex});
  Fn->addParamAttr(
      0, Attribute::get(Fn->getContext(), Attribute::ElementType, ElTy));
  if (DbgInfo)
    Fn->setMetadata(LLVMContext::MD_preserve_access_index, DbgInfo);

  return Fn;
}

Value *IRBuilderBase::createIsFPClass(Value *FPNum, unsigned Test) {
  ConstantInt *TestV = getInt32(Test);
  return CreateIntrinsic(Intrinsic::is_fpclass, {FPNum->getType()},
                         {FPNum, TestV});
}

CallInst *IRBuilderBase::CreateAlignmentAssumptionHelper(const DataLayout &DL,
                                                         Value *PtrValue,
                                                         Value *AlignValue,
                                                         Value *OffsetValue) {
  SmallVector<Value *, 4> Vals({PtrValue, AlignValue});
  if (OffsetValue)
    Vals.push_back(OffsetValue);
  OperandBundleDefT<Value *> AlignOpB("align", Vals);
  return CreateAssumption({AlignOpB});
}

CallInst *IRBuilderBase::CreateAlignmentAssumption(const DataLayout &DL,
                                                   Value *PtrValue,
                                                   uint64_t Alignment,
                                                   Value *OffsetValue) {
  assert(isa<PointerType>(PtrValue->getType()) &&
         "trying to create an alignment assumption on a non-pointer?");
  assert(Alignment != 0 && "Invalid Alignment");
  Value *AlignValue = ConstantInt::get(getInt64Ty(), Alignment);
  return CreateAlignmentAssumptionHelper(DL, PtrValue, AlignValue, OffsetValue);
}

CallInst *IRBuilderBase::CreateAlignmentAssumption(const DataLayout &DL,
                                                   Value *PtrValue,
                                                   Value *Alignment,
                                                   Value *OffsetValue) {
  assert(isa<PointerType>(PtrValue->getType()) &&
         "trying to create an alignment assumption on a non-pointer?");
  return CreateAlignmentAssumptionHelper(DL, PtrValue, Alignment, OffsetValue);
}

CallInst *IRBuilderBase::CreateDereferenceableAssumption(Value *PtrValue,
                                                         Value *SizeValue) {
  assert(isa<PointerType>(PtrValue->getType()) &&
         "trying to create a deferenceable assumption on a non-pointer?");
  SmallVector<Value *, 4> Vals({PtrValue, SizeValue});
  OperandBundleDefT<Value *> DereferenceableOpB("dereferenceable", Vals);
  return CreateAssumption({DereferenceableOpB});
}

CallInst *IRBuilderBase::CreateNonnullAssumption(Value *PtrValue) {
  assert(isa<PointerType>(PtrValue->getType()) &&
         "trying to create a nonnull assumption on a non-pointer?");
  return CreateAssumption(OperandBundleDef("nonnull", PtrValue));
}

IRBuilderDefaultInserter::~IRBuilderDefaultInserter() = default;
IRBuilderCallbackInserter::~IRBuilderCallbackInserter() = default;
IRBuilderFolder::~IRBuilderFolder() = default;
void ConstantFolder::anchor() {}
void NoFolder::anchor() {}
