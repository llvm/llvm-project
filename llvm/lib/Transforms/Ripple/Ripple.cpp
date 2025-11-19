//===--------------- Ripple.cpp - Expand RIpple intrinsics ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands Ripple intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Ripple/Ripple.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <assert.h>
#include <iterator>
#include <string>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "ripple"

////////////////////////////////////////////////////////////////////////////////
///                             TensorShapeAny                               ///
////////////////////////////////////////////////////////////////////////////////

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::operator==(
    const TensorShapeAny<SizeTy> &other) const {
  if (isScalar() && other.isScalar())
    return true;
  for (unsigned idx = 0, end = std::max(rank(), other.rank()); idx < end;
       ++idx) {
    if ((*this)[idx] != other[idx])
      return false;
  }
  return true;
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::operator<(
    const TensorShapeAny<SizeTy> &other) const {
  unsigned MaximumRank = std::max(rank(), other.rank());
  std::less<SizeTy> lessThan;
  for (unsigned Index = MaximumRank - 1; Index < MaximumRank; --Index) {
    if (lessThan((*this)[Index], other[Index]))
      return true;
    else if (lessThan(other[Index], (*this)[Index]))
      return false;
  }
  return false;
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::operator>(
    const TensorShapeAny<SizeTy> &other) const {
  return other < *this;
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::operator>=(
    const TensorShapeAny<SizeTy> &other) const {
  return other < *this || *this == other;
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::operator<=(
    const TensorShapeAny<SizeTy> &other) const {
  return *this < other || *this == other;
}

template <typename SizeTy>
Error TensorShapeAny<SizeTy>::checkDims(
    const TensorShapeAny<DimSize> &other,
    std::function<Error(unsigned idx, DimSize, DimSize)> f) const {
  for (unsigned i = 0, e = std::min(rank(), other.rank()); i < e; ++i) {
    if (Error e = f(i, (*this)[i], other[i])) {
      return e;
    }
  }
  return Error::success();
}

template <typename SizeTy>
Error TensorShapeAny<SizeTy>::combineShapeBcast(
    const TensorShapeAny<DimSize> &other) {
  if (Error e = canCombineWith(other))
    return e;
  for (unsigned i = 0, e = std::min(rank(), other.rank()); i < e; ++i) {
    shape[i] = std::max(shape[i], other[i]);
  }
  // Copy what's coming from the other shape
  for (unsigned i = rank(); i < other.rank(); ++i)
    shape.push_back(other[i]);
  return Error::success();
}

template <typename SizeTy>
Error TensorShapeAny<SizeTy>::isBroadcastError(
    const TensorShapeAny<DimSize> &other) const {
  auto check = [&](unsigned Idx, DimSize thisSize, DimSize otherSize) -> Error {
    if ((thisSize > 1 && otherSize > 1 && thisSize != otherSize) ||
        (thisSize > 1 && otherSize <= 1)) {
      // The shapes are not compatible and cannot be broadcasted
      std::string errorStr;
      raw_string_ostream os(errorStr);
      os << "dimension index " << Idx << " cannot be broadcasted from "
         << thisSize << " to " << otherSize;
      os.flush();
      return createStringError(inconvertibleErrorCode(), errorStr);
    }
    return Error::success();
  };
  return checkDims(other, check);
}

template <typename SizeTy>
Error TensorShapeAny<SizeTy>::canCombineWith(
    const TensorShapeAny<DimSize> &other) const {
  auto check = [&](unsigned Idx, DimSize thisSize, DimSize otherSize) -> Error {
    if (thisSize > 1 && otherSize > 1 && thisSize != otherSize) {
      // The shapes are not compatible and cannot be broadcasted
      std::string errorStr;
      raw_string_ostream os(errorStr);
      os << "cannot apply the broadcast rule between dimension size "
         << thisSize << " and " << otherSize << " at dimension index " << Idx;
      os.flush();
      return createStringError(inconvertibleErrorCode(), errorStr);
    }
    return Error::success();
  };
  return checkDims(other, check);
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::reduceDimensions(const BitVector &bv) {
  bool Changed = false;
  for (auto idx : bv.set_bits()) {
    if (idx < shape.size()) {
      Changed = Changed || shape[idx] != DimSize(1);
      shape[idx] = DimSize(1);
    }
  }
  return Changed;
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::keepDimensions(const BitVector &BV) {
  auto BVFlipped = BV;
  BVFlipped.flip();
  return reduceDimensions(BVFlipped);
}

template <typename SizeTy>
bool TensorShapeAny<SizeTy>::reducedToScalarBy(
    const BitVector &reduction) const {
  TensorShapeAny<SizeTy> t = *this;
  t.reduceDimensions(reduction);
  return t.isScalar();
}

template <typename SizeTy>
size_t TensorShapeAny<SizeTy>::getOffsetAt(ArrayRef<size_t> coordinate) const {
  size_t currentStride{1};
  size_t offset{0};
  for (size_t peIdx = 0; peIdx < coordinate.size(); ++peIdx) {
    DimSize peSize = (*this)[peIdx];
    if (peSize > DimSize(1)) {
      assert(coordinate[peIdx] < peSize && "Out of bound access");
      offset += coordinate[peIdx] * currentStride;
    }
    currentStride *= peSize;
  }
  return offset;
}

template <typename SizeTy>
void TensorShapeAny<SizeTy>::foreachIndex(
    std::function<void(const ArrayRef<size_t>)> f) const {
  if (isScalar())
    return;
  std::vector<size_t> indices(rank(), 0);
  for (DimSize flatIdx = 0; flatIdx < flatShape(); ++flatIdx) {
    f(indices);
    indices[0] += 1;
    for (unsigned dim = 0; dim < rank() - 1; ++dim) {
      if (indices[dim] == shape[dim]) {
        indices[dim] = 0;
        indices[dim + 1] += 1;
      }
    }
  }
}

template <typename SizeTy>
BitVector TensorShapeAny<SizeTy>::nonEmptyDims() const {
  unsigned rk = rank();
  BitVector NonEmpty(rk);
  for (unsigned i = 0; i < rk; ++i) {
    if ((*this)[i] > 1)
      NonEmpty.set(i);
  }
  return NonEmpty;
}

template <typename SizeTy>
BitVector TensorShapeAny<SizeTy>::testBothDims(
    const TensorShapeAny<SizeTy> &other,
    const std::function<bool(SizeTy, SizeTy)> &test) const {
  unsigned MaxRank = std::max(rank(), other.rank());
  BitVector testedDims(MaxRank);
  for (unsigned i = 0; i < MaxRank; i++) {
    if (test((*this)[i], other[i]))
      testedDims.set(i);
  }
  return testedDims;
}

template <typename SizeTy>
BitVector TensorShapeAny<SizeTy>::bothNonEmptyDims(
    const TensorShapeAny<SizeTy> &other) const {
  auto bothNonEmpty = [](SizeTy a, SizeTy b) { return a > 1 && b > 1; };
  return testBothDims(other, bothNonEmpty);
}

template <typename SizeTy>
BitVector TensorShapeAny<SizeTy>::reductionDimensionsBeforeBroadcast(
    const TensorShapeAny<SizeTy> &other) const {
  auto toReduce = [](SizeTy me, SizeTy them) { return me > 1 && them < 2; };
  return testBothDims(other, toReduce);
}

template <typename SizeTy>
BitVector TensorShapeAny<SizeTy>::requiredSplat(
    const TensorShapeAny<SizeTy> &other) const {
  auto needsSplat = [](SizeTy source, SizeTy destination) {
    return source == 1 && destination > 1;
  };
  return testBothDims(other, needsSplat);
}

template <typename SizeTy>
void TensorShapeAny<SizeTy>::print(raw_ostream &O) const {
  O << "Tensor";
  for (auto &len : make_range(shape.begin(), shape.end())) {
    O << "[" << len << "]";
  }
}

template <typename SizeTy>
MDNode *TensorShapeAny<SizeTy>::toConstMetadata(IntegerType *Ty) const {
  SmallVector<Metadata *, BaseTensorSize> DimsArray;
  for (auto Dim : *this) {
    auto *DimAsMetadata = ConstantAsMetadata::get(
        ConstantInt::get(Ty, APInt(Ty->getBitWidth(), Dim, false)));
    DimsArray.push_back(DimAsMetadata);
  }
  return MDNode::get(Ty->getContext(), DimsArray);
}

template <typename SizeTy>
std::unique_ptr<TensorShapeAny<SizeTy>>
TensorShapeAny<SizeTy>::fromConstMetadata(unsigned Rank, const MDNode *Node) {
  Shape S;
  // Don't fail too early, when Idx >= rank and all the upper dimensions are 1
  // (empty), we can construct the Tensor for this rank too
  for (unsigned Idx = 0, E = Node->getNumOperands(); Idx < E; Idx++) {
    if (auto *ConstantMeta =
            dyn_cast<ConstantAsMetadata>(Node->getOperand(Idx).get())) {
      if (auto *CI = dyn_cast<ConstantInt>(ConstantMeta->getValue())) {
        uint64_t Val = CI->getZExtValue();
        if (Val > 1 && Idx >= Rank)
          return nullptr;
        S.push_back(Val);
      }
    }
  }
  S.resize(Rank, DimSize(1));
  return std::make_unique<TensorShapeAny<SizeTy>>(std::move(S));
}

template <typename SizeTy>
Expected<TensorShapeAny<SizeTy>> TensorShapeAny<SizeTy>::broadcastShapeFromAll(
    ArrayRef<const TensorShapeAny<SizeTy> *> AllToBcast) {
  auto ptrToRef = [](const TensorShapeAny<SizeTy> *ShapePtr)
      -> const TensorShapeAny<SizeTy> & { return *ShapePtr; };
  return broadcastShapeFromAll(
      make_range(map_iterator(AllToBcast.begin(), ptrToRef),
                 map_iterator(AllToBcast.end(), ptrToRef)));
}

template <typename SizeTy>
template <typename IteratorT>
Expected<TensorShapeAny<SizeTy>> TensorShapeAny<SizeTy>::broadcastShapeFromAll(
    llvm::iterator_range<IteratorT> AllToBcast) {
  if (AllToBcast.begin() == AllToBcast.end())
    return createStringError(inconvertibleErrorCode(),
                             "Cannot broadcast without shapes");
  TensorShapeAny<SizeTy> BcastShape = *AllToBcast.begin();
  Error Err = Error::success();
  // Mark Err as checked
  for_each(make_range(std::next(AllToBcast.begin()), AllToBcast.end()),
           [&BcastShape, &Err](auto &TShape) {
             if (!Err)
               if (Error E = BcastShape.combineShapeBcast(TShape)) {
                 std::swap(E, Err);
                 consumeError(std::move(E));
               }
           });
  if (!Err)
    return BcastShape;
  else
    return Err;
}
