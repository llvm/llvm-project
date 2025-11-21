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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRipple.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CFGUpdate.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <array>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <queue>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "ripple"

namespace {

////////////////////////////////////////////////////////////////////////////////
///                             Local Classes                                ///
////////////////////////////////////////////////////////////////////////////////

class DiagnosticInfoRipple : public DiagnosticInfoGeneric {
private:
  // Used to store the message when the input Twine is not a single StringRef
  SmallVector<char, 0> ErrStr;
  // The error message
  StringRef ErrStrRef;
  // Keep a Twine to ErrStrRef for DiagnosticInfoGeneric
  Twine ErrTwine;

public:
  DiagnosticInfoRipple(DiagnosticSeverity Severity, const Twine &Msg)
      : DiagnosticInfoGeneric(nullptr, ErrTwine, Severity),
        ErrStrRef(Msg.toStringRef(ErrStr)), ErrTwine(ErrStrRef) {}
};

class DiagnosticInfoRippleWithLoc : public DiagnosticInfoGenericWithLoc {
private:
  // Used to store the message when the input Twine is not a single StringRef
  SmallVector<char, 0> ErrStr;
  // The error message
  StringRef ErrStrRef;
  // Keep a Twine to ErrStrRef for DiagnosticInfoGenericWithLoc
  Twine ErrTwine;

public:
  DiagnosticInfoRippleWithLoc(enum DiagnosticSeverity Severity,
                              const Function &Fn, const DiagnosticLocation &Loc,
                              const Twine &Msg)
      : DiagnosticInfoGenericWithLoc(ErrTwine, Fn, Loc, Severity),
        ErrStrRef(Msg.toStringRef(ErrStr)), ErrTwine(ErrStrRef) {}
};

////////////////////////////////////////////////////////////////////////////////
///                        Local Helper Functions                            ///
////////////////////////////////////////////////////////////////////////////////

#define RETURN_UNEXPECTED(expr)                                                \
  if (!expr) {                                                                 \
    return (expr).takeError();                                                 \
  }

/// @brief: Sets the insert points in `IRB` after the instruction "I".
void setInsertPointAfter(llvm::IRBuilder<> &IRB, llvm::Instruction *I) {
  if (isa<PHINode>(I) || isa<LandingPadInst>(I))
    IRB.SetInsertPoint(I->getParent()->getFirstInsertionPt());
  else
    IRB.SetInsertPoint(std::next(I->getIterator()));
}

std::optional<uint64_t> getConstantOperandValue(const Instruction *I,
                                                unsigned index) {
  if (index >= I->getNumOperands())
    return std::nullopt;
  auto *op = I->getOperand(index);
  auto *cst = dyn_cast<ConstantInt>(op);
  if (!cst)
    return std::nullopt;
  if (cst->getValue().getActiveBits() > 64)
    return std::nullopt;
  return cst->getValue().getZExtValue();
}

/// @brief Helper function that returns a pointer if the instruction *I* is an
/// IntrinsicInst w/ any of the IDs provided in *ids*, nullptr otherwise.
const IntrinsicInst *intrinsicWithId(const Instruction *I,
                                     ArrayRef<Intrinsic::ID> ids) {
  if (auto II = dyn_cast_if_present<IntrinsicInst>(I)) {
    if (is_contained(ids, II->getIntrinsicID()))
      return II;
  }
  return nullptr;
}

IntrinsicInst *intrinsicWithId(Instruction *I,
                               std::initializer_list<Intrinsic::ID> ids) {
  return const_cast<IntrinsicInst *>(
      intrinsicWithId(const_cast<const Instruction *>(I), ids));
}

Constant *getRippleNeutralReductionElement(Intrinsic::ID ID, Type *EltTy,
                                           FastMathFlags FMF) {
  assert(VPReductionIntrinsic::isVPReduction(ID) &&
         "Expecting a vp reduction intrinsic ID");
  if (auto OptIntinsic = VPIntrinsic::getFunctionalIntrinsicIDForVP(ID))
    return dyn_cast<Constant>(getReductionIdentity(*OptIntinsic, EltTy, FMF));
  else
    return nullptr;
}

/// If the debug location is the result of inlining some code, returns the
/// location where it has been inlined from, else returns the debug location
/// unchanged.
///
/// This is useful for Ripple's error reporting because we'd rather want point
/// to a location inside the function we are vectorizing instead of the header
/// where implementation details of the API would be implemented (e.g.,
/// overloading of ripple_shuffle in C++).
DebugLoc stripInliningFromDebugLoc(DebugLoc DL) {
  while (DL && DL->getInlinedAt())
    DL = DL->getInlinedAt();
  return DL;
}

/// @brief Creates a SelectInst (not attached to any BasicBlock) taking a vector
/// of @p Count i1 booleans and outputing a vector of integer type @p OutType
/// with values 1 when true and 0 when false.
SelectInst *createMaskSelectToTrueFalse(IntegerType *OutType,
                                        ElementCount Count,
                                        const Twine &Name = "") {
  auto &Context = OutType->getContext();
  return SelectInst::Create(
      ConstantVector::getSplat(Count, ConstantInt::getTrue(Context)),
      ConstantVector::getSplat(Count, ConstantInt::get(OutType, 1)),
      ConstantVector::getSplat(Count, ConstantInt::get(OutType, 0)), Name);
}

/** @brief Returns the set, S, of basic blocks such that for every X in S, there
 * is a path, P, BB .. -(P)-> .. S, such that ExecludedBB is not a part of a P.
 */
DenseSet<const BasicBlock *>
getAllReachableBBsExcluding(const BasicBlock *BB,
                            const BasicBlock *ExcludedBB) {
  std::queue<const BasicBlock *> Worklist;
  Worklist.push(BB);

  DenseSet<const BasicBlock *> ReachableBBs;

  while (!Worklist.empty()) {
    auto *FrontBB = Worklist.front();
    Worklist.pop();
    for (auto *SuccBB : successors(FrontBB)) {
      if (SuccBB == ExcludedBB)
        continue;
      if (ReachableBBs.contains(SuccBB))
        continue;
      ReachableBBs.insert(SuccBB);
      Worklist.push(SuccBB);
    }
  }
  return ReachableBBs;
}

} // namespace

bool llvm::hasTrivialLoopLikeBackEdge(BasicBlock *BranchingBB, BasicBlock *PDom,
                                      DominatorTreeAnalysis::Result &DT) {
  if (succ_size(BranchingBB) != 2) {
    return false;
  }
  const BasicBlock *BB1 = *succ_begin(BranchingBB);
  const BasicBlock *BB2 = *(succ_begin(BranchingBB) + 1);

  const SmallPtrSet<BasicBlock *, 1> PDomSet({PDom});
  const SmallPtrSet<BasicBlock *, 1> BranchBBSet({BranchingBB});

  if (isPotentiallyReachable(BB1, BranchingBB, &PDomSet, &DT) &&
      !isPotentiallyReachable(BB2, BranchingBB, &PDomSet, &DT) &&
      !isPotentiallyReachable(BB1, BB2, &BranchBBSet, &DT)) {
    return getAllReachableBBsExcluding(BB1, PDom).contains(BranchingBB);
  }
  if (isPotentiallyReachable(BB2, BranchingBB, &PDomSet) &&
      !isPotentiallyReachable(BB1, BranchingBB, &PDomSet) &&
      !isPotentiallyReachable(BB2, BB1, &BranchBBSet, &DT)) {
    return getAllReachableBBsExcluding(BB2, PDom).contains(BranchingBB);
  }
  return false;
}

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

////////////////////////////////////////////////////////////////////////////////
///                           NDLoadStoreAttr                                ///
////////////////////////////////////////////////////////////////////////////////

void NDLoadStoreAttr::print(raw_ostream &O) const {
  // NIY
}

NDLoadStoreAttr NDLoadStoreAttr::fromString(StringRef AttrName) {
  // NIY
  return NDLoadStoreAttr();
}

////////////////////////////////////////////////////////////////////////////////
///                         NDLoadStoreFactory                               ///
////////////////////////////////////////////////////////////////////////////////

bool NDLoadStoreFactory::analyzeCoalescing(Type *ElementTy,
                                           LinearSeries &AddressSeries,
                                           BitVector &StrideDims) {

  assert(PointerType::isValidElementType(ElementTy) &&
         "Internal type inference error");
  DataLayout const &DL = Mod.getDataLayout();

  uint64_t ElemByteSize = DL.getTypeAllocSize(ElementTy);

  // Gather address tensor slice sizes, so we can compare them with slope sizes
  // 0-dimensional slice is one element, then one rod, plane, etc.
  const TensorShape &SlopeShape = AddressSeries.getSlopeShape();
  SmallVector<uint64_t, 3> SliceSize(SlopeShape.rank());
  uint64_t CurSliceSize = ElemByteSize;
  for (int i = 0, n = SlopeShape.rank(); i < n; ++i) {
    SliceSize[i] = CurSliceSize;
    CurSliceSize *= AddressSeries.getShape(i);
  }
  LLVM_DEBUG(dbgs() << "slice sizes for " << AddressSeries << ": ";
             for_each(SliceSize, [](auto size) { dbgs() << size << ' '; });
             dbgs() << '\n');

  const TensorShape &BaseShape = AddressSeries.getBaseShape();
  bool Contiguous = true;
  // We're contiguous up to dimension i of the Address series
  // if the address tensor is made of slopes up to i and these slopes
  // correspond to a slice of the address tensor.
  int k = 0;
  for (int i = 0, n = SlopeShape.rank(); i < n; ++i) {
    // There aren't any strides as long as the slopes match the slice of
    // the load's tensor shape.
    // If the stride comes from a collection of bases,
    // we assume that this match isn't happening.
    if (BaseShape[i] > 1) {
      Contiguous = false;
      continue;
    }
    if (SlopeShape[i] == 1)
      // stride is unchanged.
      // keep trying to match the same slice shape with the next stride
      continue;
    if (Constant *ConstSlope = dyn_cast<Constant>(AddressSeries.getSlope(i))) {
      // Skip trivial and broadcast values
      assert(!ConstSlope->isZeroValue() &&
             "Unexpected splat/broadcast in linear series");
      const APInt &ApSlope = ConstSlope->getUniqueInteger();
      const APInt ApSliceSize(ApSlope.getBitWidth(), SliceSize[k],
                              /*signed=*/false);
      LLVM_DEBUG(dbgs() << "slope " << ApSlope << " compares to slice size "
                        << ApSliceSize << '\n');
      // Slopes become load/store strides from the moment that contiguity
      // is broken
      if (!Contiguous || ApSlope != ApSliceSize) {
        Contiguous = false;
        StrideDims.set(i);
      }
    } else { // non-constant slope
      Contiguous = false;
      StrideDims.set(i);
    }
    k++;
  }
  return Contiguous;
}

std::string NDLoadStoreFactory::ndFunctionName(StringRef LoadOrStore, int nDims,
                                               NDLoadStoreAttr &Attr) {
  std::string FunName;
  raw_string_ostream OS(FunName);
  OS << LoadOrStore << "." << nDims << "d" << LoadOrStore;
  if (!Attr.empty()) {
    OS << "." << Attr;
  }
  OS.flush();
  return FunName;
}

Value *NDLoadStoreFactory::genUnstructuredLoad(LoadInst *Load, Value *Address,
                                               const TensorShape &ToShape) {
  IrBuilder.SetInsertPoint(Load);
  if (ToShape.isScalar()) {
    // Base case, load
    Value *LoadVal =
        IrBuilder.CreateLoad(Load->getType(), Address, Load->getName());
    MyRipple.setRippleShape(LoadVal, ToShape);
    return LoadVal;
  } else if (AllocaInst *Alloca = dyn_cast<AllocaInst>(Address)) {
    assert(MyRipple.getRippleShape(static_cast<Value *>(Alloca)).flatShape() >=
           ToShape.flatShape());
    // Allocas that have been promoted don't require a gather
    Type *LoadVectorTy =
        VectorType::get(Alloca->getAllocatedType()->getArrayElementType(),
                        ToShape.flatShape(), /*Scalable*/ false);
    Value *LoadVal = IrBuilder.CreateLoad(LoadVectorTy, Alloca,
                                          Twine(Load->getName()) + ".ripple");
    MyRipple.setRippleShape(LoadVal, ToShape);
    return LoadVal;
  }

  LLVM_DEBUG(dbgs() << "Generating gather"
                    << " from addresses " << *Address << "\n");
  Type *ElementTy = Load->getType();
  int NElements = ToShape.flatShape();
  assert(NElements > 1);
  Type *VecTy = VectorType::get(ElementTy, NElements, /*scalable=*/false);
  Value *GatheredVal =
      IrBuilder.CreateMaskedGather(VecTy, Address,
                                   // Aligned on element boundaries
                                   Load->getAlign());
  MyRipple.setRippleShape(GatheredVal, ToShape);
  return GatheredVal;
}

Value *NDLoadStoreFactory::genLoadNoSplat(LoadInst *Load,
                                          LinearSeries &AddressSeries,
                                          Value *DefaultAddress,
                                          const TensorShape &ToShape) {
  // TODO: check that AddressSeries is splat-free
  // TODO: With opaque types, I'm not sure this will always work. Check.
  Type *ElementTy = Load->getType();
  BitVector StrideDims(AddressSeries.rank());
  bool Contiguous = analyzeCoalescing(ElementTy, AddressSeries, StrideDims);
  int NElements = ToShape.flatShape();
  Type *LoadTy = ToShape.isScalar() ? ElementTy
                                    : VectorType::get(ElementTy, NElements,
                                                      /*scalable=*/false);
  IrBuilder.SetInsertPoint(Load);
  assert(ToShape.isVector() || ToShape.isScalar() && Contiguous);
  if (Contiguous) {
    LoadInst *VectorLoad =
        IrBuilder.CreateLoad(LoadTy, AddressSeries.getBase());
    MyRipple.setRippleShape(static_cast<Value *>(VectorLoad), ToShape);
    VectorLoad->takeName(Load);
    // TODO: Take ripple_aligned() into account.
    VectorLoad->setAlignment(Load->getAlign());
    return VectorLoad;
  } else {
    return genUnstructuredLoad(Load, DefaultAddress, ToShape);
  }
}

Value *NDLoadStoreFactory::genLoad(LoadInst *Load, LinearSeries &AddressSeries,
                                   Value *DefaultAddress,
                                   const TensorShape &ToShape) {
  // Deal with broadcasts: load(a + 0*k along d) = broadcast(load(a), d)
  // So we first build the broadcast-free load, and broadcast it.
  // The reason why we expose broadcasts at this moment is that
  // it can simplify the application of masks during if-conversion.
  // A smaller-dimensional mask can apply before broadcast,
  // as opposed to having to broadcast the mask.
  BitVector SeriesSplat = AddressSeries.getSplatDims();
  // There may also be an implicit broadcast
  const TensorShape &AddressShape = AddressSeries.getShape();
  BitVector SplatDims = AddressShape.requiredSplat(ToShape);
  SplatDims |= SeriesSplat;
  Value *VectorLoad;
  if (!SplatDims.empty()) {
    // LLVM_DEBUG(dbgs() << "Splat dims: " << SplatDims << '\n');
    // Create a load without the splatting/broadcasting
    LinearSeries SplatFreeSeries = AddressSeries.removeSlopes(SeriesSplat);
    // Temporary local LS used for load optimization needs to be instantiated
    // w/o the cache
    IrBuilder.SetInsertPoint(Load);
    auto SplatFreeAddress =
        MyRipple.instantiateLinearSeriesNoCache(SplatFreeSeries);
    const TensorShape &LoadShape = SplatFreeSeries.getShape();
    LLVM_DEBUG(dbgs() << "Loaded series shape " << LoadShape << '\n');
    Value *CoreGather =
        genLoadNoSplat(Load, SplatFreeSeries, SplatFreeAddress, LoadShape);
    auto Splatted = MyRipple.tensorBcast(CoreGather, LoadShape, ToShape);
    // Checked during shape propagation
    if (!Splatted)
      report_fatal_error("Broadcast failure during codegen");
    VectorLoad = *Splatted;
  } else {
    VectorLoad = genLoadNoSplat(Load, AddressSeries, DefaultAddress, ToShape);
  }
  return VectorLoad;
}

Value *NDLoadStoreFactory::genUnstructuredStore(StoreInst *Store, Value *Val,
                                                Value *Address,
                                                const TensorShape &ToShape) {
  int NElements = ToShape.flatShape();
  // No implicit broadcasts are expected here
  assertNumElements(Val, NElements);
  assertNumElements(Address, NElements);
  LLVM_DEBUG(dbgs() << "Generating scatter w/ " << *Val << " to addresses "
                    << *Address << "\n");

  IrBuilder.SetInsertPoint(Store);
  Value *StoredVal =
      IrBuilder.CreateMaskedScatter(Val, Address, Store->getAlign());
  MyRipple.setRippleShape(StoredVal, ToShape);
  return StoredVal;
}

Value *NDLoadStoreFactory::genStore(StoreInst *Store, Value *Val,
                                    LinearSeries &AddressSeries,
                                    Value *DefaultAddress,
                                    const TensorShape &ToShape) {
  Type *ElementTy = Store->getValueOperand()->getType();
  BitVector StrideDims(AddressSeries.rank());
  bool Contiguous = analyzeCoalescing(ElementTy, AddressSeries, StrideDims);
  IrBuilder.SetInsertPoint(Store);
  Value *VectorStore;
  if (Contiguous) {
    StoreInst *VecStore = IrBuilder.CreateStore(Val, AddressSeries.getBase());
    MyRipple.setRippleShape(static_cast<Value *>(VecStore), ToShape);
    VecStore->setAlignment(Store->getAlign());
    VectorStore = VecStore;
  } else {
    return genUnstructuredStore(Store, Val, DefaultAddress, ToShape);
  }
  VectorStore->takeName(Store);
  MyRipple.setRippleShape(VectorStore, ToShape);
  return VectorStore;
}

////////////////////////////////////////////////////////////////////////////////
///                               Ripple                                     ///
////////////////////////////////////////////////////////////////////////////////

// This is a copy paste of updated UnifyFunctionExitNodes pass methods.
// It needs to be properly upstreamed, but for now we keep a copy
// to minimize upstream conflict.

static bool
unifyUnreachableBlocks(Function &F,
                       SmallVectorImpl<DominatorTree::UpdateType> *DTU) {
  std::vector<BasicBlock *> UnreachableBlocks;

  for (BasicBlock &I : F)
    if (isa<UnreachableInst>(I.getTerminator()))
      UnreachableBlocks.push_back(&I);

  if (UnreachableBlocks.size() <= 1)
    return false;

  BasicBlock *UnreachableBlock =
      BasicBlock::Create(F.getContext(), "UnifiedUnreachableBlock", &F);
  new UnreachableInst(F.getContext(), UnreachableBlock);

  for (BasicBlock *BB : UnreachableBlocks) {
    BB->back().eraseFromParent(); // Remove the unreachable inst.
    BranchInst::Create(UnreachableBlock, BB);
    if (DTU)
      DTU->push_back({DominatorTree::Insert, BB, UnreachableBlock});
  }

  return true;
}

static bool unifyReturnBlocks(Function &F,
                              SmallVectorImpl<DominatorTree::UpdateType> *DTU) {
  std::vector<BasicBlock *> ReturningBlocks;

  for (BasicBlock &I : F)
    if (isa<ReturnInst>(I.getTerminator()))
      ReturningBlocks.push_back(&I);

  if (ReturningBlocks.size() <= 1)
    return false;

  // Insert a new basic block into the function, add PHI nodes (if the function
  // returns values), and convert all of the return instructions into
  // unconditional branches.
  BasicBlock *NewRetBlock =
      BasicBlock::Create(F.getContext(), "UnifiedReturnBlock", &F);

  PHINode *PN = nullptr;
  if (F.getReturnType()->isVoidTy()) {
    ReturnInst::Create(F.getContext(), nullptr, NewRetBlock);
  } else {
    // If the function doesn't return void... add a PHI node to the block...
    PN = PHINode::Create(F.getReturnType(), ReturningBlocks.size(),
                         "UnifiedRetVal");
    PN->insertInto(NewRetBlock, NewRetBlock->end());
    ReturnInst::Create(F.getContext(), PN, NewRetBlock);
  }

  // Loop over all of the blocks, replacing the return instruction with an
  // unconditional branch.
  for (BasicBlock *BB : ReturningBlocks) {
    // Add an incoming element to the PHI node for every return instruction that
    // is merging into this new block...
    if (PN)
      PN->addIncoming(BB->getTerminator()->getOperand(0), BB);

    BB->back().eraseFromParent(); // Remove the return insn
    BranchInst::Create(NewRetBlock, BB);
    if (DTU)
      DTU->push_back({DominatorTree::Insert, BB, NewRetBlock});
  }
  return true;
}

void Ripple::setReplacementFor(Instruction *I, Value *V,
                               const TensorShape &Shape) {
  if (InstructionReplacementMapping.contains(I)) {
    LLVM_DEBUG(dbgs() << "Re-defining the mapping between " << *I << " and "
                      << *InstructionReplacementMapping[I] << " to " << *V
                      << "\n");
  }
  InstructionReplacementMapping[I] = V;
  setRippleShape(V, Shape);
}

std::pair<Value *, const TensorShape *>
Ripple::replacementValueAndShape(Value *V) const {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    ConstructedSeries CS = getCachedSeries(I);
    if (Value *Replacement = InstructionReplacementMapping.lookup(I)) {
      LLVM_DEBUG(dbgs() << "Replacement tensor value for " << *V
                        << " found: " << *Replacement << "\n");
      // During vectorization (genVectorInstruction), linear series with a
      // tensor base get a "replacement" base.
      // This method always returns the "replacement" of the base, irrespective
      // of the context. It's crucial to use the method getTensorUse to obtain
      // the expected tensor for all kinds of users (LS or not). This is because
      // an LS instantiation doesn't "replace" a value, it "creates" a tensor
      // value using an existing base and slopes, which getTensorUse handles!
      if (CS)
        return {Replacement, &CS.LS->getBaseShape()};
      else
        return {Replacement, &getRippleShape(V)};
    } else if (CS) {
      assert(CS.LS->getBaseShape() == ScalarShape &&
             "Tensor LinearSeries base has not been generated");
      return {V, &ScalarShape};
    }
  }
  LLVM_DEBUG(dbgs() << "No replacement tensor value for " << *V << "\n");
  return {V, &getRippleShape(V)};
}

std::pair<Value *, const TensorShape *> Ripple::getTensorUse(const Use &U) {
  // Ripple generates tensors for Instructions only
  Instruction *User = cast<Instruction>(U.getUser());
  Value *UseVal = U.get();
  // If the Use is a LS, we may have to instantiate it
  if (Instruction *UseI = dyn_cast<Instruction>(UseVal)) {
    if (auto UseCS = getCachedSeries(UseI)) {
      auto UserCS = getCachedSeries(User);
      // When both UserCS and UseCS are valid linear series we have two cases:
      // 1) if UserCS and UseCS have the same base shape, we return the
      //    (tensorized) base. This is because the only case, other than
      //    reduction and slicing, where we are asking for such getTensorUse is
      //    when we are generating the "Replacement" of UserCS's base. Hence we
      //    skip instantiation and fall back to getting the replacement below.
      // 2) In other cases we want to generate the instance of UseCS for the
      //    user and thus enter this condition's block
      if (!UserCS || UseCS.LS->getBaseShape() != UserCS.LS->getBaseShape() ||
          rippleReduceIntrinsics(User) || rippleSliceIntrinsic(User)) {
        auto InstantiatedLS = instantiateCachedSeries(UseCS, UseI);
        return std::make_pair(InstantiatedLS, &UseCS.LS->getShape());
      }
      // else returns the replacement below
    }
  }
  auto RepAndShape = replacementValueAndShape(UseVal);
  auto &[RepValue, RepShape] = RepAndShape;
  // Handle Alloca vector expansion shape-shifting
  if (isa<AllocaInst>(RepValue) && RepShape->isVector()) {
    auto &UserShape = getRippleShape(User);
    // That's always true assuming we don't allow taking the address of scalar
    // Alloca (which is part of Ripple pre-requisite for now). Otherwise, to
    // allow disambiguation, the fix would be to set TensorShapes to the
    // Use of AllocaInst instead.
    assert(UserShape <= *RepShape);
    RepShape = &UserShape;
  }
  return RepAndShape;
}

const TensorShape &Ripple::getRippleShape(const Value *V,
                                          bool ShapePropagation) const {
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    return getRippleShape(I);
  } else if (const Constant *C = dyn_cast<Constant>(V)) {
    return getRippleShape(C, ShapePropagation);
  } else if (const Argument *A = dyn_cast<Argument>(V)) {
    return getRippleShape(A, ShapePropagation);
  } else if (isa<BasicBlock>(V) || isa<MetadataAsValue>(V)) {
    return ShapeIgnoredByRipple;
  } else {
    std::string ErrorStr;
    raw_string_ostream OS(ErrorStr);
    OS << "The following value does not have a Ripple shape: " << *V;
    OS.flush();
    report_fatal_error(ErrorStr.c_str());
  }
}

const TensorShape &Ripple::getRippleShape(const Constant *C,
                                          bool ShapePropagation) const {
  if (ShapePropagation && C->getType()->isVectorTy())
    return ShapeIgnoredByRipple;
  else if (!C->getType()->isVectorTy())
    return ScalarShape;
  report_fatal_error(
      "Taking the Ripple shape of a vector constant is ill-defined");
}

const TensorShape &Ripple::getRippleShape(const Argument *A,
                                          bool ShapePropagation) const {
  // When specializing, arguments can have tensor shapes!
  if (ShapePropagation && A->getType()->isVectorTy())
    return ShapeIgnoredByRipple;
  else if (!A->getType()->isVectorTy())
    return ScalarShape;
  report_fatal_error(
      "Taking the Ripple shape of a vector function Argument is ill-defined");
}

const TensorShape &Ripple::getRippleShape(const Instruction *I) const {
  auto Shape = InstructionRippleShapes.find(I);
  if (Shape != InstructionRippleShapes.end()) {
    return Shape->second;
  } else {
    std::string ErrorStr;
    raw_string_ostream OS(ErrorStr);
    OS << "The following instruction does not have a Ripple shape: " << *I;
    OS.flush();
    report_fatal_error(ErrorStr.c_str());
  }
}

Error Ripple::replaceRippleGetSize() {
  Error AllErrors = Error::success();
  for (auto &I : make_early_inc_range(instructions(F))) {
    if (IntrinsicInst *BlockGetSize =
            intrinsicWithId(&I, {Intrinsic::ripple_block_getsize})) {
      if (Error E = checkRippleBlockIntrinsics(BlockGetSize)) {
        AllErrors = joinErrors(std::move(AllErrors), std::move(E));
        continue;
      }
      auto DimSize = getRippleGetSizeValue(BlockGetSize);
      Constant *C = ConstantInt::get(BlockGetSize->getType(), DimSize);
      auto BlockIterator = BlockGetSize->getIterator();
      invalidateRippleDataFor(BlockGetSize);
      ReplaceInstWithValue(BlockIterator, C);
    }
  }
  return AllErrors;
}

Error Ripple::propagateShapes(bool &WaitingForSpecialization) {
  LLVM_DEBUG(dbgs() << "Propagating shapes in function:\n"; F.print(dbgs()));

  if (Error E = checkBlockShapeUsage(F))
    return E;

  std::queue<Instruction *> WorkQueue;
  WaitingForSpecialization = false;

  std::function<void(Instruction *)> revisitUserInstructions =
      [&](Instruction *I) {
        for (auto *User : I->users())
          if (Instruction *UserInst = dyn_cast<Instruction>(User))
            if (domTree.isReachableFromEntry(UserInst->getParent()))
              WorkQueue.push(UserInst);

        if (auto MemLoc = MemoryLocation::getOrNone(I))
          if (AllocaInst *Alloca = aliasesWithPromotableAlloca(*MemLoc)) {
            if (auto *Store = dyn_cast<StoreInst>(I)) {
              visitAllInstructionsBeingClobberedBy(
                  cast<MemoryDef>(MemSSA.getMemoryAccess(Store)),
                  [&](Instruction *Clobbered) -> bool {
                    WorkQueue.push(Clobbered);
                    return true;
                  });
            } else if (auto *Load = dyn_cast<LoadInst>(I)) {
              visitAllClobberingInstructions(
                  cast<MemoryUse>(MemSSA.getMemoryAccess(Load)),
                  [&](Instruction *Clobber) -> bool {
                    WorkQueue.push(Clobber);
                    return true;
                  });
            } else {
              llvm_unreachable("Ripple doesn't support promotion of VAARG or "
                               "Atomic instructions");
            }

            // Alloca gets the largest shape of all its users
            auto &AllocaShape = getRippleShape(Alloca);
            auto &InstructionShape = getRippleShape(I);
            if (AllocaShape.flatShape() < InstructionShape.flatShape())
              setRippleShape(Alloca, InstructionShape);
          }
      };

  // Start processing the function in reverse post order
  for (auto *BB : getFuncRPOT())
    for (auto &I : *BB) {
      WorkQueue.push(&I);
      if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&I)) {
        Type *AllocatedType = Alloca->getAllocatedType();
        if (AllocatedType->isFloatingPointTy() || AllocatedType->isIntOrPtrTy())
          PromotableAlloca.insert(Alloca);
        else
          NonPromotableAlloca.insert(Alloca);
      }
    }

  while (!WorkQueue.empty()) {
    Instruction *I = WorkQueue.front();
    WorkQueue.pop();

    LLVM_DEBUG(dbgs() << "Shape inference of: " << *I << "\n");
    auto NewShape = inferShapeFromOperands(I, /*AllowPartialPhi*/ true,
                                           WaitingForSpecialization);
    if (!NewShape) {
      DiagnosticInfoRippleWithLoc Diag(
          DS_Error, F, sanitizeRippleLocation(I),
          "Ripple cannot infer the shape of this instruction");
      F.getContext().diagnose(Diag);
      return NewShape.takeError();
    }
    if (WaitingForSpecialization)
      return Error::success();

    LLVM_DEBUG(dbgs() << "Setting shape: " << *NewShape << "\n");
    // Process the users if we updates the shape of the instruction
    if (setRippleShape(I, *NewShape)) {
      // Revisit users of an instruction that changed shape
      revisitUserInstructions(I);

      if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
        // Mark masked calls
        if (NewShape->isVector()) {
          BasicBlock *VectorBlock = I->getParent();
          BasicBlock *PostDomBlock =
              postdomTree.getNode(VectorBlock)->getIDom()->getBlock();
          for (BasicBlock *BB : allBasicBlocksFromTo(VectorBlock, PostDomBlock))
            for (Instruction &I : *BB)
              if (CallInst *Call = dyn_cast<CallInst>(&I))
                MaskedCalls.insert(Call);
        }
      }
    }
    assert(getRippleShape(I) == *NewShape);
  }

  // All unreachable instructions get a scalar shape!
  for (auto &BB : F)
    if (!domTree.isReachableFromEntry(&BB))
      for (auto &I : BB)
        setRippleShape(&I, ScalarShape);

  assert(allInstructionsHaveRippleShapes() &&
         "Some instruction has no Ripple shape");

  // Now that all shapes have settled, create linear series
  for (auto *BB : getFuncRPOT()) {
    for (auto &I : *BB) {
      WorkQueue.push(&I);
    }
  }
  while (!WorkQueue.empty()) {
    Instruction *I = WorkQueue.front();
    WorkQueue.pop();
    LLVM_DEBUG(dbgs() << "Linear series processing of: " << *I << "\n");
    auto CSBefore = getCachedSeries(I);
    auto CSAfter = getLinearSeriesFor(I);
    LLVM_DEBUG(dbgs() << "Linear series is: " << CSAfter << "\n");

    if (CSBefore.getState() != CSAfter.getState())
      revisitUserInstructions(I);
  }

  // Remove non-valid series
  clearPotentialSeries();

  // Simplify and cleanup slopes
  simplifySlopes();

  // Replace ripple_get_size by their values before simplification
  if (Error E = replaceRippleGetSize())
    return E;

  // Iterate while simplification occurs
  while (simplifyFunction())
    ;

  assert(allInstructionsHaveRippleShapes() &&
         "Some instruction has no Ripple shape");
  return Error::success();
}

Ripple::PEIndex
Ripple::rippleToTensor(const RippleIntrinsicIndex &idAndIdx) const {
  auto firstFound =
      std::find_if(TensorDimIDMap.begin(), TensorDimIDMap.end(),
                   [&](auto &idx) { return idx == idAndIdx.first; });
  assert(firstFound != TensorDimIDMap.end());
  assert(PERank(idAndIdx.first) > idAndIdx.second);
  return std::distance(TensorDimIDMap.begin(), firstFound) + idAndIdx.second;
}

Ripple::RippleIntrinsicIndex Ripple::tensorToRipple(PEIndex tensorIdx) const {
  PEIdentifier id = TensorDimIDMap[tensorIdx];
  PEIndex firstTensorIndex = rippleToTensor(std::make_pair(id, 0));
  return std::make_pair(id, tensorIdx - firstTensorIndex);
}

Ripple::DimSize Ripple::PERank(PEIdentifier PEId) const {
  return PERanks.lookup_or(PEId, 0);
}

DenseMap<Ripple::PEIdentifier, Ripple::DimSize>
Ripple::gatherRippleFunctionUsage(const Function &F) {
  DenseMap<PEIdentifier, DimSize> PERanks;
  for (const auto &I : instructions(F)) {
    if (const IntrinsicInst *BSI =
            intrinsicWithId(&I, {Intrinsic::ripple_block_setshape})) {
      PEIdentifier ProcElem = *getConstantOperandValue(BSI, 0);
      DimSize Rank = 0;
      for (unsigned ArgIdx = 1, E = BSI->arg_size(); ArgIdx < E; ++ArgIdx) {
        DimSize DS = *getConstantOperandValue(BSI, ArgIdx);
        if (DS > 1)
          Rank = ArgIdx;
      }
      auto Res = PERanks.insert({ProcElem, Rank});
      // Key already present; take the maximum rank seen in this function
      if (!Res.second)
        Res.first->second = std::max(Res.first->second, Rank);
    }
  }
  return PERanks;
}

SmallVector<Ripple::PEIdentifier, 8> Ripple::buildPEIdMap() {
  SmallVector<Ripple::PEIdentifier, 8> IdMap;
  for (const auto &[DimIdentifier, DimRank] : PERanks) {
    LLVM_DEBUG(dbgs() << "PE id " << DimIdentifier << " has rank " << DimRank
                      << "\n");
    // Check that the ID is known by the compiler
    for (PEIndex rank = 0; rank < DimRank; ++rank) {
      IdMap.push_back(DimIdentifier);
    }
  }
  return IdMap;
}

IntrinsicInst *Ripple::rippleBlockIntrinsics(Instruction *I) {
  return intrinsicWithId(I, {Intrinsic::ripple_block_index,
                             Intrinsic::ripple_block_getsize,
                             Intrinsic::ripple_block_setshape});
}

IntrinsicInst *Ripple::rippleReduceIntrinsics(Instruction *I) {
  return intrinsicWithId(
      I, {Intrinsic::ripple_reduce_add, Intrinsic::ripple_reduce_mul,
          Intrinsic::ripple_reduce_and, Intrinsic::ripple_reduce_or,
          Intrinsic::ripple_reduce_xor, Intrinsic::ripple_reduce_smax,
          Intrinsic::ripple_reduce_smin, Intrinsic::ripple_reduce_umax,
          Intrinsic::ripple_reduce_umin, Intrinsic::ripple_reduce_fadd,
          Intrinsic::ripple_reduce_fmul, Intrinsic::ripple_reduce_fmin,
          Intrinsic::ripple_reduce_fmax, Intrinsic::ripple_reduce_fminimum,
          Intrinsic::ripple_reduce_fmaximum});
}

IntrinsicInst *Ripple::rippleShuffleIntrinsics(Instruction *I) {
  return intrinsicWithId(
      I, {Intrinsic::ripple_ishuffle, Intrinsic::ripple_fshuffle});
}

IntrinsicInst *Ripple::rippleBroadcastIntrinsic(Instruction *I) {
  return intrinsicWithId(I, {Intrinsic::ripple_broadcast});
}

IntrinsicInst *Ripple::rippleSliceIntrinsic(Instruction *I) {
  return intrinsicWithId(I, {Intrinsic::ripple_slice});
}

IntrinsicInst *Ripple::rippleIntrinsicsWithBlockShapeOperand(Instruction *I) {
  return intrinsicWithId(I, {Intrinsic::ripple_broadcast,
                             Intrinsic::ripple_block_getsize,
                             Intrinsic::ripple_block_index});
}

void Ripple::initFuncRPOT() {
  // Change the control flow to have only one ReturnInst and UnreachableInst in
  // the Function
  SmallVector<DominatorTree::UpdateType, 8> DomTreeUpdates;
  unifyUnreachableBlocks(F, &DomTreeUpdates);
  unifyReturnBlocks(F, &DomTreeUpdates);
  if (!DomTreeUpdates.empty()) {
    DTU.applyUpdates(DomTreeUpdates);
    DTU.flush();
  }

  FuncRPOT = new ReversePostOrderTraversal<Function *>(&F);
}

BitVector Ripple::reductionTensorDimensions(const IntrinsicInst *I) const {
  // All we need are the dimensions of Input that are not in the result of the
  // reduction
  auto InputSet = getRippleShape(I->getArgOperand(1)).nonEmptyDims();
  auto ResultSet = getRippleShape(I).nonEmptyDims();
  for (auto SetOutput : ResultSet.set_bits()) {
    InputSet.reset(SetOutput);
  }
  return InputSet;
}

unsigned Ripple::lastVectorIdx(const IntrinsicInst *I,
                               const TensorShape &IShape,
                               const unsigned SpecialArgIdx,
                               const char *OpKind) const {
  assert(rippleReduceIntrinsics(I) || rippleSliceIntrinsic(I));

  // The API only allows reducing/broadcasting vector dimensions for now; find
  // the one being reduced/bcast automatically (only one vector PE can be used
  // for a given Instruction, this is checked by the ripple semantics checker)
  unsigned SetVectorId = tensorRank();
  for (unsigned Idx = 0, EndIdx = tensorRank(); Idx < EndIdx; ++Idx) {
    if (IShape[Idx] > 1 && isVectorId(tensorToRipple(Idx).first))
      SetVectorId = Idx;
  }

  if (SetVectorId == tensorRank()) {
    // No vector dimensions being reduced/broadcasted
    std::string ErrMsg;
    llvm::raw_string_ostream RSO(ErrMsg);
    RSO << "Ripple " << OpKind
        << " used on a tensor with empty vector dimensions: "
           "input is (";
    Value *TensorArg = I->getArgOperand(SpecialArgIdx);
    if (TensorArg->getName().empty())
      RSO << *TensorArg;
    else
      RSO << TensorArg->getName();
    RSO << ") with shape " << IShape;
    RSO.flush();
    DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                   ErrMsg);
    F.getContext().diagnose(DI);
  }
  return SetVectorId;
}

Expected<TensorShape>
Ripple::computeRippleShapeForBitsetIntrinsic(const IntrinsicInst *I,
                                             const TensorShape &IShape) {
  assert(rippleReduceIntrinsics(I) || rippleBroadcastIntrinsic(I));

  bool IsReduction = rippleReduceIntrinsics(I);

  // When the machine model does not specify vector dimensions, the reduction or
  // broadcast is a no-op
  if ((IsReduction && IShape.isScalar()) || hasNoVectorDimension())
    return IShape;

  // Initialize an all false BitVector
  TensorShape OutputShape = IShape;

  unsigned BitSetArgIdx = IsReduction
                              ? &I->getArgOperandUse(0) - I->arg_begin()
                              : &I->getArgOperandUse(1) - I->arg_begin();

  // Guaranteed to be a constant because the intrinsics require an immediate
  auto BitsetArgValue = *getConstantOperandValue(I, BitSetArgIdx);

  IntrinsicInst *BcastBlockShape = nullptr;
  // Only vector dimensions are allowed
  unsigned SetVectorId = tensorRank();
  if (IsReduction)
    SetVectorId = lastVectorIdx(I, IShape, 1, "reduction");
  else {
    BcastBlockShape = getBlockShapeIntrinsic(I->getArgOperandUse(0));
    // Checked by checkBlockShapeUsage
    assert(BcastBlockShape);
    PEIdentifier PEId = *getConstantOperandValue(BcastBlockShape, 0);
    SetVectorId = rippleToTensor({PEId, 0});
  }
  if (SetVectorId >= tensorRank())
    return OutputShape;

  auto [PEBeingAffected, _] = tensorToRipple(SetVectorId);
  unsigned NumPEDimensions = PERank(PEBeingAffected);
  BitVector AffectedDimensions(tensorRank());
  for (uint64_t BitMask = 1, DimIdx = 0; BitMask != 0;
       BitMask = BitMask << 1, ++DimIdx) {
    // dimIdx ripple dimension is set
    if (BitsetArgValue & BitMask) {
      if (DimIdx < NumPEDimensions) {
        auto AffectedDim = rippleToTensor({PEBeingAffected, DimIdx});
        AffectedDimensions.set(AffectedDim);
      } else {
        auto printBinary = [](raw_ostream &OS, uint64_t value) {
          std::bitset<64> bits(value);
          std::string binaryString = bits.to_string();
          // Remove leading zeros
          binaryString.erase(0, binaryString.find_first_not_of('0'));
          OS << "0b" << binaryString;
        };

        // warn about an out of bounds reduction/broadcast dimension index
        std::string ErrMsg;
        llvm::raw_string_ostream RSO(ErrMsg);
        RSO << "Ripple " << (IsReduction ? "reduction" : "broadcast")
            << " applied on the dimension index (" << DimIdx
            << ") of processing element (PE number " << PEBeingAffected
            << ") that is out of bounds [0, " << NumPEDimensions
            << "[ (all set mask is ";
        uint64_t AllOneMaskForPE = (uint64_t(1) << NumPEDimensions) - 1;
        printBinary(RSO, AllOneMaskForPE);
        RSO << "); did you mean to use the mask (";
        printBinary(RSO, BitsetArgValue & AllOneMaskForPE);
        RSO << ")?";
        RSO.flush();
        DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                       ErrMsg);
        F.getContext().diagnose(DI);
      }
    }
  }
  if (IsReduction)
    OutputShape.reduceDimensions(AffectedDimensions);
  else {
    // Broadcast
    TensorShape BlockShape = setShapeToTensorShape(BcastBlockShape);
    LLVM_DEBUG(dbgs() << "ripple.broadcast block is " << *BcastBlockShape
                      << " with " << BlockShape << "\n");
    BlockShape.keepDimensions(AffectedDimensions);
    LLVM_DEBUG(dbgs() << "Max after reduce " << BlockShape << "\n");
    // And then broadcast InputShape by what's left
    if (auto Err = OutputShape.combineShapeBcast(BlockShape))
      return std::move(Err);
  }

  return OutputShape;
}

Error Ripple::validate() const {
  const std::string ErrorPrefix = "Ripple validation error: ";
  for (auto &[DimId, DimRank] : PERanks) {
    // Check that the ID is known by the compiler
    if (idType(DimId) == UnknownDimType) {
      std::string errStr;
      raw_string_ostream oss(errStr);
      oss << ErrorPrefix << "the dimension identifier (" << DimId
          << ") is used in function " << F.getName()
          << " but has not been registered; it is either missing from the "
             "machine model or the input program is malformed.";
      oss.flush();
      return createStringError(inconvertibleErrorCode(), errStr);
    }
  }
  return Error::success();
}

void Ripple::printValidSeries(raw_ostream &OSS) const {
  for (auto &[_, LS] : LsCache.Valid) {
    OSS << *LS << "\n";
  }
}

void Ripple::printTensorInstructions(raw_ostream &oss) const {
  auto dimTypeStr = [&](DimType type) {
    switch (type) {
    case VectorDimension:
      return "Vector";
    case ThreadDimension:
      return "Thread";
    case CoreDimension:
      return "Core";
    case DeviceDimension:
      return "Device";
    default:
      return "Unknown";
    }
  };
  auto printDimTypes = [&]() {
    oss << "DimTypes{";
    for (unsigned i = 0; i < tensorRank(); i++) {
      if (i != 0)
        oss << ", ";
      oss << dimTypeStr(tensorDimType(i));
    }
    oss << "}\n";
  };
  for (auto &shapeIt : InstructionRippleShapes) {
    if (shapeIt.second.isVector()) {
      oss << "\tInstruction " << *shapeIt.first << "\n\t\t" << shapeIt.second
          << " ";
      printDimTypes();
    }
  }
}

void Ripple::genVectorInstructions() {
  /// State to store if any changes were done to the CFG that made
  /// this->FuncRPOT to be invalidated *after* generating vector instructions.
  bool isFuncRPOTValid = true;

  auto vectorizedName = [](Instruction *I) -> Twine {
    return Twine(I->getName()) + ".ripple.vectorized";
  };

  auto processFreeze = [&](FreezeInst *Freeze,
                           const TensorShape &ToShape) -> void {
    irBuilder.SetInsertPoint(Freeze);
    auto [FreezeVal, FreezeShape] = getTensorUse(Freeze->getOperandUse(0));
    assert(*FreezeShape == ToShape);
    auto *Freezed = irBuilder.CreateFreeze(
        FreezeVal, tensorizedName(Freeze->getName(), ToShape));
    setReplacementFor(Freeze, Freezed, ToShape);
  };

  // TODO: We should probably implement ValueMaterializer and use a ValueMapper
  // to remap the values instead of this function followed by a cleanup.
  auto processRippleBlock = [&](IntrinsicInst *rippleDim,
                                const TensorShape &toShape) -> void {
    switch (rippleDim->getIntrinsicID()) {
    case Intrinsic::ripple_block_index: {
      // The propagation is handled by Linear Series, here we replace the
      // instruction by the value zero.
      assert(getCachedSeries(rippleDim));
      Constant *c = ConstantInt::get(rippleDim->getType(), 0);
      setReplacementFor(rippleDim, c, toShape);
    } break;
    case Intrinsic::ripple_block_getsize: {
      // Replace ripple.block.getsize calls by a constant
      auto size = getRippleGetSizeValue(rippleDim);
      Constant *c = ConstantInt::get(rippleDim->getType(), size);
      setReplacementFor(rippleDim, c, toShape);
    } break;
    case Intrinsic::ripple_block_setshape: {
      // Erase this intrinsic
      setReplacementFor(rippleDim, nullptr, toShape);
    } break;
    }
  };

  auto processRippleBroadcast = [&](IntrinsicInst *RippleBcast,
                                    const TensorShape &ToShape) -> void {
    // Broadcast the first argument to RippleBcast's shape
    irBuilder.SetInsertPoint(RippleBcast);
    auto VectorizedArg =
        getTensorUseAndBcast(RippleBcast->getArgOperandUse(2), ToShape);
    setReplacementFor(RippleBcast, VectorizedArg, ToShape);
  };

  auto processRippleSlice = [&](IntrinsicInst *RippleSlice,
                                const TensorShape &ToShape) -> void {
    irBuilder.SetInsertPoint(RippleSlice);
    // Expected signature: T ripple_slice(T x, int64_t idx0, int64_t idx1, ...,
    // int64_t idx9); where idx0 ... idx9 are constants
    auto [Slicee, FromShape] = getTensorUse(RippleSlice->getArgOperandUse(0));
    // No slicing -> ripple_slice is the identity
    if (ToShape == *FromShape) {
      setReplacementFor(RippleSlice, Slicee, ToShape);
      return;
    }
    SmallVector<int64_t, RippleIntrinsicsMaxDims> SliceArgs;
    llvm::transform(
        make_range(&RippleSlice->getArgOperandUse(1),
                   &RippleSlice->getArgOperandUse(1 + FromShape->rank())),
        std::back_inserter(SliceArgs), [](Use &U) -> int64_t {
          ConstantInt *Arg = cast<ConstantInt>(U.get());
          return Arg->getSExtValue();
        });
    std::vector<int> ShuffleIndices;
    Value *SliceInst;
    if (ToShape.isScalar()) {
      SmallVector<size_t, RippleIntrinsicsMaxDims> SliceIndex;
      for (unsigned Idx = 0, n = SliceArgs.size(); Idx < n; ++Idx) {
        assert(SliceArgs[Idx] != -1);
        SliceIndex.push_back((size_t)SliceArgs[Idx]);
      }
      ShuffleIndices.push_back(FromShape->getOffsetAt(SliceIndex));
      SliceInst = irBuilder.CreateExtractElement(Slicee, ShuffleIndices[0],
                                                 "ripple.slice");
    } else {
      // Capture of struct binding (FromShape) is c++20, use a copy
      auto *FromShapePtr = FromShape;
      auto computeShuffleIndices =
          [&, FromShapePtr](ArrayRef<size_t> index) -> void {
        SmallVector<size_t, RippleIntrinsicsMaxDims> SliceIndex;
        for (unsigned Idx = 0, n = FromShapePtr->rank(); Idx < n; ++Idx) {
          int64_t SliceArg = SliceArgs[Idx];
          SliceIndex.push_back(SliceArg < 0 ? (size_t)index[Idx]
                                            : (size_t)SliceArg);
        }
        ShuffleIndices.push_back(FromShapePtr->getOffsetAt(SliceIndex));
      };
      ToShape.foreachIndex(computeShuffleIndices);
      SliceInst =
          irBuilder.CreateShuffleVector(Slicee, ShuffleIndices, "ripple.slice");
    }
    setReplacementFor(RippleSlice, SliceInst, ToShape);
  };

  auto processRippleReductions = [&](IntrinsicInst *rippleReduction,
                                     const TensorShape &toShape) -> void {
    auto rippleToVPReduce = [](Intrinsic::ID rippleReduction) -> Intrinsic::ID {
      switch (rippleReduction) {
      default:
        llvm_unreachable("Not a Ripple reduction instruction");
      case Intrinsic::ripple_reduce_add:
        return Intrinsic::vp_reduce_add;
      case Intrinsic::ripple_reduce_mul:
        return Intrinsic::vp_reduce_mul;
      case Intrinsic::ripple_reduce_and:
        return Intrinsic::vp_reduce_and;
      case Intrinsic::ripple_reduce_or:
        return Intrinsic::vp_reduce_or;
      case Intrinsic::ripple_reduce_xor:
        return Intrinsic::vp_reduce_xor;
      case Intrinsic::ripple_reduce_smax:
        return Intrinsic::vp_reduce_smax;
      case Intrinsic::ripple_reduce_smin:
        return Intrinsic::vp_reduce_smin;
      case Intrinsic::ripple_reduce_umax:
        return Intrinsic::vp_reduce_umax;
      case Intrinsic::ripple_reduce_umin:
        return Intrinsic::vp_reduce_umin;
      case Intrinsic::ripple_reduce_fadd:
        return Intrinsic::vp_reduce_fadd;
      case Intrinsic::ripple_reduce_fmul:
        return Intrinsic::vp_reduce_fmul;
      case Intrinsic::ripple_reduce_fmin:
        return Intrinsic::vp_reduce_fmin;
      case Intrinsic::ripple_reduce_fmax:
        return Intrinsic::vp_reduce_fmax;
      case Intrinsic::ripple_reduce_fminimum:
        return Intrinsic::vp_reduce_fminimum;
      case Intrinsic::ripple_reduce_fmaximum:
        return Intrinsic::vp_reduce_fmaximum;
      }
    };
    irBuilder.SetInsertPoint(rippleReduction);

    auto [RedValue, RedValueShape] =
        getTensorUse(rippleReduction->getArgOperandUse(1));
    auto ReductionShape = getRippleShape(rippleReduction);

    if (*RedValueShape == ReductionShape) {
      // The reduction has the same shape as the input value; nothing to reduce!
      setReplacementFor(rippleReduction, RedValue, *RedValueShape);
    } else {
      auto VPReductionID = rippleToVPReduce(rippleReduction->getIntrinsicID());
      BitVector ReductionDims = reductionTensorDimensions(rippleReduction);
      FastMathFlags FMFRed = isa<FPMathOperator>(rippleReduction)
                                 ? rippleReduction->getFastMathFlags()
                                 : FastMathFlags();

      // Ripple reductions allow reassoc
      FMFRed.setAllowReassoc();
      Value *ReductionResult = genMultiDimReduction(
          VPReductionID, RedValue, *RedValueShape, ReductionDims, FMFRed);
      setReplacementFor(rippleReduction, ReductionResult, ReductionShape);
    }
  };

  auto createVectorPHI = [&](PHINode *oldPhi,
                             const TensorShape &toShape) -> void {
    // We create a dummy vector PHI and will fix its arguments later, once we
    // all the vector values have been generated
    Type *newPhiType = VectorType::get(oldPhi->getType()->getScalarType(),
                                       toShape.flatShape(), /*scalable*/ false);
    irBuilder.SetInsertPoint(oldPhi);
    PHINode *newPhi = irBuilder.CreatePHI(
        newPhiType, oldPhi->getNumIncomingValues(), vectorizedName(oldPhi));
    setReplacementFor(oldPhi, newPhi, toShape);
  };

  auto processComparisons = [&](CmpInst *cmp,
                                const TensorShape &toShape) -> void {
    auto bcastOps = tensorizedOperandsAndBroadcast(cmp, toShape);
    assert(bcastOps.size() == 2);
    irBuilder.SetInsertPoint(cmp);
    Value *vectorCompare = irBuilder.CreateCmp(
        cmp->getPredicate(), bcastOps[0], bcastOps[1], vectorizedName(cmp));
    setReplacementFor(cmp, vectorCompare, toShape);
  };

  auto processSelects = [&](SelectInst *Select,
                            const TensorShape &toShape) -> void {
    auto BcastOps = tensorizedOperandsAndBroadcast(Select, toShape);
    irBuilder.SetInsertPoint(Select);
    Value *VecSelect = irBuilder.CreateSelect(
        BcastOps[0], BcastOps[1], BcastOps[2], vectorizedName(Select));
    setReplacementFor(Select, VecSelect, toShape);
  };

  auto processBinaryOps = [&](BinaryOperator *binOp,
                              const TensorShape &toShape) -> void {
    auto bcastedOperands = tensorizedOperandsAndBroadcast(binOp, toShape);
    assert(bcastedOperands.size() == 2);
    irBuilder.SetInsertPoint(binOp);
    IRBuilder<>::FastMathFlagGuard FMFGuard(irBuilder);
    FastMathFlags FMF = isa<FPMathOperator>(binOp) ? binOp->getFastMathFlags()
                                                   : FastMathFlags{};
    irBuilder.setFastMathFlags(FMF);
    Value *newBinop = irBuilder.CreateBinOp(
        binOp->getOpcode(), bcastedOperands[0], bcastedOperands[1],
        vectorizedName(binOp), binOp->getMetadata(LLVMContext::MD_fpmath));
    setReplacementFor(binOp, newBinop, toShape);
  };

  auto processCasts = [&](CastInst *castInst,
                          const TensorShape &toShape) -> void {
    Type *toType = VectorType::get(castInst->getType()->getScalarType(),
                                   toShape.flatShape(), /*scalable*/ false);
    auto [cachedVal, _] = getTensorUse(castInst->getOperandUse(0));

    assert(cachedVal && "Did not visit the predecessor of a vector cast");
    assert(isa<VectorType>(cachedVal->getType()) &&
           cast<VectorType>(cachedVal->getType())
                   ->getElementCount()
                   .getKnownMinValue() == toShape.flatShape());
    irBuilder.SetInsertPoint(castInst);
    Value *vectorCast = irBuilder.CreateCast(castInst->getOpcode(), cachedVal,
                                             toType, vectorizedName(castInst));
    setReplacementFor(castInst, vectorCast, toShape);
  };

  auto processGEP = [&](GetElementPtrInst *Gep,
                        const TensorShape &toShape) -> void {
    auto BcastedPtr = getTensorUseAndBcast(
        Gep->getOperandUse(Gep->getPointerOperandIndex()), toShape);

    unsigned FirstIndex = std::distance(Gep->op_begin(), Gep->idx_begin());
    auto BcastedIndices = tensorizedOperandsAndBroadcast(
        Gep, toShape, FirstIndex, FirstIndex + Gep->getNumIndices());

    irBuilder.SetInsertPoint(Gep);
    Value *vecGEP = irBuilder.CreateGEP(Gep->getSourceElementType(), BcastedPtr,
                                        BcastedIndices, vectorizedName(Gep),
                                        Gep->isInBounds());
    LLVM_DEBUG(dbgs() << "Created GEP " << *vecGEP << " of type "
                      << *vecGEP->getType() << " source type "
                      << *Gep->getSourceElementType() << " Ptr " << *BcastedPtr
                      << " indices: ";
               std::for_each(BcastedIndices.begin(), BcastedIndices.end(),
                             [](auto *V) { dbgs() << V << " "; });
               dbgs() << "\n");
    setReplacementFor(Gep, vecGEP, toShape);
  };

  auto processLoad = [&](LoadInst *Load, const TensorShape &ToShape) -> void {
    auto [LoadPtr, LoadShape] =
        getTensorUse(Load->getOperandUse(Load->getPointerOperandIndex()));

    ConstructedSeries PointerOpSeries;

    if (AllocaInst *ThisAlloca =
            aliasesWithPromotableAlloca(MemoryLocation::get(Load))) {
      // Loads of promotable Alloca have shape-shifting capabilities

      // getRippleShape(LoadPtr) returns the maximum size of the alloca
      // used in the function, however here we want this load's shape instead
      // since we broadcasted before the store(s) that clobbers this load
      PointerOpSeries = getSplatSeries(ThisAlloca, ToShape, ToShape);
    } else {
      PointerOpSeries =
          getCachedSeries(dyn_cast<Instruction>(Load->getPointerOperand()));
      if (!PointerOpSeries) {
        PointerOpSeries = getSplatSeries(LoadPtr, *LoadShape, ToShape);
        assert(PointerOpSeries && "A load pointer can always become a series");
      }
    }

    // Here we Load from a set of addresses defined by a Linear Series.
    // The result is a tensor whose shape is that of the Linear Series.
    LinearSeries *AddressSeries = PointerOpSeries.LS;
    LLVM_DEBUG(dbgs() << "Generating series load for " << *Load << ':'
                      << *AddressSeries << '\n');
    auto NewLoad =
        NdLoadStoreFac.genLoad(Load, *AddressSeries, LoadPtr, ToShape);

    setReplacementFor(Load, NewLoad, ToShape);
  };

  auto processStore = [&](StoreInst *Store,
                          const TensorShape &ToShape) -> void {
    auto [VectorPtr, VectorShape] =
        getTensorUse(Store->getOperandUse(Store->getPointerOperandIndex()));
    if (VectorPtr == Store->getPointerOperand())
      report_fatal_error("Missing vector replacement for vectorized store");

    LLVM_DEBUG(dbgs() << "Storing to " << *VectorPtr << "\n");

    // TODO: broadcasting and storing the broadcasted value is not the most
    //       efficient way to do a broadcast store.
    //       It's best to not duplicate the regs and just store multiple times.
    //       Incorporate the broadcast dimensions into the store attributes.
    auto *VectorValue = getTensorUseAndBcast(Store->getOperandUse(0), ToShape);

    AllocaInst *AllocaPtr = dyn_cast<AllocaInst>(VectorPtr);
    // Could be an alloca promotion!
    if (AllocaPtr && VectorShape) {
      // We can issue an aligned store to the alloca!
      irBuilder.SetInsertPoint(Store);
      Value *AllocaStore = irBuilder.CreateStore(VectorValue, AllocaPtr);
      setReplacementFor(Store, AllocaStore, ToShape);
      return;
    }

    if (!(VectorPtr->getType()->isVectorTy() &&
          VectorPtr->getType()->getScalarType()->isPointerTy()))
      report_fatal_error("Expected a vector of pointers for vector store");

    // Here we load from a set of addresses defined by a Linear Series.
    // The result is a tensor whose shape is that of the Linear Series.
    auto Series =
        getCachedSeries(cast<Instruction>(Store->getPointerOperand()));
    Value *NewStore;

    if (*VectorShape != ToShape) {
      auto Bcasted = tensorBcast(VectorPtr, *VectorShape, ToShape);
      if (!Bcasted)
        report_fatal_error("Broadcast failure during codegen");
      VectorPtr = *Bcasted;
    }

    if (!Series || !Series.isValid()) {
      NewStore = NdLoadStoreFac.genUnstructuredStore(Store, VectorValue,
                                                     VectorPtr, ToShape);
    } else {
      LinearSeries *AddressSeries = Series.LS;
      NewStore = NdLoadStoreFac.genStore(Store, VectorValue, *AddressSeries,
                                         VectorPtr, ToShape);
    }

    setReplacementFor(Store, NewStore, ToShape);
  };

  auto processRippleShuffles = [&](IntrinsicInst *rippleShuffle,
                                   const TensorShape &toShape) -> void {
    auto [VecToShuffle, ShuffleShape] =
        getTensorUse(rippleShuffle->getArgOperandUse(0));
    auto VecToShuffleWith =
        getTensorUseAndBcast(rippleShuffle->getArgOperandUse(1), toShape);

    bool IsPairShuffle =
        !cast<ConstantInt>(rippleShuffle->getArgOperand(2))->isZero();

    if (IsPairShuffle) {
      // Pair may require an extra broadcast
      auto Bcasted = tensorBcast(VecToShuffle, *ShuffleShape, toShape);
      // Checked during shape propagation
      if (!Bcasted)
        report_fatal_error("Broadcast failure during codegen");
      VecToShuffle = *Bcasted;
    }

    // For PairShuffle we need to "select" the LHS or RHS depending on the index
    // value (0 or 1)
    if (!IsPairShuffle) {
      if (toShape.isScalar() && !IsPairShuffle) {
        assert(ShuffleShape->isScalar() &&
               VecToShuffle == rippleShuffle->getArgOperand(0));

        setReplacementFor(rippleShuffle, VecToShuffle, toShape);
        return;
      }
      assert(VecToShuffle != rippleShuffle->getArgOperand(0));
    }

    // We made sure that this is a valid function earlier
    Function *ShuffleFunc = cast<Function>(rippleShuffle->getArgOperand(3));
    FunctionType *ShuffleFuncType = ShuffleFunc->getFunctionType();

    irBuilder.SetInsertPoint(rippleShuffle);
    DimSize BlockSize = toShape.flatShape();
    std::vector<int> PermIdxs(BlockSize, 0);

    Evaluator Evaler(DL, &targetLibraryInfo);

    for (DimSize IIdx = 0; IIdx < BlockSize; ++IIdx) {
      Constant *RetVal;
      Constant *IdxArg =
          ConstantInt::get(ShuffleFuncType->getParamType(0), IIdx);
      Constant *BlockSizeArg =
          ConstantInt::get(ShuffleFuncType->getParamType(1), BlockSize);
      SmallVector<Constant *, 2> Args = {IdxArg, BlockSizeArg};

      if (!Evaler.EvaluateFunction(ShuffleFunc, RetVal, Args)) {
        // This is checked by checkRippleShuffleIntrinsics
        report_fatal_error("Unexpected Evaler failure");
      }
      ConstantInt *RetIntVal = cast<ConstantInt>(RetVal);
      PermIdxs[IIdx] = RetIntVal->getSExtValue();
    }

    if (IsPairShuffle && toShape.isScalar()) {
      // This case is similar to a select
      assert(PermIdxs.size() == 1);
      setReplacementFor(rippleShuffle,
                        PermIdxs[0] == 0 ? VecToShuffle : VecToShuffleWith,
                        toShape);
      return;
    }

    Value *ShuffledVec = irBuilder.CreateShuffleVector(
        VecToShuffle,
        IsPairShuffle ? VecToShuffleWith
                      : PoisonValue::get(VecToShuffle->getType()),
        PermIdxs, vectorizedName(rippleShuffle));

    setReplacementFor(rippleShuffle, ShuffledVec, toShape);
  };

  auto ProcessIntrinsicCall = [&](CallInst *Call, const TensorShape &ToShape,
                                  Intrinsic::ID VectorIntrId) -> void {
    // Replace the Call with its corresponding vector intrinsic

    // Create a vector type with the target shape and broadcast the Call
    // arguments to match the shape
    unsigned FirstArgIdx = std::distance(Call->op_begin(), Call->arg_begin());

    auto BcastedArgs = tensorizedOperandsAndBroadcast(
        Call, ToShape, FirstArgIdx, FirstArgIdx + Call->arg_size());
    irBuilder.SetInsertPoint(Call);

    // Get the Declaration of the intrinsic to verify all types of arguments
    SmallVector<Type *> BcastedArgsTypes;
    BcastedArgsTypes.reserve(Call->arg_size());
    llvm::transform(BcastedArgs, std::back_inserter(BcastedArgsTypes),
                    [](Value *V) { return V->getType(); });
    FunctionType *FTy =
        Intrinsic::getType(Call->getContext(), VectorIntrId, BcastedArgsTypes);

    // Check each arguments is overloaded type or not
    SmallVector<Value *, 8> BcastedArgsChecked;
    BcastedArgsChecked.reserve(FTy->getNumParams());
    for (unsigned Idx = 0; Idx < FTy->getNumParams(); ++Idx) {
      if (FTy->getParamType(Idx) == BcastedArgs[Idx]->getType()) {
        BcastedArgsChecked.push_back(BcastedArgs[Idx]);
      } else {
        auto *Val = Call->getOperand(Idx);
        assert(FTy->getParamType(Idx) == Val->getType());
        BcastedArgsChecked.push_back(Val);
      }
    }

    // Create a vectorized Intrinsic with arguments that should be vectorized
    CallInst *VecCall = irBuilder.CreateIntrinsic(
        FTy->getReturnType(), VectorIntrId, BcastedArgsChecked,
        isa<FPMathOperator>(Call) ? Call : nullptr, vectorizedName(Call));

    setReplacementFor(Call, VecCall, ToShape);
  };

  auto ProcessGeneralFunctionCall = [&](CallInst *Call,
                                        const TensorShape &ToShape) -> void {
    LLVM_DEBUG(dbgs() << "\nFunction before ProcessGeneralFunctionCall:\n\n";
               F.print(dbgs(), nullptr); dbgs() << "\n");
    LLVM_DEBUG(dbgs() << "Sequentializing this Call " << *Call << "\n");
    // Sequentialize the Call by extracting each element of the vector operands,
    // running the scalar function on them, and insert the scalar result into
    // the output vector.

    // The Index type for GEP of the Alloca buffers
    Type *AllocaIndexType =
        DL.getIndexType(F.getContext(), DL.getAllocaAddrSpace());

    SmallVector<Value *, 4> CallRippleArgs;
    for (auto &Arg : Call->args()) {
      if (getRippleShape(Arg).isScalar()) {
        CallRippleArgs.push_back(Arg);
      } else {
        auto BcastedVal = getTensorUseAndBcast(Arg, ToShape);
        CallRippleArgs.push_back(BcastedVal);
      }
    }

    // Setup loop blocks
    BasicBlock *BeforeLoop = Call->getParent();
    BasicBlock *LoopPreHeader =
        SplitBlock(Call->getParent(), Call->getIterator(), &DTU, nullptr,
                   nullptr, Twine(Call->getName()) + ".ripple.call.loop.pre");
    setRippleShape(BeforeLoop->getTerminator(), ScalarShape);
    assert(Call->getParent() == LoopPreHeader);
    BasicBlock *LoopHeader =
        SplitBlock(LoopPreHeader, Call->getIterator(), &DTU, nullptr, nullptr,
                   Twine(Call->getName()) + ".ripple.call.loop.header");
    setRippleShape(LoopPreHeader->getTerminator(), ScalarShape);
    assert(Call->getParent() == LoopHeader);
    BasicBlock *LoopBody =
        SplitBlock(LoopHeader, Call->getIterator(), &DTU, nullptr, nullptr,
                   Twine(Call->getName()) + ".ripple.call.loop.body");
    setRippleShape(LoopHeader->getTerminator(), ScalarShape);
    assert(Call->getParent() == LoopBody);
    BasicBlock *CallBlock = SplitBlock(
        LoopBody, Call->getIterator(), &DTU, nullptr, nullptr,
        Twine(Call->getName()) + ".ripple.call.loop.body.call.block");
    setRippleShape(LoopBody->getTerminator(), ScalarShape);
    assert(Call->getParent() == CallBlock);

    BasicBlock *ContinueBlock = SplitBlock(
        CallBlock, std::next(Call->getIterator()), &DTU, nullptr, nullptr,
        Twine(Call->getName()) + ".ripple.call.loop.body.continue.block");
    setRippleShape(CallBlock->getTerminator(), ScalarShape);

    BasicBlock *AfterLoop =
        SplitBlock(ContinueBlock, ContinueBlock->begin(), &DTU, nullptr,
                   nullptr, Twine(Call->getName()) + ".ripple.call.loop.end");
    setRippleShape(ContinueBlock->getTerminator(), ScalarShape);

    cast<BranchInst>(ContinueBlock->getTerminator())
        ->setSuccessor(0, LoopHeader);
    DTU.applyUpdates({{DominatorTree::Delete, ContinueBlock, AfterLoop},
                      {DominatorTree::Insert, ContinueBlock, LoopHeader}});

    // Create induction variable
    irBuilder.SetInsertPoint(LoopHeader->begin());
    irBuilder.SetInstDebugLocation(Call);
    PHINode *InductionVar =
        irBuilder.CreatePHI(AllocaIndexType, 2, "ripple.scalarcall.iterator");
    setRippleShape(InductionVar, ScalarShape);
    InductionVar->addIncoming(ConstantInt::get(AllocaIndexType, 0),
                              LoopPreHeader);
    VectorType *MaskVectorType = VectorType::get(
        irBuilder.getInt8Ty(), ElementCount::getFixed(ToShape.flatShape()));

    // Increment induction variable
    irBuilder.SetInsertPoint(ContinueBlock->getTerminator());
    irBuilder.SetInstDebugLocation(Call);
    Value *IncrementedInductionVar =
        irBuilder.CreateAdd(InductionVar, ConstantInt::get(AllocaIndexType, 1));
    setRippleShape(IncrementedInductionVar, ScalarShape);
    InductionVar->addIncoming(IncrementedInductionVar, ContinueBlock);

    // Initialize function call mask with as select(True, True, False)
    // which ensures getting a value of 1 for active vector
    // lanes and 0 for the rest after IfConvert is applied
    irBuilder.SetInsertPoint(LoopHeader->getTerminator());
    SelectInst *MaskSelect = irBuilder.Insert(
        createMaskSelectToTrueFalse(irBuilder.getInt8Ty(),
                                    MaskVectorType->getElementCount()),
        "zeroinit.select");
    setRippleShape(MaskSelect, ToShape);
    // Enable masking for this select
    SelectToMaskWhenIfConvert.insert(MaskSelect);

    Value *MaskVal;
    irBuilder.SetInsertPoint(LoopBody->getTerminator());
    MaskVal = irBuilder.CreateExtractElement(MaskSelect, InductionVar);
    setRippleShape(MaskVal, ScalarShape);

    // Add mask condition in LoopBody
    irBuilder.SetInsertPoint(LoopBody->getTerminator());
    irBuilder.SetInstDebugLocation(Call);
    Value *MaskCond = irBuilder.CreateICmpEQ(
        MaskVal, ConstantInt::get(irBuilder.getInt8Ty(), 1));
    setRippleShape(MaskCond, ScalarShape);
    Value *BranchInst =
        irBuilder.CreateCondBr(MaskCond, CallBlock, ContinueBlock);
    setRippleShape(BranchInst, ScalarShape);
    invalidateRippleDataFor(LoopBody->getTerminator());
    LoopBody->getTerminator()->eraseFromParent();

    // Create loop condition
    irBuilder.SetInsertPoint(LoopHeader->getTerminator());
    irBuilder.SetInstDebugLocation(Call);
    Value *LoopCond = irBuilder.CreateICmpULT(
        InductionVar, ConstantInt::get(AllocaIndexType, ToShape.flatShape()));
    setRippleShape(LoopCond, ScalarShape);

    // Create loop branch
    irBuilder.SetInsertPoint(LoopHeader);
    irBuilder.SetInstDebugLocation(Call);
    Instruction *OldTerminator = LoopHeader->getTerminator();
    Value *LoopBranchInst =
        irBuilder.CreateCondBr(LoopCond, LoopBody, AfterLoop);
    setRippleShape(LoopBranchInst, ScalarShape);
    invalidateRippleDataFor(OldTerminator);
    OldTerminator->eraseFromParent();
    DTU.applyUpdates({{DominatorTree::Insert, LoopHeader, AfterLoop}});
    DTU.flush();

    PHINode *ResValue = nullptr;

    Type *VectorCallType = Call->getType();
    if (!VectorCallType->isVoidTy()) {
      assert(VectorType::isValidElementType(VectorCallType));
      VectorCallType = VectorType::get(Call->getType(), ToShape.flatShape(),
                                       /*Scalable*/ false);
      // generate a PHI: init with Poison from the LoopPreHeader and the
      // result of InsertElement from the LoopBody
      irBuilder.SetInsertPoint(LoopHeader->begin());
      irBuilder.SetInstDebugLocation(Call);
      PHINode *ResultValue = irBuilder.CreatePHI(VectorCallType, 2u);
      setRippleShape(ResultValue, ToShape);
      ResultValue->addIncoming(PoisonValue::get(VectorCallType), LoopPreHeader);
      ResValue = ResultValue;
    }

    // Create the scalar call inside the loop body w/ the extracted/buffer
    // values
    irBuilder.SetInsertPoint(CallBlock->getFirstNonPHIIt());

    SmallVector<Value *, 4> ScalarArgs;
    for (auto *Arg : CallRippleArgs) {
      if (Arg->getType()->isVectorTy()) {
        Value *Extracted = irBuilder.CreateExtractElement(Arg, InductionVar);
        setRippleShape(Extracted, ScalarShape);
        ScalarArgs.push_back(Extracted);
      } else
        ScalarArgs.push_back(Arg);
    }
    // Function call proper
    Value *ScalarCall = irBuilder.CreateCall(
        Call->getFunctionType(), Call->getCalledOperand(), ScalarArgs);
    setRippleShape(ScalarCall, ScalarShape);
    // If the function returns a value
    if (ResValue) {
      // Insert the element into the PHINode
      Value *Inserted =
          irBuilder.CreateInsertElement(ResValue, ScalarCall, InductionVar);
      setRippleShape(Inserted, ToShape);

      irBuilder.SetInsertPoint(ContinueBlock->begin());
      irBuilder.SetInstDebugLocation(Call);
      PHINode *UpdatedResPhi = irBuilder.CreatePHI(VectorCallType, 2u);
      setRippleShape(UpdatedResPhi, ToShape);
      UpdatedResPhi->addIncoming(Inserted, CallBlock);
      UpdatedResPhi->addIncoming(ResValue, LoopBody);
      // Which is the value coming back from the LoopBody
      ResValue->addIncoming(UpdatedResPhi, ContinueBlock);
      setReplacementFor(Call, ResValue, ToShape);
    } else {
      setReplacementFor(Call, nullptr, ToShape);
    }
    LLVM_DEBUG(dbgs() << "\nFunction after ProcessGeneralFunctionCall:\n\n";
               F.print(dbgs(), nullptr); dbgs() << "\n");

    // CFG has been changed => FuncRPOT is no longer valid.
    isFuncRPOTValid = false;
  };

  auto processCallInst = [&](CallInst *call,
                             const TensorShape &toShape) -> void {
    // Check if the call matches any specific ripple intrinsic
    if (IntrinsicInst *rippleDim = rippleBlockIntrinsics(call)) {
      processRippleBlock(rippleDim, toShape);
    } else if (IntrinsicInst *RippleBroadcast =
                   rippleBroadcastIntrinsic(call)) {
      processRippleBroadcast(RippleBroadcast, toShape);
    } else if (IntrinsicInst *rippleReduction = rippleReduceIntrinsics(call)) {
      processRippleReductions(rippleReduction, toShape);
    } else if (IntrinsicInst *rippleSlice = rippleSliceIntrinsic(call)) {
      processRippleSlice(rippleSlice, toShape);
    } else if (IntrinsicInst *rippleShuffle = rippleShuffleIntrinsics(call)) {
      processRippleShuffles(rippleShuffle, toShape);
      // ToShape can be scalar because of reductions/slicing
    } else if (toShape.isVector() || rippleVectorizeCall(*call)) {
      Intrinsic::ID vectorIntrId =
          getVectorIntrinsicIDForCall(call, &targetLibraryInfo);
      if (toShape.isVector() &&
                 vectorIntrId != Intrinsic::not_intrinsic) {
        // We assume vectorizing an intrinsic call does not
        // introduce approximations
        ProcessIntrinsicCall(call, toShape, vectorIntrId);
      } else {
        {
          assert(toShape.isVector() &&
                 "ripple general function call has broadcast "
                 "semantics and cannot return a scalar");
          // Sequential execution for non-intrinsic function calls or floating
          // point intrinsic-calls without fast math
          ProcessGeneralFunctionCall(call, toShape);
        }
      }
    }
  };

  auto processFneg = [&](UnaryInstruction *fneg,
                         const TensorShape &toShape) -> void {
    auto bcastOps = tensorizedOperandsAndBroadcast(fneg, toShape);
    irBuilder.SetInsertPoint(fneg);
    Value *newFneg =
        irBuilder.CreateFNegFMF(bcastOps[0], fneg, vectorizedName(fneg));
    setReplacementFor(fneg, newFneg, toShape);
  };

  auto processAlloca = [&](AllocaInst *Alloca,
                           const TensorShape &ToShape) -> void {
    irBuilder.SetInsertPoint(Alloca);
    Type *AllocatedTy = Alloca->getAllocatedType();
    assert(!AllocatedTy->isVectorTy() && "Ripple cannot promote vector types");
    AllocaInst *AllocaVector = irBuilder.CreateAlloca(
        ArrayType::get(AllocatedTy, ToShape.flatShape()),
        /*ArraySize*/ nullptr, tensorizedName(Alloca->getName(), ToShape));
    AllocaVector->setAlignment(std::max(
        AllocaVector->getAlign(),
        DL.getPrefTypeAlign(VectorType::get(AllocatedTy, ToShape.flatShape(),
                                            /*Scalable*/ false))));
    setReplacementFor(Alloca, AllocaVector, ToShape);
  };

  for (auto *BB : getFuncRPOT()) {
    // Gather the instructions in the BB to vectorize
    SmallVector<Instruction *, 16> toProcess;
    for (auto &I : *BB) {
      // For LinearSeries we need to process a vector base
      auto CS = getCachedSeries(&I);
      auto IShape = getRippleShape(&I);

      bool ProcessLinSeries = CS && CS.LS->getBaseShape().isVector();
      bool ProcessNonSeries = !CS && IShape.isVector();

      // Process instructions that should be vectors, but not linear series
      // Call instructions are special cases because they can be external ripple
      // functions/specializations/ripple intrinsic with a scalar shape!
      if (ProcessNonSeries || ProcessLinSeries || isa<CallInst>(&I)) {
        // For PHI nodes, we generate empty vector PHIs and fix them at the end
        if (PHINode *phi = dyn_cast<PHINode>(&I)) {
          createVectorPHI(phi, CS ? CS.LS->getBaseShape() : IShape);
        } else {
          toProcess.push_back(&I);
        }
      }
    }
    // Generate the vectorized instructions
    for (auto *I : toProcess) {
      auto CS = getCachedSeries(I);
      const TensorShape &toShape =
          CS ? CS.LS->getBaseShape() : getRippleShape(I);
      // TODO: Specialize for the different instruction kinds (InstrTypes.h +
      // Instructions.h), e.g., UnaryOp, BinaryOp, Phi, etc.
      LLVM_DEBUG(dbgs() << "Vectorizing instruction " << *I << " to shape "
                        << toShape << "\n");

      if (auto *binOp = dyn_cast<BinaryOperator>(I))
        processBinaryOps(binOp, toShape);
      else if (auto *castInst = dyn_cast<CastInst>(I))
        processCasts(castInst, toShape);
      else if (auto *gep = dyn_cast<GetElementPtrInst>(I))
        processGEP(gep, toShape);
      else if (auto *Load = dyn_cast<LoadInst>(I))
        processLoad(Load, toShape);
      else if (auto *Store = dyn_cast<StoreInst>(I))
        processStore(Store, toShape);
      else {
        switch (I->getOpcode()) {
        case Instruction::Call:
          processCallInst(cast<CallInst>(I), toShape);
          break;
        case Instruction::ICmp:
        case Instruction::FCmp:
          processComparisons(cast<CmpInst>(I), toShape);
          break;
        case Instruction::FNeg:
          processFneg(cast<UnaryInstruction>(I), toShape);
          break;
        case Instruction::Select:
          processSelects(cast<SelectInst>(I), toShape);
          break;
        case Instruction::Br: {
          // Branches are handled by the if-conversion function
          BranchInst *Branch = cast<BranchInst>(I);
          assert(Branch->isConditional());
          auto [VectorCondVal, ConditionShape] =
              getTensorUse(Branch->getOperandUse(0));
          assert(*ConditionShape == toShape);
          BranchAndSwitchVecCond[Branch] = VectorCondVal;
        } break;
        case Instruction::Switch: {
          SwitchInst *Switch = cast<SwitchInst>(I);
          auto [VectorCondVal, ConditionShape] =
              getTensorUse(Switch->getOperandUse(0));
          assert(*ConditionShape == toShape);
          BranchAndSwitchVecCond[Switch] = VectorCondVal;
        } break;
        case Instruction::Ret: {
          // Checked by checkRippleFunctionReturn
          report_fatal_error("Unsupported return instruction vectorization");
        } break;
        case Instruction::Alloca:
          processAlloca(cast<AllocaInst>(I), toShape);
          break;
        case Instruction::Freeze:
          processFreeze(cast<FreezeInst>(I), toShape);
          break;
        default: {
          std::string ErrMsg;
          llvm::raw_string_ostream RSO(ErrMsg);
          RSO << "instruction type not known by the ripple vectorizer; please "
                 "fill up a support request to the Ripple team: "
              << *I;
          RSO.flush();
          DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                         ErrMsg);
          F.getContext().diagnose(DI);
          // Don't crash here to catch more potential issues; the error is
          // diagnosed already
          if (I->getType()->isVoidTy()) {
            setReplacementFor(I, nullptr, toShape);
          } else {
            auto Shape = getRippleShape(I);
            Value *Replacement = PoisonValue::get(
                VectorType::get(I->getType()->getScalarType(),
                                toShape.flatShape(), /*scalable*/ false));
            setReplacementFor(I, Replacement, toShape);
          }
        } break;
        }
      }
    }
  }
  // Fill the PHIs now that all the instructions are generated
  for (auto &[From, To] : InstructionReplacementMapping) {
    if (PHINode *OldPhi = dyn_cast<PHINode>(&*From)) {
      PHINode *NewPhi = cast<PHINode>(&*To);
      LLVM_DEBUG(dbgs() << "Fixing generated PHI: " << *NewPhi
                        << " with value generated from " << *OldPhi << "\n");
      auto PhiOperands =
          tensorizedOperandsAndBroadcast(OldPhi, getRippleShape(NewPhi));
      for (unsigned PhiIdx = 0; PhiIdx < OldPhi->getNumIncomingValues();
           ++PhiIdx) {
        NewPhi->addIncoming(PhiOperands[PhiIdx],
                            OldPhi->getIncomingBlock(PhiIdx));
      }
    }
  }

  if (!isFuncRPOTValid) {
    // We modified the control flow so FuncRPOT is not valid anymore.
    // Note that using the FuncRPOT during the replacement of scalar values
    // with vector values is sound as "isFuncRPOTValid" is supposed to store
    // whether the FuncRPOT is invalid *after* the replacement.
    invalidateFuncRPOT();
  }
}

SmallVector<Value *, 8>
Ripple::tensorizedOperandsAndBroadcast(Instruction *I,
                                       const TensorShape &ToShape,
                                       unsigned StartIdx, unsigned EndIdx) {
  SmallVector<Value *, 8> BcastedOperands;
  for (unsigned OpIdx = StartIdx, E = std::min(I->getNumOperands(), EndIdx);
       OpIdx < E; ++OpIdx) {
    auto ReplacementAndBcast =
        getTensorUseAndBcast(I->getOperandUse(OpIdx), ToShape);
    BcastedOperands.push_back(ReplacementAndBcast);
  }
  return BcastedOperands;
}

Value *Ripple::getTensorUseAndBcast(const Use &U, const TensorShape &ToShape) {
  auto [Replacement, ReplacementShape] = getTensorUse(U);
  auto Bcasted = tensorBcast(Replacement, *ReplacementShape, ToShape);
  // Checked during shape propagation
  if (!Bcasted)
    report_fatal_error("Broadcast failure during codegen");
  return *Bcasted;
}

Expected<Value *> Ripple::tensorBcast(Value *V, const TensorShape &FromShape,
                                      const TensorShape &ToShape) {

  auto broadcastConstantTensor =
      [&](Constant *C, const TensorShape &fromShape,
          const TensorShape &toShape) -> Expected<Value *> {
    if (!C->getType()->isVectorTy() && toShape.isScalar())
      return C;
    if (auto *cstInt = dyn_cast<ConstantInt>(C)) {
      auto &val = cstInt->getValue();
      Type *baseTy = cstInt->getType()->getScalarType();
      Type *vectorTy =
          VectorType::get(baseTy, toShape.flatShape(), /*scalable*/ false);
      return ConstantInt::get(vectorTy, val);
    } else if (auto *cstFP = dyn_cast<ConstantFP>(C)) {
      Type *baseTy = cstFP->getType()->getScalarType();
      Type *vectorTy =
          VectorType::get(baseTy, toShape.flatShape(), /*scalable*/ false);
      auto &val = cstFP->getValue();
      return ConstantFP::get(vectorTy, val);
    } else if (C->isNullValue()) {
      // isNullValue() handles ConstantPointerNull, ConstantAggregateZero,
      // ConstantTokenNone, ConstantTargetNone
      Type *baseTy = C->getType()->getScalarType();
      if (baseTy->isVectorTy())
        baseTy = cast<VectorType>(baseTy)->getElementType();
      Type *vectorTy =
          VectorType::get(baseTy, toShape.flatShape(), /*scalable*/ false);
      return ConstantAggregateZero::get(vectorTy);
    } else if (isa<UndefValue>(C)) {
      Type *BaseTy = C->getType()->getScalarType();
      if (BaseTy->isVectorTy())
        BaseTy = cast<VectorType>(BaseTy)->getElementType();
      Type *VectorTy =
          VectorType::get(BaseTy, toShape.flatShape(), /*scalable*/ false);
      return UndefValue::get(VectorTy);
    } else if (auto *SplatVal =
                   C->getType()->isVectorTy() ? C->getSplatValue() : nullptr) {
      return ConstantVector::getSplat(
          ElementCount::getFixed(toShape.flatShape()), SplatVal);
    } else if (auto *cstDataVector = dyn_cast<ConstantDataVector>(C)) {
      std::vector<int> shuffleMask;
      auto addOffsetToMask = [&](ArrayRef<size_t> index) {
        size_t offset = fromShape.getOffsetAt(index);
        shuffleMask.push_back(offset);
      };
      toShape.foreachIndex(addOffsetToMask);
      return ConstantExpr::getShuffleVector(
          cstDataVector, PoisonValue::get(cstDataVector->getType()),
          shuffleMask);
    } else if (auto *CstVector = dyn_cast<ConstantVector>(C)) {
      std::vector<Constant *> BcastedCstVectorVals(toShape.flatShape());

      auto BuildBcastedVector =
          [&](ArrayRef<TensorShape::DimSize> ToMultiIndex) {
            SmallVector<TensorShape::DimSize> FromMultiIndex(fromShape.rank(),
                                                             0);
            int IDim = 0;
            for (auto FromAxisLen : fromShape) {
              FromMultiIndex[IDim] =
                  (FromAxisLen == 1) ? 0 : ToMultiIndex[IDim];
              IDim++;
            }
            BcastedCstVectorVals[toShape.getOffsetAt(ToMultiIndex)] =
                CstVector->getOperand(fromShape.getOffsetAt(FromMultiIndex));
          };

      toShape.foreachIndex(BuildBcastedVector);
      return ConstantVector::get(BcastedCstVectorVals);
    } else if (isa<GlobalValue>(C)) {
      // That's a global constant address splat
      // TODO: there might be issues with global value's semantics (Linkage,
      // thread_local, etc) and Ripple's semantics
      return ConstantVector::getSplat(
          ElementCount::getFixed(toShape.flatShape()), C);
    } else {
      // We don't support vectorization of ConstantStruct, ConstantArray,
      // ConstantDataArray, ConstantTargetNone, ConstantTokenNone
      std::string ErrMsg;
      llvm::raw_string_ostream RSO(ErrMsg);
      RSO << "ripple does not know how to broadcast the value " << *C
          << " (only ConstantInt, ConstantFP, ConstantPointerNull, "
             "ConstantAggregateZero, ConstantTokenNone, ConstantTargetNone, "
             "Splats, ConstantDataVector and ConstantVector are supported)";
      RSO.flush();
      DiagnosticInfoRippleWithLoc DI(DS_Error, F, {}, ErrMsg);
      F.getContext().diagnose(DI);
      return createStringError(std::errc::invalid_argument,
                               "Unsupported Constant type being broadcasted");
    }
  };

  if (FromShape == ToShape) {
    LLVM_DEBUG(dbgs() << "Reusing " << *V << " w/ shape " << ToShape << "\n");
    return V;
  }

  if (Error e = FromShape.isBroadcastError(ToShape)) {
    std::string ErrMsg;
    llvm::raw_string_ostream RSO(ErrMsg);
    RSO << "ripple cannot broadcast the value " << *V << " from shape "
        << FromShape << " to shape " << ToShape;
    llvm::handleAllErrors(std::move(e), [&](const StringError &SE) {
      RSO << ": " << SE.getMessage();
    });
    RSO.flush();
    DebugLoc DL = isa<Instruction>(V)
                      ? sanitizeRippleLocation(cast<Instruction>(V))
                      : DebugLoc();
    DiagnosticInfoRippleWithLoc DI(DS_Error, F, DL, ErrMsg);
    F.getContext().diagnose(DI);
    return createStringError(std::errc::invalid_argument,
                             "Broadcast shapes non compatible");
  }

  // TODO: We can probably keep a cache of broadcasted values
  LLVM_DEBUG(dbgs() << "Broadcasting " << *V << " from shape " << FromShape
                    << " to shape " << ToShape << "\n");

  if (auto *C = dyn_cast<Constant>(V)) {
    return broadcastConstantTensor(C, FromShape, ToShape);
  } else if (isa<Instruction>(V) || isa<Argument>(V)) {
    auto SaveIP = irBuilder.saveIP();
    if (isa<Argument>(V)) {
      auto InsertAt = F.getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
      irBuilder.SetInsertPoint(InsertAt);
    } else {
      auto *I = cast<Instruction>(V);
      if (auto *Invoke = dyn_cast<InvokeInst>(I)) {
        // Invokeinstruction is special because it is the only terminator that
        // can have users. So we must set the insertion point in a different
        // basic block.
        irBuilder.SetInsertPoint(
            Invoke->getNormalDest()->getFirstInsertionPt());
      } else {
        setInsertPointAfter(irBuilder, I);
      }
      if (I->getStableDebugLoc())
        irBuilder.SetCurrentDebugLocation(I->getStableDebugLoc());
    }

    Value *BcastedValue = nullptr;
    if (FromShape.isScalar()) {
      // Simple vector splat
      BcastedValue = irBuilder.CreateVectorSplat(
          ToShape.flatShape(), V, Twine(V->getName()) + ".ripple.bcast");
      // A Splat is an insert followed by a shuffle
      if (ShuffleVectorInst *ShuffleInst =
              dyn_cast<ShuffleVectorInst>(BcastedValue))
        setRippleShape(ShuffleInst->getOperand(0), ToShape);
    } else {
      // To construct the shuffle mask we map each index from toShape to the
      // offset in fromShape.
      std::vector<int> shuffleMask;
      auto addOffsetToMask = [&](ArrayRef<size_t> index) {
        size_t offset = FromShape.getOffsetAt(index);
        shuffleMask.push_back(offset);
      };
      ToShape.foreachIndex(addOffsetToMask);
      assert(shuffleMask.size() == ToShape.flatShape());
      BcastedValue = irBuilder.CreateShuffleVector(
          V, shuffleMask, Twine(V->getName()) + ".ripple.bcast");
    }
    setRippleShape(BcastedValue, ToShape);
    irBuilder.restoreIP(SaveIP);
    return BcastedValue;
  } else {
    std::string ErrMsg;
    llvm::raw_string_ostream RSO(ErrMsg);
    RSO << "ripple does not know how to broadcast the value " << *V
        << " (only Instructions, Arguments, and Constants are supported)";
    RSO.flush();
    DebugLoc DL = isa<Instruction>(V)
                      ? sanitizeRippleLocation(cast<Instruction>(V))
                      : DebugLoc();
    DiagnosticInfoRippleWithLoc DI(DS_Error, F, DL, ErrMsg);
    F.getContext().diagnose(DI);
    return createStringError(std::errc::invalid_argument,
                             "Unsupported Value type being broadcasted");
  }
}

bool Ripple::maskInstructionWhenIfConvert(const Instruction *I) const {
  return isa<LoadInst>(I) || isa<StoreInst>(I) ||
         intrinsicWithId(I,
                         {Intrinsic::masked_load, Intrinsic::masked_store,
                          Intrinsic::masked_gather, Intrinsic::masked_scatter,
                          Intrinsic::masked_expandload,
                          Intrinsic::masked_compressstore}) ||
         isa<VPIntrinsic>(I) || isa<SelectInst>(I);
}

DenseSet<BasicBlock *> Ripple::allBasicBlocksFromTo(BasicBlock *from,
                                                    BasicBlock *to) const {
  assert(postdomTree.dominates(to, from));
  std::queue<BasicBlock *> toProcess;
  DenseSet<BasicBlock *> visited;
  toProcess.push(from);
  while (!toProcess.empty()) {
    BasicBlock *BB = toProcess.front();
    toProcess.pop();
    visited.insert(BB);
    for (auto *SI : successors(BB)) {
      if (!visited.contains(SI) && SI != to)
        toProcess.push(SI);
    }
  }
  visited.erase(from);
  return visited;
}

Value *Ripple::genMultiDimReduction(Intrinsic::ID reductionId, Value *vector,
                                    const TensorShape &vectorShape,
                                    const BitVector &reductionDimensions,
                                    FMFSource FMFSource) {
  auto reductionBinOp =
      [](Intrinsic::ID redId) -> std::optional<Instruction::BinaryOps> {
    switch (redId) {
    default:
      return std::nullopt;
    case Intrinsic::vp_reduce_add:
      return Instruction::Add;
    case Intrinsic::vp_reduce_or:
      return Instruction::Or;
    case Intrinsic::vp_reduce_xor:
      return Instruction::Xor;
    case Intrinsic::vp_reduce_mul:
      return Instruction::Mul;
    case Intrinsic::vp_reduce_and:
      return Instruction::And;
    case Intrinsic::vp_reduce_fadd:
      return Instruction::FAdd;
    case Intrinsic::vp_reduce_fmul:
      return Instruction::FMul;
    }
  };
  auto cmpReduction = [](Intrinsic::ID redId) -> std::optional<Intrinsic::ID> {
    switch (redId) {
    default:
      return std::nullopt;
    case Intrinsic::vp_reduce_umin:
      return Intrinsic::umin;
    case Intrinsic::vp_reduce_smin:
      return Intrinsic::smin;
    case Intrinsic::vp_reduce_umax:
      return Intrinsic::umax;
    case Intrinsic::vp_reduce_smax:
      return Intrinsic::smax;
    case Intrinsic::vp_reduce_fmin:
      return Intrinsic::minnum;
    case Intrinsic::vp_reduce_fmax:
      return Intrinsic::maxnum;
    case Intrinsic::vp_reduce_fminimum:
      return Intrinsic::minimum;
    case Intrinsic::vp_reduce_fmaximum:
      return Intrinsic::maximum;
    }
  };
  IRBuilder<>::FastMathFlagGuard FMFGuard(irBuilder);
  irBuilder.setFastMathFlags(FMFSource.get({}));

  Constant *NeutralElement = getRippleNeutralReductionElement(
      reductionId, vector->getType()->getScalarType(),
      irBuilder.getFastMathFlags());

  ElementCount VectorCount =
      cast<VectorType>(vector->getType())->getElementCount();
  // Partial reductions
  Constant *NeutralVector =
      ConstantVector::getSplat(VectorCount, NeutralElement);
  // Create a select to introduce a maskable instruction
  Constant *TrueMask = ConstantVector::getSplat(
      VectorCount, ConstantInt::getTrue(vector->getContext()));
  TensorShape CurrentVectorShape = vectorShape;
  SelectInst *SelectedValue = irBuilder.Insert(SelectInst::Create(
      TrueMask, vector, NeutralVector,
      Twine(vector->getName()) + ".ripple.reduction.partial.masking"));
  setRippleShape(SelectedValue, CurrentVectorShape);
  SelectToMaskWhenIfConvert.insert(SelectedValue);

  // Full reduction, use well supported llvm reductions instead
  if (vectorShape.reducedToScalarBy(reductionDimensions)) {
    if (auto IntrinsicId =
            VPIntrinsic::getFunctionalIntrinsicIDForVP(reductionId)) {
      SmallVector<Value *, 2> Args;
      if (*IntrinsicId == Intrinsic::vector_reduce_fadd ||
          *IntrinsicId == Intrinsic::vector_reduce_fmul) {
        Args.push_back(NeutralElement);
      }
      Args.push_back(SelectedValue);
      CallInst *ReductionCall = irBuilder.CreateIntrinsic(
          vector->getType()->getScalarType(), *IntrinsicId, Args, {},
          Twine(vector->getName()) + ".ripple.reduction");
      setRippleShape(ReductionCall, ScalarShape);
      return ReductionCall;
    }
  }

  Value *CurrentVectorValue = SelectedValue;

  std::vector<int> ShuffleMask;
  ShuffleMask.reserve(CurrentVectorShape.flatShape());
  for (auto RedDim : reductionDimensions.set_bits()) {
    NeutralVector = ConstantVector::getSplat(
        ElementCount::getFixed(CurrentVectorShape.flatShape()), NeutralElement);
    // Each non-empty inner dimension that is not being reduced adds a
    // multiple to the number of elements we start shuffling to reduce:
    // normally we reduce by shuffling at distance 1, 2, ..., 2^n
    unsigned ReductionIterCount = Log2_64_Ceil(CurrentVectorShape[RedDim]);
    for (unsigned RedIter = 0; RedIter < ReductionIterCount; ++RedIter) {
      ShuffleMask.clear();
      // Build the shuffle indices vector
      SmallVector<size_t, 0> RotateIndices(tensorRank());
      CurrentVectorShape.foreachIndex([&](auto Indices) {
        auto ReduceWith = Indices[RedDim] + (1 << RedIter);
        if (ReduceWith >= CurrentVectorShape[RedDim])
          // use the neutral element at the same offset
          ShuffleMask.push_back(CurrentVectorShape.getOffsetAt(Indices) +
                                CurrentVectorShape.flatShape());
        else {
          std::copy(Indices.begin(), Indices.end(), RotateIndices.begin());
          RotateIndices[RedDim] = ReduceWith;
          ShuffleMask.push_back(CurrentVectorShape.getOffsetAt(RotateIndices));
        }
      });
      Value *Shuff = irBuilder.CreateShuffleVector(
          CurrentVectorValue, NeutralVector, ShuffleMask,
          Twine(vector->getName()) + ".ripple.reducelog2.shuffle");
      setRippleShape(Shuff, CurrentVectorShape);
      auto BinRedOp = reductionBinOp(reductionId);
      if (BinRedOp) {
        CurrentVectorValue = irBuilder.CreateBinOp(
            *BinRedOp, CurrentVectorValue, Shuff,
            Twine(vector->getName()) + ".ripple.reducelog2.operator");
        setRippleShape(CurrentVectorValue, CurrentVectorShape);
      } else {
        auto CmpPred = cmpReduction(reductionId);
        assert(CmpPred);
        CurrentVectorValue = irBuilder.CreateIntrinsic(
            CurrentVectorValue->getType(), *CmpPred,
            {CurrentVectorValue, Shuff}, {},
            Twine(vector->getName()) + ".ripple.reducelog2.compare.op");
        setRippleShape(CurrentVectorValue, CurrentVectorShape);
      }
    }
    // Now extract the reduced value that is at index 0 (or anywhere really) in
    // the reduced dimension
    ShuffleMask.clear();
    CurrentVectorShape.foreachIndex([&](auto Indices) {
      if (Indices[RedDim] == 0)
        ShuffleMask.push_back(CurrentVectorShape.getOffsetAt(Indices));
    });
    // Get the new shape
    BitVector RedCurrentDim(reductionDimensions.size());
    RedCurrentDim.set(RedDim);
    CurrentVectorShape.reduceDimensions(RedCurrentDim);

    if (CurrentVectorShape.isScalar()) {
      CurrentVectorValue = irBuilder.CreateExtractElement(
          CurrentVectorValue, static_cast<uint64_t>(0), "ripple.red.extract");
    } else {
      CurrentVectorValue = irBuilder.CreateShuffleVector(
          CurrentVectorValue, ShuffleMask,
          Twine(vector->getName()) + ".ripple.reducelog2.dim.final");
    }
    setRippleShape(CurrentVectorValue, CurrentVectorShape);
  }

  assert(CurrentVectorShape.reduceDimensions(reductionDimensions) == false &&
         "The shape should already be a fully reduced tensor");
  return CurrentVectorValue;
}

Value *Ripple::buildLinearSeriesSlope(const LinearSeries *LS) {
  assert(LS->getSlopeShape().isVector());
  const TensorShape &LinearSeriesShape = LS->getShape();

  if (LS->hasZeroSlopes())
    return ConstantAggregateZero::get(
        VectorType::get(LS->getSlope(0)->getType(),
                        LinearSeriesShape.flatShape(), /*Scalable*/ false));

  const TensorShape &SlopeShape = LS->getSlopeShape();
  // Slope should be an integer type by construction
  Value *SlopeValue = nullptr;
  for (unsigned RankIdx = 0; RankIdx < SlopeShape.rank(); ++RankIdx) {
    auto size = SlopeShape[RankIdx];
    if (size > 1) {
      TensorShape SlopeShape =
          TensorShape(LinearSeriesShape.rank(), RankIdx, size);
      // Series [0, 1, ..., size - 1]
      Constant *BaseSeries = LinearSeries::constructLinearSeriesVector(
          cast<IntegerType>(LS->getSlope(RankIdx)->getType()), size);

      // Slope as vector
      Value *SlopeBcast = irBuilder.CreateVectorSplat(
          size, LS->getSlope(RankIdx),
          Twine(LS->getBase()->getName()) + ".ripple.LS.dim.slope.bcast");
      // A Splat is an insert followed by a shuffle
      setRippleShape(SlopeBcast, SlopeShape);
      if (ShuffleVectorInst *ShuffleInst =
              dyn_cast<ShuffleVectorInst>(SlopeBcast))
        setRippleShape(ShuffleInst->getOperand(0), SlopeShape);

      // Slope Series [0, slope, ..., slope * (size - 1)]
      Value *SlopeSeries = irBuilder.CreateBinOp(
          Instruction::Mul, BaseSeries, SlopeBcast,
          Twine(LS->getBase()->getName()) + ".ripple.LS.dim.slope");
      setRippleShape(SlopeSeries, SlopeShape);

      auto BcastedSeries =
          tensorBcast(SlopeSeries, SlopeShape, LinearSeriesShape);
      // Checked during shape propagation
      if (!BcastedSeries)
        report_fatal_error("Broadcast failure during LinearSeries codegen");
      setRippleShape(*BcastedSeries, LinearSeriesShape);

      if (SlopeValue) {
        SlopeValue = irBuilder.CreateBinOp(
            Instruction::Add, SlopeValue, *BcastedSeries,
            Twine(LS->getBase()->getName()) + ".ripple.LS.slope.combine");
        setRippleShape(SlopeValue, LinearSeriesShape);
      } else {
        SlopeValue = *BcastedSeries;
      }
    }
  }
  SlopeValue->setName(Twine(LS->getBase()->getName()) + ".ripple.LS.slope");
  assert(SlopeValue != nullptr);
  return SlopeValue;
}

Constant *LinearSeries::constructLinearSeriesVector(IntegerType *intTy,
                                                    uint64_t size) {
  std::vector<Constant *> cstVectorVals;
  for (uint64_t i = 0; i < size; ++i) {
    cstVectorVals.push_back(ConstantInt::get(
        intTy->getScalarType(), APInt(intTy->getBitWidth(), i, false)));
  }
  return ConstantVector::get(cstVectorVals);
}

IntegerType *LinearSeries::getSlopeTypeFor(const DataLayout &DL,
                                           Type *BaseType) {
  Type *BaseScalarType = BaseType->getScalarType();
  IntegerType *SlopeType = nullptr;
  if (IntegerType *IntTy = dyn_cast<IntegerType>(BaseScalarType))
    SlopeType = IntTy;
  else if (PointerType *PTy = dyn_cast<PointerType>(BaseScalarType)) {
    SlopeType =
        IntegerType::get(BaseType->getContext(),
                         DL.getPointerSizeInBits(PTy->getAddressSpace()));
  }
  return SlopeType;
}

bool Ripple::canConstructSplatSeries(Value *V, const TensorShape &FromShape,
                                     const TensorShape &ToShape) {
  return (isa<PointerType>(V->getType()->getScalarType()) ||
          isa<IntegerType>(V->getType()->getScalarType())) &&
         !FromShape.isBroadcastError(ToShape);
}

Ripple::ConstructedSeries
Ripple::getLinearSeriesFor(Constant *C, const TensorShape &FromShape,
                           const TensorShape &ToShape) {
  return getSplatSeries(C, FromShape, ToShape);
}

Ripple::ConstructedSeries
Ripple::getLinearSeriesFor(Argument *A, const TensorShape &FromShape,
                           const TensorShape &ToShape) {
  return getSplatSeries(A, FromShape, ToShape);
}

Ripple::ConstructedSeries Ripple::getSplatSeries(Value *V,
                                                 const TensorShape &FromShape,
                                                 const TensorShape &ToShape) {
  if (!canConstructSplatSeries(V, FromShape, ToShape))
    return {};

  // We can generate integer or pointer linear series (through GEP)
  Type *SlopeType =
      LinearSeries::getSlopeTypeFor(DL, V->getType()->getScalarType());

  if (!SlopeType)
    return {};

  assert(!(FromShape.isScalar() && V->getType()->isVectorTy()));

  const auto ConstantSlopeOf = [&](uint64_t val) {
    return ConstantInt::get(SlopeType, val,
                            /*signed*/ false);
  };

  // The slope shape is composed of the dimensions of ToShape - FromShape
  TensorShape SlopeShape = ToShape;
  SlopeShape.reduceDimensions(FromShape.nonEmptyDims());
  SmallVector<Value *, 4> NewSlopes;

  NewSlopes.append(ToShape.rank(), ConstantSlopeOf(0));
  LinearSeries *LS = new LinearSeries(V, FromShape, NewSlopes, SlopeShape);
  return {LS, CSState::ValidLinearSeries};
}

Ripple::ConstructedSeries Ripple::tryToPromoteLinearSeries(LinearSeries *LS) {
  Value *Base = LS->getBase();
  const TensorShape &ExpectedBaseShape = LS->getBaseShape();
  if (PHINode *Phi = dyn_cast<PHINode>(Base)) {
    LLVM_DEBUG(dbgs() << "Trying to promote " << *LS << "\n");
    // Process the incoming values that were not inserted yet
    for (unsigned IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues();
         ++IncomingIdx) {
      Value *IncomingValue = Phi->getIncomingValue(IncomingIdx);
      BasicBlock *IncomingBlock = Phi->getIncomingBlock(IncomingIdx);
      // Add if it's missing from our slopes
      if (cast<PHINode>(LS->getSlope(0))->getBasicBlockIndex(IncomingBlock) ==
          -1) {
        // The potentially missing incoming values are instructions
        LLVM_DEBUG(dbgs() << "Missing incoming from: "
                          << IncomingBlock->getName() << " value "
                          << *IncomingValue << " from " << *LS->getSlope(0)
                          << "\n");
        Instruction *IncomingInstruction = cast<Instruction>(IncomingValue);
        auto OperandCS = getCachedSeries(IncomingInstruction);
        if (OperandCS.isNotASeries()) {
          LLVM_DEBUG(dbgs() << "Incoming is not a series!\n");
          return {};
        }
        if (OperandCS.LS->getBaseShape() != ExpectedBaseShape) {
          LLVM_DEBUG(dbgs() << "Operand base shape differs: expected "
                            << ExpectedBaseShape << " and operand has "
                            << OperandCS.LS->getBaseShape() << "\n");
          return {};
        }
        LLVM_DEBUG(dbgs() << "Valid or potential linear series: "
                          << *OperandCS.LS << "\n");
        for (unsigned SlopeIdx = 0; SlopeIdx < LS->getSlopeShape().rank();
             ++SlopeIdx) {
          PHINode *SlopePhi = cast<PHINode>(LS->getSlope(SlopeIdx));
          assert(SlopePhi->getBasicBlockIndex(IncomingBlock) == -1);
          SlopePhi->addIncoming(OperandCS.LS->getSlope(SlopeIdx),
                                IncomingBlock);
        }
      }
    }
    // We must check that the roots of values flowing into the PHI are valid
    // linear series before promoting
    if (!hasValidLinearSeriesRoots(LS)) {
      LLVM_DEBUG(
          dbgs() << *LS
                 << " has non linear series root and cannot be promoted!\n");
      return {};
    }
    LLVM_DEBUG(dbgs() << *LS << " has been promoted!\n");
    for ([[maybe_unused]] auto &Slope : LS->slopes()) {
      assert(cast<PHINode>(&*Slope)->isComplete());
    }
    // All incoming values have a matching base, we can promote the PHI
    LsCache.Potential.erase(Phi);
    LsCache.Valid.insert({Phi, LS});
    return {LS, CSState::ValidLinearSeries};
  } else if (Instruction *I = dyn_cast<Instruction>(Base)) {
    for (auto &Operand : I->operands()) {
      if (Instruction *OperandInstr = dyn_cast<Instruction>(Operand)) {
        auto OperandCS = getCachedSeries(OperandInstr);
        if (OperandCS.LS->getBaseShape() != ExpectedBaseShape) {
          LLVM_DEBUG(dbgs() << "This base shape " << ExpectedBaseShape
                            << " differs from other base shape "
                            << OperandCS.LS->getBaseShape());
          return {};
        }
        if (!OperandCS.isValid())
          return {};
      }
    }
    LLVM_DEBUG(dbgs() << *LS << " has been promoted!\n");
    LsCache.Potential.erase(I);
    LsCache.Valid.insert({I, LS});
    return {LS, CSState::ValidLinearSeries};
  } else {
    // Only Instructions should become Potential Series
    report_fatal_error("A linear series base which is not an Instruction can "
                       "not be a potential series");
  }
}

Ripple::ConstructedSeries Ripple::getLinearSeriesFor(Instruction *I) {
  if (!I)
    return {};

  // TODO: Remove when llvm has entirely transitioned to record instead of
  // intrinsics for debug info!
  if (isa<DbgInfoIntrinsic>(I))
    return {};

  const TensorShape &ToShape = getRippleShape(I);

  auto isASlope = [&](Instruction *I) -> bool {
    return I && SlopeInstructions.contains(I);
  };

  IntegerType *SlopeType = LinearSeries::getSlopeTypeFor(DL, I->getType());
  // We can generate integer or pointer linear series (through GEP)
  if (!SlopeType)
    return {};

  // Use already constructed series if possible
  auto CachedCS = getCachedSeries(I);
  if (CachedCS.isValid())
    return CachedCS;

  if (CachedCS.hasPotential()) {
    assert(I == CachedCS.LS->getBase());
    if (auto PromotedCS = tryToPromoteLinearSeries(CachedCS.LS))
      return PromotedCS;
    else
      return CachedCS;
  }

  if (isASlope(I)) {
    LLVM_DEBUG(dbgs() << "We avoid creating LS for slope: " << *I << "\n");
    return {};
  }

  const auto ConstantSlopeOf = [SlopeType](uint64_t val) {
    return ConstantInt::get(SlopeType, val, /*signed*/ false);
  };

  auto getOperandSeries = [&](Value *Operand) -> ConstructedSeries {
    if (Constant *C = dyn_cast<Constant>(Operand)) {
      if (!C->getType()->isVectorTy())
        return getLinearSeriesFor(C, getRippleShape(C), ToShape);
    } else if (Argument *A = dyn_cast<Argument>(Operand)) {
      if (!A->getType()->isVectorTy())
        return getLinearSeriesFor(A, getRippleShape(A), ToShape);
    } else if (Instruction *I = dyn_cast<Instruction>(Operand))
      return getCachedSeries(I);
    return {};
  };

  // Get the linear series of the operands
  std::vector<ConstructedSeries> OperandSeries;
  for (unsigned OperandIdx = 0; OperandIdx < I->getNumOperands();
       ++OperandIdx) {
    Value *Operand = I->getOperand(OperandIdx);
    OperandSeries.push_back(getOperandSeries(Operand));
  }

  SmallVector<Value *, 3> NewSlopes;

  auto processRippleBlockIndex =
      [&](IntrinsicInst *RippleIndexInst) -> ConstructedSeries {
    // For Ripple Index, the base is zero and the shape is carried by the slope
    for (unsigned i = 0; i < ToShape.rank(); ++i)
      NewSlopes.push_back(ToShape[i] > 1 ? ConstantSlopeOf(1)
                                         : ConstantSlopeOf(0));
    LinearSeries *LS =
        new LinearSeries(ConstantSlopeOf(0), ScalarShape, NewSlopes, ToShape);
    return {LS, CSState::ValidLinearSeries};
  };

  auto processRippleGetSize =
      [&](IntrinsicInst *RippleGetSizeInst) -> ConstructedSeries {
    for (unsigned i = 0; i < ToShape.rank(); ++i)
      NewSlopes.push_back(ConstantSlopeOf(0));
    LinearSeries *LS = new LinearSeries(
        ConstantSlopeOf(getRippleGetSizeValue(RippleGetSizeInst)), ScalarShape,
        NewSlopes, ToShape);
    return {LS, CSState::ValidLinearSeries};
  };

  auto processBinOp = [&](BinaryOperator *BinOp) -> ConstructedSeries {
    auto &LhsSeries = OperandSeries[0];
    auto &RhsSeries = OperandSeries[1];
    if (LhsSeries.isNotASeries() || RhsSeries.isNotASeries())
      return {};

    CSState NewState =
        combineStatesBinaryOp(LhsSeries.getState(), RhsSeries.getState());

    const TensorShape &LhsBaseShape = LhsSeries.LS->getBaseShape();
    const TensorShape &LhsSlopeShape = LhsSeries.LS->getSlopeShape();
    const TensorShape &RhsBaseShape = RhsSeries.LS->getBaseShape();
    const TensorShape &RhsSlopeShape = RhsSeries.LS->getSlopeShape();

    // TODO: there is an optimization opportunity when the bases have different
    // shapes: we can broadcast the base if the corresponding broadcast slope
    // dimension is empty (1) or if the slope is zero and the dimension size are
    // equal. E.g. LHS[BaseShape([1, 15]) SlopeShape[5, 1] Slope(0, 0)]
    // RHS[BaseShape([5, 15]), SlopeShape([1, 1])] the LHS base can be
    // broadcasted to match RHS base shape and continue propagating a valid
    // series.
    const TensorShape &NewBaseShape = LhsBaseShape;

    TensorShape NewSlopeShape = ToShape;
    NewSlopeShape.reduceDimensions(NewBaseShape.nonEmptyDims());

    // We have to expand when:
    // 1) the bases have different shapes
    // 2) we cannot combine the slopes w/ a broadcast
    // 3) if the rhs base dimension overlaps with lhs slope dimensions
    // 4) if the lhs base dimension overlaps with rhs slope dimensions
    // Cannot combine the bases; we have to expand
    if (LhsBaseShape != RhsBaseShape ||
        RhsBaseShape.bothNonEmptyDims(LhsSlopeShape).any() ||
        LhsBaseShape.bothNonEmptyDims(RhsSlopeShape).any())
      return {};

    std::function<Value *(unsigned)> LeftSlopeOp, RightSlopeOp;
    std::function<Value *(Value *, Value *)> IRBFun;

    // Handle common signed integer truncation pattern (unsigned -> signed):
    // R2 = shl R1, P; where P = (sizeof_bit(R1) - sizeof_bit(trunc int type))
    // R3 = ashr exact R2, P
    ConstantInt *SrAmount;
    ConstantInt *SlAmount;
    Value *SlVal;
    // AShr(Shl(Any, SlAmount), SrAmount)
    if (match(BinOp, m_AShr(m_Shl(m_Value(SlVal), m_ConstantInt(SlAmount)),
                            m_ConstantInt(SrAmount))) &&
        cast<AShrOperator>(BinOp)->isExact() && SrAmount == SlAmount) {
      // We allow this transformation only if the following ops undefine
      // behavior on overflow
      if (!all_of(BinOp->users(), [](User *U) {
            if (auto *OverflowBinop = dyn_cast<OverflowingBinaryOperator>(U))
              return OverflowBinop->hasNoSignedWrap() ||
                     OverflowBinop->hasNoUnsignedWrap();
            return false;
          }))
        return {};
      // The slopes are the one from the LHS of LhsShift; don't
      // compute them using right shift or we may truncate the slopes!
      auto SlOperandOpSeries = getOperandSeries(SlVal);
      if (SlOperandOpSeries.isValid() || SlOperandOpSeries.hasPotential()) {
        IRBFun = [&](Value *Lhs, Value *) { return Lhs; };
        LeftSlopeOp = [SlOperandOpSeries](unsigned idx) {
          return SlOperandOpSeries.LS->getSlope(idx);
        };
        RightSlopeOp = [](unsigned idx) { /* Not used */ return nullptr; };
      } else {
        return {};
      }
    } else if (ShlOperator *ShlOp = dyn_cast<ShlOperator>(BinOp)) {
      IRBFun = [&, ShlOp](Value *Lhs, Value *Rhs) {
        Value *NewShl = irBuilder.CreateShl(Lhs, Rhs, "ripple.LS.slope.shl",
                                            ShlOp->hasNoUnsignedWrap(),
                                            ShlOp->hasNoSignedWrap());
        return NewShl;
      };

      if (RhsSeries.LS->isScalarOrSplat()) {
        // (ax + b) << s => (a << s)x + (b << s); (<< s) <=> (* 2^x)
        LeftSlopeOp = [&](unsigned idx) { return LhsSeries.LS->getSlope(idx); };
        RightSlopeOp = [&](unsigned idx) { return RhsSeries.LS->getBase(); };
      } else {
        return {};
      }
    } else if (MulOperator *MulOp = dyn_cast<MulOperator>(BinOp)) {
      IRBFun = [&, MulOp](Value *Lhs, Value *Rhs) {
        return irBuilder.CreateMul(Lhs, Rhs, "ripple.LS.slope.mul",
                                   MulOp->hasNoUnsignedWrap(),
                                   MulOp->hasNoSignedWrap());
      };

      // We only allow multiplication by a scalar
      if (LhsSeries.LS->isScalarOrSplat()) {
        // s * (ax + b) => (s * a)x + (s * b)
        LeftSlopeOp = [&](unsigned idx) { return LhsSeries.LS->getBase(); };
        RightSlopeOp = [&](unsigned idx) {
          return RhsSeries.LS->getSlope(idx);
        };
      } else if (RhsSeries.LS->isScalarOrSplat()) {
        // (ax + b) * s => (a * s)x + (b * s)
        LeftSlopeOp = [&](unsigned idx) { return LhsSeries.LS->getSlope(idx); };
        RightSlopeOp = [&](unsigned idx) { return RhsSeries.LS->getBase(); };
      } else {
        return {};
      }
    } else if (SubOperator *SubOp = dyn_cast<SubOperator>(BinOp)) {
      // (ax + b) - (cx + d) => (a - c)x + (b - d)
      IRBFun = [&, SubOp](Value *Lhs, Value *Rhs) {
        return irBuilder.CreateSub(Lhs, Rhs, "ripple.LS.slope.sub",
                                   SubOp->hasNoUnsignedWrap(),
                                   SubOp->hasNoSignedWrap());
      };

      LeftSlopeOp = [&](unsigned idx) { return LhsSeries.LS->getSlope(idx); };
      RightSlopeOp = [&](unsigned idx) { return RhsSeries.LS->getSlope(idx); };
    } else if (AddOperator *AddOp = dyn_cast<AddOperator>(BinOp)) {
      IRBFun = [&, AddOp](Value *Lhs, Value *Rhs) {
        return irBuilder.CreateAdd(Lhs, Rhs, "ripple.LS.slope.add",
                                   AddOp->hasNoUnsignedWrap(),
                                   AddOp->hasNoSignedWrap());
      };

      // (ax + b) + (cx + d) => (a + c)x + (b + d)
      LeftSlopeOp = [&](unsigned idx) { return LhsSeries.LS->getSlope(idx); };
      RightSlopeOp = [&](unsigned idx) { return RhsSeries.LS->getSlope(idx); };
    } else {
      // Cannot pass through a LinearSeries
      return {};
    }

    assert(ToShape.rank() == NewSlopeShape.rank());
    // Construct the new slopes
    for (unsigned i = 0; i < ToShape.rank(); ++i) {
      Value *NewSlope = IRBFun(LeftSlopeOp(i), RightSlopeOp(i));
      NewSlopes.push_back(NewSlope);
      SlopeInstructions.insert(dyn_cast<Instruction>(NewSlope));
      setRippleShape(NewSlope, ScalarShape);
    }
    auto *LS = new LinearSeries(BinOp, NewBaseShape, NewSlopes, NewSlopeShape);
    return {LS, NewState};
  };

  auto processCast = [&](CastInst *CastI) -> ConstructedSeries {
    auto &CastOpSeries = OperandSeries[0];
    if (CastOpSeries.isNotASeries())
      return {};
    auto &SlopeShape = CastOpSeries.LS->getSlopeShape();
    for (unsigned i = 0; i < SlopeShape.rank(); ++i) {
      Value *CastOp =
          irBuilder.CreateCast(CastI->getOpcode(), CastOpSeries.LS->getSlope(i),
                               CastI->getType(), "ripple.LS.slope.cast");
      setRippleShape(CastOp, ScalarShape);
      SlopeInstructions.insert(dyn_cast<Instruction>(CastOp));
      NewSlopes.push_back(CastOp);
    }
    auto *LS = new LinearSeries(CastI, CastOpSeries.LS->getBaseShape(),
                                NewSlopes, SlopeShape);
    return {LS, CastOpSeries.getState()};
  };

  auto processUnaryOp = [&](UnaryOperator *UnOp) -> ConstructedSeries {
    auto &InSeries = OperandSeries[0];
    if (InSeries.isNotASeries())
      return {};

    // Cast first because it may affect the slope type
    if (CastInst *CastI = dyn_cast<CastInst>(UnOp))
      return processCast(CastI);

    // Scalar bypass
    if (InSeries.LS->isScalar()) {
      auto *LS =
          new LinearSeries(UnOp, InSeries.LS->getBaseShape(),
                           InSeries.LS->slopes(), InSeries.LS->getSlopeShape());
      return {LS, InSeries.getState()};
    }

    return {};
  };

  auto processGEP = [&](GetElementPtrInst *GEP) -> ConstructedSeries {
    auto &PointerSeries = OperandSeries[0];
    if (PointerSeries.isNotASeries())
      return {};
    const TensorShape &NewBaseShape = PointerSeries.LS->getBaseShape();
    CSState NewState = PointerSeries.getState();
    for (unsigned Idx = 0; Idx < GEP->getNumIndices(); ++Idx) {
      auto &IdxSeries =
          OperandSeries[Idx + (GEP->idx_begin() - GEP->op_begin())];
      if (IdxSeries.isNotASeries() ||
          IdxSeries.LS->getBaseShape() != NewBaseShape)
        return {};
      NewState = combineStatesBinaryOp(NewState, IdxSeries.getState());
    }
    TensorShape NewSlopeShape = ToShape;
    NewSlopeShape.reduceDimensions(NewBaseShape.nonEmptyDims());

    // Start w/ the pointer slope
    for (unsigned i = 0; i < NewSlopeShape.rank(); ++i) {
      Value *PtrSlope = PointerSeries.LS->getSlope(i);
      assert(PtrSlope->getType() == SlopeType);
      NewSlopes.push_back(PtrSlope);
    }
    SmallVector<Value *, 4> IndicesProcessed;
    for (unsigned Idx = 0; Idx < GEP->getNumIndices(); ++Idx) {
      LinearSeries *IdxSeries =
          OperandSeries[Idx + (GEP->idx_begin() - GEP->op_begin())].LS;
      Value *GEPIndexVal = *(GEP->idx_begin() + Idx);
      // Get the type we are indexing
      IndicesProcessed.push_back(GEPIndexVal);
      Type *IndexedType = GetElementPtrInst::getIndexedType(
          GEP->getSourceElementType(), IndicesProcessed);
      LLVM_DEBUG(dbgs() << "\tProcessing GEP index " << *GEPIndexVal
                        << " affecting type " << *IndexedType << "\n");
      // This should not happen by LLVM IR construction, but better guard
      // against it
      bool IndexingStructField =
          Idx > 0 && GetElementPtrInst::getIndexedType(
                         GEP->getSourceElementType(),
                         ArrayRef(IndicesProcessed).drop_back())
                         ->isStructTy();
      if (IndexingStructField && IdxSeries->hasSlope())
        llvm_unreachable(
            "Ripple vector access to a structure field is undefined");
      // For pointers, the base is GEP() and the integer slope indexes a bytes
      // array.
      uint64_t IndexByteSize = DL.getTypeAllocSize(IndexedType);
      ConstantInt *IndexMultiple =
          ConstantInt::get(SlopeType, IndexByteSize, /*signed*/ false);
      for (unsigned SlopeIdx = 0; SlopeIdx < NewSlopeShape.rank(); ++SlopeIdx) {
        Value *IndexSlope = IdxSeries->getSlope(SlopeIdx);
        if (IndexSlope->getType() != SlopeType) {
          IndexSlope = irBuilder.CreateIntCast(IndexSlope, SlopeType,
                                               /*signed*/ false,
                                               "ripple.slope.gep.typefix");
          setRippleShape(IndexSlope, ScalarShape);
          SlopeInstructions.insert(dyn_cast<Instruction>(IndexSlope));
        }
        Value *SlopeInBytes = irBuilder.CreateMul(IndexSlope, IndexMultiple,
                                                  "ripple.slope.gep.inbytes");
        setRippleShape(SlopeInBytes, ScalarShape);
        SlopeInstructions.insert(dyn_cast<Instruction>(SlopeInBytes));
        NewSlopes[SlopeIdx] = irBuilder.CreateAdd(
            NewSlopes[SlopeIdx], SlopeInBytes, "ripple.LS.slope.gep.index");
        setRippleShape(NewSlopes[SlopeIdx], ScalarShape);
        SlopeInstructions.insert(dyn_cast<Instruction>(NewSlopes[SlopeIdx]));
      }
    }
    auto *LS = new LinearSeries(GEP, NewBaseShape, NewSlopes, NewSlopeShape);

    return {LS, NewState};
  };

  auto processPHINode = [&](PHINode *PHI) -> ConstructedSeries {
    LLVM_DEBUG(dbgs() << "Processing PHI for LS: " << *PHI << "\n");

    CSState NewState = CSState::ValidLinearSeries;
    // Get the base shape from any of the operands
    TensorShape NewBaseShape;
    assert(PHI->getNumIncomingValues() > 0);
    for (unsigned IncomingIdx = 0; IncomingIdx < PHI->getNumIncomingValues();
         ++IncomingIdx) {
      auto &OperandLS = OperandSeries[IncomingIdx];
      if (!OperandLS.isNotASeries()) {
        NewBaseShape = OperandLS.LS->getBaseShape();
      }
    }

    for (unsigned IncomingIdx = 0; IncomingIdx < PHI->getNumIncomingValues();
         ++IncomingIdx) {
      auto &OperandLS = OperandSeries[IncomingIdx];
      // When an operand is not a series, still try to potentially build the
      // PHI later
      if (OperandLS.isNotASeries()) {
        NewState =
            combineStatesBinaryOp(NewState, CSState::PotentialLinearSeries);
        continue;
      }
      // Valid keeps the state, any potential will demote
      NewState = combineStatesBinaryOp(NewState, OperandLS.getState());
      if (OperandLS.LS->getBaseShape() != NewBaseShape)
        return {};
    }
    if (NewState == CSState::NotASeries)
      return {};
    // We are creating a PHI for each Slope
    for (unsigned SlopeIdx = 0, EndIdx = ToShape.rank(); SlopeIdx < EndIdx;
         ++SlopeIdx) {
      PHINode *SlopePhi =
          irBuilder.CreatePHI(SlopeType, PHI->getNumIncomingValues(),
                              Twine(PHI->getName()) + ".ripple.slope.phi" +
                                  std::to_string(SlopeIdx));
      NewSlopes.push_back(SlopePhi);
      setRippleShape(SlopePhi, ScalarShape);
      SlopeInstructions.insert(SlopePhi);
      // Pre-populate for known LS
      for (unsigned IncomingIdx = 0; IncomingIdx < PHI->getNumIncomingValues();
           ++IncomingIdx) {
        auto &OperandLS = OperandSeries[IncomingIdx];
        // We can pre-populate the potential and valid LS
        if (!OperandLS.isNotASeries()) {
          SlopePhi->addIncoming(OperandLS.LS->getSlope(SlopeIdx),
                                PHI->getIncomingBlock(IncomingIdx));
        }
      }
    }

    // When the PHI is at the merge point of a vector branch, it will be
    // transformed into a series of select instruction. Don't create a linear
    // series when the mask cannot be applied to the base.
    if (auto *ImmDom = domTree.getNode(PHI->getParent())->getIDom())
      // PHIs always have an immediate dominator!
      if (Instruction *Terminator = ImmDom->getBlock()->getTerminator()) {
        auto &MaskShape = getRippleShape(Terminator);
        if (NewBaseShape.requiredSplat(MaskShape).any())
          return {};
      }

    TensorShape NewSlopeShape = ToShape;
    NewSlopeShape.reduceDimensions(NewBaseShape.nonEmptyDims());
    auto *LS = new LinearSeries(PHI, NewBaseShape, NewSlopes, NewSlopeShape);
    return {LS, NewState};
  };

  auto processRippleBroadcast =
      [&](IntrinsicInst *BcastOp) -> ConstructedSeries {
    // A broadcast can always be representat as a LS!

    auto &InSeries = OperandSeries[2];

    Value *BcastedVal = BcastOp->getOperand(2);

    // Constants are broadcasted to a splat-series, that's what we want!
    if (!InSeries.isNotASeries() && isa<Constant>(BcastedVal))
      return InSeries;

    // Let Arguments be re-processed (for specialization)
    if (!InSeries.isNotASeries() && isa<Argument>(BcastedVal))
      InSeries = {};

    if (!InSeries.isNotASeries())
      for (unsigned i = 0; i < ToShape.rank(); ++i)
        NewSlopes.push_back(InSeries.LS->getSlope(i));
    else
      for (unsigned i = 0; i < ToShape.rank(); ++i)
        NewSlopes.push_back(ConstantSlopeOf(0));

    Value *NewBase =
        InSeries.isNotASeries() ? BcastedVal : InSeries.LS->getBase();
    const TensorShape &NewBaseShape = InSeries.isNotASeries()
                                          ? getRippleShape(NewBase)
                                          : InSeries.LS->getBaseShape();

    auto BroadcastDimSet = InSeries.isNotASeries()
                               ? NewBaseShape.requiredSplat(ToShape)
                               : InSeries.LS->getShape().requiredSplat(ToShape);

    TensorShape NewSlopesShape = ToShape;
    NewSlopesShape.keepDimensions(BroadcastDimSet);
    LinearSeries *LS =
        new LinearSeries(NewBase, NewBaseShape, NewSlopes, NewSlopesShape);

    CSState NewState = InSeries.isNotASeries() ? CSState::ValidLinearSeries
                                               : InSeries.getState();
    return {LS, NewState};
  };

  auto IP = irBuilder.saveIP();
  irBuilder.SetInsertPoint(I);

  ConstructedSeries CS;
  if (IntrinsicInst *RippleIndexInst = intrinsicWithId(
          dyn_cast<Instruction>(I), {Intrinsic::ripple_block_index})) {
    CS = processRippleBlockIndex(RippleIndexInst);
  } else if (IntrinsicInst *RippleGetSizeInst = intrinsicWithId(
                 dyn_cast<Instruction>(I), {Intrinsic::ripple_block_getsize})) {
    CS = processRippleGetSize(RippleGetSizeInst);

  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    CS = processGEP(GEP);

  } else if (PHINode *PHI = dyn_cast<PHINode>(I)) {
    CS = processPHINode(PHI);

  } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(I)) {
    CS = processUnaryOp(UnOp);

  } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(I)) {
    CS = processBinOp(BinOp);

  } else if (IntrinsicInst *RippleBroadcast = rippleBroadcastIntrinsic(I)) {
    CS = processRippleBroadcast(RippleBroadcast);
  }

  // We can always restart a new series with a vector base since the type is
  // integral
  if (CS.isNotASeries()) {
    NewSlopes.clear();
    for (unsigned i = 0; i < ToShape.rank(); ++i)
      NewSlopes.push_back(ConstantSlopeOf(0));
    auto *LS = new LinearSeries(I, ToShape, NewSlopes, ScalarShape);
    CS = {LS, CSState::ValidLinearSeries};
  }

  irBuilder.restoreIP(IP);

  if (!CS.isNotASeries()) {
    assert(!LsCache.Valid.contains(I) && !LsCache.Potential.contains(I) &&
           "Inserting a value already in the cache");
    switch (CS.getState()) {
    case CSState::ValidLinearSeries:
      LsCache.Valid.insert({I, CS.LS});
      CS.LS->Retain();
      break;
    case CSState::PotentialLinearSeries:
      LsCache.Potential.insert({I, CS.LS});
      CS.LS->Retain();
      break;
    default:
      llvm_unreachable("State should either be Valid or Potential");
    }
  }
  return CS;
}

Ripple::CSState Ripple::combineStatesBinaryOp(CSState S1, CSState S2) {
  switch (S1) {
  case CSState::NotASeries:
  case CSState::PotentialLinearSeries:
    if (S2 == CSState::NotASeries)
      return CSState::NotASeries;
    else
      return S1;
  default:
    return S2;
  }
}

Ripple::ConstructedSeries Ripple::getCachedSeries(const Instruction *I) const {
  if (!I)
    return {};
  LinearSeries *LS = nullptr;
  LS = LsCache.Valid.lookup(I);
  if (LS)
    return {LS, CSState::ValidLinearSeries};
  LS = LsCache.Potential.lookup(I);
  if (LS)
    return {LS, CSState::PotentialLinearSeries};
  return {};
}

void Ripple::simplifySlopes() {
  DenseSet<Instruction *> InstructionsToSimplify;
  for (auto &[_, Ls] : LsCache.Valid) {
    const TensorShape &SlopeShape = Ls->getSlopeShape();
    for (unsigned SlopeIdx = 0; SlopeIdx < SlopeShape.rank(); SlopeIdx++) {
      if (SlopeShape[SlopeIdx] < 2) {
        Value *CurrentSlope = Ls->getSlope(SlopeIdx);
        Constant *Replacement = ConstantInt::get(CurrentSlope->getType(), 0);
        Ls->setSlope(SlopeIdx, Replacement);
        if (Instruction *SlopeInst = dyn_cast<Instruction>(CurrentSlope)) {
          LLVM_DEBUG(dbgs() << "Replacing slope " << *CurrentSlope << " by "
                            << *Replacement << "\n");
          InstructionsToSimplify.insert(SlopeInst);
        }
      }
    }
  }
  for (auto *I : InstructionsToSimplify) {
    Constant *Replacement = ConstantInt::get(I->getType(), 0);
    auto Iterator = I->getIterator();
    invalidateRippleDataFor(I);
    ReplaceInstWithValue(Iterator, Replacement);
  }
}


iterator_range<User::const_op_iterator>
Ripple::vectorizableOperands(const Instruction *I) {
  auto Begin = I->op_begin();
  auto End = I->op_end();
  if (const BranchInst *BrInst = dyn_cast<BranchInst>(I)) {
    // For branches, we skip the basic blocks
    if (BrInst->isConditional())
      End = std::next(Begin);
    else
      Begin = End;
  } else if (isa<SwitchInst>(I))
    // We are only interested in the switch's condition
    End = std::next(Begin);
  else if (rippleBlockIntrinsics(I))
    // No operand to vectorize for block intrinsics
    Begin = End;
  else if (rippleBroadcastIntrinsic(I)) {
    // The value being broadcasted is the third argument
    Begin = std::next(cast<CallInst>(I)->arg_begin(), 2);
    End = std::next(Begin);
  } else if (rippleSliceIntrinsic(I)) {
    // The value being sliced is the first argument
    Begin = cast<CallInst>(I)->arg_begin();
    End = std::next(Begin);
  } else if (rippleReduceIntrinsics(I)) {
    // The value being reduced is the second argument
    Begin = std::next(cast<CallInst>(I)->arg_begin());
    End = std::next(Begin);
  } else if (rippleShuffleIntrinsics(I)) {
    // The value being shuffled is the first argument
    Begin = cast<CallInst>(I)->arg_begin();
    End = std::next(Begin);
    // Pair Shuffle
    if (!cast<ConstantInt>(cast<CallInst>(I)->getArgOperand(2))->isZero())
      End = std::next(End);
  } else if (const CallInst *CallI = dyn_cast<CallInst>(I)) {
    // For other call instructions, skip the Function operand
    Begin = CallI->arg_begin();
    End = CallI->arg_end();
  }
  return make_range(Begin, End);
}

iterator_range<User::op_iterator> Ripple::vectorizableOperands(Instruction *I) {
  iterator_range<User::const_op_iterator> ConstRange =
      vectorizableOperands(const_cast<const Instruction *>(I));
  return make_range(const_cast<User::op_iterator>(ConstRange.begin()),
                    const_cast<User::op_iterator>(ConstRange.end()));
}

Expected<TensorShape>
Ripple::inferShapeFromOperands(const Instruction *I, bool AllowPartialPhi,
                               bool &RequiresWaitingForSpecialization) {
  RequiresWaitingForSpecialization = false;
  auto partialAliasCheck = [&](const Instruction *I,
                               MemoryLocation &MemLoc) -> Error {
    auto MayAliasAlloca =
        aliasesWithAlloca(MemLoc, PromotableAlloca, AliasResult::MayAlias)
            .second;
    auto PartialAliasAlloca =
        aliasesWithAlloca(MemLoc, PromotableAlloca, AliasResult::PartialAlias)
            .second;
    if (MayAliasAlloca || PartialAliasAlloca) {
      DiagnosticInfoRippleWithLoc DI(
          DS_Error, F, sanitizeRippleLocation(I),
          "Ripple cannot get a precise tensor shape shape for this "
          "instruction because it may alias with a local variable (alloca)."
          " You may avoid such situations by avoiding taking the address of "
          "local variables.");
      F.getContext().diagnose(DI);
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot get precise alloca shape  cannot be promoted");
    }
    return Error::success();
  };

  // Handle Ripple intrinsics calls that have special shape semantics
  if (const IntrinsicInst *II = rippleBlockIntrinsics(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::ripple_block_index: {
      // Get the BS shape
      auto *ShapeII = getBlockShapeIntrinsic(II->getArgOperandUse(0));
      // Checked by checkBlockShapeUsage
      assert(ShapeII);
      PEIdentifier ProcElem = *getConstantOperandValue(ShapeII, 0);
      auto Dimension = *getConstantOperandValue(II, 1);
      auto IndexShape = setShapeToTensorShape(ShapeII);

      // And keep only the dimension asked by the ID call
      BitVector DimToKeep(tensorRank());
      if (Dimension < PERank(ProcElem)) {
        auto TensorIdx = rippleToTensor({ProcElem, Dimension});
        DimToKeep.set(TensorIdx);
      }
      IndexShape.keepDimensions(DimToKeep);
      return IndexShape;
    }
    case Intrinsic::ripple_block_setshape:
    case Intrinsic::ripple_block_getsize:
      return ScalarShape;
    default:
      llvm_unreachable("Unimplemented case");
    }
  } else if (const IntrinsicInst *RippleBroadcast =
                 rippleBroadcastIntrinsic(I)) {
    Value *BroadcastingValue = RippleBroadcast->getArgOperand(2);
    return computeRippleShapeForBitsetIntrinsic(
        RippleBroadcast,
        getRippleShape(BroadcastingValue, /* ShapePropagation */ true));
  } else if (const IntrinsicInst *RippleRed = rippleReduceIntrinsics(I)) {
    Value *ReducedValue = RippleRed->getArgOperand(1);
    return computeRippleShapeForBitsetIntrinsic(
        RippleRed, getRippleShape(ReducedValue, /* ShapePropagation */ true));
  } else if (const IntrinsicInst *RippleSlice = rippleSliceIntrinsic(I)) {
    Value *Slicee = RippleSlice->getArgOperand(0);
    const TensorShape &SliceeShape =
        getRippleShape(Slicee, /* ShapePropagation */ true);

    if (SliceeShape.isScalar())
      return ScalarShape;

    TensorShape SlicedShape = SliceeShape;
    unsigned NArgs = RippleSlice->arg_size();
    BitVector toReduce(NArgs - 1);
    unsigned SetVectorId = lastVectorIdx(RippleSlice, SliceeShape,
                                         /*SpecialArg*/ 0, "slicing");
    if (SetVectorId == tensorRank()) // no non-trivial vector PE block found
      return SlicedShape;
    auto [VectorPE, _] = tensorToRipple(SetVectorId);
    for (unsigned Idx = 1; Idx < NArgs; ++Idx) {
      ConstantInt *SliceArg = cast<ConstantInt>(RippleSlice->getOperand(Idx));
      if (SliceArg->getSExtValue() >= 0) {
        if (Idx - 1 < PERank(VectorPE)) {
          auto TensorIdx = rippleToTensor({VectorPE, Idx - 1});
          toReduce.set(TensorIdx);
        }
      }
    }
    SlicedShape.reduceDimensions(toReduce);
    return SlicedShape;
  } else if (isa<DbgInfoIntrinsic>(I)) {
    return ShapeIgnoredByRipple;
  } else if (auto *Load = dyn_cast<LoadInst>(I)) {
    auto MemLoc = MemoryLocation::get(Load);

    // Check and report error to the user when the tensor shape cannot be
    // precisely inferred
    if (Error Err = partialAliasCheck(Load, MemLoc))
      return std::move(Err);

    // Checks that this
    if (aliasesWithPromotableAlloca(MemLoc)) {
      TensorShape LoadShape = ScalarShape;
      Error Err = Error::success();
      visitAllClobberingInstructions(
          cast<MemoryUse>(MemSSA.getMemoryAccess(Load)),
          [&](Instruction *Clobbering) -> bool {
            // We may be visiting the values early because of MemoryPHI so don't
            // crash because of getRippleShape
            if (AllowPartialPhi && InstructionRippleShapes.find(Clobbering) ==
                                       InstructionRippleShapes.end())
              return true;
            const TensorShape &IShape =
                getRippleShape(Clobbering, /* ShapePropagation */ true);

            auto NewShape = combineShapeBcastWithErrorReporting(
                LoadShape, IShape, "Ripple failed to broadcast the instruction",
                sanitizeRippleLocation(Load), "with the shape coming from  ",
                sanitizeRippleLocation(Clobbering));
            if (!NewShape) {
              // This is needed to check success before overriding
              if (Err)
                llvm_unreachable("Expected Success");
              Err = NewShape.takeError();
              return false;
            }
            std::swap(*NewShape, LoadShape);
            return true;
          });

      if (Err)
        return std::move(Err);
      LLVM_DEBUG(dbgs() << "Load shape final " << LoadShape << "\n");
      return LoadShape;
    }
  } else if (auto *Store = dyn_cast<StoreInst>(I)) {
    auto MemLoc = MemoryLocation::get(Store);

    // Check and report error to the user when the tensor shape cannot be
    // precisely inferred
    if (Error Err = partialAliasCheck(Store, MemLoc))
      return std::move(Err);

    if (aliasesWithPromotableAlloca(MemLoc)) {
      TensorShape StoreShape =
          getRippleShape(Store->getValueOperand(), /*ShapePropagation*/ true);
      Error Err = Error::success();
      visitAllInstructionsBeingClobberedBy(
          cast<MemoryDef>(MemSSA.getMemoryAccess(Store)),
          [&](Instruction *Clobbered) -> bool {
            if (AllowPartialPhi && InstructionRippleShapes.find(Clobbered) ==
                                       InstructionRippleShapes.end())
              return true;
            const TensorShape &ClobberedShape =
                getRippleShape(Clobbered, /* ShapePropagation */ true);

            auto NewShape = combineShapeBcastWithErrorReporting(
                StoreShape, ClobberedShape,
                "Ripple failed to broadcast the instruction",
                sanitizeRippleLocation(Store), "with the shape coming from  ",
                sanitizeRippleLocation(Clobbered));
            if (!NewShape) {
              // This is needed to check success before overriding
              if (Err)
                llvm_unreachable("Expected Success");
              Err = NewShape.takeError();
              return false;
            }
            std::swap(*NewShape, StoreShape);
            return true;
          });

      if (Err)
        return std::move(Err);
      LLVM_DEBUG(dbgs() << "Store shape final " << StoreShape << "\n");
      return StoreShape;
    }
  }

  // We assume a broadcast semantics of instruction operands
  TensorShape InstructionShape = ScalarShape;
  bool IsPhiNode = isa<PHINode>(I);
  for (auto &Op : vectorizableOperands(I)) {
    // This is only useful when propagating the shapes initially, when PHI
    // nodes have non-processed input instruction shapes
    if (IsPhiNode && AllowPartialPhi && isa<Instruction>(Op) &&
        InstructionRippleShapes.find(cast<Instruction>(Op)) ==
            InstructionRippleShapes.end())
      continue;
    auto &OperandShape = getRippleShape(Op, /* ShapePropagation */ true);
    auto NewShape = combineShapeBcastWithErrorReporting(
        InstructionShape, OperandShape,
        "Ripple failed to broadcast the instruction", sanitizeRippleLocation(I),
        "with the shape coming from the operand " +
            std::to_string(Op.getOperandNo()),
        isa<Instruction>(Op) ? sanitizeRippleLocation(cast<Instruction>(Op))
                             : DebugLoc());
    RETURN_UNEXPECTED(NewShape);
    std::swap(InstructionShape, *NewShape);
  }
  return InstructionShape;
}

void LinearSeries::print(raw_ostream &O) const {
  O << "LinearSeries[";
  O << "\n  Shape[" << getShape() << "]";
  O << "\n  Base[" << *Base << "]";
  O << "\n  BaseShape[" << baseShape << "]";
  O << "\n  Slopes[";
  for (unsigned i = slopeShape.rank() - 1; i < slopeShape.rank(); --i) {
    if (i < slopeShape.rank() - 1)
      O << ", ";
    O << *SlopeValues[i];
  }
  O << "]";
  O << "\n  SlopeShape[" << slopeShape << "]\n]";
}

bool Ripple::hasValidLinearSeriesRoots(LinearSeries *LS) const {
  LLVM_DEBUG(dbgs() << "Checking Validity of " << *LS << "\n");
  SmallPtrSet<Instruction *, 32> AlreadyProcessed;
  std::queue<Instruction *> WorkList;
  // Constant and Argument linear series are valid, only check instructions
  if (Instruction *BaseInst = dyn_cast<Instruction>(LS->getBase())) {
    WorkList.push(BaseInst);
  }
  while (!WorkList.empty()) {
    Instruction *ToProcess = WorkList.front();
    WorkList.pop();
    if (AlreadyProcessed.contains(ToProcess))
      continue;
    AlreadyProcessed.insert(ToProcess);
    LLVM_DEBUG(dbgs() << "  Checking root " << *ToProcess << "\n");
    auto CS = getCachedSeries(ToProcess);
    if (CS.isNotASeries()) {
      LLVM_DEBUG(dbgs() << "Found non-series parent: " << *ToProcess << "\n");
      return false;
    }
    for (auto &Operand : ToProcess->operands()) {
      if (Instruction *OperandInst = dyn_cast<Instruction>(Operand)) {
        WorkList.push(OperandInst);
      }
    }
  }
  return true;
}

void Ripple::clearValidSerie(const Instruction *I) {
  if (!I)
    return;
  auto It = LsCache.Valid.find(I);
  if (It != LsCache.Valid.end()) {
    It->getSecond()->Release();
    LsCache.Valid.erase(It);
  }
}

void Ripple::clearPotentialSeries() {
  DenseSet<Instruction *> ToPoison;
  for (auto &[_, LS] : LsCache.Potential) {
    // Remove introduced slopes for unused LS
    for (auto &Slope : LS->slopes()) {
      if (Instruction *I = dyn_cast<Instruction>(&*Slope)) {
        invalidateRippleDataFor(I);
        ToPoison.insert(I);
      }
    }
    assert(LS->UseCount() == 1);
    LS->Release();
  }
  LsCache.Potential.clear();
  for (auto *I : ToPoison) {
    auto IBBit = I->getIterator();
    ReplaceInstWithValue(IBBit, PoisonValue::get(I->getType()));
  }
}

void Ripple::clearLinearSeriesCache() {
  clearPotentialSeries();
  for (auto &[_, LS] : LsCache.Valid) {
    assert(LS->UseCount() == 1);
    LS->Release();
  }
  LsCache.Valid.clear();
  LsCache.GeneratedSeries.clear();
}

Value *Ripple::instantiateLinearSeries(const LinearSeries *LS,
                                       bool UseLSCache) {
  Value *CachedValue = LsCache.GeneratedSeries.lookup(LS);
  if (UseLSCache && CachedValue)
    return CachedValue;
  LLVM_DEBUG(dbgs() << "Instantiating " << *LS << "\n");

  const TensorShape &LinearSeriesShape = LS->getShape();
  // Unused RepShape when assert is gone
  [[maybe_unused]] auto [BaseReplacement, RepShape] =
      replacementValueAndShape(LS->getBase());
  // Alloca have shape-shifting capabilities
  assert(*RepShape == LS->getBaseShape() ||
         (isa<AllocaInst>(BaseReplacement) && *RepShape >= LS->getBaseShape()));

  Value *Series = nullptr;
  if (LS->getBase()->getType()->getScalarType()->isPointerTy() &&
      LS->getBaseShape().isScalar() && !LS->getSlopeShape().isScalar()) {
    // Use the fact that GEP auto-splats scalar bases to keep them as scalar
    // (helps hardware scatter/gather needing a base ptr + offset vector)
    auto SlopeValue = buildLinearSeriesSlope(LS);
    Series =
        irBuilder.CreateGEP(irBuilder.getInt8Ty(), LS->getBase(), {SlopeValue});
  } else {
    auto BaseBcast =
        tensorBcast(BaseReplacement, LS->getBaseShape(), LinearSeriesShape);
    if (!BaseBcast)
      report_fatal_error("Broadcast failure during codegen");

    // Return the Broadcast of the base when the slope is 0 or absent
    if (LS->hasZeroSlopes()) {
      Series = *BaseBcast;
    } else {
      setRippleShape(*BaseBcast, LinearSeriesShape);
      // Create the Slope
      auto SlopeValue = buildLinearSeriesSlope(LS);

      if (LS->getBase()->getType()->getScalarType()->isPointerTy()) {
        // Use a GEP to compute the final vector of addresses
        Series = irBuilder.CreateGEP(irBuilder.getInt8Ty(), *BaseBcast,
                                     {SlopeValue});
      } else {
        LLVM_DEBUG(dbgs() << "Adding base " << **BaseBcast << "\n\tto slope "
                          << *SlopeValue << "\n");
        Series =
            irBuilder.CreateBinOp(Instruction::Add, *BaseBcast, SlopeValue);
      }
    }
  }
  Series->setName(Twine(LS->getBase()->getName()) + ".ripple.LS.instance");
  LLVM_DEBUG(dbgs() << "Instantiated Linear Series: " << *LS << " as "
                    << *Series << "\n");
  setRippleShape(Series, LinearSeriesShape);
  if (UseLSCache)
    LsCache.GeneratedSeries.insert({LS, Series});
  return Series;
}

Value *Ripple::instantiateCachedSeries(ConstructedSeries &CS,
                                       Instruction *AfterI) {
  auto IP = irBuilder.saveIP();
  setInsertPointAfter(irBuilder, AfterI);
  irBuilder.SetCurrentDebugLocation(AfterI->getDebugLoc());
  auto SeriesVal = instantiateLinearSeries(CS.LS);
  irBuilder.restoreIP(IP);
  return SeriesVal;
}

Value *Ripple::getCachedInstantiationFor(const Instruction *I) const {
  auto CS = getCachedSeries(I);
  if (CS)
    return LsCache.GeneratedSeries.lookup(CS.LS);
  else
    return nullptr;
}

bool Ripple::setRippleShape(const Value *V, const TensorShape &Shape) {
  return setRippleShape(dyn_cast_if_present<Instruction>(V), Shape);
}

bool Ripple::setRippleShape(const Instruction *I, const TensorShape &Shape) {
  assert(Shape.rank() == tensorRank());
  if (!I)
    return false;
  auto inserted = InstructionRippleShapes.insert(std::make_pair(I, Shape));
  // We modify if the shape changed
  if (!inserted.second && inserted.first->second != Shape) {
    inserted.first->second = Shape;
    if (intrinsicWithId(I, {Intrinsic::ripple_block_setshape}))
      llvm_unreachable("Ripple block shape are set in stone!");
    return true;
  }
  assert(inserted.first->second.rank() == tensorRank());
  return inserted.second;
}

void Ripple::invalidateRippleDataFor(const Value *V) {
  auto *I = const_cast<Instruction *>(dyn_cast_if_present<Instruction>(V));
  if (!I)
    return;
  InstructionRippleShapes.erase(I);
  // Linear series
  SlopeInstructions.erase(I);
  clearValidSerie(I);

  // If-convert
  ToSkipMaskingWhenIfConvert.erase(I);

  // Type-specific
  if (auto *Select = dyn_cast<SelectInst>(I))
    SelectToMaskWhenIfConvert.erase(Select);
  else if (auto *Call = dyn_cast<CallInst>(I))
    MaskedCalls.erase(Call);
  else if (auto *Alloca = dyn_cast<AllocaInst>(I))
    PromotableAlloca.erase(Alloca);
}

bool Ripple::isRippleIntrinsics(const Instruction *I) {
  return rippleBlockIntrinsics(I) || rippleReduceIntrinsics(I) ||
         rippleShuffleIntrinsics(I) || rippleBroadcastIntrinsic(I) ||
         rippleSliceIntrinsic(I);
}

DebugLoc Ripple::sanitizeRippleLocation(const Instruction *I) {
  if (!I)
    return DebugLoc();
  if (Ripple::isRippleIntrinsics(I))
    return stripInliningFromDebugLoc(I->getDebugLoc());
  else
    return I->getDebugLoc();
}

bool Ripple::allInstructionsHaveRippleShapes() const {
  unsigned InstructionCount = 0;
  for (const auto &Inst : instructions(F)) {
    auto Shape = InstructionRippleShapes.find(&Inst);
    InstructionCount++;
    if (Shape == InstructionRippleShapes.end()) {
      LLVM_DEBUG(dbgs() << "Found instruction w/o a ripple shape:" << Inst
                        << "\n");
      return false;
    }
  }
  if (InstructionCount != InstructionRippleShapes.size()) {
    LLVM_DEBUG(dbgs() << "InstructionCount: " << InstructionCount
                      << "\n  Mapping size: " << InstructionRippleShapes.size()
                      << "\n");
    assert(InstructionCount == InstructionRippleShapes.size() &&
           "Mapping to instructions not part of the function");
  }
  return true;
}

bool Ripple::simplifyFunction() {
  bool SimplifiedAny = false;
  for (auto &I : make_early_inc_range(instructions(F))) {
    Instruction *TryToSimplify = &I;
    // This process does not produce new instruction, it will fold instruction
    // and return a simpler form if possible or nullptr if it cannot
    Value *Simplified = simplifyInstruction(TryToSimplify, SQ);
    if (Simplified != nullptr) {
      LLVM_DEBUG(dbgs() << "Simplified " << *TryToSimplify << " with "
                        << *Simplified << "\n");

      // Replace TryToSimplify by Simplified
      invalidateRippleDataFor(TryToSimplify);
      auto TryToSimplifyIt = TryToSimplify->getIterator();
      ReplaceInstWithValue(TryToSimplifyIt, Simplified);
      SimplifiedAny = true;
    }
  }
  return SimplifiedAny;
}

bool LinearSeries::hasZeroSlopes() const {
  return getSlopeShape().isScalar() || all_of(slopes(), [](Value *V) {
           if (Constant *C = dyn_cast<Constant>(V))
             return C->isZeroValue();
           return false;
         });
}

bool LinearSeries::isScalarOrSplat() const {
  // Splat only requires that all slopes be zero
  return getBaseShape().isScalar() && hasZeroSlopes();
}

bool Ripple::hasNoVectorDimension() const {
  return none_of(idTypes.begin(), idTypes.end(),
                 [](auto &entry) { return entry.second == VectorDimension; });
}

Error Ripple::checkRippleBlockIntrinsics(IntrinsicInst *I) {
  if (I->getIntrinsicID() == Intrinsic::ripple_block_getsize ||
      I->getIntrinsicID() == Intrinsic::ripple_block_index) {
    auto DimensionIdx = *getConstantOperandValue(I, 1);
    if (DimensionIdx >= RippleIntrinsicsMaxDims) {
      std::string ErrMsg;
      {
        raw_string_ostream RSO(ErrMsg);
        RSO << "the requested dimension index (" << DimensionIdx
            << ") exceeds the number of dimensions supported by Ripple; "
               "supported values are "
               "in the range [0, "
            << RippleIntrinsicsMaxDims - 1 << "] per block shape";
      }
      DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                     ErrMsg);
      F.getContext().diagnose(DI);
      return createStringError(inconvertibleErrorCode(),
                               "Block shape index OOB");
    }
  }

  // TODO: when adding support for other vector "kinds" (e.g., SVE, SME) we will
  // have to check that the tensor shapes with vector.
  return Error::success();
}

Error Ripple::checkRippleReductionIntrinsics(IntrinsicInst *I) {
  // The relevant checks and warnings are already processed by
  // computeRippleReductionShape during shape-propagation
  return Error::success();
}

Error Ripple::checkRippleShuffleIntrinsics(IntrinsicInst *I) {
  Function *ShuffleFunc = dyn_cast<Function>(I->getArgOperand(3));

  bool IsPairShuffle =
      !cast<ConstantInt>(cast<CallInst>(I)->getArgOperand(2))->isZero();

  if (!ShuffleFunc) {
    DiagnosticInfoRippleWithLoc DI(
        DS_Error, F, sanitizeRippleLocation(I),
        "the ripple shuffle instruction expects a function (or lambda) for the "
        "index-mapping argument");
    F.getContext().diagnose(DI);
    return createStringError(
        inconvertibleErrorCode(),
        "Shuffle index mapping function is not a function");
  }

  // Check that the index-mapping function prototype matches what we expect
  FunctionType *ShuffleFuncType = ShuffleFunc->getFunctionType();
  if (!ShuffleFuncType->getReturnType()->isIntegerTy() ||
      ShuffleFuncType->getNumParams() != 2 ||
      !ShuffleFuncType->getParamType(0)->isIntegerTy() ||
      !ShuffleFuncType->getParamType(1)->isIntegerTy()) {
    std::string ErrMsg;
    llvm::raw_string_ostream RSO(ErrMsg);
    RSO << "the ripple shuffle instruction index-mapping operand "
           "must take two integer operands and output an integer; the "
           "provided "
           "prototype is "
        << *ShuffleFunc->getFunctionType();
    RSO.flush();
    DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                   ErrMsg);
    F.getContext().diagnose(DI);
    return createStringError(
        inconvertibleErrorCode(),
        "Shuffle index mapping function w/ wrong prototype");
  }

  // Is it defined in this module
  if (ShuffleFunc->empty()) {
    DiagnosticInfoRippleWithLoc DI(
        DS_Error, F, sanitizeRippleLocation(I),
        "the index mapping function (or lambda) operand of ripple shuffle "
        "requires its definition to be accessible in the same module as the "
        "function being processed");
    F.getContext().diagnose(DI);
    return createStringError(inconvertibleErrorCode(),
                             "Shuffle index mapping is not defined in module");
  }

  // No global memory access for compile time evaluation
  bool MappingFunUsingGlobals = false;
  for (auto &BB : *ShuffleFunc) {
    for (auto &I : BB) {
      if (auto *LoadInst = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        llvm::Value *Operand = LoadInst->getPointerOperand();
        if (auto *Global = llvm::dyn_cast<llvm::GlobalVariable>(Operand)) {
          LLVM_DEBUG(dbgs() << "Found a global: " << *Global << "\n");
          if (Global->isConstant())
            continue;
          std::string ErrMsg;
          llvm::raw_string_ostream RSO(ErrMsg);
          RSO << "The ripple shuffle instruction index-mapping function (or "
                 "lambda) cannot be evaluated at compile time because it is "
                 "accessing the value of a non-constant global variable \""
              << Global->getName() << "\"";
          RSO.flush();
          DiagnosticInfoRippleWithLoc DI(DS_Error, F,
                                         sanitizeRippleLocation(&I), ErrMsg);
          F.getContext().diagnose(DI);
          MappingFunUsingGlobals = true;
        }
      }
    }
  }
  if (MappingFunUsingGlobals)
    return createStringError(
        inconvertibleErrorCode(),
        "Shuffle index mapping function accesses non-const globals");

  // Check that there is no out-of-bound access
  DimSize BlockSize = getRippleShape(I).flatShape();
  Evaluator Evaler(DL, &targetLibraryInfo);
  unsigned ReportedIssues = 0;
  for (DimSize IIdx = 0; IIdx < BlockSize; ++IIdx) {
    Constant *RetVal;
    Constant *IdxArg = ConstantInt::get(ShuffleFuncType->getParamType(0), IIdx);
    Constant *BlockSizeArg =
        ConstantInt::get(ShuffleFuncType->getParamType(1), BlockSize);
    SmallVector<Constant *, 2> Args = {IdxArg, BlockSizeArg};

    if (!Evaler.EvaluateFunction(ShuffleFunc, RetVal, Args)) {
      std::string ErrMsg;
      llvm::raw_string_ostream RSO(ErrMsg);
      RSO << "failed to evaluate the index mapping function of ripple "
             "shuffle at compile time, the call was: "
          << ShuffleFunc->getName() << " with arguments (" << IIdx << ", "
          << BlockSize << ")";
      RSO.flush();
      DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                     ErrMsg);
      F.getContext().diagnose(DI);
      return createStringError(inconvertibleErrorCode(),
                               "Shuffle index mapping evaluation failed");
    }
    ConstantInt *RetIntVal = cast<ConstantInt>(RetVal);

    auto MaxInputIndex = IsPairShuffle ? BlockSize * 2 : BlockSize;
    if (RetIntVal->uge(MaxInputIndex)) {
      std::string ErrMsg;
      llvm::raw_string_ostream RSO(ErrMsg);
      RSO << "Evaluation of the index mapping function of ripple_shuffle";
      if (IsPairShuffle)
        RSO << "_pair";
      RSO << " returned an out of bound value; the call was to ";
      char *DemangledName = itaniumDemangle(ShuffleFunc->getName().str());
      if (DemangledName) {
        StringRef DemangledString(DemangledName);
        RSO << "\"" << DemangledString << "\"";
        free(DemangledName);
      } else if (!ShuffleFunc->getName().empty()) {
        RSO << "\"" << ShuffleFunc->getName() << "\"";
      }
      RSO << " with arguments (" << *IdxArg << ", " << *BlockSizeArg
          << "). The returned value ("
          << RetIntVal->getValue().getLimitedValue()
          << ") is greater or equal to the size of ";
      if (IsPairShuffle)
        RSO << "two (pair) tensors (" << BlockSize * 2 << ")";
      else
        RSO << "the tensor (" << BlockSize << ")";
      RSO.flush();
      DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(I),
                                     ErrMsg);
      F.getContext().diagnose(DI);
      ReportedIssues++;
    }
  }
  if (ReportedIssues > 0) {
    return createStringError(
        inconvertibleErrorCode(),
        "Shuffle index mapping evaluated to out-of-bound value");
  }

  return Error::success();
}

Error Ripple::checkRippleStore(const StoreInst *Store) const {
  if (getRippleShape(Store).isScalar())
    return Error::success();

  auto &PtrShape = getRippleShape(Store->getPointerOperand());
  auto &ValueShape = getRippleShape(Store->getValueOperand());

  // We are only allowed to broadcast values to match the address but not the
  // address to match the value. This is because it can be ambiguous what the
  // semantics would be in this case. Hence, notify the user if any tensor
  // dimension of the address is smaller than the value dimension
  if (PtrShape
          .testBothDims(ValueShape,
                        [](DimSize PtrDimSize, DimSize ValueDimSize) {
                          return PtrDimSize < ValueDimSize;
                        })
          .any()) {
    std::string ErrMsg;
    llvm::raw_string_ostream RSO(ErrMsg);
    RSO << "ripple does not allow implicit broadcasting of a store address "
           "to the value address; the value has "
        << ValueShape << " and the address has " << PtrShape
        << ". Hint: use ripple_id() for the address computation or use a "
           "reduction "
           "operation";
    RSO.flush();
    DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(Store),
                                   ErrMsg);
    F.getContext().diagnose(DI);
    return createStringError(inconvertibleErrorCode(),
                             "Cannot broadcast store address");
  }
  return Error::success();
}

Error Ripple::checkVectorBranch(Instruction *BranchOrSwitch) {
  auto firstInstructionWithValidDebugLoc = [](BasicBlock *BB) -> Instruction * {
    for (auto &I : make_range(BB->getFirstNonPHIOrDbgOrAlloca(), BB->end())) {
      auto DL = I.getDebugLoc();
      if (DL && !(DL.getLine() == 0 && DL.getCol() == 0))
        return &I;
    }
    return nullptr;
  };
  auto &MaskShape = getRippleShape(BranchOrSwitch);
  auto maskCanApplyToShape = [&](const TensorShape &S) -> bool {
    TensorShape ShapeBeforeBcast = MaskShape;
    auto ReductionDims = MaskShape.reductionDimensionsBeforeBroadcast(S);
    if (ReductionDims.any())
      ShapeBeforeBcast.reduceDimensions(ReductionDims);
    if (Error e = ShapeBeforeBcast.isBroadcastError(S)) {
      consumeError(std::move(e));
      return false;
    }
    return true;
  };
  auto maskCanApplyToInstruction = [&](const Instruction *I) -> bool {
    auto &IShape = getRippleShape(I);
    return maskCanApplyToShape(IShape);
  };
  // For vector branch/switch we check that the subgraph between the basic block
  // containing the instruction and the immediate post-dominator is a single
  // entry single exit (SESE) region.
  BasicBlock *BBWithVectorSw = BranchOrSwitch->getParent();
  BasicBlock *BranchPostDom =
      postdomTree.getNode(BBWithVectorSw)->getIDom()->getBlock();
  auto BBsInBetween = allBasicBlocksFromTo(BBWithVectorSw, BranchPostDom);
  bool HasErrors = false;
  for (auto *BB : BBsInBetween) {
    for (auto &I : *BB) {
      bool CheckInstruction = false;
      CheckInstruction = maskInstructionWhenIfConvert(&I);
      // Check that maskable instructions agree w/ the mask shape
      if (CheckInstruction && !maskCanApplyToInstruction(&I)) {
        std::string ErrMsg;
        raw_string_ostream RSO(ErrMsg);
        RSO << "this instruction, with " << getRippleShape(&I)
            << " is incompatible with a vector "
               "conditional";
        RSO.flush();
        DiagnosticInfoRippleWithLoc DI(DS_Error, F, sanitizeRippleLocation(&I),
                                       ErrMsg);
        F.getContext().diagnose(DI);
        std::string NoteMsg;
        raw_string_ostream NoteRSO(NoteMsg);
        NoteRSO << "for this vector conditional instruction with "
                << getRippleShape(BranchOrSwitch);
        NoteRSO.flush();
        DiagnosticInfoRippleWithLoc Note(
            DS_Note, F, sanitizeRippleLocation(BranchOrSwitch), NoteMsg);
        F.getContext().diagnose(Note);
        HasErrors = true;
      }
    }
    for (auto *IncomingBB : predecessors(BB)) {
      bool ComingFromInBetween = BBsInBetween.contains(IncomingBB);
      bool ComingFromBranchBB = IncomingBB == BBWithVectorSw;
      // There is a branch from Incoming into the subgraph invalidating the SESE
      // assumption
      if (!(ComingFromInBetween || ComingFromBranchBB ||
            hasTrivialLoopLikeBackEdge(BBWithVectorSw, BranchPostDom,
                                       domTree)) ||
          BB->hasAddressTaken()) {
        HasErrors = true;
        // Show that it's a problem related to if-conversion of the branch
        // instruction
        std::string ErrMsg;
        llvm::raw_string_ostream RSO(ErrMsg);
        RSO << "ripple cannot vectorize the vector "
            << (isa<BranchInst>(BranchOrSwitch) ? "branch" : "switch")
            << " because it applies to a non single-entry-single-exit (SESE) "
               "region";
        RSO.flush();
        DiagnosticInfoRippleWithLoc DI(
            DS_Error, F, sanitizeRippleLocation(BranchOrSwitch), ErrMsg);
        F.getContext().diagnose(DI);
      }
      if (!(ComingFromInBetween || ComingFromBranchBB ||
            hasTrivialLoopLikeBackEdge(BBWithVectorSw, BranchPostDom,
                                       domTree))) {
        // pinpoint the illegal branching
        DiagnosticLocation DL(
            sanitizeRippleLocation(IncomingBB->getTerminator()));
        if (DL.isValid()) {
          DiagnosticInfoRippleWithLoc DI(
              DS_Note, F, DL,
              "illegally branching from this instruction into the sub-graph");
          F.getContext().diagnose(DI);
        }

        // if there is a meaningful debug location in BB, display the first
        // target of the illegal branch
        if (Instruction *BranchInstWithDebugInfo =
                firstInstructionWithValidDebugLoc(BB)) {
          DiagnosticLocation TargetDL(
              sanitizeRippleLocation(BranchInstWithDebugInfo));
          if (TargetDL.isValid()) {
            DiagnosticInfoRippleWithLoc DI(
                DS_Note, F, TargetDL,
                "illegally branching to this instruction");
            F.getContext().diagnose(DI);
          }
        }
      }
      if (BB->hasAddressTaken()) {
        // Find the instructions that take the BB address
        bool ReportedNothing = true;
        BlockAddress *BA = BlockAddress::get(BB);
        for (auto *User : BA->users()) {
          if (Instruction *I = dyn_cast<Instruction>(User)) {
            DiagnosticLocation DL(sanitizeRippleLocation(I));
            if (DL.isValid()) {
              DiagnosticInfoRippleWithLoc DI(DS_Note, F,
                                             sanitizeRippleLocation(I),
                                             "illegally taking the address at");
              F.getContext().diagnose(DI);
              ReportedNothing = false;
            }
          }
        }
        if (ReportedNothing) {
          if (Instruction *BranchInstWithDebugInfo =
                  firstInstructionWithValidDebugLoc(BB)) {
            DiagnosticInfoRippleWithLoc DI(
                DS_Note, F, sanitizeRippleLocation(BranchInstWithDebugInfo),
                "illegally taking the address of a basic block starting with "
                "this instruction");
            F.getContext().diagnose(DI);
          } else {
            DiagnosticInfoRippleWithLoc DI(
                DS_Note, F, sanitizeRippleLocation(BranchOrSwitch),
                "illegally taking the address of a basic block");
            F.getContext().diagnose(DI);
          }
        }
      }
    }
  }
  if (HasErrors)
    return createStringError(inconvertibleErrorCode(),
                             "if-conversion SESE violation");
  else
    return Error::success();
}

Error Ripple::checkTypeCanBeVectorized(const Instruction *I) {
  auto checkVectorPromotionTypeValidity = [&](const Value *V) -> bool {
    Type *ValueType = V->getType();
    if (!VectorType::isValidElementType(ValueType)) {
      LLVM_DEBUG(dbgs() << *V << " has an invalid vector type!\n");
      const char *TypeHint = "";
      if (ValueType->isArrayTy())
        TypeHint = "array ";
      else if (ValueType->isStructTy())
        TypeHint = "structure ";
      else if (ValueType->isFunctionTy())
        TypeHint = "function ";
      DiagnosticInfoRippleWithLoc DI(
          DS_Error, F, sanitizeRippleLocation(I),
          Twine("Ripple cannot create a vector type from this instruction's ") +
              TypeHint +
              "type; Allowed vector element types are integer, floating point "
              "and pointer");
      F.getContext().diagnose(DI);
      return false;
    }
    return true;
  };
  bool Valid = true;

  // Check that the instruction itself can be vectorized
  if (!I->getType()->isVoidTy())
    Valid = Valid && checkVectorPromotionTypeValidity(I);

  // And that the instruction operands can be vectorized (broadcasted)
  for (auto &U : vectorizableOperands(I))
    Valid = Valid && checkVectorPromotionTypeValidity(U);

  if (!Valid)
    return createStringError(inconvertibleErrorCode(),
                             "if-conversion SESE violation");
  else
    return Error::success();
}

Error Ripple::checkRippleFunctionReturn(const ReturnInst *Return) const {
  if (getRippleShape(Return).isVector()) {
    DiagnosticInfoRippleWithLoc DI(
        DS_Error, F, sanitizeRippleLocation(Return),
        "Ripple does not allow vectorization of the return value");
    F.getContext().diagnose(DI);
    return createStringError(inconvertibleErrorCode(),
                             "Function returns tensor");
  }
  return Error::success();
}

Error Ripple::checkRippleSemantics() {
  Error AllErrors = Error::success();
  for (auto &I : instructions(F)) {
    auto &InstructionShape = getRippleShape(&I);

    if (InstructionShape.isVector())
      AllErrors =
          llvm::joinErrors(std::move(AllErrors), checkTypeCanBeVectorized(&I));

    if ((isa<BranchInst>(&I) || isa<SwitchInst>(&I)) &&
        InstructionShape.isVector()) {
      AllErrors = llvm::joinErrors(std::move(AllErrors), checkVectorBranch(&I));

    } else if (IntrinsicInst *RippleBlockI = rippleBlockIntrinsics(&I)) {
      AllErrors = llvm::joinErrors(std::move(AllErrors),
                                   checkRippleBlockIntrinsics(RippleBlockI));

    } else if (IntrinsicInst *RippleRedI = rippleReduceIntrinsics(&I)) {
      AllErrors = llvm::joinErrors(std::move(AllErrors),
                                   checkRippleReductionIntrinsics(RippleRedI));

    } else if (IntrinsicInst *rippleShuffleI = rippleShuffleIntrinsics(&I)) {
      AllErrors = llvm::joinErrors(
          std::move(AllErrors), checkRippleShuffleIntrinsics(rippleShuffleI));

    } else if (StoreInst *Store = dyn_cast<StoreInst>(&I)) {
      AllErrors =
          llvm::joinErrors(std::move(AllErrors), checkRippleStore(Store));
    } else if (auto *Return = dyn_cast<ReturnInst>(&I)) {
      AllErrors = llvm::joinErrors(std::move(AllErrors),
                                   checkRippleFunctionReturn(Return));
    }
  }
  return AllErrors;
}

std::string Ripple::tensorizedName(StringRef Name, const TensorShape &Shape) {
  std::string TensorName;
  raw_string_ostream RSO(TensorName);
  RSO << Name << ".ripple";
  if (Shape.rank() > 0) {
    RSO << ".t" << Shape[Shape.rank() - 1];
    for (unsigned RankIdx = Shape.rank() - 2, E = Shape.rank(); RankIdx < E;
         --RankIdx)
      RSO << "x" << Shape[RankIdx];
  }
  RSO.flush();
  return TensorName;
}

Expected<TensorShape> Ripple::combineShapeBcastWithErrorReporting(
    const TensorShape &ShapeToBeBroadcasted, const TensorShape &OtherShape,
    StringRef ShapeToBeBcastedMsg, DebugLoc ShapeToBeBroadcastedLocation,
    StringRef OtherShapeMsg, DebugLoc SecondLocation) {
  TensorShape Bcasted = ShapeToBeBroadcasted;
  if (Error E = Bcasted.combineShapeBcast(OtherShape)) {
    Error Err = handleErrors(std::move(E), [&](StringError &StrErr) {
      std::string ErrMsg;
      llvm::raw_string_ostream RSO(ErrMsg);
      RSO << "broadcast failure: " << StrErr.getMessage();
      RSO.flush();
      DiagnosticInfoRippleWithLoc Diag(DS_Error, F,
                                       ShapeToBeBroadcastedLocation, ErrMsg);
      F.getContext().diagnose(Diag);
    });
    if (Err) {
      LLVM_DEBUG(dbgs() << "Error type unreported to the user: " << Err);
      return std::move(Err);
    }
    if (!ShapeToBeBcastedMsg.empty()) {
      std::string ErrMsg;
      llvm::raw_string_ostream RSO(ErrMsg);
      RSO << ShapeToBeBcastedMsg << " of shape " << ShapeToBeBroadcasted;
      RSO.flush();
      DiagnosticInfoRippleWithLoc Diag(DS_Note, F, ShapeToBeBroadcastedLocation,
                                       ErrMsg);
      F.getContext().diagnose(Diag);
    }
    if (!OtherShapeMsg.empty()) {
      std::string ErrMsg;
      llvm::raw_string_ostream RSO(ErrMsg);
      RSO << OtherShapeMsg << " of shape " << OtherShape;
      RSO.flush();
      DiagnosticInfoRippleWithLoc Diag(DS_Note, F, SecondLocation, ErrMsg);
      F.getContext().diagnose(Diag);
    }
    return createStringError(inconvertibleErrorCode(),
                             "CombineShapeBcast failure");
  }
  return Bcasted;
}

Value *Ripple::instantiateLinearSeriesNoCache(const LinearSeries &LS) {
  return instantiateLinearSeries(&LS, false);
}

void Ripple::visitAllInstructionsBeingClobberedBy(
    const MemoryDef *Def, std::function<bool(Instruction *)> Apply,
    bool VisitUnreachableFromEntry) {
  Instruction *DefInst = Def->getMemoryInst();
  auto DefLocationOpt = MemoryLocation::getOrNone(DefInst);
  if (!DefLocationOpt)
    return;
  auto &DefLocation = *DefLocationOpt;

  std::queue<MemoryAccess *> WorkQueue;
  auto addUsersToQueue = [&](MemoryAccess *Access) {
    for (User *U : Access->users())
      WorkQueue.push(static_cast<MemoryAccess *>(U));
  };

  // Todo: throw warnings when we encounter PartialAlias and MayAlias so that
  // the user knows something may have a non-deterministic shape!
  auto processAccess = [&](MemoryAccess *Access) {
    if (MemoryUse *Use = dyn_cast<MemoryUse>(Access)) {
      Instruction *MemInst = Use->getMemoryInst();
      if (auto UseLocation = MemoryLocation::getOrNone(MemInst))
        if (AA.isMustAlias(DefLocation, *UseLocation) &&
            (VisitUnreachableFromEntry ||
             domTree.isReachableFromEntry(MemInst->getParent())))
          if (!Apply(MemInst))
            // Return early when Apply returns false
            return;
    } else if (MemoryDef *OtherDef = dyn_cast<MemoryDef>(Access)) {
      // Stop processing when a new definition is encountered
      if (auto OtherDefLocation =
              MemoryLocation::getOrNone(OtherDef->getMemoryInst()))
        if (AA.isMustAlias(DefLocation, *OtherDefLocation))
          // Don't visit further because Def and OtherDef clobber
          return;
    } else
      assert(isa<MemoryPhi>(Access));
    addUsersToQueue(Access);
  };

  // Start with the Store users
  addUsersToQueue(MemSSA.getMemoryAccess(DefInst));

  SmallPtrSet<const MemoryAccess *, 0> AlreadyVisited;
  AlreadyVisited.insert(Def);
  while (!WorkQueue.empty()) {
    MemoryAccess *Access = WorkQueue.front();
    WorkQueue.pop();
    if (AlreadyVisited.contains(Access))
      continue;
    AlreadyVisited.insert(Access);
    processAccess(Access);
  }
}

void Ripple::visitAllClobberingInstructions(
    MemoryUse *Use, std::function<bool(Instruction *)> Apply,
    bool VisitUnreachableFromEntry) {
  Instruction *UseInst = Use->getMemoryInst();
  auto UseLocationOpt = MemoryLocation::getOrNone(UseInst);
  if (!UseLocationOpt)
    return;
  auto &UseLocation = *UseLocationOpt;

  // Start with the first clobber (which may be a Phi or Def)
  std::queue<MemoryAccess *> WorkQueue;
  WorkQueue.push(MemSSAWalker.getClobberingMemoryAccess(Use));

  SmallPtrSet<const MemoryAccess *, 0> AlreadyVisited;
  while (!WorkQueue.empty()) {
    MemoryAccess *Access = WorkQueue.front();
    WorkQueue.pop();
    if (AlreadyVisited.contains(Access))
      continue;
    AlreadyVisited.insert(Access);

    assert(!isa<MemoryUse>(Access) &&
           "We shouldn't be processing MemoryUse when looking for clobbering "
           "Instructions");

    if (MemSSA.isLiveOnEntryDef(Access))
      continue;
    else if (MemoryPhi *Phi = dyn_cast<MemoryPhi>(Access))
      for (auto &IncomingVal : Phi->incoming_values()) {
        auto *IncomingAccess = cast<MemoryAccess>(IncomingVal);
        WorkQueue.push(IncomingAccess);
      }
    else if (MemoryDef *Def = dyn_cast<MemoryDef>(Access)) {
      Instruction *MemInst = Def->getMemoryInst();
      if (auto DefLocation = MemoryLocation::getOrNone(MemInst))
        if (AA.isMustAlias(UseLocation, *DefLocation) &&
            (VisitUnreachableFromEntry ||
             domTree.isReachableFromEntry(MemInst->getParent()))) {
          if (Apply(MemInst))
            continue;
          else
            return;
        }
      WorkQueue.push(Def->getDefiningAccess());
    } else {
      auto *Use = cast<MemoryUse>(Access);
      WorkQueue.push(Use->getDefiningAccess());
    }
  }
}

std::pair<AliasResult, AllocaInst *>
Ripple::aliasesWithAlloca(const MemoryLocation &Loc,
                          const DenseSet<AssertingVH<AllocaInst>> &AllocaSet,
                          AliasResult::Kind AliasKind) const {
  for (AllocaInst *Alloca : AllocaSet)
    if (auto Size = Alloca->getAllocationSize(DL)) {
      auto AllocaLoc =
          MemoryLocation(Alloca, LocationSize::precise(DL.getTypeStoreSize(
                                     Alloca->getAllocatedType())));
      auto AAResult = AA.alias(AllocaLoc, Loc);
      // If kind is AliasResult::NoAlias return when any kind of aliasing happen
      bool CheckNoAlias = AliasKind == AliasResult::NoAlias;
      if ((CheckNoAlias && AAResult != AliasResult::NoAlias) ||
          (!CheckNoAlias && AAResult == AliasKind))
        return {AAResult, Alloca};
    }
  return {AliasResult::NoAlias, nullptr};
}

template <bool ReportAmbiguity>
IntrinsicInst *Ripple::getBlockShapeIntrinsic(const Use &RippleBlockShapePtr) {
  std::array<Instruction *, 2> Clobbering = {};
  auto diagnoseAmbiguity = [this, &Clobbering](Instruction *BSAccess) {
    {
      DiagnosticInfoRippleWithLoc DI(
          DS_Error, F, sanitizeRippleLocation(BSAccess),
          "block shape access is ambiguous (multiple shapes apply)");
      F.getContext().diagnose(DI);
    }
    {
      DiagnosticInfoRippleWithLoc DI(DS_Note, F,
                                     sanitizeRippleLocation(Clobbering[0]),
                                     "can come from here");
      F.getContext().diagnose(DI);
    }
    {
      DiagnosticInfoRippleWithLoc DI(
          DS_Note, F, sanitizeRippleLocation(Clobbering[1]), "and here");
      F.getContext().diagnose(DI);
    }
  };
  // Avoid infinite visits w/ cycles
  SmallPtrSet<Value *, 8> Visited;
  std::function<IntrinsicInst *(Value *)> getBlockShapeIntrinsicHelper;
  getBlockShapeIntrinsicHelper = [&](Value *BSPtr) -> IntrinsicInst * {
    if (!Visited.insert(BSPtr).second)
      return nullptr;
    // Most likely case in SSA
    if (auto *II = intrinsicWithId(dyn_cast<Instruction>(BSPtr),
                                   {Intrinsic::ripple_block_setshape})) {
      return II;
      // Most likely case in O0, coming from clang
    } else if (auto *LoadBS = dyn_cast<LoadInst>(BSPtr)) {
      IntrinsicInst *BS = nullptr;
      visitAllClobberingInstructions(
          cast<MemoryUse>(MemSSA.getMemoryAccess(LoadBS)),
          [&](Instruction *Clobber) -> bool {
            LLVM_DEBUG(dbgs() << "Visiting clobber " << *Clobber << "\n");
            if (auto *Store = dyn_cast<StoreInst>(Clobber)) {
              Value *RippleSetShape = Store->getValueOperand();
              if (auto *II =
                      intrinsicWithId(dyn_cast<Instruction>(RippleSetShape),
                                      {Intrinsic::ripple_block_setshape})) {
                LLVM_DEBUG(dbgs()
                           << "Found BS stored by clobber " << *II << "\n");
                if (ReportAmbiguity && BS) {
                  // Multiple clobbers
                  Clobbering[1] = Store;
                  diagnoseAmbiguity(LoadBS);
                  BS = nullptr;
                  return false;
                }
                Clobbering[0] = Store;
                BS = II;
                // Continue only if we look for ambiguity (another clobber)
                return ReportAmbiguity;
              } else if (auto *Load = dyn_cast<LoadInst>(RippleSetShape)) {
                IntrinsicInst *FoundBS = getBlockShapeIntrinsicHelper(Load);
                if (ReportAmbiguity && BS && FoundBS) {
                  // Multiple clobber on the same value, we have a problem
                  Clobbering[1] = Store;
                  diagnoseAmbiguity(Load);
                  BS = nullptr;
                  return false;
                }
                Clobbering[0] = Store;
                BS = FoundBS;
                // Continue if BS is not found or looking for ambiguity
                return ReportAmbiguity || !BS;
              }
            }
            return true;
          });
      return BS;
    } else if (ReportAmbiguity) {
      if (auto *PHI = dyn_cast<PHINode>(BSPtr)) {
        if (PHI->getNumIncomingValues() >= 2) {
          if (auto *IncomingI = dyn_cast<Instruction>(PHI->getIncomingValue(0)))
            Clobbering[0] = IncomingI;
          else
            Clobbering[0] = PHI->getIncomingBlock(0)->getTerminator();
          if (auto *IncomingI = dyn_cast<Instruction>(PHI->getIncomingValue(1)))
            Clobbering[1] = IncomingI;
          else
            Clobbering[1] = PHI->getIncomingBlock(1)->getTerminator();
          diagnoseAmbiguity(
              dyn_cast<Instruction>(RippleBlockShapePtr.getUser()));
        }
      }
    }
    return nullptr;
  };
  return getBlockShapeIntrinsicHelper(RippleBlockShapePtr);
}

TensorShape::DimSize
Ripple::getRippleGetSizeValue(const IntrinsicInst *RippleGetSize) {
  assert(RippleGetSize->getIntrinsicID() == Intrinsic::ripple_block_getsize);
  auto *ShapeII = getBlockShapeIntrinsic(RippleGetSize->getArgOperandUse(0));
  // Checked by checkBlockShapeUsage
  assert(ShapeII);
  PEIdentifier ProcElem = *getConstantOperandValue(ShapeII, 0);
  auto Dimension = *getConstantOperandValue(RippleGetSize, 1);
  if (Dimension < PERank(ProcElem)) {
    const auto &BlockShape = setShapeToTensorShape(ShapeII);
    return BlockShape[rippleToTensor({ProcElem, Dimension})];
  } else
    return 1;
}

TensorShape
Ripple::setShapeToTensorShape(const IntrinsicInst *RippleSetShape) const {
  assert(RippleSetShape->getIntrinsicID() == Intrinsic::ripple_block_setshape);
  TensorShape::Shape BlockShape(tensorRank(), DimSize(1));
  PEIdentifier ProcElem = *getConstantOperandValue(RippleSetShape, 0);
  for (unsigned ArgIdx = 1, E = RippleSetShape->arg_size(); ArgIdx < E;
       ++ArgIdx) {
    DimSize DS = *getConstantOperandValue(RippleSetShape, ArgIdx);
    if (DS > 1) {
      // This should be true by construction; the rank is the maximum of all the
      // ripple_set_block_shape for the same PE in this function
      assert(PERank(ProcElem) > ArgIdx - 1);
      BlockShape[rippleToTensor({ProcElem, ArgIdx - 1})] = DS;
    }
  }
  return TensorShape(std::move(BlockShape));
}

Error Ripple::checkBlockShapeUsage(const Function &F) {
  for (auto &I : instructions(F)) {
    if (const IntrinsicInst *II = rippleIntrinsicsWithBlockShapeOperand(&I)) {
      auto &BlockShapeOperandUse = II->getArgOperandUse(0);
      auto *BS = getBlockShapeIntrinsic<true>(BlockShapeOperandUse);
      if (!BS) {
        DiagnosticInfoRippleWithLoc DI(
            DS_Error, F, sanitizeRippleLocation(II),
            "Cannot locate the source block shape of this ripple intrinsic");
        F.getContext().diagnose(DI);
        return createStringError(inconvertibleErrorCode(),
                                 "Cannot find BS for ripple construct");
      }
    } else if (auto *CallI = dyn_cast<CallInst>(&I)) {
      for (auto &Arg : CallI->args()) {
        if (getBlockShapeIntrinsic(Arg)) {
          LLVM_DEBUG(dbgs() << "Found call with BS " << *CallI << "\n");
          if (!CallI->getCalledFunction() ||
              (!CallI->getCalledFunction()->isIntrinsic() &&
               CallI->getCalledFunction()->isDeclaration())) {
            DiagnosticInfoRippleWithLoc DI(
                DS_Error, F, sanitizeRippleLocation(CallI),
                "Passing a ripple block shape to a function call with no known "
                "definition is not allowed. Make sure that the function is "
                "available for ripple processing.");
            F.getContext().diagnose(DI);
            return createStringError(inconvertibleErrorCode(),
                                     "Call to declaration with BS");
          }
        }
      }
    }
  }
  return Error::success();
}

bool Ripple::rippleVectorizeCall(const CallInst &CI) const {
  return none_of(CI.args(),
                 [](auto &Use) { return Use->getType()->isVectorTy(); }) &&
         any_of(CI.args(),
                [&](auto &Use) { return getRippleShape(Use).isVector(); });
}
