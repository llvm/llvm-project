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
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRipple.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CFGUpdate.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Evaluator.h"
#include <array>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <queue>
#include <string>
#include <vector>

using namespace llvm;

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
