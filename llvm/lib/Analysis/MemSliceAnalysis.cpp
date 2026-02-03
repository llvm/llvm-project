//===- MemSliceAnalysis.cpp - Analyze memory slices -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the MemSliceAnalysis infrastructure for analyzing
/// how a pointer is accessed and partitioning it into slices.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemSliceAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mem-slice-analysis"

namespace {

/// A helper that folds a select instruction.
static Value *foldSelectInst(SelectInst &SI) {
  // If the condition being selected on is a constant or the same value is
  // being selected between, fold the select. Yes this does (rarely) happen
  // early on.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(SI.getCondition()))
    return SI.getOperand(1 + CI->isZero());
  if (SI.getOperand(1) == SI.getOperand(2))
    return SI.getOperand(1);

  return nullptr;
}

/// A helper that folds a PHI node or a select.
static Value *foldPHINodeOrSelectInst(Instruction &I) {
  if (PHINode *PN = dyn_cast<PHINode>(&I)) {
    // If PN merges together the same value, return that value.
    return PN->hasConstantValue();
  }
  return foldSelectInst(cast<SelectInst>(I));
}

} // anonymous namespace

/// Builder for the memory slices.
///
/// This class builds a set of memory slices by recursively visiting the uses
/// of a base ptr value and making a slice for each load and store at each
/// computable offset from it.
class MemSlices::SliceBuilder : public PtrUseVisitor<MemSlices::SliceBuilder> {
  friend class PtrUseVisitor<MemSlices::SliceBuilder>;
  friend class InstVisitor<MemSlices::SliceBuilder>;

  using Base = PtrUseVisitor<MemSlices::SliceBuilder>;

  const uint64_t AllocSize;
  MemSlices &AS;

  SmallDenseMap<Instruction *, unsigned> MemTransferSliceMap;
  SmallDenseMap<Instruction *, uint64_t> PHIOrSelectSizes;

  /// Set to de-duplicate dead instructions found in the use walk.
  SmallPtrSet<Instruction *, 4> VisitedDeadInsts;

  // When this access is via an llvm.protected.field.ptr intrinsic, contains
  // the second argument to the intrinsic, the discriminator.
  Value *ProtectedFieldDisc = nullptr;

public:
  SliceBuilder(const DataLayout &DL, AllocaInst &AI, MemSlices &AS)
      : PtrUseVisitor<MemSlices::SliceBuilder>(DL),
        AllocSize(AI.getAllocationSize(DL)->getFixedValue()), AS(AS) {}

private:
  void markAsDead(Instruction &I) {
    if (VisitedDeadInsts.insert(&I).second)
      AS.DeadUsers.push_back(&I);
  }

  void insertUse(Instruction &I, const APInt &Offset, uint64_t Size,
                 bool IsSplittable = false) {
    // Completely skip uses which have a zero size or start either before or
    // past the end of the allocation.
    if (Size == 0 || Offset.uge(AllocSize)) {
      LLVM_DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte use @"
                        << Offset
                        << " which has zero size or starts outside of the "
                        << AllocSize << " byte memory:\n"
                        << "  base ptr: " << AS.AI << "\n"
                        << "       use: " << I << "\n");
      return markAsDead(I);
    }

    uint64_t BeginOffset = Offset.getZExtValue();
    uint64_t EndOffset = BeginOffset + Size;

    // Clamp the end offset to the end of the allocation. Note that this is
    // formulated to handle even the case where "BeginOffset + Size" overflows.
    // This may appear superficially to be something we could ignore entirely,
    // but that is not so! There may be widened loads or PHI-node uses where
    // some instructions are dead but not others. We can't completely ignore
    // them, and so have to record at least the information here.
    assert(AllocSize >= BeginOffset); // Established above.
    if (Size > AllocSize - BeginOffset) {
      LLVM_DEBUG(dbgs() << "WARNING: Clamping a " << Size << " byte use @"
                        << Offset << " to remain within the " << AllocSize
                        << " byte memory:\n"
                        << "  base ptr: " << AS.AI << "\n"
                        << "       use: " << I << "\n");
      EndOffset = AllocSize;
    }

    AS.Slices.push_back(
        MemSlice(BeginOffset, EndOffset, U, IsSplittable, ProtectedFieldDisc));
  }

  void visitBitCastInst(BitCastInst &BC) {
    if (BC.use_empty())
      return markAsDead(BC);

    return Base::visitBitCastInst(BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    if (ASC.use_empty())
      return markAsDead(ASC);

    return Base::visitAddrSpaceCastInst(ASC);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    if (GEPI.use_empty())
      return markAsDead(GEPI);

    return Base::visitGetElementPtrInst(GEPI);
  }

  void handleLoadOrStore(Type *Ty, Instruction &I, const APInt &Offset,
                         uint64_t Size, bool IsVolatile) {
    // We allow splitting of non-volatile loads and stores where the type is an
    // integer type. These may be used to implement 'memcpy' or other "transfer
    // of bits" patterns.
    bool IsSplittable =
        Ty->isIntegerTy() && !IsVolatile && DL.typeSizeEqualsStoreSize(Ty);

    insertUse(I, Offset, Size, IsSplittable);
  }

  void visitLoadInst(LoadInst &LI) {
    assert((!LI.isSimple() || LI.getType()->isSingleValueType()) &&
           "All simple FCA loads should have been pre-split");

    // If there is a load with an unknown offset, we can still perform store
    // to load forwarding for other known-offset loads.
    if (!IsOffsetKnown)
      return PI.setEscapedReadOnly(&LI);

    TypeSize Size = DL.getTypeStoreSize(LI.getType());
    if (Size.isScalable()) {
      unsigned VScale = LI.getFunction()->getVScaleValue();
      if (!VScale)
        return PI.setAborted(&LI);

      Size = TypeSize::getFixed(Size.getKnownMinValue() * VScale);
    }

    return handleLoadOrStore(LI.getType(), LI, Offset, Size.getFixedValue(),
                             LI.isVolatile());
  }

  void visitStoreInst(StoreInst &SI) {
    Value *ValOp = SI.getValueOperand();
    if (ValOp == *U)
      return PI.setEscapedAndAborted(&SI);
    if (!IsOffsetKnown)
      return PI.setAborted(&SI);

    TypeSize StoreSize = DL.getTypeStoreSize(ValOp->getType());
    if (StoreSize.isScalable()) {
      unsigned VScale = SI.getFunction()->getVScaleValue();
      if (!VScale)
        return PI.setAborted(&SI);

      StoreSize = TypeSize::getFixed(StoreSize.getKnownMinValue() * VScale);
    }

    uint64_t Size = StoreSize.getFixedValue();

    // If this memory access can be shown to *statically* extend outside the
    // bounds of the allocation, it's behavior is undefined, so simply
    // ignore it. Note that this is more strict than the generic clamping
    // behavior of insertUse. We also try to handle cases which might run the
    // risk of overflow.
    // FIXME: We should instead consider the pointer to have escaped if this
    // function is being instrumented for addressing bugs or race conditions.
    if (Size > AllocSize || Offset.ugt(AllocSize - Size)) {
      LLVM_DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte store @"
                        << Offset << " which extends past the end of the "
                        << AllocSize << " byte memory:\n"
                        << "  base ptr: " << AS.AI << "\n"
                        << "       use: " << SI << "\n");
      return markAsDead(SI);
    }

    assert((!SI.isSimple() || ValOp->getType()->isSingleValueType()) &&
           "All simple FCA stores should have been pre-split");
    handleLoadOrStore(ValOp->getType(), SI, Offset, Size, SI.isVolatile());
  }

  void visitMemSetInst(MemSetInst &II) {
    assert(II.getRawDest() == *U && "Pointer use is not the destination?");
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    if ((Length && Length->getValue() == 0) ||
        (IsOffsetKnown && Offset.uge(AllocSize)))
      // Zero-length mem transfer intrinsics can be ignored entirely.
      return markAsDead(II);

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    insertUse(II, Offset,
              Length ? Length->getLimitedValue()
                     : AllocSize - Offset.getLimitedValue(),
              (bool)Length);
  }

  void visitMemTransferInst(MemTransferInst &II) {
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    if (Length && Length->getValue() == 0)
      // Zero-length mem transfer intrinsics can be ignored entirely.
      return markAsDead(II);

    // Because we can visit these intrinsics twice, also check to see if the
    // first time marked this instruction as dead. If so, skip it.
    if (VisitedDeadInsts.count(&II))
      return;

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    // This side of the transfer is completely out-of-bounds, and so we can
    // nuke the entire transfer. However, we also need to nuke the other side
    // if already added to our partitions.
    // FIXME: Yet another place we really should bypass this when
    // instrumenting for ASan.
    if (Offset.uge(AllocSize)) {
      SmallDenseMap<Instruction *, unsigned>::iterator MTPI =
          MemTransferSliceMap.find(&II);
      if (MTPI != MemTransferSliceMap.end())
        AS.Slices[MTPI->second].kill();
      return markAsDead(II);
    }

    uint64_t RawOffset = Offset.getLimitedValue();
    uint64_t Size = Length ? Length->getLimitedValue() : AllocSize - RawOffset;

    // Check for the special case where the same exact value is used for both
    // source and dest.
    if (*U == II.getRawDest() && *U == II.getRawSource()) {
      // For non-volatile transfers this is a no-op.
      if (!II.isVolatile())
        return markAsDead(II);

      return insertUse(II, Offset, Size, /*IsSplittable=*/false);
    }

    // If we have seen both source and destination for a mem transfer, then
    // they both point to the same memory.
    bool Inserted;
    SmallDenseMap<Instruction *, unsigned>::iterator MTPI;
    std::tie(MTPI, Inserted) =
        MemTransferSliceMap.insert(std::make_pair(&II, AS.Slices.size()));
    unsigned PrevIdx = MTPI->second;
    if (!Inserted) {
      MemSlice &PrevP = AS.Slices[PrevIdx];

      // Check if the begin offsets match and this is a non-volatile transfer.
      // In that case, we can completely elide the transfer.
      if (!II.isVolatile() && PrevP.beginOffset() == RawOffset) {
        PrevP.kill();
        return markAsDead(II);
      }

      // Otherwise we have an offset transfer within the same memory. We can't
      // split those.
      PrevP.makeUnsplittable();
    }

    // Insert the use now that we've fixed up the splittable nature.
    insertUse(II, Offset, Size, /*IsSplittable=*/Inserted && Length);

    // Check that we ended up with a valid index in the map.
    assert(AS.Slices[PrevIdx].getUse()->getUser() == &II &&
           "Map index doesn't point back to a slice with this user.");
  }

  // Disable SRoA for any intrinsics except for lifetime invariants.
  // FIXME: What about debug intrinsics? This matches old behavior, but
  // doesn't make sense.
  void visitIntrinsicInst(IntrinsicInst &II) {
    if (II.isDroppable()) {
      AS.DeadUseIfPromotable.push_back(U);
      return;
    }

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    if (II.isLifetimeStartOrEnd()) {
      insertUse(II, Offset, AllocSize, true);
      return;
    }

    if (II.getIntrinsicID() == Intrinsic::protected_field_ptr) {
      // We only handle loads and stores as users of llvm.protected.field.ptr.
      // Other uses may add items to the worklist, which will cause
      // ProtectedFieldDisc to be tracked incorrectly.
      AS.PFPUsers.push_back(&II);
      ProtectedFieldDisc = II.getArgOperand(1);
      for (Use &U : II.uses()) {
        this->U = &U;
        if (auto *LI = dyn_cast<LoadInst>(U.getUser()))
          visitLoadInst(*LI);
        else if (auto *SI = dyn_cast<StoreInst>(U.getUser()))
          visitStoreInst(*SI);
        else
          PI.setAborted(&II);
        if (PI.isAborted())
          break;
      }
      ProtectedFieldDisc = nullptr;
      return;
    }

    Base::visitIntrinsicInst(II);
  }

  Instruction *hasUnsafePHIOrSelectUse(Instruction *Root, uint64_t &Size) {
    // We consider any PHI or select that results in a direct load or store of
    // the same offset to be a viable use for slicing purposes. These uses
    // are considered unsplittable and the size is the maximum loaded or stored
    // size.
    SmallPtrSet<Instruction *, 4> Visited;
    SmallVector<std::pair<Instruction *, Instruction *>, 4> Uses;
    Visited.insert(Root);
    Uses.push_back(std::make_pair(cast<Instruction>(*U), Root));
    const DataLayout &DL = Root->getDataLayout();
    // If there are no loads or stores, the access is dead. We mark that as
    // a size zero access.
    Size = 0;
    do {
      Instruction *I, *UsedI;
      std::tie(UsedI, I) = Uses.pop_back_val();

      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        TypeSize LoadSize = DL.getTypeStoreSize(LI->getType());
        if (LoadSize.isScalable()) {
          PI.setAborted(LI);
          return nullptr;
        }
        Size = std::max(Size, LoadSize.getFixedValue());
        continue;
      }
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        Value *Op = SI->getOperand(0);
        if (Op == UsedI)
          return SI;
        TypeSize StoreSize = DL.getTypeStoreSize(Op->getType());
        if (StoreSize.isScalable()) {
          PI.setAborted(SI);
          return nullptr;
        }
        Size = std::max(Size, StoreSize.getFixedValue());
        continue;
      }

      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
        if (!GEP->hasAllZeroIndices())
          return GEP;
      } else if (!isa<BitCastInst>(I) && !isa<PHINode>(I) &&
                 !isa<SelectInst>(I) && !isa<AddrSpaceCastInst>(I)) {
        return I;
      }

      for (User *U : I->users())
        if (Visited.insert(cast<Instruction>(U)).second)
          Uses.push_back(std::make_pair(I, cast<Instruction>(U)));
    } while (!Uses.empty());

    return nullptr;
  }

  void visitPHINodeOrSelectInst(Instruction &I) {
    assert(isa<PHINode>(I) || isa<SelectInst>(I));
    if (I.use_empty())
      return markAsDead(I);

    // If this is a PHI node before a catchswitch, we cannot insert any non-PHI
    // instructions in this BB, which may be required during rewriting. Bail out
    // on these cases.
    if (isa<PHINode>(I) && !I.getParent()->hasInsertionPt())
      return PI.setAborted(&I);

    // TODO: We could use simplifyInstruction here to fold PHINodes and
    // SelectInsts. However, doing so requires to change the current
    // dead-operand-tracking mechanism. For instance, suppose neither loading
    // from %U nor %other traps. Then "load (select undef, %U, %other)" does not
    // trap either.  However, if we simply replace %U with undef using the
    // current dead-operand-tracking mechanism, "load (select undef, undef,
    // %other)" may trap because the select may return the first operand
    // "undef".
    if (Value *Result = foldPHINodeOrSelectInst(I)) {
      if (Result == *U)
        // If the result of the constant fold will be the pointer, recurse
        // through the PHI/select as if we had RAUW'ed it.
        enqueueUsers(I);
      else
        // Otherwise the operand to the PHI/select is dead, and we can replace
        // it with poison.
        AS.DeadOperands.push_back(U);

      return;
    }

    if (!IsOffsetKnown)
      return PI.setAborted(&I);

    // See if we already have computed info on this node.
    uint64_t &Size = PHIOrSelectSizes[&I];
    if (!Size) {
      // This is a new PHI/Select, check for an unsafe use of it.
      if (Instruction *UnsafeI = hasUnsafePHIOrSelectUse(&I, Size))
        return PI.setAborted(UnsafeI);
    }

    // For PHI and select operands outside the memory, we can't nuke the entire
    // phi or select -- the other side might still be relevant, so we special
    // case them here and use a separate structure to track the operands
    // themselves which should be replaced with poison.
    // FIXME: This should instead be escaped in the event we're instrumenting
    // for address sanitization.
    if (Offset.uge(AllocSize)) {
      AS.DeadOperands.push_back(U);
      return;
    }

    insertUse(I, Offset, Size);
  }

  void visitPHINode(PHINode &PN) { visitPHINodeOrSelectInst(PN); }

  void visitSelectInst(SelectInst &SI) { visitPHINodeOrSelectInst(SI); }

  /// Disable SROA entirely if there are unhandled users of the alloca.
  void visitInstruction(Instruction &I) { PI.setAborted(&I); }

  void visitCallBase(CallBase &CB) {
    // If the call operand is read-only and only does a read-only or address
    // capture, then we mark it as EscapedReadOnly.
    if (CB.isDataOperand(U) &&
        !capturesFullProvenance(CB.getCaptureInfo(U->getOperandNo())) &&
        CB.onlyReadsMemory(U->getOperandNo())) {
      PI.setEscapedReadOnly(&CB);
      return;
    }

    Base::visitCallBase(CB);
  }
};

MemSlices::MemSlices(const DataLayout &DL, AllocaInst &AI)
    :
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      AI(AI),
#endif
      PointerEscapingInstr(nullptr), PointerEscapingInstrReadOnly(nullptr) {
  SliceBuilder PB(DL, AI, *this);
  SliceBuilder::PtrInfo PtrI = PB.visitPtr(AI);
  if (PtrI.isEscaped() || PtrI.isAborted()) {
    // FIXME: We should sink the escape vs. abort info into the caller nicely,
    // possibly by just storing the PtrInfo in the MemSlices.
    PointerEscapingInstr = PtrI.getEscapingInst() ? PtrI.getEscapingInst()
                                                  : PtrI.getAbortingInst();
    assert(PointerEscapingInstr && "Did not track a bad instruction");
    return;
  }
  PointerEscapingInstrReadOnly = PtrI.getEscapedReadOnlyInst();

  llvm::erase_if(Slices, [](const MemSlice &S) { return S.isDead(); });

  // Sort the uses. This arranges for the offsets to be in ascending order,
  // and the sizes to be in descending order.
  llvm::stable_sort(Slices);
}

void MemSlices::partition_iterator::advance() {
  assert((P.SI != SE || !P.SplitTails.empty()) &&
         "Cannot advance past the end of the slices!");

  // Clear out any split uses which have ended.
  if (!P.SplitTails.empty()) {
    if (P.EndOffset >= MaxSplitSliceEndOffset) {
      // If we've finished all splits, this is easy.
      P.SplitTails.clear();
      MaxSplitSliceEndOffset = 0;
    } else {
      // Remove the uses which have ended in the prior partition. This
      // cannot change the max split slice end because we just checked that
      // the prior partition ended prior to that max.
      llvm::erase_if(P.SplitTails, [&](MemSlice *S) {
        return S->endOffset() <= P.EndOffset;
      });
      assert(llvm::any_of(P.SplitTails,
                          [&](MemSlice *S) {
                            return S->endOffset() == MaxSplitSliceEndOffset;
                          }) &&
             "Could not find the current max split slice offset!");
      assert(llvm::all_of(P.SplitTails,
                          [&](MemSlice *S) {
                            return S->endOffset() <= MaxSplitSliceEndOffset;
                          }) &&
             "Max split slice end offset is not actually the max!");
    }
  }

  // If P.SI is already at the end, then we've cleared the split tail and
  // now have an end iterator.
  if (P.SI == SE) {
    assert(P.SplitTails.empty() && "Failed to clear the split slices!");
    return;
  }

  // If we had a non-empty partition previously, set up the state for
  // subsequent partitions.
  if (P.SI != P.SJ) {
    // Accumulate all the splittable slices which started in the old
    // partition into the split list.
    for (MemSlice &S : P)
      if (S.isSplittable() && S.endOffset() > P.EndOffset) {
        P.SplitTails.push_back(&S);
        MaxSplitSliceEndOffset =
            std::max(S.endOffset(), MaxSplitSliceEndOffset);
      }

    // Start from the end of the previous partition.
    P.SI = P.SJ;

    // If P.SI is now at the end, we at most have a tail of split slices.
    if (P.SI == SE) {
      P.BeginOffset = P.EndOffset;
      P.EndOffset = MaxSplitSliceEndOffset;
      return;
    }

    // If the we have split slices and the next slice is after a gap and is
    // not splittable immediately form an empty partition for the split
    // slices up until the next slice begins.
    if (!P.SplitTails.empty() && P.SI->beginOffset() != P.EndOffset &&
        !P.SI->isSplittable()) {
      P.BeginOffset = P.EndOffset;
      P.EndOffset = P.SI->beginOffset();
      return;
    }
  }

  // OK, we need to consume new slices. Set the end offset based on the
  // current slice, and step SJ past it. The beginning offset of the
  // partition is the beginning offset of the next slice unless we have
  // pre-existing split slices.
  P.BeginOffset = P.SplitTails.empty() ? P.SI->beginOffset() : P.EndOffset;
  P.EndOffset = P.SI->endOffset();
  ++P.SJ;

  // There are two strategies to form a partition based on whether the
  // partition starts with an unsplittable slice or a splittable slice.
  if (!P.SI->isSplittable()) {
    // When we're forming an unsplittable region, it must always start at
    // the first slice and will extend through its end.
    assert(P.BeginOffset == P.SI->beginOffset());

    // Form a partition including all of the overlapping slices with this
    // unsplittable slice.
    while (P.SJ != SE && P.SJ->beginOffset() < P.EndOffset) {
      if (!P.SJ->isSplittable())
        P.EndOffset = std::max(P.EndOffset, P.SJ->endOffset());
      ++P.SJ;
    }

    // We have a partition across a set of overlapping unsplittable
    // partitions.
    return;
  }

  // If we're starting with a splittable slice, then we need to form
  // a synthetic partition spanning it and any other overlapping splittable
  // slices.
  assert(P.SI->isSplittable() && "Forming a splittable partition!");

  // Collect all of the overlapping splittable slices.
  while (P.SJ != SE && P.SJ->beginOffset() < P.EndOffset &&
         P.SJ->isSplittable()) {
    P.EndOffset = std::max(P.EndOffset, P.SJ->endOffset());
    ++P.SJ;
  }

  // Back upiEndOffset if we ended the span early when encountering an
  // unsplittable slice. This synthesizes the early end span of the partition.
  if (P.SJ != SE && P.SJ->beginOffset() < P.EndOffset) {
    assert(!P.SJ->isSplittable());
    P.EndOffset = P.SJ->beginOffset();
  }
}

/// Walk the range of a partition looking for a common type to cover this
/// sequence of slices.
std::pair<Type *, IntegerType *> MemPartition::findCommonType() const {
  Type *Ty = nullptr;
  bool TyIsCommon = true;
  IntegerType *ITy = nullptr;

  MemSlices::const_iterator B = this->begin();
  MemSlices::const_iterator E = this->end();
  uint64_t EndOffset = this->endOffset();

  // Note that we need to look at *every* slice's Use to ensure we
  // always get consistent results regardless of the order of slices.
  for (MemSlices::const_iterator I = B; I != E; ++I) {
    Use *U = I->getUse();
    if (isa<IntrinsicInst>(*U->getUser()))
      continue;
    if (I->beginOffset() != B->beginOffset() || I->endOffset() != EndOffset)
      continue;

    Type *UserTy = nullptr;
    if (LoadInst *LI = dyn_cast<LoadInst>(U->getUser())) {
      UserTy = LI->getType();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U->getUser())) {
      UserTy = SI->getValueOperand()->getType();
    }

    if (IntegerType *UserITy = dyn_cast_or_null<IntegerType>(UserTy)) {
      // If the type is larger than the partition, skip it. We only encounter
      // this for split integer operations where we want to use the type of the
      // entity causing the split. Also skip if the type is not a byte width
      // multiple.
      if (UserITy->getBitWidth() % 8 != 0 ||
          UserITy->getBitWidth() / 8 > (EndOffset - B->beginOffset()))
        continue;

      // Track the largest bitwidth integer type used in this way in case there
      // is no common type.
      if (!ITy || ITy->getBitWidth() < UserITy->getBitWidth())
        ITy = UserITy;
    }

    // To avoid depending on the order of slices, Ty and TyIsCommon must not
    // depend on types skipped above.
    if (!UserTy || (Ty && Ty != UserTy))
      TyIsCommon = false; // Give up on anything but an iN type.
    else
      Ty = UserTy;
  }

  return {TyIsCommon ? Ty : nullptr, ITy};
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

void MemSlices::print(raw_ostream &OS, const_iterator I,
                      StringRef Indent) const {
  printSlice(OS, I, Indent);
  OS << "\n";
  printUse(OS, I, Indent);
}

void MemSlices::printSlice(raw_ostream &OS, const_iterator I,
                           StringRef Indent) const {
  OS << Indent << "[" << I->beginOffset() << "," << I->endOffset() << ")"
     << " slice #" << (I - begin())
     << (I->isSplittable() ? " (splittable)" : "");
}

void MemSlices::printUse(raw_ostream &OS, const_iterator I,
                         StringRef Indent) const {
  OS << Indent << "  used by: " << *I->getUse()->getUser() << "\n";
}

void MemSlices::print(raw_ostream &OS) const {
  if (PointerEscapingInstr) {
    OS << "Can't analyze slices for base ptr: " << AI << "\n"
       << "  A pointer to this memory escaped by:\n"
       << "  " << *PointerEscapingInstr << "\n";
    return;
  }

  if (PointerEscapingInstrReadOnly)
    OS << "Escapes into ReadOnly: " << *PointerEscapingInstrReadOnly << "\n";

  OS << "Slices of base ptr: " << AI << "\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    print(OS, I);
}

LLVM_DUMP_METHOD void MemSlices::dump(const_iterator I) const {
  print(dbgs(), I);
}

LLVM_DUMP_METHOD void MemSlices::dump() const { print(dbgs()); }

#endif // !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
