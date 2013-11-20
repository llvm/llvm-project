//===- SROA.cpp - Scalar Replacement Of Aggregates ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This transformation implements the well known scalar replacement of
/// aggregates transformation. It tries to identify promotable elements of an
/// aggregate alloca, and promote them to registers. It will also try to
/// convert uses of an element (or set of elements) of an alloca into a vector
/// or bitfield-style integer scalar if appropriate.
///
/// It works to do this with minimal slicing of the alloca so that regions
/// which are merely transferred in and out of external memory remain unchanged
/// and are not decomposed to scalar code.
///
/// Because this also performs alloca promotion, it can be thought of as also
/// serving the purpose of SSA formation. The algorithm iterates on the
/// function until all opportunities for promotion have been realized.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sroa"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/DIBuilder.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

STATISTIC(NumAllocasAnalyzed, "Number of allocas analyzed for replacement");
STATISTIC(NumAllocaPartitions, "Number of alloca partitions formed");
STATISTIC(MaxPartitionsPerAlloca, "Maximum number of partitions per alloca");
STATISTIC(NumAllocaPartitionUses, "Number of alloca partition uses rewritten");
STATISTIC(MaxUsesPerAllocaPartition, "Maximum number of uses of a partition");
STATISTIC(NumNewAllocas, "Number of new, smaller allocas introduced");
STATISTIC(NumPromoted, "Number of allocas promoted to SSA values");
STATISTIC(NumLoadsSpeculated, "Number of loads speculated to allow promotion");
STATISTIC(NumDeleted, "Number of instructions deleted");
STATISTIC(NumVectorized, "Number of vectorized aggregates");

/// Hidden option to force the pass to not use DomTree and mem2reg, instead
/// forming SSA values through the SSAUpdater infrastructure.
static cl::opt<bool>
ForceSSAUpdater("force-ssa-updater", cl::init(false), cl::Hidden);

namespace {
/// \brief A custom IRBuilder inserter which prefixes all names if they are
/// preserved.
template <bool preserveNames = true>
class IRBuilderPrefixedInserter :
    public IRBuilderDefaultInserter<preserveNames> {
  std::string Prefix;

public:
  void SetNamePrefix(const Twine &P) { Prefix = P.str(); }

protected:
  void InsertHelper(Instruction *I, const Twine &Name, BasicBlock *BB,
                    BasicBlock::iterator InsertPt) const {
    IRBuilderDefaultInserter<preserveNames>::InsertHelper(
        I, Name.isTriviallyEmpty() ? Name : Prefix + Name, BB, InsertPt);
  }
};

// Specialization for not preserving the name is trivial.
template <>
class IRBuilderPrefixedInserter<false> :
    public IRBuilderDefaultInserter<false> {
public:
  void SetNamePrefix(const Twine &P) {}
};

/// \brief Provide a typedef for IRBuilder that drops names in release builds.
#ifndef NDEBUG
typedef llvm::IRBuilder<true, ConstantFolder,
                        IRBuilderPrefixedInserter<true> > IRBuilderTy;
#else
typedef llvm::IRBuilder<false, ConstantFolder,
                        IRBuilderPrefixedInserter<false> > IRBuilderTy;
#endif
}

namespace {
/// \brief A used slice of an alloca.
///
/// This structure represents a slice of an alloca used by some instruction. It
/// stores both the begin and end offsets of this use, a pointer to the use
/// itself, and a flag indicating whether we can classify the use as splittable
/// or not when forming partitions of the alloca.
class Slice {
  /// \brief The beginning offset of the range.
  uint64_t BeginOffset;

  /// \brief The ending offset, not included in the range.
  uint64_t EndOffset;

  /// \brief Storage for both the use of this slice and whether it can be
  /// split.
  PointerIntPair<Use *, 1, bool> UseAndIsSplittable;

public:
  Slice() : BeginOffset(), EndOffset() {}
  Slice(uint64_t BeginOffset, uint64_t EndOffset, Use *U, bool IsSplittable)
      : BeginOffset(BeginOffset), EndOffset(EndOffset),
        UseAndIsSplittable(U, IsSplittable) {}

  uint64_t beginOffset() const { return BeginOffset; }
  uint64_t endOffset() const { return EndOffset; }

  bool isSplittable() const { return UseAndIsSplittable.getInt(); }
  void makeUnsplittable() { UseAndIsSplittable.setInt(false); }

  Use *getUse() const { return UseAndIsSplittable.getPointer(); }

  bool isDead() const { return getUse() == 0; }
  void kill() { UseAndIsSplittable.setPointer(0); }

  /// \brief Support for ordering ranges.
  ///
  /// This provides an ordering over ranges such that start offsets are
  /// always increasing, and within equal start offsets, the end offsets are
  /// decreasing. Thus the spanning range comes first in a cluster with the
  /// same start position.
  bool operator<(const Slice &RHS) const {
    if (beginOffset() < RHS.beginOffset()) return true;
    if (beginOffset() > RHS.beginOffset()) return false;
    if (isSplittable() != RHS.isSplittable()) return !isSplittable();
    if (endOffset() > RHS.endOffset()) return true;
    return false;
  }

  /// \brief Support comparison with a single offset to allow binary searches.
  friend LLVM_ATTRIBUTE_UNUSED bool operator<(const Slice &LHS,
                                              uint64_t RHSOffset) {
    return LHS.beginOffset() < RHSOffset;
  }
  friend LLVM_ATTRIBUTE_UNUSED bool operator<(uint64_t LHSOffset,
                                              const Slice &RHS) {
    return LHSOffset < RHS.beginOffset();
  }

  bool operator==(const Slice &RHS) const {
    return isSplittable() == RHS.isSplittable() &&
           beginOffset() == RHS.beginOffset() && endOffset() == RHS.endOffset();
  }
  bool operator!=(const Slice &RHS) const { return !operator==(RHS); }
};
} // end anonymous namespace

namespace llvm {
template <typename T> struct isPodLike;
template <> struct isPodLike<Slice> {
   static const bool value = true;
};
}

namespace {
/// \brief Representation of the alloca slices.
///
/// This class represents the slices of an alloca which are formed by its
/// various uses. If a pointer escapes, we can't fully build a representation
/// for the slices used and we reflect that in this structure. The uses are
/// stored, sorted by increasing beginning offset and with unsplittable slices
/// starting at a particular offset before splittable slices.
class AllocaSlices {
public:
  /// \brief Construct the slices of a particular alloca.
  AllocaSlices(const DataLayout &DL, AllocaInst &AI);

  /// \brief Test whether a pointer to the allocation escapes our analysis.
  ///
  /// If this is true, the slices are never fully built and should be
  /// ignored.
  bool isEscaped() const { return PointerEscapingInstr; }

  /// \brief Support for iterating over the slices.
  /// @{
  typedef SmallVectorImpl<Slice>::iterator iterator;
  iterator begin() { return Slices.begin(); }
  iterator end() { return Slices.end(); }

  typedef SmallVectorImpl<Slice>::const_iterator const_iterator;
  const_iterator begin() const { return Slices.begin(); }
  const_iterator end() const { return Slices.end(); }
  /// @}

  /// \brief Allow iterating the dead users for this alloca.
  ///
  /// These are instructions which will never actually use the alloca as they
  /// are outside the allocated range. They are safe to replace with undef and
  /// delete.
  /// @{
  typedef SmallVectorImpl<Instruction *>::const_iterator dead_user_iterator;
  dead_user_iterator dead_user_begin() const { return DeadUsers.begin(); }
  dead_user_iterator dead_user_end() const { return DeadUsers.end(); }
  /// @}

  /// \brief Allow iterating the dead expressions referring to this alloca.
  ///
  /// These are operands which have cannot actually be used to refer to the
  /// alloca as they are outside its range and the user doesn't correct for
  /// that. These mostly consist of PHI node inputs and the like which we just
  /// need to replace with undef.
  /// @{
  typedef SmallVectorImpl<Use *>::const_iterator dead_op_iterator;
  dead_op_iterator dead_op_begin() const { return DeadOperands.begin(); }
  dead_op_iterator dead_op_end() const { return DeadOperands.end(); }
  /// @}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &OS, const_iterator I, StringRef Indent = "  ") const;
  void printSlice(raw_ostream &OS, const_iterator I,
                  StringRef Indent = "  ") const;
  void printUse(raw_ostream &OS, const_iterator I,
                StringRef Indent = "  ") const;
  void print(raw_ostream &OS) const;
  void LLVM_ATTRIBUTE_NOINLINE LLVM_ATTRIBUTE_USED dump(const_iterator I) const;
  void LLVM_ATTRIBUTE_NOINLINE LLVM_ATTRIBUTE_USED dump() const;
#endif

private:
  template <typename DerivedT, typename RetT = void> class BuilderBase;
  class SliceBuilder;
  friend class AllocaSlices::SliceBuilder;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// \brief Handle to alloca instruction to simplify method interfaces.
  AllocaInst &AI;
#endif

  /// \brief The instruction responsible for this alloca not having a known set
  /// of slices.
  ///
  /// When an instruction (potentially) escapes the pointer to the alloca, we
  /// store a pointer to that here and abort trying to form slices of the
  /// alloca. This will be null if the alloca slices are analyzed successfully.
  Instruction *PointerEscapingInstr;

  /// \brief The slices of the alloca.
  ///
  /// We store a vector of the slices formed by uses of the alloca here. This
  /// vector is sorted by increasing begin offset, and then the unsplittable
  /// slices before the splittable ones. See the Slice inner class for more
  /// details.
  SmallVector<Slice, 8> Slices;

  /// \brief Instructions which will become dead if we rewrite the alloca.
  ///
  /// Note that these are not separated by slice. This is because we expect an
  /// alloca to be completely rewritten or not rewritten at all. If rewritten,
  /// all these instructions can simply be removed and replaced with undef as
  /// they come from outside of the allocated space.
  SmallVector<Instruction *, 8> DeadUsers;

  /// \brief Operands which will become dead if we rewrite the alloca.
  ///
  /// These are operands that in their particular use can be replaced with
  /// undef when we rewrite the alloca. These show up in out-of-bounds inputs
  /// to PHI nodes and the like. They aren't entirely dead (there might be
  /// a GEP back into the bounds using it elsewhere) and nor is the PHI, but we
  /// want to swap this particular input for undef to simplify the use lists of
  /// the alloca.
  SmallVector<Use *, 8> DeadOperands;
};
}

static Value *foldSelectInst(SelectInst &SI) {
  // If the condition being selected on is a constant or the same value is
  // being selected between, fold the select. Yes this does (rarely) happen
  // early on.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(SI.getCondition()))
    return SI.getOperand(1+CI->isZero());
  if (SI.getOperand(1) == SI.getOperand(2))
    return SI.getOperand(1);

  return 0;
}

/// \brief Builder for the alloca slices.
///
/// This class builds a set of alloca slices by recursively visiting the uses
/// of an alloca and making a slice for each load and store at each offset.
class AllocaSlices::SliceBuilder : public PtrUseVisitor<SliceBuilder> {
  friend class PtrUseVisitor<SliceBuilder>;
  friend class InstVisitor<SliceBuilder>;
  typedef PtrUseVisitor<SliceBuilder> Base;

  const uint64_t AllocSize;
  AllocaSlices &S;

  SmallDenseMap<Instruction *, unsigned> MemTransferSliceMap;
  SmallDenseMap<Instruction *, uint64_t> PHIOrSelectSizes;

  /// \brief Set to de-duplicate dead instructions found in the use walk.
  SmallPtrSet<Instruction *, 4> VisitedDeadInsts;

public:
  SliceBuilder(const DataLayout &DL, AllocaInst &AI, AllocaSlices &S)
      : PtrUseVisitor<SliceBuilder>(DL),
        AllocSize(DL.getTypeAllocSize(AI.getAllocatedType())), S(S) {}

private:
  void markAsDead(Instruction &I) {
    if (VisitedDeadInsts.insert(&I))
      S.DeadUsers.push_back(&I);
  }

  void insertUse(Instruction &I, const APInt &Offset, uint64_t Size,
                 bool IsSplittable = false) {
    // Completely skip uses which have a zero size or start either before or
    // past the end of the allocation.
    if (Size == 0 || Offset.isNegative() || Offset.uge(AllocSize)) {
      DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte use @" << Offset
                   << " which has zero size or starts outside of the "
                   << AllocSize << " byte alloca:\n"
                   << "    alloca: " << S.AI << "\n"
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
      DEBUG(dbgs() << "WARNING: Clamping a " << Size << " byte use @" << Offset
                   << " to remain within the " << AllocSize << " byte alloca:\n"
                   << "    alloca: " << S.AI << "\n"
                   << "       use: " << I << "\n");
      EndOffset = AllocSize;
    }

    S.Slices.push_back(Slice(BeginOffset, EndOffset, U, IsSplittable));
  }

  void visitBitCastInst(BitCastInst &BC) {
    if (BC.use_empty())
      return markAsDead(BC);

    return Base::visitBitCastInst(BC);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    if (GEPI.use_empty())
      return markAsDead(GEPI);

    return Base::visitGetElementPtrInst(GEPI);
  }

  void handleLoadOrStore(Type *Ty, Instruction &I, const APInt &Offset,
                         uint64_t Size, bool IsVolatile) {
    // We allow splitting of loads and stores where the type is an integer type
    // and cover the entire alloca. This prevents us from splitting over
    // eagerly.
    // FIXME: In the great blue eventually, we should eagerly split all integer
    // loads and stores, and then have a separate step that merges adjacent
    // alloca partitions into a single partition suitable for integer widening.
    // Or we should skip the merge step and rely on GVN and other passes to
    // merge adjacent loads and stores that survive mem2reg.
    bool IsSplittable =
        Ty->isIntegerTy() && !IsVolatile && Offset == 0 && Size >= AllocSize;

    insertUse(I, Offset, Size, IsSplittable);
  }

  void visitLoadInst(LoadInst &LI) {
    assert((!LI.isSimple() || LI.getType()->isSingleValueType()) &&
           "All simple FCA loads should have been pre-split");

    if (!IsOffsetKnown)
      return PI.setAborted(&LI);

    uint64_t Size = DL.getTypeStoreSize(LI.getType());
    return handleLoadOrStore(LI.getType(), LI, Offset, Size, LI.isVolatile());
  }

  void visitStoreInst(StoreInst &SI) {
    Value *ValOp = SI.getValueOperand();
    if (ValOp == *U)
      return PI.setEscapedAndAborted(&SI);
    if (!IsOffsetKnown)
      return PI.setAborted(&SI);

    uint64_t Size = DL.getTypeStoreSize(ValOp->getType());

    // If this memory access can be shown to *statically* extend outside the
    // bounds of of the allocation, it's behavior is undefined, so simply
    // ignore it. Note that this is more strict than the generic clamping
    // behavior of insertUse. We also try to handle cases which might run the
    // risk of overflow.
    // FIXME: We should instead consider the pointer to have escaped if this
    // function is being instrumented for addressing bugs or race conditions.
    if (Offset.isNegative() || Size > AllocSize ||
        Offset.ugt(AllocSize - Size)) {
      DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte store @" << Offset
                   << " which extends past the end of the " << AllocSize
                   << " byte alloca:\n"
                   << "    alloca: " << S.AI << "\n"
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
        (IsOffsetKnown && !Offset.isNegative() && Offset.uge(AllocSize)))
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
    if ((Length && Length->getValue() == 0) ||
        (IsOffsetKnown && !Offset.isNegative() && Offset.uge(AllocSize)))
      // Zero-length mem transfer intrinsics can be ignored entirely.
      return markAsDead(II);

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    uint64_t RawOffset = Offset.getLimitedValue();
    uint64_t Size = Length ? Length->getLimitedValue()
                           : AllocSize - RawOffset;

    // Check for the special case where the same exact value is used for both
    // source and dest.
    if (*U == II.getRawDest() && *U == II.getRawSource()) {
      // For non-volatile transfers this is a no-op.
      if (!II.isVolatile())
        return markAsDead(II);

      return insertUse(II, Offset, Size, /*IsSplittable=*/false);
    }

    // If we have seen both source and destination for a mem transfer, then
    // they both point to the same alloca.
    bool Inserted;
    SmallDenseMap<Instruction *, unsigned>::iterator MTPI;
    llvm::tie(MTPI, Inserted) =
        MemTransferSliceMap.insert(std::make_pair(&II, S.Slices.size()));
    unsigned PrevIdx = MTPI->second;
    if (!Inserted) {
      Slice &PrevP = S.Slices[PrevIdx];

      // Check if the begin offsets match and this is a non-volatile transfer.
      // In that case, we can completely elide the transfer.
      if (!II.isVolatile() && PrevP.beginOffset() == RawOffset) {
        PrevP.kill();
        return markAsDead(II);
      }

      // Otherwise we have an offset transfer within the same alloca. We can't
      // split those.
      PrevP.makeUnsplittable();
    }

    // Insert the use now that we've fixed up the splittable nature.
    insertUse(II, Offset, Size, /*IsSplittable=*/Inserted && Length);

    // Check that we ended up with a valid index in the map.
    assert(S.Slices[PrevIdx].getUse()->getUser() == &II &&
           "Map index doesn't point back to a slice with this user.");
  }

  // Disable SRoA for any intrinsics except for lifetime invariants.
  // FIXME: What about debug intrinsics? This matches old behavior, but
  // doesn't make sense.
  void visitIntrinsicInst(IntrinsicInst &II) {
    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    if (II.getIntrinsicID() == Intrinsic::lifetime_start ||
        II.getIntrinsicID() == Intrinsic::lifetime_end) {
      ConstantInt *Length = cast<ConstantInt>(II.getArgOperand(0));
      uint64_t Size = std::min(AllocSize - Offset.getLimitedValue(),
                               Length->getLimitedValue());
      insertUse(II, Offset, Size, true);
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
    // If there are no loads or stores, the access is dead. We mark that as
    // a size zero access.
    Size = 0;
    do {
      Instruction *I, *UsedI;
      llvm::tie(UsedI, I) = Uses.pop_back_val();

      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        Size = std::max(Size, DL.getTypeStoreSize(LI->getType()));
        continue;
      }
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        Value *Op = SI->getOperand(0);
        if (Op == UsedI)
          return SI;
        Size = std::max(Size, DL.getTypeStoreSize(Op->getType()));
        continue;
      }

      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
        if (!GEP->hasAllZeroIndices())
          return GEP;
      } else if (!isa<BitCastInst>(I) && !isa<PHINode>(I) &&
                 !isa<SelectInst>(I)) {
        return I;
      }

      for (Value::use_iterator UI = I->use_begin(), UE = I->use_end(); UI != UE;
           ++UI)
        if (Visited.insert(cast<Instruction>(*UI)))
          Uses.push_back(std::make_pair(I, cast<Instruction>(*UI)));
    } while (!Uses.empty());

    return 0;
  }

  void visitPHINode(PHINode &PN) {
    if (PN.use_empty())
      return markAsDead(PN);
    if (!IsOffsetKnown)
      return PI.setAborted(&PN);

    // See if we already have computed info on this node.
    uint64_t &PHISize = PHIOrSelectSizes[&PN];
    if (!PHISize) {
      // This is a new PHI node, check for an unsafe use of the PHI node.
      if (Instruction *UnsafeI = hasUnsafePHIOrSelectUse(&PN, PHISize))
        return PI.setAborted(UnsafeI);
    }

    // For PHI and select operands outside the alloca, we can't nuke the entire
    // phi or select -- the other side might still be relevant, so we special
    // case them here and use a separate structure to track the operands
    // themselves which should be replaced with undef.
    // FIXME: This should instead be escaped in the event we're instrumenting
    // for address sanitization.
    if ((Offset.isNegative() && (-Offset).uge(PHISize)) ||
        (!Offset.isNegative() && Offset.uge(AllocSize))) {
      S.DeadOperands.push_back(U);
      return;
    }

    insertUse(PN, Offset, PHISize);
  }

  void visitSelectInst(SelectInst &SI) {
    if (SI.use_empty())
      return markAsDead(SI);
    if (Value *Result = foldSelectInst(SI)) {
      if (Result == *U)
        // If the result of the constant fold will be the pointer, recurse
        // through the select as if we had RAUW'ed it.
        enqueueUsers(SI);
      else
        // Otherwise the operand to the select is dead, and we can replace it
        // with undef.
        S.DeadOperands.push_back(U);

      return;
    }
    if (!IsOffsetKnown)
      return PI.setAborted(&SI);

    // See if we already have computed info on this node.
    uint64_t &SelectSize = PHIOrSelectSizes[&SI];
    if (!SelectSize) {
      // This is a new Select, check for an unsafe use of it.
      if (Instruction *UnsafeI = hasUnsafePHIOrSelectUse(&SI, SelectSize))
        return PI.setAborted(UnsafeI);
    }

    // For PHI and select operands outside the alloca, we can't nuke the entire
    // phi or select -- the other side might still be relevant, so we special
    // case them here and use a separate structure to track the operands
    // themselves which should be replaced with undef.
    // FIXME: This should instead be escaped in the event we're instrumenting
    // for address sanitization.
    if ((Offset.isNegative() && Offset.uge(SelectSize)) ||
        (!Offset.isNegative() && Offset.uge(AllocSize))) {
      S.DeadOperands.push_back(U);
      return;
    }

    insertUse(SI, Offset, SelectSize);
  }

  /// \brief Disable SROA entirely if there are unhandled users of the alloca.
  void visitInstruction(Instruction &I) {
    PI.setAborted(&I);
  }
};

AllocaSlices::AllocaSlices(const DataLayout &DL, AllocaInst &AI)
    :
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      AI(AI),
#endif
      PointerEscapingInstr(0) {
  SliceBuilder PB(DL, AI, *this);
  SliceBuilder::PtrInfo PtrI = PB.visitPtr(AI);
  if (PtrI.isEscaped() || PtrI.isAborted()) {
    // FIXME: We should sink the escape vs. abort info into the caller nicely,
    // possibly by just storing the PtrInfo in the AllocaSlices.
    PointerEscapingInstr = PtrI.getEscapingInst() ? PtrI.getEscapingInst()
                                                  : PtrI.getAbortingInst();
    assert(PointerEscapingInstr && "Did not track a bad instruction");
    return;
  }

  Slices.erase(std::remove_if(Slices.begin(), Slices.end(),
                              std::mem_fun_ref(&Slice::isDead)),
               Slices.end());

  // Sort the uses. This arranges for the offsets to be in ascending order,
  // and the sizes to be in descending order.
  std::sort(Slices.begin(), Slices.end());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

void AllocaSlices::print(raw_ostream &OS, const_iterator I,
                         StringRef Indent) const {
  printSlice(OS, I, Indent);
  printUse(OS, I, Indent);
}

void AllocaSlices::printSlice(raw_ostream &OS, const_iterator I,
                              StringRef Indent) const {
  OS << Indent << "[" << I->beginOffset() << "," << I->endOffset() << ")"
     << " slice #" << (I - begin())
     << (I->isSplittable() ? " (splittable)" : "") << "\n";
}

void AllocaSlices::printUse(raw_ostream &OS, const_iterator I,
                            StringRef Indent) const {
  OS << Indent << "  used by: " << *I->getUse()->getUser() << "\n";
}

void AllocaSlices::print(raw_ostream &OS) const {
  if (PointerEscapingInstr) {
    OS << "Can't analyze slices for alloca: " << AI << "\n"
       << "  A pointer to this alloca escaped by:\n"
       << "  " << *PointerEscapingInstr << "\n";
    return;
  }

  OS << "Slices of alloca: " << AI << "\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    print(OS, I);
}

void AllocaSlices::dump(const_iterator I) const { print(dbgs(), I); }
void AllocaSlices::dump() const { print(dbgs()); }

#endif // !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

namespace {
/// \brief Implementation of LoadAndStorePromoter for promoting allocas.
///
/// This subclass of LoadAndStorePromoter adds overrides to handle promoting
/// the loads and stores of an alloca instruction, as well as updating its
/// debug information. This is used when a domtree is unavailable and thus
/// mem2reg in its full form can't be used to handle promotion of allocas to
/// scalar values.
class AllocaPromoter : public LoadAndStorePromoter {
  AllocaInst &AI;
  DIBuilder &DIB;

  SmallVector<DbgDeclareInst *, 4> DDIs;
  SmallVector<DbgValueInst *, 4> DVIs;

public:
  AllocaPromoter(const SmallVectorImpl<Instruction *> &Insts, SSAUpdater &S,
                 AllocaInst &AI, DIBuilder &DIB)
      : LoadAndStorePromoter(Insts, S), AI(AI), DIB(DIB) {}

  void run(const SmallVectorImpl<Instruction*> &Insts) {
    // Retain the debug information attached to the alloca for use when
    // rewriting loads and stores.
    if (MDNode *DebugNode = MDNode::getIfExists(AI.getContext(), &AI)) {
      for (Value::use_iterator UI = DebugNode->use_begin(),
                               UE = DebugNode->use_end();
           UI != UE; ++UI)
        if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(*UI))
          DDIs.push_back(DDI);
        else if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(*UI))
          DVIs.push_back(DVI);
    }

    LoadAndStorePromoter::run(Insts);

    // While we have the debug information, clear it off of the alloca. The
    // caller takes care of deleting the alloca.
    while (!DDIs.empty())
      DDIs.pop_back_val()->eraseFromParent();
    while (!DVIs.empty())
      DVIs.pop_back_val()->eraseFromParent();
  }

  virtual bool isInstInList(Instruction *I,
                            const SmallVectorImpl<Instruction*> &Insts) const {
    Value *Ptr;
    if (LoadInst *LI = dyn_cast<LoadInst>(I))
      Ptr = LI->getOperand(0);
    else
      Ptr = cast<StoreInst>(I)->getPointerOperand();

    // Only used to detect cycles, which will be rare and quickly found as
    // we're walking up a chain of defs rather than down through uses.
    SmallPtrSet<Value *, 4> Visited;

    do {
      if (Ptr == &AI)
        return true;

      if (BitCastInst *BCI = dyn_cast<BitCastInst>(Ptr))
        Ptr = BCI->getOperand(0);
      else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Ptr))
        Ptr = GEPI->getPointerOperand();
      else
        return false;

    } while (Visited.insert(Ptr));

    return false;
  }

  virtual void updateDebugInfo(Instruction *Inst) const {
    for (SmallVectorImpl<DbgDeclareInst *>::const_iterator I = DDIs.begin(),
           E = DDIs.end(); I != E; ++I) {
      DbgDeclareInst *DDI = *I;
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
        ConvertDebugDeclareToDebugValue(DDI, SI, DIB);
      else if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
        ConvertDebugDeclareToDebugValue(DDI, LI, DIB);
    }
    for (SmallVectorImpl<DbgValueInst *>::const_iterator I = DVIs.begin(),
           E = DVIs.end(); I != E; ++I) {
      DbgValueInst *DVI = *I;
      Value *Arg = 0;
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        // If an argument is zero extended then use argument directly. The ZExt
        // may be zapped by an optimization pass in future.
        if (ZExtInst *ZExt = dyn_cast<ZExtInst>(SI->getOperand(0)))
          Arg = dyn_cast<Argument>(ZExt->getOperand(0));
        else if (SExtInst *SExt = dyn_cast<SExtInst>(SI->getOperand(0)))
          Arg = dyn_cast<Argument>(SExt->getOperand(0));
        if (!Arg)
          Arg = SI->getValueOperand();
      } else if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        Arg = LI->getPointerOperand();
      } else {
        continue;
      }
      Instruction *DbgVal =
        DIB.insertDbgValueIntrinsic(Arg, 0, DIVariable(DVI->getVariable()),
                                     Inst);
      DbgVal->setDebugLoc(DVI->getDebugLoc());
    }
  }
};
} // end anon namespace


namespace {
/// \brief An optimization pass providing Scalar Replacement of Aggregates.
///
/// This pass takes allocations which can be completely analyzed (that is, they
/// don't escape) and tries to turn them into scalar SSA values. There are
/// a few steps to this process.
///
/// 1) It takes allocations of aggregates and analyzes the ways in which they
///    are used to try to split them into smaller allocations, ideally of
///    a single scalar data type. It will split up memcpy and memset accesses
///    as necessary and try to isolate individual scalar accesses.
/// 2) It will transform accesses into forms which are suitable for SSA value
///    promotion. This can be replacing a memset with a scalar store of an
///    integer value, or it can involve speculating operations on a PHI or
///    select to be a PHI or select of the results.
/// 3) Finally, this will try to detect a pattern of accesses which map cleanly
///    onto insert and extract operations on a vector value, and convert them to
///    this form. By doing so, it will enable promotion of vector aggregates to
///    SSA vector values.
class SROA : public FunctionPass {
  const bool RequiresDomTree;

  LLVMContext *C;
  const DataLayout *DL;
  DominatorTree *DT;

  /// \brief Worklist of alloca instructions to simplify.
  ///
  /// Each alloca in the function is added to this. Each new alloca formed gets
  /// added to it as well to recursively simplify unless that alloca can be
  /// directly promoted. Finally, each time we rewrite a use of an alloca other
  /// the one being actively rewritten, we add it back onto the list if not
  /// already present to ensure it is re-visited.
  SetVector<AllocaInst *, SmallVector<AllocaInst *, 16> > Worklist;

  /// \brief A collection of instructions to delete.
  /// We try to batch deletions to simplify code and make things a bit more
  /// efficient.
  SetVector<Instruction *, SmallVector<Instruction *, 8> > DeadInsts;

  /// \brief Post-promotion worklist.
  ///
  /// Sometimes we discover an alloca which has a high probability of becoming
  /// viable for SROA after a round of promotion takes place. In those cases,
  /// the alloca is enqueued here for re-processing.
  ///
  /// Note that we have to be very careful to clear allocas out of this list in
  /// the event they are deleted.
  SetVector<AllocaInst *, SmallVector<AllocaInst *, 16> > PostPromotionWorklist;

  /// \brief A collection of alloca instructions we can directly promote.
  std::vector<AllocaInst *> PromotableAllocas;

  /// \brief A worklist of PHIs to speculate prior to promoting allocas.
  ///
  /// All of these PHIs have been checked for the safety of speculation and by
  /// being speculated will allow promoting allocas currently in the promotable
  /// queue.
  SetVector<PHINode *, SmallVector<PHINode *, 2> > SpeculatablePHIs;

  /// \brief A worklist of select instructions to speculate prior to promoting
  /// allocas.
  ///
  /// All of these select instructions have been checked for the safety of
  /// speculation and by being speculated will allow promoting allocas
  /// currently in the promotable queue.
  SetVector<SelectInst *, SmallVector<SelectInst *, 2> > SpeculatableSelects;

public:
  SROA(bool RequiresDomTree = true)
      : FunctionPass(ID), RequiresDomTree(RequiresDomTree),
        C(0), DL(0), DT(0) {
    initializeSROAPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F);
  void getAnalysisUsage(AnalysisUsage &AU) const;

  const char *getPassName() const { return "SROA"; }
  static char ID;

private:
  friend class PHIOrSelectSpeculator;
  friend class AllocaSliceRewriter;

  bool rewritePartition(AllocaInst &AI, AllocaSlices &S,
                        AllocaSlices::iterator B, AllocaSlices::iterator E,
                        int64_t BeginOffset, int64_t EndOffset,
                        ArrayRef<AllocaSlices::iterator> SplitUses);
  bool splitAlloca(AllocaInst &AI, AllocaSlices &S);
  bool runOnAlloca(AllocaInst &AI);
  void deleteDeadInstructions(SmallPtrSet<AllocaInst *, 4> &DeletedAllocas);
  bool promoteAllocas(Function &F);
};
}

char SROA::ID = 0;

FunctionPass *llvm::createSROAPass(bool RequiresDomTree) {
  return new SROA(RequiresDomTree);
}

INITIALIZE_PASS_BEGIN(SROA, "sroa", "Scalar Replacement Of Aggregates",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(SROA, "sroa", "Scalar Replacement Of Aggregates",
                    false, false)

/// Walk the range of a partitioning looking for a common type to cover this
/// sequence of slices.
static Type *findCommonType(AllocaSlices::const_iterator B,
                            AllocaSlices::const_iterator E,
                            uint64_t EndOffset) {
  Type *Ty = 0;
  bool IgnoreNonIntegralTypes = false;
  for (AllocaSlices::const_iterator I = B; I != E; ++I) {
    Use *U = I->getUse();
    if (isa<IntrinsicInst>(*U->getUser()))
      continue;
    if (I->beginOffset() != B->beginOffset() || I->endOffset() != EndOffset)
      continue;

    Type *UserTy = 0;
    if (LoadInst *LI = dyn_cast<LoadInst>(U->getUser())) {
      UserTy = LI->getType();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U->getUser())) {
      UserTy = SI->getValueOperand()->getType();
    } else {
      IgnoreNonIntegralTypes = true; // Give up on anything but an iN type.
      continue;
    }

    if (IntegerType *ITy = dyn_cast<IntegerType>(UserTy)) {
      // If the type is larger than the partition, skip it. We only encounter
      // this for split integer operations where we want to use the type of the
      // entity causing the split. Also skip if the type is not a byte width
      // multiple.
      if (ITy->getBitWidth() % 8 != 0 ||
          ITy->getBitWidth() / 8 > (EndOffset - B->beginOffset()))
        continue;

      // If we have found an integer type use covering the alloca, use that
      // regardless of the other types, as integers are often used for
      // a "bucket of bits" type.
      //
      // NB: This *must* be the only return from inside the loop so that the
      // order of slices doesn't impact the computed type.
      return ITy;
    } else if (IgnoreNonIntegralTypes) {
      continue;
    }

    if (Ty && Ty != UserTy)
      IgnoreNonIntegralTypes = true; // Give up on anything but an iN type.

    Ty = UserTy;
  }
  return Ty;
}

/// PHI instructions that use an alloca and are subsequently loaded can be
/// rewritten to load both input pointers in the pred blocks and then PHI the
/// results, allowing the load of the alloca to be promoted.
/// From this:
///   %P2 = phi [i32* %Alloca, i32* %Other]
///   %V = load i32* %P2
/// to:
///   %V1 = load i32* %Alloca      -> will be mem2reg'd
///   ...
///   %V2 = load i32* %Other
///   ...
///   %V = phi [i32 %V1, i32 %V2]
///
/// We can do this to a select if its only uses are loads and if the operands
/// to the select can be loaded unconditionally.
///
/// FIXME: This should be hoisted into a generic utility, likely in
/// Transforms/Util/Local.h
static bool isSafePHIToSpeculate(PHINode &PN,
                                 const DataLayout *DL = 0) {
  // For now, we can only do this promotion if the load is in the same block
  // as the PHI, and if there are no stores between the phi and load.
  // TODO: Allow recursive phi users.
  // TODO: Allow stores.
  BasicBlock *BB = PN.getParent();
  unsigned MaxAlign = 0;
  bool HaveLoad = false;
  for (Value::use_iterator UI = PN.use_begin(), UE = PN.use_end(); UI != UE;
       ++UI) {
    LoadInst *LI = dyn_cast<LoadInst>(*UI);
    if (LI == 0 || !LI->isSimple())
      return false;

    // For now we only allow loads in the same block as the PHI.  This is
    // a common case that happens when instcombine merges two loads through
    // a PHI.
    if (LI->getParent() != BB)
      return false;

    // Ensure that there are no instructions between the PHI and the load that
    // could store.
    for (BasicBlock::iterator BBI = &PN; &*BBI != LI; ++BBI)
      if (BBI->mayWriteToMemory())
        return false;

    MaxAlign = std::max(MaxAlign, LI->getAlignment());
    HaveLoad = true;
  }

  if (!HaveLoad)
    return false;

  // We can only transform this if it is safe to push the loads into the
  // predecessor blocks. The only thing to watch out for is that we can't put
  // a possibly trapping load in the predecessor if it is a critical edge.
  for (unsigned Idx = 0, Num = PN.getNumIncomingValues(); Idx != Num; ++Idx) {
    TerminatorInst *TI = PN.getIncomingBlock(Idx)->getTerminator();
    Value *InVal = PN.getIncomingValue(Idx);

    // If the value is produced by the terminator of the predecessor (an
    // invoke) or it has side-effects, there is no valid place to put a load
    // in the predecessor.
    if (TI == InVal || TI->mayHaveSideEffects())
      return false;

    // If the predecessor has a single successor, then the edge isn't
    // critical.
    if (TI->getNumSuccessors() == 1)
      continue;

    // If this pointer is always safe to load, or if we can prove that there
    // is already a load in the block, then we can move the load to the pred
    // block.
    if (InVal->isDereferenceablePointer() ||
        isSafeToLoadUnconditionally(InVal, TI, MaxAlign, DL))
      continue;

    return false;
  }

  return true;
}

static void speculatePHINodeLoads(PHINode &PN) {
  DEBUG(dbgs() << "    original: " << PN << "\n");

  Type *LoadTy = cast<PointerType>(PN.getType())->getElementType();
  IRBuilderTy PHIBuilder(&PN);
  PHINode *NewPN = PHIBuilder.CreatePHI(LoadTy, PN.getNumIncomingValues(),
                                        PN.getName() + ".sroa.speculated");

  // Get the TBAA tag and alignment to use from one of the loads.  It doesn't
  // matter which one we get and if any differ.
  LoadInst *SomeLoad = cast<LoadInst>(*PN.use_begin());
  MDNode *TBAATag = SomeLoad->getMetadata(LLVMContext::MD_tbaa);
  unsigned Align = SomeLoad->getAlignment();

  // Rewrite all loads of the PN to use the new PHI.
  while (!PN.use_empty()) {
    LoadInst *LI = cast<LoadInst>(*PN.use_begin());
    LI->replaceAllUsesWith(NewPN);
    LI->eraseFromParent();
  }

  // Inject loads into all of the pred blocks.
  for (unsigned Idx = 0, Num = PN.getNumIncomingValues(); Idx != Num; ++Idx) {
    BasicBlock *Pred = PN.getIncomingBlock(Idx);
    TerminatorInst *TI = Pred->getTerminator();
    Value *InVal = PN.getIncomingValue(Idx);
    IRBuilderTy PredBuilder(TI);

    LoadInst *Load = PredBuilder.CreateLoad(
        InVal, (PN.getName() + ".sroa.speculate.load." + Pred->getName()));
    ++NumLoadsSpeculated;
    Load->setAlignment(Align);
    if (TBAATag)
      Load->setMetadata(LLVMContext::MD_tbaa, TBAATag);
    NewPN->addIncoming(Load, Pred);
  }

  DEBUG(dbgs() << "          speculated to: " << *NewPN << "\n");
  PN.eraseFromParent();
}

/// Select instructions that use an alloca and are subsequently loaded can be
/// rewritten to load both input pointers and then select between the result,
/// allowing the load of the alloca to be promoted.
/// From this:
///   %P2 = select i1 %cond, i32* %Alloca, i32* %Other
///   %V = load i32* %P2
/// to:
///   %V1 = load i32* %Alloca      -> will be mem2reg'd
///   %V2 = load i32* %Other
///   %V = select i1 %cond, i32 %V1, i32 %V2
///
/// We can do this to a select if its only uses are loads and if the operand
/// to the select can be loaded unconditionally.
static bool isSafeSelectToSpeculate(SelectInst &SI, const DataLayout *DL = 0) {
  Value *TValue = SI.getTrueValue();
  Value *FValue = SI.getFalseValue();
  bool TDerefable = TValue->isDereferenceablePointer();
  bool FDerefable = FValue->isDereferenceablePointer();

  for (Value::use_iterator UI = SI.use_begin(), UE = SI.use_end(); UI != UE;
       ++UI) {
    LoadInst *LI = dyn_cast<LoadInst>(*UI);
    if (LI == 0 || !LI->isSimple())
      return false;

    // Both operands to the select need to be dereferencable, either
    // absolutely (e.g. allocas) or at this point because we can see other
    // accesses to it.
    if (!TDerefable &&
        !isSafeToLoadUnconditionally(TValue, LI, LI->getAlignment(), DL))
      return false;
    if (!FDerefable &&
        !isSafeToLoadUnconditionally(FValue, LI, LI->getAlignment(), DL))
      return false;
  }

  return true;
}

static void speculateSelectInstLoads(SelectInst &SI) {
  DEBUG(dbgs() << "    original: " << SI << "\n");

  IRBuilderTy IRB(&SI);
  Value *TV = SI.getTrueValue();
  Value *FV = SI.getFalseValue();
  // Replace the loads of the select with a select of two loads.
  while (!SI.use_empty()) {
    LoadInst *LI = cast<LoadInst>(*SI.use_begin());
    assert(LI->isSimple() && "We only speculate simple loads");

    IRB.SetInsertPoint(LI);
    LoadInst *TL =
        IRB.CreateLoad(TV, LI->getName() + ".sroa.speculate.load.true");
    LoadInst *FL =
        IRB.CreateLoad(FV, LI->getName() + ".sroa.speculate.load.false");
    NumLoadsSpeculated += 2;

    // Transfer alignment and TBAA info if present.
    TL->setAlignment(LI->getAlignment());
    FL->setAlignment(LI->getAlignment());
    if (MDNode *Tag = LI->getMetadata(LLVMContext::MD_tbaa)) {
      TL->setMetadata(LLVMContext::MD_tbaa, Tag);
      FL->setMetadata(LLVMContext::MD_tbaa, Tag);
    }

    Value *V = IRB.CreateSelect(SI.getCondition(), TL, FL,
                                LI->getName() + ".sroa.speculated");

    DEBUG(dbgs() << "          speculated to: " << *V << "\n");
    LI->replaceAllUsesWith(V);
    LI->eraseFromParent();
  }
  SI.eraseFromParent();
}

/// \brief Build a GEP out of a base pointer and indices.
///
/// This will return the BasePtr if that is valid, or build a new GEP
/// instruction using the IRBuilder if GEP-ing is needed.
static Value *buildGEP(IRBuilderTy &IRB, Value *BasePtr,
                       SmallVectorImpl<Value *> &Indices) {
  if (Indices.empty())
    return BasePtr;

  // A single zero index is a no-op, so check for this and avoid building a GEP
  // in that case.
  if (Indices.size() == 1 && cast<ConstantInt>(Indices.back())->isZero())
    return BasePtr;

  return IRB.CreateInBoundsGEP(BasePtr, Indices, "idx");
}

/// \brief Get a natural GEP off of the BasePtr walking through Ty toward
/// TargetTy without changing the offset of the pointer.
///
/// This routine assumes we've already established a properly offset GEP with
/// Indices, and arrived at the Ty type. The goal is to continue to GEP with
/// zero-indices down through type layers until we find one the same as
/// TargetTy. If we can't find one with the same type, we at least try to use
/// one with the same size. If none of that works, we just produce the GEP as
/// indicated by Indices to have the correct offset.
static Value *getNaturalGEPWithType(IRBuilderTy &IRB, const DataLayout &DL,
                                    Value *BasePtr, Type *Ty, Type *TargetTy,
                                    SmallVectorImpl<Value *> &Indices) {
  if (Ty == TargetTy)
    return buildGEP(IRB, BasePtr, Indices);

  // See if we can descend into a struct and locate a field with the correct
  // type.
  unsigned NumLayers = 0;
  Type *ElementTy = Ty;
  do {
    if (ElementTy->isPointerTy())
      break;
    if (SequentialType *SeqTy = dyn_cast<SequentialType>(ElementTy)) {
      ElementTy = SeqTy->getElementType();
      // Note that we use the default address space as this index is over an
      // array or a vector, not a pointer.
      Indices.push_back(IRB.getInt(APInt(DL.getPointerSizeInBits(0), 0)));
    } else if (StructType *STy = dyn_cast<StructType>(ElementTy)) {
      if (STy->element_begin() == STy->element_end())
        break; // Nothing left to descend into.
      ElementTy = *STy->element_begin();
      Indices.push_back(IRB.getInt32(0));
    } else {
      break;
    }
    ++NumLayers;
  } while (ElementTy != TargetTy);
  if (ElementTy != TargetTy)
    Indices.erase(Indices.end() - NumLayers, Indices.end());

  return buildGEP(IRB, BasePtr, Indices);
}

/// \brief Recursively compute indices for a natural GEP.
///
/// This is the recursive step for getNaturalGEPWithOffset that walks down the
/// element types adding appropriate indices for the GEP.
static Value *getNaturalGEPRecursively(IRBuilderTy &IRB, const DataLayout &DL,
                                       Value *Ptr, Type *Ty, APInt &Offset,
                                       Type *TargetTy,
                                       SmallVectorImpl<Value *> &Indices) {
  if (Offset == 0)
    return getNaturalGEPWithType(IRB, DL, Ptr, Ty, TargetTy, Indices);

  // We can't recurse through pointer types.
  if (Ty->isPointerTy())
    return 0;

  // We try to analyze GEPs over vectors here, but note that these GEPs are
  // extremely poorly defined currently. The long-term goal is to remove GEPing
  // over a vector from the IR completely.
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty)) {
    unsigned ElementSizeInBits = DL.getTypeSizeInBits(VecTy->getScalarType());
    if (ElementSizeInBits % 8)
      return 0; // GEPs over non-multiple of 8 size vector elements are invalid.
    APInt ElementSize(Offset.getBitWidth(), ElementSizeInBits / 8);
    APInt NumSkippedElements = Offset.sdiv(ElementSize);
    if (NumSkippedElements.ugt(VecTy->getNumElements()))
      return 0;
    Offset -= NumSkippedElements * ElementSize;
    Indices.push_back(IRB.getInt(NumSkippedElements));
    return getNaturalGEPRecursively(IRB, DL, Ptr, VecTy->getElementType(),
                                    Offset, TargetTy, Indices);
  }

  if (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
    Type *ElementTy = ArrTy->getElementType();
    APInt ElementSize(Offset.getBitWidth(), DL.getTypeAllocSize(ElementTy));
    APInt NumSkippedElements = Offset.sdiv(ElementSize);
    if (NumSkippedElements.ugt(ArrTy->getNumElements()))
      return 0;

    Offset -= NumSkippedElements * ElementSize;
    Indices.push_back(IRB.getInt(NumSkippedElements));
    return getNaturalGEPRecursively(IRB, DL, Ptr, ElementTy, Offset, TargetTy,
                                    Indices);
  }

  StructType *STy = dyn_cast<StructType>(Ty);
  if (!STy)
    return 0;

  const StructLayout *SL = DL.getStructLayout(STy);
  uint64_t StructOffset = Offset.getZExtValue();
  if (StructOffset >= SL->getSizeInBytes())
    return 0;
  unsigned Index = SL->getElementContainingOffset(StructOffset);
  Offset -= APInt(Offset.getBitWidth(), SL->getElementOffset(Index));
  Type *ElementTy = STy->getElementType(Index);
  if (Offset.uge(DL.getTypeAllocSize(ElementTy)))
    return 0; // The offset points into alignment padding.

  Indices.push_back(IRB.getInt32(Index));
  return getNaturalGEPRecursively(IRB, DL, Ptr, ElementTy, Offset, TargetTy,
                                  Indices);
}

/// \brief Get a natural GEP from a base pointer to a particular offset and
/// resulting in a particular type.
///
/// The goal is to produce a "natural" looking GEP that works with the existing
/// composite types to arrive at the appropriate offset and element type for
/// a pointer. TargetTy is the element type the returned GEP should point-to if
/// possible. We recurse by decreasing Offset, adding the appropriate index to
/// Indices, and setting Ty to the result subtype.
///
/// If no natural GEP can be constructed, this function returns null.
static Value *getNaturalGEPWithOffset(IRBuilderTy &IRB, const DataLayout &DL,
                                      Value *Ptr, APInt Offset, Type *TargetTy,
                                      SmallVectorImpl<Value *> &Indices) {
  PointerType *Ty = cast<PointerType>(Ptr->getType());

  // Don't consider any GEPs through an i8* as natural unless the TargetTy is
  // an i8.
  if (Ty == IRB.getInt8PtrTy() && TargetTy->isIntegerTy(8))
    return 0;

  Type *ElementTy = Ty->getElementType();
  if (!ElementTy->isSized())
    return 0; // We can't GEP through an unsized element.
  APInt ElementSize(Offset.getBitWidth(), DL.getTypeAllocSize(ElementTy));
  if (ElementSize == 0)
    return 0; // Zero-length arrays can't help us build a natural GEP.
  APInt NumSkippedElements = Offset.sdiv(ElementSize);

  Offset -= NumSkippedElements * ElementSize;
  Indices.push_back(IRB.getInt(NumSkippedElements));
  return getNaturalGEPRecursively(IRB, DL, Ptr, ElementTy, Offset, TargetTy,
                                  Indices);
}

/// \brief Compute an adjusted pointer from Ptr by Offset bytes where the
/// resulting pointer has PointerTy.
///
/// This tries very hard to compute a "natural" GEP which arrives at the offset
/// and produces the pointer type desired. Where it cannot, it will try to use
/// the natural GEP to arrive at the offset and bitcast to the type. Where that
/// fails, it will try to use an existing i8* and GEP to the byte offset and
/// bitcast to the type.
///
/// The strategy for finding the more natural GEPs is to peel off layers of the
/// pointer, walking back through bit casts and GEPs, searching for a base
/// pointer from which we can compute a natural GEP with the desired
/// properties. The algorithm tries to fold as many constant indices into
/// a single GEP as possible, thus making each GEP more independent of the
/// surrounding code.
static Value *getAdjustedPtr(IRBuilderTy &IRB, const DataLayout &DL,
                             Value *Ptr, APInt Offset, Type *PointerTy) {
  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<Value *, 4> Visited;
  Visited.insert(Ptr);
  SmallVector<Value *, 4> Indices;

  // We may end up computing an offset pointer that has the wrong type. If we
  // never are able to compute one directly that has the correct type, we'll
  // fall back to it, so keep it around here.
  Value *OffsetPtr = 0;

  // Remember any i8 pointer we come across to re-use if we need to do a raw
  // byte offset.
  Value *Int8Ptr = 0;
  APInt Int8PtrOffset(Offset.getBitWidth(), 0);

  Type *TargetTy = PointerTy->getPointerElementType();

  do {
    // First fold any existing GEPs into the offset.
    while (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr)) {
      APInt GEPOffset(Offset.getBitWidth(), 0);
      if (!GEP->accumulateConstantOffset(DL, GEPOffset))
        break;
      Offset += GEPOffset;
      Ptr = GEP->getPointerOperand();
      if (!Visited.insert(Ptr))
        break;
    }

    // See if we can perform a natural GEP here.
    Indices.clear();
    if (Value *P = getNaturalGEPWithOffset(IRB, DL, Ptr, Offset, TargetTy,
                                           Indices)) {
      if (P->getType() == PointerTy) {
        // Zap any offset pointer that we ended up computing in previous rounds.
        if (OffsetPtr && OffsetPtr->use_empty())
          if (Instruction *I = dyn_cast<Instruction>(OffsetPtr))
            I->eraseFromParent();
        return P;
      }
      if (!OffsetPtr) {
        OffsetPtr = P;
      }
    }

    // Stash this pointer if we've found an i8*.
    if (Ptr->getType()->isIntegerTy(8)) {
      Int8Ptr = Ptr;
      Int8PtrOffset = Offset;
    }

    // Peel off a layer of the pointer and update the offset appropriately.
    if (Operator::getOpcode(Ptr) == Instruction::BitCast) {
      Ptr = cast<Operator>(Ptr)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(Ptr)) {
      if (GA->mayBeOverridden())
        break;
      Ptr = GA->getAliasee();
    } else {
      break;
    }
    assert(Ptr->getType()->isPointerTy() && "Unexpected operand type!");
  } while (Visited.insert(Ptr));

  if (!OffsetPtr) {
    if (!Int8Ptr) {
      Int8Ptr = IRB.CreateBitCast(Ptr, IRB.getInt8PtrTy(),
                                  "raw_cast");
      Int8PtrOffset = Offset;
    }

    OffsetPtr = Int8PtrOffset == 0 ? Int8Ptr :
      IRB.CreateInBoundsGEP(Int8Ptr, IRB.getInt(Int8PtrOffset),
                            "raw_idx");
  }
  Ptr = OffsetPtr;

  // On the off chance we were targeting i8*, guard the bitcast here.
  if (Ptr->getType() != PointerTy)
    Ptr = IRB.CreateBitCast(Ptr, PointerTy, "cast");

  return Ptr;
}

/// \brief Test whether we can convert a value from the old to the new type.
///
/// This predicate should be used to guard calls to convertValue in order to
/// ensure that we only try to convert viable values. The strategy is that we
/// will peel off single element struct and array wrappings to get to an
/// underlying value, and convert that value.
static bool canConvertValue(const DataLayout &DL, Type *OldTy, Type *NewTy) {
  if (OldTy == NewTy)
    return true;
  if (IntegerType *OldITy = dyn_cast<IntegerType>(OldTy))
    if (IntegerType *NewITy = dyn_cast<IntegerType>(NewTy))
      if (NewITy->getBitWidth() >= OldITy->getBitWidth())
        return true;
  if (DL.getTypeSizeInBits(NewTy) != DL.getTypeSizeInBits(OldTy))
    return false;
  if (!NewTy->isSingleValueType() || !OldTy->isSingleValueType())
    return false;

  // We can convert pointers to integers and vice-versa. Same for vectors
  // of pointers and integers.
  OldTy = OldTy->getScalarType();
  NewTy = NewTy->getScalarType();
  if (NewTy->isPointerTy() || OldTy->isPointerTy()) {
    if (NewTy->isPointerTy() && OldTy->isPointerTy())
      return true;
    if (NewTy->isIntegerTy() || OldTy->isIntegerTy())
      return true;
    return false;
  }

  return true;
}

/// \brief Generic routine to convert an SSA value to a value of a different
/// type.
///
/// This will try various different casting techniques, such as bitcasts,
/// inttoptr, and ptrtoint casts. Use the \c canConvertValue predicate to test
/// two types for viability with this routine.
static Value *convertValue(const DataLayout &DL, IRBuilderTy &IRB, Value *V,
                           Type *NewTy) {
  Type *OldTy = V->getType();
  assert(canConvertValue(DL, OldTy, NewTy) && "Value not convertable to type");

  if (OldTy == NewTy)
    return V;

  if (IntegerType *OldITy = dyn_cast<IntegerType>(OldTy))
    if (IntegerType *NewITy = dyn_cast<IntegerType>(NewTy))
      if (NewITy->getBitWidth() > OldITy->getBitWidth())
        return IRB.CreateZExt(V, NewITy);

  // See if we need inttoptr for this type pair. A cast involving both scalars
  // and vectors requires and additional bitcast.
  if (OldTy->getScalarType()->isIntegerTy() &&
      NewTy->getScalarType()->isPointerTy()) {
    // Expand <2 x i32> to i8* --> <2 x i32> to i64 to i8*
    if (OldTy->isVectorTy() && !NewTy->isVectorTy())
      return IRB.CreateIntToPtr(IRB.CreateBitCast(V, DL.getIntPtrType(NewTy)),
                                NewTy);

    // Expand i128 to <2 x i8*> --> i128 to <2 x i64> to <2 x i8*>
    if (!OldTy->isVectorTy() && NewTy->isVectorTy())
      return IRB.CreateIntToPtr(IRB.CreateBitCast(V, DL.getIntPtrType(NewTy)),
                                NewTy);

    return IRB.CreateIntToPtr(V, NewTy);
  }

  // See if we need ptrtoint for this type pair. A cast involving both scalars
  // and vectors requires and additional bitcast.
  if (OldTy->getScalarType()->isPointerTy() &&
      NewTy->getScalarType()->isIntegerTy()) {
    // Expand <2 x i8*> to i128 --> <2 x i8*> to <2 x i64> to i128
    if (OldTy->isVectorTy() && !NewTy->isVectorTy())
      return IRB.CreateBitCast(IRB.CreatePtrToInt(V, DL.getIntPtrType(OldTy)),
                               NewTy);

    // Expand i8* to <2 x i32> --> i8* to i64 to <2 x i32>
    if (!OldTy->isVectorTy() && NewTy->isVectorTy())
      return IRB.CreateBitCast(IRB.CreatePtrToInt(V, DL.getIntPtrType(OldTy)),
                               NewTy);

    return IRB.CreatePtrToInt(V, NewTy);
  }

  return IRB.CreateBitCast(V, NewTy);
}

/// \brief Test whether the given slice use can be promoted to a vector.
///
/// This function is called to test each entry in a partioning which is slated
/// for a single slice.
static bool isVectorPromotionViableForSlice(
    const DataLayout &DL, AllocaSlices &S, uint64_t SliceBeginOffset,
    uint64_t SliceEndOffset, VectorType *Ty, uint64_t ElementSize,
    AllocaSlices::const_iterator I) {
  // First validate the slice offsets.
  uint64_t BeginOffset =
      std::max(I->beginOffset(), SliceBeginOffset) - SliceBeginOffset;
  uint64_t BeginIndex = BeginOffset / ElementSize;
  if (BeginIndex * ElementSize != BeginOffset ||
      BeginIndex >= Ty->getNumElements())
    return false;
  uint64_t EndOffset =
      std::min(I->endOffset(), SliceEndOffset) - SliceBeginOffset;
  uint64_t EndIndex = EndOffset / ElementSize;
  if (EndIndex * ElementSize != EndOffset || EndIndex > Ty->getNumElements())
    return false;

  assert(EndIndex > BeginIndex && "Empty vector!");
  uint64_t NumElements = EndIndex - BeginIndex;
  Type *SliceTy =
      (NumElements == 1) ? Ty->getElementType()
                         : VectorType::get(Ty->getElementType(), NumElements);

  Type *SplitIntTy =
      Type::getIntNTy(Ty->getContext(), NumElements * ElementSize * 8);

  Use *U = I->getUse();

  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(U->getUser())) {
    if (MI->isVolatile())
      return false;
    if (!I->isSplittable())
      return false; // Skip any unsplittable intrinsics.
  } else if (U->get()->getType()->getPointerElementType()->isStructTy()) {
    // Disable vector promotion when there are loads or stores of an FCA.
    return false;
  } else if (LoadInst *LI = dyn_cast<LoadInst>(U->getUser())) {
    if (LI->isVolatile())
      return false;
    Type *LTy = LI->getType();
    if (SliceBeginOffset > I->beginOffset() ||
        SliceEndOffset < I->endOffset()) {
      assert(LTy->isIntegerTy());
      LTy = SplitIntTy;
    }
    if (!canConvertValue(DL, SliceTy, LTy))
      return false;
  } else if (StoreInst *SI = dyn_cast<StoreInst>(U->getUser())) {
    if (SI->isVolatile())
      return false;
    Type *STy = SI->getValueOperand()->getType();
    if (SliceBeginOffset > I->beginOffset() ||
        SliceEndOffset < I->endOffset()) {
      assert(STy->isIntegerTy());
      STy = SplitIntTy;
    }
    if (!canConvertValue(DL, STy, SliceTy))
      return false;
  } else {
    return false;
  }

  return true;
}

/// \brief Test whether the given alloca partitioning and range of slices can be
/// promoted to a vector.
///
/// This is a quick test to check whether we can rewrite a particular alloca
/// partition (and its newly formed alloca) into a vector alloca with only
/// whole-vector loads and stores such that it could be promoted to a vector
/// SSA value. We only can ensure this for a limited set of operations, and we
/// don't want to do the rewrites unless we are confident that the result will
/// be promotable, so we have an early test here.
static bool
isVectorPromotionViable(const DataLayout &DL, Type *AllocaTy, AllocaSlices &S,
                        uint64_t SliceBeginOffset, uint64_t SliceEndOffset,
                        AllocaSlices::const_iterator I,
                        AllocaSlices::const_iterator E,
                        ArrayRef<AllocaSlices::iterator> SplitUses) {
  VectorType *Ty = dyn_cast<VectorType>(AllocaTy);
  if (!Ty)
    return false;

  uint64_t ElementSize = DL.getTypeSizeInBits(Ty->getScalarType());

  // While the definition of LLVM vectors is bitpacked, we don't support sizes
  // that aren't byte sized.
  if (ElementSize % 8)
    return false;
  assert((DL.getTypeSizeInBits(Ty) % 8) == 0 &&
         "vector size not a multiple of element size?");
  ElementSize /= 8;

  for (; I != E; ++I)
    if (!isVectorPromotionViableForSlice(DL, S, SliceBeginOffset,
                                         SliceEndOffset, Ty, ElementSize, I))
      return false;

  for (ArrayRef<AllocaSlices::iterator>::const_iterator SUI = SplitUses.begin(),
                                                        SUE = SplitUses.end();
       SUI != SUE; ++SUI)
    if (!isVectorPromotionViableForSlice(DL, S, SliceBeginOffset,
                                         SliceEndOffset, Ty, ElementSize, *SUI))
      return false;

  return true;
}

/// \brief Test whether a slice of an alloca is valid for integer widening.
///
/// This implements the necessary checking for the \c isIntegerWideningViable
/// test below on a single slice of the alloca.
static bool isIntegerWideningViableForSlice(const DataLayout &DL,
                                            Type *AllocaTy,
                                            uint64_t AllocBeginOffset,
                                            uint64_t Size, AllocaSlices &S,
                                            AllocaSlices::const_iterator I,
                                            bool &WholeAllocaOp) {
  uint64_t RelBegin = I->beginOffset() - AllocBeginOffset;
  uint64_t RelEnd = I->endOffset() - AllocBeginOffset;

  // We can't reasonably handle cases where the load or store extends past
  // the end of the aloca's type and into its padding.
  if (RelEnd > Size)
    return false;

  Use *U = I->getUse();

  if (LoadInst *LI = dyn_cast<LoadInst>(U->getUser())) {
    if (LI->isVolatile())
      return false;
    if (RelBegin == 0 && RelEnd == Size)
      WholeAllocaOp = true;
    if (IntegerType *ITy = dyn_cast<IntegerType>(LI->getType())) {
      if (ITy->getBitWidth() < DL.getTypeStoreSizeInBits(ITy))
        return false;
    } else if (RelBegin != 0 || RelEnd != Size ||
               !canConvertValue(DL, AllocaTy, LI->getType())) {
      // Non-integer loads need to be convertible from the alloca type so that
      // they are promotable.
      return false;
    }
  } else if (StoreInst *SI = dyn_cast<StoreInst>(U->getUser())) {
    Type *ValueTy = SI->getValueOperand()->getType();
    if (SI->isVolatile())
      return false;
    if (RelBegin == 0 && RelEnd == Size)
      WholeAllocaOp = true;
    if (IntegerType *ITy = dyn_cast<IntegerType>(ValueTy)) {
      if (ITy->getBitWidth() < DL.getTypeStoreSizeInBits(ITy))
        return false;
    } else if (RelBegin != 0 || RelEnd != Size ||
               !canConvertValue(DL, ValueTy, AllocaTy)) {
      // Non-integer stores need to be convertible to the alloca type so that
      // they are promotable.
      return false;
    }
  } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(U->getUser())) {
    if (MI->isVolatile() || !isa<Constant>(MI->getLength()))
      return false;
    if (!I->isSplittable())
      return false; // Skip any unsplittable intrinsics.
  } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U->getUser())) {
    if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
        II->getIntrinsicID() != Intrinsic::lifetime_end)
      return false;
  } else {
    return false;
  }

  return true;
}

/// \brief Test whether the given alloca partition's integer operations can be
/// widened to promotable ones.
///
/// This is a quick test to check whether we can rewrite the integer loads and
/// stores to a particular alloca into wider loads and stores and be able to
/// promote the resulting alloca.
static bool
isIntegerWideningViable(const DataLayout &DL, Type *AllocaTy,
                        uint64_t AllocBeginOffset, AllocaSlices &S,
                        AllocaSlices::const_iterator I,
                        AllocaSlices::const_iterator E,
                        ArrayRef<AllocaSlices::iterator> SplitUses) {
  uint64_t SizeInBits = DL.getTypeSizeInBits(AllocaTy);
  // Don't create integer types larger than the maximum bitwidth.
  if (SizeInBits > IntegerType::MAX_INT_BITS)
    return false;

  // Don't try to handle allocas with bit-padding.
  if (SizeInBits != DL.getTypeStoreSizeInBits(AllocaTy))
    return false;

  // We need to ensure that an integer type with the appropriate bitwidth can
  // be converted to the alloca type, whatever that is. We don't want to force
  // the alloca itself to have an integer type if there is a more suitable one.
  Type *IntTy = Type::getIntNTy(AllocaTy->getContext(), SizeInBits);
  if (!canConvertValue(DL, AllocaTy, IntTy) ||
      !canConvertValue(DL, IntTy, AllocaTy))
    return false;

  uint64_t Size = DL.getTypeStoreSize(AllocaTy);

  // While examining uses, we ensure that the alloca has a covering load or
  // store. We don't want to widen the integer operations only to fail to
  // promote due to some other unsplittable entry (which we may make splittable
  // later). However, if there are only splittable uses, go ahead and assume
  // that we cover the alloca.
  bool WholeAllocaOp = (I != E) ? false : DL.isLegalInteger(SizeInBits);

  for (; I != E; ++I)
    if (!isIntegerWideningViableForSlice(DL, AllocaTy, AllocBeginOffset, Size,
                                         S, I, WholeAllocaOp))
      return false;

  for (ArrayRef<AllocaSlices::iterator>::const_iterator SUI = SplitUses.begin(),
                                                        SUE = SplitUses.end();
       SUI != SUE; ++SUI)
    if (!isIntegerWideningViableForSlice(DL, AllocaTy, AllocBeginOffset, Size,
                                         S, *SUI, WholeAllocaOp))
      return false;

  return WholeAllocaOp;
}

static Value *extractInteger(const DataLayout &DL, IRBuilderTy &IRB, Value *V,
                             IntegerType *Ty, uint64_t Offset,
                             const Twine &Name) {
  DEBUG(dbgs() << "       start: " << *V << "\n");
  IntegerType *IntTy = cast<IntegerType>(V->getType());
  assert(DL.getTypeStoreSize(Ty) + Offset <= DL.getTypeStoreSize(IntTy) &&
         "Element extends past full value");
  uint64_t ShAmt = 8*Offset;
  if (DL.isBigEndian())
    ShAmt = 8*(DL.getTypeStoreSize(IntTy) - DL.getTypeStoreSize(Ty) - Offset);
  if (ShAmt) {
    V = IRB.CreateLShr(V, ShAmt, Name + ".shift");
    DEBUG(dbgs() << "     shifted: " << *V << "\n");
  }
  assert(Ty->getBitWidth() <= IntTy->getBitWidth() &&
         "Cannot extract to a larger integer!");
  if (Ty != IntTy) {
    V = IRB.CreateTrunc(V, Ty, Name + ".trunc");
    DEBUG(dbgs() << "     trunced: " << *V << "\n");
  }
  return V;
}

static Value *insertInteger(const DataLayout &DL, IRBuilderTy &IRB, Value *Old,
                            Value *V, uint64_t Offset, const Twine &Name) {
  IntegerType *IntTy = cast<IntegerType>(Old->getType());
  IntegerType *Ty = cast<IntegerType>(V->getType());
  assert(Ty->getBitWidth() <= IntTy->getBitWidth() &&
         "Cannot insert a larger integer!");
  DEBUG(dbgs() << "       start: " << *V << "\n");
  if (Ty != IntTy) {
    V = IRB.CreateZExt(V, IntTy, Name + ".ext");
    DEBUG(dbgs() << "    extended: " << *V << "\n");
  }
  assert(DL.getTypeStoreSize(Ty) + Offset <= DL.getTypeStoreSize(IntTy) &&
         "Element store outside of alloca store");
  uint64_t ShAmt = 8*Offset;
  if (DL.isBigEndian())
    ShAmt = 8*(DL.getTypeStoreSize(IntTy) - DL.getTypeStoreSize(Ty) - Offset);
  if (ShAmt) {
    V = IRB.CreateShl(V, ShAmt, Name + ".shift");
    DEBUG(dbgs() << "     shifted: " << *V << "\n");
  }

  if (ShAmt || Ty->getBitWidth() < IntTy->getBitWidth()) {
    APInt Mask = ~Ty->getMask().zext(IntTy->getBitWidth()).shl(ShAmt);
    Old = IRB.CreateAnd(Old, Mask, Name + ".mask");
    DEBUG(dbgs() << "      masked: " << *Old << "\n");
    V = IRB.CreateOr(Old, V, Name + ".insert");
    DEBUG(dbgs() << "    inserted: " << *V << "\n");
  }
  return V;
}

static Value *extractVector(IRBuilderTy &IRB, Value *V,
                            unsigned BeginIndex, unsigned EndIndex,
                            const Twine &Name) {
  VectorType *VecTy = cast<VectorType>(V->getType());
  unsigned NumElements = EndIndex - BeginIndex;
  assert(NumElements <= VecTy->getNumElements() && "Too many elements!");

  if (NumElements == VecTy->getNumElements())
    return V;

  if (NumElements == 1) {
    V = IRB.CreateExtractElement(V, IRB.getInt32(BeginIndex),
                                 Name + ".extract");
    DEBUG(dbgs() << "     extract: " << *V << "\n");
    return V;
  }

  SmallVector<Constant*, 8> Mask;
  Mask.reserve(NumElements);
  for (unsigned i = BeginIndex; i != EndIndex; ++i)
    Mask.push_back(IRB.getInt32(i));
  V = IRB.CreateShuffleVector(V, UndefValue::get(V->getType()),
                              ConstantVector::get(Mask),
                              Name + ".extract");
  DEBUG(dbgs() << "     shuffle: " << *V << "\n");
  return V;
}

static Value *insertVector(IRBuilderTy &IRB, Value *Old, Value *V,
                           unsigned BeginIndex, const Twine &Name) {
  VectorType *VecTy = cast<VectorType>(Old->getType());
  assert(VecTy && "Can only insert a vector into a vector");

  VectorType *Ty = dyn_cast<VectorType>(V->getType());
  if (!Ty) {
    // Single element to insert.
    V = IRB.CreateInsertElement(Old, V, IRB.getInt32(BeginIndex),
                                Name + ".insert");
    DEBUG(dbgs() <<  "     insert: " << *V << "\n");
    return V;
  }

  assert(Ty->getNumElements() <= VecTy->getNumElements() &&
         "Too many elements!");
  if (Ty->getNumElements() == VecTy->getNumElements()) {
    assert(V->getType() == VecTy && "Vector type mismatch");
    return V;
  }
  unsigned EndIndex = BeginIndex + Ty->getNumElements();

  // When inserting a smaller vector into the larger to store, we first
  // use a shuffle vector to widen it with undef elements, and then
  // a second shuffle vector to select between the loaded vector and the
  // incoming vector.
  SmallVector<Constant*, 8> Mask;
  Mask.reserve(VecTy->getNumElements());
  for (unsigned i = 0; i != VecTy->getNumElements(); ++i)
    if (i >= BeginIndex && i < EndIndex)
      Mask.push_back(IRB.getInt32(i - BeginIndex));
    else
      Mask.push_back(UndefValue::get(IRB.getInt32Ty()));
  V = IRB.CreateShuffleVector(V, UndefValue::get(V->getType()),
                              ConstantVector::get(Mask),
                              Name + ".expand");
  DEBUG(dbgs() << "    shuffle: " << *V << "\n");

  Mask.clear();
  for (unsigned i = 0; i != VecTy->getNumElements(); ++i)
    Mask.push_back(IRB.getInt1(i >= BeginIndex && i < EndIndex));

  V = IRB.CreateSelect(ConstantVector::get(Mask), V, Old, Name + "blend");

  DEBUG(dbgs() << "    blend: " << *V << "\n");
  return V;
}

namespace {
/// \brief Visitor to rewrite instructions using p particular slice of an alloca
/// to use a new alloca.
///
/// Also implements the rewriting to vector-based accesses when the partition
/// passes the isVectorPromotionViable predicate. Most of the rewriting logic
/// lives here.
class AllocaSliceRewriter : public InstVisitor<AllocaSliceRewriter, bool> {
  // Befriend the base class so it can delegate to private visit methods.
  friend class llvm::InstVisitor<AllocaSliceRewriter, bool>;
  typedef llvm::InstVisitor<AllocaSliceRewriter, bool> Base;

  const DataLayout &DL;
  AllocaSlices &S;
  SROA &Pass;
  AllocaInst &OldAI, &NewAI;
  const uint64_t NewAllocaBeginOffset, NewAllocaEndOffset;
  Type *NewAllocaTy;

  // If we are rewriting an alloca partition which can be written as pure
  // vector operations, we stash extra information here. When VecTy is
  // non-null, we have some strict guarantees about the rewritten alloca:
  //   - The new alloca is exactly the size of the vector type here.
  //   - The accesses all either map to the entire vector or to a single
  //     element.
  //   - The set of accessing instructions is only one of those handled above
  //     in isVectorPromotionViable. Generally these are the same access kinds
  //     which are promotable via mem2reg.
  VectorType *VecTy;
  Type *ElementTy;
  uint64_t ElementSize;

  // This is a convenience and flag variable that will be null unless the new
  // alloca's integer operations should be widened to this integer type due to
  // passing isIntegerWideningViable above. If it is non-null, the desired
  // integer type will be stored here for easy access during rewriting.
  IntegerType *IntTy;

  // The offset of the slice currently being rewritten.
  uint64_t BeginOffset, EndOffset;
  bool IsSplittable;
  bool IsSplit;
  Use *OldUse;
  Instruction *OldPtr;

  // Output members carrying state about the result of visiting and rewriting
  // the slice of the alloca.
  bool IsUsedByRewrittenSpeculatableInstructions;

  // Utility IR builder, whose name prefix is setup for each visited use, and
  // the insertion point is set to point to the user.
  IRBuilderTy IRB;

public:
  AllocaSliceRewriter(const DataLayout &DL, AllocaSlices &S, SROA &Pass,
                      AllocaInst &OldAI, AllocaInst &NewAI,
                      uint64_t NewBeginOffset, uint64_t NewEndOffset,
                      bool IsVectorPromotable = false,
                      bool IsIntegerPromotable = false)
      : DL(DL), S(S), Pass(Pass), OldAI(OldAI), NewAI(NewAI),
        NewAllocaBeginOffset(NewBeginOffset), NewAllocaEndOffset(NewEndOffset),
        NewAllocaTy(NewAI.getAllocatedType()),
        VecTy(IsVectorPromotable ? cast<VectorType>(NewAllocaTy) : 0),
        ElementTy(VecTy ? VecTy->getElementType() : 0),
        ElementSize(VecTy ? DL.getTypeSizeInBits(ElementTy) / 8 : 0),
        IntTy(IsIntegerPromotable
                  ? Type::getIntNTy(
                        NewAI.getContext(),
                        DL.getTypeSizeInBits(NewAI.getAllocatedType()))
                  : 0),
        BeginOffset(), EndOffset(), IsSplittable(), IsSplit(), OldUse(),
        OldPtr(), IsUsedByRewrittenSpeculatableInstructions(false),
        IRB(NewAI.getContext(), ConstantFolder()) {
    if (VecTy) {
      assert((DL.getTypeSizeInBits(ElementTy) % 8) == 0 &&
             "Only multiple-of-8 sized vector elements are viable");
      ++NumVectorized;
    }
    assert((!IsVectorPromotable && !IsIntegerPromotable) ||
           IsVectorPromotable != IsIntegerPromotable);
  }

  bool visit(AllocaSlices::const_iterator I) {
    bool CanSROA = true;
    BeginOffset = I->beginOffset();
    EndOffset = I->endOffset();
    IsSplittable = I->isSplittable();
    IsSplit =
        BeginOffset < NewAllocaBeginOffset || EndOffset > NewAllocaEndOffset;

    OldUse = I->getUse();
    OldPtr = cast<Instruction>(OldUse->get());

    Instruction *OldUserI = cast<Instruction>(OldUse->getUser());
    IRB.SetInsertPoint(OldUserI);
    IRB.SetCurrentDebugLocation(OldUserI->getDebugLoc());
    IRB.SetNamePrefix(Twine(NewAI.getName()) + "." + Twine(BeginOffset) + ".");

    CanSROA &= visit(cast<Instruction>(OldUse->getUser()));
    if (VecTy || IntTy)
      assert(CanSROA);
    return CanSROA;
  }

  /// \brief Query whether this slice is used by speculatable instructions after
  /// rewriting.
  ///
  /// These instructions (PHIs and Selects currently) require the alloca slice
  /// to run back through the rewriter. Thus, they are promotable, but not on
  /// this iteration. This is distinct from a slice which is unpromotable for
  /// some other reason, in which case we don't even want to perform the
  /// speculation. This can be querried at any time and reflects whether (at
  /// that point) a visit call has rewritten a speculatable instruction on the
  /// current slice.
  bool isUsedByRewrittenSpeculatableInstructions() const {
    return IsUsedByRewrittenSpeculatableInstructions;
  }

private:
  // Make sure the other visit overloads are visible.
  using Base::visit;

  // Every instruction which can end up as a user must have a rewrite rule.
  bool visitInstruction(Instruction &I) {
    DEBUG(dbgs() << "    !!!! Cannot rewrite: " << I << "\n");
    llvm_unreachable("No rewrite rule for this instruction!");
  }

  Value *getAdjustedAllocaPtr(IRBuilderTy &IRB, uint64_t Offset,
                              Type *PointerTy) {
    assert(Offset >= NewAllocaBeginOffset);
    return getAdjustedPtr(IRB, DL, &NewAI, APInt(DL.getPointerSizeInBits(),
                                                 Offset - NewAllocaBeginOffset),
                          PointerTy);
  }

  /// \brief Compute suitable alignment to access an offset into the new alloca.
  unsigned getOffsetAlign(uint64_t Offset) {
    unsigned NewAIAlign = NewAI.getAlignment();
    if (!NewAIAlign)
      NewAIAlign = DL.getABITypeAlignment(NewAI.getAllocatedType());
    return MinAlign(NewAIAlign, Offset);
  }

  /// \brief Compute suitable alignment to access a type at an offset of the
  /// new alloca.
  ///
  /// \returns zero if the type's ABI alignment is a suitable alignment,
  /// otherwise returns the maximal suitable alignment.
  unsigned getOffsetTypeAlign(Type *Ty, uint64_t Offset) {
    unsigned Align = getOffsetAlign(Offset);
    return Align == DL.getABITypeAlignment(Ty) ? 0 : Align;
  }

  unsigned getIndex(uint64_t Offset) {
    assert(VecTy && "Can only call getIndex when rewriting a vector");
    uint64_t RelOffset = Offset - NewAllocaBeginOffset;
    assert(RelOffset / ElementSize < UINT32_MAX && "Index out of bounds");
    uint32_t Index = RelOffset / ElementSize;
    assert(Index * ElementSize == RelOffset);
    return Index;
  }

  void deleteIfTriviallyDead(Value *V) {
    Instruction *I = cast<Instruction>(V);
    if (isInstructionTriviallyDead(I))
      Pass.DeadInsts.insert(I);
  }

  Value *rewriteVectorizedLoadInst(uint64_t NewBeginOffset,
                                   uint64_t NewEndOffset) {
    unsigned BeginIndex = getIndex(NewBeginOffset);
    unsigned EndIndex = getIndex(NewEndOffset);
    assert(EndIndex > BeginIndex && "Empty vector!");

    Value *V = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                     "load");
    return extractVector(IRB, V, BeginIndex, EndIndex, "vec");
  }

  Value *rewriteIntegerLoad(LoadInst &LI, uint64_t NewBeginOffset,
                            uint64_t NewEndOffset) {
    assert(IntTy && "We cannot insert an integer to the alloca");
    assert(!LI.isVolatile());
    Value *V = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                     "load");
    V = convertValue(DL, IRB, V, IntTy);
    assert(NewBeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
    uint64_t Offset = NewBeginOffset - NewAllocaBeginOffset;
    if (Offset > 0 || NewEndOffset < NewAllocaEndOffset)
      V = extractInteger(DL, IRB, V, cast<IntegerType>(LI.getType()), Offset,
                         "extract");
    return V;
  }

  bool visitLoadInst(LoadInst &LI) {
    DEBUG(dbgs() << "    original: " << LI << "\n");
    Value *OldOp = LI.getOperand(0);
    assert(OldOp == OldPtr);

    // Compute the intersecting offset range.
    assert(BeginOffset < NewAllocaEndOffset);
    assert(EndOffset > NewAllocaBeginOffset);
    uint64_t NewBeginOffset = std::max(BeginOffset, NewAllocaBeginOffset);
    uint64_t NewEndOffset = std::min(EndOffset, NewAllocaEndOffset);

    uint64_t Size = NewEndOffset - NewBeginOffset;

    Type *TargetTy = IsSplit ? Type::getIntNTy(LI.getContext(), Size * 8)
                             : LI.getType();
    bool IsPtrAdjusted = false;
    Value *V;
    if (VecTy) {
      V = rewriteVectorizedLoadInst(NewBeginOffset, NewEndOffset);
    } else if (IntTy && LI.getType()->isIntegerTy()) {
      V = rewriteIntegerLoad(LI, NewBeginOffset, NewEndOffset);
    } else if (NewBeginOffset == NewAllocaBeginOffset &&
               canConvertValue(DL, NewAllocaTy, LI.getType())) {
      V = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                LI.isVolatile(), "load");
    } else {
      Type *LTy = TargetTy->getPointerTo();
      V = IRB.CreateAlignedLoad(
          getAdjustedAllocaPtr(IRB, NewBeginOffset, LTy),
          getOffsetTypeAlign(TargetTy, NewBeginOffset - NewAllocaBeginOffset),
          LI.isVolatile(), "load");
      IsPtrAdjusted = true;
    }
    V = convertValue(DL, IRB, V, TargetTy);

    if (IsSplit) {
      assert(!LI.isVolatile());
      assert(LI.getType()->isIntegerTy() &&
             "Only integer type loads and stores are split");
      assert(Size < DL.getTypeStoreSize(LI.getType()) &&
             "Split load isn't smaller than original load");
      assert(LI.getType()->getIntegerBitWidth() ==
             DL.getTypeStoreSizeInBits(LI.getType()) &&
             "Non-byte-multiple bit width");
      // Move the insertion point just past the load so that we can refer to it.
      IRB.SetInsertPoint(llvm::next(BasicBlock::iterator(&LI)));
      // Create a placeholder value with the same type as LI to use as the
      // basis for the new value. This allows us to replace the uses of LI with
      // the computed value, and then replace the placeholder with LI, leaving
      // LI only used for this computation.
      Value *Placeholder
        = new LoadInst(UndefValue::get(LI.getType()->getPointerTo()));
      V = insertInteger(DL, IRB, Placeholder, V, NewBeginOffset,
                        "insert");
      LI.replaceAllUsesWith(V);
      Placeholder->replaceAllUsesWith(&LI);
      delete Placeholder;
    } else {
      LI.replaceAllUsesWith(V);
    }

    Pass.DeadInsts.insert(&LI);
    deleteIfTriviallyDead(OldOp);
    DEBUG(dbgs() << "          to: " << *V << "\n");
    return !LI.isVolatile() && !IsPtrAdjusted;
  }

  bool rewriteVectorizedStoreInst(Value *V, StoreInst &SI, Value *OldOp,
                                  uint64_t NewBeginOffset,
                                  uint64_t NewEndOffset) {
    if (V->getType() != VecTy) {
      unsigned BeginIndex = getIndex(NewBeginOffset);
      unsigned EndIndex = getIndex(NewEndOffset);
      assert(EndIndex > BeginIndex && "Empty vector!");
      unsigned NumElements = EndIndex - BeginIndex;
      assert(NumElements <= VecTy->getNumElements() && "Too many elements!");
      Type *SliceTy =
          (NumElements == 1) ? ElementTy
                             : VectorType::get(ElementTy, NumElements);
      if (V->getType() != SliceTy)
        V = convertValue(DL, IRB, V, SliceTy);

      // Mix in the existing elements.
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         "load");
      V = insertVector(IRB, Old, V, BeginIndex, "vec");
    }
    StoreInst *Store = IRB.CreateAlignedStore(V, &NewAI, NewAI.getAlignment());
    Pass.DeadInsts.insert(&SI);

    (void)Store;
    DEBUG(dbgs() << "          to: " << *Store << "\n");
    return true;
  }

  bool rewriteIntegerStore(Value *V, StoreInst &SI,
                           uint64_t NewBeginOffset, uint64_t NewEndOffset) {
    assert(IntTy && "We cannot extract an integer from the alloca");
    assert(!SI.isVolatile());
    if (DL.getTypeSizeInBits(V->getType()) != IntTy->getBitWidth()) {
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         "oldload");
      Old = convertValue(DL, IRB, Old, IntTy);
      assert(BeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
      uint64_t Offset = BeginOffset - NewAllocaBeginOffset;
      V = insertInteger(DL, IRB, Old, SI.getValueOperand(), Offset,
                        "insert");
    }
    V = convertValue(DL, IRB, V, NewAllocaTy);
    StoreInst *Store = IRB.CreateAlignedStore(V, &NewAI, NewAI.getAlignment());
    Pass.DeadInsts.insert(&SI);
    (void)Store;
    DEBUG(dbgs() << "          to: " << *Store << "\n");
    return true;
  }

  bool visitStoreInst(StoreInst &SI) {
    DEBUG(dbgs() << "    original: " << SI << "\n");
    Value *OldOp = SI.getOperand(1);
    assert(OldOp == OldPtr);

    Value *V = SI.getValueOperand();

    // Strip all inbounds GEPs and pointer casts to try to dig out any root
    // alloca that should be re-examined after promoting this alloca.
    if (V->getType()->isPointerTy())
      if (AllocaInst *AI = dyn_cast<AllocaInst>(V->stripInBoundsOffsets()))
        Pass.PostPromotionWorklist.insert(AI);

    // Compute the intersecting offset range.
    assert(BeginOffset < NewAllocaEndOffset);
    assert(EndOffset > NewAllocaBeginOffset);
    uint64_t NewBeginOffset = std::max(BeginOffset, NewAllocaBeginOffset);
    uint64_t NewEndOffset = std::min(EndOffset, NewAllocaEndOffset);

    uint64_t Size = NewEndOffset - NewBeginOffset;
    if (Size < DL.getTypeStoreSize(V->getType())) {
      assert(!SI.isVolatile());
      assert(V->getType()->isIntegerTy() &&
             "Only integer type loads and stores are split");
      assert(V->getType()->getIntegerBitWidth() ==
             DL.getTypeStoreSizeInBits(V->getType()) &&
             "Non-byte-multiple bit width");
      IntegerType *NarrowTy = Type::getIntNTy(SI.getContext(), Size * 8);
      V = extractInteger(DL, IRB, V, NarrowTy, NewBeginOffset,
                         "extract");
    }

    if (VecTy)
      return rewriteVectorizedStoreInst(V, SI, OldOp, NewBeginOffset,
                                        NewEndOffset);
    if (IntTy && V->getType()->isIntegerTy())
      return rewriteIntegerStore(V, SI, NewBeginOffset, NewEndOffset);

    StoreInst *NewSI;
    if (NewBeginOffset == NewAllocaBeginOffset &&
        NewEndOffset == NewAllocaEndOffset &&
        canConvertValue(DL, V->getType(), NewAllocaTy)) {
      V = convertValue(DL, IRB, V, NewAllocaTy);
      NewSI = IRB.CreateAlignedStore(V, &NewAI, NewAI.getAlignment(),
                                     SI.isVolatile());
    } else {
      Value *NewPtr = getAdjustedAllocaPtr(IRB, NewBeginOffset,
                                           V->getType()->getPointerTo());
      NewSI = IRB.CreateAlignedStore(
          V, NewPtr, getOffsetTypeAlign(
                         V->getType(), NewBeginOffset - NewAllocaBeginOffset),
          SI.isVolatile());
    }
    (void)NewSI;
    Pass.DeadInsts.insert(&SI);
    deleteIfTriviallyDead(OldOp);

    DEBUG(dbgs() << "          to: " << *NewSI << "\n");
    return NewSI->getPointerOperand() == &NewAI && !SI.isVolatile();
  }

  /// \brief Compute an integer value from splatting an i8 across the given
  /// number of bytes.
  ///
  /// Note that this routine assumes an i8 is a byte. If that isn't true, don't
  /// call this routine.
  /// FIXME: Heed the advice above.
  ///
  /// \param V The i8 value to splat.
  /// \param Size The number of bytes in the output (assuming i8 is one byte)
  Value *getIntegerSplat(Value *V, unsigned Size) {
    assert(Size > 0 && "Expected a positive number of bytes.");
    IntegerType *VTy = cast<IntegerType>(V->getType());
    assert(VTy->getBitWidth() == 8 && "Expected an i8 value for the byte");
    if (Size == 1)
      return V;

    Type *SplatIntTy = Type::getIntNTy(VTy->getContext(), Size*8);
    V = IRB.CreateMul(IRB.CreateZExt(V, SplatIntTy, "zext"),
                      ConstantExpr::getUDiv(
                        Constant::getAllOnesValue(SplatIntTy),
                        ConstantExpr::getZExt(
                          Constant::getAllOnesValue(V->getType()),
                          SplatIntTy)),
                      "isplat");
    return V;
  }

  /// \brief Compute a vector splat for a given element value.
  Value *getVectorSplat(Value *V, unsigned NumElements) {
    V = IRB.CreateVectorSplat(NumElements, V, "vsplat");
    DEBUG(dbgs() << "       splat: " << *V << "\n");
    return V;
  }

  bool visitMemSetInst(MemSetInst &II) {
    DEBUG(dbgs() << "    original: " << II << "\n");
    assert(II.getRawDest() == OldPtr);

    // If the memset has a variable size, it cannot be split, just adjust the
    // pointer to the new alloca.
    if (!isa<Constant>(II.getLength())) {
      assert(!IsSplit);
      assert(BeginOffset >= NewAllocaBeginOffset);
      II.setDest(
          getAdjustedAllocaPtr(IRB, BeginOffset, II.getRawDest()->getType()));
      Type *CstTy = II.getAlignmentCst()->getType();
      II.setAlignment(ConstantInt::get(CstTy, getOffsetAlign(BeginOffset)));

      deleteIfTriviallyDead(OldPtr);
      return false;
    }

    // Record this instruction for deletion.
    Pass.DeadInsts.insert(&II);

    Type *AllocaTy = NewAI.getAllocatedType();
    Type *ScalarTy = AllocaTy->getScalarType();

    // Compute the intersecting offset range.
    assert(BeginOffset < NewAllocaEndOffset);
    assert(EndOffset > NewAllocaBeginOffset);
    uint64_t NewBeginOffset = std::max(BeginOffset, NewAllocaBeginOffset);
    uint64_t NewEndOffset = std::min(EndOffset, NewAllocaEndOffset);
    uint64_t SliceOffset = NewBeginOffset - NewAllocaBeginOffset;

    // If this doesn't map cleanly onto the alloca type, and that type isn't
    // a single value type, just emit a memset.
    if (!VecTy && !IntTy &&
        (BeginOffset > NewAllocaBeginOffset ||
         EndOffset < NewAllocaEndOffset ||
         !AllocaTy->isSingleValueType() ||
         !DL.isLegalInteger(DL.getTypeSizeInBits(ScalarTy)) ||
         DL.getTypeSizeInBits(ScalarTy)%8 != 0)) {
      Type *SizeTy = II.getLength()->getType();
      Constant *Size = ConstantInt::get(SizeTy, NewEndOffset - NewBeginOffset);
      CallInst *New = IRB.CreateMemSet(
          getAdjustedAllocaPtr(IRB, NewBeginOffset, II.getRawDest()->getType()),
          II.getValue(), Size, getOffsetAlign(SliceOffset), II.isVolatile());
      (void)New;
      DEBUG(dbgs() << "          to: " << *New << "\n");
      return false;
    }

    // If we can represent this as a simple value, we have to build the actual
    // value to store, which requires expanding the byte present in memset to
    // a sensible representation for the alloca type. This is essentially
    // splatting the byte to a sufficiently wide integer, splatting it across
    // any desired vector width, and bitcasting to the final type.
    Value *V;

    if (VecTy) {
      // If this is a memset of a vectorized alloca, insert it.
      assert(ElementTy == ScalarTy);

      unsigned BeginIndex = getIndex(NewBeginOffset);
      unsigned EndIndex = getIndex(NewEndOffset);
      assert(EndIndex > BeginIndex && "Empty vector!");
      unsigned NumElements = EndIndex - BeginIndex;
      assert(NumElements <= VecTy->getNumElements() && "Too many elements!");

      Value *Splat =
          getIntegerSplat(II.getValue(), DL.getTypeSizeInBits(ElementTy) / 8);
      Splat = convertValue(DL, IRB, Splat, ElementTy);
      if (NumElements > 1)
        Splat = getVectorSplat(Splat, NumElements);

      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         "oldload");
      V = insertVector(IRB, Old, Splat, BeginIndex, "vec");
    } else if (IntTy) {
      // If this is a memset on an alloca where we can widen stores, insert the
      // set integer.
      assert(!II.isVolatile());

      uint64_t Size = NewEndOffset - NewBeginOffset;
      V = getIntegerSplat(II.getValue(), Size);

      if (IntTy && (BeginOffset != NewAllocaBeginOffset ||
                    EndOffset != NewAllocaBeginOffset)) {
        Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                           "oldload");
        Old = convertValue(DL, IRB, Old, IntTy);
        uint64_t Offset = NewBeginOffset - NewAllocaBeginOffset;
        V = insertInteger(DL, IRB, Old, V, Offset, "insert");
      } else {
        assert(V->getType() == IntTy &&
               "Wrong type for an alloca wide integer!");
      }
      V = convertValue(DL, IRB, V, AllocaTy);
    } else {
      // Established these invariants above.
      assert(NewBeginOffset == NewAllocaBeginOffset);
      assert(NewEndOffset == NewAllocaEndOffset);

      V = getIntegerSplat(II.getValue(), DL.getTypeSizeInBits(ScalarTy) / 8);
      if (VectorType *AllocaVecTy = dyn_cast<VectorType>(AllocaTy))
        V = getVectorSplat(V, AllocaVecTy->getNumElements());

      V = convertValue(DL, IRB, V, AllocaTy);
    }

    Value *New = IRB.CreateAlignedStore(V, &NewAI, NewAI.getAlignment(),
                                        II.isVolatile());
    (void)New;
    DEBUG(dbgs() << "          to: " << *New << "\n");
    return !II.isVolatile();
  }

  bool visitMemTransferInst(MemTransferInst &II) {
    // Rewriting of memory transfer instructions can be a bit tricky. We break
    // them into two categories: split intrinsics and unsplit intrinsics.

    DEBUG(dbgs() << "    original: " << II << "\n");

    // Compute the intersecting offset range.
    assert(BeginOffset < NewAllocaEndOffset);
    assert(EndOffset > NewAllocaBeginOffset);
    uint64_t NewBeginOffset = std::max(BeginOffset, NewAllocaBeginOffset);
    uint64_t NewEndOffset = std::min(EndOffset, NewAllocaEndOffset);

    assert(II.getRawSource() == OldPtr || II.getRawDest() == OldPtr);
    bool IsDest = II.getRawDest() == OldPtr;

    // Compute the relative offset within the transfer.
    unsigned IntPtrWidth = DL.getPointerSizeInBits();
    APInt RelOffset(IntPtrWidth, NewBeginOffset - BeginOffset);

    unsigned Align = II.getAlignment();
    uint64_t SliceOffset = NewBeginOffset - NewAllocaBeginOffset;
    if (Align > 1)
      Align =
          MinAlign(RelOffset.zextOrTrunc(64).getZExtValue(),
                   MinAlign(II.getAlignment(), getOffsetAlign(SliceOffset)));

    // For unsplit intrinsics, we simply modify the source and destination
    // pointers in place. This isn't just an optimization, it is a matter of
    // correctness. With unsplit intrinsics we may be dealing with transfers
    // within a single alloca before SROA ran, or with transfers that have
    // a variable length. We may also be dealing with memmove instead of
    // memcpy, and so simply updating the pointers is the necessary for us to
    // update both source and dest of a single call.
    if (!IsSplittable) {
      Value *OldOp = IsDest ? II.getRawDest() : II.getRawSource();
      if (IsDest)
        II.setDest(
            getAdjustedAllocaPtr(IRB, BeginOffset, II.getRawDest()->getType()));
      else
        II.setSource(getAdjustedAllocaPtr(IRB, BeginOffset,
                                          II.getRawSource()->getType()));

      Type *CstTy = II.getAlignmentCst()->getType();
      II.setAlignment(ConstantInt::get(CstTy, Align));

      DEBUG(dbgs() << "          to: " << II << "\n");
      deleteIfTriviallyDead(OldOp);
      return false;
    }
    // For split transfer intrinsics we have an incredibly useful assurance:
    // the source and destination do not reside within the same alloca, and at
    // least one of them does not escape. This means that we can replace
    // memmove with memcpy, and we don't need to worry about all manner of
    // downsides to splitting and transforming the operations.

    // If this doesn't map cleanly onto the alloca type, and that type isn't
    // a single value type, just emit a memcpy.
    bool EmitMemCpy
      = !VecTy && !IntTy && (BeginOffset > NewAllocaBeginOffset ||
                             EndOffset < NewAllocaEndOffset ||
                             !NewAI.getAllocatedType()->isSingleValueType());

    // If we're just going to emit a memcpy, the alloca hasn't changed, and the
    // size hasn't been shrunk based on analysis of the viable range, this is
    // a no-op.
    if (EmitMemCpy && &OldAI == &NewAI) {
      // Ensure the start lines up.
      assert(NewBeginOffset == BeginOffset);

      // Rewrite the size as needed.
      if (NewEndOffset != EndOffset)
        II.setLength(ConstantInt::get(II.getLength()->getType(),
                                      NewEndOffset - NewBeginOffset));
      return false;
    }
    // Record this instruction for deletion.
    Pass.DeadInsts.insert(&II);

    // Strip all inbounds GEPs and pointer casts to try to dig out any root
    // alloca that should be re-examined after rewriting this instruction.
    Value *OtherPtr = IsDest ? II.getRawSource() : II.getRawDest();
    if (AllocaInst *AI
          = dyn_cast<AllocaInst>(OtherPtr->stripInBoundsOffsets()))
      Pass.Worklist.insert(AI);

    if (EmitMemCpy) {
      Type *OtherPtrTy = IsDest ? II.getRawSource()->getType()
                                : II.getRawDest()->getType();

      // Compute the other pointer, folding as much as possible to produce
      // a single, simple GEP in most cases.
      OtherPtr = getAdjustedPtr(IRB, DL, OtherPtr, RelOffset, OtherPtrTy);

      Value *OurPtr = getAdjustedAllocaPtr(
          IRB, NewBeginOffset,
          IsDest ? II.getRawDest()->getType() : II.getRawSource()->getType());
      Type *SizeTy = II.getLength()->getType();
      Constant *Size = ConstantInt::get(SizeTy, NewEndOffset - NewBeginOffset);

      CallInst *New = IRB.CreateMemCpy(IsDest ? OurPtr : OtherPtr,
                                       IsDest ? OtherPtr : OurPtr,
                                       Size, Align, II.isVolatile());
      (void)New;
      DEBUG(dbgs() << "          to: " << *New << "\n");
      return false;
    }

    // Note that we clamp the alignment to 1 here as a 0 alignment for a memcpy
    // is equivalent to 1, but that isn't true if we end up rewriting this as
    // a load or store.
    if (!Align)
      Align = 1;

    bool IsWholeAlloca = NewBeginOffset == NewAllocaBeginOffset &&
                         NewEndOffset == NewAllocaEndOffset;
    uint64_t Size = NewEndOffset - NewBeginOffset;
    unsigned BeginIndex = VecTy ? getIndex(NewBeginOffset) : 0;
    unsigned EndIndex = VecTy ? getIndex(NewEndOffset) : 0;
    unsigned NumElements = EndIndex - BeginIndex;
    IntegerType *SubIntTy
      = IntTy ? Type::getIntNTy(IntTy->getContext(), Size*8) : 0;

    Type *OtherPtrTy = NewAI.getType();
    if (VecTy && !IsWholeAlloca) {
      if (NumElements == 1)
        OtherPtrTy = VecTy->getElementType();
      else
        OtherPtrTy = VectorType::get(VecTy->getElementType(), NumElements);

      OtherPtrTy = OtherPtrTy->getPointerTo();
    } else if (IntTy && !IsWholeAlloca) {
      OtherPtrTy = SubIntTy->getPointerTo();
    }

    Value *SrcPtr = getAdjustedPtr(IRB, DL, OtherPtr, RelOffset, OtherPtrTy);
    Value *DstPtr = &NewAI;
    if (!IsDest)
      std::swap(SrcPtr, DstPtr);

    Value *Src;
    if (VecTy && !IsWholeAlloca && !IsDest) {
      Src = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                  "load");
      Src = extractVector(IRB, Src, BeginIndex, EndIndex, "vec");
    } else if (IntTy && !IsWholeAlloca && !IsDest) {
      Src = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                  "load");
      Src = convertValue(DL, IRB, Src, IntTy);
      uint64_t Offset = NewBeginOffset - NewAllocaBeginOffset;
      Src = extractInteger(DL, IRB, Src, SubIntTy, Offset, "extract");
    } else {
      Src = IRB.CreateAlignedLoad(SrcPtr, Align, II.isVolatile(),
                                  "copyload");
    }

    if (VecTy && !IsWholeAlloca && IsDest) {
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         "oldload");
      Src = insertVector(IRB, Old, Src, BeginIndex, "vec");
    } else if (IntTy && !IsWholeAlloca && IsDest) {
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         "oldload");
      Old = convertValue(DL, IRB, Old, IntTy);
      uint64_t Offset = NewBeginOffset - NewAllocaBeginOffset;
      Src = insertInteger(DL, IRB, Old, Src, Offset, "insert");
      Src = convertValue(DL, IRB, Src, NewAllocaTy);
    }

    StoreInst *Store = cast<StoreInst>(
      IRB.CreateAlignedStore(Src, DstPtr, Align, II.isVolatile()));
    (void)Store;
    DEBUG(dbgs() << "          to: " << *Store << "\n");
    return !II.isVolatile();
  }

  bool visitIntrinsicInst(IntrinsicInst &II) {
    assert(II.getIntrinsicID() == Intrinsic::lifetime_start ||
           II.getIntrinsicID() == Intrinsic::lifetime_end);
    DEBUG(dbgs() << "    original: " << II << "\n");
    assert(II.getArgOperand(1) == OldPtr);

    // Compute the intersecting offset range.
    assert(BeginOffset < NewAllocaEndOffset);
    assert(EndOffset > NewAllocaBeginOffset);
    uint64_t NewBeginOffset = std::max(BeginOffset, NewAllocaBeginOffset);
    uint64_t NewEndOffset = std::min(EndOffset, NewAllocaEndOffset);

    // Record this instruction for deletion.
    Pass.DeadInsts.insert(&II);

    ConstantInt *Size
      = ConstantInt::get(cast<IntegerType>(II.getArgOperand(0)->getType()),
                         NewEndOffset - NewBeginOffset);
    Value *Ptr =
        getAdjustedAllocaPtr(IRB, NewBeginOffset, II.getArgOperand(1)->getType());
    Value *New;
    if (II.getIntrinsicID() == Intrinsic::lifetime_start)
      New = IRB.CreateLifetimeStart(Ptr, Size);
    else
      New = IRB.CreateLifetimeEnd(Ptr, Size);

    (void)New;
    DEBUG(dbgs() << "          to: " << *New << "\n");
    return true;
  }

  bool visitPHINode(PHINode &PN) {
    DEBUG(dbgs() << "    original: " << PN << "\n");
    assert(BeginOffset >= NewAllocaBeginOffset && "PHIs are unsplittable");
    assert(EndOffset <= NewAllocaEndOffset && "PHIs are unsplittable");

    // We would like to compute a new pointer in only one place, but have it be
    // as local as possible to the PHI. To do that, we re-use the location of
    // the old pointer, which necessarily must be in the right position to
    // dominate the PHI.
    IRBuilderTy PtrBuilder(OldPtr);
    PtrBuilder.SetNamePrefix(Twine(NewAI.getName()) + "." + Twine(BeginOffset) +
                             ".");

    Value *NewPtr =
        getAdjustedAllocaPtr(PtrBuilder, BeginOffset, OldPtr->getType());
    // Replace the operands which were using the old pointer.
    std::replace(PN.op_begin(), PN.op_end(), cast<Value>(OldPtr), NewPtr);

    DEBUG(dbgs() << "          to: " << PN << "\n");
    deleteIfTriviallyDead(OldPtr);

    // Check whether we can speculate this PHI node, and if so remember that
    // fact and queue it up for another iteration after the speculation
    // occurs.
    if (isSafePHIToSpeculate(PN, &DL)) {
      Pass.SpeculatablePHIs.insert(&PN);
      IsUsedByRewrittenSpeculatableInstructions = true;
      return true;
    }

    return false; // PHIs can't be promoted on their own.
  }

  bool visitSelectInst(SelectInst &SI) {
    DEBUG(dbgs() << "    original: " << SI << "\n");
    assert((SI.getTrueValue() == OldPtr || SI.getFalseValue() == OldPtr) &&
           "Pointer isn't an operand!");
    assert(BeginOffset >= NewAllocaBeginOffset && "Selects are unsplittable");
    assert(EndOffset <= NewAllocaEndOffset && "Selects are unsplittable");

    Value *NewPtr = getAdjustedAllocaPtr(IRB, BeginOffset, OldPtr->getType());
    // Replace the operands which were using the old pointer.
    if (SI.getOperand(1) == OldPtr)
      SI.setOperand(1, NewPtr);
    if (SI.getOperand(2) == OldPtr)
      SI.setOperand(2, NewPtr);

    DEBUG(dbgs() << "          to: " << SI << "\n");
    deleteIfTriviallyDead(OldPtr);

    // Check whether we can speculate this select instruction, and if so
    // remember that fact and queue it up for another iteration after the
    // speculation occurs.
    if (isSafeSelectToSpeculate(SI, &DL)) {
      Pass.SpeculatableSelects.insert(&SI);
      IsUsedByRewrittenSpeculatableInstructions = true;
      return true;
    }

    return false; // Selects can't be promoted on their own.
  }

};
}

namespace {
/// \brief Visitor to rewrite aggregate loads and stores as scalar.
///
/// This pass aggressively rewrites all aggregate loads and stores on
/// a particular pointer (or any pointer derived from it which we can identify)
/// with scalar loads and stores.
class AggLoadStoreRewriter : public InstVisitor<AggLoadStoreRewriter, bool> {
  // Befriend the base class so it can delegate to private visit methods.
  friend class llvm::InstVisitor<AggLoadStoreRewriter, bool>;

  const DataLayout &DL;

  /// Queue of pointer uses to analyze and potentially rewrite.
  SmallVector<Use *, 8> Queue;

  /// Set to prevent us from cycling with phi nodes and loops.
  SmallPtrSet<User *, 8> Visited;

  /// The current pointer use being rewritten. This is used to dig up the used
  /// value (as opposed to the user).
  Use *U;

public:
  AggLoadStoreRewriter(const DataLayout &DL) : DL(DL) {}

  /// Rewrite loads and stores through a pointer and all pointers derived from
  /// it.
  bool rewrite(Instruction &I) {
    DEBUG(dbgs() << "  Rewriting FCA loads and stores...\n");
    enqueueUsers(I);
    bool Changed = false;
    while (!Queue.empty()) {
      U = Queue.pop_back_val();
      Changed |= visit(cast<Instruction>(U->getUser()));
    }
    return Changed;
  }

private:
  /// Enqueue all the users of the given instruction for further processing.
  /// This uses a set to de-duplicate users.
  void enqueueUsers(Instruction &I) {
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end(); UI != UE;
         ++UI)
      if (Visited.insert(*UI))
        Queue.push_back(&UI.getUse());
  }

  // Conservative default is to not rewrite anything.
  bool visitInstruction(Instruction &I) { return false; }

  /// \brief Generic recursive split emission class.
  template <typename Derived>
  class OpSplitter {
  protected:
    /// The builder used to form new instructions.
    IRBuilderTy IRB;
    /// The indices which to be used with insert- or extractvalue to select the
    /// appropriate value within the aggregate.
    SmallVector<unsigned, 4> Indices;
    /// The indices to a GEP instruction which will move Ptr to the correct slot
    /// within the aggregate.
    SmallVector<Value *, 4> GEPIndices;
    /// The base pointer of the original op, used as a base for GEPing the
    /// split operations.
    Value *Ptr;

    /// Initialize the splitter with an insertion point, Ptr and start with a
    /// single zero GEP index.
    OpSplitter(Instruction *InsertionPoint, Value *Ptr)
      : IRB(InsertionPoint), GEPIndices(1, IRB.getInt32(0)), Ptr(Ptr) {}

  public:
    /// \brief Generic recursive split emission routine.
    ///
    /// This method recursively splits an aggregate op (load or store) into
    /// scalar or vector ops. It splits recursively until it hits a single value
    /// and emits that single value operation via the template argument.
    ///
    /// The logic of this routine relies on GEPs and insertvalue and
    /// extractvalue all operating with the same fundamental index list, merely
    /// formatted differently (GEPs need actual values).
    ///
    /// \param Ty  The type being split recursively into smaller ops.
    /// \param Agg The aggregate value being built up or stored, depending on
    /// whether this is splitting a load or a store respectively.
    void emitSplitOps(Type *Ty, Value *&Agg, const Twine &Name) {
      if (Ty->isSingleValueType())
        return static_cast<Derived *>(this)->emitFunc(Ty, Agg, Name);

      if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
        unsigned OldSize = Indices.size();
        (void)OldSize;
        for (unsigned Idx = 0, Size = ATy->getNumElements(); Idx != Size;
             ++Idx) {
          assert(Indices.size() == OldSize && "Did not return to the old size");
          Indices.push_back(Idx);
          GEPIndices.push_back(IRB.getInt32(Idx));
          emitSplitOps(ATy->getElementType(), Agg, Name + "." + Twine(Idx));
          GEPIndices.pop_back();
          Indices.pop_back();
        }
        return;
      }

      if (StructType *STy = dyn_cast<StructType>(Ty)) {
        unsigned OldSize = Indices.size();
        (void)OldSize;
        for (unsigned Idx = 0, Size = STy->getNumElements(); Idx != Size;
             ++Idx) {
          assert(Indices.size() == OldSize && "Did not return to the old size");
          Indices.push_back(Idx);
          GEPIndices.push_back(IRB.getInt32(Idx));
          emitSplitOps(STy->getElementType(Idx), Agg, Name + "." + Twine(Idx));
          GEPIndices.pop_back();
          Indices.pop_back();
        }
        return;
      }

      llvm_unreachable("Only arrays and structs are aggregate loadable types");
    }
  };

  struct LoadOpSplitter : public OpSplitter<LoadOpSplitter> {
    LoadOpSplitter(Instruction *InsertionPoint, Value *Ptr)
      : OpSplitter<LoadOpSplitter>(InsertionPoint, Ptr) {}

    /// Emit a leaf load of a single value. This is called at the leaves of the
    /// recursive emission to actually load values.
    void emitFunc(Type *Ty, Value *&Agg, const Twine &Name) {
      assert(Ty->isSingleValueType());
      // Load the single value and insert it using the indices.
      Value *GEP = IRB.CreateInBoundsGEP(Ptr, GEPIndices, Name + ".gep");
      Value *Load = IRB.CreateLoad(GEP, Name + ".load");
      Agg = IRB.CreateInsertValue(Agg, Load, Indices, Name + ".insert");
      DEBUG(dbgs() << "          to: " << *Load << "\n");
    }
  };

  bool visitLoadInst(LoadInst &LI) {
    assert(LI.getPointerOperand() == *U);
    if (!LI.isSimple() || LI.getType()->isSingleValueType())
      return false;

    // We have an aggregate being loaded, split it apart.
    DEBUG(dbgs() << "    original: " << LI << "\n");
    LoadOpSplitter Splitter(&LI, *U);
    Value *V = UndefValue::get(LI.getType());
    Splitter.emitSplitOps(LI.getType(), V, LI.getName() + ".fca");
    LI.replaceAllUsesWith(V);
    LI.eraseFromParent();
    return true;
  }

  struct StoreOpSplitter : public OpSplitter<StoreOpSplitter> {
    StoreOpSplitter(Instruction *InsertionPoint, Value *Ptr)
      : OpSplitter<StoreOpSplitter>(InsertionPoint, Ptr) {}

    /// Emit a leaf store of a single value. This is called at the leaves of the
    /// recursive emission to actually produce stores.
    void emitFunc(Type *Ty, Value *&Agg, const Twine &Name) {
      assert(Ty->isSingleValueType());
      // Extract the single value and store it using the indices.
      Value *Store = IRB.CreateStore(
        IRB.CreateExtractValue(Agg, Indices, Name + ".extract"),
        IRB.CreateInBoundsGEP(Ptr, GEPIndices, Name + ".gep"));
      (void)Store;
      DEBUG(dbgs() << "          to: " << *Store << "\n");
    }
  };

  bool visitStoreInst(StoreInst &SI) {
    if (!SI.isSimple() || SI.getPointerOperand() != *U)
      return false;
    Value *V = SI.getValueOperand();
    if (V->getType()->isSingleValueType())
      return false;

    // We have an aggregate being stored, split it apart.
    DEBUG(dbgs() << "    original: " << SI << "\n");
    StoreOpSplitter Splitter(&SI, *U);
    Splitter.emitSplitOps(V->getType(), V, V->getName() + ".fca");
    SI.eraseFromParent();
    return true;
  }

  bool visitBitCastInst(BitCastInst &BC) {
    enqueueUsers(BC);
    return false;
  }

  bool visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    enqueueUsers(GEPI);
    return false;
  }

  bool visitPHINode(PHINode &PN) {
    enqueueUsers(PN);
    return false;
  }

  bool visitSelectInst(SelectInst &SI) {
    enqueueUsers(SI);
    return false;
  }
};
}

/// \brief Strip aggregate type wrapping.
///
/// This removes no-op aggregate types wrapping an underlying type. It will
/// strip as many layers of types as it can without changing either the type
/// size or the allocated size.
static Type *stripAggregateTypeWrapping(const DataLayout &DL, Type *Ty) {
  if (Ty->isSingleValueType())
    return Ty;

  uint64_t AllocSize = DL.getTypeAllocSize(Ty);
  uint64_t TypeSize = DL.getTypeSizeInBits(Ty);

  Type *InnerTy;
  if (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
    InnerTy = ArrTy->getElementType();
  } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(STy);
    unsigned Index = SL->getElementContainingOffset(0);
    InnerTy = STy->getElementType(Index);
  } else {
    return Ty;
  }

  if (AllocSize > DL.getTypeAllocSize(InnerTy) ||
      TypeSize > DL.getTypeSizeInBits(InnerTy))
    return Ty;

  return stripAggregateTypeWrapping(DL, InnerTy);
}

/// \brief Try to find a partition of the aggregate type passed in for a given
/// offset and size.
///
/// This recurses through the aggregate type and tries to compute a subtype
/// based on the offset and size. When the offset and size span a sub-section
/// of an array, it will even compute a new array type for that sub-section,
/// and the same for structs.
///
/// Note that this routine is very strict and tries to find a partition of the
/// type which produces the *exact* right offset and size. It is not forgiving
/// when the size or offset cause either end of type-based partition to be off.
/// Also, this is a best-effort routine. It is reasonable to give up and not
/// return a type if necessary.
static Type *getTypePartition(const DataLayout &DL, Type *Ty,
                              uint64_t Offset, uint64_t Size) {
  if (Offset == 0 && DL.getTypeAllocSize(Ty) == Size)
    return stripAggregateTypeWrapping(DL, Ty);
  if (Offset > DL.getTypeAllocSize(Ty) ||
      (DL.getTypeAllocSize(Ty) - Offset) < Size)
    return 0;

  if (SequentialType *SeqTy = dyn_cast<SequentialType>(Ty)) {
    // We can't partition pointers...
    if (SeqTy->isPointerTy())
      return 0;

    Type *ElementTy = SeqTy->getElementType();
    uint64_t ElementSize = DL.getTypeAllocSize(ElementTy);
    uint64_t NumSkippedElements = Offset / ElementSize;
    if (ArrayType *ArrTy = dyn_cast<ArrayType>(SeqTy)) {
      if (NumSkippedElements >= ArrTy->getNumElements())
        return 0;
    } else if (VectorType *VecTy = dyn_cast<VectorType>(SeqTy)) {
      if (NumSkippedElements >= VecTy->getNumElements())
        return 0;
    }
    Offset -= NumSkippedElements * ElementSize;

    // First check if we need to recurse.
    if (Offset > 0 || Size < ElementSize) {
      // Bail if the partition ends in a different array element.
      if ((Offset + Size) > ElementSize)
        return 0;
      // Recurse through the element type trying to peel off offset bytes.
      return getTypePartition(DL, ElementTy, Offset, Size);
    }
    assert(Offset == 0);

    if (Size == ElementSize)
      return stripAggregateTypeWrapping(DL, ElementTy);
    assert(Size > ElementSize);
    uint64_t NumElements = Size / ElementSize;
    if (NumElements * ElementSize != Size)
      return 0;
    return ArrayType::get(ElementTy, NumElements);
  }

  StructType *STy = dyn_cast<StructType>(Ty);
  if (!STy)
    return 0;

  const StructLayout *SL = DL.getStructLayout(STy);
  if (Offset >= SL->getSizeInBytes())
    return 0;
  uint64_t EndOffset = Offset + Size;
  if (EndOffset > SL->getSizeInBytes())
    return 0;

  unsigned Index = SL->getElementContainingOffset(Offset);
  Offset -= SL->getElementOffset(Index);

  Type *ElementTy = STy->getElementType(Index);
  uint64_t ElementSize = DL.getTypeAllocSize(ElementTy);
  if (Offset >= ElementSize)
    return 0; // The offset points into alignment padding.

  // See if any partition must be contained by the element.
  if (Offset > 0 || Size < ElementSize) {
    if ((Offset + Size) > ElementSize)
      return 0;
    return getTypePartition(DL, ElementTy, Offset, Size);
  }
  assert(Offset == 0);

  if (Size == ElementSize)
    return stripAggregateTypeWrapping(DL, ElementTy);

  StructType::element_iterator EI = STy->element_begin() + Index,
                               EE = STy->element_end();
  if (EndOffset < SL->getSizeInBytes()) {
    unsigned EndIndex = SL->getElementContainingOffset(EndOffset);
    if (Index == EndIndex)
      return 0; // Within a single element and its padding.

    // Don't try to form "natural" types if the elements don't line up with the
    // expected size.
    // FIXME: We could potentially recurse down through the last element in the
    // sub-struct to find a natural end point.
    if (SL->getElementOffset(EndIndex) != EndOffset)
      return 0;

    assert(Index < EndIndex);
    EE = STy->element_begin() + EndIndex;
  }

  // Try to build up a sub-structure.
  StructType *SubTy = StructType::get(STy->getContext(), makeArrayRef(EI, EE),
                                      STy->isPacked());
  const StructLayout *SubSL = DL.getStructLayout(SubTy);
  if (Size != SubSL->getSizeInBytes())
    return 0; // The sub-struct doesn't have quite the size needed.

  return SubTy;
}

/// \brief Rewrite an alloca partition's users.
///
/// This routine drives both of the rewriting goals of the SROA pass. It tries
/// to rewrite uses of an alloca partition to be conducive for SSA value
/// promotion. If the partition needs a new, more refined alloca, this will
/// build that new alloca, preserving as much type information as possible, and
/// rewrite the uses of the old alloca to point at the new one and have the
/// appropriate new offsets. It also evaluates how successful the rewrite was
/// at enabling promotion and if it was successful queues the alloca to be
/// promoted.
bool SROA::rewritePartition(AllocaInst &AI, AllocaSlices &S,
                            AllocaSlices::iterator B, AllocaSlices::iterator E,
                            int64_t BeginOffset, int64_t EndOffset,
                            ArrayRef<AllocaSlices::iterator> SplitUses) {
  assert(BeginOffset < EndOffset);
  uint64_t SliceSize = EndOffset - BeginOffset;

  // Try to compute a friendly type for this partition of the alloca. This
  // won't always succeed, in which case we fall back to a legal integer type
  // or an i8 array of an appropriate size.
  Type *SliceTy = 0;
  if (Type *CommonUseTy = findCommonType(B, E, EndOffset))
    if (DL->getTypeAllocSize(CommonUseTy) >= SliceSize)
      SliceTy = CommonUseTy;
  if (!SliceTy)
    if (Type *TypePartitionTy = getTypePartition(*DL, AI.getAllocatedType(),
                                                 BeginOffset, SliceSize))
      SliceTy = TypePartitionTy;
  if ((!SliceTy || (SliceTy->isArrayTy() &&
                    SliceTy->getArrayElementType()->isIntegerTy())) &&
      DL->isLegalInteger(SliceSize * 8))
    SliceTy = Type::getIntNTy(*C, SliceSize * 8);
  if (!SliceTy)
    SliceTy = ArrayType::get(Type::getInt8Ty(*C), SliceSize);
  assert(DL->getTypeAllocSize(SliceTy) >= SliceSize);

  bool IsVectorPromotable = isVectorPromotionViable(
      *DL, SliceTy, S, BeginOffset, EndOffset, B, E, SplitUses);

  bool IsIntegerPromotable =
      !IsVectorPromotable &&
      isIntegerWideningViable(*DL, SliceTy, BeginOffset, S, B, E, SplitUses);

  // Check for the case where we're going to rewrite to a new alloca of the
  // exact same type as the original, and with the same access offsets. In that
  // case, re-use the existing alloca, but still run through the rewriter to
  // perform phi and select speculation.
  AllocaInst *NewAI;
  if (SliceTy == AI.getAllocatedType()) {
    assert(BeginOffset == 0 &&
           "Non-zero begin offset but same alloca type");
    NewAI = &AI;
    // FIXME: We should be able to bail at this point with "nothing changed".
    // FIXME: We might want to defer PHI speculation until after here.
  } else {
    unsigned Alignment = AI.getAlignment();
    if (!Alignment) {
      // The minimum alignment which users can rely on when the explicit
      // alignment is omitted or zero is that required by the ABI for this
      // type.
      Alignment = DL->getABITypeAlignment(AI.getAllocatedType());
    }
    Alignment = MinAlign(Alignment, BeginOffset);
    // If we will get at least this much alignment from the type alone, leave
    // the alloca's alignment unconstrained.
    if (Alignment <= DL->getABITypeAlignment(SliceTy))
      Alignment = 0;
    NewAI = new AllocaInst(SliceTy, 0, Alignment,
                           AI.getName() + ".sroa." + Twine(B - S.begin()), &AI);
    ++NumNewAllocas;
  }

  DEBUG(dbgs() << "Rewriting alloca partition "
               << "[" << BeginOffset << "," << EndOffset << ") to: " << *NewAI
               << "\n");

  // Track the high watermark on several worklists that are only relevant for
  // promoted allocas. We will reset it to this point if the alloca is not in
  // fact scheduled for promotion.
  unsigned PPWOldSize = PostPromotionWorklist.size();
  unsigned SPOldSize = SpeculatablePHIs.size();
  unsigned SSOldSize = SpeculatableSelects.size();
  unsigned NumUses = 0;

  AllocaSliceRewriter Rewriter(*DL, S, *this, AI, *NewAI, BeginOffset,
                               EndOffset, IsVectorPromotable,
                               IsIntegerPromotable);
  bool Promotable = true;
  for (ArrayRef<AllocaSlices::iterator>::const_iterator SUI = SplitUses.begin(),
                                                        SUE = SplitUses.end();
       SUI != SUE; ++SUI) {
    DEBUG(dbgs() << "  rewriting split ");
    DEBUG(S.printSlice(dbgs(), *SUI, ""));
    Promotable &= Rewriter.visit(*SUI);
    ++NumUses;
  }
  for (AllocaSlices::iterator I = B; I != E; ++I) {
    DEBUG(dbgs() << "  rewriting ");
    DEBUG(S.printSlice(dbgs(), I, ""));
    Promotable &= Rewriter.visit(I);
    ++NumUses;
  }

  NumAllocaPartitionUses += NumUses;
  MaxUsesPerAllocaPartition =
      std::max<unsigned>(NumUses, MaxUsesPerAllocaPartition);

  if (Promotable && !Rewriter.isUsedByRewrittenSpeculatableInstructions()) {
    DEBUG(dbgs() << "  and queuing for promotion\n");
    PromotableAllocas.push_back(NewAI);
  } else if (NewAI != &AI ||
             (Promotable &&
              Rewriter.isUsedByRewrittenSpeculatableInstructions())) {
    // If we can't promote the alloca, iterate on it to check for new
    // refinements exposed by splitting the current alloca. Don't iterate on an
    // alloca which didn't actually change and didn't get promoted.
    //
    // Alternatively, if we could promote the alloca but have speculatable
    // instructions then we will speculate them after finishing our processing
    // of the original alloca. Mark the new one for re-visiting in the next
    // iteration so the speculated operations can be rewritten.
    //
    // FIXME: We should actually track whether the rewriter changed anything.
    Worklist.insert(NewAI);
  }

  // Drop any post-promotion work items if promotion didn't happen.
  if (!Promotable) {
    while (PostPromotionWorklist.size() > PPWOldSize)
      PostPromotionWorklist.pop_back();
    while (SpeculatablePHIs.size() > SPOldSize)
      SpeculatablePHIs.pop_back();
    while (SpeculatableSelects.size() > SSOldSize)
      SpeculatableSelects.pop_back();
  }

  return true;
}

namespace {
struct IsSliceEndLessOrEqualTo {
  uint64_t UpperBound;

  IsSliceEndLessOrEqualTo(uint64_t UpperBound) : UpperBound(UpperBound) {}

  bool operator()(const AllocaSlices::iterator &I) {
    return I->endOffset() <= UpperBound;
  }
};
}

static void
removeFinishedSplitUses(SmallVectorImpl<AllocaSlices::iterator> &SplitUses,
                        uint64_t &MaxSplitUseEndOffset, uint64_t Offset) {
  if (Offset >= MaxSplitUseEndOffset) {
    SplitUses.clear();
    MaxSplitUseEndOffset = 0;
    return;
  }

  size_t SplitUsesOldSize = SplitUses.size();
  SplitUses.erase(std::remove_if(SplitUses.begin(), SplitUses.end(),
                                 IsSliceEndLessOrEqualTo(Offset)),
                  SplitUses.end());
  if (SplitUsesOldSize == SplitUses.size())
    return;

  // Recompute the max. While this is linear, so is remove_if.
  MaxSplitUseEndOffset = 0;
  for (SmallVectorImpl<AllocaSlices::iterator>::iterator
           SUI = SplitUses.begin(),
           SUE = SplitUses.end();
       SUI != SUE; ++SUI)
    MaxSplitUseEndOffset = std::max((*SUI)->endOffset(), MaxSplitUseEndOffset);
}

/// \brief Walks the slices of an alloca and form partitions based on them,
/// rewriting each of their uses.
bool SROA::splitAlloca(AllocaInst &AI, AllocaSlices &S) {
  if (S.begin() == S.end())
    return false;

  unsigned NumPartitions = 0;
  bool Changed = false;
  SmallVector<AllocaSlices::iterator, 4> SplitUses;
  uint64_t MaxSplitUseEndOffset = 0;

  uint64_t BeginOffset = S.begin()->beginOffset();

  for (AllocaSlices::iterator SI = S.begin(), SJ = llvm::next(SI), SE = S.end();
       SI != SE; SI = SJ) {
    uint64_t MaxEndOffset = SI->endOffset();

    if (!SI->isSplittable()) {
      // When we're forming an unsplittable region, it must always start at the
      // first slice and will extend through its end.
      assert(BeginOffset == SI->beginOffset());

      // Form a partition including all of the overlapping slices with this
      // unsplittable slice.
      while (SJ != SE && SJ->beginOffset() < MaxEndOffset) {
        if (!SJ->isSplittable())
          MaxEndOffset = std::max(MaxEndOffset, SJ->endOffset());
        ++SJ;
      }
    } else {
      assert(SI->isSplittable()); // Established above.

      // Collect all of the overlapping splittable slices.
      while (SJ != SE && SJ->beginOffset() < MaxEndOffset &&
             SJ->isSplittable()) {
        MaxEndOffset = std::max(MaxEndOffset, SJ->endOffset());
        ++SJ;
      }

      // Back up MaxEndOffset and SJ if we ended the span early when
      // encountering an unsplittable slice.
      if (SJ != SE && SJ->beginOffset() < MaxEndOffset) {
        assert(!SJ->isSplittable());
        MaxEndOffset = SJ->beginOffset();
      }
    }

    // Check if we have managed to move the end offset forward yet. If so,
    // we'll have to rewrite uses and erase old split uses.
    if (BeginOffset < MaxEndOffset) {
      // Rewrite a sequence of overlapping slices.
      Changed |=
          rewritePartition(AI, S, SI, SJ, BeginOffset, MaxEndOffset, SplitUses);
      ++NumPartitions;

      removeFinishedSplitUses(SplitUses, MaxSplitUseEndOffset, MaxEndOffset);
    }

    // Accumulate all the splittable slices from the [SI,SJ) region which
    // overlap going forward.
    for (AllocaSlices::iterator SK = SI; SK != SJ; ++SK)
      if (SK->isSplittable() && SK->endOffset() > MaxEndOffset) {
        SplitUses.push_back(SK);
        MaxSplitUseEndOffset = std::max(SK->endOffset(), MaxSplitUseEndOffset);
      }

    // If we're already at the end and we have no split uses, we're done.
    if (SJ == SE && SplitUses.empty())
      break;

    // If we have no split uses or no gap in offsets, we're ready to move to
    // the next slice.
    if (SplitUses.empty() || (SJ != SE && MaxEndOffset == SJ->beginOffset())) {
      BeginOffset = SJ->beginOffset();
      continue;
    }

    // Even if we have split slices, if the next slice is splittable and the
    // split slices reach it, we can simply set up the beginning offset of the
    // next iteration to bridge between them.
    if (SJ != SE && SJ->isSplittable() &&
        MaxSplitUseEndOffset > SJ->beginOffset()) {
      BeginOffset = MaxEndOffset;
      continue;
    }

    // Otherwise, we have a tail of split slices. Rewrite them with an empty
    // range of slices.
    uint64_t PostSplitEndOffset =
        SJ == SE ? MaxSplitUseEndOffset : SJ->beginOffset();

    Changed |= rewritePartition(AI, S, SJ, SJ, MaxEndOffset, PostSplitEndOffset,
                                SplitUses);
    ++NumPartitions;

    if (SJ == SE)
      break; // Skip the rest, we don't need to do any cleanup.

    removeFinishedSplitUses(SplitUses, MaxSplitUseEndOffset,
                            PostSplitEndOffset);

    // Now just reset the begin offset for the next iteration.
    BeginOffset = SJ->beginOffset();
  }

  NumAllocaPartitions += NumPartitions;
  MaxPartitionsPerAlloca =
      std::max<unsigned>(NumPartitions, MaxPartitionsPerAlloca);

  return Changed;
}

/// \brief Analyze an alloca for SROA.
///
/// This analyzes the alloca to ensure we can reason about it, builds
/// the slices of the alloca, and then hands it off to be split and
/// rewritten as needed.
bool SROA::runOnAlloca(AllocaInst &AI) {
  DEBUG(dbgs() << "SROA alloca: " << AI << "\n");
  ++NumAllocasAnalyzed;

  // Special case dead allocas, as they're trivial.
  if (AI.use_empty()) {
    AI.eraseFromParent();
    return true;
  }

  // Skip alloca forms that this analysis can't handle.
  if (AI.isArrayAllocation() || !AI.getAllocatedType()->isSized() ||
      DL->getTypeAllocSize(AI.getAllocatedType()) == 0)
    return false;

  bool Changed = false;

  // First, split any FCA loads and stores touching this alloca to promote
  // better splitting and promotion opportunities.
  AggLoadStoreRewriter AggRewriter(*DL);
  Changed |= AggRewriter.rewrite(AI);

  // Build the slices using a recursive instruction-visiting builder.
  AllocaSlices S(*DL, AI);
  DEBUG(S.print(dbgs()));
  if (S.isEscaped())
    return Changed;

  // Delete all the dead users of this alloca before splitting and rewriting it.
  for (AllocaSlices::dead_user_iterator DI = S.dead_user_begin(),
                                        DE = S.dead_user_end();
       DI != DE; ++DI) {
    Changed = true;
    (*DI)->replaceAllUsesWith(UndefValue::get((*DI)->getType()));
    DeadInsts.insert(*DI);
  }
  for (AllocaSlices::dead_op_iterator DO = S.dead_op_begin(),
                                      DE = S.dead_op_end();
       DO != DE; ++DO) {
    Value *OldV = **DO;
    // Clobber the use with an undef value.
    **DO = UndefValue::get(OldV->getType());
    if (Instruction *OldI = dyn_cast<Instruction>(OldV))
      if (isInstructionTriviallyDead(OldI)) {
        Changed = true;
        DeadInsts.insert(OldI);
      }
  }

  // No slices to split. Leave the dead alloca for a later pass to clean up.
  if (S.begin() == S.end())
    return Changed;

  Changed |= splitAlloca(AI, S);

  DEBUG(dbgs() << "  Speculating PHIs\n");
  while (!SpeculatablePHIs.empty())
    speculatePHINodeLoads(*SpeculatablePHIs.pop_back_val());

  DEBUG(dbgs() << "  Speculating Selects\n");
  while (!SpeculatableSelects.empty())
    speculateSelectInstLoads(*SpeculatableSelects.pop_back_val());

  return Changed;
}

/// \brief Delete the dead instructions accumulated in this run.
///
/// Recursively deletes the dead instructions we've accumulated. This is done
/// at the very end to maximize locality of the recursive delete and to
/// minimize the problems of invalidated instruction pointers as such pointers
/// are used heavily in the intermediate stages of the algorithm.
///
/// We also record the alloca instructions deleted here so that they aren't
/// subsequently handed to mem2reg to promote.
void SROA::deleteDeadInstructions(SmallPtrSet<AllocaInst*, 4> &DeletedAllocas) {
  while (!DeadInsts.empty()) {
    Instruction *I = DeadInsts.pop_back_val();
    DEBUG(dbgs() << "Deleting dead instruction: " << *I << "\n");

    I->replaceAllUsesWith(UndefValue::get(I->getType()));

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        // Zero out the operand and see if it becomes trivially dead.
        *OI = 0;
        if (isInstructionTriviallyDead(U))
          DeadInsts.insert(U);
      }

    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      DeletedAllocas.insert(AI);

    ++NumDeleted;
    I->eraseFromParent();
  }
}

static void enqueueUsersInWorklist(Instruction &I,
                                   SmallVectorImpl<Instruction *> &Worklist,
                                   SmallPtrSet<Instruction *, 8> &Visited) {
  for (Value::use_iterator UI = I.use_begin(), UE = I.use_end(); UI != UE;
       ++UI)
    if (Visited.insert(cast<Instruction>(*UI)))
      Worklist.push_back(cast<Instruction>(*UI));
}

/// \brief Promote the allocas, using the best available technique.
///
/// This attempts to promote whatever allocas have been identified as viable in
/// the PromotableAllocas list. If that list is empty, there is nothing to do.
/// If there is a domtree available, we attempt to promote using the full power
/// of mem2reg. Otherwise, we build and use the AllocaPromoter above which is
/// based on the SSAUpdater utilities. This function returns whether any
/// promotion occurred.
bool SROA::promoteAllocas(Function &F) {
  if (PromotableAllocas.empty())
    return false;

  NumPromoted += PromotableAllocas.size();

  if (DT && !ForceSSAUpdater) {
    DEBUG(dbgs() << "Promoting allocas with mem2reg...\n");
    PromoteMemToReg(PromotableAllocas, *DT);
    PromotableAllocas.clear();
    return true;
  }

  DEBUG(dbgs() << "Promoting allocas with SSAUpdater...\n");
  SSAUpdater SSA;
  DIBuilder DIB(*F.getParent());
  SmallVector<Instruction *, 64> Insts;

  // We need a worklist to walk the uses of each alloca.
  SmallVector<Instruction *, 8> Worklist;
  SmallPtrSet<Instruction *, 8> Visited;
  SmallVector<Instruction *, 32> DeadInsts;

  for (unsigned Idx = 0, Size = PromotableAllocas.size(); Idx != Size; ++Idx) {
    AllocaInst *AI = PromotableAllocas[Idx];
    Insts.clear();
    Worklist.clear();
    Visited.clear();

    enqueueUsersInWorklist(*AI, Worklist, Visited);

    while (!Worklist.empty()) {
      Instruction *I = Worklist.pop_back_val();

      // FIXME: Currently the SSAUpdater infrastructure doesn't reason about
      // lifetime intrinsics and so we strip them (and the bitcasts+GEPs
      // leading to them) here. Eventually it should use them to optimize the
      // scalar values produced.
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        assert(II->getIntrinsicID() == Intrinsic::lifetime_start ||
               II->getIntrinsicID() == Intrinsic::lifetime_end);
        II->eraseFromParent();
        continue;
      }

      // Push the loads and stores we find onto the list. SROA will already
      // have validated that all loads and stores are viable candidates for
      // promotion.
      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        assert(LI->getType() == AI->getAllocatedType());
        Insts.push_back(LI);
        continue;
      }
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        assert(SI->getValueOperand()->getType() == AI->getAllocatedType());
        Insts.push_back(SI);
        continue;
      }

      // For everything else, we know that only no-op bitcasts and GEPs will
      // make it this far, just recurse through them and recall them for later
      // removal.
      DeadInsts.push_back(I);
      enqueueUsersInWorklist(*I, Worklist, Visited);
    }
    AllocaPromoter(Insts, SSA, *AI, DIB).run(Insts);
    while (!DeadInsts.empty())
      DeadInsts.pop_back_val()->eraseFromParent();
    AI->eraseFromParent();
  }

  PromotableAllocas.clear();
  return true;
}

namespace {
  /// \brief A predicate to test whether an alloca belongs to a set.
  class IsAllocaInSet {
    typedef SmallPtrSet<AllocaInst *, 4> SetType;
    const SetType &Set;

  public:
    typedef AllocaInst *argument_type;

    IsAllocaInSet(const SetType &Set) : Set(Set) {}
    bool operator()(AllocaInst *AI) const { return Set.count(AI); }
  };
}

bool SROA::runOnFunction(Function &F) {
  DEBUG(dbgs() << "SROA function: " << F.getName() << "\n");
  C = &F.getContext();
  DL = getAnalysisIfAvailable<DataLayout>();
  if (!DL) {
    DEBUG(dbgs() << "  Skipping SROA -- no target data!\n");
    return false;
  }
  DT = getAnalysisIfAvailable<DominatorTree>();

  BasicBlock &EntryBB = F.getEntryBlock();
  for (BasicBlock::iterator I = EntryBB.begin(), E = llvm::prior(EntryBB.end());
       I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      Worklist.insert(AI);

  bool Changed = false;
  // A set of deleted alloca instruction pointers which should be removed from
  // the list of promotable allocas.
  SmallPtrSet<AllocaInst *, 4> DeletedAllocas;

  do {
    while (!Worklist.empty()) {
      Changed |= runOnAlloca(*Worklist.pop_back_val());
      deleteDeadInstructions(DeletedAllocas);

      // Remove the deleted allocas from various lists so that we don't try to
      // continue processing them.
      if (!DeletedAllocas.empty()) {
        Worklist.remove_if(IsAllocaInSet(DeletedAllocas));
        PostPromotionWorklist.remove_if(IsAllocaInSet(DeletedAllocas));
        PromotableAllocas.erase(std::remove_if(PromotableAllocas.begin(),
                                               PromotableAllocas.end(),
                                               IsAllocaInSet(DeletedAllocas)),
                                PromotableAllocas.end());
        DeletedAllocas.clear();
      }
    }

    Changed |= promoteAllocas(F);

    Worklist = PostPromotionWorklist;
    PostPromotionWorklist.clear();
  } while (!Worklist.empty());

  return Changed;
}

void SROA::getAnalysisUsage(AnalysisUsage &AU) const {
  if (RequiresDomTree)
    AU.addRequired<DominatorTree>();
  AU.setPreservesCFG();
}
