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
#include "llvm/Constants.h"
#include "llvm/DIBuilder.h"
#include "llvm/DebugInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Operator.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

STATISTIC(NumAllocasAnalyzed, "Number of allocas analyzed for replacement");
STATISTIC(NumNewAllocas,      "Number of new, smaller allocas introduced");
STATISTIC(NumPromoted,        "Number of allocas promoted to SSA values");
STATISTIC(NumLoadsSpeculated, "Number of loads speculated to allow promotion");
STATISTIC(NumDeleted,         "Number of instructions deleted");
STATISTIC(NumVectorized,      "Number of vectorized aggregates");

/// Hidden option to force the pass to not use DomTree and mem2reg, instead
/// forming SSA values through the SSAUpdater infrastructure.
static cl::opt<bool>
ForceSSAUpdater("force-ssa-updater", cl::init(false), cl::Hidden);

namespace {
/// \brief Alloca partitioning representation.
///
/// This class represents a partitioning of an alloca into slices, and
/// information about the nature of uses of each slice of the alloca. The goal
/// is that this information is sufficient to decide if and how to split the
/// alloca apart and replace slices with scalars. It is also intended that this
/// structure can capture the relevant information needed both to decide about
/// and to enact these transformations.
class AllocaPartitioning {
public:
  /// \brief A common base class for representing a half-open byte range.
  struct ByteRange {
    /// \brief The beginning offset of the range.
    uint64_t BeginOffset;

    /// \brief The ending offset, not included in the range.
    uint64_t EndOffset;

    ByteRange() : BeginOffset(), EndOffset() {}
    ByteRange(uint64_t BeginOffset, uint64_t EndOffset)
        : BeginOffset(BeginOffset), EndOffset(EndOffset) {}

    /// \brief Support for ordering ranges.
    ///
    /// This provides an ordering over ranges such that start offsets are
    /// always increasing, and within equal start offsets, the end offsets are
    /// decreasing. Thus the spanning range comes first in a cluster with the
    /// same start position.
    bool operator<(const ByteRange &RHS) const {
      if (BeginOffset < RHS.BeginOffset) return true;
      if (BeginOffset > RHS.BeginOffset) return false;
      if (EndOffset > RHS.EndOffset) return true;
      return false;
    }

    /// \brief Support comparison with a single offset to allow binary searches.
    friend bool operator<(const ByteRange &LHS, uint64_t RHSOffset) {
      return LHS.BeginOffset < RHSOffset;
    }

    friend LLVM_ATTRIBUTE_UNUSED bool operator<(uint64_t LHSOffset,
                                                const ByteRange &RHS) {
      return LHSOffset < RHS.BeginOffset;
    }

    bool operator==(const ByteRange &RHS) const {
      return BeginOffset == RHS.BeginOffset && EndOffset == RHS.EndOffset;
    }
    bool operator!=(const ByteRange &RHS) const { return !operator==(RHS); }
  };

  /// \brief A partition of an alloca.
  ///
  /// This structure represents a contiguous partition of the alloca. These are
  /// formed by examining the uses of the alloca. During formation, they may
  /// overlap but once an AllocaPartitioning is built, the Partitions within it
  /// are all disjoint.
  struct Partition : public ByteRange {
    /// \brief Whether this partition is splittable into smaller partitions.
    ///
    /// We flag partitions as splittable when they are formed entirely due to
    /// accesses by trivially splittable operations such as memset and memcpy.
    bool IsSplittable;

    /// \brief Test whether a partition has been marked as dead.
    bool isDead() const {
      if (BeginOffset == UINT64_MAX) {
        assert(EndOffset == UINT64_MAX);
        return true;
      }
      return false;
    }

    /// \brief Kill a partition.
    /// This is accomplished by setting both its beginning and end offset to
    /// the maximum possible value.
    void kill() {
      assert(!isDead() && "He's Dead, Jim!");
      BeginOffset = EndOffset = UINT64_MAX;
    }

    Partition() : ByteRange(), IsSplittable() {}
    Partition(uint64_t BeginOffset, uint64_t EndOffset, bool IsSplittable)
        : ByteRange(BeginOffset, EndOffset), IsSplittable(IsSplittable) {}
  };

  /// \brief A particular use of a partition of the alloca.
  ///
  /// This structure is used to associate uses of a partition with it. They
  /// mark the range of bytes which are referenced by a particular instruction,
  /// and includes a handle to the user itself and the pointer value in use.
  /// The bounds of these uses are determined by intersecting the bounds of the
  /// memory use itself with a particular partition. As a consequence there is
  /// intentionally overlap between various uses of the same partition.
  struct PartitionUse : public ByteRange {
    /// \brief The use in question. Provides access to both user and used value.
    ///
    /// Note that this may be null if the partition use is *dead*, that is, it
    /// should be ignored.
    Use *U;

    PartitionUse() : ByteRange(), U() {}
    PartitionUse(uint64_t BeginOffset, uint64_t EndOffset, Use *U)
        : ByteRange(BeginOffset, EndOffset), U(U) {}
  };

  /// \brief Construct a partitioning of a particular alloca.
  ///
  /// Construction does most of the work for partitioning the alloca. This
  /// performs the necessary walks of users and builds a partitioning from it.
  AllocaPartitioning(const DataLayout &TD, AllocaInst &AI);

  /// \brief Test whether a pointer to the allocation escapes our analysis.
  ///
  /// If this is true, the partitioning is never fully built and should be
  /// ignored.
  bool isEscaped() const { return PointerEscapingInstr; }

  /// \brief Support for iterating over the partitions.
  /// @{
  typedef SmallVectorImpl<Partition>::iterator iterator;
  iterator begin() { return Partitions.begin(); }
  iterator end() { return Partitions.end(); }

  typedef SmallVectorImpl<Partition>::const_iterator const_iterator;
  const_iterator begin() const { return Partitions.begin(); }
  const_iterator end() const { return Partitions.end(); }
  /// @}

  /// \brief Support for iterating over and manipulating a particular
  /// partition's uses.
  ///
  /// The iteration support provided for uses is more limited, but also
  /// includes some manipulation routines to support rewriting the uses of
  /// partitions during SROA.
  /// @{
  typedef SmallVectorImpl<PartitionUse>::iterator use_iterator;
  use_iterator use_begin(unsigned Idx) { return Uses[Idx].begin(); }
  use_iterator use_begin(const_iterator I) { return Uses[I - begin()].begin(); }
  use_iterator use_end(unsigned Idx) { return Uses[Idx].end(); }
  use_iterator use_end(const_iterator I) { return Uses[I - begin()].end(); }

  typedef SmallVectorImpl<PartitionUse>::const_iterator const_use_iterator;
  const_use_iterator use_begin(unsigned Idx) const { return Uses[Idx].begin(); }
  const_use_iterator use_begin(const_iterator I) const {
    return Uses[I - begin()].begin();
  }
  const_use_iterator use_end(unsigned Idx) const { return Uses[Idx].end(); }
  const_use_iterator use_end(const_iterator I) const {
    return Uses[I - begin()].end();
  }

  unsigned use_size(unsigned Idx) const { return Uses[Idx].size(); }
  unsigned use_size(const_iterator I) const { return Uses[I - begin()].size(); }
  const PartitionUse &getUse(unsigned PIdx, unsigned UIdx) const {
    return Uses[PIdx][UIdx];
  }
  const PartitionUse &getUse(const_iterator I, unsigned UIdx) const {
    return Uses[I - begin()][UIdx];
  }

  void use_push_back(unsigned Idx, const PartitionUse &PU) {
    Uses[Idx].push_back(PU);
  }
  void use_push_back(const_iterator I, const PartitionUse &PU) {
    Uses[I - begin()].push_back(PU);
  }
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

  /// \brief MemTransferInst auxiliary data.
  /// This struct provides some auxiliary data about memory transfer
  /// intrinsics such as memcpy and memmove. These intrinsics can use two
  /// different ranges within the same alloca, and provide other challenges to
  /// correctly represent. We stash extra data to help us untangle this
  /// after the partitioning is complete.
  struct MemTransferOffsets {
    /// The destination begin and end offsets when the destination is within
    /// this alloca. If the end offset is zero the destination is not within
    /// this alloca.
    uint64_t DestBegin, DestEnd;

    /// The source begin and end offsets when the source is within this alloca.
    /// If the end offset is zero, the source is not within this alloca.
    uint64_t SourceBegin, SourceEnd;

    /// Flag for whether an alloca is splittable.
    bool IsSplittable;
  };
  MemTransferOffsets getMemTransferOffsets(MemTransferInst &II) const {
    return MemTransferInstData.lookup(&II);
  }

  /// \brief Map from a PHI or select operand back to a partition.
  ///
  /// When manipulating PHI nodes or selects, they can use more than one
  /// partition of an alloca. We store a special mapping to allow finding the
  /// partition referenced by each of these operands, if any.
  iterator findPartitionForPHIOrSelectOperand(Use *U) {
    SmallDenseMap<Use *, std::pair<unsigned, unsigned> >::const_iterator MapIt
      = PHIOrSelectOpMap.find(U);
    if (MapIt == PHIOrSelectOpMap.end())
      return end();

    return begin() + MapIt->second.first;
  }

  /// \brief Map from a PHI or select operand back to the specific use of
  /// a partition.
  ///
  /// Similar to mapping these operands back to the partitions, this maps
  /// directly to the use structure of that partition.
  use_iterator findPartitionUseForPHIOrSelectOperand(Use *U) {
    SmallDenseMap<Use *, std::pair<unsigned, unsigned> >::const_iterator MapIt
      = PHIOrSelectOpMap.find(U);
    assert(MapIt != PHIOrSelectOpMap.end());
    return Uses[MapIt->second.first].begin() + MapIt->second.second;
  }

  /// \brief Compute a common type among the uses of a particular partition.
  ///
  /// This routines walks all of the uses of a particular partition and tries
  /// to find a common type between them. Untyped operations such as memset and
  /// memcpy are ignored.
  Type *getCommonType(iterator I) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &OS, const_iterator I, StringRef Indent = "  ") const;
  void printUsers(raw_ostream &OS, const_iterator I,
                  StringRef Indent = "  ") const;
  void print(raw_ostream &OS) const;
  void LLVM_ATTRIBUTE_NOINLINE LLVM_ATTRIBUTE_USED dump(const_iterator I) const;
  void LLVM_ATTRIBUTE_NOINLINE LLVM_ATTRIBUTE_USED dump() const;
#endif

private:
  template <typename DerivedT, typename RetT = void> class BuilderBase;
  class PartitionBuilder;
  friend class AllocaPartitioning::PartitionBuilder;
  class UseBuilder;
  friend class AllocaPartitioning::UseBuilder;

#ifndef NDEBUG
  /// \brief Handle to alloca instruction to simplify method interfaces.
  AllocaInst &AI;
#endif

  /// \brief The instruction responsible for this alloca having no partitioning.
  ///
  /// When an instruction (potentially) escapes the pointer to the alloca, we
  /// store a pointer to that here and abort trying to partition the alloca.
  /// This will be null if the alloca is partitioned successfully.
  Instruction *PointerEscapingInstr;

  /// \brief The partitions of the alloca.
  ///
  /// We store a vector of the partitions over the alloca here. This vector is
  /// sorted by increasing begin offset, and then by decreasing end offset. See
  /// the Partition inner class for more details. Initially (during
  /// construction) there are overlaps, but we form a disjoint sequence of
  /// partitions while finishing construction and a fully constructed object is
  /// expected to always have this as a disjoint space.
  SmallVector<Partition, 8> Partitions;

  /// \brief The uses of the partitions.
  ///
  /// This is essentially a mapping from each partition to a list of uses of
  /// that partition. The mapping is done with a Uses vector that has the exact
  /// same number of entries as the partition vector. Each entry is itself
  /// a vector of the uses.
  SmallVector<SmallVector<PartitionUse, 2>, 8> Uses;

  /// \brief Instructions which will become dead if we rewrite the alloca.
  ///
  /// Note that these are not separated by partition. This is because we expect
  /// a partitioned alloca to be completely rewritten or not rewritten at all.
  /// If rewritten, all these instructions can simply be removed and replaced
  /// with undef as they come from outside of the allocated space.
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

  /// \brief The underlying storage for auxiliary memcpy and memset info.
  SmallDenseMap<MemTransferInst *, MemTransferOffsets, 4> MemTransferInstData;

  /// \brief A side datastructure used when building up the partitions and uses.
  ///
  /// This mapping is only really used during the initial building of the
  /// partitioning so that we can retain information about PHI and select nodes
  /// processed.
  SmallDenseMap<Instruction *, std::pair<uint64_t, bool> > PHIOrSelectSizes;

  /// \brief Auxiliary information for particular PHI or select operands.
  SmallDenseMap<Use *, std::pair<unsigned, unsigned>, 4> PHIOrSelectOpMap;

  /// \brief A utility routine called from the constructor.
  ///
  /// This does what it says on the tin. It is the key of the alloca partition
  /// splitting and merging. After it is called we have the desired disjoint
  /// collection of partitions.
  void splitAndMergePartitions();
};
}

template <typename DerivedT, typename RetT>
class AllocaPartitioning::BuilderBase
    : public InstVisitor<DerivedT, RetT> {
public:
  BuilderBase(const DataLayout &TD, AllocaInst &AI, AllocaPartitioning &P)
      : TD(TD),
        AllocSize(TD.getTypeAllocSize(AI.getAllocatedType())),
        P(P) {
    enqueueUsers(AI, 0);
  }

protected:
  const DataLayout &TD;
  const uint64_t AllocSize;
  AllocaPartitioning &P;

  SmallPtrSet<Use *, 8> VisitedUses;

  struct OffsetUse {
    Use *U;
    int64_t Offset;
  };
  SmallVector<OffsetUse, 8> Queue;

  // The active offset and use while visiting.
  Use *U;
  int64_t Offset;

  void enqueueUsers(Instruction &I, int64_t UserOffset) {
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
         UI != UE; ++UI) {
      if (VisitedUses.insert(&UI.getUse())) {
        OffsetUse OU = { &UI.getUse(), UserOffset };
        Queue.push_back(OU);
      }
    }
  }

  bool computeConstantGEPOffset(GetElementPtrInst &GEPI, int64_t &GEPOffset) {
    GEPOffset = Offset;
    for (gep_type_iterator GTI = gep_type_begin(GEPI), GTE = gep_type_end(GEPI);
         GTI != GTE; ++GTI) {
      ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());
      if (!OpC)
        return false;
      if (OpC->isZero())
        continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        unsigned ElementIdx = OpC->getZExtValue();
        const StructLayout *SL = TD.getStructLayout(STy);
        uint64_t ElementOffset = SL->getElementOffset(ElementIdx);
        // Check that we can continue to model this GEP in a signed 64-bit offset.
        if (ElementOffset > INT64_MAX ||
            (GEPOffset >= 0 &&
             ((uint64_t)GEPOffset + ElementOffset) > INT64_MAX)) {
          DEBUG(dbgs() << "WARNING: Encountered a cumulative offset exceeding "
                       << "what can be represented in an int64_t!\n"
                       << "  alloca: " << P.AI << "\n");
          return false;
        }
        if (GEPOffset < 0)
          GEPOffset = ElementOffset + (uint64_t)-GEPOffset;
        else
          GEPOffset += ElementOffset;
        continue;
      }

      APInt Index = OpC->getValue().sextOrTrunc(TD.getPointerSizeInBits());
      Index *= APInt(Index.getBitWidth(),
                     TD.getTypeAllocSize(GTI.getIndexedType()));
      Index += APInt(Index.getBitWidth(), (uint64_t)GEPOffset,
                     /*isSigned*/true);
      // Check if the result can be stored in our int64_t offset.
      if (!Index.isSignedIntN(sizeof(GEPOffset) * 8)) {
        DEBUG(dbgs() << "WARNING: Encountered a cumulative offset exceeding "
                     << "what can be represented in an int64_t!\n"
                     << "  alloca: " << P.AI << "\n");
        return false;
      }

      GEPOffset = Index.getSExtValue();
    }
    return true;
  }

  Value *foldSelectInst(SelectInst &SI) {
    // If the condition being selected on is a constant or the same value is
    // being selected between, fold the select. Yes this does (rarely) happen
    // early on.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(SI.getCondition()))
      return SI.getOperand(1+CI->isZero());
    if (SI.getOperand(1) == SI.getOperand(2)) {
      assert(*U == SI.getOperand(1));
      return SI.getOperand(1);
    }
    return 0;
  }
};

/// \brief Builder for the alloca partitioning.
///
/// This class builds an alloca partitioning by recursively visiting the uses
/// of an alloca and splitting the partitions for each load and store at each
/// offset.
class AllocaPartitioning::PartitionBuilder
    : public BuilderBase<PartitionBuilder, bool> {
  friend class InstVisitor<PartitionBuilder, bool>;

  SmallDenseMap<Instruction *, unsigned> MemTransferPartitionMap;

public:
  PartitionBuilder(const DataLayout &TD, AllocaInst &AI, AllocaPartitioning &P)
      : BuilderBase<PartitionBuilder, bool>(TD, AI, P) {}

  /// \brief Run the builder over the allocation.
  bool operator()() {
    // Note that we have to re-evaluate size on each trip through the loop as
    // the queue grows at the tail.
    for (unsigned Idx = 0; Idx < Queue.size(); ++Idx) {
      U = Queue[Idx].U;
      Offset = Queue[Idx].Offset;
      if (!visit(cast<Instruction>(U->getUser())))
        return false;
    }
    return true;
  }

private:
  bool markAsEscaping(Instruction &I) {
    P.PointerEscapingInstr = &I;
    return false;
  }

  void insertUse(Instruction &I, int64_t Offset, uint64_t Size,
                 bool IsSplittable = false) {
    // Completely skip uses which have a zero size or don't overlap the
    // allocation.
    if (Size == 0 ||
        (Offset >= 0 && (uint64_t)Offset >= AllocSize) ||
        (Offset < 0 && (uint64_t)-Offset >= Size)) {
      DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte use @" << Offset
                   << " which starts past the end of the " << AllocSize
                   << " byte alloca:\n"
                   << "    alloca: " << P.AI << "\n"
                   << "       use: " << I << "\n");
      return;
    }

    // Clamp the start to the beginning of the allocation.
    if (Offset < 0) {
      DEBUG(dbgs() << "WARNING: Clamping a " << Size << " byte use @" << Offset
                   << " to start at the beginning of the alloca:\n"
                   << "    alloca: " << P.AI << "\n"
                   << "       use: " << I << "\n");
      Size -= (uint64_t)-Offset;
      Offset = 0;
    }

    uint64_t BeginOffset = Offset, EndOffset = BeginOffset + Size;

    // Clamp the end offset to the end of the allocation. Note that this is
    // formulated to handle even the case where "BeginOffset + Size" overflows.
    assert(AllocSize >= BeginOffset); // Established above.
    if (Size > AllocSize - BeginOffset) {
      DEBUG(dbgs() << "WARNING: Clamping a " << Size << " byte use @" << Offset
                   << " to remain within the " << AllocSize << " byte alloca:\n"
                   << "    alloca: " << P.AI << "\n"
                   << "       use: " << I << "\n");
      EndOffset = AllocSize;
    }

    Partition New(BeginOffset, EndOffset, IsSplittable);
    P.Partitions.push_back(New);
  }

  bool handleLoadOrStore(Type *Ty, Instruction &I, int64_t Offset,
                         bool IsVolatile) {
    uint64_t Size = TD.getTypeStoreSize(Ty);

    // If this memory access can be shown to *statically* extend outside the
    // bounds of of the allocation, it's behavior is undefined, so simply
    // ignore it. Note that this is more strict than the generic clamping
    // behavior of insertUse. We also try to handle cases which might run the
    // risk of overflow.
    // FIXME: We should instead consider the pointer to have escaped if this
    // function is being instrumented for addressing bugs or race conditions.
    if (Offset < 0 || (uint64_t)Offset >= AllocSize ||
        Size > (AllocSize - (uint64_t)Offset)) {
      DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte "
                   << (isa<LoadInst>(I) ? "load" : "store") << " @" << Offset
                   << " which extends past the end of the " << AllocSize
                   << " byte alloca:\n"
                   << "    alloca: " << P.AI << "\n"
                   << "       use: " << I << "\n");
      return true;
    }

    // We allow splitting of loads and stores where the type is an integer type
    // and which cover the entire alloca. Such integer loads and stores
    // often require decomposition into fine grained loads and stores.
    bool IsSplittable = false;
    if (IntegerType *ITy = dyn_cast<IntegerType>(Ty))
      IsSplittable = !IsVolatile && ITy->getBitWidth() == AllocSize*8;

    insertUse(I, Offset, Size, IsSplittable);
    return true;
  }

  bool visitBitCastInst(BitCastInst &BC) {
    enqueueUsers(BC, Offset);
    return true;
  }

  bool visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    int64_t GEPOffset;
    if (!computeConstantGEPOffset(GEPI, GEPOffset))
      return markAsEscaping(GEPI);

    enqueueUsers(GEPI, GEPOffset);
    return true;
  }

  bool visitLoadInst(LoadInst &LI) {
    assert((!LI.isSimple() || LI.getType()->isSingleValueType()) &&
           "All simple FCA loads should have been pre-split");
    return handleLoadOrStore(LI.getType(), LI, Offset, LI.isVolatile());
  }

  bool visitStoreInst(StoreInst &SI) {
    Value *ValOp = SI.getValueOperand();
    if (ValOp == *U)
      return markAsEscaping(SI);

    assert((!SI.isSimple() || ValOp->getType()->isSingleValueType()) &&
           "All simple FCA stores should have been pre-split");
    return handleLoadOrStore(ValOp->getType(), SI, Offset, SI.isVolatile());
  }


  bool visitMemSetInst(MemSetInst &II) {
    assert(II.getRawDest() == *U && "Pointer use is not the destination?");
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    uint64_t Size = Length ? Length->getZExtValue() : AllocSize - Offset;
    insertUse(II, Offset, Size, Length);
    return true;
  }

  bool visitMemTransferInst(MemTransferInst &II) {
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    uint64_t Size = Length ? Length->getZExtValue() : AllocSize - Offset;
    if (!Size)
      // Zero-length mem transfer intrinsics can be ignored entirely.
      return true;

    MemTransferOffsets &Offsets = P.MemTransferInstData[&II];

    // Only intrinsics with a constant length can be split.
    Offsets.IsSplittable = Length;

    if (*U == II.getRawDest()) {
      Offsets.DestBegin = Offset;
      Offsets.DestEnd = Offset + Size;
    }
    if (*U == II.getRawSource()) {
      Offsets.SourceBegin = Offset;
      Offsets.SourceEnd = Offset + Size;
    }

    // If we have set up end offsets for both the source and the destination,
    // we have found both sides of this transfer pointing at the same alloca.
    bool SeenBothEnds = Offsets.SourceEnd && Offsets.DestEnd;
    if (SeenBothEnds && II.getRawDest() != II.getRawSource()) {
      unsigned PrevIdx = MemTransferPartitionMap[&II];

      // Check if the begin offsets match and this is a non-volatile transfer.
      // In that case, we can completely elide the transfer.
      if (!II.isVolatile() && Offsets.SourceBegin == Offsets.DestBegin) {
        P.Partitions[PrevIdx].kill();
        return true;
      }

      // Otherwise we have an offset transfer within the same alloca. We can't
      // split those.
      P.Partitions[PrevIdx].IsSplittable = Offsets.IsSplittable = false;
    } else if (SeenBothEnds) {
      // Handle the case where this exact use provides both ends of the
      // operation.
      assert(II.getRawDest() == II.getRawSource());

      // For non-volatile transfers this is a no-op.
      if (!II.isVolatile())
        return true;

      // Otherwise just suppress splitting.
      Offsets.IsSplittable = false;
    }


    // Insert the use now that we've fixed up the splittable nature.
    insertUse(II, Offset, Size, Offsets.IsSplittable);

    // Setup the mapping from intrinsic to partition of we've not seen both
    // ends of this transfer.
    if (!SeenBothEnds) {
      unsigned NewIdx = P.Partitions.size() - 1;
      bool Inserted
        = MemTransferPartitionMap.insert(std::make_pair(&II, NewIdx)).second;
      assert(Inserted &&
             "Already have intrinsic in map but haven't seen both ends");
      (void)Inserted;
    }

    return true;
  }

  // Disable SRoA for any intrinsics except for lifetime invariants.
  // FIXME: What about debug instrinsics? This matches old behavior, but
  // doesn't make sense.
  bool visitIntrinsicInst(IntrinsicInst &II) {
    if (II.getIntrinsicID() == Intrinsic::lifetime_start ||
        II.getIntrinsicID() == Intrinsic::lifetime_end) {
      ConstantInt *Length = cast<ConstantInt>(II.getArgOperand(0));
      uint64_t Size = std::min(AllocSize - Offset, Length->getLimitedValue());
      insertUse(II, Offset, Size, true);
      return true;
    }

    return markAsEscaping(II);
  }

  Instruction *hasUnsafePHIOrSelectUse(Instruction *Root, uint64_t &Size) {
    // We consider any PHI or select that results in a direct load or store of
    // the same offset to be a viable use for partitioning purposes. These uses
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
        Size = std::max(Size, TD.getTypeStoreSize(LI->getType()));
        continue;
      }
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        Value *Op = SI->getOperand(0);
        if (Op == UsedI)
          return SI;
        Size = std::max(Size, TD.getTypeStoreSize(Op->getType()));
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

  bool visitPHINode(PHINode &PN) {
    // See if we already have computed info on this node.
    std::pair<uint64_t, bool> &PHIInfo = P.PHIOrSelectSizes[&PN];
    if (PHIInfo.first) {
      PHIInfo.second = true;
      insertUse(PN, Offset, PHIInfo.first);
      return true;
    }

    // Check for an unsafe use of the PHI node.
    if (Instruction *EscapingI = hasUnsafePHIOrSelectUse(&PN, PHIInfo.first))
      return markAsEscaping(*EscapingI);

    insertUse(PN, Offset, PHIInfo.first);
    return true;
  }

  bool visitSelectInst(SelectInst &SI) {
    if (Value *Result = foldSelectInst(SI)) {
      if (Result == *U)
        // If the result of the constant fold will be the pointer, recurse
        // through the select as if we had RAUW'ed it.
        enqueueUsers(SI, Offset);

      return true;
    }

    // See if we already have computed info on this node.
    std::pair<uint64_t, bool> &SelectInfo = P.PHIOrSelectSizes[&SI];
    if (SelectInfo.first) {
      SelectInfo.second = true;
      insertUse(SI, Offset, SelectInfo.first);
      return true;
    }

    // Check for an unsafe use of the PHI node.
    if (Instruction *EscapingI = hasUnsafePHIOrSelectUse(&SI, SelectInfo.first))
      return markAsEscaping(*EscapingI);

    insertUse(SI, Offset, SelectInfo.first);
    return true;
  }

  /// \brief Disable SROA entirely if there are unhandled users of the alloca.
  bool visitInstruction(Instruction &I) { return markAsEscaping(I); }
};


/// \brief Use adder for the alloca partitioning.
///
/// This class adds the uses of an alloca to all of the partitions which they
/// use. For splittable partitions, this can end up doing essentially a linear
/// walk of the partitions, but the number of steps remains bounded by the
/// total result instruction size:
/// - The number of partitions is a result of the number unsplittable
///   instructions using the alloca.
/// - The number of users of each partition is at worst the total number of
///   splittable instructions using the alloca.
/// Thus we will produce N * M instructions in the end, where N are the number
/// of unsplittable uses and M are the number of splittable. This visitor does
/// the exact same number of updates to the partitioning.
///
/// In the more common case, this visitor will leverage the fact that the
/// partition space is pre-sorted, and do a logarithmic search for the
/// partition needed, making the total visit a classical ((N + M) * log(N))
/// complexity operation.
class AllocaPartitioning::UseBuilder : public BuilderBase<UseBuilder> {
  friend class InstVisitor<UseBuilder>;

  /// \brief Set to de-duplicate dead instructions found in the use walk.
  SmallPtrSet<Instruction *, 4> VisitedDeadInsts;

public:
  UseBuilder(const DataLayout &TD, AllocaInst &AI, AllocaPartitioning &P)
      : BuilderBase<UseBuilder>(TD, AI, P) {}

  /// \brief Run the builder over the allocation.
  void operator()() {
    // Note that we have to re-evaluate size on each trip through the loop as
    // the queue grows at the tail.
    for (unsigned Idx = 0; Idx < Queue.size(); ++Idx) {
      U = Queue[Idx].U;
      Offset = Queue[Idx].Offset;
      this->visit(cast<Instruction>(U->getUser()));
    }
  }

private:
  void markAsDead(Instruction &I) {
    if (VisitedDeadInsts.insert(&I))
      P.DeadUsers.push_back(&I);
  }

  void insertUse(Instruction &User, int64_t Offset, uint64_t Size) {
    // If the use has a zero size or extends outside of the allocation, record
    // it as a dead use for elimination later.
    if (Size == 0 || (uint64_t)Offset >= AllocSize ||
        (Offset < 0 && (uint64_t)-Offset >= Size))
      return markAsDead(User);

    // Clamp the start to the beginning of the allocation.
    if (Offset < 0) {
      Size -= (uint64_t)-Offset;
      Offset = 0;
    }

    uint64_t BeginOffset = Offset, EndOffset = BeginOffset + Size;

    // Clamp the end offset to the end of the allocation. Note that this is
    // formulated to handle even the case where "BeginOffset + Size" overflows.
    assert(AllocSize >= BeginOffset); // Established above.
    if (Size > AllocSize - BeginOffset)
      EndOffset = AllocSize;

    // NB: This only works if we have zero overlapping partitions.
    iterator B = std::lower_bound(P.begin(), P.end(), BeginOffset);
    if (B != P.begin() && llvm::prior(B)->EndOffset > BeginOffset)
      B = llvm::prior(B);
    for (iterator I = B, E = P.end(); I != E && I->BeginOffset < EndOffset;
         ++I) {
      PartitionUse NewPU(std::max(I->BeginOffset, BeginOffset),
                         std::min(I->EndOffset, EndOffset), U);
      P.use_push_back(I, NewPU);
      if (isa<PHINode>(U->getUser()) || isa<SelectInst>(U->getUser()))
        P.PHIOrSelectOpMap[U]
          = std::make_pair(I - P.begin(), P.Uses[I - P.begin()].size() - 1);
    }
  }

  void handleLoadOrStore(Type *Ty, Instruction &I, int64_t Offset) {
    uint64_t Size = TD.getTypeStoreSize(Ty);

    // If this memory access can be shown to *statically* extend outside the
    // bounds of of the allocation, it's behavior is undefined, so simply
    // ignore it. Note that this is more strict than the generic clamping
    // behavior of insertUse.
    if (Offset < 0 || (uint64_t)Offset >= AllocSize ||
        Size > (AllocSize - (uint64_t)Offset))
      return markAsDead(I);

    insertUse(I, Offset, Size);
  }

  void visitBitCastInst(BitCastInst &BC) {
    if (BC.use_empty())
      return markAsDead(BC);

    enqueueUsers(BC, Offset);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    if (GEPI.use_empty())
      return markAsDead(GEPI);

    int64_t GEPOffset;
    if (!computeConstantGEPOffset(GEPI, GEPOffset))
      llvm_unreachable("Unable to compute constant offset for use");

    enqueueUsers(GEPI, GEPOffset);
  }

  void visitLoadInst(LoadInst &LI) {
    handleLoadOrStore(LI.getType(), LI, Offset);
  }

  void visitStoreInst(StoreInst &SI) {
    handleLoadOrStore(SI.getOperand(0)->getType(), SI, Offset);
  }

  void visitMemSetInst(MemSetInst &II) {
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    uint64_t Size = Length ? Length->getZExtValue() : AllocSize - Offset;
    insertUse(II, Offset, Size);
  }

  void visitMemTransferInst(MemTransferInst &II) {
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    uint64_t Size = Length ? Length->getZExtValue() : AllocSize - Offset;
    if (!Size)
      return markAsDead(II);

    MemTransferOffsets &Offsets = P.MemTransferInstData[&II];
    if (!II.isVolatile() && Offsets.DestEnd && Offsets.SourceEnd &&
        Offsets.DestBegin == Offsets.SourceBegin)
      return markAsDead(II); // Skip identity transfers without side-effects.

    insertUse(II, Offset, Size);
  }

  void visitIntrinsicInst(IntrinsicInst &II) {
    assert(II.getIntrinsicID() == Intrinsic::lifetime_start ||
           II.getIntrinsicID() == Intrinsic::lifetime_end);

    ConstantInt *Length = cast<ConstantInt>(II.getArgOperand(0));
    insertUse(II, Offset,
              std::min(AllocSize - Offset, Length->getLimitedValue()));
  }

  void insertPHIOrSelect(Instruction &User, uint64_t Offset) {
    uint64_t Size = P.PHIOrSelectSizes.lookup(&User).first;

    // For PHI and select operands outside the alloca, we can't nuke the entire
    // phi or select -- the other side might still be relevant, so we special
    // case them here and use a separate structure to track the operands
    // themselves which should be replaced with undef.
    if (Offset >= AllocSize) {
      P.DeadOperands.push_back(U);
      return;
    }

    insertUse(User, Offset, Size);
  }
  void visitPHINode(PHINode &PN) {
    if (PN.use_empty())
      return markAsDead(PN);

    insertPHIOrSelect(PN, Offset);
  }
  void visitSelectInst(SelectInst &SI) {
    if (SI.use_empty())
      return markAsDead(SI);

    if (Value *Result = foldSelectInst(SI)) {
      if (Result == *U)
        // If the result of the constant fold will be the pointer, recurse
        // through the select as if we had RAUW'ed it.
        enqueueUsers(SI, Offset);
      else
        // Otherwise the operand to the select is dead, and we can replace it
        // with undef.
        P.DeadOperands.push_back(U);

      return;
    }

    insertPHIOrSelect(SI, Offset);
  }

  /// \brief Unreachable, we've already visited the alloca once.
  void visitInstruction(Instruction &I) {
    llvm_unreachable("Unhandled instruction in use builder.");
  }
};

void AllocaPartitioning::splitAndMergePartitions() {
  size_t NumDeadPartitions = 0;

  // Track the range of splittable partitions that we pass when accumulating
  // overlapping unsplittable partitions.
  uint64_t SplitEndOffset = 0ull;

  Partition New(0ull, 0ull, false);

  for (unsigned i = 0, j = i, e = Partitions.size(); i != e; i = j) {
    ++j;

    if (!Partitions[i].IsSplittable || New.BeginOffset == New.EndOffset) {
      assert(New.BeginOffset == New.EndOffset);
      New = Partitions[i];
    } else {
      assert(New.IsSplittable);
      New.EndOffset = std::max(New.EndOffset, Partitions[i].EndOffset);
    }
    assert(New.BeginOffset != New.EndOffset);

    // Scan the overlapping partitions.
    while (j != e && New.EndOffset > Partitions[j].BeginOffset) {
      // If the new partition we are forming is splittable, stop at the first
      // unsplittable partition.
      if (New.IsSplittable && !Partitions[j].IsSplittable)
        break;

      // Grow the new partition to include any equally splittable range. 'j' is
      // always equally splittable when New is splittable, but when New is not
      // splittable, we may subsume some (or part of some) splitable partition
      // without growing the new one.
      if (New.IsSplittable == Partitions[j].IsSplittable) {
        New.EndOffset = std::max(New.EndOffset, Partitions[j].EndOffset);
      } else {
        assert(!New.IsSplittable);
        assert(Partitions[j].IsSplittable);
        SplitEndOffset = std::max(SplitEndOffset, Partitions[j].EndOffset);
      }

      Partitions[j].kill();
      ++NumDeadPartitions;
      ++j;
    }

    // If the new partition is splittable, chop off the end as soon as the
    // unsplittable subsequent partition starts and ensure we eventually cover
    // the splittable area.
    if (j != e && New.IsSplittable) {
      SplitEndOffset = std::max(SplitEndOffset, New.EndOffset);
      New.EndOffset = std::min(New.EndOffset, Partitions[j].BeginOffset);
    }

    // Add the new partition if it differs from the original one and is
    // non-empty. We can end up with an empty partition here if it was
    // splittable but there is an unsplittable one that starts at the same
    // offset.
    if (New != Partitions[i]) {
      if (New.BeginOffset != New.EndOffset)
        Partitions.push_back(New);
      // Mark the old one for removal.
      Partitions[i].kill();
      ++NumDeadPartitions;
    }

    New.BeginOffset = New.EndOffset;
    if (!New.IsSplittable) {
      New.EndOffset = std::max(New.EndOffset, SplitEndOffset);
      if (j != e && !Partitions[j].IsSplittable)
        New.EndOffset = std::min(New.EndOffset, Partitions[j].BeginOffset);
      New.IsSplittable = true;
      // If there is a trailing splittable partition which won't be fused into
      // the next splittable partition go ahead and add it onto the partitions
      // list.
      if (New.BeginOffset < New.EndOffset &&
          (j == e || !Partitions[j].IsSplittable ||
           New.EndOffset < Partitions[j].BeginOffset)) {
        Partitions.push_back(New);
        New.BeginOffset = New.EndOffset = 0ull;
      }
    }
  }

  // Re-sort the partitions now that they have been split and merged into
  // disjoint set of partitions. Also remove any of the dead partitions we've
  // replaced in the process.
  std::sort(Partitions.begin(), Partitions.end());
  if (NumDeadPartitions) {
    assert(Partitions.back().isDead());
    assert((ptrdiff_t)NumDeadPartitions ==
           std::count(Partitions.begin(), Partitions.end(), Partitions.back()));
  }
  Partitions.erase(Partitions.end() - NumDeadPartitions, Partitions.end());
}

AllocaPartitioning::AllocaPartitioning(const DataLayout &TD, AllocaInst &AI)
    :
#ifndef NDEBUG
      AI(AI),
#endif
      PointerEscapingInstr(0) {
  PartitionBuilder PB(TD, AI, *this);
  if (!PB())
    return;

  // Sort the uses. This arranges for the offsets to be in ascending order,
  // and the sizes to be in descending order.
  std::sort(Partitions.begin(), Partitions.end());

  // Remove any partitions from the back which are marked as dead.
  while (!Partitions.empty() && Partitions.back().isDead())
    Partitions.pop_back();

  if (Partitions.size() > 1) {
    // Intersect splittability for all partitions with equal offsets and sizes.
    // Then remove all but the first so that we have a sequence of non-equal but
    // potentially overlapping partitions.
    for (iterator I = Partitions.begin(), J = I, E = Partitions.end(); I != E;
         I = J) {
      ++J;
      while (J != E && *I == *J) {
        I->IsSplittable &= J->IsSplittable;
        ++J;
      }
    }
    Partitions.erase(std::unique(Partitions.begin(), Partitions.end()),
                     Partitions.end());

    // Split splittable and merge unsplittable partitions into a disjoint set
    // of partitions over the used space of the allocation.
    splitAndMergePartitions();
  }

  // Now build up the user lists for each of these disjoint partitions by
  // re-walking the recursive users of the alloca.
  Uses.resize(Partitions.size());
  UseBuilder UB(TD, AI, *this);
  UB();
}

Type *AllocaPartitioning::getCommonType(iterator I) const {
  Type *Ty = 0;
  for (const_use_iterator UI = use_begin(I), UE = use_end(I); UI != UE; ++UI) {
    if (!UI->U)
      continue; // Skip dead uses.
    if (isa<IntrinsicInst>(*UI->U->getUser()))
      continue;
    if (UI->BeginOffset != I->BeginOffset || UI->EndOffset != I->EndOffset)
      continue;

    Type *UserTy = 0;
    if (LoadInst *LI = dyn_cast<LoadInst>(UI->U->getUser())) {
      UserTy = LI->getType();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(UI->U->getUser())) {
      UserTy = SI->getValueOperand()->getType();
    } else {
      return 0; // Bail if we have weird uses.
    }

    if (IntegerType *ITy = dyn_cast<IntegerType>(UserTy)) {
      // If the type is larger than the partition, skip it. We only encounter
      // this for split integer operations where we want to use the type of the
      // entity causing the split.
      if (ITy->getBitWidth() > (I->EndOffset - I->BeginOffset)*8)
        continue;

      // If we have found an integer type use covering the alloca, use that
      // regardless of the other types, as integers are often used for a "bucket
      // of bits" type.
      return ITy;
    }

    if (Ty && Ty != UserTy)
      return 0;

    Ty = UserTy;
  }
  return Ty;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

void AllocaPartitioning::print(raw_ostream &OS, const_iterator I,
                               StringRef Indent) const {
  OS << Indent << "partition #" << (I - begin())
     << " [" << I->BeginOffset << "," << I->EndOffset << ")"
     << (I->IsSplittable ? " (splittable)" : "")
     << (Uses[I - begin()].empty() ? " (zero uses)" : "")
     << "\n";
}

void AllocaPartitioning::printUsers(raw_ostream &OS, const_iterator I,
                                    StringRef Indent) const {
  for (const_use_iterator UI = use_begin(I), UE = use_end(I);
       UI != UE; ++UI) {
    if (!UI->U)
      continue; // Skip dead uses.
    OS << Indent << "  [" << UI->BeginOffset << "," << UI->EndOffset << ") "
       << "used by: " << *UI->U->getUser() << "\n";
    if (MemTransferInst *II = dyn_cast<MemTransferInst>(UI->U->getUser())) {
      const MemTransferOffsets &MTO = MemTransferInstData.lookup(II);
      bool IsDest;
      if (!MTO.IsSplittable)
        IsDest = UI->BeginOffset == MTO.DestBegin;
      else
        IsDest = MTO.DestBegin != 0u;
      OS << Indent << "    (original " << (IsDest ? "dest" : "source") << ": "
         << "[" << (IsDest ? MTO.DestBegin : MTO.SourceBegin)
         << "," << (IsDest ? MTO.DestEnd : MTO.SourceEnd) << ")\n";
    }
  }
}

void AllocaPartitioning::print(raw_ostream &OS) const {
  if (PointerEscapingInstr) {
    OS << "No partitioning for alloca: " << AI << "\n"
       << "  A pointer to this alloca escaped by:\n"
       << "  " << *PointerEscapingInstr << "\n";
    return;
  }

  OS << "Partitioning of alloca: " << AI << "\n";
  unsigned Num = 0;
  for (const_iterator I = begin(), E = end(); I != E; ++I, ++Num) {
    print(OS, I);
    printUsers(OS, I);
  }
}

void AllocaPartitioning::dump(const_iterator I) const { print(dbgs(), I); }
void AllocaPartitioning::dump() const { print(dbgs()); }

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
  AllocaPromoter(const SmallVectorImpl<Instruction*> &Insts, SSAUpdater &S,
                 AllocaInst &AI, DIBuilder &DIB)
    : LoadAndStorePromoter(Insts, S), AI(AI), DIB(DIB) {}

  void run(const SmallVectorImpl<Instruction*> &Insts) {
    // Remember which alloca we're promoting (for isInstInList).
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
    AI.eraseFromParent();
    while (!DDIs.empty())
      DDIs.pop_back_val()->eraseFromParent();
    while (!DVIs.empty())
      DVIs.pop_back_val()->eraseFromParent();
  }

  virtual bool isInstInList(Instruction *I,
                            const SmallVectorImpl<Instruction*> &Insts) const {
    if (LoadInst *LI = dyn_cast<LoadInst>(I))
      return LI->getOperand(0) == &AI;
    return cast<StoreInst>(I)->getPointerOperand() == &AI;
  }

  virtual void updateDebugInfo(Instruction *Inst) const {
    for (SmallVector<DbgDeclareInst *, 4>::const_iterator I = DDIs.begin(),
           E = DDIs.end(); I != E; ++I) {
      DbgDeclareInst *DDI = *I;
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
        ConvertDebugDeclareToDebugValue(DDI, SI, DIB);
      else if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
        ConvertDebugDeclareToDebugValue(DDI, LI, DIB);
    }
    for (SmallVector<DbgValueInst *, 4>::const_iterator I = DVIs.begin(),
           E = DVIs.end(); I != E; ++I) {
      DbgValueInst *DVI = *I;
      Value *Arg = NULL;
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        // If an argument is zero extended then use argument directly. The ZExt
        // may be zapped by an optimization pass in future.
        if (ZExtInst *ZExt = dyn_cast<ZExtInst>(SI->getOperand(0)))
          Arg = dyn_cast<Argument>(ZExt->getOperand(0));
        if (SExtInst *SExt = dyn_cast<SExtInst>(SI->getOperand(0)))
          Arg = dyn_cast<Argument>(SExt->getOperand(0));
        if (!Arg)
          Arg = SI->getOperand(0);
      } else if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        Arg = LI->getOperand(0);
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
///    as necessary and try to isolate invidual scalar accesses.
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
  const DataLayout *TD;
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
  SmallVector<Instruction *, 8> DeadInsts;

  /// \brief A set to prevent repeatedly marking an instruction split into many
  /// uses as dead. Only used to guard insertion into DeadInsts.
  SmallPtrSet<Instruction *, 4> DeadSplitInsts;

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

public:
  SROA(bool RequiresDomTree = true)
      : FunctionPass(ID), RequiresDomTree(RequiresDomTree),
        C(0), TD(0), DT(0) {
    initializeSROAPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F);
  void getAnalysisUsage(AnalysisUsage &AU) const;

  const char *getPassName() const { return "SROA"; }
  static char ID;

private:
  friend class PHIOrSelectSpeculator;
  friend class AllocaPartitionRewriter;
  friend class AllocaPartitionVectorRewriter;

  bool rewriteAllocaPartition(AllocaInst &AI,
                              AllocaPartitioning &P,
                              AllocaPartitioning::iterator PI);
  bool splitAlloca(AllocaInst &AI, AllocaPartitioning &P);
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

namespace {
/// \brief Visitor to speculate PHIs and Selects where possible.
class PHIOrSelectSpeculator : public InstVisitor<PHIOrSelectSpeculator> {
  // Befriend the base class so it can delegate to private visit methods.
  friend class llvm::InstVisitor<PHIOrSelectSpeculator>;

  const DataLayout &TD;
  AllocaPartitioning &P;
  SROA &Pass;

public:
  PHIOrSelectSpeculator(const DataLayout &TD, AllocaPartitioning &P, SROA &Pass)
    : TD(TD), P(P), Pass(Pass) {}

  /// \brief Visit the users of an alloca partition and rewrite them.
  void visitUsers(AllocaPartitioning::const_iterator PI) {
    // Note that we need to use an index here as the underlying vector of uses
    // may be grown during speculation. However, we never need to re-visit the
    // new uses, and so we can use the initial size bound.
    for (unsigned Idx = 0, Size = P.use_size(PI); Idx != Size; ++Idx) {
      const AllocaPartitioning::PartitionUse &PU = P.getUse(PI, Idx);
      if (!PU.U)
        continue; // Skip dead use.

      visit(cast<Instruction>(PU.U->getUser()));
    }
  }

private:
  // By default, skip this instruction.
  void visitInstruction(Instruction &I) {}

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
  bool isSafePHIToSpeculate(PHINode &PN, SmallVectorImpl<LoadInst *> &Loads) {
    // For now, we can only do this promotion if the load is in the same block
    // as the PHI, and if there are no stores between the phi and load.
    // TODO: Allow recursive phi users.
    // TODO: Allow stores.
    BasicBlock *BB = PN.getParent();
    unsigned MaxAlign = 0;
    for (Value::use_iterator UI = PN.use_begin(), UE = PN.use_end();
         UI != UE; ++UI) {
      LoadInst *LI = dyn_cast<LoadInst>(*UI);
      if (LI == 0 || !LI->isSimple()) return false;

      // For now we only allow loads in the same block as the PHI.  This is
      // a common case that happens when instcombine merges two loads through
      // a PHI.
      if (LI->getParent() != BB) return false;

      // Ensure that there are no instructions between the PHI and the load that
      // could store.
      for (BasicBlock::iterator BBI = &PN; &*BBI != LI; ++BBI)
        if (BBI->mayWriteToMemory())
          return false;

      MaxAlign = std::max(MaxAlign, LI->getAlignment());
      Loads.push_back(LI);
    }

    // We can only transform this if it is safe to push the loads into the
    // predecessor blocks. The only thing to watch out for is that we can't put
    // a possibly trapping load in the predecessor if it is a critical edge.
    for (unsigned Idx = 0, Num = PN.getNumIncomingValues(); Idx != Num;
         ++Idx) {
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
          isSafeToLoadUnconditionally(InVal, TI, MaxAlign, &TD))
        continue;

      return false;
    }

    return true;
  }

  void visitPHINode(PHINode &PN) {
    DEBUG(dbgs() << "    original: " << PN << "\n");

    SmallVector<LoadInst *, 4> Loads;
    if (!isSafePHIToSpeculate(PN, Loads))
      return;

    assert(!Loads.empty());

    Type *LoadTy = cast<PointerType>(PN.getType())->getElementType();
    IRBuilder<> PHIBuilder(&PN);
    PHINode *NewPN = PHIBuilder.CreatePHI(LoadTy, PN.getNumIncomingValues(),
                                          PN.getName() + ".sroa.speculated");

    // Get the TBAA tag and alignment to use from one of the loads.  It doesn't
    // matter which one we get and if any differ, it doesn't matter.
    LoadInst *SomeLoad = cast<LoadInst>(Loads.back());
    MDNode *TBAATag = SomeLoad->getMetadata(LLVMContext::MD_tbaa);
    unsigned Align = SomeLoad->getAlignment();

    // Rewrite all loads of the PN to use the new PHI.
    do {
      LoadInst *LI = Loads.pop_back_val();
      LI->replaceAllUsesWith(NewPN);
      Pass.DeadInsts.push_back(LI);
    } while (!Loads.empty());

    // Inject loads into all of the pred blocks.
    for (unsigned Idx = 0, Num = PN.getNumIncomingValues(); Idx != Num; ++Idx) {
      BasicBlock *Pred = PN.getIncomingBlock(Idx);
      TerminatorInst *TI = Pred->getTerminator();
      Use *InUse = &PN.getOperandUse(PN.getOperandNumForIncomingValue(Idx));
      Value *InVal = PN.getIncomingValue(Idx);
      IRBuilder<> PredBuilder(TI);

      LoadInst *Load
        = PredBuilder.CreateLoad(InVal, (PN.getName() + ".sroa.speculate.load." +
                                         Pred->getName()));
      ++NumLoadsSpeculated;
      Load->setAlignment(Align);
      if (TBAATag)
        Load->setMetadata(LLVMContext::MD_tbaa, TBAATag);
      NewPN->addIncoming(Load, Pred);

      Instruction *Ptr = dyn_cast<Instruction>(InVal);
      if (!Ptr)
        // No uses to rewrite.
        continue;

      // Try to lookup and rewrite any partition uses corresponding to this phi
      // input.
      AllocaPartitioning::iterator PI
        = P.findPartitionForPHIOrSelectOperand(InUse);
      if (PI == P.end())
        continue;

      // Replace the Use in the PartitionUse for this operand with the Use
      // inside the load.
      AllocaPartitioning::use_iterator UI
        = P.findPartitionUseForPHIOrSelectOperand(InUse);
      assert(isa<PHINode>(*UI->U->getUser()));
      UI->U = &Load->getOperandUse(Load->getPointerOperandIndex());
    }
    DEBUG(dbgs() << "          speculated to: " << *NewPN << "\n");
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
  bool isSafeSelectToSpeculate(SelectInst &SI,
                               SmallVectorImpl<LoadInst *> &Loads) {
    Value *TValue = SI.getTrueValue();
    Value *FValue = SI.getFalseValue();
    bool TDerefable = TValue->isDereferenceablePointer();
    bool FDerefable = FValue->isDereferenceablePointer();

    for (Value::use_iterator UI = SI.use_begin(), UE = SI.use_end();
         UI != UE; ++UI) {
      LoadInst *LI = dyn_cast<LoadInst>(*UI);
      if (LI == 0 || !LI->isSimple()) return false;

      // Both operands to the select need to be dereferencable, either
      // absolutely (e.g. allocas) or at this point because we can see other
      // accesses to it.
      if (!TDerefable && !isSafeToLoadUnconditionally(TValue, LI,
                                                      LI->getAlignment(), &TD))
        return false;
      if (!FDerefable && !isSafeToLoadUnconditionally(FValue, LI,
                                                      LI->getAlignment(), &TD))
        return false;
      Loads.push_back(LI);
    }

    return true;
  }

  void visitSelectInst(SelectInst &SI) {
    DEBUG(dbgs() << "    original: " << SI << "\n");
    IRBuilder<> IRB(&SI);

    // If the select isn't safe to speculate, just use simple logic to emit it.
    SmallVector<LoadInst *, 4> Loads;
    if (!isSafeSelectToSpeculate(SI, Loads))
      return;

    Use *Ops[2] = { &SI.getOperandUse(1), &SI.getOperandUse(2) };
    AllocaPartitioning::iterator PIs[2];
    AllocaPartitioning::PartitionUse PUs[2];
    for (unsigned i = 0, e = 2; i != e; ++i) {
      PIs[i] = P.findPartitionForPHIOrSelectOperand(Ops[i]);
      if (PIs[i] != P.end()) {
        // If the pointer is within the partitioning, remove the select from
        // its uses. We'll add in the new loads below.
        AllocaPartitioning::use_iterator UI
          = P.findPartitionUseForPHIOrSelectOperand(Ops[i]);
        PUs[i] = *UI;
        // Clear out the use here so that the offsets into the use list remain
        // stable but this use is ignored when rewriting.
        UI->U = 0;
      }
    }

    Value *TV = SI.getTrueValue();
    Value *FV = SI.getFalseValue();
    // Replace the loads of the select with a select of two loads.
    while (!Loads.empty()) {
      LoadInst *LI = Loads.pop_back_val();

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

      LoadInst *Loads[2] = { TL, FL };
      for (unsigned i = 0, e = 2; i != e; ++i) {
        if (PIs[i] != P.end()) {
          Use *LoadUse = &Loads[i]->getOperandUse(0);
          assert(PUs[i].U->get() == LoadUse->get());
          PUs[i].U = LoadUse;
          P.use_push_back(PIs[i], PUs[i]);
        }
      }

      DEBUG(dbgs() << "          speculated to: " << *V << "\n");
      LI->replaceAllUsesWith(V);
      Pass.DeadInsts.push_back(LI);
    }
  }
};
}

/// \brief Accumulate the constant offsets in a GEP into a single APInt offset.
///
/// If the provided GEP is all-constant, the total byte offset formed by the
/// GEP is computed and Offset is set to it. If the GEP has any non-constant
/// operands, the function returns false and the value of Offset is unmodified.
static bool accumulateGEPOffsets(const DataLayout &TD, GEPOperator &GEP,
                                 APInt &Offset) {
  APInt GEPOffset(Offset.getBitWidth(), 0);
  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());
    if (!OpC)
      return false;
    if (OpC->isZero()) continue;

    // Handle a struct index, which adds its field offset to the pointer.
    if (StructType *STy = dyn_cast<StructType>(*GTI)) {
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = TD.getStructLayout(STy);
      GEPOffset += APInt(Offset.getBitWidth(),
                         SL->getElementOffset(ElementIdx));
      continue;
    }

    APInt TypeSize(Offset.getBitWidth(),
                   TD.getTypeAllocSize(GTI.getIndexedType()));
    if (VectorType *VTy = dyn_cast<VectorType>(*GTI)) {
      assert((VTy->getScalarSizeInBits() % 8) == 0 &&
             "vector element size is not a multiple of 8, cannot GEP over it");
      TypeSize = VTy->getScalarSizeInBits() / 8;
    }

    GEPOffset += OpC->getValue().sextOrTrunc(Offset.getBitWidth()) * TypeSize;
  }
  Offset = GEPOffset;
  return true;
}

/// \brief Build a GEP out of a base pointer and indices.
///
/// This will return the BasePtr if that is valid, or build a new GEP
/// instruction using the IRBuilder if GEP-ing is needed.
static Value *buildGEP(IRBuilder<> &IRB, Value *BasePtr,
                       SmallVectorImpl<Value *> &Indices,
                       const Twine &Prefix) {
  if (Indices.empty())
    return BasePtr;

  // A single zero index is a no-op, so check for this and avoid building a GEP
  // in that case.
  if (Indices.size() == 1 && cast<ConstantInt>(Indices.back())->isZero())
    return BasePtr;

  return IRB.CreateInBoundsGEP(BasePtr, Indices, Prefix + ".idx");
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
static Value *getNaturalGEPWithType(IRBuilder<> &IRB, const DataLayout &TD,
                                    Value *BasePtr, Type *Ty, Type *TargetTy,
                                    SmallVectorImpl<Value *> &Indices,
                                    const Twine &Prefix) {
  if (Ty == TargetTy)
    return buildGEP(IRB, BasePtr, Indices, Prefix);

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
      Indices.push_back(IRB.getInt(APInt(TD.getPointerSizeInBits(0), 0)));
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

  return buildGEP(IRB, BasePtr, Indices, Prefix);
}

/// \brief Recursively compute indices for a natural GEP.
///
/// This is the recursive step for getNaturalGEPWithOffset that walks down the
/// element types adding appropriate indices for the GEP.
static Value *getNaturalGEPRecursively(IRBuilder<> &IRB, const DataLayout &TD,
                                       Value *Ptr, Type *Ty, APInt &Offset,
                                       Type *TargetTy,
                                       SmallVectorImpl<Value *> &Indices,
                                       const Twine &Prefix) {
  if (Offset == 0)
    return getNaturalGEPWithType(IRB, TD, Ptr, Ty, TargetTy, Indices, Prefix);

  // We can't recurse through pointer types.
  if (Ty->isPointerTy())
    return 0;

  // We try to analyze GEPs over vectors here, but note that these GEPs are
  // extremely poorly defined currently. The long-term goal is to remove GEPing
  // over a vector from the IR completely.
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty)) {
    unsigned ElementSizeInBits = VecTy->getScalarSizeInBits();
    if (ElementSizeInBits % 8)
      return 0; // GEPs over non-multiple of 8 size vector elements are invalid.
    APInt ElementSize(Offset.getBitWidth(), ElementSizeInBits / 8);
    APInt NumSkippedElements = Offset.sdiv(ElementSize);
    if (NumSkippedElements.ugt(VecTy->getNumElements()))
      return 0;
    Offset -= NumSkippedElements * ElementSize;
    Indices.push_back(IRB.getInt(NumSkippedElements));
    return getNaturalGEPRecursively(IRB, TD, Ptr, VecTy->getElementType(),
                                    Offset, TargetTy, Indices, Prefix);
  }

  if (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
    Type *ElementTy = ArrTy->getElementType();
    APInt ElementSize(Offset.getBitWidth(), TD.getTypeAllocSize(ElementTy));
    APInt NumSkippedElements = Offset.sdiv(ElementSize);
    if (NumSkippedElements.ugt(ArrTy->getNumElements()))
      return 0;

    Offset -= NumSkippedElements * ElementSize;
    Indices.push_back(IRB.getInt(NumSkippedElements));
    return getNaturalGEPRecursively(IRB, TD, Ptr, ElementTy, Offset, TargetTy,
                                    Indices, Prefix);
  }

  StructType *STy = dyn_cast<StructType>(Ty);
  if (!STy)
    return 0;

  const StructLayout *SL = TD.getStructLayout(STy);
  uint64_t StructOffset = Offset.getZExtValue();
  if (StructOffset >= SL->getSizeInBytes())
    return 0;
  unsigned Index = SL->getElementContainingOffset(StructOffset);
  Offset -= APInt(Offset.getBitWidth(), SL->getElementOffset(Index));
  Type *ElementTy = STy->getElementType(Index);
  if (Offset.uge(TD.getTypeAllocSize(ElementTy)))
    return 0; // The offset points into alignment padding.

  Indices.push_back(IRB.getInt32(Index));
  return getNaturalGEPRecursively(IRB, TD, Ptr, ElementTy, Offset, TargetTy,
                                  Indices, Prefix);
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
static Value *getNaturalGEPWithOffset(IRBuilder<> &IRB, const DataLayout &TD,
                                      Value *Ptr, APInt Offset, Type *TargetTy,
                                      SmallVectorImpl<Value *> &Indices,
                                      const Twine &Prefix) {
  PointerType *Ty = cast<PointerType>(Ptr->getType());

  // Don't consider any GEPs through an i8* as natural unless the TargetTy is
  // an i8.
  if (Ty == IRB.getInt8PtrTy() && TargetTy->isIntegerTy(8))
    return 0;

  Type *ElementTy = Ty->getElementType();
  if (!ElementTy->isSized())
    return 0; // We can't GEP through an unsized element.
  APInt ElementSize(Offset.getBitWidth(), TD.getTypeAllocSize(ElementTy));
  if (ElementSize == 0)
    return 0; // Zero-length arrays can't help us build a natural GEP.
  APInt NumSkippedElements = Offset.sdiv(ElementSize);

  Offset -= NumSkippedElements * ElementSize;
  Indices.push_back(IRB.getInt(NumSkippedElements));
  return getNaturalGEPRecursively(IRB, TD, Ptr, ElementTy, Offset, TargetTy,
                                  Indices, Prefix);
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
/// properities. The algorithm tries to fold as many constant indices into
/// a single GEP as possible, thus making each GEP more independent of the
/// surrounding code.
static Value *getAdjustedPtr(IRBuilder<> &IRB, const DataLayout &TD,
                             Value *Ptr, APInt Offset, Type *PointerTy,
                             const Twine &Prefix) {
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
      if (!accumulateGEPOffsets(TD, *GEP, GEPOffset))
        break;
      Offset += GEPOffset;
      Ptr = GEP->getPointerOperand();
      if (!Visited.insert(Ptr))
        break;
    }

    // See if we can perform a natural GEP here.
    Indices.clear();
    if (Value *P = getNaturalGEPWithOffset(IRB, TD, Ptr, Offset, TargetTy,
                                           Indices, Prefix)) {
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
                                  Prefix + ".raw_cast");
      Int8PtrOffset = Offset;
    }

    OffsetPtr = Int8PtrOffset == 0 ? Int8Ptr :
      IRB.CreateInBoundsGEP(Int8Ptr, IRB.getInt(Int8PtrOffset),
                            Prefix + ".raw_idx");
  }
  Ptr = OffsetPtr;

  // On the off chance we were targeting i8*, guard the bitcast here.
  if (Ptr->getType() != PointerTy)
    Ptr = IRB.CreateBitCast(Ptr, PointerTy, Prefix + ".cast");

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
  if (DL.getTypeSizeInBits(NewTy) != DL.getTypeSizeInBits(OldTy))
    return false;
  if (!NewTy->isSingleValueType() || !OldTy->isSingleValueType())
    return false;

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
static Value *convertValue(const DataLayout &DL, IRBuilder<> &IRB, Value *V,
                           Type *Ty) {
  assert(canConvertValue(DL, V->getType(), Ty) &&
         "Value not convertable to type");
  if (V->getType() == Ty)
    return V;
  if (V->getType()->isIntegerTy() && Ty->isPointerTy())
    return IRB.CreateIntToPtr(V, Ty);
  if (V->getType()->isPointerTy() && Ty->isIntegerTy())
    return IRB.CreatePtrToInt(V, Ty);

  return IRB.CreateBitCast(V, Ty);
}

/// \brief Test whether the given alloca partition can be promoted to a vector.
///
/// This is a quick test to check whether we can rewrite a particular alloca
/// partition (and its newly formed alloca) into a vector alloca with only
/// whole-vector loads and stores such that it could be promoted to a vector
/// SSA value. We only can ensure this for a limited set of operations, and we
/// don't want to do the rewrites unless we are confident that the result will
/// be promotable, so we have an early test here.
static bool isVectorPromotionViable(const DataLayout &TD,
                                    Type *AllocaTy,
                                    AllocaPartitioning &P,
                                    uint64_t PartitionBeginOffset,
                                    uint64_t PartitionEndOffset,
                                    AllocaPartitioning::const_use_iterator I,
                                    AllocaPartitioning::const_use_iterator E) {
  VectorType *Ty = dyn_cast<VectorType>(AllocaTy);
  if (!Ty)
    return false;

  uint64_t VecSize = TD.getTypeSizeInBits(Ty);
  uint64_t ElementSize = Ty->getScalarSizeInBits();

  // While the definition of LLVM vectors is bitpacked, we don't support sizes
  // that aren't byte sized.
  if (ElementSize % 8)
    return false;
  assert((VecSize % 8) == 0 && "vector size not a multiple of element size?");
  VecSize /= 8;
  ElementSize /= 8;

  for (; I != E; ++I) {
    if (!I->U)
      continue; // Skip dead use.

    uint64_t BeginOffset = I->BeginOffset - PartitionBeginOffset;
    uint64_t BeginIndex = BeginOffset / ElementSize;
    if (BeginIndex * ElementSize != BeginOffset ||
        BeginIndex >= Ty->getNumElements())
      return false;
    uint64_t EndOffset = I->EndOffset - PartitionBeginOffset;
    uint64_t EndIndex = EndOffset / ElementSize;
    if (EndIndex * ElementSize != EndOffset ||
        EndIndex > Ty->getNumElements())
      return false;

    // FIXME: We should build shuffle vector instructions to handle
    // non-element-sized accesses.
    if ((EndOffset - BeginOffset) != ElementSize &&
        (EndOffset - BeginOffset) != VecSize)
      return false;

    if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I->U->getUser())) {
      if (MI->isVolatile())
        return false;
      if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(I->U->getUser())) {
        const AllocaPartitioning::MemTransferOffsets &MTO
          = P.getMemTransferOffsets(*MTI);
        if (!MTO.IsSplittable)
          return false;
      }
    } else if (I->U->get()->getType()->getPointerElementType()->isStructTy()) {
      // Disable vector promotion when there are loads or stores of an FCA.
      return false;
    } else if (!isa<LoadInst>(I->U->getUser()) &&
               !isa<StoreInst>(I->U->getUser())) {
      return false;
    }
  }
  return true;
}

/// \brief Test whether the given alloca partition's integer operations can be
/// widened to promotable ones.
///
/// This is a quick test to check whether we can rewrite the integer loads and
/// stores to a particular alloca into wider loads and stores and be able to
/// promote the resulting alloca.
static bool isIntegerWideningViable(const DataLayout &TD,
                                    Type *AllocaTy,
                                    uint64_t AllocBeginOffset,
                                    AllocaPartitioning &P,
                                    AllocaPartitioning::const_use_iterator I,
                                    AllocaPartitioning::const_use_iterator E) {
  uint64_t SizeInBits = TD.getTypeSizeInBits(AllocaTy);

  // Don't try to handle allocas with bit-padding.
  if (SizeInBits != TD.getTypeStoreSizeInBits(AllocaTy))
    return false;

  // We need to ensure that an integer type with the appropriate bitwidth can
  // be converted to the alloca type, whatever that is. We don't want to force
  // the alloca itself to have an integer type if there is a more suitable one.
  Type *IntTy = Type::getIntNTy(AllocaTy->getContext(), SizeInBits);
  if (!canConvertValue(TD, AllocaTy, IntTy) ||
      !canConvertValue(TD, IntTy, AllocaTy))
    return false;

  uint64_t Size = TD.getTypeStoreSize(AllocaTy);

  // Check the uses to ensure the uses are (likely) promoteable integer uses.
  // Also ensure that the alloca has a covering load or store. We don't want
  // to widen the integer operotains only to fail to promote due to some other
  // unsplittable entry (which we may make splittable later).
  bool WholeAllocaOp = false;
  for (; I != E; ++I) {
    if (!I->U)
      continue; // Skip dead use.

    uint64_t RelBegin = I->BeginOffset - AllocBeginOffset;
    uint64_t RelEnd = I->EndOffset - AllocBeginOffset;

    // We can't reasonably handle cases where the load or store extends past
    // the end of the aloca's type and into its padding.
    if (RelEnd > Size)
      return false;

    if (LoadInst *LI = dyn_cast<LoadInst>(I->U->getUser())) {
      if (LI->isVolatile())
        return false;
      if (RelBegin == 0 && RelEnd == Size)
        WholeAllocaOp = true;
      if (IntegerType *ITy = dyn_cast<IntegerType>(LI->getType())) {
        if (ITy->getBitWidth() < TD.getTypeStoreSize(ITy))
          return false;
        continue;
      }
      // Non-integer loads need to be convertible from the alloca type so that
      // they are promotable.
      if (RelBegin != 0 || RelEnd != Size ||
          !canConvertValue(TD, AllocaTy, LI->getType()))
        return false;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I->U->getUser())) {
      Type *ValueTy = SI->getValueOperand()->getType();
      if (SI->isVolatile())
        return false;
      if (RelBegin == 0 && RelEnd == Size)
        WholeAllocaOp = true;
      if (IntegerType *ITy = dyn_cast<IntegerType>(ValueTy)) {
        if (ITy->getBitWidth() < TD.getTypeStoreSize(ITy))
          return false;
        continue;
      }
      // Non-integer stores need to be convertible to the alloca type so that
      // they are promotable.
      if (RelBegin != 0 || RelEnd != Size ||
          !canConvertValue(TD, ValueTy, AllocaTy))
        return false;
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I->U->getUser())) {
      if (MI->isVolatile())
        return false;
      if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(I->U->getUser())) {
        const AllocaPartitioning::MemTransferOffsets &MTO
          = P.getMemTransferOffsets(*MTI);
        if (!MTO.IsSplittable)
          return false;
      }
    } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I->U->getUser())) {
      if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
          II->getIntrinsicID() != Intrinsic::lifetime_end)
        return false;
    } else {
      return false;
    }
  }
  return WholeAllocaOp;
}

static Value *extractInteger(const DataLayout &DL, IRBuilder<> &IRB, Value *V,
                             IntegerType *Ty, uint64_t Offset,
                             const Twine &Name) {
  IntegerType *IntTy = cast<IntegerType>(V->getType());
  assert(DL.getTypeStoreSize(Ty) + Offset <= DL.getTypeStoreSize(IntTy) &&
         "Element extends past full value");
  uint64_t ShAmt = 8*Offset;
  if (DL.isBigEndian())
    ShAmt = 8*(DL.getTypeStoreSize(IntTy) - DL.getTypeStoreSize(Ty) - Offset);
  if (ShAmt)
    V = IRB.CreateLShr(V, ShAmt, Name + ".shift");
  assert(Ty->getBitWidth() <= IntTy->getBitWidth() &&
         "Cannot extract to a larger integer!");
  if (Ty != IntTy)
    V = IRB.CreateTrunc(V, Ty, Name + ".trunc");
  return V;
}

static Value *insertInteger(const DataLayout &DL, IRBuilder<> &IRB, Value *Old,
                            Value *V, uint64_t Offset, const Twine &Name) {
  IntegerType *IntTy = cast<IntegerType>(Old->getType());
  IntegerType *Ty = cast<IntegerType>(V->getType());
  assert(Ty->getBitWidth() <= IntTy->getBitWidth() &&
         "Cannot insert a larger integer!");
  if (Ty != IntTy)
    V = IRB.CreateZExt(V, IntTy, Name + ".ext");
  assert(DL.getTypeStoreSize(Ty) + Offset <= DL.getTypeStoreSize(IntTy) &&
         "Element store outside of alloca store");
  uint64_t ShAmt = 8*Offset;
  if (DL.isBigEndian())
    ShAmt = 8*(DL.getTypeStoreSize(IntTy) - DL.getTypeStoreSize(Ty) - Offset);
  if (ShAmt)
    V = IRB.CreateShl(V, ShAmt, Name + ".shift");

  if (ShAmt || Ty->getBitWidth() < IntTy->getBitWidth()) {
    APInt Mask = ~Ty->getMask().zext(IntTy->getBitWidth()).shl(ShAmt);
    Old = IRB.CreateAnd(Old, Mask, Name + ".mask");
    V = IRB.CreateOr(Old, V, Name + ".insert");
  }
  return V;
}

namespace {
/// \brief Visitor to rewrite instructions using a partition of an alloca to
/// use a new alloca.
///
/// Also implements the rewriting to vector-based accesses when the partition
/// passes the isVectorPromotionViable predicate. Most of the rewriting logic
/// lives here.
class AllocaPartitionRewriter : public InstVisitor<AllocaPartitionRewriter,
                                                   bool> {
  // Befriend the base class so it can delegate to private visit methods.
  friend class llvm::InstVisitor<AllocaPartitionRewriter, bool>;

  const DataLayout &TD;
  AllocaPartitioning &P;
  SROA &Pass;
  AllocaInst &OldAI, &NewAI;
  const uint64_t NewAllocaBeginOffset, NewAllocaEndOffset;
  Type *NewAllocaTy;

  // If we are rewriting an alloca partition which can be written as pure
  // vector operations, we stash extra information here. When VecTy is
  // non-null, we have some strict guarantees about the rewriten alloca:
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

  // The offset of the partition user currently being rewritten.
  uint64_t BeginOffset, EndOffset;
  Use *OldUse;
  Instruction *OldPtr;

  // The name prefix to use when rewriting instructions for this alloca.
  std::string NamePrefix;

public:
  AllocaPartitionRewriter(const DataLayout &TD, AllocaPartitioning &P,
                          AllocaPartitioning::iterator PI,
                          SROA &Pass, AllocaInst &OldAI, AllocaInst &NewAI,
                          uint64_t NewBeginOffset, uint64_t NewEndOffset)
    : TD(TD), P(P), Pass(Pass),
      OldAI(OldAI), NewAI(NewAI),
      NewAllocaBeginOffset(NewBeginOffset),
      NewAllocaEndOffset(NewEndOffset),
      NewAllocaTy(NewAI.getAllocatedType()),
      VecTy(), ElementTy(), ElementSize(), IntTy(),
      BeginOffset(), EndOffset() {
  }

  /// \brief Visit the users of the alloca partition and rewrite them.
  bool visitUsers(AllocaPartitioning::const_use_iterator I,
                  AllocaPartitioning::const_use_iterator E) {
    if (isVectorPromotionViable(TD, NewAI.getAllocatedType(), P,
                                NewAllocaBeginOffset, NewAllocaEndOffset,
                                I, E)) {
      ++NumVectorized;
      VecTy = cast<VectorType>(NewAI.getAllocatedType());
      ElementTy = VecTy->getElementType();
      assert((VecTy->getScalarSizeInBits() % 8) == 0 &&
             "Only multiple-of-8 sized vector elements are viable");
      ElementSize = VecTy->getScalarSizeInBits() / 8;
    } else if (isIntegerWideningViable(TD, NewAI.getAllocatedType(),
                                       NewAllocaBeginOffset, P, I, E)) {
      IntTy = Type::getIntNTy(NewAI.getContext(),
                              TD.getTypeSizeInBits(NewAI.getAllocatedType()));
    }
    bool CanSROA = true;
    for (; I != E; ++I) {
      if (!I->U)
        continue; // Skip dead uses.
      BeginOffset = I->BeginOffset;
      EndOffset = I->EndOffset;
      OldUse = I->U;
      OldPtr = cast<Instruction>(I->U->get());
      NamePrefix = (Twine(NewAI.getName()) + "." + Twine(BeginOffset)).str();
      CanSROA &= visit(cast<Instruction>(I->U->getUser()));
    }
    if (VecTy) {
      assert(CanSROA);
      VecTy = 0;
      ElementTy = 0;
      ElementSize = 0;
    }
    if (IntTy) {
      assert(CanSROA);
      IntTy = 0;
    }
    return CanSROA;
  }

private:
  // Every instruction which can end up as a user must have a rewrite rule.
  bool visitInstruction(Instruction &I) {
    DEBUG(dbgs() << "    !!!! Cannot rewrite: " << I << "\n");
    llvm_unreachable("No rewrite rule for this instruction!");
  }

  Twine getName(const Twine &Suffix) {
    return NamePrefix + Suffix;
  }

  Value *getAdjustedAllocaPtr(IRBuilder<> &IRB, Type *PointerTy) {
    assert(BeginOffset >= NewAllocaBeginOffset);
    APInt Offset(TD.getPointerSizeInBits(), BeginOffset - NewAllocaBeginOffset);
    return getAdjustedPtr(IRB, TD, &NewAI, Offset, PointerTy, getName(""));
  }

  /// \brief Compute suitable alignment to access an offset into the new alloca.
  unsigned getOffsetAlign(uint64_t Offset) {
    unsigned NewAIAlign = NewAI.getAlignment();
    if (!NewAIAlign)
      NewAIAlign = TD.getABITypeAlignment(NewAI.getAllocatedType());
    return MinAlign(NewAIAlign, Offset);
  }

  /// \brief Compute suitable alignment to access this partition of the new
  /// alloca.
  unsigned getPartitionAlign() {
    return getOffsetAlign(BeginOffset - NewAllocaBeginOffset);
  }

  /// \brief Compute suitable alignment to access a type at an offset of the
  /// new alloca.
  ///
  /// \returns zero if the type's ABI alignment is a suitable alignment,
  /// otherwise returns the maximal suitable alignment.
  unsigned getOffsetTypeAlign(Type *Ty, uint64_t Offset) {
    unsigned Align = getOffsetAlign(Offset);
    return Align == TD.getABITypeAlignment(Ty) ? 0 : Align;
  }

  /// \brief Compute suitable alignment to access a type at the beginning of
  /// this partition of the new alloca.
  ///
  /// See \c getOffsetTypeAlign for details; this routine delegates to it.
  unsigned getPartitionTypeAlign(Type *Ty) {
    return getOffsetTypeAlign(Ty, BeginOffset - NewAllocaBeginOffset);
  }

  ConstantInt *getIndex(IRBuilder<> &IRB, uint64_t Offset) {
    assert(VecTy && "Can only call getIndex when rewriting a vector");
    uint64_t RelOffset = Offset - NewAllocaBeginOffset;
    assert(RelOffset / ElementSize < UINT32_MAX && "Index out of bounds");
    uint32_t Index = RelOffset / ElementSize;
    assert(Index * ElementSize == RelOffset);
    return IRB.getInt32(Index);
  }

  void deleteIfTriviallyDead(Value *V) {
    Instruction *I = cast<Instruction>(V);
    if (isInstructionTriviallyDead(I))
      Pass.DeadInsts.push_back(I);
  }

  bool rewriteVectorizedLoadInst(IRBuilder<> &IRB, LoadInst &LI, Value *OldOp) {
    Value *Result;
    if (LI.getType() == VecTy->getElementType() ||
        BeginOffset > NewAllocaBeginOffset || EndOffset < NewAllocaEndOffset) {
      Result = IRB.CreateExtractElement(
        IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(), getName(".load")),
        getIndex(IRB, BeginOffset), getName(".extract"));
    } else {
      Result = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                     getName(".load"));
    }
    if (Result->getType() != LI.getType())
      Result = convertValue(TD, IRB, Result, LI.getType());
    LI.replaceAllUsesWith(Result);
    Pass.DeadInsts.push_back(&LI);

    DEBUG(dbgs() << "          to: " << *Result << "\n");
    return true;
  }

  bool rewriteIntegerLoad(IRBuilder<> &IRB, LoadInst &LI) {
    assert(IntTy && "We cannot insert an integer to the alloca");
    assert(!LI.isVolatile());
    Value *V = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                     getName(".load"));
    V = convertValue(TD, IRB, V, IntTy);
    assert(BeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
    uint64_t Offset = BeginOffset - NewAllocaBeginOffset;
    V = extractInteger(TD, IRB, V, cast<IntegerType>(LI.getType()), Offset,
                       getName(".extract"));
    LI.replaceAllUsesWith(V);
    Pass.DeadInsts.push_back(&LI);
    DEBUG(dbgs() << "          to: " << *V << "\n");
    return true;
  }

  bool visitLoadInst(LoadInst &LI) {
    DEBUG(dbgs() << "    original: " << LI << "\n");
    Value *OldOp = LI.getOperand(0);
    assert(OldOp == OldPtr);
    IRBuilder<> IRB(&LI);

    uint64_t Size = EndOffset - BeginOffset;
    if (Size < TD.getTypeStoreSize(LI.getType())) {
      assert(!LI.isVolatile());
      assert(LI.getType()->isIntegerTy() &&
             "Only integer type loads and stores are split");
      assert(LI.getType()->getIntegerBitWidth() ==
             TD.getTypeStoreSizeInBits(LI.getType()) &&
             "Non-byte-multiple bit width");
      assert(LI.getType()->getIntegerBitWidth() ==
             TD.getTypeAllocSizeInBits(OldAI.getAllocatedType()) &&
             "Only alloca-wide loads can be split and recomposed");
      IntegerType *NarrowTy = Type::getIntNTy(LI.getContext(), Size * 8);
      bool IsConvertable = (BeginOffset - NewAllocaBeginOffset == 0) &&
                           canConvertValue(TD, NewAllocaTy, NarrowTy);
      Value *V;
      // Move the insertion point just past the load so that we can refer to it.
      IRB.SetInsertPoint(llvm::next(BasicBlock::iterator(&LI)));
      if (IsConvertable)
        V = convertValue(TD, IRB,
                         IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                               getName(".load")),
                         NarrowTy);
      else
        V = IRB.CreateAlignedLoad(
          getAdjustedAllocaPtr(IRB, NarrowTy->getPointerTo()),
          getPartitionTypeAlign(NarrowTy), getName(".load"));
      // Create a placeholder value with the same type as LI to use as the
      // basis for the new value. This allows us to replace the uses of LI with
      // the computed value, and then replace the placeholder with LI, leaving
      // LI only used for this computation.
      Value *Placeholder
        = new LoadInst(UndefValue::get(LI.getType()->getPointerTo()));
      V = insertInteger(TD, IRB, Placeholder, V, BeginOffset,
                        getName(".insert"));
      LI.replaceAllUsesWith(V);
      Placeholder->replaceAllUsesWith(&LI);
      delete Placeholder;
      if (Pass.DeadSplitInsts.insert(&LI))
        Pass.DeadInsts.push_back(&LI);
      DEBUG(dbgs() << "          to: " << *V << "\n");
      return IsConvertable;
    }

    if (VecTy)
      return rewriteVectorizedLoadInst(IRB, LI, OldOp);
    if (IntTy && LI.getType()->isIntegerTy())
      return rewriteIntegerLoad(IRB, LI);

    if (BeginOffset == NewAllocaBeginOffset &&
        canConvertValue(TD, NewAllocaTy, LI.getType())) {
      Value *NewLI = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                           LI.isVolatile(), getName(".load"));
      Value *NewV = convertValue(TD, IRB, NewLI, LI.getType());
      LI.replaceAllUsesWith(NewV);
      Pass.DeadInsts.push_back(&LI);

      DEBUG(dbgs() << "          to: " << *NewLI << "\n");
      return !LI.isVolatile();
    }

    assert(!IntTy && "Invalid load found with int-op widening enabled");

    Value *NewPtr = getAdjustedAllocaPtr(IRB,
                                         LI.getPointerOperand()->getType());
    LI.setOperand(0, NewPtr);
    LI.setAlignment(getPartitionTypeAlign(LI.getType()));
    DEBUG(dbgs() << "          to: " << LI << "\n");

    deleteIfTriviallyDead(OldOp);
    return NewPtr == &NewAI && !LI.isVolatile();
  }

  bool rewriteVectorizedStoreInst(IRBuilder<> &IRB, StoreInst &SI,
                                  Value *OldOp) {
    Value *V = SI.getValueOperand();
    if (V->getType() == ElementTy ||
        BeginOffset > NewAllocaBeginOffset || EndOffset < NewAllocaEndOffset) {
      if (V->getType() != ElementTy)
        V = convertValue(TD, IRB, V, ElementTy);
      LoadInst *LI = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                           getName(".load"));
      V = IRB.CreateInsertElement(LI, V, getIndex(IRB, BeginOffset),
                                  getName(".insert"));
    } else if (V->getType() != VecTy) {
      V = convertValue(TD, IRB, V, VecTy);
    }
    StoreInst *Store = IRB.CreateAlignedStore(V, &NewAI, NewAI.getAlignment());
    Pass.DeadInsts.push_back(&SI);

    (void)Store;
    DEBUG(dbgs() << "          to: " << *Store << "\n");
    return true;
  }

  bool rewriteIntegerStore(IRBuilder<> &IRB, StoreInst &SI) {
    assert(IntTy && "We cannot extract an integer from the alloca");
    assert(!SI.isVolatile());
    Value *V = SI.getValueOperand();
    if (TD.getTypeSizeInBits(V->getType()) != IntTy->getBitWidth()) {
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         getName(".oldload"));
      Old = convertValue(TD, IRB, Old, IntTy);
      assert(BeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
      uint64_t Offset = BeginOffset - NewAllocaBeginOffset;
      V = insertInteger(TD, IRB, Old, SI.getValueOperand(), Offset,
                        getName(".insert"));
    }
    V = convertValue(TD, IRB, V, NewAllocaTy);
    StoreInst *Store = IRB.CreateAlignedStore(V, &NewAI, NewAI.getAlignment());
    Pass.DeadInsts.push_back(&SI);
    (void)Store;
    DEBUG(dbgs() << "          to: " << *Store << "\n");
    return true;
  }

  bool visitStoreInst(StoreInst &SI) {
    DEBUG(dbgs() << "    original: " << SI << "\n");
    Value *OldOp = SI.getOperand(1);
    assert(OldOp == OldPtr);
    IRBuilder<> IRB(&SI);

    if (VecTy)
      return rewriteVectorizedStoreInst(IRB, SI, OldOp);
    Type *ValueTy = SI.getValueOperand()->getType();

    uint64_t Size = EndOffset - BeginOffset;
    if (Size < TD.getTypeStoreSize(ValueTy)) {
      assert(!SI.isVolatile());
      assert(ValueTy->isIntegerTy() &&
             "Only integer type loads and stores are split");
      assert(ValueTy->getIntegerBitWidth() ==
             TD.getTypeStoreSizeInBits(ValueTy) &&
             "Non-byte-multiple bit width");
      assert(ValueTy->getIntegerBitWidth() ==
             TD.getTypeSizeInBits(OldAI.getAllocatedType()) &&
             "Only alloca-wide stores can be split and recomposed");
      IntegerType *NarrowTy = Type::getIntNTy(SI.getContext(), Size * 8);
      Value *V = extractInteger(TD, IRB, SI.getValueOperand(), NarrowTy,
                                BeginOffset, getName(".extract"));
      StoreInst *NewSI;
      bool IsConvertable = (BeginOffset - NewAllocaBeginOffset == 0) &&
                           canConvertValue(TD, NarrowTy, NewAllocaTy);
      if (IsConvertable)
        NewSI = IRB.CreateAlignedStore(convertValue(TD, IRB, V, NewAllocaTy),
                                       &NewAI, NewAI.getAlignment());
      else
        NewSI = IRB.CreateAlignedStore(
          V, getAdjustedAllocaPtr(IRB, NarrowTy->getPointerTo()),
          getPartitionTypeAlign(NarrowTy));
      (void)NewSI;
      if (Pass.DeadSplitInsts.insert(&SI))
        Pass.DeadInsts.push_back(&SI);

      DEBUG(dbgs() << "          to: " << *NewSI << "\n");
      return IsConvertable;
    }

    if (IntTy && ValueTy->isIntegerTy())
      return rewriteIntegerStore(IRB, SI);

    // Strip all inbounds GEPs and pointer casts to try to dig out any root
    // alloca that should be re-examined after promoting this alloca.
    if (ValueTy->isPointerTy())
      if (AllocaInst *AI = dyn_cast<AllocaInst>(SI.getValueOperand()
                                                  ->stripInBoundsOffsets()))
        Pass.PostPromotionWorklist.insert(AI);

    if (BeginOffset == NewAllocaBeginOffset &&
        canConvertValue(TD, ValueTy, NewAllocaTy)) {
      Value *NewV = convertValue(TD, IRB, SI.getValueOperand(), NewAllocaTy);
      StoreInst *NewSI = IRB.CreateAlignedStore(NewV, &NewAI, NewAI.getAlignment(),
                                                SI.isVolatile());
      (void)NewSI;
      Pass.DeadInsts.push_back(&SI);

      DEBUG(dbgs() << "          to: " << *NewSI << "\n");
      return !SI.isVolatile();
    }

    assert(!IntTy && "Invalid store found with int-op widening enabled");

    Value *NewPtr = getAdjustedAllocaPtr(IRB,
                                         SI.getPointerOperand()->getType());
    SI.setOperand(1, NewPtr);
    SI.setAlignment(getPartitionTypeAlign(SI.getValueOperand()->getType()));
    DEBUG(dbgs() << "          to: " << SI << "\n");

    deleteIfTriviallyDead(OldOp);
    return NewPtr == &NewAI && !SI.isVolatile();
  }

  bool visitMemSetInst(MemSetInst &II) {
    DEBUG(dbgs() << "    original: " << II << "\n");
    IRBuilder<> IRB(&II);
    assert(II.getRawDest() == OldPtr);

    // If the memset has a variable size, it cannot be split, just adjust the
    // pointer to the new alloca.
    if (!isa<Constant>(II.getLength())) {
      II.setDest(getAdjustedAllocaPtr(IRB, II.getRawDest()->getType()));
      Type *CstTy = II.getAlignmentCst()->getType();
      II.setAlignment(ConstantInt::get(CstTy, getPartitionAlign()));

      deleteIfTriviallyDead(OldPtr);
      return false;
    }

    // Record this instruction for deletion.
    if (Pass.DeadSplitInsts.insert(&II))
      Pass.DeadInsts.push_back(&II);

    Type *AllocaTy = NewAI.getAllocatedType();
    Type *ScalarTy = AllocaTy->getScalarType();

    // If this doesn't map cleanly onto the alloca type, and that type isn't
    // a single value type, just emit a memset.
    if (!VecTy && !IntTy &&
        (BeginOffset != NewAllocaBeginOffset ||
         EndOffset != NewAllocaEndOffset ||
         !AllocaTy->isSingleValueType() ||
         !TD.isLegalInteger(TD.getTypeSizeInBits(ScalarTy)))) {
      Type *SizeTy = II.getLength()->getType();
      Constant *Size = ConstantInt::get(SizeTy, EndOffset - BeginOffset);
      CallInst *New
        = IRB.CreateMemSet(getAdjustedAllocaPtr(IRB,
                                                II.getRawDest()->getType()),
                           II.getValue(), Size, getPartitionAlign(),
                           II.isVolatile());
      (void)New;
      DEBUG(dbgs() << "          to: " << *New << "\n");
      return false;
    }

    // If we can represent this as a simple value, we have to build the actual
    // value to store, which requires expanding the byte present in memset to
    // a sensible representation for the alloca type. This is essentially
    // splatting the byte to a sufficiently wide integer, bitcasting to the
    // desired scalar type, and splatting it across any desired vector type.
    uint64_t Size = EndOffset - BeginOffset;
    Value *V = II.getValue();
    IntegerType *VTy = cast<IntegerType>(V->getType());
    Type *SplatIntTy = Type::getIntNTy(VTy->getContext(), Size*8);
    if (Size*8 > VTy->getBitWidth())
      V = IRB.CreateMul(IRB.CreateZExt(V, SplatIntTy, getName(".zext")),
                        ConstantExpr::getUDiv(
                          Constant::getAllOnesValue(SplatIntTy),
                          ConstantExpr::getZExt(
                            Constant::getAllOnesValue(V->getType()),
                            SplatIntTy)),
                        getName(".isplat"));

    // If this is an element-wide memset of a vectorizable alloca, insert it.
    if (VecTy && (BeginOffset > NewAllocaBeginOffset ||
                  EndOffset < NewAllocaEndOffset)) {
      if (V->getType() != ScalarTy)
        V = convertValue(TD, IRB, V, ScalarTy);
      StoreInst *Store = IRB.CreateAlignedStore(
        IRB.CreateInsertElement(IRB.CreateAlignedLoad(&NewAI,
                                                      NewAI.getAlignment(),
                                                      getName(".load")),
                                V, getIndex(IRB, BeginOffset),
                                getName(".insert")),
        &NewAI, NewAI.getAlignment());
      (void)Store;
      DEBUG(dbgs() << "          to: " << *Store << "\n");
      return true;
    }

    // If this is a memset on an alloca where we can widen stores, insert the
    // set integer.
    if (IntTy && (BeginOffset > NewAllocaBeginOffset ||
                  EndOffset < NewAllocaEndOffset)) {
      assert(!II.isVolatile());
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         getName(".oldload"));
      Old = convertValue(TD, IRB, Old, IntTy);
      assert(BeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
      uint64_t Offset = BeginOffset - NewAllocaBeginOffset;
      V = insertInteger(TD, IRB, Old, V, Offset, getName(".insert"));
    }

    if (V->getType() != AllocaTy)
      V = convertValue(TD, IRB, V, AllocaTy);

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
    IRBuilder<> IRB(&II);

    assert(II.getRawSource() == OldPtr || II.getRawDest() == OldPtr);
    bool IsDest = II.getRawDest() == OldPtr;

    const AllocaPartitioning::MemTransferOffsets &MTO
      = P.getMemTransferOffsets(II);

    // Compute the relative offset within the transfer.
    unsigned IntPtrWidth = TD.getPointerSizeInBits();
    APInt RelOffset(IntPtrWidth, BeginOffset - (IsDest ? MTO.DestBegin
                                                       : MTO.SourceBegin));

    unsigned Align = II.getAlignment();
    if (Align > 1)
      Align = MinAlign(RelOffset.zextOrTrunc(64).getZExtValue(),
                       MinAlign(II.getAlignment(), getPartitionAlign()));

    // For unsplit intrinsics, we simply modify the source and destination
    // pointers in place. This isn't just an optimization, it is a matter of
    // correctness. With unsplit intrinsics we may be dealing with transfers
    // within a single alloca before SROA ran, or with transfers that have
    // a variable length. We may also be dealing with memmove instead of
    // memcpy, and so simply updating the pointers is the necessary for us to
    // update both source and dest of a single call.
    if (!MTO.IsSplittable) {
      Value *OldOp = IsDest ? II.getRawDest() : II.getRawSource();
      if (IsDest)
        II.setDest(getAdjustedAllocaPtr(IRB, II.getRawDest()->getType()));
      else
        II.setSource(getAdjustedAllocaPtr(IRB, II.getRawSource()->getType()));

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
      = !VecTy && !IntTy && (BeginOffset != NewAllocaBeginOffset ||
                             EndOffset != NewAllocaEndOffset ||
                             !NewAI.getAllocatedType()->isSingleValueType());

    // If we're just going to emit a memcpy, the alloca hasn't changed, and the
    // size hasn't been shrunk based on analysis of the viable range, this is
    // a no-op.
    if (EmitMemCpy && &OldAI == &NewAI) {
      uint64_t OrigBegin = IsDest ? MTO.DestBegin : MTO.SourceBegin;
      uint64_t OrigEnd = IsDest ? MTO.DestEnd : MTO.SourceEnd;
      // Ensure the start lines up.
      assert(BeginOffset == OrigBegin);
      (void)OrigBegin;

      // Rewrite the size as needed.
      if (EndOffset != OrigEnd)
        II.setLength(ConstantInt::get(II.getLength()->getType(),
                                      EndOffset - BeginOffset));
      return false;
    }
    // Record this instruction for deletion.
    if (Pass.DeadSplitInsts.insert(&II))
      Pass.DeadInsts.push_back(&II);

    bool IsWholeAlloca = BeginOffset == NewAllocaBeginOffset &&
                         EndOffset == NewAllocaEndOffset;
    bool IsVectorElement = VecTy && !IsWholeAlloca;
    uint64_t Size = EndOffset - BeginOffset;
    IntegerType *SubIntTy
      = IntTy ? Type::getIntNTy(IntTy->getContext(), Size*8) : 0;

    Type *OtherPtrTy = IsDest ? II.getRawSource()->getType()
                              : II.getRawDest()->getType();
    if (!EmitMemCpy) {
      if (IsVectorElement)
        OtherPtrTy = VecTy->getElementType()->getPointerTo();
      else if (IntTy && !IsWholeAlloca)
        OtherPtrTy = SubIntTy->getPointerTo();
      else
        OtherPtrTy = NewAI.getType();
    }

    // Compute the other pointer, folding as much as possible to produce
    // a single, simple GEP in most cases.
    Value *OtherPtr = IsDest ? II.getRawSource() : II.getRawDest();
    OtherPtr = getAdjustedPtr(IRB, TD, OtherPtr, RelOffset, OtherPtrTy,
                              getName("." + OtherPtr->getName()));

    // Strip all inbounds GEPs and pointer casts to try to dig out any root
    // alloca that should be re-examined after rewriting this instruction.
    if (AllocaInst *AI
          = dyn_cast<AllocaInst>(OtherPtr->stripInBoundsOffsets()))
      Pass.Worklist.insert(AI);

    if (EmitMemCpy) {
      Value *OurPtr
        = getAdjustedAllocaPtr(IRB, IsDest ? II.getRawDest()->getType()
                                           : II.getRawSource()->getType());
      Type *SizeTy = II.getLength()->getType();
      Constant *Size = ConstantInt::get(SizeTy, EndOffset - BeginOffset);

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

    Value *SrcPtr = OtherPtr;
    Value *DstPtr = &NewAI;
    if (!IsDest)
      std::swap(SrcPtr, DstPtr);

    Value *Src;
    if (IsVectorElement && !IsDest) {
      // We have to extract rather than load.
      Src = IRB.CreateExtractElement(
        IRB.CreateAlignedLoad(SrcPtr, Align, getName(".copyload")),
        getIndex(IRB, BeginOffset),
        getName(".copyextract"));
    } else if (IntTy && !IsWholeAlloca && !IsDest) {
      Src = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                  getName(".load"));
      Src = convertValue(TD, IRB, Src, IntTy);
      assert(BeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
      uint64_t Offset = BeginOffset - NewAllocaBeginOffset;
      Src = extractInteger(TD, IRB, Src, SubIntTy, Offset, getName(".extract"));
    } else {
      Src = IRB.CreateAlignedLoad(SrcPtr, Align, II.isVolatile(),
                                  getName(".copyload"));
    }

    if (IntTy && !IsWholeAlloca && IsDest) {
      Value *Old = IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(),
                                         getName(".oldload"));
      Old = convertValue(TD, IRB, Old, IntTy);
      assert(BeginOffset >= NewAllocaBeginOffset && "Out of bounds offset");
      uint64_t Offset = BeginOffset - NewAllocaBeginOffset;
      Src = insertInteger(TD, IRB, Old, Src, Offset, getName(".insert"));
      Src = convertValue(TD, IRB, Src, NewAllocaTy);
    }

    if (IsVectorElement && IsDest) {
      // We have to insert into a loaded copy before storing.
      Src = IRB.CreateInsertElement(
        IRB.CreateAlignedLoad(&NewAI, NewAI.getAlignment(), getName(".load")),
        Src, getIndex(IRB, BeginOffset),
        getName(".insert"));
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
    IRBuilder<> IRB(&II);
    assert(II.getArgOperand(1) == OldPtr);

    // Record this instruction for deletion.
    if (Pass.DeadSplitInsts.insert(&II))
      Pass.DeadInsts.push_back(&II);

    ConstantInt *Size
      = ConstantInt::get(cast<IntegerType>(II.getArgOperand(0)->getType()),
                         EndOffset - BeginOffset);
    Value *Ptr = getAdjustedAllocaPtr(IRB, II.getArgOperand(1)->getType());
    Value *New;
    if (II.getIntrinsicID() == Intrinsic::lifetime_start)
      New = IRB.CreateLifetimeStart(Ptr, Size);
    else
      New = IRB.CreateLifetimeEnd(Ptr, Size);

    DEBUG(dbgs() << "          to: " << *New << "\n");
    return true;
  }

  bool visitPHINode(PHINode &PN) {
    DEBUG(dbgs() << "    original: " << PN << "\n");

    // We would like to compute a new pointer in only one place, but have it be
    // as local as possible to the PHI. To do that, we re-use the location of
    // the old pointer, which necessarily must be in the right position to
    // dominate the PHI.
    IRBuilder<> PtrBuilder(cast<Instruction>(OldPtr));

    Value *NewPtr = getAdjustedAllocaPtr(PtrBuilder, OldPtr->getType());
    // Replace the operands which were using the old pointer.
    std::replace(PN.op_begin(), PN.op_end(), cast<Value>(OldPtr), NewPtr);

    DEBUG(dbgs() << "          to: " << PN << "\n");
    deleteIfTriviallyDead(OldPtr);
    return false;
  }

  bool visitSelectInst(SelectInst &SI) {
    DEBUG(dbgs() << "    original: " << SI << "\n");
    IRBuilder<> IRB(&SI);

    // Find the operand we need to rewrite here.
    bool IsTrueVal = SI.getTrueValue() == OldPtr;
    if (IsTrueVal)
      assert(SI.getFalseValue() != OldPtr && "Pointer is both operands!");
    else
      assert(SI.getFalseValue() == OldPtr && "Pointer isn't an operand!");

    Value *NewPtr = getAdjustedAllocaPtr(IRB, OldPtr->getType());
    SI.setOperand(IsTrueVal ? 1 : 2, NewPtr);
    DEBUG(dbgs() << "          to: " << SI << "\n");
    deleteIfTriviallyDead(OldPtr);
    return false;
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

  const DataLayout &TD;

  /// Queue of pointer uses to analyze and potentially rewrite.
  SmallVector<Use *, 8> Queue;

  /// Set to prevent us from cycling with phi nodes and loops.
  SmallPtrSet<User *, 8> Visited;

  /// The current pointer use being rewritten. This is used to dig up the used
  /// value (as opposed to the user).
  Use *U;

public:
  AggLoadStoreRewriter(const DataLayout &TD) : TD(TD) {}

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
    IRBuilder<> IRB;
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
      Value *Load = IRB.CreateLoad(IRB.CreateInBoundsGEP(Ptr, GEPIndices,
                                                         Name + ".gep"),
                                   Name + ".load");
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
static Type *getTypePartition(const DataLayout &TD, Type *Ty,
                              uint64_t Offset, uint64_t Size) {
  if (Offset == 0 && TD.getTypeAllocSize(Ty) == Size)
    return stripAggregateTypeWrapping(TD, Ty);
  if (Offset > TD.getTypeAllocSize(Ty) ||
      (TD.getTypeAllocSize(Ty) - Offset) < Size)
    return 0;

  if (SequentialType *SeqTy = dyn_cast<SequentialType>(Ty)) {
    // We can't partition pointers...
    if (SeqTy->isPointerTy())
      return 0;

    Type *ElementTy = SeqTy->getElementType();
    uint64_t ElementSize = TD.getTypeAllocSize(ElementTy);
    uint64_t NumSkippedElements = Offset / ElementSize;
    if (ArrayType *ArrTy = dyn_cast<ArrayType>(SeqTy))
      if (NumSkippedElements >= ArrTy->getNumElements())
        return 0;
    if (VectorType *VecTy = dyn_cast<VectorType>(SeqTy))
      if (NumSkippedElements >= VecTy->getNumElements())
        return 0;
    Offset -= NumSkippedElements * ElementSize;

    // First check if we need to recurse.
    if (Offset > 0 || Size < ElementSize) {
      // Bail if the partition ends in a different array element.
      if ((Offset + Size) > ElementSize)
        return 0;
      // Recurse through the element type trying to peel off offset bytes.
      return getTypePartition(TD, ElementTy, Offset, Size);
    }
    assert(Offset == 0);

    if (Size == ElementSize)
      return stripAggregateTypeWrapping(TD, ElementTy);
    assert(Size > ElementSize);
    uint64_t NumElements = Size / ElementSize;
    if (NumElements * ElementSize != Size)
      return 0;
    return ArrayType::get(ElementTy, NumElements);
  }

  StructType *STy = dyn_cast<StructType>(Ty);
  if (!STy)
    return 0;

  const StructLayout *SL = TD.getStructLayout(STy);
  if (Offset >= SL->getSizeInBytes())
    return 0;
  uint64_t EndOffset = Offset + Size;
  if (EndOffset > SL->getSizeInBytes())
    return 0;

  unsigned Index = SL->getElementContainingOffset(Offset);
  Offset -= SL->getElementOffset(Index);

  Type *ElementTy = STy->getElementType(Index);
  uint64_t ElementSize = TD.getTypeAllocSize(ElementTy);
  if (Offset >= ElementSize)
    return 0; // The offset points into alignment padding.

  // See if any partition must be contained by the element.
  if (Offset > 0 || Size < ElementSize) {
    if ((Offset + Size) > ElementSize)
      return 0;
    return getTypePartition(TD, ElementTy, Offset, Size);
  }
  assert(Offset == 0);

  if (Size == ElementSize)
    return stripAggregateTypeWrapping(TD, ElementTy);

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
  const StructLayout *SubSL = TD.getStructLayout(SubTy);
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
bool SROA::rewriteAllocaPartition(AllocaInst &AI,
                                  AllocaPartitioning &P,
                                  AllocaPartitioning::iterator PI) {
  uint64_t AllocaSize = PI->EndOffset - PI->BeginOffset;
  bool IsLive = false;
  for (AllocaPartitioning::use_iterator UI = P.use_begin(PI),
                                        UE = P.use_end(PI);
       UI != UE && !IsLive; ++UI)
    if (UI->U)
      IsLive = true;
  if (!IsLive)
    return false; // No live uses left of this partition.

  DEBUG(dbgs() << "Speculating PHIs and selects in partition "
               << "[" << PI->BeginOffset << "," << PI->EndOffset << ")\n");

  PHIOrSelectSpeculator Speculator(*TD, P, *this);
  DEBUG(dbgs() << "  speculating ");
  DEBUG(P.print(dbgs(), PI, ""));
  Speculator.visitUsers(PI);

  // Try to compute a friendly type for this partition of the alloca. This
  // won't always succeed, in which case we fall back to a legal integer type
  // or an i8 array of an appropriate size.
  Type *AllocaTy = 0;
  if (Type *PartitionTy = P.getCommonType(PI))
    if (TD->getTypeAllocSize(PartitionTy) >= AllocaSize)
      AllocaTy = PartitionTy;
  if (!AllocaTy)
    if (Type *PartitionTy = getTypePartition(*TD, AI.getAllocatedType(),
                                             PI->BeginOffset, AllocaSize))
      AllocaTy = PartitionTy;
  if ((!AllocaTy ||
       (AllocaTy->isArrayTy() &&
        AllocaTy->getArrayElementType()->isIntegerTy())) &&
      TD->isLegalInteger(AllocaSize * 8))
    AllocaTy = Type::getIntNTy(*C, AllocaSize * 8);
  if (!AllocaTy)
    AllocaTy = ArrayType::get(Type::getInt8Ty(*C), AllocaSize);
  assert(TD->getTypeAllocSize(AllocaTy) >= AllocaSize);

  // Check for the case where we're going to rewrite to a new alloca of the
  // exact same type as the original, and with the same access offsets. In that
  // case, re-use the existing alloca, but still run through the rewriter to
  // performe phi and select speculation.
  AllocaInst *NewAI;
  if (AllocaTy == AI.getAllocatedType()) {
    assert(PI->BeginOffset == 0 &&
           "Non-zero begin offset but same alloca type");
    assert(PI == P.begin() && "Begin offset is zero on later partition");
    NewAI = &AI;
  } else {
    unsigned Alignment = AI.getAlignment();
    if (!Alignment) {
      // The minimum alignment which users can rely on when the explicit
      // alignment is omitted or zero is that required by the ABI for this
      // type.
      Alignment = TD->getABITypeAlignment(AI.getAllocatedType());
    }
    Alignment = MinAlign(Alignment, PI->BeginOffset);
    // If we will get at least this much alignment from the type alone, leave
    // the alloca's alignment unconstrained.
    if (Alignment <= TD->getABITypeAlignment(AllocaTy))
      Alignment = 0;
    NewAI = new AllocaInst(AllocaTy, 0, Alignment,
                           AI.getName() + ".sroa." + Twine(PI - P.begin()),
                           &AI);
    ++NumNewAllocas;
  }

  DEBUG(dbgs() << "Rewriting alloca partition "
               << "[" << PI->BeginOffset << "," << PI->EndOffset << ") to: "
               << *NewAI << "\n");

  // Track the high watermark of the post-promotion worklist. We will reset it
  // to this point if the alloca is not in fact scheduled for promotion.
  unsigned PPWOldSize = PostPromotionWorklist.size();

  AllocaPartitionRewriter Rewriter(*TD, P, PI, *this, AI, *NewAI,
                                   PI->BeginOffset, PI->EndOffset);
  DEBUG(dbgs() << "  rewriting ");
  DEBUG(P.print(dbgs(), PI, ""));
  bool Promotable = Rewriter.visitUsers(P.use_begin(PI), P.use_end(PI));
  if (Promotable) {
    DEBUG(dbgs() << "  and queuing for promotion\n");
    PromotableAllocas.push_back(NewAI);
  } else if (NewAI != &AI) {
    // If we can't promote the alloca, iterate on it to check for new
    // refinements exposed by splitting the current alloca. Don't iterate on an
    // alloca which didn't actually change and didn't get promoted.
    Worklist.insert(NewAI);
  }

  // Drop any post-promotion work items if promotion didn't happen.
  if (!Promotable)
    while (PostPromotionWorklist.size() > PPWOldSize)
      PostPromotionWorklist.pop_back();

  return true;
}

/// \brief Walks the partitioning of an alloca rewriting uses of each partition.
bool SROA::splitAlloca(AllocaInst &AI, AllocaPartitioning &P) {
  bool Changed = false;
  for (AllocaPartitioning::iterator PI = P.begin(), PE = P.end(); PI != PE;
       ++PI)
    Changed |= rewriteAllocaPartition(AI, P, PI);

  return Changed;
}

/// \brief Analyze an alloca for SROA.
///
/// This analyzes the alloca to ensure we can reason about it, builds
/// a partitioning of the alloca, and then hands it off to be split and
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
      TD->getTypeAllocSize(AI.getAllocatedType()) == 0)
    return false;

  bool Changed = false;

  // First, split any FCA loads and stores touching this alloca to promote
  // better splitting and promotion opportunities.
  AggLoadStoreRewriter AggRewriter(*TD);
  Changed |= AggRewriter.rewrite(AI);

  // Build the partition set using a recursive instruction-visiting builder.
  AllocaPartitioning P(*TD, AI);
  DEBUG(P.print(dbgs()));
  if (P.isEscaped())
    return Changed;

  // Delete all the dead users of this alloca before splitting and rewriting it.
  for (AllocaPartitioning::dead_user_iterator DI = P.dead_user_begin(),
                                              DE = P.dead_user_end();
       DI != DE; ++DI) {
    Changed = true;
    (*DI)->replaceAllUsesWith(UndefValue::get((*DI)->getType()));
    DeadInsts.push_back(*DI);
  }
  for (AllocaPartitioning::dead_op_iterator DO = P.dead_op_begin(),
                                            DE = P.dead_op_end();
       DO != DE; ++DO) {
    Value *OldV = **DO;
    // Clobber the use with an undef value.
    **DO = UndefValue::get(OldV->getType());
    if (Instruction *OldI = dyn_cast<Instruction>(OldV))
      if (isInstructionTriviallyDead(OldI)) {
        Changed = true;
        DeadInsts.push_back(OldI);
      }
  }

  // No partitions to split. Leave the dead alloca for a later pass to clean up.
  if (P.begin() == P.end())
    return Changed;

  return splitAlloca(AI, P) || Changed;
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
  DeadSplitInsts.clear();
  while (!DeadInsts.empty()) {
    Instruction *I = DeadInsts.pop_back_val();
    DEBUG(dbgs() << "Deleting dead instruction: " << *I << "\n");

    I->replaceAllUsesWith(UndefValue::get(I->getType()));

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        // Zero out the operand and see if it becomes trivially dead.
        *OI = 0;
        if (isInstructionTriviallyDead(U))
          DeadInsts.push_back(U);
      }

    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      DeletedAllocas.insert(AI);

    ++NumDeleted;
    I->eraseFromParent();
  }
}

/// \brief Promote the allocas, using the best available technique.
///
/// This attempts to promote whatever allocas have been identified as viable in
/// the PromotableAllocas list. If that list is empty, there is nothing to do.
/// If there is a domtree available, we attempt to promote using the full power
/// of mem2reg. Otherwise, we build and use the AllocaPromoter above which is
/// based on the SSAUpdater utilities. This function returns whether any
/// promotion occured.
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
  SmallVector<Instruction*, 64> Insts;

  for (unsigned Idx = 0, Size = PromotableAllocas.size(); Idx != Size; ++Idx) {
    AllocaInst *AI = PromotableAllocas[Idx];
    for (Value::use_iterator UI = AI->use_begin(), UE = AI->use_end();
         UI != UE;) {
      Instruction *I = cast<Instruction>(*UI++);
      // FIXME: Currently the SSAUpdater infrastructure doesn't reason about
      // lifetime intrinsics and so we strip them (and the bitcasts+GEPs
      // leading to them) here. Eventually it should use them to optimize the
      // scalar values produced.
      if (isa<BitCastInst>(I) || isa<GetElementPtrInst>(I)) {
        assert(onlyUsedByLifetimeMarkers(I) &&
               "Found a bitcast used outside of a lifetime marker.");
        while (!I->use_empty())
          cast<Instruction>(*I->use_begin())->eraseFromParent();
        I->eraseFromParent();
        continue;
      }
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        assert(II->getIntrinsicID() == Intrinsic::lifetime_start ||
               II->getIntrinsicID() == Intrinsic::lifetime_end);
        II->eraseFromParent();
        continue;
      }

      Insts.push_back(I);
    }
    AllocaPromoter(Insts, SSA, *AI, DIB).run(Insts);
    Insts.clear();
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
  TD = getAnalysisIfAvailable<DataLayout>();
  if (!TD) {
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
