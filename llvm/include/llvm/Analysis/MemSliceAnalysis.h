//===- MemSliceAnalysis.h - Analyze memory slices -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines the MemSliceAnalysis infrastructure for analyzing
/// how memory is accessed and partitioning it into slices. Each slice
/// represents a contiguous region of the memory that is accessed.
///
/// The analysis provides:
/// - MemPartition iteration for non-overlapping regions
/// - Slice information (offsets, sizes, uses, splittability)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMSLICEANALYSIS_H
#define LLVM_ANALYSIS_MEMSLICEANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

class AllocaInst;
class DataLayout;
class IntrinsicInst;
class Type;
class Value;
class raw_ostream;

/// A used slice of an memory region.
///
/// This structure represents a slice of a ptr used by some instruction. It
/// stores both the begin and end offsets of this use, a pointer to the use
/// itself, and a flag indicating whether we can classify the use as splittable
/// or not when forming partitions of the memory.
class MemSlice {
  /// The beginning offset of the range.
  uint64_t BeginOffset = 0;

  /// The ending offset, not included in the range.
  uint64_t EndOffset = 0;

  /// Storage for both the use of this slice and whether it can be
  /// split.
  PointerIntPair<Use *, 1, bool> UseAndIsSplittable;

  /// When this access is via an llvm.protected.field.ptr intrinsic, contains
  /// the second argument to the intrinsic, the discriminator.
  Value *ProtectedFieldDisc = nullptr;

public:
  MemSlice() = default;

  MemSlice(uint64_t BeginOffset, uint64_t EndOffset, Use *U, bool IsSplittable,
           Value *ProtectedFieldDisc = nullptr)
      : BeginOffset(BeginOffset), EndOffset(EndOffset),
        UseAndIsSplittable(U, IsSplittable),
        ProtectedFieldDisc(ProtectedFieldDisc) {}

  uint64_t beginOffset() const { return BeginOffset; }
  uint64_t endOffset() const { return EndOffset; }

  bool isSplittable() const { return UseAndIsSplittable.getInt(); }
  void makeUnsplittable() { UseAndIsSplittable.setInt(false); }

  Use *getUse() const { return UseAndIsSplittable.getPointer(); }
  Value *getProtectedFieldDisc() const { return ProtectedFieldDisc; }

  bool isDead() const { return getUse() == nullptr; }
  void kill() { UseAndIsSplittable.setPointer(nullptr); }

  /// Support for ordering ranges.
  ///
  /// This provides an ordering over ranges such that start offsets are
  /// always increasing, and within equal start offsets, the end offsets are
  /// decreasing. Thus the spanning range comes first in a cluster with the
  /// same start position.
  bool operator<(const MemSlice &RHS) const {
    if (beginOffset() < RHS.beginOffset())
      return true;
    if (beginOffset() > RHS.beginOffset())
      return false;
    if (isSplittable() != RHS.isSplittable())
      return !isSplittable();
    if (endOffset() > RHS.endOffset())
      return true;
    return false;
  }

  /// Support comparison with a single offset to allow binary searches.
  [[maybe_unused]] friend bool operator<(const MemSlice &LHS,
                                         uint64_t RHSOffset) {
    return LHS.beginOffset() < RHSOffset;
  }
  [[maybe_unused]] friend bool operator<(uint64_t LHSOffset,
                                         const MemSlice &RHS) {
    return LHSOffset < RHS.beginOffset();
  }

  bool operator==(const MemSlice &RHS) const {
    return isSplittable() == RHS.isSplittable() &&
           beginOffset() == RHS.beginOffset() && endOffset() == RHS.endOffset();
  }
  bool operator!=(const MemSlice &RHS) const { return !operator==(RHS); }
};

/// Representation of the memory slices.
///
/// This class represents the slices of a memory ptr which are formed by its
/// various uses. If a pointer escapes, we can't fully build a representation
/// for the slices used and we reflect that in this structure. The uses are
/// stored, sorted by increasing beginning offset and with unsplittable slices
/// starting at a particular offset before splittable slices.
class MemSlices {
public:
  /// Construct the slices of a particular alloca.
  MemSlices(const DataLayout &DL, AllocaInst &AI);

  /// Test whether a pointer to the allocation escapes our analysis.
  ///
  /// If this is true, the slices are never fully built and should be
  /// ignored.
  bool isEscaped() const { return PointerEscapingInstr; }
  bool isEscapedReadOnly() const { return PointerEscapingInstrReadOnly; }

  /// Support for iterating over the slices.
  /// @{
  using iterator = SmallVectorImpl<MemSlice>::iterator;
  using range = iterator_range<iterator>;

  iterator begin() { return Slices.begin(); }
  iterator end() { return Slices.end(); }

  using const_iterator = SmallVectorImpl<MemSlice>::const_iterator;
  using const_range = iterator_range<const_iterator>;

  const_iterator begin() const { return Slices.begin(); }
  const_iterator end() const { return Slices.end(); }
  /// @}

  /// Erase a range of slices.
  void erase(iterator Start, iterator Stop) { Slices.erase(Start, Stop); }

  /// Insert new slices for this memory.
  ///
  /// This moves the slices into the memory's slices collection, and re-sorts
  /// everything so that the usual ordering properties of the memory's slices
  /// hold.
  void insert(ArrayRef<MemSlice> NewSlices) {
    int OldSize = Slices.size();
    Slices.append(NewSlices.begin(), NewSlices.end());
    auto SliceI = Slices.begin() + OldSize;
    std::stable_sort(SliceI, Slices.end());
    std::inplace_merge(Slices.begin(), SliceI, Slices.end());
  }

  // Forward declare the iterator and range accessor for walking the
  // partitions.
  class partition_iterator;
  iterator_range<partition_iterator> partitions();

  /// Access the dead users for this memory.
  ArrayRef<Instruction *> getDeadUsers() const { return DeadUsers; }

  /// Access the users for this memory that are llvm.protected.field.ptr
  /// intrinsics.
  ArrayRef<IntrinsicInst *> getPFPUsers() const { return PFPUsers; }

  /// Access Uses that should be dropped if the memory is promotable.
  ArrayRef<Use *> getDeadUsesIfPromotable() const {
    return DeadUseIfPromotable;
  }

  /// Access the dead operands referring to this memory.
  ///
  /// These are operands which have cannot actually be used to refer to the
  /// memory as they are outside its range and the user doesn't correct for
  /// that. These mostly consist of PHI node inputs and the like which we just
  /// need to replace with undef.
  ArrayRef<Use *> getDeadOperands() const { return DeadOperands; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &OS, const_iterator I, StringRef Indent = "  ") const;
  void printSlice(raw_ostream &OS, const_iterator I,
                  StringRef Indent = "  ") const;
  void printUse(raw_ostream &OS, const_iterator I,
                StringRef Indent = "  ") const;
  void print(raw_ostream &OS) const;
  void dump(const_iterator I) const;
  void dump() const;
#endif

private:
  template <typename DerivedT, typename RetT = void> class BuilderBase;
  class SliceBuilder;

  friend class MemSlices::SliceBuilder;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Handle to memory instruction to simplify method interfaces.
  AllocaInst &AI;
#endif

  /// The instruction responsible for this memory not having a known set
  /// of slices.
  ///
  /// When an instruction (potentially) escapes the pointer to the memory, we
  /// store a pointer to that here and abort trying to form slices of the
  /// memory. This will be null if the memory slices are analyzed successfully.
  Instruction *PointerEscapingInstr;
  Instruction *PointerEscapingInstrReadOnly;

  /// The slices of the memory.
  ///
  /// We store a vector of the slices formed by uses of the memory here. This
  /// vector is sorted by increasing begin offset, and then the unsplittable
  /// slices before the splittable ones. See the MemSlice class for more
  /// details.
  SmallVector<MemSlice, 8> Slices;

  /// Instructions which will become dead if we rewrite the memory.
  ///
  /// Note that these are not separated by slice. This is because we expect an
  /// memory to be completely rewritten or not rewritten at all. If rewritten,
  /// all these instructions can simply be removed and replaced with poison as
  /// they come from outside of the memory space.
  SmallVector<Instruction *, 8> DeadUsers;

  /// Users that are llvm.protected.field.ptr intrinsics. These will be RAUW'd
  /// to their first argument if we rewrite the memory.
  SmallVector<IntrinsicInst *, 0> PFPUsers;

  /// Uses which will become dead if can promote the memory.
  SmallVector<Use *, 8> DeadUseIfPromotable;

  /// Operands which will become dead if we rewrite the memory.
  ///
  /// These are operands that in their particular use can be replaced with
  /// poison when we rewrite the memory. These show up in out-of-bounds inputs
  /// to PHI nodes and the like. They aren't entirely dead (there might be
  /// a GEP back into the bounds using it elsewhere) and nor is the PHI, but we
  /// want to swap this particular input for poison to simplify the use lists of
  /// the memory.
  SmallVector<Use *, 8> DeadOperands;
};

/// A partition of the slices.
///
/// An ephemeral representation for a range of slices which can be viewed as
/// a partition of the memory. This range represents a span of the memory's
/// memory which cannot be split, and provides access to all of the slices
/// overlapping some part of the partition.
///
/// Objects of this type are produced by traversing the memory's slices, but
/// are only ephemeral and not persistent.
class MemPartition {
private:
  friend class MemSlices;
  friend class MemSlices::partition_iterator;

  using iterator = MemSlices::iterator;

  /// The beginning and ending offsets of the memory for this
  /// partition.
  uint64_t BeginOffset = 0, EndOffset = 0;

  /// The start and end iterators of this partition.
  iterator SI, SJ;

  /// A collection of split slice tails overlapping the partition.
  SmallVector<MemSlice *, 4> SplitTails;

  /// Raw constructor builds an empty partition starting and ending at
  /// the given iterator.
  MemPartition(iterator SI) : SI(SI), SJ(SI) {}

public:
  /// The start offset of this partition.
  ///
  /// All of the contained slices start at or after this offset.
  uint64_t beginOffset() const { return BeginOffset; }

  /// The end offset of this partition.
  ///
  /// All of the contained slices end at or before this offset.
  uint64_t endOffset() const { return EndOffset; }

  /// The size of the partition.
  ///
  /// Note that this can never be zero.
  uint64_t size() const {
    assert(BeginOffset < EndOffset && "Partitions must span some bytes!");
    return EndOffset - BeginOffset;
  }

  /// Test whether this partition contains no slices, and merely spans
  /// a region occupied by split slices.
  bool empty() const { return SI == SJ; }

  /// \name Iterate slices that start within the partition.
  /// These may be splittable or unsplittable. They have a begin offset >= the
  /// partition begin offset.
  /// @{
  // FIXME: We should probably define a "concat_iterator" helper and use that
  // to stitch together pointee_iterators over the split tails and the
  // contiguous iterators of the partition. That would give a much nicer
  // interface here. We could then additionally expose filtered iterators for
  // split, unsplit, and unsplittable splices based on the usage patterns.
  iterator begin() const { return SI; }
  iterator end() const { return SJ; }
  /// @}

  /// Get the sequence of split slice tails.
  ///
  /// These tails are of slices which start before this partition but are
  /// split and overlap into the partition. We accumulate these while forming
  /// partitions.
  ArrayRef<MemSlice *> splitSliceTails() const { return SplitTails; }

  /// Find the common type for a partition of slices.
  ///
  /// This walks the range of slices in a partition and determines if there
  /// is a common type used across all loads and stores. It also tracks the
  /// largest integer type used.
  ///
  /// Returns a pair of {CommonType, LargestIntegerType}.
  std::pair<Type *, IntegerType *> findCommonType() const;
};

/// An iterator over partitions of the memory's slices.
///
/// This iterator implements the core algorithm for partitioning the memory's
/// slices. It is a forward iterator as we don't support backtracking for
/// efficiency reasons, and re-use a single storage area to maintain the
/// current set of split slices.
///
/// It is templated on the slice iterator type to use so that it can operate
/// with either const or non-const slice iterators.
class MemSlices::partition_iterator
    : public iterator_facade_base<partition_iterator, std::forward_iterator_tag,
                                  MemPartition> {
  friend class MemSlices;

  /// Most of the state for walking the partitions is held in a class
  /// with a nice interface for examining them.
  MemPartition P;

  /// We need to keep the end of the slices to know when to stop.
  MemSlices::iterator SE;

  /// We also need to keep track of the maximum split end offset seen.
  /// FIXME: Do we really?
  uint64_t MaxSplitSliceEndOffset = 0;

  /// Sets the partition to be empty at given iterator, and sets the
  /// end iterator.
  partition_iterator(MemSlices::iterator SI, MemSlices::iterator SE)
      : P(SI), SE(SE) {
    // If not already at the end, advance our state to form the initial
    // partition.
    if (SI != SE)
      advance();
  }

  /// Advance the iterator to the next partition.
  ///
  /// Requires that the iterator not be at the end of the slices.
  void advance();

public:
  bool operator==(const partition_iterator &RHS) const {
    assert(SE == RHS.SE &&
           "End iterators don't match between compared partition iterators!");

    // The observed positions of partitions is marked by the P.SI iterator and
    // the emptiness of the split slices. The latter is only relevant when
    // P.SI == SE, as the end iterator will additionally have an empty split
    // slices list, but the prior may have the same P.SI and a tail of split
    // slices.
    if (P.SI == RHS.P.SI && P.SplitTails.empty() == RHS.P.SplitTails.empty()) {
      assert(P.SJ == RHS.P.SJ &&
             "Same set of slices formed two different sized partitions!");
      assert(P.SplitTails.size() == RHS.P.SplitTails.size() &&
             "Same slice position with differently sized non-empty split "
             "slice tails!");
      return true;
    }
    return false;
  }

  partition_iterator &operator++() {
    advance();
    return *this;
  }

  const MemPartition &operator*() const { return P; }
};

/// A forward range over the partitions of the memory's slices.
///
/// This accesses an iterator range over the partitions of the memory's
/// slices. It computes these partitions on the fly based on the overlapping
/// offsets of the slices and the ability to split them. It will visit "empty"
/// partitions to cover regions of the memory only accessed via split
/// slices.
inline iterator_range<MemSlices::partition_iterator> MemSlices::partitions() {
  return make_range(partition_iterator(begin(), end()),
                    partition_iterator(end(), end()));
}

} // end namespace llvm

#endif // LLVM_ANALYSIS_MEMSLICEANALYSIS_H
