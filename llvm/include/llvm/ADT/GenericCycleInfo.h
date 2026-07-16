//===- GenericCycleInfo.h - Info for Cycles in any IR ------*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Find all cycles in a control-flow graph, including irreducible loops.
///
/// See docs/CycleTerminology.md for a formal definition of cycles.
///
/// Briefly:
/// - A cycle is a generalization of a loop which can represent
///   irreducible control flow.
/// - Cycles identified in a program are implementation defined,
///   depending on the DFS traversal chosen.
/// - Cycles are well-nested, and form a forest with a parent-child
///   relationship.
/// - In any choice of DFS, every natural loop L is represented by a
///   unique cycle C which is a superset of L.
/// - In the absence of irreducible control flow, the cycles are
///   exactly the natural loops in the program.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_GENERICCYCLEINFO_H
#define LLVM_ADT_GENERICCYCLEINFO_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GenericSSAContext.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {

template <typename ContextT> class GenericCycleInfo;
template <typename ContextT> class GenericCycleInfoCompute;

/// A possibly irreducible generalization of a \ref Loop.
///
/// Cycles are stored by value in GenericCycleInfo::Cycles in preorder of the
/// cycle forest: a cycle is immediately followed by its descendants. Child
/// iteration is therefore pointer arithmetic over that array.
template <typename ContextT> class GenericCycle {
public:
  using BlockT = typename ContextT::BlockT;
  using FunctionT = typename ContextT::FunctionT;
  template <typename> friend class GenericCycleInfo;
  template <typename> friend class GenericCycleInfoCompute;

private:
  /// The parent cycle. Is null for top-level cycles.
  GenericCycle *ParentCycle = nullptr;

  /// The entry block(s) of the cycle. The header is the only entry if
  /// this is a loop.
  SmallVector<BlockT *, 1> Entries;

  /// This cycle's blocks (its own and its nested cycles') occupy the half-open
  /// range [IdxBegin, IdxEnd) of GenericCycleInfo::BlockLayout. The
  /// ranges are nested like an Euler tour of the cycle tree, so containment is
  /// an interval test (see contains()).
  ///
  /// During construction (before the forest is flattened), IdxEnd accumulates
  /// the number of this cycle's own blocks (those whose innermost cycle is
  /// this one).
  unsigned IdxBegin = 0, IdxEnd = 0;

  /// Depth of the cycle in the tree: top-level cycles are at depth 1 and each
  /// nested cycle is one deeper (getCycleDepth() returns 0 for blocks outside
  /// any cycle). Sibling cycles share a depth.
  unsigned Depth = 0;

  /// Number of cycles nested inside this one: the subtree occupies
  /// [this, this + 1 + NumDescendants) of GenericCycleInfo::Cycles.
  unsigned NumDescendants = 0;

  void appendEntry(BlockT *Block) { Entries.push_back(Block); }

  GenericCycle(const GenericCycle &) = delete;
  GenericCycle &operator=(const GenericCycle &) = delete;
  GenericCycle(GenericCycle &&Rhs) = delete;
  GenericCycle &operator=(GenericCycle &&Rhs) = delete;

public:
  GenericCycle() = default;
};

/// Opaque handle to a cycle within a GenericCycleInfo. Wraps the cycle's
/// preorder index; a default-constructed handle is invalid ("no cycle"). All
/// queries live on GenericCycleInfo, which resolves the handle to storage.
///
/// Handles remain valid as long as the cycle forest is not recomputed.
/// addBlockToCycle() adds a block but never adds, removes, or reorders cycles,
/// so it leaves every handle valid.
template <typename ContextT> class GenericCycleRef {
  static constexpr unsigned InvalidIndex = ~0u;
  unsigned Index = InvalidIndex;

  explicit GenericCycleRef(unsigned Index) : Index(Index) {}
  friend class GenericCycleInfo<ContextT>;
  friend struct DenseMapInfo<GenericCycleRef<ContextT>>;

public:
  GenericCycleRef() = default;
  bool isValid() const { return Index != InvalidIndex; }
  explicit operator bool() const { return isValid(); }
  bool operator==(GenericCycleRef O) const { return Index == O.Index; }
  bool operator!=(GenericCycleRef O) const { return Index != O.Index; }
};

/// The empty/tombstone keys are distinct from the invalid handle, so an invalid
/// handle stays a legal key -- matching the pointer world where nullptr is a
/// legal DenseMap/SmallPtrSet key.
template <typename ContextT> struct DenseMapInfo<GenericCycleRef<ContextT>> {
  using T = GenericCycleRef<ContextT>;
  static T getEmptyKey() { return T(~0u - 1); }
  static T getTombstoneKey() { return T(~0u - 2); }
  static unsigned getHashValue(T C) { return C.Index; }
  static bool isEqual(T A, T B) { return A.Index == B.Index; }
};

/// \brief Cycle information for a function.
template <typename ContextT> class GenericCycleInfo {
public:
  using BlockT = typename ContextT::BlockT;
  /// The internal, by-value storage type for a cycle.
  using CycleT = GenericCycle<ContextT>;
  /// The opaque handle by which consumers refer to a cycle.
  using Cycle = GenericCycleRef<ContextT>;
  using FunctionT = typename ContextT::FunctionT;
  template <typename> friend class GenericCycleInfoCompute;

private:
  ContextT Context;
  unsigned BlockNumberEpoch;

  /// Map basic block numbers to their inner-most containing cycle.
  SmallVector<CycleT *> BlockMap;

  /// Euler tour of the cycle forest: every cycle's blocks form a contiguous
  /// slice [IdxBegin, IdxEnd) of this array, nested inside its parent's.
  SmallVector<BlockT *, 8> BlockLayout;

  /// All cycles in forest preorder: every cycle is immediately followed by
  /// its descendants, and skipping a top-level cycle's subtree lands on the
  /// next top-level cycle.
  std::unique_ptr<CycleT[]> Cycles;
  unsigned NumCycles = 0;

  /// getExitBlocks caches, indexed by the cycle's preorder index. Empty until
  /// the first query, then sized to NumCycles.
  mutable SmallVector<SmallVector<BlockT *, 0>, 0> ExitBlocksCaches;

  /// The preorder index of \p C, i.e. its offset in the Cycles array.
  unsigned getCycleIndex(const CycleT &C) const { return &C - Cycles.get(); }

  /// Resolve a handle to its stored cycle. The assert catches deref of an
  /// invalid handle and (partially) of a handle from another CycleInfo.
  CycleT &deref(Cycle C) {
    assert(C.Index < NumCycles);
    return Cycles[C.Index];
  }
  const CycleT &deref(Cycle C) const {
    assert(C.Index < NumCycles);
    return Cycles[C.Index];
  }
  /// The handle for a stored cycle.
  Cycle ref(const CycleT &C) const { return Cycle(getCycleIndex(C)); }

  /// The innermost cycle containing \p Block as a raw pointer, or null. Used
  /// internally and during construction, where handles are not yet meaningful
  /// because the flat array does not exist.
  CycleT *getCyclePtr(const BlockT *Block) const {
    verifyBlockNumberEpoch(Block->getParent());
    unsigned Number = GraphTraits<const BlockT *>::getNumber(Block);
    return Number < BlockMap.size() ? BlockMap[Number] : nullptr;
  }

  void verifyBlockNumberEpoch(const FunctionT *Fn) const {
    assert(BlockNumberEpoch ==
               GraphTraits<const FunctionT *>::getNumberEpoch(Fn) &&
           "CycleInfo used with outdated block number epoch");
  }
  void addToBlockMap(BlockT *Block, CycleT *Cycle);

public:
  /// Iteration over child cycles, yielding handles. The first child (if any)
  /// immediately follows this cycle in the preorder array, and each next
  /// sibling follows the previous child's subtree.
  struct const_child_iterator
      : iterator_facade_base<const_child_iterator, std::forward_iterator_tag,
                             Cycle, std::ptrdiff_t, Cycle, Cycle> {
    const GenericCycleInfo *CI = nullptr;
    unsigned Index = 0;

    const_child_iterator() = default;
    const_child_iterator(const GenericCycleInfo &CI, unsigned Index)
        : CI(&CI), Index(Index) {}

    Cycle operator*() const { return Cycle(Index); }
    const_child_iterator &operator++() {
      Index += 1 + CI->Cycles[Index].NumDescendants;
      return *this;
    }
    bool operator==(const const_child_iterator &Other) const {
      return Index == Other.Index;
    }
  };

  /// Sequential iteration over all cycles in forest preorder, yielding handles.
  struct const_cycle_iterator
      : iterator_facade_base<const_cycle_iterator, std::forward_iterator_tag,
                             Cycle, std::ptrdiff_t, Cycle, Cycle> {
    unsigned Index = 0;

    const_cycle_iterator() = default;
    explicit const_cycle_iterator(unsigned Index) : Index(Index) {}

    Cycle operator*() const { return Cycle(Index); }
    const_cycle_iterator &operator++() {
      ++Index;
      return *this;
    }
    bool operator==(const const_cycle_iterator &Other) const {
      return Index == Other.Index;
    }
  };

  GenericCycleInfo() = default;
  GenericCycleInfo(GenericCycleInfo &&) = default;
  GenericCycleInfo &operator=(GenericCycleInfo &&) = default;

  void clear();
  void compute(FunctionT &F);
  void splitCriticalEdge(BlockT *Pred, BlockT *Succ, BlockT *New);

  const FunctionT *getFunction() const { return Context.getFunction(); }
  const ContextT &getSSAContext() const { return Context; }

  /// All cycles in forest preorder.
  iterator_range<const_cycle_iterator> cycles() const {
    return llvm::make_range(const_cycle_iterator(0),
                            const_cycle_iterator(NumCycles));
  }

  /// \brief Find the innermost cycle containing \p Block.
  ///
  /// \returns the innermost cycle containing \p Block or an invalid handle if
  ///          it is not contained in any cycle.
  Cycle getCycle(const BlockT *Block) const {
    CycleT *C = getCyclePtr(Block);
    return C ? ref(*C) : Cycle();
  }

  BlockT *getHeader(Cycle C) const { return deref(C).Entries[0]; }
  bool isReducible(Cycle C) const { return deref(C).Entries.size() == 1; }
  Cycle getParentCycle(Cycle C) const {
    CycleT *P = deref(C).ParentCycle;
    return P ? ref(*P) : Cycle();
  }
  unsigned getDepth(Cycle C) const { return deref(C).Depth; }
  size_t getNumBlocks(Cycle C) const {
    const CycleT &Cyc = deref(C);
    return Cyc.IdxEnd - Cyc.IdxBegin;
  }

  ArrayRef<BlockT *> getEntries(Cycle C) const { return deref(C).Entries; }
  bool isEntry(Cycle C, const BlockT *Block) const {
    return is_contained(deref(C).Entries, Block);
  }
  void setSingleEntry(Cycle C, BlockT *Block) {
    auto &Entries = deref(C).Entries;
    Entries.clear();
    Entries.push_back(Block);
  }
  /// Returns true iff \p Outer contains \p Inner. O(1). Non-strict.
  bool contains(Cycle Outer, Cycle Inner) const {
    const CycleT &O = deref(Outer);
    const CycleT &I = deref(Inner);
    return O.IdxBegin <= I.IdxBegin && I.IdxEnd <= O.IdxEnd;
  }
  iterator_range<const_child_iterator> children(Cycle C) const {
    unsigned First = C.Index + 1;
    return llvm::make_range(
        const_child_iterator(*this, First),
        const_child_iterator(*this, First + deref(C).NumDescendants));
  }
  Printable printEntries(Cycle C, const ContextT &Ctx) const {
    return Printable([this, C, &Ctx](raw_ostream &Out) {
      ListSeparator LS(" ");
      for (auto *Entry : deref(C).Entries)
        Out << LS << Ctx.print(Entry);
    });
  }

  /// \brief Return whether \p Block is contained in \p C. O(1).
  bool contains(Cycle C, const BlockT *Block) const {
    Cycle Inner = getCycle(Block);
    return Inner.isValid() && contains(C, Inner);
  }

  /// \brief Return the blocks of \p C, including those of nested cycles.
  ArrayRef<BlockT *> getBlocks(Cycle C) const {
    const CycleT &Cyc = deref(C);
    return ArrayRef<BlockT *>(BlockLayout.begin() + Cyc.IdxBegin,
                              BlockLayout.begin() + Cyc.IdxEnd);
  }

  Cycle getSmallestCommonCycle(Cycle A, Cycle B) const;
  Cycle getSmallestCommonCycle(BlockT *A, BlockT *B) const;

  /// \brief Return the depth of the innermost cycle containing \p Block, or 0
  /// if it is not contained in any cycle.
  unsigned getCycleDepth(const BlockT *Block) const {
    Cycle C = getCycle(Block);
    return C.isValid() ? getDepth(C) : 0;
  }

  Cycle getTopLevelParentCycle(const BlockT *Block) const {
    CycleT *C = getCyclePtr(Block);
    if (!C)
      return Cycle();
    while (C->ParentCycle)
      C = C->ParentCycle;
    return ref(*C);
  }

  /// Return all of the successor blocks of \p C: the blocks outside of \p C
  /// which are branched to from within it.
  void getExitBlocks(Cycle C, SmallVectorImpl<BlockT *> &TmpStorage) const;

  /// Return all blocks of \p C that have a successor outside of \p C.
  void getExitingBlocks(Cycle C, SmallVectorImpl<BlockT *> &TmpStorage) const;

  /// Return the preheader block for \p C. Pre-header is well-defined for
  /// reducible cycle in docs/LoopTerminology.md as: the only one entering
  /// block and its only edge is to the entry block. Return null for
  /// irreducible cycles.
  BlockT *getCyclePreheader(Cycle C) const;

  /// If \p C has exactly one entry with exactly one predecessor, return it,
  /// otherwise return nullptr.
  BlockT *getCyclePredecessor(Cycle C) const;

  /// Verify that \p C is actually a well-formed cycle in the CFG.
  void verifyCycle(Cycle C) const;

  /// Verify the parent-child relations of \p C.
  ///
  /// Note that this does \em not check that \p C is really a cycle in the CFG.
  void verifyCycleNest(Cycle C) const;

  /// Assumes that \p C is the innermost cycle containing \p Block.
  /// \p Block will be appended to \p C and all of its parent cycles.
  /// \p Block will be added to BlockMap with \p C.
  void addBlockToCycle(BlockT *Block, Cycle C);

  /// Methods for debug and self-test.
  //@{
  void verifyCycleNest(bool VerifyFull = false) const;
  void verify() const;
  void print(raw_ostream &Out) const;
  void dump() const { print(dbgs()); }
  Printable print(Cycle C) const;
  //@}

  /// Iteration over top-level cycles.
  //@{
  using const_toplevel_iterator = const_child_iterator;

  const_toplevel_iterator toplevel_begin() const {
    return const_toplevel_iterator(*this, 0);
  }
  const_toplevel_iterator toplevel_end() const {
    return const_toplevel_iterator(*this, NumCycles);
  }

  iterator_range<const_toplevel_iterator> toplevel_cycles() const {
    return llvm::make_range(toplevel_begin(), toplevel_end());
  }
  //@}
};

} // namespace llvm

#endif // LLVM_ADT_GENERICCYCLEINFO_H
