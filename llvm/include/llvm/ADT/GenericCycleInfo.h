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

  /// \brief Whether the cycle is a natural loop.
  bool isReducible() const { return Entries.size() == 1; }

  BlockT *getHeader() const { return Entries[0]; }

  const SmallVectorImpl<BlockT *> & getEntries() const {
    return Entries;
  }

  /// \brief Return whether \p Block is an entry block of the cycle.
  bool isEntry(const BlockT *Block) const {
    return is_contained(Entries, Block);
  }

  /// \brief Replace all entries with \p Block as single entry.
  /// \p Block must be contained in the cycle.
  void setSingleEntry(BlockT *Block) {
    Entries.clear();
    Entries.push_back(Block);
  }

  /// \brief Returns true iff this cycle contains \p C. O(1). Non-strict, i.e.
  /// returns true if C is the same cycle.
  bool contains(const GenericCycle *C) const {
    return C && IdxBegin <= C->IdxBegin && C->IdxEnd <= IdxEnd;
  }

  const GenericCycle *getParentCycle() const { return ParentCycle; }
  GenericCycle *getParentCycle() { return ParentCycle; }
  unsigned getDepth() const { return Depth; }

  size_t getNumBlocks() const { return IdxEnd - IdxBegin; }

  /// Iteration over child cycles: the first child (if any) immediately
  /// follows this cycle in the preorder array, and each next sibling follows
  /// the previous child's subtree.
  //@{
  struct const_child_iterator
      : iterator_facade_base<const_child_iterator, std::forward_iterator_tag,
                             GenericCycle *, std::ptrdiff_t, GenericCycle *,
                             GenericCycle *> {
    const GenericCycle *C = nullptr;

    const_child_iterator() = default;
    explicit const_child_iterator(const GenericCycle *C) : C(C) {}

    GenericCycle *operator*() const { return const_cast<GenericCycle *>(C); }
    const_child_iterator &operator++() {
      C += 1 + C->NumDescendants;
      return *this;
    }
    bool operator==(const const_child_iterator &Other) const {
      return C == Other.C;
    }
  };

  const_child_iterator child_begin() const {
    return const_child_iterator{this + 1};
  }
  const_child_iterator child_end() const {
    return const_child_iterator{this + 1 + NumDescendants};
  }
  iterator_range<const_child_iterator> children() const {
    return llvm::make_range(child_begin(), child_end());
  }
  //@}

  /// Iteration over entry blocks.
  //@{
  using const_entry_iterator =
      typename SmallVectorImpl<BlockT *>::const_iterator;
  const_entry_iterator entry_begin() const { return Entries.begin(); }
  const_entry_iterator entry_end() const { return Entries.end(); }
  size_t getNumEntries() const { return Entries.size(); }
  iterator_range<const_entry_iterator> entries() const {
    return llvm::make_range(entry_begin(), entry_end());
  }
  using const_reverse_entry_iterator =
      typename SmallVectorImpl<BlockT *>::const_reverse_iterator;
  const_reverse_entry_iterator entry_rbegin() const { return Entries.rbegin(); }
  const_reverse_entry_iterator entry_rend() const { return Entries.rend(); }
  //@}

  Printable printEntries(const ContextT &Ctx) const {
    return Printable([this, &Ctx](raw_ostream &Out) {
      ListSeparator LS(" ");
      for (auto *Entry : Entries)
        Out << LS << Ctx.print(Entry);
    });
  }
};

/// \brief Cycle information for a function.
template <typename ContextT> class GenericCycleInfo {
public:
  using BlockT = typename ContextT::BlockT;
  using CycleT = GenericCycle<ContextT>;
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

  void verifyBlockNumberEpoch(const FunctionT *Fn) const {
    assert(BlockNumberEpoch ==
               GraphTraits<const FunctionT *>::getNumberEpoch(Fn) &&
           "CycleInfo used with outdated block number epoch");
  }
  void addToBlockMap(BlockT *Block, CycleT *Cycle);

public:
  GenericCycleInfo() = default;
  GenericCycleInfo(GenericCycleInfo &&) = default;
  GenericCycleInfo &operator=(GenericCycleInfo &&) = default;

  void clear();
  void compute(FunctionT &F);
  void splitCriticalEdge(BlockT *Pred, BlockT *Succ, BlockT *New);

  const FunctionT *getFunction() const { return Context.getFunction(); }
  const ContextT &getSSAContext() const { return Context; }

  /// All cycles in forest preorder.
  MutableArrayRef<CycleT> cycles() { return {Cycles.get(), NumCycles}; }
  ArrayRef<CycleT> cycles() const { return {Cycles.get(), NumCycles}; }

  /// \brief Find the innermost cycle containing \p Block.
  ///
  /// \returns the innermost cycle containing \p Block or nullptr if
  ///          it is not contained in any cycle.
  CycleT *getCycle(const BlockT *Block) const {
    verifyBlockNumberEpoch(Block->getParent());
    unsigned Number = GraphTraits<const BlockT *>::getNumber(Block);
    return Number < BlockMap.size() ? BlockMap[Number] : nullptr;
  }

  /// \brief Return whether \p Block is contained in \p C. O(1).
  bool contains(const CycleT &C, const BlockT *Block) const {
    return C.contains(getCycle(Block));
  }

  /// \brief Return the blocks of \p C, including those of nested cycles.
  ArrayRef<BlockT *> getBlocks(const CycleT &C) const {
    return ArrayRef<BlockT *>(BlockLayout.begin() + C.IdxBegin,
                              BlockLayout.begin() + C.IdxEnd);
  }

  CycleT *getSmallestCommonCycle(CycleT *A, CycleT *B) const;
  CycleT *getSmallestCommonCycle(BlockT *A, BlockT *B) const;

  /// \brief Return the depth of the innermost cycle containing \p Block, or 0
  /// if it is not contained in any cycle.
  unsigned getCycleDepth(const BlockT *Block) const {
    CycleT *Cycle = getCycle(Block);
    return Cycle ? Cycle->getDepth() : 0;
  }

  CycleT *getTopLevelParentCycle(const BlockT *Block) const {
    CycleT *Cycle = getCycle(Block);
    while (Cycle && Cycle->ParentCycle)
      Cycle = Cycle->ParentCycle;
    return Cycle;
  }

  /// Return all of the successor blocks of \p C: the blocks outside of \p C
  /// which are branched to from within it.
  void getExitBlocks(const CycleT &C,
                     SmallVectorImpl<BlockT *> &TmpStorage) const;

  /// Return all blocks of \p C that have a successor outside of \p C.
  void getExitingBlocks(const CycleT &C,
                        SmallVectorImpl<BlockT *> &TmpStorage) const;

  /// Return the preheader block for \p C. Pre-header is well-defined for
  /// reducible cycle in docs/LoopTerminology.md as: the only one entering
  /// block and its only edge is to the entry block. Return null for
  /// irreducible cycles.
  BlockT *getCyclePreheader(const CycleT &C) const;

  /// If \p C has exactly one entry with exactly one predecessor, return it,
  /// otherwise return nullptr.
  BlockT *getCyclePredecessor(const CycleT &C) const;

  /// Verify that \p C is actually a well-formed cycle in the CFG.
  void verifyCycle(const CycleT &C) const;

  /// Verify the parent-child relations of \p C.
  ///
  /// Note that this does \em not check that \p C is really a cycle in the CFG.
  void verifyCycleNest(const CycleT &C) const;

  /// Assumes that \p Cycle is the innermost cycle containing \p Block.
  /// \p Block will be appended to \p Cycle and all of its parent cycles.
  /// \p Block will be added to BlockMap with \p Cycle and
  /// BlockMapTopLevel with \p Cycle's top level parent cycle.
  void addBlockToCycle(BlockT *Block, CycleT *Cycle);

  /// Methods for debug and self-test.
  //@{
  void verifyCycleNest(bool VerifyFull = false) const;
  void verify() const;
  void print(raw_ostream &Out) const;
  void dump() const { print(dbgs()); }
  Printable print(const CycleT *Cycle) const;
  //@}

  /// Iteration over top-level cycles.
  //@{
  using const_toplevel_iterator = typename CycleT::const_child_iterator;

  const_toplevel_iterator toplevel_begin() const {
    return const_toplevel_iterator{Cycles.get()};
  }
  const_toplevel_iterator toplevel_end() const {
    return const_toplevel_iterator{Cycles.get() + NumCycles};
  }

  iterator_range<const_toplevel_iterator> toplevel_cycles() const {
    return llvm::make_range(toplevel_begin(), toplevel_end());
  }
  //@}
};

} // namespace llvm

#endif // LLVM_ADT_GENERICCYCLEINFO_H
