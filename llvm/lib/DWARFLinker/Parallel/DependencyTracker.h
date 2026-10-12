//===- "DependencyTracker.h" ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKER_PARALLEL_DEPENDENCYTRACKER_H
#define LLVM_LIB_DWARFLINKER_PARALLEL_DEPENDENCYTRACKER_H

#include "DWARFLinkerCompileUnit.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class DWARFDebugInfoEntry;
class DWARFDie;

namespace dwarf_linker {
namespace parallel {

/// This class discovers DIEs dependencies: marks "live" DIEs, marks DIE
/// locations (whether DIE should be cloned as regular DIE or it should be put
/// into the artificial type unit).
class DependencyTracker {
public:
  DependencyTracker(CompileUnit &CU) : CU(CU) {}

  /// Recursively walk the \p DIE tree and look for DIEs to keep. Store that
  /// information in \p CU's DIEInfo.
  ///
  /// This function is the entry point of the DIE selection algorithm. It is
  /// expected to walk the DIE tree and(through the mediation of
  /// Context.File.Addresses) ask for relocation adjustment value on each
  /// DIE that might be a 'root DIE'(f.e. subprograms, variables).
  ///
  /// Returns true if all dependencies are correctly discovered. Inter-CU
  /// dependencies cannot be discovered if referenced CU is not analyzed yet.
  /// If that is the case this method returns false.
  bool resolveDependenciesAndMarkLiveness(
      bool InterCUProcessingStarted,
      std::atomic<bool> &HasNewInterconnectedCUs);

  /// Check if dependencies have incompatible placement.
  /// If that is the case modify placement to be compatible.
  /// \returns true if any placement was updated, otherwise returns false.
  /// This method should be called as a followup processing after
  /// resolveDependenciesAndMarkLiveness().
  bool updateDependenciesCompleteness();

  /// Recursively walk the \p DIE tree and check "keepness" and "placement"
  /// information. It is an error if parent node does not have "keep" flag,
  /// while child has one. It is an error if parent node has "TypeTable"
  /// placement while child has "PlainDwarf" placement. This function dump error
  /// at stderr in that case.
  void verifyKeepChain();

protected:
  enum class LiveRootWorklistActionTy : uint8_t {
    /// Mark current item as live entry.
    MarkSingleLiveEntry = 0,

    /// Mark current item as type entry.
    MarkSingleTypeEntry,

    /// Mark current item and all its children as live entry.
    MarkLiveEntryRec,

    /// Mark current item and all its children as type entry.
    MarkTypeEntryRec,

    /// Mark all children of current item as live entry.
    MarkLiveChildrenRec,

    /// Mark all children of current item as type entry.
    MarkTypeChildrenRec,
  };

  /// \returns true if the specified action is for the "PlainDwarf".
  bool isLiveAction(LiveRootWorklistActionTy Action) {
    switch (Action) {
    default:
      return false;

    case LiveRootWorklistActionTy::MarkSingleLiveEntry:
    case LiveRootWorklistActionTy::MarkLiveEntryRec:
    case LiveRootWorklistActionTy::MarkLiveChildrenRec:
      return true;
    }
  }

  /// \returns true if the specified action is for the "TypeTable".
  bool isTypeAction(LiveRootWorklistActionTy Action) {
    switch (Action) {
    default:
      return false;

    case LiveRootWorklistActionTy::MarkSingleTypeEntry:
    case LiveRootWorklistActionTy::MarkTypeEntryRec:
    case LiveRootWorklistActionTy::MarkTypeChildrenRec:
      return true;
    }
  }

  /// \returns true if the specified action affects only Root entry
  /// itself and does not affect it`s children.
  bool isSingleAction(LiveRootWorklistActionTy Action) {
    switch (Action) {
    default:
      return false;

    case LiveRootWorklistActionTy::MarkSingleLiveEntry:
    case LiveRootWorklistActionTy::MarkSingleTypeEntry:
      return true;
    }
  }

  /// \returns true if the specified action affects only Root entry
  /// itself and does not affect it`s children.
  bool isChildrenAction(LiveRootWorklistActionTy Action) {
    switch (Action) {
    default:
      return false;

    case LiveRootWorklistActionTy::MarkLiveChildrenRec:
    case LiveRootWorklistActionTy::MarkTypeChildrenRec:
      return true;
    }
  }

  /// What a tree walk does, and for a walk that only records dependencies,
  /// which root the dependencies it finds are recorded under. Only a
  /// DW_TAG_subprogram re-anchors that root, so the distinction cannot be
  /// recovered by comparing root entries: a walk of a subprogram subtree starts
  /// out anchored to the subprogram itself.
  enum class TreeWalkKindTy : uint8_t {
    /// Mark the tree as kept and schedule the roots it references.
    MarkTree,

    /// Do not mark. Record the dependencies as belonging to whichever root
    /// references the walked subtree.
    RecordSubtreeDeps,

    /// Do not mark. Record the dependencies as belonging to a subprogram inside
    /// the walked subtree, which makes them the same for every referencing
    /// root.
    RecordNestedSubprogramDeps,
  };

  /// \returns true if the specified walk records dependencies instead of
  /// marking the tree.
  static bool recordsDepsOnly(TreeWalkKindTy Kind) {
    return Kind != TreeWalkKindTy::MarkTree;
  }

  /// Class keeping live worklist item data.
  class LiveRootWorklistItemTy {
  public:
    LiveRootWorklistItemTy() = default;
    LiveRootWorklistItemTy(const LiveRootWorklistItemTy &) = default;
    LiveRootWorklistItemTy(LiveRootWorklistActionTy Action,
                           UnitEntryPairTy RootEntry) {
      RootCU.setInt(Action);
      RootCU.setPointer(RootEntry.CU);

      RootDieEntry = RootEntry.DieEntry;
    }
    LiveRootWorklistItemTy(
        LiveRootWorklistActionTy Action, UnitEntryPairTy RootEntry,
        UnitEntryPairTy ReferencedBy,
        const DWARFDebugInfoEntry *ReferencedTypeDieEntry = nullptr) {
      RootCU.setPointer(RootEntry.CU);
      RootCU.setInt(Action);
      RootDieEntry = RootEntry.DieEntry;

      ReferencedByCU = ReferencedBy.CU;
      ReferencedByDieEntry = ReferencedBy.DieEntry;

      this->ReferencedTypeDieEntry = ReferencedTypeDieEntry;
    }

    UnitEntryPairTy getRootEntry() const {
      return UnitEntryPairTy{RootCU.getPointer(), RootDieEntry};
    }

    CompileUnit::DieOutputPlacement getPlacement() const {
      return static_cast<CompileUnit::DieOutputPlacement>(RootCU.getInt());
    }

    bool hasReferencedByOtherEntry() const { return ReferencedByCU != nullptr; }

    UnitEntryPairTy getReferencedByEntry() const {
      assert(ReferencedByCU);
      assert(ReferencedByDieEntry);
      return UnitEntryPairTy{ReferencedByCU, ReferencedByDieEntry};
    }

    /// \returns the DIE actually referenced by ReferencedByDieEntry, whose
    /// placement (rather than the enclosing RootDieEntry's) determines whether
    /// ReferencedByDieEntry may remain in the type table. Null when the
    /// referenced DIE is RootDieEntry itself, in which case RootDieEntry's
    /// placement is used instead.
    const DWARFDebugInfoEntry *getReferencedTypeDieEntry() const {
      return ReferencedTypeDieEntry;
    }

    LiveRootWorklistActionTy getAction() const {
      return static_cast<LiveRootWorklistActionTy>(RootCU.getInt());
    }

  protected:
    /// Root entry.
    /// ASSUMPTION: 3 bits are used to store LiveRootWorklistActionTy value.
    /// Thus LiveRootWorklistActionTy should have no more eight elements.

    /// Pointer traits for CompileUnit.
    struct CompileUnitPointerTraits {
      static inline void *getAsVoidPointer(CompileUnit *P) { return P; }
      static inline CompileUnit *getFromVoidPointer(void *P) {
        return (CompileUnit *)P;
      }
      static constexpr int NumLowBitsAvailable = 3;
      static_assert(
          alignof(CompileUnit) >= (1 << NumLowBitsAvailable),
          "CompileUnit insufficiently aligned to have enough low bits.");
    };

    PointerIntPair<CompileUnit *, 3, LiveRootWorklistActionTy,
                   CompileUnitPointerTraits>
        RootCU;
    const DWARFDebugInfoEntry *RootDieEntry = nullptr;

    /// Another root entry which references this RootDieEntry.
    /// ReferencedByDieEntry is kept to update placement.
    /// if RootDieEntry has placement incompatible with placement
    /// of ReferencedByDieEntry then it should be updated.
    CompileUnit *ReferencedByCU = nullptr;
    const DWARFDebugInfoEntry *ReferencedByDieEntry = nullptr;

    /// The DIE actually referenced by ReferencedByDieEntry. It lives in the
    /// same CU as RootDieEntry, but its placement can differ: RootDieEntry is
    /// the enclosing root that is marked as kept, whereas this DIE may be a
    /// nested type demoted independently. That placement, not RootDieEntry's,
    /// determines whether ReferencedByDieEntry may remain in the type table.
    /// Null when RootDieEntry is the referenced DIE itself.
    const DWARFDebugInfoEntry *ReferencedTypeDieEntry = nullptr;
  };

  using RootEntriesListTy = SmallVector<LiveRootWorklistItemTy>;

  /// A completeness dependency of a subtree that belongs to whichever root
  /// references the subtree, which is not known while the subtree is walked.
  struct SubtreeDependencyTy {
    LiveRootWorklistActionTy Action;
    UnitEntryPairTy Root;
    const DWARFDebugInfoEntry *ReferencedTypeDieEntry;
  };

  using SubtreeDependenciesTy = SmallVector<SubtreeDependencyTy>;

  /// A subtree paired with the action it is walked with, which selects both the
  /// visited children and the action recorded for a reference.
  using SubtreeDependenciesKeyTy =
      std::tuple<CompileUnit *, const DWARFDebugInfoEntry *,
                 LiveRootWorklistActionTy>;

  /// A root referencing an already-marked subtree, standing in for all of that
  /// subtree's dependencies. The subtree is walked when completeness is
  /// checked, once per subtree rather than once per referencing root, which is
  /// what keeps recording linear in the number of shared subtrees.
  ///
  /// Deferring the walk out of marking also keeps it off the state marking is
  /// still mutating. A walk that only records dependencies reads a DIE's ODR
  /// availability and whether it has an address, both settled before marking
  /// begins, and never the keep and placement bits that sibling units raise as
  /// they mark. Walking during marking would consult those bits through
  /// isAlreadyMarked and yield a result that depends on how the units
  /// interleave.
  struct SubtreeDependencyRefTy {
    UnitEntryPairTy Subtree;
    LiveRootWorklistActionTy Action;
    UnitEntryPairTy ReferencedBy;
  };

  /// This function navigates DIEs tree starting from specified \p Entry.
  /// It puts found 'root DIE' into the worklist. The \p CollectLiveEntries
  /// instructs to collect either live roots(like subprograms having live
  /// DW_AT_low_pc) or otherwise roots which is not live(they need to be
  /// collected if they are imported f.e. by DW_TAG_imported_module).
  void collectRootsToKeep(const UnitEntryPairTy &Entry,
                          std::optional<UnitEntryPairTy> ReferencedBy,
                          bool IsLiveParent);

  /// Returns true if specified variable references live code section.
  static bool isLiveVariableEntry(const UnitEntryPairTy &Entry,
                                  bool IsLiveParent);

  /// Returns true if specified subprogram references live code section.
  static bool isLiveSubprogramEntry(const UnitEntryPairTy &Entry);

  /// Examine worklist and mark all 'root DIE's as kept and set "Placement"
  /// property.
  bool markCollectedLiveRootsAsKept(bool InterCUProcessingStarted,
                                    std::atomic<bool> &HasNewInterconnectedCUs);

  /// Mark whole DIE tree as kept recursively. A walk that only records
  /// dependencies (see \p Kind) does not mark the tree. Instead its
  /// completeness dependencies are collected (see maybeAddReferencedRoots) so
  /// they can be applied to every root referencing the tree.
  /// \see materializeSubtreeSummaries.
  bool markDIEEntryAsKeptRec(LiveRootWorklistActionTy Action,
                             const UnitEntryPairTy &RootEntry,
                             const UnitEntryPairTy &Entry,
                             bool InterCUProcessingStarted,
                             std::atomic<bool> &HasNewInterconnectedCUs,
                             TreeWalkKindTy Kind = TreeWalkKindTy::MarkTree);

  /// Record that \p RootEntry references the already-marked subtree \p Entry,
  /// and therefore carries the completeness dependencies of that subtree. The
  /// subtree itself is walked later, by materializeSubtreeSummaries().
  void recordSubtreeDependencies(LiveRootWorklistActionTy Action,
                                 const UnitEntryPairTy &RootEntry,
                                 const UnitEntryPairTy &Entry);

  /// Walk every subtree that a recorded reference stands for, once per subtree
  /// and action, and summarize the dependencies it contributes. Called when
  /// completeness is checked, so that liveness marking and inter-unit reference
  /// resolution have settled and the summary no longer depends on the order the
  /// units were processed in.
  void materializeSubtreeSummaries();

  /// Apply each summarized subtree's dependencies to every root recorded as
  /// referencing it.
  /// \returns true if any placement was updated.
  bool applySubtreeSummaries();

  /// Demote \p ReferencedBy to plain DWARF if it may not stay in the type table
  /// while the DIE it references through \p Root is not placed there.
  /// \returns true if the placement was updated.
  bool demoteIfIncomplete(const UnitEntryPairTy &Root,
                          const DWARFDebugInfoEntry *ReferencedTypeDieEntry,
                          const UnitEntryPairTy &ReferencedBy);

  /// Mark parents as keeping children.
  void markParentsAsKeepingChildren(const UnitEntryPairTy &Entry);

  /// Mark whole DIE tree as placed in "PlainDwarf".
  void setPlainDwarfPlacementRec(const UnitEntryPairTy &Entry);

  /// Check referenced DIEs and add them into the worklist. A walk that only
  /// records dependencies (see \p Kind) schedules nothing, so it triggers no
  /// reference-following recursion. Each dependency it finds is instead
  /// collected for the root that carries it, which is either whichever root
  /// references the walked subtree or a subprogram nested inside it. This is
  /// used when \p Entry was already marked by a racing CU/root: the marking and
  /// subtree are handled elsewhere, but the referencing root's dependencies
  /// must still be recorded so the completeness fixpoint sees a complete,
  /// order-independent dependency set.
  bool maybeAddReferencedRoots(LiveRootWorklistActionTy Action,
                               const UnitEntryPairTy &RootEntry,
                               const UnitEntryPairTy &Entry,
                               bool InterCUProcessingStarted,
                               std::atomic<bool> &HasNewInterconnectedCUs,
                               TreeWalkKindTy Kind = TreeWalkKindTy::MarkTree);

  /// \returns true if \p DIEEntry can possibly be put into the artificial type
  /// unit.
  bool isTypeTableCandidate(const DWARFDebugInfoEntry *DIEEntry);

  /// \returns root for the specified \p Entry.
  UnitEntryPairTy getRootForSpecifiedEntry(UnitEntryPairTy Entry);

  /// Add action item to the work list.
  void addActionToRootEntriesWorkList(
      LiveRootWorklistActionTy Action, const UnitEntryPairTy &Entry,
      std::optional<UnitEntryPairTy> ReferencedBy,
      const DWARFDebugInfoEntry *ReferencedTypeDieEntry = nullptr);

  CompileUnit &CU;

  /// List of entries which are 'root DIE's.
  RootEntriesListTy RootEntriesWorkList;

  /// List of entries dependencies.
  RootEntriesListTy Dependencies;

  /// Dependency summaries of already-marked subtrees, keyed by subtree and
  /// action. Filled once, when completeness is first checked.
  DenseMap<SubtreeDependenciesKeyTy, SubtreeDependenciesTy> SubtreeSummaries;

  /// Roots referencing an already-marked subtree.
  SmallVector<SubtreeDependencyRefTy> SubtreeDependencyRefs;

  /// Number of leading SubtreeDependencyRefs whose subtree is summarized.
  size_t MaterializedRefs = 0;

  /// Where the walk in progress collects the dependencies that belong to the
  /// root referencing the walked subtree, or null when no such walk is in
  /// progress. Scoped by materializeSubtreeSummaries().
  SubtreeDependenciesTy *CollectedSubtreeDeps = nullptr;

  /// Whether inter-unit references could be resolved during marking. Reused
  /// when the recorded subtrees are walked, which happens outside of marking.
  bool InterCUProcessingWasStarted = false;
};

} // end of namespace parallel
} // end of namespace dwarf_linker
} // end of namespace llvm

#endif // LLVM_LIB_DWARFLINKER_PARALLEL_DEPENDENCYTRACKER_H
