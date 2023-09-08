//===- "DependencyTracker.h" ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DEPENDENCYTRACKER_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DEPENDENCYTRACKER_H

#include "DWARFLinkerCompileUnit.h"
#include "DWARFLinkerImpl.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class DWARFDebugInfoEntry;
class DWARFDie;

namespace dwarflinker_parallel {

/// This class discovers DIEs dependencies and marks "live" DIEs.
class DependencyTracker {
public:
  DependencyTracker(DWARFLinkerImpl::LinkContext &Context) : Context(Context) {}

  /// Recursively walk the \p DIE tree and look for DIEs to keep. Store that
  /// information in \p CU's DIEInfo.
  ///
  /// This function is the entry point of the DIE selection algorithm. It is
  /// expected to walk the DIE tree and(through the mediation of
  /// Context.File.Addresses) ask for relocation adjustment value on each
  /// DIE that might be a 'root DIE'.
  ///
  /// Returns true if all dependencies are correctly discovered. Inter-CU
  /// dependencies cannot be discovered if referenced CU is not analyzed yet.
  /// If that is the case this method returns false.
  bool resolveDependenciesAndMarkLiveness(CompileUnit &CU);

  /// Recursively walk the \p DIE tree and check "keepness" information.
  /// It is an error if parent node does not have "keep" flag, while
  /// child have one. This function dump error at stderr in that case.
#ifndef NDEBUG
  static void verifyKeepChain(CompileUnit &CU);
#endif

protected:
  struct RootEntryTy {
    RootEntryTy(CompileUnit &CU, const DWARFDebugInfoEntry *RootEntry)
        : CU(CU), RootEntry(RootEntry) {}

    // Compile unit keeping root entry.
    CompileUnit &CU;

    // Root entry.
    const DWARFDebugInfoEntry *RootEntry;
  };

  using RootEntriesListTy = SmallVector<RootEntryTy>;

  /// This function navigates DIEs tree starting from specified \p Entry.
  /// It puts 'root DIE' into the worklist.
  void collectRootsToKeep(CompileUnit &CU, const DWARFDebugInfoEntry *Entry);

  /// Returns true if specified variable references live code section.
  bool isLiveVariableEntry(CompileUnit &CU, const DWARFDebugInfoEntry *Entry);

  /// Returns true if specified subprogram references live code section.
  bool isLiveSubprogramEntry(CompileUnit &CU, const DWARFDebugInfoEntry *Entry);

  /// Examine worklist and mark all 'root DIE's as kept.
  bool markLiveRootsAsKept();

  /// Mark whole DIE tree as kept recursively.
  bool markDIEEntryAsKeptRec(const RootEntryTy &RootItem, CompileUnit &CU,
                             const DWARFDebugInfoEntry *Entry);

  /// Check referenced DIEs and add them into the worklist if neccessary.
  bool maybeAddReferencedRoots(const RootEntryTy &RootItem, CompileUnit &CU,
                               const DWARFDebugInfoEntry *Entry);

  /// Add 'root DIE' into the worklist.
  void addItemToWorklist(CompileUnit &CU, const DWARFDebugInfoEntry *Entry);

  /// Set kind of placement(whether it goes into type table, plain dwarf or
  /// both) for the specified die \p DieIdx.
  void setDIEPlacementAndTypename(CompileUnit::DIEInfo &Info);

  /// Flag indicating whether liveness information should be examined.
  bool TrackLiveness = false;

  /// List of CU, Entry pairs which are 'root DIE's.
  RootEntriesListTy RootEntriesWorkList;

  /// Link context for the analyzed CU.
  DWARFLinkerImpl::LinkContext &Context;
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DEPENDENCYTRACKER_H
