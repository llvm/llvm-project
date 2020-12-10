//===------ Simplify.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simplify a SCoP by removing unnecessary statements and accesses.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_TRANSFORM_SIMPLIFY_H
#define POLLY_TRANSFORM_SIMPLIFY_H

#include "polly/ScopPass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class PassRegistry;
class Pass;
} // namespace llvm

namespace polly {
class SimplifyVisitor {
private:
  /// The invocation id (if there are multiple instances in the pass manager's
  /// pipeline) to determine which statistics to update.
  int CallNo;

  /// The last/current SCoP that is/has been processed.
  Scop *S;

  /// Number of statements with empty domains removed from the SCoP.
  int EmptyDomainsRemoved = 0;

  /// Number of writes that are overwritten anyway.
  int OverwritesRemoved = 0;

  /// Number of combined writes.
  int WritesCoalesced = 0;

  /// Number of redundant writes removed from this SCoP.
  int RedundantWritesRemoved = 0;

  /// Number of writes with empty access domain removed.
  int EmptyPartialAccessesRemoved = 0;

  /// Number of unused accesses removed from this SCoP.
  int DeadAccessesRemoved = 0;

  /// Number of unused instructions removed from this SCoP.
  int DeadInstructionsRemoved = 0;

  /// Number of unnecessary statements removed from the SCoP.
  int StmtsRemoved = 0;

  /// Return whether at least one simplification has been applied.
  bool isModified() const;

  /// Remove statements that are never executed due to their domains being
  /// empty.
  ///
  /// In contrast to Scop::simplifySCoP, this removes based on the SCoP's
  /// effective domain, i.e. including the SCoP's context as used by some other
  /// simplification methods in this pass. This is necessary because the
  /// analysis on empty domains is unreliable, e.g. remove a scalar value
  /// definition MemoryAccesses, but not its use.
  void removeEmptyDomainStmts();

  /// Remove writes that are overwritten unconditionally later in the same
  /// statement.
  ///
  /// There must be no read of the same value between the write (that is to be
  /// removed) and the overwrite.
  void removeOverwrites();

  /// Combine writes that write the same value if possible.
  ///
  /// This function is able to combine:
  /// - Partial writes with disjoint domain.
  /// - Writes that write to the same array element.
  ///
  /// In all cases, both writes must write the same values.
  void coalesceWrites();

  /// Remove writes that just write the same value already stored in the
  /// element.
  void removeRedundantWrites();

  /// Remove statements without side effects.
  void removeUnnecessaryStmts();

  /// Remove accesses that have an empty domain.
  void removeEmptyPartialAccesses();

  /// Mark all reachable instructions and access, and sweep those that are not
  /// reachable.
  void markAndSweep(LoopInfo *LI);

  /// Print simplification statistics to @p OS.
  void printStatistics(llvm::raw_ostream &OS, int Indent = 0) const;

  /// Print the current state of all MemoryAccesses to @p OS.
  void printAccesses(llvm::raw_ostream &OS, int Indent = 0) const;

public:
  explicit SimplifyVisitor(int CallNo = 0) : CallNo(CallNo) {}

  bool visit(Scop &S, LoopInfo *LI);

  void printScop(raw_ostream &OS, Scop &S) const;

  void releaseMemory();
};
} // namespace polly

namespace polly {

class MemoryAccess;
class ScopStmt;

/// Return a vector that contains MemoryAccesses in the order in
/// which they are executed.
///
/// The order is:
/// - Implicit reads (BlockGenerator::generateScalarLoads)
/// - Explicit reads and writes (BlockGenerator::generateArrayLoad,
///   BlockGenerator::generateArrayStore)
///   - In block statements, the accesses are in order in which their
///     instructions are executed.
///   - In region statements, that order of execution is not predictable at
///     compile-time.
/// - Implicit writes (BlockGenerator::generateScalarStores)
///   The order in which implicit writes are executed relative to each other is
///   undefined.
llvm::SmallVector<MemoryAccess *, 32> getAccessesInOrder(ScopStmt &Stmt);

/// Create a Simplify pass
///
/// @param CallNo Disambiguates this instance for when there are multiple
///               instances of this pass in the pass manager. It is used only to
///               keep the statistics apart and has no influence on the
///               simplification itself.
///
/// @return The Simplify pass.
llvm::Pass *createSimplifyPass(int CallNo = 0);

struct SimplifyPass : public PassInfoMixin<SimplifyPass> {
  SimplifyPass(int CallNo = 0) : Imp(CallNo) {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &AR, SPMUpdater &U);

  SimplifyVisitor Imp;
};

struct SimplifyPrinterPass : public PassInfoMixin<SimplifyPrinterPass> {
  SimplifyPrinterPass(raw_ostream &OS, int CallNo = 0) : OS(OS), Imp(CallNo) {}

  PreservedAnalyses run(Scop &S, ScopAnalysisManager &,
                        ScopStandardAnalysisResults &, SPMUpdater &);

  raw_ostream &OS;
  SimplifyVisitor Imp;
};
} // namespace polly

namespace llvm {
void initializeSimplifyLegacyPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_TRANSFORM_SIMPLIFY_H */
