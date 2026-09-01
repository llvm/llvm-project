//===- Transforms/IPO/PGOVerify.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the pass-instrumentation registration hook for
/// `-verify-ipgo` diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PGOVERIFY_H
#define LLVM_TRANSFORMS_IPO_PGOVERIFY_H

#include "llvm/ADT/Any.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class BasicBlock;
class BlockFrequencyInfo;
class Function;
class Loop;
class Module;
class PassInstrumentationCallbacks;

/// Registers `-verify-ipgo` diagnostics with pass instrumentation.
class IPGOVerifier {
public:
  /// Per-block frequency state used by PGOVerifier flow checks.
  struct BlockFreqInfo {
    unsigned numUnknownIn = 0;
    unsigned numUnknownOut = 0;
    uint64_t sumIn = 0;
    uint64_t sumOut = 0;
  };

  /// Frequency cache for all basic blocks in a function.
  using AllBlockFreqInfo = MapVector<const BasicBlock *, BlockFreqInfo>;

  /// Register post-pass callback hooks used by `-verify-ipgo` diagnostics.
  ///
  /// \param PIC Pass instrumentation callback registry.
  LLVM_ABI void registerCallbacks(PassInstrumentationCallbacks &PIC);

  /// Dispatch post-pass handling by IR unit type.
  ///
  /// \param PassID Name of the pass that completed.
  /// \param IR IR unit received from pass instrumentation callbacks.
  LLVM_ABI void runAfterPass(StringRef PassID, Any IR);

  /// Compute/infer block frequency state for flow-conservation checks.
  LLVM_ABI void computeBlockFrequencies(const Function *F,
                                        const BlockFrequencyInfo &BFI);

  /// Retrieve cached per-block frequency information for a function.
  ///
  /// \note The cache is keyed by function pointer and is invalidated after
  ///       pass callbacks when IR may have changed.
  ///
  /// \return A pointer to cached frequency data for \p F, or `nullptr` when
  ///         no cache entry exists.
  LLVM_ABI const AllBlockFreqInfo *
  getCachedBlockFreqInfo(const Function *F) const;

private:
  /// Invalidate cached block-frequency entries for changed IR scopes.
  void invalidateFunctionFrequencyCache(Any IR);

  /// Return true if a function is eligible for verification.
  ///
  /// Applies verifier-local exclusions and optional command-line filtering.
  bool shouldVerifyFunction(const Function *F) const;

  /// Handle module callbacks by delegating each function to function handler.
  void runAfterPass(const Module *M);

  /// Per-function callback handler.
  void runAfterPass(Function *F);

  /// Handle SCC callbacks by delegating each function to function handler.
  void runAfterPass(const LazyCallGraph::SCC *C);

  /// Handle loop callbacks by delegating to containing function handler.
  void runAfterPass(const Loop *L);

  /// Check whether function-local profile counts may overflow 32-bit ranges.
  ///
  /// This guards strict flow-conservation checks that rely on bounded profile
  /// counts derived from entry and block-level profile metadata.
  ///
  /// \return `true` if a possible overflow is detected for \p F, otherwise
  ///         `false`.
  bool hasFunctionLocalCountOverflow(const Function *F, const llvm::BlockFrequencyInfo&) const;

  /// Check whether a module carries an instrumentation-profile use summary.
  ///
  /// The verifier uses this as a signal that summary-based profile limits are
  /// available to conservatively reason about local-count overflow.
  ///
  /// \return `true` if \p M has an instrumentation profile summary, otherwise
  ///         `false`.
  bool hasInstrProfUseSummary(const Module *M) const;

  /// Validate block-level flow conservation for known incoming/outgoing sums.
  ///
  /// For basic blocks whose incoming and outgoing frequency contributions are
  /// fully known, this checks whether incoming sum equals outgoing sum.
  /// Diagnostics are emitted in debug mode for mismatches or unknown states.
  void validateBlockFrequencies(const Function *F);

  /// Validate function entry count against summed direct-caller profile counts.
  ///
  /// This check runs only when the function has an entry count and all direct
  /// callsites to the function have extractable profile totals.
  void validateEntryCountAgainstCallerSum(const Function *F);

  /// Validate instrumentation-generation phase invariants.
  ///
  /// Checks for gen-phase violations such as:
  /// - InstrProf intrinsic names matching their containing function
  /// - Counter global loads from the correct function
  void verifyGenPhase(const Function *F);

  /// Per-instance cache of inferred block-frequency data keyed by function.
  DenseMap<const Function *, AllBlockFreqInfo> FunctionBlockFreqInfoCache;
};

} // end namespace llvm
#endif // LLVM_TRANSFORMS_IPO_PGOVERIFY_H
