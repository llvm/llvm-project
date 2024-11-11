//===- CSE.h - Common Subexpression Elimination -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for eliminating common subexpressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CSE_H_
#define MLIR_TRANSFORMS_CSE_H_

#include <functional>

namespace mlir {

class DominanceInfo;
class Operation;
class RewriterBase;

/// Configuration for CSE.
struct CSEConfig {
  /// If set, matching ops act as a CSE'ing barrier: ops are not CSE'd across
  /// matching ops.
  ///
  /// Note: IsolatedFromAbove ops are always a CSE'ing barrier, regardless of
  /// this filter.
  ///
  /// Example:
  /// %0 = arith.constant 0 : index
  /// scf.for ... {
  ///   %1 = arith.constant 0 : index
  ///   ...
  /// }
  /// If "scf.for" is marked as a CSE'ing barrier, %0 and %1 are *not* CSE'd.
  std::function<bool(Operation *)> barrierOpFilter = nullptr;

  /// If set, matching ops are not eliminated (neither CSE'd nor DCE'd). All
  /// non-matching ops are subject to elimination.
  std::function<bool(Operation *)> eliminateOpFilter = nullptr;
};

/// Eliminate common subexpressions within the given operation. This transform
/// looks for and deduplicates equivalent operations.
///
/// `changed` indicates whether the IR was modified or not.
void eliminateCommonSubExpressions(RewriterBase &rewriter,
                                   DominanceInfo &domInfo, Operation *op,
                                   bool *changed = nullptr,
                                   CSEConfig config = CSEConfig());

} // namespace mlir

#endif // MLIR_TRANSFORMS_CSE_H_
