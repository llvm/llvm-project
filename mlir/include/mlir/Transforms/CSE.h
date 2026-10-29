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

#include <cstdint>

namespace mlir {

class DominanceInfo;
class Operation;
class Region;
class RewriterBase;

/// Eliminate common subexpressions within the given operation. This transform
/// looks for and deduplicates equivalent operations.
///
/// `changed` indicates whether the IR was modified or not. `numCSE` and
/// `numDCE` receive counts of operations deduplicated and dead operations
/// erased, respectively.
void eliminateCommonSubExpressions(RewriterBase &rewriter,
                                   DominanceInfo &domInfo, Operation *op,
                                   bool *changed = nullptr,
                                   int64_t *numCSE = nullptr,
                                   int64_t *numDCE = nullptr);

/// Eliminate common subexpressions within the given region.
///
/// `changed` indicates whether the IR was modified or not. Statistics are not
/// reported through this overload; use the `Operation *` overload when CSE /
/// DCE counts are needed.
void eliminateCommonSubExpressions(RewriterBase &rewriter,
                                   DominanceInfo &domInfo, Region &region,
                                   bool *changed = nullptr);

} // namespace mlir

#endif // MLIR_TRANSFORMS_CSE_H_
