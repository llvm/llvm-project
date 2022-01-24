//===- Utils.h - General utilities for Presburger library ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions required by the Presburger Library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_UTILS_H
#define MLIR_ANALYSIS_PRESBURGER_UTILS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

class IntegerPolyhedron;

namespace presburger_utils {

/// `ReprKind` enum is used to set the constraint type in `MaybeLocalRepr`.
enum class ReprKind { Inequality, Equality, None };

/// `MaybeLocalRepr` contains the indices of the contraints that can be
/// expressed as a floordiv of an affine function. If it's an `equality`
/// contraint `equalityIdx` is set, in case of `inequality` the `lowerBoundIdx`
/// and `upperBoundIdx` is set. By default the kind attribute is set to None.
struct MaybeLocalRepr {
  ReprKind kind = ReprKind::None;
  union {
    unsigned equalityIdx;
    struct {
      unsigned lowerBoundIdx, upperBoundIdx;
    } inEqualityPair;
  } repr;
};

/// Check if the pos^th identifier can be expressed as a floordiv of an affine
/// function of other identifiers (where the divisor is a positive constant).
/// `foundRepr` contains a boolean for each identifier indicating if the
/// explicit representation for that identifier has already been computed.
/// Returns the upper and lower bound inequalities using which the floordiv
/// can be computed. If the representation could be computed, `dividend` and
/// `denominator` are set. If the representation could not be computed,
/// `llvm::None` is returned.
MaybeLocalRepr computeSingleVarRepr(const IntegerPolyhedron &cst,
                                    ArrayRef<bool> foundRepr, unsigned pos,
                                    SmallVector<int64_t, 8> &dividend,
                                    unsigned &divisor);

/// Given dividends of divisions `divs` and denominators `denoms`, detects and
/// removes duplicate divisions. `localOffset` is the offset in dividend of a
/// division from where local identifiers start.
///
/// On every possible duplicate division found, `merge(i, j)`, where `i`, `j`
/// are current index of the duplicate divisions, is called and division at
/// index `j` is merged into division at index `i`. If `merge(i, j)` returns
/// `true`, the divisions are merged i.e. `j^th` division gets eliminated and
/// it's each instance is replaced by `i^th` division. If it returns `false`,
/// the divisions are not merged. `merge` can also do side effects, For example
/// it can merge the local identifiers in IntegerPolyhedron.
void removeDuplicateDivs(
    std::vector<SmallVector<int64_t, 8>> &divs,
    SmallVectorImpl<unsigned> &denoms, unsigned localOffset,
    llvm::function_ref<bool(unsigned i, unsigned j)> merge);

} // namespace presburger_utils
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_UTILS_H
