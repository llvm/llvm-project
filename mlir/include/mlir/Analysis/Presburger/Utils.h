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

} // namespace presburger_utils
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_UTILS_H
