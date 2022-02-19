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

/// This class represents the result of operations optimizing something subject
/// to some constraints. If the constraints were not satisfiable the, kind will
/// be Empty. If the optimum is unbounded, the kind is Unbounded, and if the
/// optimum is bounded, the kind will be Bounded and `optimum` holds the optimal
/// value.
enum class OptimumKind { Empty, Unbounded, Bounded };
template <typename T>
class MaybeOptimum {
public:
private:
  OptimumKind kind = OptimumKind::Empty;
  T optimum;

public:
  MaybeOptimum() = default;
  MaybeOptimum(OptimumKind kind) : kind(kind) {
    assert(kind != OptimumKind::Bounded &&
           "Bounded optima should be constructed by specifying the optimum!");
  }
  MaybeOptimum(const T &optimum)
      : kind(OptimumKind::Bounded), optimum(optimum) {}

  OptimumKind getKind() const { return kind; }
  bool isBounded() const { return kind == OptimumKind::Bounded; }
  bool isUnbounded() const { return kind == OptimumKind::Unbounded; }
  bool isEmpty() const { return kind == OptimumKind::Empty; }

  Optional<T> getOptimumIfBounded() const { return optimum; }
  const T &getBoundedOptimum() const {
    assert(kind == OptimumKind::Bounded &&
           "This should be called only for bounded optima");
    return optimum;
  }
  T &getBoundedOptimum() {
    assert(kind == OptimumKind::Bounded &&
           "This should be called only for bounded optima");
    return optimum;
  }
  const T &operator*() const { return getBoundedOptimum(); }
  T &operator*() { return getBoundedOptimum(); }
  const T *operator->() const { return &getBoundedOptimum(); }
  T *operator->() { return &getBoundedOptimum(); }
  bool operator==(const MaybeOptimum<T> &other) const {
    if (kind != other.kind)
      return false;
    if (kind != OptimumKind::Bounded)
      return true;
    return optimum == other.optimum;
  }

  // Given f that takes a T and returns a U, convert this `MaybeOptimum<T>` to
  // a `MaybeOptimum<U>` by applying `f` to the bounded optimum if it exists, or
  // returning a MaybeOptimum of the same kind otherwise.
  template <class Function>
  auto map(const Function &f) const & -> MaybeOptimum<decltype(f(optimum))> {
    if (kind == OptimumKind::Bounded)
      return f(optimum);
    return kind;
  }
};

/// `ReprKind` enum is used to set the constraint type in `MaybeLocalRepr`.
enum class ReprKind { Inequality, Equality, None };

/// `MaybeLocalRepr` contains the indices of the contraints that can be
/// expressed as a floordiv of an affine function. If it's an `equality`
/// contraint `equalityIdx` is set, in case of `inequality` the `lowerBoundIdx`
/// and `upperBoundIdx` is set. By default the kind attribute is set to None.
struct MaybeLocalRepr {
  ReprKind kind = ReprKind::None;
  explicit operator bool() const { return kind != ReprKind::None; }
  union {
    unsigned equalityIdx;
    struct {
      unsigned lowerBoundIdx, upperBoundIdx;
    } inequalityPair;
  } repr;
};

/// Check if the pos^th identifier can be expressed as a floordiv of an affine
/// function of other identifiers (where the divisor is a positive constant).
/// `foundRepr` contains a boolean for each identifier indicating if the
/// explicit representation for that identifier has already been computed.
/// Returns the `MaybeLocalRepr` struct which contains the indices of the
/// constraints that can be expressed as a floordiv of an affine function. If
/// the representation could be computed, `dividend` and `denominator` are set.
/// If the representation could not be computed, the kind attribute in
/// `MaybeLocalRepr` is set to None.
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
