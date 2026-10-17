//===- OpFoldResult.h - Results of an operation fold ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines OpFoldResult, the result of a fold for one result of an
// operation, and OpFoldResults and NormalizedOpFoldResults, the result of a
// fold of any operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPFOLDRESULT_H
#define MLIR_IR_OPFOLDRESULT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace mlir {
class Operation;

/// This class represents a single result from folding an operation.
class OpFoldResult : public PointerUnion<Attribute, Value> {
  using PointerUnion<Attribute, Value>::PointerUnion;

public:
  LLVM_DUMP_METHOD void dump() const;

  MLIRContext *getContext() const {
    PointerUnion pu = *this;
    return isa<Attribute>(pu) ? cast<Attribute>(pu).getContext()
                              : cast<Value>(pu).getContext();
  }
};

// Temporarily exit the MLIR namespace to add casting support as later code in
// this uses it. The CastInfo must come after the OpFoldResult definition and
// before any cast function calls depending on CastInfo.
} // namespace mlir

namespace llvm {
// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::OpFoldResult>
    : public CastInfo<To, mlir::OpFoldResult::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::OpFoldResult>
    : public CastInfo<To, const mlir::OpFoldResult::PointerUnion> {};
} // namespace llvm

namespace mlir {
/// Allow printing to a stream.
raw_ostream &operator<<(raw_ostream &os, OpFoldResult ofr);

namespace detail {
/// Storage shared by OpFoldResults and NormalizedOpFoldResults.
class OpFoldResultsBase {
protected:
  OpFoldResultsBase() = default;
  OpFoldResultsBase(const OpFoldResultsBase &) = default;
  OpFoldResultsBase(OpFoldResultsBase &&) = default;
  OpFoldResultsBase &operator=(const OpFoldResultsBase &) = default;
  OpFoldResultsBase &operator=(OpFoldResultsBase &&) = default;
  ~OpFoldResultsBase() = default;

  /// The replacements of the results of the op.
  SmallVector<OpFoldResult, 2> replacements;
  /// Whether the fold changed the op in place.
  bool inPlace = false;
};
} // namespace detail

/// The result of a fold of any op, as the fold returns it. Replacement i is an
/// Attribute (replace result i with a constant), a Value (replace result i with
/// that value), or null or op->getResult(i) (keep result i). A separate bit
/// records an in-place change of the op. The fold dispatch turns it into a
/// NormalizedOpFoldResults before a driver reads it.
class [[nodiscard]] OpFoldResults : public detail::OpFoldResultsBase {
public:
  /// success(): the op changed in place. failure(): the fold did not apply.
  OpFoldResults(LogicalResult status);
  /// One replacement per range element. An empty range is a failure.
  template <typename RangeT,
            typename = std::enable_if_t<
                !std::is_convertible_v<RangeT, Attribute> &&
                !std::is_convertible_v<RangeT, Value> &&
                std::is_convertible_v<llvm::detail::ValueOfRange<RangeT>,
                                      OpFoldResult>>>
  OpFoldResults(RangeT &&range) {
    llvm::append_range(replacements, range);
  }
};

/// The result of a fold as drivers read it: no replacement, or one replacement
/// per result of the op, where null keeps the result (i = op->getResult(i) is
/// not a valid normalized replacement). Only the fold dispatch builds it.
class [[nodiscard]] NormalizedOpFoldResults : public detail::OpFoldResultsBase {
public:
  /// Failure: the fold did not apply and the IR is unchanged.
  NormalizedOpFoldResults() = default;
  /// Normalize `results`, a fold result of `op`. A replacement that is the
  /// op's own result keeps that result, and no replacement remains if the fold
  /// keeps every result.
  NormalizedOpFoldResults(Operation *op, OpFoldResults &&results);

  /// Set whether the fold changed the op in place (operands, attributes,
  /// properties, or regions).
  void setModifiedInPlace(bool modified = true);

  /// Return true if the fold changed the op in place.
  bool modifiedInPlace() const;
  /// Return true if the fold replaces at least one result.
  bool replacesAny() const;
  /// Return true if the op has at least one result and the fold replaces all
  /// of them.
  bool replacesAll() const;
  /// Return the replacements: none, or one per result of the op, where null
  /// keeps the result. The returned range points into this object.
  ArrayRef<OpFoldResult> getReplacements() const LLVM_LIFETIME_BOUND;
};

/// Return true if `results` changed the op in place or replaces a result.
inline bool succeeded(const NormalizedOpFoldResults &results) {
  return results.modifiedInPlace() || results.replacesAny();
}

/// Return true if `results` is a failure.
inline bool failed(const NormalizedOpFoldResults &results) {
  return !succeeded(results);
}
} // namespace mlir

#endif // MLIR_IR_OPFOLDRESULT_H
