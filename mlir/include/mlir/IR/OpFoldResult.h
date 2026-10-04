//===- OpFoldResult.h - Results of an operation fold ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines OpFoldResult, the result of a fold of an operation with
// exactly one result, and OpFoldResults, the result of a fold of any
// operation.
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

/// The result of a fold of any op. An op with exactly one fixed result defines
/// its folder with OpFoldResult; any other op can return OpFoldResults
/// directly. Replacement i is an Attribute (replace result i with a
/// constant), a Value (replace result i with that value), or null /
/// op->getResult(i) (keep result i). A separate bit records an in-place change
/// of the op.
class [[nodiscard]] OpFoldResults {
public:
  /// Failure: the fold did not apply and the IR is unchanged.
  OpFoldResults() = default;
  /// Failure, the same as the default constructor.
  OpFoldResults(std::nullptr_t);
  /// success(): the op changed in place. failure(): failure.
  OpFoldResults(LogicalResult status);
  /// One replacement. Valid only when the op has exactly one result at run
  /// time. A null replacement, or the op's own result, keeps the result, so the
  /// fold fails. Unlike `OpFoldResult fold`, the op's own result does not mean
  /// in place; use success() or setModifiedInPlace() for an in-place change.
  OpFoldResults(OpFoldResult replacement);
  OpFoldResults(Value replacement);
  OpFoldResults(Attribute replacement);
  /// One replacement per result.
  OpFoldResults(std::initializer_list<OpFoldResult> replacements);
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
  /// Incremental form: every result is kept and the op is not changed in
  /// place.
  explicit OpFoldResults(Operation *op);

  /// Set the replacement of `result`, which must be a result of the op. If
  /// another op owns `result`, or if normalize() already ran, the behavior is
  /// undefined. The object must already hold one replacement per result, for
  /// example from OpFoldResults(op). A null replacement, or
  /// `replacement == result`, keeps the result. The last write wins.
  void replace(Value result, OpFoldResult replacement);
  /// Same as above, with the index of the result. normalize() maps a
  /// replacement that is the op's own result to "keep".
  void replace(unsigned resultIndex, OpFoldResult replacement);
  /// Set whether the fold changed the op in place (operands, attributes,
  /// properties, or regions).
  void setModifiedInPlace(bool modified = true);

  // Queries for drivers. They are valid after normalize().

  /// Return true if the fold changed the op in place or replaces a result.
  bool succeeded() const;
  /// Return true if the fold did not apply.
  bool failed() const;
  /// Return true if the fold changed the op in place.
  bool modifiedInPlace() const;
  /// Return true if the fold replaces at least one result.
  bool replacesAny() const;
  /// Return true if the operation has at least one result and all the results
  /// are replaced.
  bool replacesAll() const;
  /// Return the number of replacements: 0 if no replacement, or one per result
  /// of the op.
  unsigned size() const;
  /// Return the replacement of result `i`; null means keep. If size() == 0,
  /// every index reads as keep; otherwise `i` must be less than size().
  OpFoldResult operator[](unsigned i) const;
  /// Return the replacements. The returned range points into this object.
  ArrayRef<OpFoldResult> getReplacements() const LLVM_LIFETIME_BOUND;

  /// If replacement i matches op's result i, set replacement i to null. Clear
  /// replacements if all end up null. Idempotent.
  void normalize(Operation *op);

private:
  SmallVector<OpFoldResult, 2> replacements;
  bool inPlace = false;
};

/// Return true if `result` changed the op in place or replaces a result.
inline bool succeeded(const OpFoldResults &result) {
  return result.succeeded();
}

/// Return true if `result` is a failure.
inline bool failed(const OpFoldResults &result) { return result.failed(); }

namespace detail {
/// Convert the result of a legacy vector fold with the strict legacy contract:
/// failure stays failure, success with an empty vector means "in place", and
/// success with a full vector has one replacement per result. The result is
/// then normalized.
OpFoldResults convertLegacyFoldResults(LogicalResult status,
                                       ArrayRef<OpFoldResult> results);

/// Drop the replacements of the normalized `result` if a replacement names
/// another result of `op` that `result` also replaces, because the outcome
/// would depend on the order in which a driver replaces the results. A
/// forwarding fold can return such a result where an op can use its own
/// results: in a graph region or in an unreachable block. The in-place bit
/// stays.
void dropReplacementsOfReplacedResults(Operation *op, OpFoldResults &result);
} // namespace detail

} // namespace mlir

#endif // MLIR_IR_OPFOLDRESULT_H
