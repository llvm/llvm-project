//===- MergerNewtypes.h - Newtypes for the `Merger` class -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: This header currently defines some typedefs to avoid confusion
// between several different things which are all represented as `unsigned`.
// Over the next few commits, these typedefs will be replaced with "newtypes"
// (i.e., data types which are zero-cost abstractions for wrapping some
// underlying type while ensuring that the compiler keeps the new type
// distinct from the old type), along with related classes for iterating
// over them, etc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_UTILS_MERGERNEWTYPES_H_
#define MLIR_DIALECT_SPARSETENSOR_UTILS_MERGERNEWTYPES_H_

#include <cassert>
#include <type_traits>

namespace mlir {
namespace sparse_tensor {

namespace detail {
/// A constant serving as the canonically invalid identifier,
/// regardless of the identifier type.
static constexpr unsigned kInvalidId = -1u;
} // namespace detail

//===----------------------------------------------------------------------===//
/// Tensor identifiers.
///
/// Semantically, tensor identifiers could be chosen to be anything;
/// but operationally, they must be chosen such that the `Merger`
/// and `GenericOpSparsifier` agree.  Therefore, the numeric values of
/// tensor identifiers are chosen to be the `BlockArgument::getArgNumber`
/// of the value passed to `Merger::buildTensorExp`, which ranges from
/// zero to `linalg::GenericOp::getNumOperands` for the op passed to
/// `GenericOpSparsifier::matchAndRewrite`.
using TensorId = unsigned;

//===----------------------------------------------------------------------===//
/// Loop identifiers.
///
/// These identifiers serve as proxies for the `$dim` argument to
/// `linalg::IndexOp`, however the numerical value of a `LoopId` should
/// not necessarily be equated with the numerical value of the corresponding
/// `$dim` argument.  The `$dim` arguments are De Bruijn indices: that
/// is, they identify the loop which binds the loop-variable by counting
/// the enclosing loops from innermost to outermost, starting from zero.
/// Whereas `LoopId` are considered to be arbitrary names for identifying
/// loops; since the `Merger` does not care about the actual ordering of
/// loops, and leaves it up to the `LoopEmitter` to specify the actual
/// loop ordering (`LoopOrd`).
///
/// TODO: Despite the above claim that `$dim` and `LoopId` need not be
/// numerically equal, some code in the `Merger` class does equate them
/// (e.g., `buildTensorExp`).  So we need to explicate the exact relationship
/// between `$dim`, `LoopId`, and `LoopOrd`; especially with regards to their
/// providence.  If `LoopId` really is supposed to be equated with `$dim`,
/// then we should change the name to `LoopIdx` or similar, to capture the
/// fact that its numerical value is not invariant when entering/exiting
/// loops (unlike `TensorId`, `ExprId`, `LatPointId`, and `LatSetId` which
/// are invariant identifiers).
using LoopId = unsigned;

//===----------------------------------------------------------------------===//
/// A compressed representation of `std::pair<TensorId, LoopId>`.
/// The compression scheme is such that this also serves as an index
/// into the bitvector stored in `LatPoint` (since that bitvector is
/// just the implementation for a set of `TensorLoopId` values).
using TensorLoopId = unsigned;

//===----------------------------------------------------------------------===//
/// `TensorExp` identifiers.  These are allocated by `Merger::addExp`,
/// and serve as unique identifiers for the corresponding `TensorExp` object.
using ExprId = unsigned;

//===----------------------------------------------------------------------===//
/// `LatPoint` identifiers.  These are allocated by `Merger::addLat`,
/// and serve as unique identifiers for the corresponding `LatPoint` object.
using LatPointId = unsigned;

//===----------------------------------------------------------------------===//
/// `LatSet` identifiers.  These are allocated by `Merger::addSet` (and
/// by other methods calling that one), and serve as unique identifiers
/// for the corresponding `SmallVector<LatPointId>` object.
using LatSetId = unsigned;

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_UTILS_MERGERNEWTYPES_H_
