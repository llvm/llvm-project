//===- Transforms.h - XeGPU Dialect transformations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <functional>
#include <optional>
#include <utility>

namespace mlir {
class RewritePatternSet;

namespace xegpu {

/// Options to control the XeGPU unrolling. Its main purpose is to
/// provide a way to customize the native shape of the operation.
struct UnrollOptions {
  /// Callback function that indicates whether vector unrolling should be
  /// attempted on the operation.
  using FilterConstraintFnType = std::function<LogicalResult(Operation *op)>;
  FilterConstraintFnType filterConstraint = nullptr;
  UnrollOptions &setFilterConstraint(FilterConstraintFnType constraint) {
    filterConstraint = std::move(constraint);
    return *this;
  }

  /// Function that computes the target shape for unrolling. It returns an
  /// optional vector of integers representing the shape. If it returns
  /// `std::nullopt`, unrolling is aborted for the given operation.
  using NativeShapeFnType =
      std::function<std::optional<SmallVector<int64_t>>(Operation *op)>;
  NativeShapeFnType nativeShape = nullptr;
  UnrollOptions &setNativeShapeFn(NativeShapeFnType fn) {
    nativeShape = std::move(fn);
    return *this;
  }

  /// Function that converts a ShapedType (TensorDescType or VectorType)
  /// into the unrolled type based on the tileShape. It returns a vector of
  /// types representing the unrolled types for simplicity. When
  /// `returnSingleType` is true, it returns a vector containing only one single
  /// unrolled type.
  using UnrolledTypeFnType = std::function<SmallVector<Type>(
      ShapedType type, ArrayRef<int64_t> tileShape, bool returnSingleType)>;
  UnrolledTypeFnType getUnrolledTypes = nullptr;
  UnrollOptions &setUnrolledTypesFn(UnrolledTypeFnType fn) {
    getUnrolledTypes = std::move(fn);
    return *this;
  }
};

/// Appends patterns for folding aliasing ops into XeGPU ops into `patterns`.
void populateXeGPUFoldAliasOpsPatterns(RewritePatternSet &patterns);

/// Appends patterns for XeGPU SIMT distribution into `patterns`.
void populateXeGPUSubgroupDistributePatterns(RewritePatternSet &patterns);
/// Appends patterns for moving function body into gpu.warp_execute_on_lane0 op.
void populateXeGPUMoveFuncBodyToWarpOpPatterns(RewritePatternSet &patterns);
/// Appends patterns for XeGPU workgroup to subgroup distribution into
/// `patterns`.
void populateXeGPUWgToSgDistributePatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to unroll xegpu operations to a smaller shapes.
/// Users can control whether an operation to be unrolled or not, as well as
/// its target shape via `options` structure. (via setting filterConstraint
/// and nativeShape respectively, both of them are function refs taking `op` as
/// input).
/// An `op` is unrolled to the `targetShape` as follows, for each of its
/// operands:
///   1. the unrolled type `unrolledType` and number of unrolled instances
///   `numUnrolledInstances` are computed from the `targetShape`.
///   2. pack each operand. ExtractStridedSlice are created to break-up the
///   vector operands. And BuiltinUnrealizedCastop are created to break-up
///    the TensorDesc operands.
///   3. the original op is cloned `numUnrolledInstances` times, once for each
///   result.
///   4. unpack the results. InsertStridedSlice are inserted for VectorType
///   result, and BuiltinUnrealizedCastOp are inserted for TensorDescType result
///   to re-assemble the slices into the original shape.
void populateXeGPUUnrollPatterns(RewritePatternSet &patterns,
                                 const UnrollOptions &options);

} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H
