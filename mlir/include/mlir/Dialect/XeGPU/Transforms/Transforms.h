//===- Transforms.h - XeGPU Dialect transformations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H

namespace mlir {
class RewritePatternSet;


namespace xegpu {
struct UnrollOptions {
  using FilterConstraintFnType = std::function<LogicalResult(Operation *op)>;
  /// Callback function that indicates whether vector unrolling should be
  /// attempted on the operation.
  FilterConstraintFnType filterConstraint = nullptr;
  UnrollOptions &setFilterConstraint(FilterConstraintFnType constraint) {
    filterConstraint = std::move(constraint);
    return *this;
  }

  using NativeShapeFnType =
      std::function<std::optional<SmallVector<int64_t>>(Operation *op)>;
  /// Function that returns the shape of the vector to unroll to for a given
  /// operation. The unrolling is aborted if the function returns
  /// `std::nullopt`.
  NativeShapeFnType nativeShape = nullptr;
  UnrollOptions &setNativeShapeFn(NativeShapeFnType fn) {
    nativeShape = std::move(fn);
    return *this;
  }
};


/// Appends patterns for folding aliasing ops into XeGPU ops into `patterns`.
void populateXeGPUFoldAliasOpsPatterns(RewritePatternSet &patterns);

void populateXeGPUUnrollPatterns(RewritePatternSet &patterns,
                                 const UnrollOptions &options);

} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_TRANSFORMS_H
