//===- Passes.h - Sparse tensor pass entry points ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Defines a parallelization strategy. Any independent loop is a candidate
/// for parallelization. The loop is made parallel if (1) allowed by the
/// strategy (e.g., AnyStorageOuterLoop considers either a dense or sparse
/// outermost loop only), and (2) the generated code is an actual for-loop
/// (and not a co-iterating while-loop).
enum class SparseParallelizationStrategy {
  kNone,
  kDenseOuterLoop,
  kAnyStorageOuterLoop,
  kDenseAnyLoop,
  kAnyStorageAnyLoop
  // TODO: support reduction parallelization too?
};

/// Defines a vectorization strategy. Any inner loop is a candidate (full SIMD
/// for parallel loops and horizontal SIMD for reduction loops). A loop is
/// actually vectorized if (1) allowed by the strategy, and (2) the emitted
/// code is an actual for-loop (and not a co-iterating while-loop).
enum class SparseVectorizationStrategy {
  kNone,
  kDenseInnerLoop,
  kAnyStorageInnerLoop
};

/// Defines a type for "pointer" and "index" storage in the sparse storage
/// scheme, with a choice between the native platform-dependent index width
/// or any of 64-/32-/16-/8-bit integers. A narrow width obviously reduces
/// the memory footprint of the sparse storage scheme, but the width should
/// suffice to define the total required range (viz. the maximum number of
/// stored entries per indirection level for the "pointers" and the maximum
/// value of each tensor index over all dimensions for the "indices").
enum class SparseIntType { kNative, kI64, kI32, kI16, kI8 };

/// Sparsification options.
struct SparsificationOptions {
  SparsificationOptions(SparseParallelizationStrategy p,
                        SparseVectorizationStrategy v, unsigned vl,
                        SparseIntType pt, SparseIntType it, bool fo)
      : parallelizationStrategy(p), vectorizationStrategy(v), vectorLength(vl),
        ptrType(pt), indType(it), fastOutput(fo) {}
  SparsificationOptions()
      : SparsificationOptions(SparseParallelizationStrategy::kNone,
                              SparseVectorizationStrategy::kNone, 1u,
                              SparseIntType::kNative, SparseIntType::kNative,
                              false) {}
  SparseParallelizationStrategy parallelizationStrategy;
  SparseVectorizationStrategy vectorizationStrategy;
  unsigned vectorLength;
  SparseIntType ptrType;
  SparseIntType indType;
  bool fastOutput; // experimental: fast output buffers
};

/// Sets up sparsification rewriting rules with the given options.
void populateSparsificationPatterns(
    RewritePatternSet &patterns,
    const SparsificationOptions &options = SparsificationOptions());

/// Sets up sparse tensor conversion rules.
void populateSparseTensorConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createSparsificationPass();
std::unique_ptr<Pass> createSparseTensorConversionPass();

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
