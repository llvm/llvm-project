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

// Forward.
class TypeConverter;

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

/// Converts command-line parallelization flag to the strategy enum.
SparseParallelizationStrategy sparseParallelizationStrategy(int32_t flag);

/// Defines a vectorization strategy. Any inner loop is a candidate (full SIMD
/// for parallel loops and horizontal SIMD for reduction loops). A loop is
/// actually vectorized if (1) allowed by the strategy, and (2) the emitted
/// code is an actual for-loop (and not a co-iterating while-loop).
enum class SparseVectorizationStrategy {
  kNone,
  kDenseInnerLoop,
  kAnyStorageInnerLoop
};

/// Converts command-line vectorization flag to the strategy enum.
SparseVectorizationStrategy sparseVectorizationStrategy(int32_t flag);

/// Sparsification options.
struct SparsificationOptions {
  SparsificationOptions(SparseParallelizationStrategy p,
                        SparseVectorizationStrategy v, unsigned vl, bool e)
      : parallelizationStrategy(p), vectorizationStrategy(v), vectorLength(vl),
        enableSIMDIndex32(e) {}
  SparsificationOptions()
      : SparsificationOptions(SparseParallelizationStrategy::kNone,
                              SparseVectorizationStrategy::kNone, 1u, false) {}
  SparseParallelizationStrategy parallelizationStrategy;
  SparseVectorizationStrategy vectorizationStrategy;
  unsigned vectorLength;
  bool enableSIMDIndex32;
};

/// Sets up sparsification rewriting rules with the given options.
void populateSparsificationPatterns(
    RewritePatternSet &patterns,
    const SparsificationOptions &options = SparsificationOptions());

/// Sets up sparse tensor conversion rules.
void populateSparseTensorConversionPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns);

std::unique_ptr<Pass> createSparsificationPass();
std::unique_ptr<Pass>
createSparsificationPass(const SparsificationOptions &options);
std::unique_ptr<Pass> createSparseTensorConversionPass();

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
