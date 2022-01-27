//===- Passes.h - Sparse tensor pipeline entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_
#define MLIR_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace sparse_tensor {

/// Options for the "sparse-compiler" pipeline.  So far this contains
/// only the same options as the sparsification pass, and must be kept
/// in sync with the `SparseTensor/Transforms/Passes.td` file.  In the
/// future this may be extended with options for other passes in the pipeline.
struct SparseCompilerOptions
    : public PassPipelineOptions<SparseCompilerOptions> {
  PassOptions::Option<int32_t> parallelization{
      *this, "parallelization-strategy",
      desc("Set the parallelization strategy"), init(0)};
  PassOptions::Option<int32_t> vectorization{
      *this, "vectorization-strategy", desc("Set the vectorization strategy"),
      init(0)};
  PassOptions::Option<int32_t> vectorLength{
      *this, "vl", desc("Set the vector length"), init(1)};
  PassOptions::Option<bool> enableSIMDIndex32{
      *this, "enable-simd-index32",
      desc("Enable i32 indexing into vectors (for efficiency)"), init(false)};

  /// Projects out the options for the sparsification pass.
  SparsificationOptions sparsificationOptions() const {
    return SparsificationOptions(sparseParallelizationStrategy(parallelization),
                                 sparseVectorizationStrategy(vectorization),
                                 vectorLength, enableSIMDIndex32);
  }
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "sparse-compiler" pipeline to the `OpPassManager`.  This
/// is the standard pipeline for taking sparsity-agnostic IR using
/// the sparse-tensor type and lowering it to LLVM IR with concrete
/// representations and algorithms for sparse tensors.
void buildSparseCompiler(OpPassManager &pm,
                         const SparseCompilerOptions &options);

/// Registers all pipelines for the `sparse_tensor` dialect.  At present,
/// this includes only "sparse-compiler".
void registerSparseTensorPipelines();

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_
