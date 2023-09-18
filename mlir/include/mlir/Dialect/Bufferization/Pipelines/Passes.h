//===- Passes.h - Bufferization pipeline entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all bufferization pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_PIPELINES_PASSES_H
#define MLIR_DIALECT_BUFFERIZATION_PIPELINES_PASSES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace bufferization {
struct DeallocationOptions;

/// Options for the buffer deallocation pipeline.
struct BufferDeallocationPipelineOptions
    : public PassPipelineOptions<BufferDeallocationPipelineOptions> {
  PassOptions::Option<bool> privateFunctionDynamicOwnership{
      *this, "private-function-dynamic-ownership",
      llvm::cl::desc(
          "Allows to add additional arguments to private functions to "
          "dynamically pass ownership of memrefs to callees. This can enable "
          "earlier deallocations."),
      llvm::cl::init(false)};
  PassOptions::Option<bool> allowCloning{
      *this, "allow-cloning",
      llvm::cl::desc(
          "Allows the pass to insert `bufferization.clone` operations. This is "
          "useful for supporting IR that does not adhere to the function "
          "boundary ABI initially (excl. external functions) and to support "
          "operations with results with 'Unknown' ownership. However, it "
          "requires that all buffer writes dominate all buffer reads (i.e., "
          "only enable this option if your IR is guaranteed to have this "
          "property)."),
      llvm::cl::init(false)};

  /// Convert this BufferDeallocationPipelineOptions struct to a
  /// DeallocationOptions struct to be passed to the
  /// OwnershipBasedBufferDeallocationPass.
  DeallocationOptions asDeallocationOptions() const;
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the buffer deallocation pipeline to the `OpPassManager`. This
/// is the standard pipeline for deallocating the MemRefs introduced by the
/// One-Shot bufferization pass.
void buildBufferDeallocationPipeline(
    OpPassManager &pm, const BufferDeallocationPipelineOptions &options);

/// Registers all pipelines for the `bufferization` dialect. Currently,
/// this includes only the "buffer-deallocation-pipeline".
void registerBufferizationPipelines();

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_PIPELINES_PASSES_H
