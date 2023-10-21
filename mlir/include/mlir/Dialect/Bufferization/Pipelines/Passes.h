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
      llvm::cl::init(true)};
  PassOptions::Option<bool> verifyFunctionBoundaryABI{
      *this, "verify-function-boundary-abi",
      llvm::cl::desc(
          "Inserts `cf.assert` operations to verify the function boundary ABI "
          "at runtime. Currently, it is only checked that the ownership of "
          "returned MemRefs is 'true'. This makes sure that ownership is "
          "yielded and the returned MemRef does not originate from the same "
          "allocation as a function argument. If it can be determined "
          "statically that the ABI is not adhered to, an error will already be "
          "emitted at compile time. This cannot be changed with this option."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> removeExistingDeallocations{
      *this, "remove-existing-deallocations",
      llvm::cl::desc("Removes all pre-existing memref.dealloc operations and "
                     "insert all deallocations according to the buffer "
                     "deallocation rules."),
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
