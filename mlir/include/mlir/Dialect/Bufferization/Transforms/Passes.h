#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace bufferization {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates an instance of the BufferDeallocation pass to free all allocated
/// buffers.
std::unique_ptr<Pass> createBufferDeallocationPass();

/// Creates a pass that finalizes a partial bufferization by removing remaining
/// bufferization.to_tensor and bufferization.to_memref operations.
std::unique_ptr<FunctionPass> createFinalizingBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
