#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace func {
class FuncOp;
} // namespace func

namespace bufferization {
struct OneShotBufferizationOptions;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates an instance of the BufferDeallocation pass to free all allocated
/// buffers.
std::unique_ptr<Pass> createBufferDeallocationPass();

/// Run buffer deallocation.
LogicalResult deallocateBuffers(Operation *op);

/// Creates a pass that moves allocations upwards to reduce the number of
/// required copies that are inserted during the BufferDeallocation pass.
std::unique_ptr<Pass> createBufferHoistingPass();

/// Creates a pass that moves allocations upwards out of loops. This avoids
/// reallocations inside of loops.
std::unique_ptr<Pass> createBufferLoopHoistingPass();

/// Creates a pass that converts memref function results to out-params.
std::unique_ptr<Pass> createBufferResultsToOutParamsPass();

/// Replace buffers that are returned from a function with an out parameter.
/// Also update all call sites.
LogicalResult promoteBufferResultsToOutParams(ModuleOp module);

/// Creates a pass that finalizes a partial bufferization by removing remaining
/// bufferization.to_tensor and bufferization.to_memref operations.
std::unique_ptr<OperationPass<func::FuncOp>> createFinalizingBufferizePass();

/// Create a pass that bufferizes all ops that implement BufferizableOpInterface
/// with One-Shot Bufferize.
std::unique_ptr<Pass> createOneShotBufferizePass();

/// Create a pass that bufferizes all ops that implement BufferizableOpInterface
/// with One-Shot Bufferize and the specified bufferization options.
std::unique_ptr<Pass>
createOneShotBufferizePass(const OneShotBufferizationOptions &options);

/// Creates a pass that promotes heap-based allocations to stack-based ones.
/// Only buffers smaller than the provided size are promoted.
/// Dynamic shaped buffers are promoted up to the given rank.
std::unique_ptr<Pass>
createPromoteBuffersToStackPass(unsigned maxAllocSizeInBytes = 1024,
                                unsigned maxRankOfAllocatedMemRef = 1);

/// Creates a pass that promotes heap-based allocations to stack-based ones.
/// Only buffers smaller with `isSmallAlloc(alloc) == true` are promoted.
std::unique_ptr<Pass>
createPromoteBuffersToStackPass(std::function<bool(Value)> isSmallAlloc);

/// Create a pass that tries to eliminate alloc_tensor ops that are anchored on
/// insert_slice ops.
std::unique_ptr<Pass> createAllocTensorEliminationPass();

/// Create a pass that bufferizes ops from the bufferization dialect.
std::unique_ptr<Pass> createBufferizationBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register external models for AllocationOpInterface.
void registerAllocationOpInterfaceExternalModels(DialectRegistry &registry);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
