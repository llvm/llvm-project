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

#define GEN_PASS_DECL
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"

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

// Options struct for BufferResultsToOutParams pass.
// Note: defined only here, not in tablegen.
struct BufferResultsToOutParamsOptions {
  // Filter function; returns true if the function should be converted.
  // Defaults to true, i.e. all functions are converted.
  llvm::function_ref<bool(func::FuncOp *)> filterFn = [](func::FuncOp *func) {
    return true;
  };
};

/// Creates a pass that converts memref function results to out-params.
std::unique_ptr<Pass> createBufferResultsToOutParamsPass(
    const BufferResultsToOutParamsOptions &options = {});

/// Replace buffers that are returned from a function with an out parameter.
/// Also update all call sites.
LogicalResult
promoteBufferResultsToOutParams(ModuleOp module,
                                const BufferResultsToOutParamsOptions &options);

/// Creates a pass that drops memref function results that are equivalent to a
/// function argument.
std::unique_ptr<Pass> createDropEquivalentBufferResultsPass();

/// Create a pass that rewrites tensor.empty to bufferization.alloc_tensor.
std::unique_ptr<Pass> createEmptyTensorToAllocTensorPass();

/// Drop all memref function results that are equivalent to a function argument.
LogicalResult dropEquivalentBufferResults(ModuleOp module);

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

/// Create a pass that resolves out-of-place tensor OpOperands with copies.
std::unique_ptr<Pass> createTensorCopyInsertionPass();
std::unique_ptr<Pass>
createTensorCopyInsertionPass(const OneShotBufferizationOptions &options);

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
