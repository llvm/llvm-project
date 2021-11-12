//===- ComprehensiveBufferize.h - Linalg bufferization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetOperations.h"

namespace mlir {

class DominanceInfo;
class FuncOp;
class GlobalCreator;
class ModuleOp;

namespace linalg {
namespace comprehensive_bufferize {

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

struct BufferizationState;

/// Analyze the `ops` to determine which OpResults are inplaceable.
LogicalResult inPlaceAnalysis(SmallVector<Operation *> &ops,
                              BufferizationAliasInfo &aliasInfo,
                              const DominanceInfo &domInfo,
                              unsigned analysisFuzzerSeed = 0);

// TODO: Do not expose those functions in the header file.
/// Default allocation function that is used by the comprehensive bufferization
/// pass. The default currently creates a ranked memref using `memref.alloc`.
Optional<Value> defaultAllocationFn(OpBuilder &b, Location loc, MemRefType type,
                                    const SmallVector<Value> &dynShape);

/// Default deallocation function that is used by the comprehensive
/// bufferization pass. It expects to recieve back the value called from the
/// `defaultAllocationFn`.
void defaultDeallocationFn(OpBuilder &b, Location loc, Value allocatedBuffer);

/// Default memory copy function that is used by the comprehensive bufferization
/// pass. Creates a `linalg.copy` op.
void defaultMemCpyFn(OpBuilder &b, Location loc, Value from, Value to);

/// Return default allocation callbacks.
std::unique_ptr<AllocationCallbacks> defaultAllocationCallbacks();

/// Bufferize one particular op.
/// `bufferizedFunctionTypes` (resp. `globalCreator`) are expected to be
/// non-null if `op` is a CallOpInterface (resp. GlobalCreator).
LogicalResult
bufferizeOp(Operation *op, BufferizationState &state,
            DenseMap<FuncOp, FunctionType> *bufferizedFunctionTypes = nullptr);

/// Register external models implemented for the `BufferizableOpInterface`.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

struct BufferizationOptions {
  BufferizationOptions();

  /// Register a "post analysis" step. Such steps are executed after the
  /// analysis, but before bufferization.
  template <typename Step, typename... Args>
  void addPostAnalysisStep(Args... args) {
    postAnalysisSteps.emplace_back(
        std::make_unique<Step>(std::forward<Args>(args)...));
  }

  std::unique_ptr<AllocationCallbacks> allocationFns;
  bool allowReturnMemref = false;
  unsigned analysisFuzzerSeed = 0;
  bool testAnalysisOnly = false;
  std::vector<std::unique_ptr<PostAnalysisStep>> postAnalysisSteps;
};

LogicalResult runComprehensiveBufferize(ModuleOp moduleOp,
                                        const BufferizationOptions &options);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
