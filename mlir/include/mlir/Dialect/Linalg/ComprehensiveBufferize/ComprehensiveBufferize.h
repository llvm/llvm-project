//===- ComprehensiveBufferize.h - Linalg bufferization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H

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

class BufferizationAliasInfo;

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

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

/// Callback functions that are used by the comprehensive bufferization pass to
/// allocate/deallocate memory. These default to use the
/// `defaultAllocationFn`/`defaultDeallocationFn`, but can be overridden by the
/// caller. The `deallocationFn` is gauranteed to recieve the `Value` returned
/// by the `allocationFn`.
struct AllocationCallbacks {
  using AllocationFn = std::function<Optional<Value>(
      OpBuilder &, Location, MemRefType, const SmallVector<Value> &)>;
  using DeallocationFn = std::function<void(OpBuilder &, Location, Value)>;
  using MemCpyFn = std::function<void(OpBuilder &, Location, Value, Value)>;

  AllocationCallbacks(AllocationFn allocFn, DeallocationFn deallocFn,
                      MemCpyFn copyFn)
      : allocationFn(allocFn), deallocationFn(deallocFn), memCpyFn(copyFn) {}

  AllocationCallbacks()
      : allocationFn(defaultAllocationFn),
        deallocationFn(defaultDeallocationFn), memCpyFn(defaultMemCpyFn) {}

  AllocationFn allocationFn;
  DeallocationFn deallocationFn;
  MemCpyFn memCpyFn;
};

/// Bufferize one particular op.
/// `bufferizedFunctionTypes` (resp. `globalCreator`) are expected to be
/// non-null if `op` is a CallOpInterface (resp. GlobalCreator).
LogicalResult
bufferizeOp(Operation *op, BlockAndValueMapping &bvm,
            BufferizationAliasInfo &aliasInfo,
            AllocationCallbacks allocationFns,
            DenseMap<FuncOp, FunctionType> *bufferizedFunctionTypes = nullptr);

/// Register external models implemented for the `BufferizableOpInterface`.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

/// Try to eliminate InitTensorOps inside `funcOp`.
///
/// * `rewriteFunc` generates the replacement for the InitTensorOp.
/// * Only InitTensorOps that are anchored on a matching OpOperand as per
///   `anchorMatchFunc` are considered. "Anchored" means that there is a path on
///   the reverse SSA use-def chain, starting from the OpOperand and always
///   following the aliasing  OpOperand, that eventually ends at a single
///   InitTensorOp.
/// * The result of `rewriteFunc` must usually be analyzed for inplacability.
///   This analysis can be skipped with `skipAnalysis`.
LogicalResult initTensorElimination(
    FuncOp funcOp, BufferizationAliasInfo &aliasInfo, DominanceInfo &domInfo,
    std::function<bool(OpOperand &)> anchorMatchFunc,
    std::function<Value(OpBuilder &, Location, OpOperand &)> rewriteFunc,
    bool skipAnalysis = false);

/// Try to eliminate InitTensorOps inside funcOp that are anchored on an
/// InsertSliceOp, i.e., if it is eventually inserted into another tensor
/// (and some other conditions are met).
LogicalResult eliminateInsertSliceAnchoredInitTensorOps(
    FuncOp funcOp, BufferizationAliasInfo &aliasInfo, DominanceInfo &domInfo);

struct BufferizationOptions {
  BufferizationOptions()
      : allocationFns(std::make_unique<AllocationCallbacks>()) {}

  std::unique_ptr<AllocationCallbacks> allocationFns;
  bool allowReturnMemref = false;
  unsigned analysisFuzzerSeed = 0;
  bool testAnalysisOnly = false;
};

LogicalResult runComprehensiveBufferize(ModuleOp moduleOp,
                                        const BufferizationOptions &options);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
