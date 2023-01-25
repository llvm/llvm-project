//===- BufferResultsToOutParams.cpp - Calling convention conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_BUFFERRESULTSTOOUTPARAMS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;

/// Return `true` if the given MemRef type has a fully dynamic layout.
static bool hasFullyDynamicLayoutMap(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;
  if (!llvm::all_of(strides, ShapedType::isDynamic))
    return false;
  if (!ShapedType::isDynamic(offset))
    return false;
  return true;
}

/// Return `true` if the given MemRef type has a static identity layout (i.e.,
/// no layout).
static bool hasStaticIdentityLayout(MemRefType type) {
  return type.getLayout().isIdentity();
}

// Updates the func op and entry block.
//
// Any args appended to the entry block are added to `appendedEntryArgs`.
static LogicalResult
updateFuncOp(func::FuncOp func,
             SmallVectorImpl<BlockArgument> &appendedEntryArgs) {
  auto functionType = func.getFunctionType();

  // Collect information about the results will become appended arguments.
  SmallVector<Type, 6> erasedResultTypes;
  BitVector erasedResultIndices(functionType.getNumResults());
  for (const auto &resultType : llvm::enumerate(functionType.getResults())) {
    if (auto memrefType = resultType.value().dyn_cast<MemRefType>()) {
      if (!hasStaticIdentityLayout(memrefType) &&
          !hasFullyDynamicLayoutMap(memrefType)) {
        // Only buffers with static identity layout can be allocated. These can
        // be casted to memrefs with fully dynamic layout map. Other layout maps
        // are not supported.
        return func->emitError()
               << "cannot create out param for result with unsupported layout";
      }
      erasedResultIndices.set(resultType.index());
      erasedResultTypes.push_back(memrefType);
    }
  }

  // Add the new arguments to the function type.
  auto newArgTypes = llvm::to_vector<6>(
      llvm::concat<const Type>(functionType.getInputs(), erasedResultTypes));
  auto newFunctionType = FunctionType::get(func.getContext(), newArgTypes,
                                           functionType.getResults());
  func.setType(newFunctionType);

  // Transfer the result attributes to arg attributes.
  auto erasedIndicesIt = erasedResultIndices.set_bits_begin();
  for (int i = 0, e = erasedResultTypes.size(); i < e; ++i, ++erasedIndicesIt) {
    func.setArgAttrs(functionType.getNumInputs() + i,
                     func.getResultAttrs(*erasedIndicesIt));
  }

  // Erase the results.
  func.eraseResults(erasedResultIndices);

  // Add the new arguments to the entry block if the function is not external.
  if (func.isExternal())
    return success();
  Location loc = func.getLoc();
  for (Type type : erasedResultTypes)
    appendedEntryArgs.push_back(func.front().addArgument(type, loc));

  return success();
}

// Updates all ReturnOps in the scope of the given func::FuncOp by either
// keeping them as return values or copying the associated buffer contents into
// the given out-params.
static void updateReturnOps(func::FuncOp func,
                            ArrayRef<BlockArgument> appendedEntryArgs) {
  func.walk([&](func::ReturnOp op) {
    SmallVector<Value, 6> copyIntoOutParams;
    SmallVector<Value, 6> keepAsReturnOperands;
    for (Value operand : op.getOperands()) {
      if (operand.getType().isa<MemRefType>())
        copyIntoOutParams.push_back(operand);
      else
        keepAsReturnOperands.push_back(operand);
    }
    OpBuilder builder(op);
    for (auto t : llvm::zip(copyIntoOutParams, appendedEntryArgs))
      builder.create<memref::CopyOp>(op.getLoc(), std::get<0>(t),
                                     std::get<1>(t));
    builder.create<func::ReturnOp>(op.getLoc(), keepAsReturnOperands);
    op.erase();
  });
}

// Updates all CallOps in the scope of the given ModuleOp by allocating
// temporary buffers for newly introduced out params.
static LogicalResult
updateCalls(ModuleOp module,
            const bufferization::BufferResultsToOutParamsOptions &options) {
  bool didFail = false;
  SymbolTable symtab(module);
  module.walk([&](func::CallOp op) {
    auto callee = symtab.lookup<func::FuncOp>(op.getCallee());
    if (!callee) {
      op.emitError() << "cannot find callee '" << op.getCallee() << "' in "
                     << "symbol table";
      didFail = true;
      return;
    }
    if (!options.filterFn(&callee))
      return;
    SmallVector<Value, 6> replaceWithNewCallResults;
    SmallVector<Value, 6> replaceWithOutParams;
    for (OpResult result : op.getResults()) {
      if (result.getType().isa<MemRefType>())
        replaceWithOutParams.push_back(result);
      else
        replaceWithNewCallResults.push_back(result);
    }
    SmallVector<Value, 6> outParams;
    OpBuilder builder(op);
    for (Value memref : replaceWithOutParams) {
      if (!memref.getType().cast<MemRefType>().hasStaticShape()) {
        op.emitError()
            << "cannot create out param for dynamically shaped result";
        didFail = true;
        return;
      }
      auto memrefType = memref.getType().cast<MemRefType>();
      auto allocType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                          AffineMap(), memrefType.getMemorySpace());
      Value outParam = builder.create<memref::AllocOp>(op.getLoc(), allocType);
      if (!hasStaticIdentityLayout(memrefType)) {
        // Layout maps are already checked in `updateFuncOp`.
        assert(hasFullyDynamicLayoutMap(memrefType) &&
               "layout map not supported");
        outParam =
            builder.create<memref::CastOp>(op.getLoc(), memrefType, outParam);
      }
      memref.replaceAllUsesWith(outParam);
      outParams.push_back(outParam);
    }

    auto newOperands = llvm::to_vector<6>(op.getOperands());
    newOperands.append(outParams.begin(), outParams.end());
    auto newResultTypes = llvm::to_vector<6>(llvm::map_range(
        replaceWithNewCallResults, [](Value v) { return v.getType(); }));
    auto newCall = builder.create<func::CallOp>(op.getLoc(), op.getCalleeAttr(),
                                                newResultTypes, newOperands);
    for (auto t : llvm::zip(replaceWithNewCallResults, newCall.getResults()))
      std::get<0>(t).replaceAllUsesWith(std::get<1>(t));
    op.erase();
  });

  return failure(didFail);
}

LogicalResult mlir::bufferization::promoteBufferResultsToOutParams(
    ModuleOp module,
    const bufferization::BufferResultsToOutParamsOptions &options) {
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!options.filterFn(&func))
      continue;
    SmallVector<BlockArgument, 6> appendedEntryArgs;
    if (failed(updateFuncOp(func, appendedEntryArgs)))
      return failure();
    if (func.isExternal())
      continue;
    updateReturnOps(func, appendedEntryArgs);
  }
  if (failed(updateCalls(module, options)))
    return failure();
  return success();
}

namespace {
struct BufferResultsToOutParamsPass
    : bufferization::impl::BufferResultsToOutParamsBase<
          BufferResultsToOutParamsPass> {
  explicit BufferResultsToOutParamsPass(
      const bufferization::BufferResultsToOutParamsOptions &options)
      : options(options) {}

  void runOnOperation() override {
    if (failed(bufferization::promoteBufferResultsToOutParams(getOperation(),
                                                              options)))
      return signalPassFailure();
  }

private:
  bufferization::BufferResultsToOutParamsOptions options;
};
} // namespace

std::unique_ptr<Pass> mlir::bufferization::createBufferResultsToOutParamsPass(
    const bufferization::BufferResultsToOutParamsOptions &options) {
  return std::make_unique<BufferResultsToOutParamsPass>(options);
}
