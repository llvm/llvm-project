//===- BufferResultsToOutParams.cpp - Calling convention conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_BUFFERRESULTSTOOUTPARAMSPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using AllocationFn = bufferization::BufferResultsToOutParamsOpts::AllocationFn;
using MemCpyFn = bufferization::BufferResultsToOutParamsOpts::MemCpyFn;
using AllocDynamicSizesMap =
    llvm::DenseMap<func::FuncOp, SmallVector<SmallVector<Value>>>;

/// Return `true` if the given MemRef type has a fully dynamic layout.
static bool hasFullyDynamicLayoutMap(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  if (!llvm::all_of(strides, ShapedType::isDynamic))
    return false;
  if (ShapedType::isStatic(offset))
    return false;
  return true;
}

/// Return `true` if the given MemRef type has a static identity layout (i.e.,
/// no layout).
static bool hasStaticIdentityLayout(MemRefType type) {
  return type.getLayout().isIdentity();
}

/// Return the dynamic shapes of the `memref` based on the defining op. If the
/// complete dynamic shape fails to be captured, return an empty value.
/// Currently, only function block arguments are supported for capturing.
static SmallVector<Value> getDynamicSize(Value memref, func::FuncOp funcOp) {
  Operation *defOp = memref.getDefiningOp();
  if (!defOp)
    return {};
  auto operands = defOp->getOperands();
  SmallVector<Value> dynamicSizes;
  for (Value size : operands) {
    if (!isa<IndexType>(size.getType()))
      continue;

    BlockArgument sizeSrc = dyn_cast<BlockArgument>(size);
    if (!sizeSrc)
      return {};
    auto arguments = funcOp.getArguments();
    auto iter = llvm::find(arguments, sizeSrc);
    if (iter == arguments.end())
      return {};
    dynamicSizes.push_back(*iter);
  }
  return dynamicSizes;
}

/// Returns the dynamic sizes at the callee, through the call relationship
/// between the caller and callee.
static SmallVector<Value> mapDynamicSizeAtCaller(func::CallOp call,
                                                 func::FuncOp callee,
                                                 ValueRange dynamicSizes) {
  SmallVector<Value> mappedDynamicSizes;
  for (Value size : dynamicSizes) {
    for (auto [src, dst] :
         llvm::zip_first(call.getOperands(), callee.getArguments())) {
      if (size != dst)
        continue;
      mappedDynamicSizes.push_back(src);
    }
  }
  assert(mappedDynamicSizes.size() == dynamicSizes.size() &&
         "could not find all dynamic sizes");
  return mappedDynamicSizes;
}

// Updates the func op and entry block.
//
// Any args appended to the entry block are added to `appendedEntryArgs`.
// If `addResultAttribute` is true, adds the unit attribute `bufferize.result`
// to each newly created function argument.
static LogicalResult
updateFuncOp(func::FuncOp func,
             SmallVectorImpl<BlockArgument> &appendedEntryArgs,
             bool addResultAttribute) {
  auto functionType = func.getFunctionType();

  // Collect information about the results will become appended arguments.
  SmallVector<Type, 6> erasedResultTypes;
  BitVector erasedResultIndices(functionType.getNumResults());
  for (const auto &resultType : llvm::enumerate(functionType.getResults())) {
    if (auto memrefType = dyn_cast<MemRefType>(resultType.value())) {
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
    if (addResultAttribute)
      func.setArgAttr(functionType.getNumInputs() + i,
                      StringAttr::get(func.getContext(), "bufferize.result"),
                      UnitAttr::get(func.getContext()));
  }

  // Erase the results.
  if (failed(func.eraseResults(erasedResultIndices)))
    return failure();

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
static LogicalResult
updateReturnOps(func::FuncOp func, ArrayRef<BlockArgument> appendedEntryArgs,
                AllocDynamicSizesMap &map,
                const bufferization::BufferResultsToOutParamsOpts &options) {
  auto res = func.walk([&](func::ReturnOp op) {
    SmallVector<Value, 6> copyIntoOutParams;
    SmallVector<Value, 6> keepAsReturnOperands;
    for (Value operand : op.getOperands()) {
      if (isa<MemRefType>(operand.getType()))
        copyIntoOutParams.push_back(operand);
      else
        keepAsReturnOperands.push_back(operand);
    }
    OpBuilder builder(op);
    SmallVector<SmallVector<Value>> dynamicSizes;
    for (auto [orig, arg] : llvm::zip(copyIntoOutParams, appendedEntryArgs)) {
      bool hoistStaticAllocs =
          options.hoistStaticAllocs &&
          cast<MemRefType>(orig.getType()).hasStaticShape();
      bool hoistDynamicAllocs =
          options.hoistDynamicAllocs &&
          !cast<MemRefType>(orig.getType()).hasStaticShape();
      if ((hoistStaticAllocs || hoistDynamicAllocs) &&
          isa_and_nonnull<bufferization::AllocationOpInterface>(
              orig.getDefiningOp())) {
        orig.replaceAllUsesWith(arg);
        if (hoistDynamicAllocs) {
          SmallVector<Value> dynamicSize = getDynamicSize(orig, func);
          dynamicSizes.push_back(dynamicSize);
        }
        orig.getDefiningOp()->erase();
      } else {
        if (failed(options.memCpyFn(builder, op.getLoc(), orig, arg)))
          return WalkResult::interrupt();
      }
    }
    func::ReturnOp::create(builder, op.getLoc(), keepAsReturnOperands);
    op.erase();
    auto dynamicSizePair =
        std::pair<func::FuncOp, SmallVector<SmallVector<Value>>>(func,
                                                                 dynamicSizes);
    map.insert(dynamicSizePair);
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

// Updates all CallOps in the scope of the given ModuleOp by allocating
// temporary buffers for newly introduced out params.
static LogicalResult
updateCalls(ModuleOp module, const AllocDynamicSizesMap &map,
            const bufferization::BufferResultsToOutParamsOpts &options) {
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
      if (isa<MemRefType>(result.getType()))
        replaceWithOutParams.push_back(result);
      else
        replaceWithNewCallResults.push_back(result);
    }
    SmallVector<Value, 6> outParams;
    OpBuilder builder(op);
    SmallVector<SmallVector<Value>> dynamicSizes = map.lookup(callee);
    size_t dynamicSizesIndex = 0;
    for (Value memref : replaceWithOutParams) {
      SmallVector<Value> dynamicSize = dynamicSizes.size() > dynamicSizesIndex
                                           ? dynamicSizes[dynamicSizesIndex]
                                           : SmallVector<Value>();
      bool memrefStaticShape =
          cast<MemRefType>(memref.getType()).hasStaticShape();
      if (!memrefStaticShape && dynamicSize.empty()) {
        op.emitError()
            << "cannot create out param for dynamically shaped result";
        didFail = true;
        return;
      }
      auto memrefType = cast<MemRefType>(memref.getType());
      auto allocType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                          AffineMap(), memrefType.getMemorySpace());

      if (memrefStaticShape) {
        dynamicSize = {};
      } else {
        ++dynamicSizesIndex;
        dynamicSize = mapDynamicSizeAtCaller(op, callee, dynamicSize);
      }
      auto maybeOutParam =
          options.allocationFn(builder, op.getLoc(), allocType, dynamicSize);
      if (failed(maybeOutParam)) {
        op.emitError() << "failed to create allocation op";
        didFail = true;
        return;
      }
      Value outParam = maybeOutParam.value();
      if (!hasStaticIdentityLayout(memrefType)) {
        // Layout maps are already checked in `updateFuncOp`.
        assert(hasFullyDynamicLayoutMap(memrefType) &&
               "layout map not supported");
        outParam =
            memref::CastOp::create(builder, op.getLoc(), memrefType, outParam);
      }
      memref.replaceAllUsesWith(outParam);
      outParams.push_back(outParam);
    }

    auto newOperands = llvm::to_vector<6>(op.getOperands());
    newOperands.append(outParams.begin(), outParams.end());
    auto newResultTypes = llvm::to_vector<6>(llvm::map_range(
        replaceWithNewCallResults, [](Value v) { return v.getType(); }));
    auto newCall = func::CallOp::create(
        builder, op.getLoc(), op.getCalleeAttr(), newResultTypes, newOperands);
    for (auto t : llvm::zip(replaceWithNewCallResults, newCall.getResults()))
      std::get<0>(t).replaceAllUsesWith(std::get<1>(t));
    op.erase();
  });

  return failure(didFail);
}

LogicalResult mlir::bufferization::promoteBufferResultsToOutParams(
    ModuleOp module,
    const bufferization::BufferResultsToOutParamsOpts &options) {
  // It maps the shape source of the dynamic shape memref returned by each
  // function.
  AllocDynamicSizesMap map;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!options.filterFn(&func))
      continue;
    SmallVector<BlockArgument, 6> appendedEntryArgs;
    if (failed(
            updateFuncOp(func, appendedEntryArgs, options.addResultAttribute)))
      return failure();
    if (func.isExternal())
      continue;
    if (failed(updateReturnOps(func, appendedEntryArgs, map, options))) {
      return failure();
    }
  }
  if (failed(updateCalls(module, map, options)))
    return failure();
  return success();
}

namespace {
struct BufferResultsToOutParamsPass
    : bufferization::impl::BufferResultsToOutParamsPassBase<
          BufferResultsToOutParamsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Convert from pass options in tablegen to BufferResultsToOutParamsOpts.
    if (addResultAttribute)
      options.addResultAttribute = true;
    if (hoistStaticAllocs)
      options.hoistStaticAllocs = true;
    if (hoistDynamicAllocs)
      options.hoistDynamicAllocs = true;

    if (failed(bufferization::promoteBufferResultsToOutParams(getOperation(),
                                                              options)))
      return signalPassFailure();
  }

private:
  bufferization::BufferResultsToOutParamsOpts options;
};
} // namespace
