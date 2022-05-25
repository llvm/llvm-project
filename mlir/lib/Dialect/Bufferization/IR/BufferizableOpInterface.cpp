//===- BufferizableOpInterface.cpp - Bufferizable Ops  ---=----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace bufferization {

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.cpp.inc"

} // namespace bufferization
} // namespace mlir

#define DEBUG_TYPE "bufferizable-op-interface"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X))

using namespace mlir;
using namespace bufferization;

/// Attribute name used to mark region arguments that can be bufferized
/// in-place during linalg comprehensive bufferization.
constexpr const ::llvm::StringLiteral
    bufferization::BufferizableOpInterface::kInplaceableAttrName;

/// Attribute name used to mark allocs that are created by the bufferization.
static const char *kBufferAllocationAttr = "bufferization.allocation";

/// Attribute name used to mark allocs that should not be deallocated.
static const char *kSkipDeallocAttr = "bufferization.skip_dealloc";

//===----------------------------------------------------------------------===//
// BufferizationOptions
//===----------------------------------------------------------------------===//

// Default constructor for BufferizationOptions.
BufferizationOptions::BufferizationOptions() = default;

bool BufferizationOptions::isOpAllowed(Operation *op) const {
  // Special case: If function boundary bufferization is deactivated, do not
  // allow ops that belong to the `func` dialect.
  bool isFuncBoundaryOp = isa_and_nonnull<func::FuncDialect>(op->getDialect());
  if (!bufferizeFunctionBoundaries && isFuncBoundaryOp)
    return false;

  // All other ops: Allow/disallow according to filter.
  bool isAllowed = !filterHasAllowRule();
  for (const OpFilterEntry &entry : opFilter) {
    bool filterResult = entry.fn(op);
    switch (entry.type) {
    case OpFilterEntry::ALLOW:
      isAllowed |= filterResult;
      break;
    case OpFilterEntry::DENY:
      if (filterResult)
        // DENY filter matches. This op is no allowed. (Even if other ALLOW
        // filters may match.)
        return false;
    };
  }
  return isAllowed;
}

BufferizableOpInterface
BufferizationOptions::dynCastBufferizableOp(Operation *op) const {
  auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op);
  if (!bufferizableOp)
    return nullptr;
  if (!isOpAllowed(op))
    return nullptr;
  return bufferizableOp;
}

BufferizableOpInterface
BufferizationOptions::dynCastBufferizableOp(Value value) const {
  if (auto bufferizableOp = value.getDefiningOp<BufferizableOpInterface>())
    if (isOpAllowed(bufferizableOp.getOperation()))
      return bufferizableOp;
  return nullptr;
}

void BufferizationOptions::addDialectStateInitializer(
    StringRef name, const DialectStateInitFn &fn) {
  stateInitializers.push_back(
      [=](AnalysisState &state) { state.insertDialectState(name, fn()); });
}

//===----------------------------------------------------------------------===//
// Helper functions for BufferizableOpInterface
//===----------------------------------------------------------------------===//

static void setInsertionPointAfter(OpBuilder &b, Value value) {
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
  } else {
    b.setInsertionPointAfter(value.getDefiningOp());
  }
}

/// Determine which OpOperand* will alias with `result` if the op is bufferized
/// in place. Return an empty vector if the op is not bufferizable.
SmallVector<OpOperand *>
AnalysisState::getAliasingOpOperand(OpResult result) const {
  if (Operation *op = result.getDefiningOp())
    if (auto bufferizableOp = getOptions().dynCastBufferizableOp(op))
      return bufferizableOp.getAliasingOpOperand(result, *this);
  return {};
}

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. Return an empty vector if the op is not bufferizable.
SmallVector<OpResult>
AnalysisState::getAliasingOpResult(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          getOptions().dynCastBufferizableOp(opOperand.getOwner()))
    return bufferizableOp.getAliasingOpResult(opOperand, *this);
  return {};
}

/// Return true if `opOperand` bufferizes to a memory read. Return `true` if the
/// op is not bufferizable.
bool AnalysisState::bufferizesToMemoryRead(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          getOptions().dynCastBufferizableOp(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryRead(opOperand, *this);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` bufferizes to a memory write. Return
/// `true` if the op is not bufferizable.
bool AnalysisState::bufferizesToMemoryWrite(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          getOptions().dynCastBufferizableOp(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryWrite(opOperand, *this);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` does neither read nor write but bufferizes to an
/// alias. Return false if the op is not bufferizable.
bool AnalysisState::bufferizesToAliasOnly(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          getOptions().dynCastBufferizableOp(opOperand.getOwner()))
    return bufferizableOp.bufferizesToAliasOnly(opOperand, *this);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return false.
  return false;
}

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool AnalysisState::isValueRead(Value value) const {
  assert(value.getType().isa<TensorType>() && "expected TensorType");
  SmallVector<OpOperand *> workingSet;
  for (OpOperand &use : value.getUses())
    workingSet.push_back(&use);

  while (!workingSet.empty()) {
    OpOperand *uMaybeReading = workingSet.pop_back_val();
    // Skip over all ops that neither read nor write (but create an alias).
    if (bufferizesToAliasOnly(*uMaybeReading))
      for (OpResult opResult : getAliasingOpResult(*uMaybeReading))
        for (OpOperand &use : opResult.getUses())
          workingSet.push_back(&use);
    if (bufferizesToMemoryRead(*uMaybeReading))
      return true;
  }

  return false;
}

// Starting from `value`, follow the use-def chain in reverse, always selecting
// the aliasing OpOperands. Find and return Values for which `condition`
// evaluates to true. OpOperands of such matching Values are not traversed any
// further.
llvm::SetVector<Value> AnalysisState::findValueInReverseUseDefChain(
    Value value, llvm::function_ref<bool(Value)> condition) const {
  llvm::SetVector<Value> result, workingSet;
  workingSet.insert(value);

  while (!workingSet.empty()) {
    Value value = workingSet.pop_back_val();
    if (condition(value) || value.isa<BlockArgument>()) {
      result.insert(value);
      continue;
    }

    OpResult opResult = value.cast<OpResult>();
    SmallVector<OpOperand *> opOperands = getAliasingOpOperand(opResult);
    if (opOperands.empty() || !options.isOpAllowed(value.getDefiningOp())) {
      result.insert(value);
      continue;
    }

    for (OpOperand *o : opOperands)
      workingSet.insert(o->get());
  }

  return result;
}

// Find the Values of the last preceding write of a given Value.
llvm::SetVector<Value>
AnalysisState::findLastPrecedingWrite(Value value) const {
  return findValueInReverseUseDefChain(value, [&](Value value) {
    Operation *op = value.getDefiningOp();
    if (!op)
      return true;
    auto bufferizableOp = options.dynCastBufferizableOp(op);
    if (!bufferizableOp)
      return true;
    return bufferizableOp.isMemoryWrite(value.cast<OpResult>(), *this);
  });
}

AnalysisState::AnalysisState(const BufferizationOptions &options)
    : options(options) {
  for (const BufferizationOptions::AnalysisStateInitFn &fn :
       options.stateInitializers)
    fn(*this);
}

// bufferization.to_memref is not allowed to change the rank.
static void ensureToMemrefOpIsValid(Value tensor, Type memrefType) {
#ifndef NDEBUG
  auto rankedTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  assert((!rankedTensorType || memrefType.cast<MemRefType>().getRank() ==
                                   rankedTensorType.getRank()) &&
         "to_memref would be invalid: mismatching ranks");
#endif
}

Value mlir::bufferization::lookupBuffer(RewriterBase &rewriter, Value tensor,
                                        const BufferizationOptions &options) {
  auto tensorType = tensor.getType().dyn_cast<TensorType>();
  assert(tensorType && "unexpected non-tensor type");

  // Replace "%t = to_tensor %m" with %m.
  if (auto toTensorOp = tensor.getDefiningOp<bufferization::ToTensorOp>())
    return toTensorOp.memref();

  // Insert to_memref op.
  OpBuilder::InsertionGuard g(rewriter);
  setInsertionPointAfter(rewriter, tensor);
  Type memrefType = getMemRefType(tensorType, options);
  ensureToMemrefOpIsValid(tensor, memrefType);
  return rewriter.create<bufferization::ToMemrefOp>(tensor.getLoc(), memrefType,
                                                    tensor);
}

/// Return the buffer (memref) for a given OpOperand (tensor). Allocate
/// a new buffer and copy over data from the existing buffer if out-of-place
/// bufferization was decided.
FailureOr<Value>
BufferizationState::getBuffer(RewriterBase &rewriter, OpOperand &opOperand,
                              Optional<ForceInPlacability> overrideInPlace,
                              Optional<Operation *> customCopyInsertionPoint) {
  const BufferizationOptions &options = analysisState.getOptions();
  OpBuilder::InsertionGuard guard(rewriter);
  Operation *op = opOperand.getOwner();
  Location loc = op->getLoc();
  SmallVector<OpResult> aliasingOpResults =
      analysisState.getAliasingOpResult(opOperand);
  Value operand = opOperand.get();
  Value operandBuffer = lookupBuffer(rewriter, operand, options);

  // Can `operandBuffer` be used directly or do we need a copy?
  bool inplace =
      overrideInPlace != FORCE_OUT_OF_PLACE &&
      (overrideInPlace == FORCE_INPLACE || analysisState.isInPlace(opOperand));
  if (inplace)
    return operandBuffer;

  // Bufferizing out-of-place: Allocate a new buffer.
  // Move insertion point right after `operandBuffer`. That is where the
  // allocation should be inserted (in the absence of allocation hoisting).
  setInsertionPointAfter(rewriter, operandBuffer);
  // Allocate the result buffer. The buffer should be deallocated if the tensor
  // is not yielded and deallocs are enabled in general.
  bool dealloc = llvm::none_of(aliasingOpResults, [&](Value v) {
    return getAnalysisState().isTensorYielded(v);
  });
  FailureOr<Value> resultBuffer = createAlloc(
      rewriter, loc, operandBuffer, dealloc && getOptions().createDeallocs);
  if (failed(resultBuffer))
    return failure();
  // Do not copy the buffer if its contents are undefined.
  if (analysisState.hasUndefinedContents(&opOperand))
    return resultBuffer;
  // Do not copy if the copied data is never read.
  if (!aliasingOpResults.empty() &&
      !analysisState.bufferizesToMemoryRead(opOperand) &&
      llvm::none_of(aliasingOpResults, [&](OpResult opResult) {
        return analysisState.isValueRead(opResult);
      }))
    return resultBuffer;
  // Do not copy if this op does not read the data, but writes it.
  if (analysisState.bufferizesToMemoryWrite(opOperand) &&
      !analysisState.bufferizesToMemoryRead(opOperand))
    return resultBuffer;

  if (customCopyInsertionPoint) {
    rewriter.setInsertionPoint(*customCopyInsertionPoint);
  } else {
    // The copy happens right before the op that is bufferized.
    rewriter.setInsertionPoint(op);
  }
  if (failed(options.createMemCpy(rewriter, loc, operandBuffer, *resultBuffer)))
    return failure();

  return resultBuffer;
}

/// Return the buffer type for a given Value (tensor) after bufferization.
BaseMemRefType BufferizationState::getBufferType(Value value) const {
  auto tensorType = value.getType().dyn_cast<TensorType>();
  assert(tensorType && "unexpected non-tensor type");

  if (auto toTensorOp = value.getDefiningOp<bufferization::ToTensorOp>())
    return toTensorOp.memref().getType().cast<BaseMemRefType>();

  return getMemRefType(tensorType, getOptions());
}

void bufferization::replaceOpWithBufferizedValues(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange values) {
  assert(values.size() == op->getNumResults() &&
         "expected one value per OpResult");
  OpBuilder::InsertionGuard g(rewriter);

  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (OpResult opResult : op->getOpResults()) {
    Value replacement = values[opResult.getResultNumber()];
    if (opResult.getType().isa<TensorType>()) {
      // The OpResult is a tensor. Such values are replaced with memrefs during
      // bufferization.
      assert((replacement.getType().isa<MemRefType>() ||
              replacement.getType().isa<UnrankedMemRefType>()) &&
             "tensor op result should be replaced with a memref value");
      // The existing uses of the OpResult still expect a tensor. Insert a
      // ToTensorOp. Throughout bufferization, this ToTensorOp will gradually
      // loose all of its users and eventually DCE away.
      rewriter.setInsertionPointAfter(op);
      replacement = rewriter.create<bufferization::ToTensorOp>(
          replacement.getLoc(), replacement);
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

AlwaysCopyAnalysisState::AlwaysCopyAnalysisState(
    const BufferizationOptions &options)
    : AnalysisState(options) {
  // Note: Allocations must be deallocated with a subsequent run of the buffer
  // deallocation pass.
  assert(!options.createDeallocs &&
         "cannot create deallocs with AlwaysCopyBufferizationState");
}

/// Return `true` if the given OpResult has been decided to bufferize inplace.
bool AlwaysCopyAnalysisState::isInPlace(OpOperand &opOperand) const {
  // OpOperands that bufferize to a memory write are out-of-place, i.e., an
  // alloc and copy is inserted.
  return !bufferizesToMemoryWrite(opOperand);
}

/// Return true if `v1` and `v2` bufferize to equivalent buffers.
bool AlwaysCopyAnalysisState::areEquivalentBufferizedValues(Value v1,
                                                            Value v2) const {
  // There is no analysis, so we do not know if the values are equivalent. The
  // conservative answer is "false".
  return false;
}

/// Return `true` if the given tensor has undefined contents.
bool AlwaysCopyAnalysisState::hasUndefinedContents(OpOperand *opOperand) const {
  // There is no analysis, so the conservative answer is "false".
  return false;
}

/// Return true if the given tensor (or an aliasing tensor) is yielded from
/// the containing block. Also include all aliasing tensors in the same block.
bool AlwaysCopyAnalysisState::isTensorYielded(Value tensor) const {
  // There is no analysis, so conservatively answer "true".
  return true;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

/// Create a memref allocation with the given type and dynamic extents.
FailureOr<Value> BufferizationOptions::createAlloc(OpBuilder &b, Location loc,
                                                   MemRefType type,
                                                   ValueRange dynShape) const {
  if (allocationFn)
    return (*allocationFn)(b, loc, type, dynShape, bufferAlignment);

  // Default bufferallocation via AllocOp.
  Value allocated = b.create<memref::AllocOp>(
      loc, type, dynShape, b.getI64IntegerAttr(bufferAlignment));
  return allocated;
}

/// Creates a memref deallocation. The given memref buffer must have been
/// allocated using `createAlloc`.
LogicalResult BufferizationOptions::createDealloc(OpBuilder &b, Location loc,
                                                  Value allocatedBuffer) const {
  if (deallocationFn)
    return (*deallocationFn)(b, loc, allocatedBuffer);

  // Default buffer deallocation via DeallocOp.
  b.create<memref::DeallocOp>(loc, allocatedBuffer);
  return success();
}

static MemRefType getContiguousMemRefType(ShapedType shapedType,
                                          Attribute memorySpace = {}) {
  MemRefLayoutAttrInterface layout = {};
  return MemRefType::get(shapedType.getShape(), shapedType.getElementType(),
                         layout, memorySpace);
}

/// Compute the type of the `memref` to use for allocating the buffer for
/// `shapedValue`. Also returns (by reference in `dynShape`), the value for the
/// dynamic dimensions in the returned `memref` type.
static MemRefType getAllocationTypeAndShape(OpBuilder &b, Location loc,
                                            Value shapedValue,
                                            SmallVectorImpl<Value> &dynShape) {
  MemRefType allocMemRefType =
      getContiguousMemRefType(shapedValue.getType().cast<ShapedType>());

  // Compute the dynamic part of the shape.
  bool reifiedShapes = false;
  if (auto rankedOp = dyn_cast_or_null<ReifyRankedShapedTypeOpInterface>(
          shapedValue.getDefiningOp())) {
    ReifiedRankedShapedTypeDims resultDims;
    if (succeeded(rankedOp.reifyResultShapes(b, resultDims))) {
      reifiedShapes = true;
      OpResult resultValue = shapedValue.dyn_cast<OpResult>();
      auto &shape = resultDims[resultValue.getResultNumber()];
      for (const auto &dim : enumerate(allocMemRefType.getShape()))
        if (ShapedType::isDynamic(dim.value()))
          dynShape.push_back(shape[dim.index()]);
    }
  }

  if (!reifiedShapes) {
    for (const auto &dim : enumerate(allocMemRefType.getShape()))
      if (ShapedType::isDynamic(dim.value())) {
        assert((shapedValue.getType().isa<UnrankedMemRefType>() ||
                shapedValue.getType().isa<MemRefType>()) &&
               "expected MemRef type");
        dynShape.push_back(
            b.create<memref::DimOp>(loc, shapedValue, dim.index()));
      }
  }

  return allocMemRefType;
}

static Value createBufferAllocation(OpBuilder &b, Location loc, MemRefType type,
                                    ValueRange dynShape, bool skipDealloc) {
  auto allocaOp = b.create<memref::AllocaOp>(loc, type, dynShape);
  allocaOp->setAttr(kBufferAllocationAttr, b.getUnitAttr());
  if (skipDealloc)
    allocaOp->setAttr(kSkipDeallocAttr, b.getUnitAttr());
  return allocaOp.getResult();
}

/// Create an allocation after `shapedValue.getDefiningOp` (or at the top of the
/// block in case of a bbArg).
FailureOr<Value> BufferizationState::createAlloc(OpBuilder &b, Location loc,
                                                 Value shapedValue,
                                                 Optional<bool> dealloc) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Compute allocation memref type.
  assert(shapedValue.getType().isa<ShapedType>());
  SmallVector<Value> dynShape;
  MemRefType allocMemRefType =
      getAllocationTypeAndShape(b, loc, shapedValue, dynShape);

  // Should be the buffer be deallocated again or should we let it leak?
  bool skipDealloc;
  if (dealloc) {
    skipDealloc = !dealloc.getValue();
  } else {
    assert(shapedValue.getType().isa<TensorType>() &&
           "must specify `dealloc` if non-tensor value is passed");
    // Buffer should be not be deallocated if deallocs are generally deactivated
    // or if the tensor is yielded from a block.
    skipDealloc = !getOptions().createDeallocs ||
                  getAnalysisState().isTensorYielded(shapedValue);
  }

  // Create the buffer allocation.
  return createBufferAllocation(b, loc, allocMemRefType, dynShape, skipDealloc);
}

/// Create a memory copy between two memref buffers.
LogicalResult BufferizationOptions::createMemCpy(OpBuilder &b, Location loc,
                                                 Value from, Value to) const {
  if (memCpyFn)
    return (*memCpyFn)(b, loc, from, to);

  b.create<memref::CopyOp>(loc, from, to);
  return success();
}

LogicalResult
bufferization::createAllocDeallocOps(Operation *op,
                                     const BufferizationOptions &options,
                                     bool onlyLeakingAllocs, bool *changed) {
  IRRewriter rewriter(op->getContext());
  if (changed)
    *changed = false;

  // Bufferization creates memref.alloca ops. After bufferization, these must be
  // rewritten to alloc/dealloc ops as specified in the bufferization options.
  WalkResult status = op->walk([&](memref::AllocaOp allocaOp) {
    // Ignore memref.alloca ops that were not created by the bufferization.
    if (!allocaOp->hasAttr(kBufferAllocationAttr))
      return WalkResult::skip();
    // If `onlyLeakingAllocs`, process only ops that are marked as
    // "skip dealloc".
    bool skipDealloc = allocaOp->hasAttr(kSkipDeallocAttr);
    if (onlyLeakingAllocs && !skipDealloc)
      return WalkResult::skip();

    // Create alloc.
    Block *block = allocaOp->getBlock();
    rewriter.setInsertionPoint(allocaOp);
    FailureOr<Value> alloc =
        options.createAlloc(rewriter, allocaOp->getLoc(), allocaOp.getType(),
                            allocaOp.dynamicSizes());
    if (failed(alloc))
      return WalkResult::interrupt();
    rewriter.replaceOp(allocaOp, *alloc);
    if (changed)
      *changed = true;

    // Stop here if the buffer should not be deallocated.
    if (skipDealloc)
      return WalkResult::advance();

    // Create dealloc.
    rewriter.setInsertionPoint(block->getTerminator());
    if (failed(options.createDealloc(rewriter, alloc->getLoc(), *alloc)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return success(!status.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

bool bufferization::isFunctionArgument(Value value) {
  auto bbArg = value.dyn_cast<BlockArgument>();
  if (!bbArg)
    return false;
  return isa<func::FuncOp>(bbArg.getOwner()->getParentOp());
}

BaseMemRefType bufferization::getMemRefType(TensorType tensorType,
                                            const BufferizationOptions &options,
                                            MemRefLayoutAttrInterface layout,
                                            Attribute memorySpace) {
  // Case 1: Unranked memref type.
  if (auto unrankedTensorType = tensorType.dyn_cast<UnrankedTensorType>()) {
    assert(!layout && "UnrankedTensorType cannot have a layout map");
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type with specified layout.
  auto rankedTensorType = tensorType.cast<RankedTensorType>();
  if (layout) {
    return MemRefType::get(rankedTensorType.getShape(),
                           rankedTensorType.getElementType(), layout,
                           memorySpace);
  }

  // Case 3: Configured with "fully dynamic layout maps".
  if (options.unknownTypeConversion ==
      BufferizationOptions::LayoutMapOption::FullyDynamicLayoutMap)
    return getMemRefTypeWithFullyDynamicLayout(tensorType, memorySpace);

  // Case 4: Configured with "static identity layout maps".
  if (options.unknownTypeConversion ==
      BufferizationOptions::LayoutMapOption::IdentityLayoutMap)
    return getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace);

  llvm_unreachable("InferLayoutMap is an invalid option");
}

BaseMemRefType
bufferization::getMemRefTypeWithFullyDynamicLayout(TensorType tensorType,
                                                   Attribute memorySpace) {
  // Case 1: Unranked memref type.
  if (auto unrankedTensorType = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type.
  auto rankedTensorType = tensorType.cast<RankedTensorType>();
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(rankedTensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, rankedTensorType.getContext());
  return MemRefType::get(rankedTensorType.getShape(),
                         rankedTensorType.getElementType(), stridedLayout,
                         memorySpace);
}

/// Return a MemRef type with a static identity layout (i.e., no layout map). If
/// the given tensor type is unranked, return an unranked MemRef type.
BaseMemRefType
bufferization::getMemRefTypeWithStaticIdentityLayout(TensorType tensorType,
                                                     Attribute memorySpace) {
  // Case 1: Unranked memref type.
  if (auto unrankedTensorType = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type.
  auto rankedTensorType = tensorType.cast<RankedTensorType>();
  MemRefLayoutAttrInterface layout = {};
  return MemRefType::get(rankedTensorType.getShape(),
                         rankedTensorType.getElementType(), layout,
                         memorySpace);
}
