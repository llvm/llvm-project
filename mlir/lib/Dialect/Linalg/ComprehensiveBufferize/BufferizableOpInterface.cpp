//===- BufferizableOpInterface.cpp - Comprehensive Bufferize --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.cpp.inc"

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#define DEBUG_TYPE "bufferizable-op-interface"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X))

using namespace mlir;
using namespace linalg::comprehensive_bufferize;

//===----------------------------------------------------------------------===//
// BufferizationOptions
//===----------------------------------------------------------------------===//

// Default constructor for BufferizationOptions.
BufferizationOptions::BufferizationOptions() {}

BufferizableOpInterface mlir::linalg::comprehensive_bufferize::
    BufferizationOptions::dynCastBufferizableOp(Operation *op) const {
  if (isOpAllowed(op))
    return dyn_cast<BufferizableOpInterface>(op);
  return nullptr;
}

BufferizableOpInterface mlir::linalg::comprehensive_bufferize::
    BufferizationOptions::dynCastBufferizableOp(Value value) const {
  if (auto bufferizableOp = value.getDefiningOp<BufferizableOpInterface>())
    if (isOpAllowed(bufferizableOp.getOperation()))
      return bufferizableOp;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// BufferizationAliasInfo
//===----------------------------------------------------------------------===//

BufferizationAliasInfo::BufferizationAliasInfo(Operation *rootOp) {
  rootOp->walk([&](Operation *op) {
    for (Value v : op->getResults())
      if (v.getType().isa<TensorType>())
        createAliasInfoEntry(v);
    for (Region &r : op->getRegions())
      for (Block &b : r.getBlocks())
        for (auto bbArg : b.getArguments())
          if (bbArg.getType().isa<TensorType>())
            createAliasInfoEntry(bbArg);
  });
}

/// Add a new entry for `v` in the `aliasInfo` and `equivalentInfo`. In the
/// beginning the alias and equivalence sets only contain `v` itself.
void BufferizationAliasInfo::createAliasInfoEntry(Value v) {
  aliasInfo.insert(v);
  equivalentInfo.insert(v);
}

/// Insert an info entry for `newValue` and merge its alias set with that of
/// `alias`.
void BufferizationAliasInfo::insertNewBufferAlias(Value newValue, Value alias) {
  createAliasInfoEntry(newValue);
  aliasInfo.unionSets(newValue, alias);
}

/// Insert an info entry for `newValue` and merge its alias set with that of
/// `alias`. Additionally, merge their equivalence classes.
void BufferizationAliasInfo::insertNewBufferEquivalence(Value newValue,
                                                        Value alias) {
  insertNewBufferAlias(newValue, alias);
  equivalentInfo.unionSets(newValue, alias);
}

/// Return `true` if a value was marked as in-place bufferized.
bool BufferizationAliasInfo::isInPlace(OpOperand &operand) const {
  return inplaceBufferized.contains(&operand);
}

/// Set the inPlace bufferization spec to true.
void BufferizationAliasInfo::bufferizeInPlace(OpOperand &operand,
                                              BufferizationState &state) {
  markInPlace(operand);
  if (OpResult result = state.getAliasingOpResult(operand))
    aliasInfo.unionSets(result, operand.get());
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpOperand &operand) {
  assert(!inplaceBufferized.contains(&operand) &&
         "OpOperand was already decided to bufferize inplace");
}

/// Apply `fun` to all the members of the equivalence class of `v`.
void BufferizationAliasInfo::applyOnEquivalenceClass(
    Value v, function_ref<void(Value)> fun) const {
  auto leaderIt = equivalentInfo.findLeader(v);
  for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
       ++mit) {
    fun(*mit);
  }
}

/// Apply `fun` to all aliases of `v`.
void BufferizationAliasInfo::applyOnAliases(
    Value v, function_ref<void(Value)> fun) const {
  auto leaderIt = aliasInfo.findLeader(v);
  for (auto mit = leaderIt, meit = aliasInfo.member_end(); mit != meit; ++mit) {
    fun(*mit);
  }
}

BufferizationAliasInfo::EquivalenceClassRangeType
BufferizationAliasInfo::getAliases(Value v) const {
  DenseSet<Value> res;
  auto it = aliasInfo.findValue(aliasInfo.getLeaderValue(v));
  for (auto mit = aliasInfo.member_begin(it), meit = aliasInfo.member_end();
       mit != meit; ++mit) {
    res.insert(static_cast<Value>(*mit));
  }
  return BufferizationAliasInfo::EquivalenceClassRangeType(
      aliasInfo.member_begin(it), aliasInfo.member_end());
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
mlir::linalg::comprehensive_bufferize::BufferizationState::getAliasingOpOperand(
    OpResult result) const {
  if (Operation *op = result.getDefiningOp())
    if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
      return bufferizableOp.getAliasingOpOperand(result, *this);
  return {};
}

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. Return an empty OpResult if the op is not bufferizable.
OpResult
mlir::linalg::comprehensive_bufferize::BufferizationState::getAliasingOpResult(
    OpOperand &opOperand) const {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.getAliasingOpResult(opOperand, *this);
  return OpResult();
}

/// Return true if `opOperand` bufferizes to a memory read. Return `true` if the
/// op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::BufferizationState::
    bufferizesToMemoryRead(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryRead(opOperand, *this);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` bufferizes to a memory write. Return
/// `true` if the op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::BufferizationState::
    bufferizesToMemoryWrite(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryWrite(opOperand, *this);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` does neither read nor write but bufferizes to an
/// alias. Return false if the op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::BufferizationState::
    bufferizesToAliasOnly(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToAliasOnly(opOperand, *this);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return false.
  return false;
}

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool mlir::linalg::comprehensive_bufferize::BufferizationState::isValueRead(
    Value value) const {
  assert(value.getType().isa<TensorType>() && "expected TensorType");
  SmallVector<OpOperand *> workingSet;
  for (OpOperand &use : value.getUses())
    workingSet.push_back(&use);

  while (!workingSet.empty()) {
    OpOperand *uMaybeReading = workingSet.pop_back_val();
    // Skip over all ops that neither read nor write (but create an alias).
    if (bufferizesToAliasOnly(*uMaybeReading))
      for (OpOperand &use : getAliasingOpResult(*uMaybeReading).getUses())
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
llvm::SetVector<Value> mlir::linalg::comprehensive_bufferize::
    BufferizationState::findValueInReverseUseDefChain(
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
llvm::SetVector<Value> mlir::linalg::comprehensive_bufferize::
    BufferizationState::findLastPrecedingWrite(Value value) const {
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

mlir::linalg::comprehensive_bufferize::BufferizationState::BufferizationState(
    const BufferizationOptions &options)
    : options(options) {}

mlir::linalg::comprehensive_bufferize::AnalysisBufferizationState::
    AnalysisBufferizationState(Operation *op,
                               const AnalysisBufferizationOptions &options)
    : BufferizationState(options), aliasInfo(op) {
  // Set up alias sets for OpResults that must bufferize in-place. This should
  // be done before making any other bufferization decisions.
  op->walk([&](BufferizableOpInterface bufferizableOp) {
    if (!options.isOpAllowed(bufferizableOp))
      return WalkResult::skip();
    for (OpOperand &opOperand : bufferizableOp->getOpOperands()) {
      if (opOperand.get().getType().isa<TensorType>())
        if (bufferizableOp.mustBufferizeInPlace(opOperand, *this)) {
          if (OpResult opResult =
                  bufferizableOp.getAliasingOpResult(opOperand, *this))
            aliasInfo.unionAliasSets(opOperand.get(), opResult);
          aliasInfo.markInPlace(opOperand);
        }
    }
    return WalkResult::advance();
  });
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

static Value lookupBuffer(RewriterBase &rewriter, Value tensor) {
  assert(tensor.getType().isa<TensorType>() && "unexpected non-tensor type");

  // Replace "%t = to_tensor %m" with %m.
  if (auto toTensorOp = tensor.getDefiningOp<bufferization::ToTensorOp>())
    return toTensorOp.memref();

  // Insert to_memref op.
  OpBuilder::InsertionGuard g(rewriter);
  setInsertionPointAfter(rewriter, tensor);
  Type memrefType;
  if (auto rankedTensorType = tensor.getType().dyn_cast<RankedTensorType>()) {
    memrefType = getDynamicMemRefType(rankedTensorType);
  } else {
    memrefType = getUnrankedMemRefType(
        tensor.getType().cast<TensorType>().getElementType());
  }
  ensureToMemrefOpIsValid(tensor, memrefType);
  return rewriter.create<bufferization::ToMemrefOp>(tensor.getLoc(), memrefType,
                                                    tensor);
}

/// Return the result buffer (memref) for a given OpResult (tensor). Allocate
/// a new buffer and copy over data from the existing buffer if out-of-place
/// bufferization is necessary.
FailureOr<Value>
mlir::linalg::comprehensive_bufferize::BufferizationState::getBuffer(
    RewriterBase &rewriter, OpOperand &opOperand, bool forceInPlace,
    Optional<Operation *> customCopyInsertionPoint) const {
  OpBuilder::InsertionGuard guard(rewriter);
  Operation *op = opOperand.getOwner();
  Location loc = op->getLoc();
  Value operand = opOperand.get();
  Value operandBuffer = lookupBuffer(rewriter, operand);

  if (forceInPlace || isInPlace(opOperand))
    return operandBuffer;

  // Bufferizing out-of-place: Allocate a new buffer.
  // Move insertion point right after `operandBuffer`. That is where the
  // allocation should be inserted (in the absence of allocation hoisting).
  setInsertionPointAfter(rewriter, operandBuffer);
  // Allocate the result buffer.
  FailureOr<Value> resultBuffer = createAlloc(rewriter, loc, operandBuffer,
                                              options.createDeallocs, options);
  if (failed(resultBuffer))
    return failure();
  // Do not copy if the last preceding writes of `operand` are ops that do
  // not write (skipping ops that merely create aliases). E.g., InitTensorOp.
  // Note: If `findLastPrecedingWrite` reaches the end of the reverse SSA
  // use-def chain, it returns that value, regardless of whether it is a
  // memory write or not.
  SetVector<Value> lastWrites = findLastPrecedingWrite(operand);
  if (llvm::none_of(lastWrites, [&](Value lastWrite) {
        if (auto bufferizableOp = options.dynCastBufferizableOp(lastWrite))
          return bufferizableOp.isMemoryWrite(lastWrite.cast<OpResult>(),
                                              *this);
        return true;
      }))
    return resultBuffer;
  // Do not copy if the copied data is never read.
  OpResult aliasingOpResult = getAliasingOpResult(opOperand);
  if (aliasingOpResult && !bufferizesToMemoryRead(opOperand) &&
      !isValueRead(aliasingOpResult))
    return resultBuffer;
  // Do not copy if this op does not read the data, but writes it.
  if (bufferizesToMemoryWrite(opOperand) && !bufferizesToMemoryRead(opOperand))
    return resultBuffer;

  if (customCopyInsertionPoint) {
    rewriter.setInsertionPoint(*customCopyInsertionPoint);
  } else {
    // The copy happens right before the op that is bufferized.
    rewriter.setInsertionPoint(op);
  }
  if (failed(
          createMemCpy(rewriter, loc, operandBuffer, *resultBuffer, options)))
    return failure();

  return resultBuffer;
}

void mlir::linalg::comprehensive_bufferize::replaceOpWithBufferizedValues(
    RewriterBase &rewriter, Operation *op, ValueRange values) {
  OpBuilder::InsertionGuard g(rewriter);

  // Replace all OpResults with the given values.
  for (OpResult opResult : op->getOpResults()) {
    // Skip OpResult if it has no uses.
    if (opResult.getUses().empty())
      continue;

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
      setInsertionPointAfter(rewriter, replacement);
      replacement = rewriter.create<bufferization::ToTensorOp>(
          replacement.getLoc(), replacement);
    }
    opResult.replaceAllUsesWith(replacement);
  }

  rewriter.eraseOp(op);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

/// Move the insertion point of the given builder to the beginning of a
/// surrounding block as much as possible, while not crossing any allocation
/// hoisting barriers.
static void moveInsertionPointToAllocationHoistingBarrier(OpBuilder &b) {
  Operation *op = b.getInsertionBlock()->getParentOp();
  while (op) {
    if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
      if (bufferizableOp.isAllocationHoistingBarrier())
        break;
    op = op->getParentOp();
  }

  if (!op) {
    // No allocation hoisting barrier found. Hoist to FuncOp.
    op = b.getInsertionBlock()->getParentOp();
    if (!isa<FuncOp>(op))
      op = op->getParentOfType<FuncOp>();
    assert(op && "could not find enclosing FuncOp");
  }

  // TODO: Handle cases where allocation hoisting barrier has more than one
  // region or block.
  assert(op->getNumRegions() == 1 &&
         "allocation hoisting barriers with >1 regions not supported");
  assert(op->getRegion(0).getBlocks().size() == 1 &&
         "allocation hoisting barriers with >1 blocks not supported");
  b.setInsertionPointToStart(&(op->getRegion(0).front()));
}

/// Compute the type of the `memref` to use for allocating the buffer for
/// `shapedValue`. Also returns (by reference in `dynShape`), the value for the
/// dynamic dimensions in the returned `memref` type. The function may also set
/// the insertion point to an earlier location, where the allocation should
/// happen ("allocation hoisting").
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

  // If the buffer is statically shaped, try to hoist it to the first enclosing
  // parallel region.
  // TODO: also hoist in the dynamic case. For now this relies on subsequent
  // calls to LICM and buffer hoisting which will most likely not succeed.
  // TODO: when packing, allocate a static bounding box which will enable more
  // hoisting.
  if (dynShape.empty())
    moveInsertionPointToAllocationHoistingBarrier(b);

  return allocMemRefType;
}

/// Create an AllocOp/DeallocOp pair, where the AllocOp is after
/// `shapedValue.getDefiningOp` (or at the top of the block in case of a
/// bbArg) and the DeallocOp is at the end of the block.
FailureOr<Value> mlir::linalg::comprehensive_bufferize::createAlloc(
    OpBuilder &b, Location loc, Value shapedValue, bool deallocMemref,
    const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // 1. Create memory allocation.
  assert(shapedValue.getType().isa<ShapedType>());
  MemRefType memRefType = shapedValue.getType().dyn_cast<MemRefType>();
  SmallVector<Value> dynShape;
  // Note: getAllocationTypeAndShape also sets the insertion point.
  MemRefType allocMemRefType =
      getAllocationTypeAndShape(b, loc, shapedValue, dynShape);
  FailureOr<Value> allocated =
      createAlloc(b, loc, allocMemRefType, dynShape, options);
  if (failed(allocated))
    return failure();
  Value casted = allocated.getValue();
  if (memRefType && memRefType != allocMemRefType) {
    assert(memref::CastOp::areCastCompatible(allocated.getValue().getType(),
                                             memRefType) &&
           "createAlloc: cast incompatible");
    casted = b.create<memref::CastOp>(loc, memRefType, allocated.getValue());
  }

  if (deallocMemref) {
    // 2. Create memory deallocation.
    b.setInsertionPoint(allocated.getValue().getParentBlock()->getTerminator());
    if (failed(createDealloc(b, loc, allocated.getValue(), options)))
      return failure();
  }

  return casted;
}

/// Create a memref allocation.
FailureOr<Value> mlir::linalg::comprehensive_bufferize::createAlloc(
    OpBuilder &b, Location loc, MemRefType type, ArrayRef<Value> dynShape,
    const BufferizationOptions &options) {
  if (options.allocationFn)
    return (*options.allocationFn)(b, loc, type, dynShape);

  // Default bufferallocation via AllocOp.
  Value allocated = b.create<memref::AllocOp>(
      loc, type, dynShape, b.getI64IntegerAttr(kBufferAlignments));
  return allocated;
}

/// Create a memref deallocation.
LogicalResult mlir::linalg::comprehensive_bufferize::createDealloc(
    OpBuilder &b, Location loc, Value allocatedBuffer,
    const BufferizationOptions &options) {
  if (options.deallocationFn)
    return (*options.deallocationFn)(b, loc, allocatedBuffer);

  // Default buffer deallocation via DeallocOp.
  b.create<memref::DeallocOp>(loc, allocatedBuffer);
  return success();
}

/// Create a memory copy between two memref buffers.
LogicalResult mlir::linalg::comprehensive_bufferize::createMemCpy(
    OpBuilder &b, Location loc, Value from, Value to,
    const BufferizationOptions &options) {
  if (options.memCpyFn)
    return (*options.memCpyFn)(b, loc, from, to);

  b.create<memref::CopyOp>(loc, from, to);
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

bool mlir::linalg::comprehensive_bufferize::isFunctionArgument(Value value) {
  auto bbArg = value.dyn_cast<BlockArgument>();
  if (!bbArg)
    return false;
  return isa<FuncOp>(bbArg.getOwner()->getParentOp());
}

bool mlir::linalg::comprehensive_bufferize::AnalysisBufferizationState::
    isInPlace(OpOperand &opOperand) const {
  return aliasInfo.isInPlace(opOperand);
}

bool mlir::linalg::comprehensive_bufferize::AnalysisBufferizationState::
    areEquivalentBufferizedValues(Value v1, Value v2) const {
  return aliasInfo.areEquivalentBufferizedValues(v1, v2);
}

MemRefType mlir::linalg::comprehensive_bufferize::getContiguousMemRefType(
    ShapedType shapedType, MemRefLayoutAttrInterface layout,
    Attribute memorySpace) {
  return MemRefType::get(shapedType.getShape(), shapedType.getElementType(),
                         layout, memorySpace);
}

UnrankedMemRefType mlir::linalg::comprehensive_bufferize::getUnrankedMemRefType(
    Type elementType, Attribute memorySpace) {
  return UnrankedMemRefType::get(elementType, memorySpace);
}

MemRefType mlir::linalg::comprehensive_bufferize::getDynamicMemRefType(
    RankedTensorType tensorType, unsigned addressSpace) {
  // TODO: address space decisions to connect with the actual alloc.
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(tensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, tensorType.getContext());
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         stridedLayout, addressSpace);
}
