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
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

using namespace mlir;
using namespace linalg::comprehensive_bufferize;

//===----------------------------------------------------------------------===//
// BufferizationOptions
//===----------------------------------------------------------------------===//

/// Default allocation function that is used by the comprehensive bufferization
/// pass. The default currently creates a ranked memref using `memref.alloc`.
static Optional<Value> defaultAllocationFn(OpBuilder &b, Location loc,
                                           MemRefType type,
                                           ArrayRef<Value> dynShape) {
  Value allocated = b.create<memref::AllocOp>(
      loc, type, dynShape, b.getI64IntegerAttr(kBufferAlignments));
  return allocated;
}

/// Default deallocation function that is used by the comprehensive
/// bufferization pass. It expects to recieve back the value called from the
/// `defaultAllocationFn`.
static void defaultDeallocationFn(OpBuilder &b, Location loc,
                                  Value allocatedBuffer) {
  b.create<memref::DeallocOp>(loc, allocatedBuffer);
}

/// Default memory copy function that is used by the comprehensive bufferization
/// pass. Creates a `memref.copy` op.
static void defaultMemCpyFn(OpBuilder &b, Location loc, Value from, Value to) {
  b.create<memref::CopyOp>(loc, from, to);
}

std::unique_ptr<AllocationCallbacks>
mlir::linalg::comprehensive_bufferize::defaultAllocationCallbacks() {
  return std::make_unique<AllocationCallbacks>(
      defaultAllocationFn, defaultDeallocationFn, defaultMemCpyFn);
}

// Default constructor for BufferizationOptions that sets all allocation
// callbacks to their default functions.
BufferizationOptions::BufferizationOptions()
    : allocationFns(defaultAllocationCallbacks()) {}

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

  // Set up alias sets for OpResults that must bufferize in-place. This should
  // be done before making any other bufferization decisions.
  rootOp->walk([&](BufferizableOpInterface bufferizableOp) {
    for (OpResult opResult : bufferizableOp->getOpResults()) {
      if (opResult.getType().isa<TensorType>())
        if (bufferizableOp.mustBufferizeInPlace(opResult)) {
          SmallVector<OpOperand *> operands =
              bufferizableOp.getAliasingOpOperand(opResult);
          assert(!operands.empty() &&
                 "expected that OpResult has aliasing OpOperand");
          for (OpOperand *operand : operands)
            aliasInfo.unionSets(operand->get(), opResult);
          markInPlace(opResult);
        }
    }
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

bool BufferizationAliasInfo::bufferizesToWritableMemory(Value v) const {
  return bufferizeToWritableMemory.count(v) > 0;
}

/// Specify that the value is known to bufferize to writable memory.
void BufferizationAliasInfo::setBufferizesToWritableMemory(Value v) {
  bufferizeToWritableMemory.insert(v);
}

/// Return `true` if a value was marked as in-place bufferized.
bool BufferizationAliasInfo::isInPlace(OpResult opResult) const {
  bool inplace = inplaceBufferized.contains(opResult);
#ifndef NDEBUG
  if (inplace) {
    auto bufferizableOp =
        dyn_cast<BufferizableOpInterface>(opResult.getDefiningOp());
    assert(bufferizableOp &&
           "expected that in-place bufferized op is bufferizable");
    SmallVector<OpOperand *> operands =
        bufferizableOp.getAliasingOpOperand(opResult);
    for (OpOperand *operand : operands)
      assert(areAliasingBufferizedValues(operand->get(), opResult) &&
             "expected that in-place bufferized OpResult aliases with "
             "aliasing OpOperand");
  }
#endif // NDEBUG
  return inplace;
}

/// Set the inPlace bufferization spec to true.
void BufferizationAliasInfo::bufferizeInPlace(OpResult result,
                                              OpOperand &operand) {
  LLVM_DEBUG(llvm::dbgs() << "bufferizeInPlace: ");
  LLVM_DEBUG(result.print(llvm::dbgs()));

  markInPlace(result);
  aliasInfo.unionSets(result, operand.get());
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpResult result) {
  LLVM_DEBUG(llvm::dbgs() << "bufferizeOutOfPlace: ");
  LLVM_DEBUG(result.print(llvm::dbgs()));

  if (inplaceBufferized.contains(result))
    inplaceBufferized.erase(result);
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
mlir::linalg::comprehensive_bufferize::getAliasingOpOperand(OpResult result) {
  if (Operation *op = result.getDefiningOp())
    if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
      return bufferizableOp.getAliasingOpOperand(result);
  return {};
}

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. Return an empty OpResult if the op is not bufferizable.
OpResult mlir::linalg::comprehensive_bufferize::getAliasingOpResult(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.getAliasingOpResult(opOperand);
  return OpResult();
}

/// Return true if `opOperand` bufferizes to a memory read. Return `true` if the
/// op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::bufferizesToMemoryRead(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryRead(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` bufferizes to a memory write. Return
/// `true` if the op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::bufferizesToMemoryWrite(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryWrite(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` does neither read nor write but bufferizes to an
/// alias. Return false if the op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::bufferizesToAliasOnly(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToAliasOnly(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return false.
  return false;
}

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool mlir::linalg::comprehensive_bufferize::isValueRead(Value value) {
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
llvm::SetVector<Value>
mlir::linalg::comprehensive_bufferize::findValueInReverseUseDefChain(
    Value value, std::function<bool(Value)> condition) {
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
    if (opOperands.empty()) {
      result.insert(value);
      continue;
    }

    for (OpOperand *o : opOperands)
      workingSet.insert(o->get());
  }

  return result;
}

// Find the Value of the last preceding write of a given Value.
Value mlir::linalg::comprehensive_bufferize::findLastPrecedingWrite(
    Value value) {
  SetVector<Value> result =
      findValueInReverseUseDefChain(value, [](Value value) {
        Operation *op = value.getDefiningOp();
        if (!op)
          return true;
        auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op);
        if (!bufferizableOp)
          return true;
        return bufferizableOp.isMemoryWrite(value.cast<OpResult>());
      });

  // To simplify the analysis, `scf.if` ops are considered memory writes. There
  // are currently no other ops where one OpResult may alias with multiple
  // OpOperands. Therefore, this function should return exactly one result at
  // the moment.
  assert(result.size() == 1 && "expected exactly one result");
  return result.front();
}

/// Return the result buffer (memref) for a given OpResult (tensor). Allocate
/// a new buffer and copy over data from the existing buffer if out-of-place
/// bufferization is necessary.
Value mlir::linalg::comprehensive_bufferize::BufferizationState::
    getResultBuffer(OpResult result) {
  OpBuilder::InsertionGuard guard(builder);
  Operation *op = result.getOwner();
  SmallVector<OpOperand *> aliasingOperands = getAliasingOpOperand(result);
  assert(!aliasingOperands.empty() && "could not get aliasing OpOperand");
  OpOperand *opOperand = aliasingOperands.front();
  Value operand = opOperand->get();
  Value operandBuffer = lookupBuffer(operand);
  // Make sure that all OpOperands are the same buffer. If this is not the case,
  // we would have to materialize a memref value.
  // TODO: Should be looking for checking for "equivalent buffers" instead of
  // operator== here, but equivalent buffers for scf.if yield values are not
  // set up yet.
  if (aliasingOperands.size() > 1 &&
      !llvm::all_of(aliasingOperands, [&](OpOperand *o) {
        return lookupBuffer(o->get()) == operandBuffer;
      })) {
    op->emitError("result buffer is ambiguous");
    return Value();
  }

  // If bufferizing out-of-place, allocate a new buffer.
  if (!aliasInfo.isInPlace(result)) {
    // Ops with multiple aliasing operands can currently not bufferize
    // out-of-place.
    assert(
        aliasingOperands.size() == 1 &&
        "ops with multiple aliasing OpOperands cannot bufferize out-of-place");
    Location loc = op->getLoc();
    // Move insertion point right after `operandBuffer`. That is where the
    // allocation should be inserted (in the absence of allocation hoisting).
    setInsertionPointAfter(builder, operandBuffer);
    // Allocate the result buffer.
    Value resultBuffer = createAllocDeallocFn(builder, loc, operandBuffer);
    bool skipCopy = false;
    // Do not copy if the last preceding write of `operand` is an op that does
    // not write (skipping ops that merely create aliases). E.g., InitTensorOp.
    // Note: If `findLastPrecedingWrite` reaches the end of the reverse SSA
    // use-def chain, it returns that value, regardless of whether it is a
    // memory write or not.
    Value lastWrite = findLastPrecedingWrite(operand);
    if (auto bufferizableOp =
            lastWrite.getDefiningOp<BufferizableOpInterface>())
      if (!bufferizableOp.isMemoryWrite(lastWrite.cast<OpResult>()))
        skipCopy = true;
    // Do not copy if the copied data is never read.
    if (!isValueRead(result))
      skipCopy = true;
    // Do not copy if this op does not read the data, but writes it.
    if (bufferizesToMemoryWrite(*opOperand) &&
        !bufferizesToMemoryRead(*opOperand))
      skipCopy = true;
    if (!skipCopy) {
      // The copy happens right before the op that is bufferized.
      builder.setInsertionPoint(op);
      options.allocationFns->memCpyFn(builder, loc, operandBuffer,
                                      resultBuffer);
    }
    return resultBuffer;
  }

  // Bufferizing in-place. No need to allocate a new buffer.
  return operandBuffer;
}

LogicalResult
mlir::linalg::comprehensive_bufferize::bufferize(Region *region,
                                                 BufferizationState &state) {
  for (Block &block : *region)
    if (failed(bufferize(&block, state)))
      return failure();
  return success();
}

LogicalResult
mlir::linalg::comprehensive_bufferize::bufferize(Block *block,
                                                 BufferizationState &state) {
  for (Operation &op : *block)
    if (failed(bufferize(&op, state)))
      return failure();
  return success();
}

LogicalResult
mlir::linalg::comprehensive_bufferize::bufferize(Operation *op,
                                                 BufferizationState &state) {
  OpBuilder &b = state.builder;

  // Check if op has tensor results or operands.
  auto isaTensor = [](Type t) { return t.isa<TensorType>(); };
  bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
  bool hasTensorOperand = any_of(op->getOperandTypes(), isaTensor);

  // No tensor results or operands: Simply bufferize all nested ops.
  if (!hasTensorResult && !hasTensorOperand) {
    for (Region &region : op->getRegions())
      if (failed(bufferize(&region, state)))
        return failure();
    return success();
  }

  // Bufferize using `BufferizableOpInterface`. Interface implementations are
  // responsible for bufferizing nested ops.
  if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op)) {
    b.setInsertionPoint(op);
    return bufferizableOp.bufferize(b, state);
  }

  // `op` is an unbufferizable tensor op.
  if (!state.options.allowUnknownOps)
    return op->emitError() << "unsupported op with tensors";

  // Replace all OpOperands with "to-tensor casted" bufferized values.
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get().getType().isa<TensorType>() &&
        state.isMapped(operand.get())) {
      b.setInsertionPoint(op);
      Value toTensorOp = b.create<bufferization::ToTensorOp>(
          op->getLoc(), state.lookupBuffer(operand.get()));
      operand.set(toTensorOp);
    }
  }

  for (Region &region : op->getRegions())
    if (failed(bufferize(&region, state)))
      return failure();

  return success();
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

/// Create an Allocop/DeAllocOp pair, where the AllocOp is after
/// `shapedValue.getDefiningOp` (or at the top of the block in case of a
/// bbArg) and the DeallocOp is at the end of the block.
Value mlir::linalg::comprehensive_bufferize::BufferizationState::
    createAllocDeallocFn(OpBuilder &b, Location loc, Value shapedValue) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // 1. Create memory allocation.
  assert(shapedValue.getType().isa<ShapedType>());
  MemRefType memRefType = shapedValue.getType().dyn_cast<MemRefType>();
  SmallVector<Value> dynShape;
  // Note: getAllocationTypeAndShape also sets the insertion point.
  MemRefType allocMemRefType =
      getAllocationTypeAndShape(b, loc, shapedValue, dynShape);
  Optional<Value> allocated =
      options.allocationFns->allocationFn(b, loc, allocMemRefType, dynShape);
  // TODO: For now just assert the value is returned. Eventually need to
  // error-propagate.
  assert(allocated && "allocation failed");
  Value casted = allocated.getValue();
  if (memRefType && memRefType != allocMemRefType) {
    casted = b.create<memref::CastOp>(loc, memRefType, allocated.getValue());
    aliasInfo.insertNewBufferEquivalence(casted, allocated.getValue());
  }

  // 2. Create memory deallocation.
  b.setInsertionPoint(allocated.getValue().getParentBlock()->getTerminator());
  options.allocationFns->deallocationFn(b, loc, allocated.getValue());
  return casted;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

/// Wrapper for better debugging.
void mlir::linalg::comprehensive_bufferize::BufferizationState::mapBuffer(
    ValueRange tensors, ValueRange buffers) {
  assert(!tensors.empty() && "unexpected empty tensors");
#ifndef NDEBUG
  for (Value tensor : tensors) {
    assert(tensor && "unexpected empty tensor");
    assert(tensor.getType().isa<TensorType>() && "unexpected non-tensor type");
  }
  for (Value buffer : buffers) {
    assert(buffer && "unexpected empty buffer");
    assert((buffer.getType().isa<MemRefType>() ||
            buffer.getType().isa<UnrankedMemRefType>()) &&
           "expected that tensor is mapped to memref");
  }
#endif // NDEBUG
  return mapping.map(tensors, buffers);
}

/// Wrapper for better debugging.
void mlir::linalg::comprehensive_bufferize::BufferizationState::mapBuffer(
    Value tensor, Value buffer) {
  assert(tensor && "unexpected empty tensor");
  assert(tensor.getType().isa<TensorType>() && "unexpected non-tensor type");
  assert(buffer && "unexpected empty buffer");
  assert((buffer.getType().isa<MemRefType>() ||
          buffer.getType().isa<UnrankedMemRefType>()) &&
         "expected that tensor is mapped to memref");
  return mapping.map(tensor, buffer);
}

/// Wrapper for better debugging.
Value mlir::linalg::comprehensive_bufferize::BufferizationState::lookupBuffer(
    Value tensor) {
  // TODO: if key comes from bbArg, forward.
  assert(tensor.getType().isa<TensorType>() && "unexpected non-tensor type");
  Value buffer = mapping.lookupOrNull(tensor);

  if (!buffer) {
    if (options.allowUnknownOps) {
      // `tensor` was not bufferized yet. This should never happen with
      // bufferizable ops.
      assert(!tensor.getDefiningOp<BufferizableOpInterface>() &&
             "tensor is not mapped");
      // Insert to_memref op.
      OpBuilder b(tensor.getContext());
      setInsertionPointAfter(b, tensor);
      return b.create<bufferization::ToMemrefOp>(
          tensor.getLoc(),
          getDynamicMemRefType(tensor.getType().cast<RankedTensorType>()),
          tensor);
    }

    // Dump tensor for easier debugging.
    tensor.dump();
    llvm_unreachable("tensor is not mapped");
    return Value();
  }

  assert((buffer.getType().isa<MemRefType>() ||
          buffer.getType().isa<UnrankedMemRefType>()) &&
         "expected that tensor is mapped to memref");
  return buffer;
}

bool mlir::linalg::comprehensive_bufferize::BufferizationState::isMapped(
    Value value) const {
  return mapping.contains(value);
}

void mlir::linalg::comprehensive_bufferize::BufferizationState::markOpObsolete(
    Operation *op) {
  obsoleteOps.push_back(op);
}

void mlir::linalg::comprehensive_bufferize::BufferizationState::
    eraseObsoleteOps() {
  for (Operation *op : obsoleteOps)
    op->erase();
  obsoleteOps.clear();
}

MemRefType mlir::linalg::comprehensive_bufferize::getContiguousMemRefType(
    ShapedType shapedType, MemRefLayoutAttrInterface layout,
    Attribute memorySpace) {
  return MemRefType::get(shapedType.getShape(), shapedType.getElementType(),
                         layout, memorySpace);
}

Type mlir::linalg::comprehensive_bufferize::getContiguousOrUnrankedMemRefType(
    Type type, MemRefLayoutAttrInterface layout, Attribute memorySpace) {
  if (type.isa<RankedTensorType, MemRefType>())
    return getContiguousMemRefType(type.cast<ShapedType>(), layout,
                                   memorySpace);
  assert(!layout && "expected empty layout with UnrankedMemRefType");
  return UnrankedMemRefType::get(getElementTypeOrSelf(type), memorySpace);
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
