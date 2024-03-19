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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// BufferizableOpInterface
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bufferization {

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.cpp.inc"

} // namespace bufferization
} // namespace mlir

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::bufferization::AnalysisState)

#define DEBUG_TYPE "bufferizable-op-interface"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X))

using namespace mlir;
using namespace bufferization;

static bool isRepetitiveRegion(Region *region,
                               const BufferizationOptions &options) {
  Operation *op = region->getParentOp();
  if (auto bufferizableOp = options.dynCastBufferizableOp(op))
    if (bufferizableOp.isRepetitiveRegion(region->getRegionNumber()))
      return true;
  return false;
}

Region *AnalysisState::getEnclosingRepetitiveRegion(
    Operation *op, const BufferizationOptions &options) {
  if (!op->getBlock())
    return nullptr;
  if (auto iter = enclosingRepetitiveRegionCache.find_as(op);
      iter != enclosingRepetitiveRegionCache.end())
    return iter->second;
  return enclosingRepetitiveRegionCache[op] =
             getEnclosingRepetitiveRegion(op->getBlock(), options);
}

Region *AnalysisState::getEnclosingRepetitiveRegion(
    Value value, const BufferizationOptions &options) {
  if (auto iter = enclosingRepetitiveRegionCache.find_as(value);
      iter != enclosingRepetitiveRegionCache.end())
    return iter->second;

  Region *region = value.getParentRegion();
  // Collect all visited regions since we only know the repetitive region we
  // want to map it to later on
  SmallVector<Region *> visitedRegions;
  while (region) {
    visitedRegions.push_back(region);
    if (isRepetitiveRegion(region, options))
      break;
    region = region->getParentRegion();
  }
  enclosingRepetitiveRegionCache[value] = region;
  for (Region *r : visitedRegions)
    enclosingRepetitiveRegionCache[r] = region;
  return region;
}

Region *AnalysisState::getEnclosingRepetitiveRegion(
    Block *block, const BufferizationOptions &options) {
  if (auto iter = enclosingRepetitiveRegionCache.find_as(block);
      iter != enclosingRepetitiveRegionCache.end())
    return iter->second;

  Region *region = block->getParent();
  Operation *op = nullptr;
  // Collect all visited regions since we only know the repetitive region we
  // want to map it to later on
  SmallVector<Region *> visitedRegions;
  do {
    op = region->getParentOp();
    if (isRepetitiveRegion(region, options))
      break;
  } while ((region = op->getParentRegion()));

  enclosingRepetitiveRegionCache[block] = region;
  for (Region *r : visitedRegions)
    enclosingRepetitiveRegionCache[r] = region;
  return region;
}

void AnalysisState::resetCache() { enclosingRepetitiveRegionCache.clear(); }

Region *bufferization::getNextEnclosingRepetitiveRegion(
    Region *region, const BufferizationOptions &options) {
  assert(isRepetitiveRegion(region, options) && "expected repetitive region");
  while ((region = region->getParentRegion())) {
    if (isRepetitiveRegion(region, options))
      break;
  }
  return region;
}

Region *bufferization::getParallelRegion(Region *region,
                                         const BufferizationOptions &options) {
  while (region) {
    auto bufferizableOp = options.dynCastBufferizableOp(region->getParentOp());
    if (bufferizableOp &&
        bufferizableOp.isParallelRegion(region->getRegionNumber())) {
      assert(isRepetitiveRegion(region, options) &&
             "expected that all parallel regions are also repetitive regions");
      return region;
    }
    region = region->getParentRegion();
  }
  return nullptr;
}

Operation *bufferization::getOwnerOfValue(Value value) {
  if (auto opResult = llvm::dyn_cast<OpResult>(value))
    return opResult.getDefiningOp();
  return llvm::cast<BlockArgument>(value).getOwner()->getParentOp();
}

/// Create an AllocTensorOp for the given shaped value. If `copy` is set, the
/// shaped value is copied. Otherwise, a tensor with undefined contents is
/// allocated.
FailureOr<Value> bufferization::allocateTensorForShapedValue(
    OpBuilder &b, Location loc, Value shapedValue,
    const BufferizationOptions &options, bool copy) {
  Value tensor;
  if (llvm::isa<RankedTensorType>(shapedValue.getType())) {
    tensor = shapedValue;
  } else if (llvm::isa<MemRefType>(shapedValue.getType())) {
    tensor = b.create<ToTensorOp>(loc, shapedValue);
  } else if (llvm::isa<UnrankedTensorType>(shapedValue.getType()) ||
             llvm::isa<UnrankedMemRefType>(shapedValue.getType())) {
    return getOwnerOfValue(shapedValue)
        ->emitError("copying of unranked tensors is not implemented");
  } else {
    llvm_unreachable("expected RankedTensorType or MemRefType");
  }
  RankedTensorType tensorType = llvm::cast<RankedTensorType>(tensor.getType());
  SmallVector<Value> dynamicSizes;
  if (!copy) {
    // Compute the dynamic part of the shape.
    // First try to query the shape via ReifyRankedShapedTypeOpInterface.
    bool reifiedShapes = false;
    if (llvm::isa<RankedTensorType>(shapedValue.getType()) &&
        llvm::isa<OpResult>(shapedValue)) {
      ReifiedRankedShapedTypeDims resultDims;
      if (succeeded(
              reifyResultShapes(b, shapedValue.getDefiningOp(), resultDims))) {
        reifiedShapes = true;
        auto &shape =
            resultDims[llvm::cast<OpResult>(shapedValue).getResultNumber()];
        for (const auto &dim : enumerate(tensorType.getShape()))
          if (ShapedType::isDynamic(dim.value()))
            dynamicSizes.push_back(shape[dim.index()].get<Value>());
      }
    }

    // If the shape could not be reified, create DimOps.
    if (!reifiedShapes)
      populateDynamicDimSizes(b, loc, tensor, dynamicSizes);
  }

  // Create AllocTensorOp.
  auto allocTensorOp = b.create<AllocTensorOp>(loc, tensorType, dynamicSizes,
                                               copy ? tensor : Value());

  // Add 'memory_space' attribute. Not needed if 'copy' operand is specified.
  if (copy)
    return allocTensorOp.getResult();
  FailureOr<BaseMemRefType> copyBufferType = getBufferType(tensor, options);
  if (failed(copyBufferType))
    return failure();
  Attribute memorySpace = copyBufferType->getMemorySpace();
  if (!memorySpace)
    memorySpace = b.getI64IntegerAttr(0);
  allocTensorOp.setMemorySpaceAttr(memorySpace);
  return allocTensorOp.getResult();
}

LogicalResult BufferizableOpInterface::resolveTensorOpOperandConflicts(
    RewriterBase &rewriter, const AnalysisState &state) {
  OpBuilder::InsertionGuard g(rewriter);
  Operation *op = getOperation();
  SmallVector<OpOperand *> outOfPlaceOpOperands;
  DenseSet<OpOperand *> copiedOpOperands;
  SmallVector<Value> outOfPlaceValues;
  DenseSet<Value> copiedOpValues;

  // Find all out-of-place OpOperands.
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (!llvm::isa<TensorType>(operandType))
      continue;
    if (state.isInPlace(opOperand))
      continue;
    if (llvm::isa<UnrankedTensorType>(operandType))
      return op->emitError("copying of unranked tensors is not implemented");

    AliasingValueList aliasingValues = state.getAliasingValues(opOperand);
    if (aliasingValues.getNumAliases() == 1 &&
        isa<OpResult>(aliasingValues.getAliases()[0].value) &&
        !state.bufferizesToMemoryWrite(opOperand) &&
        state.getAliasingOpOperands(aliasingValues.getAliases()[0].value)
                .getNumAliases() == 1 &&
        !isa<UnrankedTensorType>(
            aliasingValues.getAliases()[0].value.getType())) {
      // The op itself does not write but may create exactly one alias. Instead
      // of copying the OpOperand, copy the OpResult. The OpResult can sometimes
      // be smaller than the OpOperand (e.g., in the case of an extract_slice,
      // where the result is usually a smaller part of the source). Do not apply
      // this optimization if the OpResult is an unranked tensor (because those
      // cannot be copied at the moment).
      Value value = aliasingValues.getAliases()[0].value;
      outOfPlaceValues.push_back(value);
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpValues.insert(value);
    } else {
      // In all other cases, make a copy of the OpOperand.
      outOfPlaceOpOperands.push_back(&opOperand);
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpOperands.insert(&opOperand);
    }
  }

  // Insert copies of OpOperands.
  rewriter.setInsertionPoint(op);
  for (OpOperand *opOperand : outOfPlaceOpOperands) {
    FailureOr<Value> copy = allocateTensorForShapedValue(
        rewriter, op->getLoc(), opOperand->get(), state.getOptions(),
        copiedOpOperands.contains(opOperand));
    if (failed(copy))
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { opOperand->set(*copy); });
  }

  // Insert copies of Values.
  rewriter.setInsertionPointAfter(op);
  for (Value value : outOfPlaceValues) {
    FailureOr<Value> copy = allocateTensorForShapedValue(
        rewriter, op->getLoc(), value, state.getOptions(),
        copiedOpValues.count(value));
    if (failed(copy))
      return failure();
    SmallVector<OpOperand *> uses = llvm::to_vector(
        llvm::map_range(value.getUses(), [](OpOperand &use) { return &use; }));
    for (OpOperand *use : uses) {
      // Do not update the alloc_tensor op that we just created.
      if (use->getOwner() == copy->getDefiningOp())
        continue;
      // tensor.dim ops may have been created to be used as alloc_tensor op
      // dynamic extents. Do not update these either.
      if (isa<tensor::DimOp>(use->getOwner()))
        continue;
      rewriter.modifyOpInPlace(use->getOwner(), [&]() { use->set(*copy); });
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OpFilter
//===----------------------------------------------------------------------===//

bool OpFilter::isOpAllowed(Operation *op) const {
  // All other ops: Allow/disallow according to filter.
  bool isAllowed = !hasAllowRule();
  for (const Entry &entry : entries) {
    bool filterResult = entry.fn(op);
    switch (entry.type) {
    case Entry::ALLOW:
      isAllowed |= filterResult;
      break;
    case Entry::DENY:
      if (filterResult)
        // DENY filter matches. This op is no allowed. (Even if other ALLOW
        // filters may match.)
        return false;
    };
  }
  return isAllowed;
}

//===----------------------------------------------------------------------===//
// BufferizationOptions
//===----------------------------------------------------------------------===//

namespace {

/// Default function arg type converter: Use a fully dynamic layout map.
BaseMemRefType
defaultFunctionArgTypeConverter(TensorType type, Attribute memorySpace,
                                func::FuncOp funcOp,
                                const BufferizationOptions &options) {
  return getMemRefTypeWithFullyDynamicLayout(type, memorySpace);
}
/// Default unknown type converter: Use a fully dynamic layout map.
BaseMemRefType
defaultUnknownTypeConverter(Value value, Attribute memorySpace,
                            const BufferizationOptions &options) {
  return getMemRefTypeWithFullyDynamicLayout(
      llvm::cast<TensorType>(value.getType()), memorySpace);
}

} // namespace

// Default constructor for BufferizationOptions.
BufferizationOptions::BufferizationOptions()
    : functionArgTypeConverterFn(defaultFunctionArgTypeConverter),
      unknownTypeConverterFn(defaultUnknownTypeConverter) {}

bool BufferizationOptions::isOpAllowed(Operation *op) const {
  // Special case: If function boundary bufferization is deactivated, do not
  // allow ops that belong to the `func` dialect.
  bool isFuncBoundaryOp = isa_and_nonnull<func::FuncDialect>(op->getDialect());
  if (!bufferizeFunctionBoundaries && isFuncBoundaryOp)
    return false;

  return opFilter.isOpAllowed(op);
}

BufferizableOpInterface
BufferizationOptions::dynCastBufferizableOp(Operation *op) const {
  if (!isOpAllowed(op))
    return nullptr;
  auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op);
  if (!bufferizableOp)
    return nullptr;
  return bufferizableOp;
}

BufferizableOpInterface
BufferizationOptions::dynCastBufferizableOp(Value value) const {
  return dynCastBufferizableOp(getOwnerOfValue(value));
}

void BufferizationOptions::setFunctionBoundaryTypeConversion(
    LayoutMapOption layoutMapOption) {
  functionArgTypeConverterFn = [=](TensorType tensorType, Attribute memorySpace,
                                   func::FuncOp funcOp,
                                   const BufferizationOptions &options) {
    if (layoutMapOption == LayoutMapOption::IdentityLayoutMap)
      return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                  memorySpace);
    return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                              memorySpace);
  };
  inferFunctionResultLayout =
      layoutMapOption == LayoutMapOption::InferLayoutMap;
}

//===----------------------------------------------------------------------===//
// Helper functions for BufferizableOpInterface
//===----------------------------------------------------------------------===//

static void setInsertionPointAfter(OpBuilder &b, Value value) {
  if (auto bbArg = llvm::dyn_cast<BlockArgument>(value)) {
    b.setInsertionPointToStart(bbArg.getOwner());
  } else {
    b.setInsertionPointAfter(value.getDefiningOp());
  }
}

/// Determine which OpOperand* will alias with `value` if the op is bufferized
/// in place. Return all tensor OpOperand* if the op is not bufferizable.
AliasingOpOperandList AnalysisState::getAliasingOpOperands(Value value) const {
  if (Operation *op = getOwnerOfValue(value))
    if (auto bufferizableOp = getOptions().dynCastBufferizableOp(op))
      return bufferizableOp.getAliasingOpOperands(value, *this);

  // The op is not bufferizable.
  return detail::unknownGetAliasingOpOperands(value);
}

/// Determine which Values will alias with `opOperand` if the op is bufferized
/// in place. Return all tensor Values if the op is not bufferizable.
AliasingValueList AnalysisState::getAliasingValues(OpOperand &opOperand) const {
  if (auto bufferizableOp =
          getOptions().dynCastBufferizableOp(opOperand.getOwner()))
    return bufferizableOp.getAliasingValues(opOperand, *this);

  // The op is not bufferizable.
  return detail::unknownGetAliasingValues(opOperand);
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

bool AnalysisState::bufferizesToMemoryWrite(Value value) const {
  auto opResult = llvm::dyn_cast<OpResult>(value);
  if (!opResult)
    return true;
  auto bufferizableOp = getOptions().dynCastBufferizableOp(value);
  if (!bufferizableOp)
    return true;
  return bufferizableOp.resultBufferizesToMemoryWrite(opResult, *this);
}

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool AnalysisState::isValueRead(Value value) const {
  assert(llvm::isa<TensorType>(value.getType()) && "expected TensorType");
  SmallVector<OpOperand *> workingSet;
  DenseSet<OpOperand *> visited;
  for (OpOperand &use : value.getUses())
    workingSet.push_back(&use);

  while (!workingSet.empty()) {
    OpOperand *uMaybeReading = workingSet.pop_back_val();
    if (visited.contains(uMaybeReading))
      continue;
    visited.insert(uMaybeReading);

    // Skip over all ops that neither read nor write (but create an alias).
    if (bufferizesToAliasOnly(*uMaybeReading))
      for (AliasingValue alias : getAliasingValues(*uMaybeReading))
        for (OpOperand &use : alias.value.getUses())
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
    Value value, llvm::function_ref<bool(Value)> condition,
    TraversalConfig config) const {
  llvm::DenseSet<Value> visited;
  llvm::SetVector<Value> result, workingSet;
  workingSet.insert(value);

  while (!workingSet.empty()) {
    Value value = workingSet.pop_back_val();

    if (!config.revisitAlreadyVisitedValues && visited.contains(value)) {
      // Stop traversal if value was already visited.
      if (config.alwaysIncludeLeaves)
        result.insert(value);
      continue;
    }
    visited.insert(value);

    if (condition(value)) {
      result.insert(value);
      continue;
    }

    if (!config.followUnknownOps && !options.dynCastBufferizableOp(value)) {
      // Stop iterating if `followUnknownOps` is unset and the op is either
      // not bufferizable or excluded in the OpFilter.
      if (config.alwaysIncludeLeaves)
        result.insert(value);
      continue;
    }

    AliasingOpOperandList aliases = getAliasingOpOperands(value);
    if (aliases.getNumAliases() == 0) {
      // The traversal ends naturally if there are no more OpOperands that
      // could be followed.
      if (config.alwaysIncludeLeaves)
        result.insert(value);
      continue;
    }

    for (AliasingOpOperand a : aliases) {
      if (config.followEquivalentOnly &&
          a.relation != BufferRelation::Equivalent) {
        // Stop iterating if `followEquivalentOnly` is set but the alias is not
        // equivalent.
        if (config.alwaysIncludeLeaves)
          result.insert(value);
        continue;
      }

      if (config.followInPlaceOnly && !isInPlace(*a.opOperand)) {
        // Stop iterating if `followInPlaceOnly` is set but the alias is
        // out-of-place.
        if (config.alwaysIncludeLeaves)
          result.insert(value);
        continue;
      }

      if (config.followSameTypeOrCastsOnly &&
          a.opOperand->get().getType() != value.getType() &&
          !value.getDefiningOp<CastOpInterface>()) {
        // Stop iterating if `followSameTypeOrCastsOnly` is set but the alias is
        // has a different type and the op is not a cast.
        if (config.alwaysIncludeLeaves)
          result.insert(value);
        continue;
      }

      workingSet.insert(a.opOperand->get());
    }
  }

  return result;
}

// Find the values that define the contents of the given value.
llvm::SetVector<Value> AnalysisState::findDefinitions(Value value) const {
  TraversalConfig config;
  config.alwaysIncludeLeaves = false;
  return findValueInReverseUseDefChain(
      value, [&](Value v) { return this->bufferizesToMemoryWrite(v); }, config);
}

AnalysisState::AnalysisState(const BufferizationOptions &options)
    : AnalysisState(options, TypeID::get<AnalysisState>()) {}

AnalysisState::AnalysisState(const BufferizationOptions &options, TypeID type)
    : options(options), type(type) {
  for (const BufferizationOptions::AnalysisStateInitFn &fn :
       options.stateInitializers)
    fn(*this);
}

bool AnalysisState::canOmitTensorCopy(OpOperand &opOperand) const {
  // Do not copy if the tensor has undefined contents.
  if (hasUndefinedContents(&opOperand))
    return true;

  // Do not copy if the buffer of the tensor is entirely overwritten (with
  // values that do not depend on the old tensor).
  if (bufferizesToMemoryWrite(opOperand) && !bufferizesToMemoryRead(opOperand))
    return true;

  // Do not copy if the tensor is never read.
  AliasingValueList aliases = getAliasingValues(opOperand);
  if (!bufferizesToMemoryRead(opOperand) &&
      llvm::none_of(aliases,
                    [&](AliasingValue a) { return isValueRead(a.value); }))
    return true;

  // Default: Cannot omit the copy.
  return false;
}

bool AnalysisState::isInPlace(OpOperand &opOperand) const {
  // ToMemrefOps are always in-place.
  if (isa<ToMemrefOp>(opOperand.getOwner()))
    return true;

  // In the absence of analysis information, OpOperands that bufferize to a
  // memory write are out-of-place, i.e., an alloc and copy is inserted.
  return !bufferizesToMemoryWrite(opOperand);
}

bool AnalysisState::areEquivalentBufferizedValues(Value v1, Value v2) const {
  // In the absence of analysis information, we do not know if the values are
  // equivalent. The conservative answer is "false".
  return false;
}

bool AnalysisState::areAliasingBufferizedValues(Value v1, Value v2) const {
  // In the absence of analysis information, we do not know if the values may be
  // aliasing. The conservative answer is "true".
  return true;
}

bool AnalysisState::hasUndefinedContents(OpOperand *opOperand) const {
  // In the absence of analysis information, the conservative answer is "false".
  return false;
}

// bufferization.to_memref is not allowed to change the rank.
static void ensureToMemrefOpIsValid(Value tensor, Type memrefType) {
#ifndef NDEBUG
  auto rankedTensorType = llvm::dyn_cast<RankedTensorType>(tensor.getType());
  assert((!rankedTensorType || llvm::cast<MemRefType>(memrefType).getRank() ==
                                   rankedTensorType.getRank()) &&
         "to_memref would be invalid: mismatching ranks");
#endif
}

FailureOr<Value> bufferization::getBuffer(RewriterBase &rewriter, Value value,
                                          const BufferizationOptions &options) {
#ifndef NDEBUG
  auto tensorType = llvm::dyn_cast<TensorType>(value.getType());
  assert(tensorType && "unexpected non-tensor type");
#endif // NDEBUG

  // Replace "%t = to_tensor %m" with %m.
  if (auto toTensorOp = value.getDefiningOp<bufferization::ToTensorOp>())
    return toTensorOp.getMemref();

  // Insert to_memref op.
  OpBuilder::InsertionGuard g(rewriter);
  setInsertionPointAfter(rewriter, value);
  FailureOr<BaseMemRefType> memrefType = getBufferType(value, options);
  if (failed(memrefType))
    return failure();
  ensureToMemrefOpIsValid(value, *memrefType);
  return rewriter
      .create<bufferization::ToMemrefOp>(value.getLoc(), *memrefType, value)
      .getResult();
}

/// Return the buffer type for a given Value (tensor) after bufferization.
FailureOr<BaseMemRefType>
bufferization::getBufferType(Value value, const BufferizationOptions &options) {
  SmallVector<Value> invocationStack;
  return getBufferType(value, options, invocationStack);
}

/// Return the buffer type for a given Value (tensor) after bufferization.
FailureOr<BaseMemRefType>
bufferization::getBufferType(Value value, const BufferizationOptions &options,
                             SmallVector<Value> &invocationStack) {
  assert(llvm::isa<TensorType>(value.getType()) &&
         "unexpected non-tensor type");
  invocationStack.push_back(value);
  auto popFromStack =
      llvm::make_scope_exit([&]() { invocationStack.pop_back(); });

  // Try querying BufferizableOpInterface.
  Operation *op = getOwnerOfValue(value);
  auto bufferizableOp = options.dynCastBufferizableOp(op);
  if (bufferizableOp)
    return bufferizableOp.getBufferType(value, options, invocationStack);

  // Op is not bufferizable.
  auto memSpace =
      options.defaultMemorySpaceFn(value.getType().cast<TensorType>());
  if (!memSpace.has_value())
    return op->emitError("could not infer memory space");

  return getMemRefType(value, options, /*layout=*/{}, *memSpace);
}

bool bufferization::hasTensorSemantics(Operation *op) {
  if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
    return bufferizableOp.hasTensorSemantics();
  return detail::defaultHasTensorSemantics(op);
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
    if (llvm::isa<TensorType>(opResult.getType())) {
      // The OpResult is a tensor. Such values are replaced with memrefs during
      // bufferization.
      assert((llvm::isa<MemRefType>(replacement.getType()) ||
              llvm::isa<UnrankedMemRefType>(replacement.getType())) &&
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

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc insertion support.
//===----------------------------------------------------------------------===//

/// Create a memref allocation with the given type and dynamic extents.
FailureOr<Value> BufferizationOptions::createAlloc(OpBuilder &b, Location loc,
                                                   MemRefType type,
                                                   ValueRange dynShape) const {
  if (allocationFn)
    return (*allocationFn)(b, loc, type, dynShape, bufferAlignment);

  // Default bufferallocation via AllocOp.
  if (bufferAlignment != 0)
    return b
        .create<memref::AllocOp>(loc, type, dynShape,
                                 b.getI64IntegerAttr(bufferAlignment))
        .getResult();
  return b.create<memref::AllocOp>(loc, type, dynShape).getResult();
}

/// Create a memory copy between two memref buffers.
LogicalResult BufferizationOptions::createMemCpy(OpBuilder &b, Location loc,
                                                 Value from, Value to) const {
  if (memCpyFn)
    return (*memCpyFn)(b, loc, from, to);

  b.create<memref::CopyOp>(loc, from, to);
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization-specific IRMapping support with debugging.
//===----------------------------------------------------------------------===//

BaseMemRefType bufferization::getMemRefType(Value value,
                                            const BufferizationOptions &options,
                                            MemRefLayoutAttrInterface layout,
                                            Attribute memorySpace) {
  auto tensorType = llvm::cast<TensorType>(value.getType());

  // Case 1: Unranked memref type.
  if (auto unrankedTensorType =
          llvm::dyn_cast<UnrankedTensorType>(tensorType)) {
    assert(!layout && "UnrankedTensorType cannot have a layout map");
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type with specified layout.
  auto rankedTensorType = llvm::cast<RankedTensorType>(tensorType);
  if (layout) {
    return MemRefType::get(rankedTensorType.getShape(),
                           rankedTensorType.getElementType(), layout,
                           memorySpace);
  }

  return options.unknownTypeConverterFn(value, memorySpace, options);
}

BaseMemRefType
bufferization::getMemRefTypeWithFullyDynamicLayout(TensorType tensorType,
                                                   Attribute memorySpace) {
  // Case 1: Unranked memref type.
  if (auto unrankedTensorType =
          llvm::dyn_cast<UnrankedTensorType>(tensorType)) {
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type.
  auto rankedTensorType = llvm::cast<RankedTensorType>(tensorType);
  int64_t dynamicOffset = ShapedType::kDynamic;
  SmallVector<int64_t> dynamicStrides(rankedTensorType.getRank(),
                                      ShapedType::kDynamic);
  auto stridedLayout = StridedLayoutAttr::get(tensorType.getContext(),
                                              dynamicOffset, dynamicStrides);
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
  if (auto unrankedTensorType =
          llvm::dyn_cast<UnrankedTensorType>(tensorType)) {
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type.
  auto rankedTensorType = llvm::cast<RankedTensorType>(tensorType);
  MemRefLayoutAttrInterface layout = {};
  return MemRefType::get(rankedTensorType.getShape(),
                         rankedTensorType.getElementType(), layout,
                         memorySpace);
}

//===----------------------------------------------------------------------===//
// Default implementations of interface methods
//===----------------------------------------------------------------------===//

bool bufferization::detail::defaultResultBufferizesToMemoryWrite(
    OpResult opResult, const AnalysisState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(opResult.getDefiningOp());
  AliasingOpOperandList opOperands =
      bufferizableOp.getAliasingOpOperands(opResult, state);

  // Case 1: OpResults that have no aliasing OpOperand usually bufferize to
  // memory writes.
  if (opOperands.getAliases().empty())
    return true;

  // Case 2: If an aliasing OpOperand bufferizes to a memory write, the OpResult
  // may bufferize to a memory write.
  if (llvm::any_of(opOperands, [&](AliasingOpOperand alias) {
        return state.bufferizesToMemoryWrite(*alias.opOperand);
      }))
    return true;

  // Case 3: Check if a nested aliasing OpOperand value bufferizes to a memory
  // write. (Or: The reverse SSA use-def chain ends inside the reigon.) In that
  // case, the OpResult bufferizes to a memory write. E.g.:
  //
  // %0 = "some_writing_op" : tensor<?xf32>
  // %r = scf.if ... -> tensor<?xf32> {
  //   scf.yield %0 : tensor<?xf32>
  // } else {
  //   %1 = "another_writing_op"(%0) : tensor<?xf32>
  //   scf.yield %1 : tensor<?xf32>
  // }
  // "some_reading_op"(%r)
  //
  // %r bufferizes to a memory write because an aliasing OpOperand value (%1)
  // bufferizes to a memory write and the defining op is inside the scf.if.
  //
  // Note: This treatment of surrouding ops is useful for ops that have a
  // region but no OpOperand such as scf.if or scf.execute_region. It simplifies
  // the analysis considerably.
  //
  // "another_writing_op" in the above example should be able to bufferize
  // inplace in the absence of another read of %0. However, if the scf.if op
  // would not be considered a "write", the analysis would detect the
  // following conflict:
  //
  // * read = some_reading_op
  // * lastWrite = %0  (Note: The last write of %r would be a set: {%0, %1}.)
  // * conflictingWrite = %1
  //
  auto isMemoryWriteInsideOp = [&](Value v) {
    Operation *op = getOwnerOfValue(v);
    if (!opResult.getDefiningOp()->isAncestor(op))
      return false;
    return state.bufferizesToMemoryWrite(v);
  };
  TraversalConfig config;
  config.alwaysIncludeLeaves = false;
  for (AliasingOpOperand alias : opOperands) {
    if (!state
             .findValueInReverseUseDefChain(alias.opOperand->get(),
                                            isMemoryWriteInsideOp, config)
             .empty())
      return true;
  }
  return false;
}

// Compute the AliasingOpOperandList for a given Value based on
// getAliasingValues.
AliasingOpOperandList bufferization::detail::defaultGetAliasingOpOperands(
    Value value, const AnalysisState &state) {
  Operation *op = getOwnerOfValue(value);
  SmallVector<AliasingOpOperand> result;
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (!llvm::isa<TensorType>(opOperand.get().getType()))
      continue;
    AliasingValueList aliasingValues = state.getAliasingValues(opOperand);
    for (const auto &it : aliasingValues)
      if (it.value == value)
        result.emplace_back(&opOperand, it.relation, it.isDefinite);
  }
  return AliasingOpOperandList(std::move(result));
}

FailureOr<BaseMemRefType> bufferization::detail::defaultGetBufferType(
    Value value, const BufferizationOptions &options,
    SmallVector<Value> &invocationStack) {
  assert(llvm::isa<TensorType>(value.getType()) && "expected tensor type");

  // No further analysis is possible for a block argument.
  if (llvm::isa<BlockArgument>(value))
    return bufferization::getMemRefType(value, options);

  // Value is an OpResult.
  Operation *op = getOwnerOfValue(value);
  auto opResult = llvm::cast<OpResult>(value);
  AnalysisState state(options);
  AliasingOpOperandList aliases = state.getAliasingOpOperands(opResult);
  if (aliases.getNumAliases() > 0 &&
      aliases.getAliases()[0].relation == BufferRelation::Equivalent) {
    // If the OpResult has an equivalent OpOperand, both OpResult and
    // OpOperand bufferize to the exact same buffer type.
    Value equivalentOperand = aliases.getAliases().front().opOperand->get();
    return getBufferType(equivalentOperand, options, invocationStack);
  }

  // If we do not know the memory space and there is no default memory space,
  // report a failure.
  auto memSpace =
      options.defaultMemorySpaceFn(value.getType().cast<TensorType>());
  if (!memSpace.has_value())
    return op->emitError("could not infer memory space");

  return getMemRefType(value, options, /*layout=*/{}, *memSpace);
}

bool bufferization::detail::defaultIsRepetitiveRegion(
    BufferizableOpInterface bufferizableOp, unsigned index) {
  assert(index < bufferizableOp->getNumRegions() && "invalid region index");
  auto regionInterface =
      dyn_cast<RegionBranchOpInterface>(bufferizableOp.getOperation());
  if (!regionInterface)
    return false;
  return regionInterface.isRepetitiveRegion(index);
}

AliasingOpOperandList
bufferization::detail::unknownGetAliasingOpOperands(Value value) {
  // TODO: Take into account successor blocks.
  // No aliasing in case of non-entry blocks.
  if (auto bbArg = dyn_cast<BlockArgument>(value))
    if (bbArg.getOwner() != &bbArg.getOwner()->getParent()->getBlocks().front())
      return {};

  // Unknown op: Conservatively assume that each OpResult may alias with every
  // OpOperand. In addition, each block argument of an entry block may alias
  // with every OpOperand.
  AliasingOpOperandList r;
  for (OpOperand &operand : value.getDefiningOp()->getOpOperands())
    if (isa<TensorType>(operand.get().getType()))
      r.addAlias({&operand, BufferRelation::Unknown, /*isDefinite=*/false});
  return r;
}

AliasingValueList
bufferization::detail::unknownGetAliasingValues(OpOperand &opOperand) {
  // TODO: Take into account successor blocks.
  // Unknown op: Conservatively assume that each OpResult may alias with every
  // OpOperand. In addition, each block argument of an entry block may alias
  // with every OpOperand.
  AliasingValueList r;
  for (OpResult result : opOperand.getOwner()->getOpResults())
    if (llvm::isa<TensorType>(result.getType()))
      r.addAlias({result, BufferRelation::Unknown, /*isDefinite=*/false});
  for (Region &region : opOperand.getOwner()->getRegions())
    if (!region.getBlocks().empty())
      for (BlockArgument bbArg : region.getBlocks().front().getArguments())
        if (bbArg.getType().isa<TensorType>())
          r.addAlias({bbArg, BufferRelation::Unknown, /*isDefinite=*/false});
  return r;
}

bool bufferization::detail::defaultHasTensorSemantics(Operation *op) {
  auto isaTensor = [](Type t) { return isa<TensorType>(t); };
  bool hasTensorBlockArgument = any_of(op->getRegions(), [&](Region &r) {
    return any_of(r.getBlocks(), [&](Block &b) {
      return any_of(b.getArguments(), [&](BlockArgument bbArg) {
        return isaTensor(bbArg.getType());
      });
    });
  });
  if (hasTensorBlockArgument)
    return true;

  if (any_of(op->getResultTypes(), isaTensor))
    return true;
  return any_of(op->getOperandTypes(), isaTensor);
}
