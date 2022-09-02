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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// BufferizableOpInterface
//===----------------------------------------------------------------------===//

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

Operation *bufferization::getOwnerOfValue(Value value) {
  if (auto opResult = value.dyn_cast<OpResult>())
    return opResult.getDefiningOp();
  return value.cast<BlockArgument>().getOwner()->getParentOp();
}

bool bufferization::allocationDoesNotEscape(OpResult opResult) {
#ifndef NDEBUG
  auto bufferizableOp = opResult.getDefiningOp<BufferizableOpInterface>();
  assert(bufferizableOp && bufferizableOp.bufferizesToAllocation(opResult) &&
         "expected op that bufferizes to an allocation");
#endif // NDEBUG

  Operation *op = opResult.getDefiningOp();
  // If there is no 'escape' attribute, we cannot say for sure.
  if (!op->hasAttr(BufferizationDialect::kEscapeAttrName))
    return false;
  auto attr =
      op->getAttrOfType<ArrayAttr>(BufferizationDialect::kEscapeAttrName);
  return !attr[opResult.getResultNumber()].cast<BoolAttr>().getValue();
}

/// Create an AllocTensorOp for the given shaped value. If `copy` is set, the
/// shaped value is copied. Otherwise, a tensor with undefined contents is
/// allocated.
FailureOr<Value> bufferization::allocateTensorForShapedValue(
    OpBuilder &b, Location loc, Value shapedValue, bool escape,
    const BufferizationOptions &options, bool copy) {
  Value tensor;
  if (shapedValue.getType().isa<RankedTensorType>()) {
    tensor = shapedValue;
  } else if (shapedValue.getType().isa<MemRefType>()) {
    tensor = b.create<ToTensorOp>(loc, shapedValue);
  } else {
    llvm_unreachable("expected RankedTensorType or MemRefType");
  }
  RankedTensorType tensorType = tensor.getType().cast<RankedTensorType>();
  SmallVector<Value> dynamicSizes;
  if (!copy) {
    // Compute the dynamic part of the shape.
    // First try to query the shape via ReifyRankedShapedTypeOpInterface.
    bool reifiedShapes = false;
    if (shapedValue.getType().isa<RankedTensorType>() &&
        shapedValue.isa<OpResult>()) {
      if (auto rankedOp = dyn_cast_or_null<ReifyRankedShapedTypeOpInterface>(
              shapedValue.getDefiningOp())) {
        ReifiedRankedShapedTypeDims resultDims;
        if (succeeded(rankedOp.reifyResultShapes(b, resultDims))) {
          reifiedShapes = true;
          auto &shape =
              resultDims[shapedValue.cast<OpResult>().getResultNumber()];
          for (const auto &dim : enumerate(tensorType.getShape()))
            if (ShapedType::isDynamic(dim.value()))
              dynamicSizes.push_back(shape[dim.index()]);
        }
      }
    }

    // If the shape could not be reified, create DimOps.
    if (!reifiedShapes)
      populateDynamicDimSizes(b, loc, tensor, dynamicSizes);
  }

  // Create AllocTensorOp.
  auto allocTensorOp = b.create<AllocTensorOp>(loc, tensorType, dynamicSizes,
                                               copy ? tensor : Value());
  allocTensorOp->setAttr(BufferizationDialect::kEscapeAttrName,
                         b.getBoolArrayAttr({escape}));

  // Add 'memory_space' attribute. Not needed if 'copy' operand is specified.
  if (copy)
    return allocTensorOp.getResult();
  FailureOr<BaseMemRefType> copyBufferType = getBufferType(tensor, options);
  if (failed(copyBufferType))
    return failure();
  allocTensorOp.setMemorySpaceAttr(
      b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/false),
                       copyBufferType->getMemorySpaceAsInt()));
  return allocTensorOp.getResult();
}

LogicalResult BufferizableOpInterface::resolveTensorOpOperandConflicts(
    RewriterBase &rewriter, const AnalysisState &state) {
  OpBuilder::InsertionGuard g(rewriter);
  Operation *op = getOperation();
  SmallVector<OpOperand *> outOfPlaceOpOperands;
  DenseSet<OpOperand *> copiedOpOperands;
  DenseSet<OpOperand *> escapingOpOperandCopies;
  SmallVector<OpResult> outOfPlaceOpResults;
  DenseSet<OpResult> copiedOpResults;
  DenseSet<OpResult> escapingOpResultCopies;

  // Find all out-of-place OpOperands.
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (!operandType.isa<TensorType>())
      continue;
    if (state.isInPlace(opOperand))
      continue;
    if (operandType.isa<UnrankedTensorType>())
      return op->emitError("copies of unranked tensors are not supported");

    SmallVector<OpResult> aliasingOpResults =
        state.getAliasingOpResult(opOperand);
    // Is the result yielded from a block? Or are deallocations turned off
    // entirely? In either case, mark the allocation as "escaping", so that it
    // will not be deallocated.
    bool escape = !state.getOptions().createDeallocs ||
                  llvm::any_of(aliasingOpResults, [&](Value v) {
                    return state.isTensorYielded(v);
                  });

    if (aliasingOpResults.size() == 1 &&
        !state.bufferizesToMemoryWrite(opOperand) &&
        state.getAliasingOpOperand(aliasingOpResults.front()).size() == 1) {
      // The op itself does not write but may create exactly one alias. Instead
      // of copying the OpOperand, copy the OpResult. The OpResult can sometimes
      // be smaller than the OpOperand (e.g., in the case of an extract_slice,
      // where the result is usually a smaller part of the source).
      outOfPlaceOpResults.push_back(aliasingOpResults.front());
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpResults.insert(aliasingOpResults.front());
      if (escape)
        escapingOpResultCopies.insert(aliasingOpResults.front());
    } else {
      // In all other cases, make a copy of the OpOperand.
      outOfPlaceOpOperands.push_back(&opOperand);
      if (!state.canOmitTensorCopy(opOperand))
        copiedOpOperands.insert(&opOperand);
      if (escape)
        escapingOpOperandCopies.insert(&opOperand);
    }
  }

  // Insert copies of OpOperands.
  rewriter.setInsertionPoint(op);
  for (OpOperand *opOperand : outOfPlaceOpOperands) {
    FailureOr<Value> copy = allocateTensorForShapedValue(
        rewriter, op->getLoc(), opOperand->get(),
        escapingOpOperandCopies.contains(opOperand), state.getOptions(),
        copiedOpOperands.contains(opOperand));
    if (failed(copy))
      return failure();
    rewriter.updateRootInPlace(op, [&]() { opOperand->set(*copy); });
  }

  // Insert copies of OpResults.
  rewriter.setInsertionPointAfter(op);
  for (OpResult opResult : outOfPlaceOpResults) {
    FailureOr<Value> copy = allocateTensorForShapedValue(
        rewriter, op->getLoc(), opResult,
        escapingOpResultCopies.contains(opResult), state.getOptions(),
        copiedOpResults.count(opResult));
    if (failed(copy))
      return failure();
    SmallVector<OpOperand *> uses = llvm::to_vector(llvm::map_range(
        opResult.getUses(), [](OpOperand &use) { return &use; }));
    for (OpOperand *use : uses) {
      // Do not update the alloc_tensor op that we just created.
      if (use->getOwner() != copy->getDefiningOp())
        rewriter.updateRootInPlace(use->getOwner(), [&]() { use->set(*copy); });
    }
  }

  return success();
}

bool bufferization::shouldDeallocateOpResult(
    OpResult opResult, const BufferizationOptions &options) {
  Operation *op = opResult.getOwner();
  assert(options.dynCastBufferizableOp(op).bufferizesToAllocation(opResult) &&
         "expected that op allocates");

  AnalysisState analysisState(options);
  if (op->hasAttr(BufferizationDialect::kEscapeAttrName)) {
    // AllocTensorOp has one result.
    ArrayAttr escapeAttr =
        op->getAttr(BufferizationDialect::kEscapeAttrName).cast<ArrayAttr>();
    return !escapeAttr[0].cast<BoolAttr>().getValue();
  }

  // No "escape" annotation found.
  if (options.createDeallocs) {
    // Perform an ad-hoc analysis.
    return !analysisState.isTensorYielded(opResult);
  }

  return false;
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

/// Default unknown type converter: Use a fully dynamic layout map.
static BaseMemRefType
defaultUnknownTypeConverter(Value value, unsigned memorySpace,
                            const BufferizationOptions &options) {
  return getMemRefTypeWithFullyDynamicLayout(value.getType().cast<TensorType>(),
                                             memorySpace);
}

// Default constructor for BufferizationOptions.
BufferizationOptions::BufferizationOptions()
    : unknownTypeConverterFn(defaultUnknownTypeConverter) {}

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

bool AnalysisState::canOmitTensorCopy(OpOperand &opOperand) const {
  // Do not copy if the tensor has undefined contents.
  if (hasUndefinedContents(&opOperand))
    return true;

  // Do not copy if the buffer of the tensor is entirely overwritten (with
  // values that do not depend on the old tensor).
  if (bufferizesToMemoryWrite(opOperand) && !bufferizesToMemoryRead(opOperand))
    return true;

  // Do not copy if the tensor is never read.
  SmallVector<OpResult> aliasingOpResults = getAliasingOpResult(opOperand);
  if (!bufferizesToMemoryRead(opOperand) &&
      llvm::none_of(aliasingOpResults,
                    [&](OpResult opResult) { return isValueRead(opResult); }))
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

bool AnalysisState::isTensorYielded(Value tensor) const {
  // In the absence of analysis information, the conservative answer is "true".
  if (!tensor.getDefiningOp<AllocTensorOp>())
    return true;

  // For AllocTensorOp results, we can do better: They do not alias with any
  // preceding value, so we can follow SSA use-def chains and do a simple
  // analysis.
  SmallVector<OpOperand *> worklist;
  for (OpOperand &use : tensor.getUses())
    worklist.push_back(&use);

  while (!worklist.empty()) {
    OpOperand *operand = worklist.pop_back_val();
    Operation *op = operand->getOwner();

    // If the op is not bufferizable, we can safely assume that the value is not
    // yielded. (When bufferizing that op, it must handle such cases.)
    if (!options.dynCastBufferizableOp(op))
      continue;

    // We cannot analyze through ToMemrefOps, so we have to conservatively
    // assume that the value is yielded.
    if (isa<ToMemrefOp>(op))
      return true;

    // Check if the op is returning/yielding.
    if (isRegionReturnLike(op))
      return true;

    // Add all aliasing OpResults to the worklist.
    // Note: In the absence of detailed analysis information (e.g., there may be
    // no function call analysis information), this `getAliasingOpResult` is
    // conservative and may report additional OpResults as potentially aliasing.
    for (OpResult opResult : getAliasingOpResult(*operand))
      for (OpOperand &use : opResult.getUses())
        worklist.push_back(&use);
  }

  // No ReturnLike op found: The value is not yielded.
  return false;
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

FailureOr<Value> bufferization::getBuffer(RewriterBase &rewriter, Value value,
                                          const BufferizationOptions &options) {
#ifndef NDEBUG
  auto tensorType = value.getType().dyn_cast<TensorType>();
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

FailureOr<BaseMemRefType> bufferization::detail::defaultGetBufferType(
    Value value, const BufferizationOptions &options,
    const DenseMap<Value, BaseMemRefType> &fixedTypes) {
  assert(value.getType().isa<TensorType>() && "expected tensor type");

  // No further analysis is possible for a block argument.
  if (value.isa<BlockArgument>())
    return bufferization::getMemRefType(value, options);

  // Value is an OpResult.
  Operation *op = getOwnerOfValue(value);
  auto opResult = value.cast<OpResult>();
  auto bufferizableOp = cast<BufferizableOpInterface>(op);
  AnalysisState state(options);
  auto aliasingOperands = bufferizableOp.getAliasingOpOperand(opResult, state);
  if (!aliasingOperands.empty() &&
      bufferizableOp.bufferRelation(opResult, state) ==
          BufferRelation::Equivalent) {
    // If the OpResult has an equivalent OpOperand, both OpResult and
    // OpOperand bufferize to the exact same buffer type.
    Value equivalentOperand = aliasingOperands.front()->get();
    return getBufferType(equivalentOperand, options, fixedTypes);
  }

  // If we do not know the memory space and there is no default memory space,
  // report a failure.
  if (!options.defaultMemorySpace.has_value())
    return op->emitError("could not infer memory space");

  return getMemRefType(value, options, /*layout=*/{},
                       *options.defaultMemorySpace);
}

/// Return the buffer type for a given Value (tensor) after bufferization.
FailureOr<BaseMemRefType>
bufferization::getBufferType(Value value, const BufferizationOptions &options) {
  DenseMap<Value, BaseMemRefType> fixedTypes;
  return getBufferType(value, options, fixedTypes);
}

/// Return the buffer type for a given Value (tensor) after bufferization.
FailureOr<BaseMemRefType> bufferization::getBufferType(
    Value value, const BufferizationOptions &options,
    const DenseMap<Value, BaseMemRefType> &fixedTypes) {
  assert(value.getType().isa<TensorType>() && "unexpected non-tensor type");

  // If the `value` is in `fixedTypes`, return the mapped type.
  const auto &it = fixedTypes.find(value);
  if (it != fixedTypes.end())
    return it->second;

  // Try querying BufferizableOpInterface.
  Operation *op = getOwnerOfValue(value);
  auto bufferizableOp = options.dynCastBufferizableOp(op);
  if (bufferizableOp)
    return bufferizableOp.getBufferType(value, options, fixedTypes);

  // Op is not bufferizable.
  if (!options.defaultMemorySpace.has_value())
    return op->emitError("could not infer memory space");

  return getMemRefType(value, options, /*layout=*/{},
                       *options.defaultMemorySpace);
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
  if (bufferAlignment != 0)
    return b
        .create<memref::AllocOp>(loc, type, dynShape,
                                 b.getI64IntegerAttr(bufferAlignment))
        .getResult();
  return b.create<memref::AllocOp>(loc, type, dynShape).getResult();
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

/// Create a memory copy between two memref buffers.
LogicalResult BufferizationOptions::createMemCpy(OpBuilder &b, Location loc,
                                                 Value from, Value to) const {
  if (memCpyFn)
    return (*memCpyFn)(b, loc, from, to);

  b.create<memref::CopyOp>(loc, from, to);
  return success();
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

BaseMemRefType bufferization::getMemRefType(Value value,
                                            const BufferizationOptions &options,
                                            MemRefLayoutAttrInterface layout,
                                            unsigned memorySpace) {
  auto tensorType = value.getType().cast<TensorType>();
  auto memorySpaceAttr = IntegerAttr::get(
      IntegerType::get(tensorType.getContext(), 64), memorySpace);

  // Case 1: Unranked memref type.
  if (auto unrankedTensorType = tensorType.dyn_cast<UnrankedTensorType>()) {
    assert(!layout && "UnrankedTensorType cannot have a layout map");
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpaceAttr);
  }

  // Case 2: Ranked memref type with specified layout.
  auto rankedTensorType = tensorType.cast<RankedTensorType>();
  if (layout) {
    return MemRefType::get(rankedTensorType.getShape(),
                           rankedTensorType.getElementType(), layout,
                           memorySpaceAttr);
  }

  return options.unknownTypeConverterFn(value, memorySpace, options);
}

BaseMemRefType
bufferization::getMemRefTypeWithFullyDynamicLayout(TensorType tensorType,
                                                   unsigned memorySpace) {
  // Case 1: Unranked memref type.
  if (auto unrankedTensorType = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type.
  auto memorySpaceAttr = IntegerAttr::get(
      IntegerType::get(tensorType.getContext(), 64), memorySpace);
  auto rankedTensorType = tensorType.cast<RankedTensorType>();
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(rankedTensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, rankedTensorType.getContext());
  return MemRefType::get(rankedTensorType.getShape(),
                         rankedTensorType.getElementType(), stridedLayout,
                         memorySpaceAttr);
}

/// Return a MemRef type with a static identity layout (i.e., no layout map). If
/// the given tensor type is unranked, return an unranked MemRef type.
BaseMemRefType
bufferization::getMemRefTypeWithStaticIdentityLayout(TensorType tensorType,
                                                     unsigned memorySpace) {
  // Case 1: Unranked memref type.
  if (auto unrankedTensorType = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedMemRefType::get(unrankedTensorType.getElementType(),
                                   memorySpace);
  }

  // Case 2: Ranked memref type.
  auto rankedTensorType = tensorType.cast<RankedTensorType>();
  auto memorySpaceAttr = IntegerAttr::get(
      IntegerType::get(tensorType.getContext(), 64), memorySpace);
  MemRefLayoutAttrInterface layout = {};
  return MemRefType::get(rankedTensorType.getShape(),
                         rankedTensorType.getElementType(), layout,
                         memorySpaceAttr);
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
