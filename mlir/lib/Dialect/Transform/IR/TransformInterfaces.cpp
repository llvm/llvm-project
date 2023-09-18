//===- TransformInterfaces.cpp - Transform Dialect Interfaces -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "transform-dialect"
#define DEBUG_TYPE_FULL "transform-dialect-full"
#define DEBUG_PRINT_AFTER_ALL "transform-dialect-print-top-level-after-all"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X))
#define FULL_LDBG(X) DEBUG_WITH_TYPE(DEBUG_TYPE_FULL, (DBGS() << (X)))

using namespace mlir;

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//

constexpr const Value transform::TransformState::kTopLevelValue;

transform::TransformState::TransformState(
    Region *region, Operation *payloadRoot,
    const RaggedArray<MappedValue> &extraMappings,
    const TransformOptions &options)
    : topLevel(payloadRoot), options(options) {
  topLevelMappedValues.reserve(extraMappings.size());
  for (ArrayRef<MappedValue> mapping : extraMappings)
    topLevelMappedValues.push_back(mapping);

  auto result =
      mappings.insert(std::make_pair(region, std::make_unique<Mappings>()));
  assert(result.second && "the region scope is already present");
  (void)result;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  regionStack.push_back(region);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
}

Operation *transform::TransformState::getTopLevel() const { return topLevel; }

ArrayRef<Operation *>
transform::TransformState::getPayloadOpsView(Value value) const {
  const TransformOpMapping &operationMapping = getMapping(value).direct;
  auto iter = operationMapping.find(value);
  assert(
      iter != operationMapping.end() &&
      "cannot find mapping for payload handle (param/value handle provided?)");
  return iter->getSecond();
}

ArrayRef<Attribute> transform::TransformState::getParams(Value value) const {
  const ParamMapping &mapping = getMapping(value).params;
  auto iter = mapping.find(value);
  assert(iter != mapping.end() && "cannot find mapping for param handle "
                                  "(operation/value handle provided?)");
  return iter->getSecond();
}

ArrayRef<Value>
transform::TransformState::getPayloadValues(Value handleValue) const {
  const ValueMapping &mapping = getMapping(handleValue).values;
  auto iter = mapping.find(handleValue);
  assert(iter != mapping.end() && "cannot find mapping for value handle "
                                  "(param/operation handle provided?)");
  return iter->getSecond();
}

LogicalResult transform::TransformState::getHandlesForPayloadOp(
    Operation *op, SmallVectorImpl<Value> &handles,
    bool includeOutOfScope) const {
  bool found = false;
  for (const auto &[region, mapping] : llvm::reverse(mappings)) {
    auto iterator = mapping->reverse.find(op);
    if (iterator != mapping->reverse.end()) {
      llvm::append_range(handles, iterator->getSecond());
      found = true;
    }
    // Stop looking when reaching a region that is isolated from above.
    if (!includeOutOfScope &&
        region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
  }

  return success(found);
}

LogicalResult transform::TransformState::getHandlesForPayloadValue(
    Value payloadValue, SmallVectorImpl<Value> &handles,
    bool includeOutOfScope) const {
  bool found = false;
  for (const auto &[region, mapping] : llvm::reverse(mappings)) {
    auto iterator = mapping->reverseValues.find(payloadValue);
    if (iterator != mapping->reverseValues.end()) {
      llvm::append_range(handles, iterator->getSecond());
      found = true;
    }
    // Stop looking when reaching a region that is isolated from above.
    if (!includeOutOfScope &&
        region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
  }

  return success(found);
}

/// Given a list of MappedValues, cast them to the value kind implied by the
/// interface of the handle type, and dispatch to one of the callbacks.
static DiagnosedSilenceableFailure dispatchMappedValues(
    Value handle, ArrayRef<transform::MappedValue> values,
    function_ref<LogicalResult(ArrayRef<Operation *>)> operationsFn,
    function_ref<LogicalResult(ArrayRef<transform::Param>)> paramsFn,
    function_ref<LogicalResult(ValueRange)> valuesFn) {
  if (llvm::isa<transform::TransformHandleTypeInterface>(handle.getType())) {
    SmallVector<Operation *> operations;
    operations.reserve(values.size());
    for (transform::MappedValue value : values) {
      if (auto *op = llvm::dyn_cast_if_present<Operation *>(value)) {
        operations.push_back(op);
        continue;
      }
      return emitSilenceableFailure(handle.getLoc())
             << "wrong kind of value provided for top-level operation handle";
    }
    if (failed(operationsFn(operations)))
      return DiagnosedSilenceableFailure::definiteFailure();
    return DiagnosedSilenceableFailure::success();
  }

  if (llvm::isa<transform::TransformValueHandleTypeInterface>(
          handle.getType())) {
    SmallVector<Value> payloadValues;
    payloadValues.reserve(values.size());
    for (transform::MappedValue value : values) {
      if (auto v = llvm::dyn_cast_if_present<Value>(value)) {
        payloadValues.push_back(v);
        continue;
      }
      return emitSilenceableFailure(handle.getLoc())
             << "wrong kind of value provided for the top-level value handle";
    }
    if (failed(valuesFn(payloadValues)))
      return DiagnosedSilenceableFailure::definiteFailure();
    return DiagnosedSilenceableFailure::success();
  }

  assert(llvm::isa<transform::TransformParamTypeInterface>(handle.getType()) &&
         "unsupported kind of block argument");
  SmallVector<transform::Param> parameters;
  parameters.reserve(values.size());
  for (transform::MappedValue value : values) {
    if (auto attr = llvm::dyn_cast_if_present<Attribute>(value)) {
      parameters.push_back(attr);
      continue;
    }
    return emitSilenceableFailure(handle.getLoc())
           << "wrong kind of value provided for top-level parameter";
  }
  if (failed(paramsFn(parameters)))
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

LogicalResult
transform::TransformState::mapBlockArgument(BlockArgument argument,
                                            ArrayRef<MappedValue> values) {
  return dispatchMappedValues(
             argument, values,
             [&](ArrayRef<Operation *> operations) {
               return setPayloadOps(argument, operations);
             },
             [&](ArrayRef<Param> params) {
               return setParams(argument, params);
             },
             [&](ValueRange payloadValues) {
               return setPayloadValues(argument, payloadValues);
             })
      .checkAndReport();
}

LogicalResult
transform::TransformState::setPayloadOps(Value value,
                                         ArrayRef<Operation *> targets) {
  assert(value != kTopLevelValue &&
         "attempting to reset the transformation root");
  assert(llvm::isa<TransformHandleTypeInterface>(value.getType()) &&
         "wrong handle type");

  for (Operation *target : targets) {
    if (target)
      continue;
    return emitError(value.getLoc())
           << "attempting to assign a null payload op to this transform value";
  }

  auto iface = llvm::cast<TransformHandleTypeInterface>(value.getType());
  DiagnosedSilenceableFailure result =
      iface.checkPayload(value.getLoc(), targets);
  if (failed(result.checkAndReport()))
    return failure();

  // Setting new payload for the value without cleaning it first is a misuse of
  // the API, assert here.
  SmallVector<Operation *> storedTargets(targets.begin(), targets.end());
  Mappings &mappings = getMapping(value);
  bool inserted =
      mappings.direct.insert({value, std::move(storedTargets)}).second;
  assert(inserted && "value is already associated with another list");
  (void)inserted;

  for (Operation *op : targets)
    mappings.reverse[op].push_back(value);

  return success();
}

LogicalResult
transform::TransformState::setPayloadValues(Value handle,
                                            ValueRange payloadValues) {
  assert(handle != nullptr && "attempting to set params for a null value");
  assert(llvm::isa<TransformValueHandleTypeInterface>(handle.getType()) &&
         "wrong handle type");

  for (Value payload : payloadValues) {
    if (payload)
      continue;
    return emitError(handle.getLoc()) << "attempting to assign a null payload "
                                         "value to this transform handle";
  }

  auto iface = llvm::cast<TransformValueHandleTypeInterface>(handle.getType());
  SmallVector<Value> payloadValueVector = llvm::to_vector(payloadValues);
  DiagnosedSilenceableFailure result =
      iface.checkPayload(handle.getLoc(), payloadValueVector);
  if (failed(result.checkAndReport()))
    return failure();

  Mappings &mappings = getMapping(handle);
  bool inserted =
      mappings.values.insert({handle, std::move(payloadValueVector)}).second;
  assert(
      inserted &&
      "value handle is already associated with another list of payload values");
  (void)inserted;

  for (Value payload : payloadValues)
    mappings.reverseValues[payload].push_back(handle);

  return success();
}

LogicalResult transform::TransformState::setParams(Value value,
                                                   ArrayRef<Param> params) {
  assert(value != nullptr && "attempting to set params for a null value");

  for (Attribute attr : params) {
    if (attr)
      continue;
    return emitError(value.getLoc())
           << "attempting to assign a null parameter to this transform value";
  }

  auto valueType = llvm::dyn_cast<TransformParamTypeInterface>(value.getType());
  assert(value &&
         "cannot associate parameter with a value of non-parameter type");
  DiagnosedSilenceableFailure result =
      valueType.checkPayload(value.getLoc(), params);
  if (failed(result.checkAndReport()))
    return failure();

  Mappings &mappings = getMapping(value);
  bool inserted =
      mappings.params.insert({value, llvm::to_vector(params)}).second;
  assert(inserted && "value is already associated with another list of params");
  (void)inserted;
  return success();
}

template <typename Mapping, typename Key, typename Mapped>
void dropMappingEntry(Mapping &mapping, Key key, Mapped mapped) {
  auto it = mapping.find(key);
  if (it == mapping.end())
    return;

  llvm::erase_value(it->getSecond(), mapped);
  if (it->getSecond().empty())
    mapping.erase(it);
}

void transform::TransformState::forgetMapping(Value opHandle,
                                              ValueRange origOpFlatResults) {
  Mappings &mappings = getMapping(opHandle);
  for (Operation *op : mappings.direct[opHandle])
    dropMappingEntry(mappings.reverse, op, opHandle);
  mappings.direct.erase(opHandle);
#ifndef LLVM_ENABLE_ABI_BREAKING_CHECKS
  // Payload IR is removed from the mapping. This invalidates the respective
  // iterators.
  mappings.incrementTimestamp(opHandle);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

  for (Value opResult : origOpFlatResults) {
    SmallVector<Value> resultHandles;
    (void)getHandlesForPayloadValue(opResult, resultHandles);
    for (Value resultHandle : resultHandles) {
      Mappings &localMappings = getMapping(resultHandle);
      dropMappingEntry(localMappings.values, resultHandle, opResult);
      dropMappingEntry(localMappings.reverseValues, opResult, resultHandle);
    }
  }
}

void transform::TransformState::forgetValueMapping(
    Value valueHandle, ArrayRef<Operation *> payloadOperations) {
  Mappings &mappings = getMapping(valueHandle);
  for (Value payloadValue : mappings.reverseValues[valueHandle])
    dropMappingEntry(mappings.reverseValues, payloadValue, valueHandle);
  mappings.values.erase(valueHandle);

  for (Operation *payloadOp : payloadOperations) {
    SmallVector<Value> opHandles;
    (void)getHandlesForPayloadOp(payloadOp, opHandles);
    for (Value opHandle : opHandles) {
      Mappings &localMappings = getMapping(opHandle);
      dropMappingEntry(localMappings.direct, opHandle, payloadOp);
      dropMappingEntry(localMappings.reverse, payloadOp, opHandle);

#ifndef LLVM_ENABLE_ABI_BREAKING_CHECKS
      // Payload IR is removed from the mapping. This invalidates the respective
      // iterators.
      localMappings.incrementTimestamp(opHandle);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    }
  }
}

LogicalResult
transform::TransformState::replacePayloadOp(Operation *op,
                                            Operation *replacement) {
  // TODO: consider invalidating the handles to nested objects here.

#ifndef NDEBUG
  for (Value opResult : op->getResults()) {
    SmallVector<Value> valueHandles;
    (void)getHandlesForPayloadValue(opResult, valueHandles,
                                    /*includeOutOfScope=*/true);
    assert(valueHandles.empty() && "expected no mapping to old results");
  }
#endif // NDEBUG

  // Drop the mapping between the op and all handles that point to it. Fail if
  // there are no handles.
  SmallVector<Value> opHandles;
  if (failed(getHandlesForPayloadOp(op, opHandles, /*includeOutOfScope=*/true)))
    return failure();
  for (Value handle : opHandles) {
    Mappings &mappings = getMapping(handle, /*allowOutOfScope=*/true);
    dropMappingEntry(mappings.reverse, op, handle);
  }

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (options.getExpensiveChecksEnabled()) {
    auto it = cachedNames.find(op);
    assert(it != cachedNames.end() && "entry not found");
    assert(it->second == op->getName() && "operation name mismatch");
    cachedNames.erase(it);
    if (replacement) {
      auto insertion =
          cachedNames.insert({replacement, replacement->getName()});
      if (!insertion.second) {
        assert(insertion.first->second == replacement->getName() &&
               "operation is already cached with a different name");
      }
    }
  }
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

  // Replace the pointed-to object of all handles with the replacement object.
  // In case a payload op was erased (replacement object is nullptr), a nullptr
  // is stored in the mapping. These nullptrs are removed after each transform.
  // Furthermore, nullptrs are not enumerated by payload op iterators. The
  // relative order of ops is preserved.
  //
  // Removing an op from the mapping would be problematic because removing an
  // element from an array invalidates iterators; merely changing the value of
  // elements does not.
  for (Value handle : opHandles) {
    Mappings &mappings = getMapping(handle, /*allowOutOfScope=*/true);
    auto it = mappings.direct.find(handle);
    if (it == mappings.direct.end())
      continue;

    SmallVector<Operation *, 2> &association = it->getSecond();
    // Note that an operation may be associated with the handle more than once.
    for (Operation *&mapped : association) {
      if (mapped == op)
        mapped = replacement;
    }

    if (replacement) {
      mappings.reverse[replacement].push_back(handle);
    } else {
      opHandlesToCompact.insert(handle);
    }
  }

  return success();
}

LogicalResult
transform::TransformState::replacePayloadValue(Value value, Value replacement) {
  SmallVector<Value> valueHandles;
  if (failed(getHandlesForPayloadValue(value, valueHandles,
                                       /*includeOutOfScope=*/true)))
    return failure();

  for (Value handle : valueHandles) {
    Mappings &mappings = getMapping(handle, /*allowOutOfScope=*/true);
    dropMappingEntry(mappings.reverseValues, value, handle);

    // If replacing with null, that is erasing the mapping, drop the mapping
    // between the handles and the IR objects
    if (!replacement) {
      dropMappingEntry(mappings.values, handle, value);
    } else {
      auto it = mappings.values.find(handle);
      if (it == mappings.values.end())
        continue;

      SmallVector<Value> &association = it->getSecond();
      for (Value &mapped : association) {
        if (mapped == value)
          mapped = replacement;
      }
      mappings.reverseValues[replacement].push_back(handle);
    }
  }

  return success();
}

void transform::TransformState::recordOpHandleInvalidationOne(
    OpOperand &consumingHandle, ArrayRef<Operation *> potentialAncestors,
    Operation *payloadOp, Value otherHandle, Value throughValue,
    transform::TransformState::InvalidatedHandleMap &newlyInvalidated) const {
  // If the op is associated with invalidated handle, skip the check as it
  // may be reading invalid IR. This also ensures we report the first
  // invalidation and not the last one.
  if (invalidatedHandles.count(otherHandle) ||
      newlyInvalidated.count(otherHandle))
    return;

  FULL_LDBG("--recordOpHandleInvalidationOne\n");
  DEBUG_WITH_TYPE(
      DEBUG_TYPE_FULL,
      llvm::interleaveComma(potentialAncestors, DBGS() << "--ancestors: ",
                            [](Operation *op) { llvm::dbgs() << *op; });
      llvm::dbgs() << "\n");

  Operation *owner = consumingHandle.getOwner();
  unsigned operandNo = consumingHandle.getOperandNumber();
  for (Operation *ancestor : potentialAncestors) {
    // clang-format off
    DEBUG_WITH_TYPE(DEBUG_TYPE_FULL, 
      { (DBGS() << "----handle one ancestor: " << *ancestor << "\n"); });
    DEBUG_WITH_TYPE(DEBUG_TYPE_FULL, 
      { (DBGS() << "----of payload with name: " 
                << payloadOp->getName().getIdentifier() << "\n"); });
    DEBUG_WITH_TYPE(DEBUG_TYPE_FULL,
      { (DBGS() << "----of payload: " << *payloadOp << "\n"); });
    // clang-format on
    if (!ancestor->isAncestor(payloadOp))
      continue;

    // Make sure the error-reporting lambda doesn't capture anything
    // by-reference because it will go out of scope. Additionally, extract
    // location from Payload IR ops because the ops themselves may be
    // deleted before the lambda gets called.
    Location ancestorLoc = ancestor->getLoc();
    Location opLoc = payloadOp->getLoc();
    std::optional<Location> throughValueLoc =
        throughValue ? std::make_optional(throughValue.getLoc()) : std::nullopt;
    newlyInvalidated[otherHandle] = [ancestorLoc, opLoc, owner, operandNo,
                                     otherHandle,
                                     throughValueLoc](Location currentLoc) {
      InFlightDiagnostic diag = emitError(currentLoc)
                                << "op uses a handle invalidated by a "
                                   "previously executed transform op";
      diag.attachNote(otherHandle.getLoc()) << "handle to invalidated ops";
      diag.attachNote(owner->getLoc())
          << "invalidated by this transform op that consumes its operand #"
          << operandNo
          << " and invalidates all handles to payload IR entities associated "
             "with this operand and entities nested in them";
      diag.attachNote(ancestorLoc) << "ancestor payload op";
      diag.attachNote(opLoc) << "nested payload op";
      if (throughValueLoc) {
        diag.attachNote(*throughValueLoc)
            << "consumed handle points to this payload value";
      }
    };
  }
}

void transform::TransformState::recordValueHandleInvalidationByOpHandleOne(
    OpOperand &opHandle, ArrayRef<Operation *> potentialAncestors,
    Value payloadValue, Value valueHandle,
    transform::TransformState::InvalidatedHandleMap &newlyInvalidated) const {
  // If the op is associated with invalidated handle, skip the check as it
  // may be reading invalid IR. This also ensures we report the first
  // invalidation and not the last one.
  if (invalidatedHandles.count(valueHandle) ||
      newlyInvalidated.count(valueHandle))
    return;

  for (Operation *ancestor : potentialAncestors) {
    Operation *definingOp;
    std::optional<unsigned> resultNo;
    unsigned argumentNo = std::numeric_limits<unsigned>::max();
    unsigned blockNo = std::numeric_limits<unsigned>::max();
    unsigned regionNo = std::numeric_limits<unsigned>::max();
    if (auto opResult = llvm::dyn_cast<OpResult>(payloadValue)) {
      definingOp = opResult.getOwner();
      resultNo = opResult.getResultNumber();
    } else {
      auto arg = llvm::cast<BlockArgument>(payloadValue);
      definingOp = arg.getParentBlock()->getParentOp();
      argumentNo = arg.getArgNumber();
      blockNo = std::distance(arg.getOwner()->getParent()->begin(),
                              arg.getOwner()->getIterator());
      regionNo = arg.getOwner()->getParent()->getRegionNumber();
    }
    assert(definingOp && "expected the value to be defined by an op as result "
                         "or block argument");
    if (!ancestor->isAncestor(definingOp))
      continue;

    Operation *owner = opHandle.getOwner();
    unsigned operandNo = opHandle.getOperandNumber();
    Location ancestorLoc = ancestor->getLoc();
    Location opLoc = definingOp->getLoc();
    Location valueLoc = payloadValue.getLoc();
    newlyInvalidated[valueHandle] = [valueHandle, owner, operandNo, resultNo,
                                     argumentNo, blockNo, regionNo, ancestorLoc,
                                     opLoc, valueLoc](Location currentLoc) {
      InFlightDiagnostic diag = emitError(currentLoc)
                                << "op uses a handle invalidated by a "
                                   "previously executed transform op";
      diag.attachNote(valueHandle.getLoc()) << "invalidated handle";
      diag.attachNote(owner->getLoc())
          << "invalidated by this transform op that consumes its operand #"
          << operandNo
          << " and invalidates all handles to payload IR entities "
             "associated with this operand and entities nested in them";
      diag.attachNote(ancestorLoc)
          << "ancestor op associated with the consumed handle";
      if (resultNo) {
        diag.attachNote(opLoc)
            << "op defining the value as result #" << *resultNo;
      } else {
        diag.attachNote(opLoc)
            << "op defining the value as block argument #" << argumentNo
            << " of block #" << blockNo << " in region #" << regionNo;
      }
      diag.attachNote(valueLoc) << "payload value";
    };
  }
}

void transform::TransformState::recordOpHandleInvalidation(
    OpOperand &handle, ArrayRef<Operation *> potentialAncestors,
    Value throughValue,
    transform::TransformState::InvalidatedHandleMap &newlyInvalidated) const {

  if (potentialAncestors.empty()) {
    DEBUG_WITH_TYPE(DEBUG_TYPE_FULL, {
      (DBGS() << "----recording invalidation for empty handle: " << handle.get()
              << "\n");
    });

    Operation *owner = handle.getOwner();
    unsigned operandNo = handle.getOperandNumber();
    newlyInvalidated[handle.get()] = [owner, operandNo](Location currentLoc) {
      InFlightDiagnostic diag = emitError(currentLoc)
                                << "op uses a handle associated with empty "
                                   "payload and invalidated by a "
                                   "previously executed transform op";
      diag.attachNote(owner->getLoc())
          << "invalidated by this transform op that consumes its operand #"
          << operandNo;
    };
    return;
  }

  // Iterate over the mapping and invalidate aliasing handles. This is quite
  // expensive and only necessary for error reporting in case of transform
  // dialect misuse with dangling handles. Iteration over the handles is based
  // on the assumption that the number of handles is significantly less than the
  // number of IR objects (operations and values). Alternatively, we could walk
  // the IR nested in each payload op associated with the given handle and look
  // for handles associated with each operation and value.
  for (const auto &[region, mapping] : llvm::reverse(mappings)) {
    // Stop lookup when reaching a region that is isolated from above.
    if (region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
    // Go over all op handle mappings and mark as invalidated any handle
    // pointing to any of the payload ops associated with the given handle or
    // any op nested in them.
    for (const auto &[payloadOp, otherHandles] : mapping->reverse) {
      for (Value otherHandle : otherHandles)
        recordOpHandleInvalidationOne(handle, potentialAncestors, payloadOp,
                                      otherHandle, throughValue,
                                      newlyInvalidated);
    }
    // Go over all value handle mappings and mark as invalidated any handle
    // pointing to any result of the payload op associated with the given handle
    // or any op nested in them. Similarly invalidate handles to argument of
    // blocks belonging to any region of any payload op associated with the
    // given handle or any op nested in them.
    for (const auto &[payloadValue, valueHandles] : mapping->reverseValues) {
      for (Value valueHandle : valueHandles)
        recordValueHandleInvalidationByOpHandleOne(handle, potentialAncestors,
                                                   payloadValue, valueHandle,
                                                   newlyInvalidated);
    }
  }
}

void transform::TransformState::recordValueHandleInvalidation(
    OpOperand &valueHandle,
    transform::TransformState::InvalidatedHandleMap &newlyInvalidated) const {
  // Invalidate other handles to the same value.
  for (Value payloadValue : getPayloadValues(valueHandle.get())) {
    SmallVector<Value> otherValueHandles;
    (void)getHandlesForPayloadValue(payloadValue, otherValueHandles);
    for (Value otherHandle : otherValueHandles) {
      Operation *owner = valueHandle.getOwner();
      unsigned operandNo = valueHandle.getOperandNumber();
      Location valueLoc = payloadValue.getLoc();
      newlyInvalidated[otherHandle] = [otherHandle, owner, operandNo,
                                       valueLoc](Location currentLoc) {
        InFlightDiagnostic diag = emitError(currentLoc)
                                  << "op uses a handle invalidated by a "
                                     "previously executed transform op";
        diag.attachNote(otherHandle.getLoc()) << "invalidated handle";
        diag.attachNote(owner->getLoc())
            << "invalidated by this transform op that consumes its operand #"
            << operandNo
            << " and invalidates handles to the same values as associated with "
               "it";
        diag.attachNote(valueLoc) << "payload value";
      };
    }

    if (auto opResult = llvm::dyn_cast<OpResult>(payloadValue)) {
      Operation *payloadOp = opResult.getOwner();
      recordOpHandleInvalidation(valueHandle, payloadOp, payloadValue,
                                 newlyInvalidated);
    } else {
      auto arg = llvm::dyn_cast<BlockArgument>(payloadValue);
      for (Operation &payloadOp : *arg.getOwner())
        recordOpHandleInvalidation(valueHandle, &payloadOp, payloadValue,
                                   newlyInvalidated);
    }
  }
}

/// Checks that the operation does not use invalidated handles as operands.
/// Reports errors and returns failure if it does. Otherwise, invalidates the
/// handles consumed by the operation as well as any handles pointing to payload
/// IR operations nested in the operations associated with the consumed handles.
LogicalResult transform::TransformState::checkAndRecordHandleInvalidationImpl(
    transform::TransformOpInterface transform,
    transform::TransformState::InvalidatedHandleMap &newlyInvalidated) const {
  FULL_LDBG("--Start checkAndRecordHandleInvalidation\n");
  auto memoryEffectsIface =
      cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectsIface.getEffectsOnResource(
      transform::TransformMappingResource::get(), effects);

  for (OpOperand &target : transform->getOpOperands()) {
    DEBUG_WITH_TYPE(DEBUG_TYPE_FULL, {
      (DBGS() << "----iterate on handle: " << target.get() << "\n");
    });
    // If the operand uses an invalidated handle, report it. If the operation
    // allows handles to point to repeated payload operations, only report
    // pre-existing invalidation errors. Otherwise, also report invalidations
    // caused by the current transform operation affecting its other operands.
    auto it = invalidatedHandles.find(target.get());
    auto nit = newlyInvalidated.find(target.get());
    if (it != invalidatedHandles.end()) {
      FULL_LDBG("--End checkAndRecordHandleInvalidation, found already "
                "invalidated -> FAILURE\n");
      return it->getSecond()(transform->getLoc()), failure();
    }
    if (!transform.allowsRepeatedHandleOperands() &&
        nit != newlyInvalidated.end()) {
      FULL_LDBG("--End checkAndRecordHandleInvalidation, found newly "
                "invalidated (by this op) -> FAILURE\n");
      return nit->getSecond()(transform->getLoc()), failure();
    }

    // Invalidate handles pointing to the operations nested in the operation
    // associated with the handle consumed by this operation.
    auto consumesTarget = [&](const MemoryEffects::EffectInstance &effect) {
      return isa<MemoryEffects::Free>(effect.getEffect()) &&
             effect.getValue() == target.get();
    };
    if (llvm::any_of(effects, consumesTarget)) {
      FULL_LDBG("----found consume effect\n");
      if (llvm::isa<transform::TransformHandleTypeInterface>(
              target.get().getType())) {
        FULL_LDBG("----recordOpHandleInvalidation\n");
        SmallVector<Operation *> payloadOps =
            llvm::to_vector(getPayloadOps(target.get()));
        recordOpHandleInvalidation(target, payloadOps, nullptr,
                                   newlyInvalidated);
      } else if (llvm::isa<transform::TransformValueHandleTypeInterface>(
                     target.get().getType())) {
        FULL_LDBG("----recordValueHandleInvalidation\n");
        recordValueHandleInvalidation(target, newlyInvalidated);
      } else {
        FULL_LDBG("----not a TransformHandle -> SKIP AND DROP ON THE FLOOR\n");
      }
    } else {
      FULL_LDBG("----no consume effect -> SKIP\n");
    }
  }

  FULL_LDBG("--End checkAndRecordHandleInvalidation -> SUCCESS\n");
  return success();
}

LogicalResult transform::TransformState::checkAndRecordHandleInvalidation(
    transform::TransformOpInterface transform) {
  InvalidatedHandleMap newlyInvalidated;
  LogicalResult checkResult =
      checkAndRecordHandleInvalidationImpl(transform, newlyInvalidated);
  invalidatedHandles.insert(std::make_move_iterator(newlyInvalidated.begin()),
                            std::make_move_iterator(newlyInvalidated.end()));
  return checkResult;
}

template <typename T>
DiagnosedSilenceableFailure
checkRepeatedConsumptionInOperand(ArrayRef<T> payload,
                                  transform::TransformOpInterface transform,
                                  unsigned operandNumber) {
  DenseSet<T> seen;
  for (T p : payload) {
    if (!seen.insert(p).second) {
      DiagnosedSilenceableFailure diag =
          transform.emitSilenceableError()
          << "a handle passed as operand #" << operandNumber
          << " and consumed by this operation points to a payload "
             "entity more than once";
      if constexpr (std::is_pointer_v<T>)
        diag.attachNote(p->getLoc()) << "repeated target op";
      else
        diag.attachNote(p.getLoc()) << "repeated target value";
      return diag;
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::TransformState::compactOpHandles() {
  for (Value handle : opHandlesToCompact) {
    Mappings &mappings = getMapping(handle, /*allowOutOfScope=*/true);
#ifndef LLVM_ENABLE_ABI_BREAKING_CHECKS
    if (llvm::find(mappings.direct[handle], nullptr) !=
        mappings.direct[handle].end())
      // Payload IR is removed from the mapping. This invalidates the respective
      // iterators.
      mappings.incrementTimestamp(handle);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    llvm::erase_value(mappings.direct[handle], nullptr);
  }
  opHandlesToCompact.clear();
}

DiagnosedSilenceableFailure
transform::TransformState::applyTransform(TransformOpInterface transform) {
  LLVM_DEBUG({
    DBGS() << "applying: ";
    transform->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  DEBUG_WITH_TYPE(DEBUG_TYPE_FULL,
                  DBGS() << "Top-level payload before application:\n"
                         << *getTopLevel() << "\n");
  auto printOnFailureRAII = llvm::make_scope_exit([this] {
    (void)this;
    LLVM_DEBUG(DBGS() << "Failing Top-level payload:\n"; getTopLevel()->print(
        llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm()););
  });
  if (options.getExpensiveChecksEnabled()) {
    FULL_LDBG("ExpensiveChecksEnabled\n");
    if (failed(checkAndRecordHandleInvalidation(transform)))
      return DiagnosedSilenceableFailure::definiteFailure();

    for (OpOperand &operand : transform->getOpOperands()) {
      DEBUG_WITH_TYPE(DEBUG_TYPE_FULL, {
        (DBGS() << "iterate on handle: " << operand.get() << "\n");
      });
      if (!isHandleConsumed(operand.get(), transform)) {
        FULL_LDBG("--handle not consumed -> SKIP\n");
        continue;
      }
      if (transform.allowsRepeatedHandleOperands()) {
        FULL_LDBG("--op allows repeated handles -> SKIP\n");
        continue;
      }
      FULL_LDBG("--handle is consumed\n");

      Type operandType = operand.get().getType();
      if (llvm::isa<TransformHandleTypeInterface>(operandType)) {
        FULL_LDBG("--checkRepeatedConsumptionInOperand for Operation*\n");
        DiagnosedSilenceableFailure check =
            checkRepeatedConsumptionInOperand<Operation *>(
                getPayloadOpsView(operand.get()), transform,
                operand.getOperandNumber());
        if (!check.succeeded()) {
          FULL_LDBG("----FAILED\n");
          return check;
        }
      } else if (llvm::isa<TransformValueHandleTypeInterface>(operandType)) {
        FULL_LDBG("--checkRepeatedConsumptionInOperand For Value\n");
        DiagnosedSilenceableFailure check =
            checkRepeatedConsumptionInOperand<Value>(
                getPayloadValues(operand.get()), transform,
                operand.getOperandNumber());
        if (!check.succeeded()) {
          FULL_LDBG("----FAILED\n");
          return check;
        }
      } else {
        FULL_LDBG("--not a TransformHandle -> SKIP AND DROP ON THE FLOOR\n");
      }
    }

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    // Cache Operation* -> OperationName mappings. These will be checked after
    // the transform has been applied to detect incorrect memory side effects
    // and missing op tracking.
    for (std::unique_ptr<Mappings> &mapping :
         llvm::make_second_range(mappings)) {
      for (Operation *op : llvm::make_first_range(mapping->reverse)) {
        auto insertion = cachedNames.insert({op, op->getName()});
        if (!insertion.second) {
          if (insertion.first->second != op->getName()) {
            // Operation is already in the cache, but with a different name.
            DiagnosedDefiniteFailure diag =
                emitDefiniteFailure(transform->getLoc())
                << "expensive checks failure: operation mismatch, expected "
                << insertion.first->second;
            diag.attachNote(op->getLoc()) << "payload op: " << op->getName();
            return diag;
          }
        }
      }
    }
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }

  // Find which operands are consumed.
  SmallVector<OpOperand *> consumedOperands =
      transform.getConsumedHandleOpOperands();

  // Remember the results of the payload ops associated with the consumed
  // op handles or the ops defining the value handles so we can drop the
  // association with them later. This must happen here because the
  // transformation may destroy or mutate them so we cannot traverse the payload
  // IR after that.
  SmallVector<Value> origOpFlatResults;
  SmallVector<Operation *> origAssociatedOps;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  DenseSet<Operation *> consumedPayloadOps;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  for (OpOperand *opOperand : consumedOperands) {
    Value operand = opOperand->get();
    if (llvm::isa<TransformHandleTypeInterface>(operand.getType())) {
      for (Operation *payloadOp : getPayloadOps(operand)) {
        llvm::append_range(origOpFlatResults, payloadOp->getResults());
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
        if (options.getExpensiveChecksEnabled()) {
          // Store all consumed payload ops (and their nested ops) in a set for
          // extra error checking.
          payloadOp->walk(
              [&](Operation *op) { consumedPayloadOps.insert(op); });
        }
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
      }
      continue;
    }
    if (llvm::isa<TransformValueHandleTypeInterface>(operand.getType())) {
      for (Value payloadValue : getPayloadValues(operand)) {
        if (llvm::isa<OpResult>(payloadValue)) {
          origAssociatedOps.push_back(payloadValue.getDefiningOp());
          continue;
        }
        llvm::append_range(
            origAssociatedOps,
            llvm::map_range(*llvm::cast<BlockArgument>(payloadValue).getOwner(),
                            [](Operation &op) { return &op; }));
      }
      continue;
    }
    DiagnosedDefiniteFailure diag =
        emitDefiniteFailure(transform->getLoc())
        << "unexpectedly consumed a value that is not a handle as operand #"
        << opOperand->getOperandNumber();
    diag.attachNote(operand.getLoc())
        << "value defined here with type " << operand.getType();
    return diag;
  }

  // Prepare rewriter and listener.
  transform::ErrorCheckingTrackingListener trackingListener(*this, transform);
  transform::TransformRewriter rewriter(transform->getContext(),
                                        &trackingListener);

  // Compute the result but do not short-circuit the silenceable failure case as
  // we still want the handles to propagate properly so the "suppress" mode can
  // proceed on a best effort basis.
  transform::TransformResults results(transform->getNumResults());
  DiagnosedSilenceableFailure result(transform.apply(rewriter, results, *this));
  compactOpHandles();

  // Error handling: fail if transform or listener failed.
  DiagnosedSilenceableFailure trackingFailure =
      trackingListener.checkAndResetError();
  if (!transform->hasTrait<ReportTrackingListenerFailuresOpTrait>() ||
      transform->hasAttr(
          transform::TransformDialect::kSilenceTrackingFailuresAttrName)) {
    // Only report failures for ReportTrackingListenerFailuresOpTrait ops. Also
    // do not report failures if the above mentioned attribute is set.
    if (trackingFailure.isSilenceableFailure())
      (void)trackingFailure.silence();
    trackingFailure = DiagnosedSilenceableFailure::success();
  }
  if (!trackingFailure.succeeded()) {
    if (result.succeeded()) {
      result = std::move(trackingFailure);
    } else {
      // Transform op errors have precedence, report those first.
      if (result.isSilenceableFailure())
        result.attachNote() << "tracking listener also failed: "
                            << trackingFailure.getMessage();
      (void)trackingFailure.silence();
    }
  }
  if (result.isDefiniteFailure())
    return result;

  // If a silenceable failure was produced, some results may be unset, set them
  // to empty lists.
  if (result.isSilenceableFailure())
    results.setRemainingToEmpty(transform);

  // Remove the mapping for the operand if it is consumed by the operation. This
  // allows us to catch use-after-free with assertions later on.
  for (OpOperand *opOperand : consumedOperands) {
    Value operand = opOperand->get();
    if (llvm::isa<TransformHandleTypeInterface>(operand.getType())) {
      forgetMapping(operand, origOpFlatResults);
    } else if (llvm::isa<TransformValueHandleTypeInterface>(
                   operand.getType())) {
      forgetValueMapping(operand, origAssociatedOps);
    }
  }

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (options.getExpensiveChecksEnabled()) {
    // Remove erased ops from the transform state.
    for (Operation *op : consumedPayloadOps) {
      // This payload op was consumed but it may still be mapped to one or
      // multiple handles. Forget all handles that are mapped to the op, so that
      // there are no dangling pointers in the transform dialect state. This is
      // necessary so that the `cachedNames`-based checks work correctly.
      //
      // Note: Dangling pointers to erased payload ops are allowed if the
      // corresponding handles are not used anymore. There is another
      // "expensive-check" that looks for future uses of dangling payload op
      // pointers (through arbitrary handles). Removing handles to erased ops
      // does not interfere with the other expensive checks: handle invalidation
      // happens earlier and keeps track of invalidated handles with
      // pre-generated error messages, so we do not need the association to
      // still be there when the invalidated handle is accessed.
      SmallVector<Value> handles;
      (void)getHandlesForPayloadOp(op, handles);
      for (Value handle : handles)
        forgetMapping(handle, /*origOpFlatResults=*/ValueRange());
      cachedNames.erase(op);
    }

    // Check cached operation names.
    for (std::unique_ptr<Mappings> &mapping :
         llvm::make_second_range(mappings)) {
      for (Operation *op : llvm::make_first_range(mapping->reverse)) {
        // Make sure that the name of the op has not changed. If it has changed,
        // the op was removed and a new op was allocated at the same memory
        // location. This means that we are missing op tracking somewhere.
        auto cacheIt = cachedNames.find(op);
        if (cacheIt == cachedNames.end()) {
          DiagnosedDefiniteFailure diag =
              emitDefiniteFailure(transform->getLoc())
              << "expensive checks failure: operation not found in cache";
          diag.attachNote(op->getLoc()) << "payload op";
          return diag;
        }
        // If the `getName` call (or the above `attachNote`) is crashing, we
        // have a dangling pointer. This usually means that an op was erased but
        // the transform dialect was not made aware of that; e.g., missing
        // "consumesHandle" or rewriter usage.
        if (cacheIt->second != op->getName()) {
          DiagnosedDefiniteFailure diag =
              emitDefiniteFailure(transform->getLoc())
              << "expensive checks failure: operation mismatch, expected "
              << cacheIt->second;
          diag.attachNote(op->getLoc()) << "payload op: " << op->getName();
          return diag;
        }
      }
    }
  }
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

  if (failed(updateStateFromResults(results, transform->getResults())))
    return DiagnosedSilenceableFailure::definiteFailure();

  printOnFailureRAII.release();
  DEBUG_WITH_TYPE(DEBUG_PRINT_AFTER_ALL, {
    DBGS() << "Top-level payload:\n";
    getTopLevel()->print(llvm::dbgs());
  });
  return result;
}

LogicalResult transform::TransformState::updateStateFromResults(
    const TransformResults &results, ResultRange opResults) {
  for (OpResult result : opResults) {
    if (llvm::isa<TransformParamTypeInterface>(result.getType())) {
      assert(results.isParam(result.getResultNumber()) &&
             "expected parameters for the parameter-typed result");
      if (failed(
              setParams(result, results.getParams(result.getResultNumber())))) {
        return failure();
      }
    } else if (llvm::isa<TransformValueHandleTypeInterface>(result.getType())) {
      assert(results.isValue(result.getResultNumber()) &&
             "expected values for value-type-result");
      if (failed(setPayloadValues(
              result, results.getValues(result.getResultNumber())))) {
        return failure();
      }
    } else {
      assert(!results.isParam(result.getResultNumber()) &&
             "expected payload ops for the non-parameter typed result");
      if (failed(
              setPayloadOps(result, results.get(result.getResultNumber())))) {
        return failure();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransformState::Extension
//===----------------------------------------------------------------------===//

transform::TransformState::Extension::~Extension() = default;

LogicalResult
transform::TransformState::Extension::replacePayloadOp(Operation *op,
                                                       Operation *replacement) {
  // TODO: we may need to invalidate handles to operations and values nested in
  // the operation being replaced.
  return state.replacePayloadOp(op, replacement);
}

LogicalResult
transform::TransformState::Extension::replacePayloadValue(Value value,
                                                          Value replacement) {
  return state.replacePayloadValue(value, replacement);
}

//===----------------------------------------------------------------------===//
// TransformState::RegionScope
//===----------------------------------------------------------------------===//

transform::TransformState::RegionScope::~RegionScope() {
  // Remove handle invalidation notices as handles are going out of scope.
  // The same region may be re-entered leading to incorrect invalidation
  // errors.
  for (Block &block : *region) {
    for (Value handle : block.getArguments()) {
      state.invalidatedHandles.erase(handle);
    }
    for (Operation &op : block) {
      for (Value handle : op.getResults()) {
        state.invalidatedHandles.erase(handle);
      }
    }
  }

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  // Remember pointers to payload ops referenced by the handles going out of
  // scope.
  SmallVector<Operation *> referencedOps =
      llvm::to_vector(llvm::make_first_range(state.mappings[region]->reverse));
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

  state.mappings.erase(region);

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  // If the last handle to a payload op has gone out of scope, we no longer
  // need to store the cached name. Pointers may get reused, leading to
  // incorrect associations in the cache.
  for (Operation *op : referencedOps) {
    SmallVector<Value> handles;
    if (succeeded(state.getHandlesForPayloadOp(op, handles)))
      continue;
    state.cachedNames.erase(op);
  }

  state.regionStack.pop_back();
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
}

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//

transform::TransformResults::TransformResults(unsigned numSegments) {
  operations.appendEmptyRows(numSegments);
  params.appendEmptyRows(numSegments);
  values.appendEmptyRows(numSegments);
}

void transform::TransformResults::setParams(
    OpResult value, ArrayRef<transform::TransformState::Param> params) {
  int64_t position = value.getResultNumber();
  assert(position < static_cast<int64_t>(this->params.size()) &&
         "setting params for a non-existent handle");
  assert(this->params[position].data() == nullptr && "params already set");
  assert(operations[position].data() == nullptr &&
         "another kind of results already set");
  assert(values[position].data() == nullptr &&
         "another kind of results already set");
  this->params.replace(position, params);
}

void transform::TransformResults::setValues(OpResult handle,
                                            ValueRange values) {
  int64_t position = handle.getResultNumber();
  assert(position < static_cast<int64_t>(this->values.size()) &&
         "setting values for a non-existent handle");
  assert(this->values[position].data() == nullptr && "values already set");
  assert(operations[position].data() == nullptr &&
         "another kind of results already set");
  assert(params[position].data() == nullptr &&
         "another kind of results already set");
  this->values.replace(position, values);
}

void transform::TransformResults::setMappedValues(
    OpResult handle, ArrayRef<MappedValue> values) {
  DiagnosedSilenceableFailure diag = dispatchMappedValues(
      handle, values,
      [&](ArrayRef<Operation *> operations) {
        return set(handle, operations), success();
      },
      [&](ArrayRef<Param> params) {
        return setParams(handle, params), success();
      },
      [&](ValueRange payloadValues) {
        return setValues(handle, payloadValues), success();
      });
#ifndef NDEBUG
  if (!diag.succeeded())
    llvm::dbgs() << diag.getStatusString() << "\n";
  assert(diag.succeeded() && "incorrect mapping");
#endif // NDEBUG
  (void)diag.silence();
}

void transform::TransformResults::setRemainingToEmpty(
    transform::TransformOpInterface transform) {
  for (OpResult opResult : transform->getResults()) {
    if (!isSet(opResult.getResultNumber()))
      setMappedValues(opResult, {});
  }
}

ArrayRef<Operation *>
transform::TransformResults::get(unsigned resultNumber) const {
  assert(resultNumber < operations.size() &&
         "querying results for a non-existent handle");
  assert(operations[resultNumber].data() != nullptr &&
         "querying unset results (values or params expected?)");
  return operations[resultNumber];
}

ArrayRef<transform::TransformState::Param>
transform::TransformResults::getParams(unsigned resultNumber) const {
  assert(resultNumber < params.size() &&
         "querying params for a non-existent handle");
  assert(params[resultNumber].data() != nullptr &&
         "querying unset params (ops or values expected?)");
  return params[resultNumber];
}

ArrayRef<Value>
transform::TransformResults::getValues(unsigned resultNumber) const {
  assert(resultNumber < values.size() &&
         "querying values for a non-existent handle");
  assert(values[resultNumber].data() != nullptr &&
         "querying unset values (ops or params expected?)");
  return values[resultNumber];
}

bool transform::TransformResults::isParam(unsigned resultNumber) const {
  assert(resultNumber < params.size() &&
         "querying association for a non-existent handle");
  return params[resultNumber].data() != nullptr;
}

bool transform::TransformResults::isValue(unsigned resultNumber) const {
  assert(resultNumber < values.size() &&
         "querying association for a non-existent handle");
  return values[resultNumber].data() != nullptr;
}

bool transform::TransformResults::isSet(unsigned resultNumber) const {
  assert(resultNumber < params.size() &&
         "querying association for a non-existent handle");
  return params[resultNumber].data() != nullptr ||
         operations[resultNumber].data() != nullptr ||
         values[resultNumber].data() != nullptr;
}

//===----------------------------------------------------------------------===//
// TrackingListener
//===----------------------------------------------------------------------===//

transform::TrackingListener::TrackingListener(TransformState &state,
                                              TransformOpInterface op)
    : TransformState::Extension(state), transformOp(op) {
  if (op) {
    for (OpOperand *opOperand : transformOp.getConsumedHandleOpOperands()) {
      consumedHandles.insert(opOperand->get());
    }
  }
}

Operation *transform::TrackingListener::getCommonDefiningOp(ValueRange values) {
  Operation *defOp = nullptr;
  for (Value v : values) {
    // Skip empty values.
    if (!v)
      continue;
    if (!defOp) {
      defOp = v.getDefiningOp();
      continue;
    }
    if (defOp != v.getDefiningOp())
      return nullptr;
  }
  return defOp;
}

FailureOr<Operation *>
transform::TrackingListener::findReplacementOp(Operation *op,
                                               ValueRange newValues) const {
  assert(op->getNumResults() == newValues.size() &&
         "invalid number of replacement values");
  SmallVector<Value> values(newValues.begin(), newValues.end());

  do {
    // If the replacement values belong to different ops, drop the mapping.
    Operation *defOp = getCommonDefiningOp(values);
    if (!defOp)
      return failure();

    // If the defining op has the same type, we take it as a replacement.
    if (op->getName() == defOp->getName())
      return defOp;

    // Replacing an op with a constant-like equivalent is a common
    // canonicalization.
    if (defOp->hasTrait<OpTrait::ConstantLike>())
      return defOp;

    values.clear();

    // Skip through ops that implement FindPayloadReplacementOpInterface.
    if (auto findReplacementOpInterface =
            dyn_cast<FindPayloadReplacementOpInterface>(defOp)) {
      values.assign(findReplacementOpInterface.getNextOperands());
      continue;
    }

    // Skip through ops that implement CastOpInterface.
    if (isa<CastOpInterface>(defOp)) {
      values.assign(defOp->getOperands().begin(), defOp->getOperands().end());
      continue;
    }
  } while (!values.empty());

  return failure();
}

LogicalResult transform::TrackingListener::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    DBGS() << "Match Failure : " << diag.str() << "\n";
  });
  return failure();
}

void transform::TrackingListener::notifyOperationRemoved(Operation *op) {
  // TODO: Walk can be removed when D144193 has landed.
  op->walk([&](Operation *op) {
    // Remove mappings for result values.
    for (OpResult value : op->getResults())
      (void)replacePayloadValue(value, nullptr);
    // Remove mapping for op.
    (void)replacePayloadOp(op, nullptr);
  });
}

/// Return true if `a` happens before `b`, i.e., `a` or one of its ancestors
/// properly dominates `b` and `b` is not inside `a`.
static bool happensBefore(Operation *a, Operation *b) {
  do {
    if (a->isProperAncestor(b))
      return false;
    if (Operation *bAncestor = a->getBlock()->findAncestorOpInBlock(*b)) {
      return a->isBeforeInBlock(bAncestor);
    }
  } while ((a = a->getParentOp()));
  return false;
}

void transform::TrackingListener::notifyOperationReplaced(
    Operation *op, ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "invalid number of replacement values");

  // Replace value handles.
  for (auto [oldValue, newValue] : llvm::zip(op->getResults(), newValues))
    (void)replacePayloadValue(oldValue, newValue);

  // Replace op handle.
  SmallVector<Value> opHandles;
  if (failed(getTransformState().getHandlesForPayloadOp(
          op, opHandles, /*includeOutOfScope=*/true))) {
    // Op is not tracked.
    return;
  }

  // Helper function to check if the current transform op consumes any handle
  // that is mapped to `op`.
  //
  // Note: If a handle was consumed, there shouldn't be any alive users, so it
  // is not really necessary to check for consumed handles. However, in case
  // there are indeed alive handles that were consumed (which is undefined
  // behavior) and a replacement op could not be found, we want to fail with a
  // nicer error message: "op uses a handle invalidated..." instead of "could
  // not find replacement op". This nicer error is produced later.
  auto handleWasConsumed = [&] {
    return llvm::any_of(opHandles,
                        [&](Value h) { return consumedHandles.contains(h); });
  };

  // Helper function to check if the handle is alive.
  auto hasAliveUser = [&]() {
    for (Value v : opHandles) {
      for (Operation *user : v.getUsers())
        if (user != transformOp && !happensBefore(user, transformOp))
          return true;
    }
    return false;
  };

  if (!hasAliveUser() || handleWasConsumed()) {
    // The op is tracked but the corresponding handles are dead or were
    // consumed. Drop the op form the mapping.
    (void)replacePayloadOp(op, nullptr);
    return;
  }

  FailureOr<Operation *> replacement = findReplacementOp(op, newValues);
  // If the op is tracked but no replacement op was found, send a
  // notification.
  if (failed(replacement)) {
    notifyPayloadReplacementNotFound(op, newValues);
    (void)replacePayloadOp(op, nullptr);
    return;
  }

  (void)replacePayloadOp(op, *replacement);
}

transform::ErrorCheckingTrackingListener::~ErrorCheckingTrackingListener() {
  // The state of the ErrorCheckingTrackingListener must be checked and reset
  // if there was an error. This is to prevent errors from accidentally being
  // missed.
  assert(status.succeeded() && "listener state was not checked");
}

DiagnosedSilenceableFailure
transform::ErrorCheckingTrackingListener::checkAndResetError() {
  DiagnosedSilenceableFailure s = std::move(status);
  status = DiagnosedSilenceableFailure::success();
  errorCounter = 0;
  return s;
}

bool transform::ErrorCheckingTrackingListener::failed() const {
  return !status.succeeded();
}

void transform::ErrorCheckingTrackingListener::notifyPayloadReplacementNotFound(
    Operation *op, ValueRange values) {
  if (status.succeeded()) {
    status = emitSilenceableFailure(
        getTransformOp(), "tracking listener failed to find replacement op");
  }

  status.attachNote(op->getLoc()) << "[" << errorCounter << "] replaced op";
  for (auto &&[index, value] : llvm::enumerate(values))
    status.attachNote(value.getLoc())
        << "[" << errorCounter << "] replacement value " << index;

  ++errorCounter;
}

//===----------------------------------------------------------------------===//
// TransformRewriter
//===----------------------------------------------------------------------===//

transform::TransformRewriter::TransformRewriter(
    MLIRContext *ctx, ErrorCheckingTrackingListener *listener)
    : RewriterBase(ctx), listener(listener) {
  setListener(listener);
}

bool transform::TransformRewriter::hasTrackingFailures() const {
  return listener->failed();
}

/// Silence all tracking failures that have been encountered so far.
void transform::TransformRewriter::silenceTrackingFailure() {
  if (hasTrackingFailures()) {
    DiagnosedSilenceableFailure status = listener->checkAndResetError();
    (void)status.silence();
  }
}

LogicalResult transform::TransformRewriter::notifyPayloadOperationReplaced(
    Operation *op, Operation *replacement) {
  return listener->replacePayloadOp(op, replacement);
}

//===----------------------------------------------------------------------===//
// Utilities for TransformEachOpTrait.
//===----------------------------------------------------------------------===//

LogicalResult
transform::detail::checkNestedConsumption(Location loc,
                                          ArrayRef<Operation *> targets) {
  for (auto &&[position, parent] : llvm::enumerate(targets)) {
    for (Operation *child : targets.drop_front(position + 1)) {
      if (parent->isAncestor(child)) {
        InFlightDiagnostic diag =
            emitError(loc)
            << "transform operation consumes a handle pointing to an ancestor "
               "payload operation before its descendant";
        diag.attachNote()
            << "the ancestor is likely erased or rewritten before the "
               "descendant is accessed, leading to undefined behavior";
        diag.attachNote(parent->getLoc()) << "ancestor payload op";
        diag.attachNote(child->getLoc()) << "descendant payload op";
        return diag;
      }
    }
  }
  return success();
}

LogicalResult
transform::detail::checkApplyToOne(Operation *transformOp,
                                   Location payloadOpLoc,
                                   const ApplyToEachResultList &partialResult) {
  Location transformOpLoc = transformOp->getLoc();
  StringRef transformOpName = transformOp->getName().getStringRef();
  unsigned expectedNumResults = transformOp->getNumResults();

  // Reuse the emission of the diagnostic note.
  auto emitDiag = [&]() {
    auto diag = mlir::emitError(transformOpLoc);
    diag.attachNote(payloadOpLoc) << "when applied to this op";
    return diag;
  };

  if (partialResult.size() != expectedNumResults) {
    auto diag = emitDiag() << "application of " << transformOpName
                           << " expected to produce " << expectedNumResults
                           << " results (actually produced "
                           << partialResult.size() << ").";
    diag.attachNote(transformOpLoc)
        << "if you need variadic results, consider a generic `apply` "
        << "instead of the specialized `applyToOne`.";
    return failure();
  }

  // Check that the right kind of value was produced.
  for (const auto &[ptr, res] :
       llvm::zip(partialResult, transformOp->getResults())) {
    if (ptr.isNull())
      continue;
    if (llvm::isa<TransformHandleTypeInterface>(res.getType()) &&
        !ptr.is<Operation *>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce an Operation * for result #"
                        << res.getResultNumber();
    }
    if (llvm::isa<TransformParamTypeInterface>(res.getType()) &&
        !ptr.is<Attribute>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce an Attribute for result #"
                        << res.getResultNumber();
    }
    if (llvm::isa<TransformValueHandleTypeInterface>(res.getType()) &&
        !ptr.is<Value>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce a Value for result #"
                        << res.getResultNumber();
    }
  }
  return success();
}

template <typename T>
static SmallVector<T> castVector(ArrayRef<transform::MappedValue> range) {
  return llvm::to_vector(llvm::map_range(
      range, [](transform::MappedValue value) { return value.get<T>(); }));
}

void transform::detail::setApplyToOneResults(
    Operation *transformOp, TransformResults &transformResults,
    ArrayRef<ApplyToEachResultList> results) {
  SmallVector<SmallVector<MappedValue>> transposed;
  transposed.resize(transformOp->getNumResults());
  for (const ApplyToEachResultList &partialResults : results) {
    if (llvm::any_of(partialResults,
                     [](MappedValue value) { return value.isNull(); }))
      continue;
    assert(transformOp->getNumResults() == partialResults.size() &&
           "expected as many partial results as op as results");
    for (auto [i, value] : llvm::enumerate(partialResults))
      transposed[i].push_back(value);
  }

  for (OpResult r : transformOp->getResults()) {
    unsigned position = r.getResultNumber();
    if (llvm::isa<TransformParamTypeInterface>(r.getType())) {
      transformResults.setParams(r,
                                 castVector<Attribute>(transposed[position]));
    } else if (llvm::isa<TransformValueHandleTypeInterface>(r.getType())) {
      transformResults.setValues(r, castVector<Value>(transposed[position]));
    } else {
      transformResults.set(r, castVector<Operation *>(transposed[position]));
    }
  }
}

//===----------------------------------------------------------------------===//
// Utilities for implementing transform ops with regions.
//===----------------------------------------------------------------------===//

void transform::detail::prepareValueMappings(
    SmallVectorImpl<SmallVector<transform::MappedValue>> &mappings,
    ValueRange values, const transform::TransformState &state) {
  for (Value operand : values) {
    SmallVector<MappedValue> &mapped = mappings.emplace_back();
    if (llvm::isa<TransformHandleTypeInterface>(operand.getType())) {
      llvm::append_range(mapped, state.getPayloadOps(operand));
    } else if (llvm::isa<TransformValueHandleTypeInterface>(
                   operand.getType())) {
      llvm::append_range(mapped, state.getPayloadValues(operand));
    } else {
      assert(llvm::isa<TransformParamTypeInterface>(operand.getType()) &&
             "unsupported kind of transform dialect value");
      llvm::append_range(mapped, state.getParams(operand));
    }
  }
}

void transform::detail::forwardTerminatorOperands(
    Block *block, transform::TransformState &state,
    transform::TransformResults &results) {
  for (auto &&[terminatorOperand, result] :
       llvm::zip(block->getTerminator()->getOperands(),
                 block->getParentOp()->getOpResults())) {
    if (llvm::isa<transform::TransformHandleTypeInterface>(result.getType())) {
      results.set(result, state.getPayloadOps(terminatorOperand));
    } else if (llvm::isa<transform::TransformValueHandleTypeInterface>(
                   result.getType())) {
      results.setValues(result, state.getPayloadValues(terminatorOperand));
    } else {
      assert(
          llvm::isa<transform::TransformParamTypeInterface>(result.getType()) &&
          "unhandled transform type interface");
      results.setParams(result, state.getParams(terminatorOperand));
    }
  }
}

transform::TransformState
transform::detail::makeTransformStateForTesting(Region *region,
                                                Operation *payloadRoot) {
  return TransformState(region, payloadRoot);
}

//===----------------------------------------------------------------------===//
// Utilities for PossibleTopLevelTransformOpTrait.
//===----------------------------------------------------------------------===//

/// Appends to `effects` the memory effect instances on `target` with the same
/// resource and effect as the ones the operation `iface` having on `source`.
static void
remapEffects(MemoryEffectOpInterface iface, BlockArgument source, Value target,
             SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  SmallVector<MemoryEffects::EffectInstance> nestedEffects;
  iface.getEffectsOnValue(source, nestedEffects);
  for (const auto &effect : nestedEffects)
    effects.emplace_back(effect.getEffect(), target, effect.getResource());
}

/// Appends to `effects` the same effects as the operations of `block` have on
/// block arguments but associated with `operands.`
static void
remapArgumentEffects(Block &block, ValueRange operands,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (Operation &op : block) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!iface)
      continue;

    for (auto &&[source, target] : llvm::zip(block.getArguments(), operands)) {
      remapEffects(iface, source, target, effects);
    }

    SmallVector<MemoryEffects::EffectInstance> nestedEffects;
    iface.getEffectsOnResource(transform::PayloadIRResource::get(),
                               nestedEffects);
    llvm::append_range(effects, nestedEffects);
  }
}

void transform::detail::getPotentialTopLevelEffects(
    Operation *operation, Value root, Block &body,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(operation->getOperands(), effects);
  transform::producesHandle(operation->getResults(), effects);

  if (!root) {
    for (Operation &op : body) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
      if (!iface)
        continue;

      SmallVector<MemoryEffects::EffectInstance, 2> nestedEffects;
      iface.getEffects(effects);
    }
    return;
  }

  // Carry over all effects on arguments of the entry block as those on the
  // operands, this is the same value just remapped.
  remapArgumentEffects(body, operation->getOperands(), effects);
}

LogicalResult transform::detail::mapPossibleTopLevelTransformOpBlockArguments(
    TransformState &state, Operation *op, Region &region) {
  SmallVector<Operation *> targets;
  SmallVector<SmallVector<MappedValue>> extraMappings;
  if (op->getNumOperands() != 0) {
    llvm::append_range(targets, state.getPayloadOps(op->getOperand(0)));
    prepareValueMappings(extraMappings, op->getOperands().drop_front(), state);
  } else {
    if (state.getNumTopLevelMappings() !=
        region.front().getNumArguments() - 1) {
      return emitError(op->getLoc())
             << "operation expects " << region.front().getNumArguments() - 1
             << " extra value bindings, but " << state.getNumTopLevelMappings()
             << " were provided to the interpreter";
    }

    // Top-level transforms can be used for matching. If no concrete operation
    // type is specified, the block argument is mapped to the top-level op.
    // Otherwise, it is mapped to all ops of the specified type within the
    // top-level op (including the top-level op itself). Once an op is added as
    // a target, its descendants are not explored any further.
    BlockArgument bbArg = region.front().getArgument(0);
    if (auto bbArgType = dyn_cast<transform::OperationType>(bbArg.getType())) {
      state.getTopLevel()->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->getName().getStringRef() == bbArgType.getOperationName()) {
          targets.push_back(op);
          return WalkResult::skip();
        }
        return WalkResult::advance();
      });
    } else {
      targets.push_back(state.getTopLevel());
    }

    for (unsigned i = 0, e = state.getNumTopLevelMappings(); i < e; ++i)
      extraMappings.push_back(llvm::to_vector(state.getTopLevelMapping(i)));
  }

  if (failed(state.mapBlockArguments(region.front().getArgument(0), targets)))
    return failure();

  for (BlockArgument argument : region.front().getArguments().drop_front()) {
    if (failed(state.mapBlockArgument(
            argument, extraMappings[argument.getArgNumber() - 1])))
      return failure();
  }

  return success();
}

LogicalResult
transform::detail::verifyPossibleTopLevelTransformOpTrait(Operation *op) {
  // Attaching this trait without the interface is a misuse of the API, but it
  // cannot be caught via a static_assert because interface registration is
  // dynamic.
  assert(isa<TransformOpInterface>(op) &&
         "should implement TransformOpInterface to have "
         "PossibleTopLevelTransformOpTrait");

  if (op->getNumRegions() < 1)
    return op->emitOpError() << "expects at least one region";

  Region *bodyRegion = &op->getRegion(0);
  if (!llvm::hasNItems(*bodyRegion, 1))
    return op->emitOpError() << "expects a single-block region";

  Block *body = &bodyRegion->front();
  if (body->getNumArguments() == 0) {
    return op->emitOpError()
           << "expects the entry block to have at least one argument";
  }
  if (!llvm::isa<TransformHandleTypeInterface>(
          body->getArgument(0).getType())) {
    return op->emitOpError()
           << "expects the first entry block argument to be of type "
              "implementing TransformHandleTypeInterface";
  }
  BlockArgument arg = body->getArgument(0);
  if (op->getNumOperands() != 0) {
    if (arg.getType() != op->getOperand(0).getType()) {
      return op->emitOpError()
             << "expects the type of the block argument to match "
                "the type of the operand";
    }
  }
  for (BlockArgument arg : body->getArguments().drop_front()) {
    if (llvm::isa<TransformHandleTypeInterface, TransformParamTypeInterface,
                  TransformValueHandleTypeInterface>(arg.getType()))
      continue;

    InFlightDiagnostic diag =
        op->emitOpError()
        << "expects trailing entry block arguments to be of type implementing "
           "TransformHandleTypeInterface, TransformValueHandleTypeInterface or "
           "TransformParamTypeInterface";
    diag.attachNote() << "argument #" << arg.getArgNumber() << " does not";
    return diag;
  }

  if (auto *parent =
          op->getParentWithTrait<PossibleTopLevelTransformOpTrait>()) {
    if (op->getNumOperands() != body->getNumArguments()) {
      InFlightDiagnostic diag =
          op->emitOpError()
          << "expects operands to be provided for a nested op";
      diag.attachNote(parent->getLoc())
          << "nested in another possible top-level op";
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Utilities for ParamProducedTransformOpTrait.
//===----------------------------------------------------------------------===//

void transform::detail::getParamProducerTransformOpTraitEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(op->getResults(), effects);
  bool hasPayloadOperands = false;
  for (Value operand : op->getOperands()) {
    onlyReadsHandle(operand, effects);
    if (llvm::isa<TransformHandleTypeInterface,
                  TransformValueHandleTypeInterface>(operand.getType()))
      hasPayloadOperands = true;
  }
  if (hasPayloadOperands)
    onlyReadsPayload(effects);
}

LogicalResult
transform::detail::verifyParamProducerTransformOpTrait(Operation *op) {
  // Interfaces can be attached dynamically, so this cannot be a static
  // assert.
  if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
    llvm::report_fatal_error(
        Twine("ParamProducerTransformOpTrait must be attached to an op that "
              "implements MemoryEffectsOpInterface, found on ") +
        op->getName().getStringRef());
  }
  for (Value result : op->getResults()) {
    if (llvm::isa<TransformParamTypeInterface>(result.getType()))
      continue;
    return op->emitOpError()
           << "ParamProducerTransformOpTrait attached to this op expects "
              "result types to implement TransformParamTypeInterface";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Memory effects.
//===----------------------------------------------------------------------===//

void transform::consumesHandle(
    ValueRange handles,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (Value handle : handles) {
    effects.emplace_back(MemoryEffects::Read::get(), handle,
                         TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Free::get(), handle,
                         TransformMappingResource::get());
  }
}

/// Returns `true` if the given list of effects instances contains an instance
/// with the effect type specified as template parameter.
template <typename EffectTy, typename ResourceTy, typename Range>
static bool hasEffect(Range &&effects) {
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return isa<EffectTy>(effect.getEffect()) &&
           isa<ResourceTy>(effect.getResource());
  });
}

bool transform::isHandleConsumed(Value handle,
                                 transform::TransformOpInterface transform) {
  auto iface = cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffectsOnValue(handle, effects);
  return ::hasEffect<MemoryEffects::Read, TransformMappingResource>(effects) &&
         ::hasEffect<MemoryEffects::Free, TransformMappingResource>(effects);
}

void transform::producesHandle(
    ValueRange handles,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (Value handle : handles) {
    effects.emplace_back(MemoryEffects::Allocate::get(), handle,
                         TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), handle,
                         TransformMappingResource::get());
  }
}

void transform::onlyReadsHandle(
    ValueRange handles,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (Value handle : handles) {
    effects.emplace_back(MemoryEffects::Read::get(), handle,
                         TransformMappingResource::get());
  }
}

void transform::modifiesPayload(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), PayloadIRResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), PayloadIRResource::get());
}

void transform::onlyReadsPayload(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), PayloadIRResource::get());
}

bool transform::doesModifyPayload(transform::TransformOpInterface transform) {
  auto iface = cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  return ::hasEffect<MemoryEffects::Write, PayloadIRResource>(effects);
}

bool transform::doesReadPayload(transform::TransformOpInterface transform) {
  auto iface = cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  return ::hasEffect<MemoryEffects::Read, PayloadIRResource>(effects);
}

void transform::getConsumedBlockArguments(
    Block &block, llvm::SmallDenseSet<unsigned int> &consumedArguments) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  for (Operation &nested : block) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(nested);
    if (!iface)
      continue;

    effects.clear();
    iface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      BlockArgument argument =
          dyn_cast_or_null<BlockArgument>(effect.getValue());
      if (!argument || argument.getOwner() != &block ||
          !isa<MemoryEffects::Free>(effect.getEffect()) ||
          effect.getResource() != transform::TransformMappingResource::get()) {
        continue;
      }
      consumedArguments.insert(argument.getArgNumber());
    }
  }
}

//===----------------------------------------------------------------------===//
// Utilities for TransformOpInterface.
//===----------------------------------------------------------------------===//

SmallVector<OpOperand *> transform::detail::getConsumedHandleOpOperands(
    TransformOpInterface transformOp) {
  SmallVector<OpOperand *> consumedOperands;
  consumedOperands.reserve(transformOp->getNumOperands());
  auto memEffectInterface =
      cast<MemoryEffectOpInterface>(transformOp.getOperation());
  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  for (OpOperand &target : transformOp->getOpOperands()) {
    effects.clear();
    memEffectInterface.getEffectsOnValue(target.get(), effects);
    if (llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
          return isa<transform::TransformMappingResource>(
                     effect.getResource()) &&
                 isa<MemoryEffects::Free>(effect.getEffect());
        })) {
      consumedOperands.push_back(&target);
    }
  }
  return consumedOperands;
}

LogicalResult transform::detail::verifyTransformOpInterface(Operation *op) {
  auto iface = cast<MemoryEffectOpInterface>(op);
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);

  auto effectsOn = [&](Value value) {
    return llvm::make_filter_range(
        effects, [value](const MemoryEffects::EffectInstance &instance) {
          return instance.getValue() == value;
        });
  };

  std::optional<unsigned> firstConsumedOperand;
  for (OpOperand &operand : op->getOpOperands()) {
    auto range = effectsOn(operand.get());
    if (range.empty()) {
      InFlightDiagnostic diag =
          op->emitError() << "TransformOpInterface requires memory effects "
                             "on operands to be specified";
      diag.attachNote() << "no effects specified for operand #"
                        << operand.getOperandNumber();
      return diag;
    }
    if (::hasEffect<MemoryEffects::Allocate, TransformMappingResource>(range)) {
      InFlightDiagnostic diag = op->emitError()
                                << "TransformOpInterface did not expect "
                                   "'allocate' memory effect on an operand";
      diag.attachNote() << "specified for operand #"
                        << operand.getOperandNumber();
      return diag;
    }
    if (!firstConsumedOperand &&
        ::hasEffect<MemoryEffects::Free, TransformMappingResource>(range)) {
      firstConsumedOperand = operand.getOperandNumber();
    }
  }

  if (firstConsumedOperand &&
      !::hasEffect<MemoryEffects::Write, PayloadIRResource>(effects)) {
    InFlightDiagnostic diag =
        op->emitError()
        << "TransformOpInterface expects ops consuming operands to have a "
           "'write' effect on the payload resource";
    diag.attachNote() << "consumes operand #" << *firstConsumedOperand;
    return diag;
  }

  for (OpResult result : op->getResults()) {
    auto range = effectsOn(result);
    if (!::hasEffect<MemoryEffects::Allocate, TransformMappingResource>(
            range)) {
      InFlightDiagnostic diag =
          op->emitError() << "TransformOpInterface requires 'allocate' memory "
                             "effect to be specified for results";
      diag.attachNote() << "no 'allocate' effect specified for result #"
                        << result.getResultNumber();
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entry point.
//===----------------------------------------------------------------------===//

LogicalResult
transform::applyTransforms(Operation *payloadRoot,
                           TransformOpInterface transform,
                           const RaggedArray<MappedValue> &extraMapping,
                           const TransformOptions &options) {
#ifndef NDEBUG
  if (!transform->hasTrait<PossibleTopLevelTransformOpTrait>() ||
      transform->getNumOperands() != 0) {
    transform->emitError()
        << "expected transform to start at the top-level transform op";
    llvm::report_fatal_error("could not run transforms",
                             /*gen_crash_diag=*/false);
  }
#endif // NDEBUG

  TransformState state(transform->getParentRegion(), payloadRoot, extraMapping,
                       options);
  return state.applyTransform(transform).checkAndReport();
}

//===----------------------------------------------------------------------===//
// Generated interface implementation.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.cpp.inc"
