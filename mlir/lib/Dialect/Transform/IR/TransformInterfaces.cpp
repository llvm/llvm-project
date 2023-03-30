//===- TransformInterfaces.cpp - Transform Dialect Interfaces -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
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

  auto result = mappings.try_emplace(region);
  assert(result.second && "the region scope is already present");
  (void)result;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  regionStack.push_back(region);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
}

Operation *transform::TransformState::getTopLevel() const { return topLevel; }

ArrayRef<Operation *>
transform::TransformState::getPayloadOps(Value value) const {
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
    Operation *op, SmallVectorImpl<Value> &handles) const {
  bool found = false;
  for (const Mappings &mapping : llvm::make_second_range(mappings)) {
    auto iterator = mapping.reverse.find(op);
    if (iterator != mapping.reverse.end()) {
      llvm::append_range(handles, iterator->getSecond());
      found = true;
    }
  }

  return success(found);
}

LogicalResult transform::TransformState::getHandlesForPayloadValue(
    Value payloadValue, SmallVectorImpl<Value> &handles) const {
  bool found = false;
  for (const Mappings &mapping : llvm::make_second_range(mappings)) {
    auto iterator = mapping.reverseValues.find(payloadValue);
    if (iterator != mapping.reverseValues.end()) {
      llvm::append_range(handles, iterator->getSecond());
      found = true;
    }
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
  if (handle.getType().isa<transform::TransformHandleTypeInterface>()) {
    SmallVector<Operation *> operations;
    operations.reserve(values.size());
    for (transform::MappedValue value : values) {
      if (auto *op = value.dyn_cast<Operation *>()) {
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

  if (handle.getType().isa<transform::TransformValueHandleTypeInterface>()) {
    SmallVector<Value> payloadValues;
    payloadValues.reserve(values.size());
    for (transform::MappedValue value : values) {
      if (auto v = value.dyn_cast<Value>()) {
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

  assert(handle.getType().isa<transform::TransformParamTypeInterface>() &&
         "unsupported kind of block argument");
  SmallVector<transform::Param> parameters;
  parameters.reserve(values.size());
  for (transform::MappedValue value : values) {
    if (auto attr = value.dyn_cast<Attribute>()) {
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
  assert(value.getType().isa<TransformHandleTypeInterface>() &&
         "wrong handle type");

  for (Operation *target : targets) {
    if (target)
      continue;
    return emitError(value.getLoc())
           << "attempting to assign a null payload op to this transform value";
  }

  auto iface = value.getType().cast<TransformHandleTypeInterface>();
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
  assert(handle.getType().isa<TransformValueHandleTypeInterface>() &&
         "wrong handle type");

  for (Value payload : payloadValues) {
    if (payload)
      continue;
    return emitError(handle.getLoc()) << "attempting to assign a null payload "
                                         "value to this transform handle";
  }

  auto iface = handle.getType().cast<TransformValueHandleTypeInterface>();
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

  auto valueType = value.getType().dyn_cast<TransformParamTypeInterface>();
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
    }
  }
}

LogicalResult
transform::TransformState::replacePayloadOp(Operation *op,
                                            Operation *replacement) {
  // Drop the mapping between the op and all handles that point to it. Don't
  // care if there are on such handles.
  SmallVector<Value> opHandles;
  (void)getHandlesForPayloadOp(op, opHandles);
  for (Value handle : opHandles) {
    Mappings &mappings = getMapping(handle);
    dropMappingEntry(mappings.reverse, op, handle);
  }

#ifndef NDEBUG
  for (Value opResult : op->getResults()) {
    SmallVector<Value> valueHandles;
    (void)getHandlesForPayloadValue(opResult, valueHandles);
    assert(valueHandles.empty() && "expected no mapping to old results");
  }
#endif // NDEBUG

  // TODO: consider invalidating the handles to nested objects here.

  // If replacing with null, that is erasing the mapping, drop the mapping
  // between the handles and the IR objects and return.
  if (!replacement) {
    for (Value handle : opHandles) {
      Mappings &mappings = getMapping(handle);
      dropMappingEntry(mappings.direct, handle, op);
    }
    return success();
  }

  // Otherwise, replace the pointed-to object of all handles while preserving
  // their relative order. First, replace the mapped operation if present.
  for (Value handle : opHandles) {
    Mappings &mappings = getMapping(handle);
    auto it = mappings.direct.find(handle);
    if (it == mappings.direct.end())
      continue;

    SmallVector<Operation *, 2> &association = it->getSecond();
    // Note that an operation may be associated with the handle more than once.
    for (Operation *&mapped : association) {
      if (mapped == op)
        mapped = replacement;
    }
    mappings.reverse[replacement].push_back(handle);
  }

  return success();
}

LogicalResult
transform::TransformState::replacePayloadValue(Value value, Value replacement) {
  SmallVector<Value> valueHandles;
  (void)getHandlesForPayloadValue(value, valueHandles);

  for (Value handle : valueHandles) {
    Mappings &mappings = getMapping(handle);
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
    Operation *payloadOp, Value otherHandle, Value throughValue) {
  // If the op is associated with invalidated handle, skip the check as it
  // may be reading invalid IR.
  if (invalidatedHandles.count(otherHandle))
    return;

  LDBG("--recordOpHandleInvalidationOne\n");
  LLVM_DEBUG(llvm::interleaveComma(potentialAncestors,
                                   DBGS() << "--ancestors: ",
                                   [](Operation *op) { llvm::dbgs() << *op; });
             llvm::dbgs() << "\n");
  for (Operation *ancestor : potentialAncestors) {
    LLVM_DEBUG(DBGS() << "----handle one ancestor: " << *ancestor << "\n");
    LLVM_DEBUG(DBGS() << "----of payload with name: "
                      << payloadOp->getName().getIdentifier() << "\n");
    DEBUG_WITH_TYPE(DEBUG_TYPE_FULL,
                    { (DBGS() << "----of payload: " << *payloadOp << "\n"); });
    if (!ancestor->isAncestor(payloadOp))
      continue;

    // Make sure the error-reporting lambda doesn't capture anything
    // by-reference because it will go out of scope. Additionally, extract
    // location from Payload IR ops because the ops themselves may be
    // deleted before the lambda gets called.
    Location ancestorLoc = ancestor->getLoc();
    Location opLoc = payloadOp->getLoc();
    Operation *owner = consumingHandle.getOwner();
    unsigned operandNo = consumingHandle.getOperandNumber();
    std::optional<Location> throughValueLoc =
        throughValue ? std::make_optional(throughValue.getLoc()) : std::nullopt;
    invalidatedHandles[otherHandle] = [ancestorLoc, opLoc, owner, operandNo,
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
    OpOperand &consumingHandle, ArrayRef<Operation *> potentialAncestors,
    Value payloadValue, Value valueHandle) {
  // If the op is associated with invalidated handle, skip the check as it
  // may be reading invalid IR.
  if (invalidatedHandles.count(valueHandle))
    return;

  for (Operation *ancestor : potentialAncestors) {
    Operation *definingOp;
    std::optional<unsigned> resultNo = std::nullopt;
    unsigned argumentNo, blockNo, regionNo;
    if (auto opResult = payloadValue.dyn_cast<OpResult>()) {
      definingOp = opResult.getOwner();
      resultNo = opResult.getResultNumber();
    } else {
      auto arg = payloadValue.cast<BlockArgument>();
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

    Operation *owner = consumingHandle.getOwner();
    unsigned operandNo = consumingHandle.getOperandNumber();
    Location ancestorLoc = ancestor->getLoc();
    Location opLoc = definingOp->getLoc();
    Location valueLoc = payloadValue.getLoc();
    invalidatedHandles[valueHandle] =
        [valueHandle, owner, operandNo, resultNo, argumentNo, blockNo, regionNo,
         ancestorLoc, opLoc, valueLoc](Location currentLoc) {
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
    Value throughValue) {
  // Iterate over the mapping and invalidate aliasing handles. This is quite
  // expensive and only necessary for error reporting in case of transform
  // dialect misuse with dangling handles. Iteration over the handles is based
  // on the assumption that the number of handles is significantly less than the
  // number of IR objects (operations and values). Alternatively, we could walk
  // the IR nested in each payload op associated with the given handle and look
  // for handles associated with each operation and value.
  for (const Mappings &mapping : llvm::make_second_range(mappings)) {
    // Go over all op handle mappings and mark as invalidated any handle
    // pointing to any of the payload ops associated with the given handle or
    // any op nested in them.
    for (const auto &[payloadOp, otherHandles] : mapping.reverse) {
      for (Value otherHandle : otherHandles)
        recordOpHandleInvalidationOne(handle, potentialAncestors, payloadOp,
                                      otherHandle, throughValue);
    }
    // Go over all value handle mappings and mark as invalidated any handle
    // pointing to any result of the payload op associated with the given handle
    // or any op nested in them. Similarly invalidate handles to argument of
    // blocks belonging to any region of any payload op associated with the
    // given handle or any op nested in them.
    for (const auto &[payloadValue, valueHandles] : mapping.reverseValues) {
      for (Value valueHandle : valueHandles)
        recordValueHandleInvalidationByOpHandleOne(handle, potentialAncestors,
                                                   payloadValue, valueHandle);
    }
  }
}

void transform::TransformState::recordValueHandleInvalidation(
    OpOperand &valueHandle) {
  // Invalidate other handles to the same value.
  for (Value payloadValue : getPayloadValues(valueHandle.get())) {
    SmallVector<Value> otherValueHandles;
    (void)getHandlesForPayloadValue(payloadValue, otherValueHandles);
    for (Value otherHandle : otherValueHandles) {
      Operation *owner = valueHandle.getOwner();
      unsigned operandNo = valueHandle.getOperandNumber();
      Location valueLoc = payloadValue.getLoc();
      invalidatedHandles[otherHandle] = [otherHandle, owner, operandNo,
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

    if (auto opResult = payloadValue.dyn_cast<OpResult>()) {
      Operation *payloadOp = opResult.getOwner();
      recordOpHandleInvalidation(valueHandle, payloadOp, payloadValue);
    } else {
      auto arg = payloadValue.dyn_cast<BlockArgument>();
      for (Operation &payloadOp : *arg.getOwner())
        recordOpHandleInvalidation(valueHandle, &payloadOp, payloadValue);
    }
  }
}

LogicalResult transform::TransformState::checkAndRecordHandleInvalidation(
    TransformOpInterface transform) {
  LDBG("--Start checkAndRecordHandleInvalidation\n");
  auto memoryEffectsIface =
      cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectsIface.getEffectsOnResource(
      transform::TransformMappingResource::get(), effects);

  for (OpOperand &target : transform->getOpOperands()) {
    LLVM_DEBUG(DBGS() << "----iterate on handle: " << target.get() << "\n");
    // If the operand uses an invalidated handle, report it.
    auto it = invalidatedHandles.find(target.get());
    if (!transform.allowsRepeatedHandleOperands() &&
        it != invalidatedHandles.end()) {
      LLVM_DEBUG(
          DBGS() << "--End checkAndRecordHandleInvalidation -> FAILURE\n");
      return it->getSecond()(transform->getLoc()), failure();
    }

    // Invalidate handles pointing to the operations nested in the operation
    // associated with the handle consumed by this operation.
    auto consumesTarget = [&](const MemoryEffects::EffectInstance &effect) {
      return isa<MemoryEffects::Free>(effect.getEffect()) &&
             effect.getValue() == target.get();
    };
    if (llvm::any_of(effects, consumesTarget)) {
      LLVM_DEBUG(DBGS() << "----found consume effect -> SKIP\n");
      if (target.get().getType().isa<TransformHandleTypeInterface>()) {
        LDBG("----recordOpHandleInvalidation\n");
        ArrayRef<Operation *> payloadOps = getPayloadOps(target.get());
        recordOpHandleInvalidation(target, payloadOps);
      } else if (target.get()
                     .getType()
                     .isa<TransformValueHandleTypeInterface>()) {
        LDBG("----recordValueHandleInvalidation\n");
        recordValueHandleInvalidation(target);
      } else {
        LDBG("----not a TransformHandle -> SKIP AND DROP ON THE FLOOR\n");
      }
    } else {
      LLVM_DEBUG(DBGS() << "----no consume effect -> SKIP\n");
    }
  }

  LDBG("--End checkAndRecordHandleInvalidation -> SUCCESS\n");
  return success();
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

DiagnosedSilenceableFailure
transform::TransformState::applyTransform(TransformOpInterface transform) {
  LLVM_DEBUG(DBGS() << "\n"; DBGS() << "applying: " << transform << "\n");
  auto printOnFailureRAII = llvm::make_scope_exit([this] {
    (void)this;
    DEBUG_WITH_TYPE(DEBUG_PRINT_AFTER_ALL, {
      DBGS() << "Top-level payload:\n";
      getTopLevel()->print(llvm::dbgs(),
                           mlir::OpPrintingFlags().printGenericOpForm());
    });
  });
  if (options.getExpensiveChecksEnabled()) {
    LDBG("ExpensiveChecksEnabled\n");
    if (failed(checkAndRecordHandleInvalidation(transform)))
      return DiagnosedSilenceableFailure::definiteFailure();

    for (OpOperand &operand : transform->getOpOperands()) {
      LLVM_DEBUG(DBGS() << "iterate on handle: " << operand.get() << "\n");
      if (!isHandleConsumed(operand.get(), transform)) {
        LDBG("--handle not consumed -> SKIP\n");
        continue;
      }
      LDBG("--handle is consumed\n");

      Type operandType = operand.get().getType();
      if (operandType.isa<TransformHandleTypeInterface>()) {
        LLVM_DEBUG(
            DBGS() << "--checkRepeatedConsumptionInOperand for Operation*\n");
        DiagnosedSilenceableFailure check =
            checkRepeatedConsumptionInOperand<Operation *>(
                getPayloadOps(operand.get()), transform,
                operand.getOperandNumber());
        if (!check.succeeded()) {
          LDBG("----FAILED\n");
          return check;
        }
      } else if (operandType.isa<TransformValueHandleTypeInterface>()) {
        LDBG("--checkRepeatedConsumptionInOperand For Value\n");
        DiagnosedSilenceableFailure check =
            checkRepeatedConsumptionInOperand<Value>(
                getPayloadValues(operand.get()), transform,
                operand.getOperandNumber());
        if (!check.succeeded()) {
          LDBG("----FAILED\n");
          return check;
        }
      } else {
        LDBG("--not a TransformHandle -> SKIP AND DROP ON THE FLOOR\n");
      }
    }
  }

  // Find which operands are consumed.
  DenseSet<unsigned> consumedOperands;
  auto memEffectInterface =
      cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  for (OpOperand &target : transform->getOpOperands()) {
    effects.clear();
    memEffectInterface.getEffectsOnValue(target.get(), effects);
    if (llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
          return isa<transform::TransformMappingResource>(
                     effect.getResource()) &&
                 isa<MemoryEffects::Free>(effect.getEffect());
        })) {
      consumedOperands.insert(target.getOperandNumber());
    }
  }

  // Remember the results of the payload ops associated with the consumed
  // op handles or the ops defining the value handles so we can drop the
  // association with them later. This must happen here because the
  // transformation may destroy or mutate them so we cannot traverse the payload
  // IR after that.
  SmallVector<Value> origOpFlatResults;
  SmallVector<Operation *> origAssociatedOps;
  for (unsigned index : consumedOperands) {
    Value operand = transform->getOperand(index);
    if (operand.getType().isa<TransformHandleTypeInterface>()) {
      for (Operation *payloadOp : getPayloadOps(operand))
        llvm::append_range(origOpFlatResults, payloadOp->getResults());
      continue;
    }
    if (operand.getType().isa<TransformValueHandleTypeInterface>()) {
      for (Value payloadValue : getPayloadValues(operand)) {
        if (payloadValue.isa<OpResult>()) {
          origAssociatedOps.push_back(payloadValue.getDefiningOp());
          continue;
        }
        llvm::append_range(
            origAssociatedOps,
            llvm::map_range(*payloadValue.cast<BlockArgument>().getOwner(),
                            [](Operation &op) { return &op; }));
      }
      continue;
    }
    DiagnosedDefiniteFailure diag =
        emitDefiniteFailure(transform->getLoc())
        << "unexpectedly consumed a value that is not a handle as operand #"
        << index;
    diag.attachNote(operand.getLoc())
        << "value defined here with type " << operand.getType();
    return diag;
  }

  // Compute the result but do not short-circuit the silenceable failure case as
  // we still want the handles to propagate properly so the "suppress" mode can
  // proceed on a best effort basis.
  transform::TransformResults results(transform->getNumResults());
  DiagnosedSilenceableFailure result(transform.apply(results, *this));
  if (result.isDefiniteFailure())
    return result;

  // If a silenceable failure was produced, some results may be unset, set them
  // to empty lists.
  if (result.isSilenceableFailure()) {
    for (OpResult opResult : transform->getResults()) {
      if (results.isSet(opResult.getResultNumber()))
        continue;

      if (opResult.getType().isa<TransformParamTypeInterface>())
        results.setParams(opResult, {});
      else if (opResult.getType().isa<TransformValueHandleTypeInterface>())
        results.setValues(opResult, {});
      else
        results.set(opResult, {});
    }
  }

  // Remove the mapping for the operand if it is consumed by the operation. This
  // allows us to catch use-after-free with assertions later on.
  for (unsigned index : consumedOperands) {
    Value operand = transform->getOperand(index);
    if (operand.getType().isa<TransformHandleTypeInterface>()) {
      forgetMapping(operand, origOpFlatResults);
    } else if (operand.getType().isa<TransformValueHandleTypeInterface>()) {
      forgetValueMapping(operand, origAssociatedOps);
    }
  }

  for (OpResult result : transform->getResults()) {
    assert(result.getDefiningOp() == transform.getOperation() &&
           "payload IR association for a value other than the result of the "
           "current transform op");
    if (result.getType().isa<TransformParamTypeInterface>()) {
      assert(results.isParam(result.getResultNumber()) &&
             "expected parameters for the parameter-typed result");
      if (failed(
              setParams(result, results.getParams(result.getResultNumber())))) {
        return DiagnosedSilenceableFailure::definiteFailure();
      }
    } else if (result.getType().isa<TransformValueHandleTypeInterface>()) {
      assert(results.isValue(result.getResultNumber()) &&
             "expected values for value-type-result");
      if (failed(setPayloadValues(
              result, results.getValues(result.getResultNumber())))) {
        return DiagnosedSilenceableFailure::definiteFailure();
      }
    } else {
      assert(!results.isParam(result.getResultNumber()) &&
             "expected payload ops for the non-parameter typed result");
      if (failed(
              setPayloadOps(result, results.get(result.getResultNumber())))) {
        return DiagnosedSilenceableFailure::definiteFailure();
      }
    }
  }

  printOnFailureRAII.release();
  DEBUG_WITH_TYPE(DEBUG_PRINT_AFTER_ALL, {
    DBGS() << "Top-level payload:\n";
    getTopLevel()->print(llvm::dbgs());
  });
  return result;
}

//===----------------------------------------------------------------------===//
// TransformState::Extension
//===----------------------------------------------------------------------===//

transform::TransformState::Extension::~Extension() = default;

LogicalResult
transform::TransformState::Extension::replacePayloadOp(Operation *op,
                                                       Operation *replacement) {
  SmallVector<Value> handles;
  if (failed(state.getHandlesForPayloadOp(op, handles)))
    return failure();

  // TODO: we may need to invalidate handles to operations and values nested in
  // the operation being replaced.
  return state.replacePayloadOp(op, replacement);
}

LogicalResult
transform::TransformState::Extension::replacePayloadValue(Value value,
                                                          Value replacement) {
  SmallVector<Value> handles;
  if (failed(state.getHandlesForPayloadValue(value, handles)))
    return failure();

  return state.replacePayloadValue(value, replacement);
}

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//

transform::TransformResults::TransformResults(unsigned numSegments) {
  operations.appendEmptyRows(numSegments);
  params.appendEmptyRows(numSegments);
  values.appendEmptyRows(numSegments);
}

void transform::TransformResults::set(OpResult value,
                                      ArrayRef<Operation *> ops) {
  int64_t position = value.getResultNumber();
  assert(position < static_cast<int64_t>(operations.size()) &&
         "setting results for a non-existent handle");
  assert(operations[position].data() == nullptr && "results already set");
  assert(params[position].data() == nullptr &&
         "another kind of results already set");
  assert(values[position].data() == nullptr &&
         "another kind of results already set");
  operations.replace(position, ops);
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
// Utilities for TransformEachOpTrait.
//===----------------------------------------------------------------------===//

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
    if (res.getType().template isa<TransformHandleTypeInterface>() &&
        !ptr.is<Operation *>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce an Operation * for result #"
                        << res.getResultNumber();
    }
    if (res.getType().template isa<TransformParamTypeInterface>() &&
        !ptr.is<Attribute>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce an Attribute for result #"
                        << res.getResultNumber();
    }
    if (res.getType().template isa<TransformValueHandleTypeInterface>() &&
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
    if (r.getType().isa<TransformParamTypeInterface>()) {
      transformResults.setParams(r,
                                 castVector<Attribute>(transposed[position]));
    } else if (r.getType().isa<TransformValueHandleTypeInterface>()) {
      transformResults.setValues(r, castVector<Value>(transposed[position]));
    } else {
      transformResults.set(r, castVector<Operation *>(transposed[position]));
    }
  }
}

//===----------------------------------------------------------------------===//
// Utilities for PossibleTopLevelTransformOpTrait.
//===----------------------------------------------------------------------===//

void transform::detail::prepareValueMappings(
    SmallVectorImpl<SmallVector<transform::MappedValue>> &mappings,
    ValueRange values, const transform::TransformState &state) {
  for (Value operand : values) {
    SmallVector<MappedValue> &mapped = mappings.emplace_back();
    if (operand.getType().isa<TransformHandleTypeInterface>()) {
      llvm::append_range(mapped, state.getPayloadOps(operand));
    } else if (operand.getType().isa<TransformValueHandleTypeInterface>()) {
      llvm::append_range(mapped, state.getPayloadValues(operand));
    } else {
      assert(operand.getType().isa<TransformParamTypeInterface>() &&
             "unsupported kind of transform dialect value");
      llvm::append_range(mapped, state.getParams(operand));
    }
  }
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

    targets.push_back(state.getTopLevel());
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
  if (!body->getArgument(0).getType().isa<TransformHandleTypeInterface>()) {
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
    if (arg.getType()
            .isa<TransformHandleTypeInterface, TransformParamTypeInterface,
                 TransformValueHandleTypeInterface>())
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
    if (operand.getType()
            .isa<TransformHandleTypeInterface,
                 TransformValueHandleTypeInterface>())
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
    if (result.getType().isa<TransformParamTypeInterface>())
      continue;
    return op->emitOpError()
           << "ParamProducerTransformOpTrait attached to this op expects "
              "result types to implement TransformParamTypeInterface";
  }
  return success();
}

DiagnosedSilenceableFailure transform::detail::transformWithPatternsApply(
    Operation *transformOp, Operation *target, ApplyToEachResultList &results,
    TransformState &state,
    function_ref<void(RewritePatternSet &)> populatePatterns) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return emitDefiniteFailure(transformOp)
           << "applies only to isolated-from-above targets because it needs to "
              "apply patterns greedily";
  }
  RewritePatternSet patterns(transformOp->getContext());
  populatePatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return emitDefiniteFailure(transformOp) << "failed to apply patterns";

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
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

//===----------------------------------------------------------------------===//
// Utilities for TransformOpInterface.
//===----------------------------------------------------------------------===//

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

  std::optional<unsigned> firstConsumedOperand = std::nullopt;
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
