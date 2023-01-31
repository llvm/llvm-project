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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "transform-dialect"
#define DEBUG_PRINT_AFTER_ALL "transform-dialect-print-top-level-after-all"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

using namespace mlir;

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//

constexpr const Value transform::TransformState::kTopLevelValue;

transform::TransformState::TransformState(
    Region *region, Operation *payloadRoot,
    ArrayRef<ArrayRef<MappedValue>> extraMappings,
    const TransformOptions &options)
    : topLevel(payloadRoot), options(options) {
  topLevelMappedValues.reserve(extraMappings.size());
  for (ArrayRef<MappedValue> mapping : extraMappings) {
    size_t start = topLevelMappedValueStorage.size();
    llvm::append_range(topLevelMappedValueStorage, mapping);
    topLevelMappedValues.push_back(
        ArrayRef<MappedValue>(topLevelMappedValueStorage)
            .slice(start, mapping.size()));
  }

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
  assert(iter != operationMapping.end() &&
         "cannot find mapping for payload handle (param handle provided?)");
  return iter->getSecond();
}

ArrayRef<Attribute> transform::TransformState::getParams(Value value) const {
  const ParamMapping &mapping = getMapping(value).params;
  auto iter = mapping.find(value);
  assert(iter != mapping.end() &&
         "cannot find mapping for param handle (payload handle provided?)");
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

LogicalResult
transform::TransformState::mapBlockArgument(BlockArgument argument,
                                            ArrayRef<MappedValue> values) {
  if (argument.getType().isa<TransformHandleTypeInterface>()) {
    SmallVector<Operation *> operations;
    operations.reserve(values.size());
    for (MappedValue value : values) {
      if (auto *op = value.dyn_cast<Operation *>()) {
        operations.push_back(op);
        continue;
      }
      return emitError(argument.getLoc())
             << "wrong kind of value provided for top-level operation handle";
    }
    return setPayloadOps(argument, operations);
  }

  assert(argument.getType().isa<TransformParamTypeInterface>() &&
         "unsupported kind of block argument");
  SmallVector<Param> parameters;
  parameters.reserve(values.size());
  for (MappedValue value : values) {
    if (auto attr = value.dyn_cast<Attribute>()) {
      parameters.push_back(attr);
      continue;
    }
    return emitError(argument.getLoc())
           << "wrong kind of value provided for top-level parameter";
  }
  return setParams(argument, parameters);
}

LogicalResult
transform::TransformState::setPayloadOps(Value value,
                                         ArrayRef<Operation *> targets) {
  assert(value != kTopLevelValue &&
         "attempting to reset the transformation root");
  assert(!value.getType().isa<TransformParamTypeInterface>() &&
         "cannot associate payload ops with a value of parameter type");

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

void transform::TransformState::dropReverseMapping(Mappings &mappings,
                                                   Operation *op, Value value) {
  auto it = mappings.reverse.find(op);
  if (it == mappings.reverse.end())
    return;

  llvm::erase_value(it->getSecond(), value);
  if (it->getSecond().empty())
    mappings.reverse.erase(it);
}

void transform::TransformState::removePayloadOps(Value value) {
  Mappings &mappings = getMapping(value);
  for (Operation *op : mappings.direct[value])
    dropReverseMapping(mappings, op, value);
  mappings.direct.erase(value);
}

LogicalResult transform::TransformState::updatePayloadOps(
    Value value, function_ref<Operation *(Operation *)> callback) {
  Mappings &mappings = getMapping(value);
  auto it = mappings.direct.find(value);
  assert(it != mappings.direct.end() && "unknown handle");
  SmallVector<Operation *, 2> &association = it->getSecond();
  SmallVector<Operation *, 2> updated;
  updated.reserve(association.size());

  for (Operation *op : association) {
    dropReverseMapping(mappings, op, value);
    if (Operation *updatedOp = callback(op)) {
      updated.push_back(updatedOp);
      mappings.reverse[updatedOp].push_back(value);
    }
  }

  auto iface = value.getType().cast<TransformHandleTypeInterface>();
  DiagnosedSilenceableFailure result =
      iface.checkPayload(value.getLoc(), updated);
  if (failed(result.checkAndReport()))
    return failure();

  it->second = updated;
  return success();
}

void transform::TransformState::recordHandleInvalidationOne(
    OpOperand &handle, Operation *payloadOp, Value otherHandle) {
  ArrayRef<Operation *> potentialAncestors = getPayloadOps(handle.get());
  // If the op is associated with invalidated handle, skip the check as it
  // may be reading invalid IR.
  if (invalidatedHandles.count(otherHandle))
    return;

  for (Operation *ancestor : potentialAncestors) {
    if (!ancestor->isAncestor(payloadOp))
      continue;

    // Make sure the error-reporting lambda doesn't capture anything
    // by-reference because it will go out of scope. Additionally, extract
    // location from Payload IR ops because the ops themselves may be
    // deleted before the lambda gets called.
    Location ancestorLoc = ancestor->getLoc();
    Location opLoc = payloadOp->getLoc();
    Operation *owner = handle.getOwner();
    unsigned operandNo = handle.getOperandNumber();
    invalidatedHandles[otherHandle] = [ancestorLoc, opLoc, owner, operandNo,
                                       otherHandle](Location currentLoc) {
      InFlightDiagnostic diag = emitError(currentLoc)
                                << "op uses a handle invalidated by a "
                                   "previously executed transform op";
      diag.attachNote(otherHandle.getLoc()) << "handle to invalidated ops";
      diag.attachNote(owner->getLoc())
          << "invalidated by this transform op that consumes its operand #"
          << operandNo
          << " and invalidates handles to payload ops nested in payload "
             "ops associated with the consumed handle";
      diag.attachNote(ancestorLoc) << "ancestor payload op";
      diag.attachNote(opLoc) << "nested payload op";
    };
  }
}

void transform::TransformState::recordHandleInvalidation(OpOperand &handle) {
  for (const Mappings &mapping : llvm::make_second_range(mappings))
    for (const auto &[payloadOp, otherHandles] : mapping.reverse)
      for (Value otherHandle : otherHandles)
        recordHandleInvalidationOne(handle, payloadOp, otherHandle);
}

LogicalResult transform::TransformState::checkAndRecordHandleInvalidation(
    TransformOpInterface transform) {
  auto memoryEffectsIface =
      cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance> effects;
  memoryEffectsIface.getEffectsOnResource(
      transform::TransformMappingResource::get(), effects);

  for (OpOperand &target : transform->getOpOperands()) {
    // If the operand uses an invalidated handle, report it.
    auto it = invalidatedHandles.find(target.get());
    if (!transform.allowsRepeatedHandleOperands() &&
        it != invalidatedHandles.end())
      return it->getSecond()(transform->getLoc()), failure();

    // Invalidate handles pointing to the operations nested in the operation
    // associated with the handle consumed by this operation.
    auto consumesTarget = [&](const MemoryEffects::EffectInstance &effect) {
      return isa<MemoryEffects::Free>(effect.getEffect()) &&
             effect.getValue() == target.get();
    };
    if (llvm::any_of(effects, consumesTarget))
      recordHandleInvalidation(target);
  }

  return success();
}

DiagnosedSilenceableFailure
transform::TransformState::applyTransform(TransformOpInterface transform) {
  LLVM_DEBUG(DBGS() << "applying: " << transform << "\n");
  auto printOnFailureRAII = llvm::make_scope_exit([this] {
    (void)this;
    DEBUG_WITH_TYPE(DEBUG_PRINT_AFTER_ALL, {
      DBGS() << "Top-level payload:\n";
      getTopLevel()->print(llvm::dbgs(),
                           mlir::OpPrintingFlags().printGenericOpForm());
    });
  });
  if (options.getExpensiveChecksEnabled()) {
    if (failed(checkAndRecordHandleInvalidation(transform)))
      return DiagnosedSilenceableFailure::definiteFailure();

    for (OpOperand &operand : transform->getOpOperands()) {
      if (!isHandleConsumed(operand.get(), transform))
        continue;

      DenseSet<Operation *> seen;
      for (Operation *op : getPayloadOps(operand.get())) {
        if (!seen.insert(op).second) {
          DiagnosedSilenceableFailure diag =
              transform.emitSilenceableError()
              << "a handle passed as operand #" << operand.getOperandNumber()
              << " and consumed by this operation points to a payload "
                 "operation more than once";
          diag.attachNote(op->getLoc()) << "repeated target op";
          return diag;
        }
      }
    }
  }

  transform::TransformResults results(transform->getNumResults());
  // Compute the result but do not short-circuit the silenceable failure case as
  // we still want the handles to propagate properly so the "suppress" mode can
  // proceed on a best effort basis.
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
      else
        results.set(opResult, {});
    }
  }

  // Remove the mapping for the operand if it is consumed by the operation. This
  // allows us to catch use-after-free with assertions later on.
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
      removePayloadOps(target.get());
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

  for (Value handle : handles) {
    LogicalResult result =
        state.updatePayloadOps(handle, [&](Operation *current) {
          return current == op ? replacement : current;
        });
    if (failed(result))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//

transform::TransformResults::TransformResults(unsigned numSegments) {
  segments.resize(numSegments,
                  ArrayRef<Operation *>(nullptr, static_cast<size_t>(0)));
  paramSegments.resize(numSegments, ArrayRef<TransformState::Param>(
                                        nullptr, static_cast<size_t>(0)));
}

void transform::TransformResults::set(OpResult value,
                                      ArrayRef<Operation *> ops) {
  int64_t position = value.getResultNumber();
  assert(position < static_cast<int64_t>(segments.size()) &&
         "setting results for a non-existent handle");
  assert(segments[position].data() == nullptr && "results already set");
  int64_t start = operations.size();
  llvm::append_range(operations, ops);
  segments[position] = ArrayRef(operations).drop_front(start);
}

void transform::TransformResults::setParams(
    OpResult value, ArrayRef<transform::TransformState::Param> params) {
  int64_t position = value.getResultNumber();
  assert(position < static_cast<int64_t>(paramSegments.size()) &&
         "setting params for a non-existent handle");
  assert(paramSegments[position].data() == nullptr && "params already set");
  size_t start = this->params.size();
  llvm::append_range(this->params, params);
  paramSegments[position] = ArrayRef(this->params).drop_front(start);
}

ArrayRef<Operation *>
transform::TransformResults::get(unsigned resultNumber) const {
  assert(resultNumber < segments.size() &&
         "querying results for a non-existent handle");
  assert(segments[resultNumber].data() != nullptr &&
         "querying unset results (param expected?)");
  return segments[resultNumber];
}

ArrayRef<transform::TransformState::Param>
transform::TransformResults::getParams(unsigned resultNumber) const {
  assert(resultNumber < paramSegments.size() &&
         "querying params for a non-existent handle");
  assert(paramSegments[resultNumber].data() != nullptr &&
         "querying unset params (payload ops expected?)");
  return paramSegments[resultNumber];
}

bool transform::TransformResults::isParam(unsigned resultNumber) const {
  assert(resultNumber < paramSegments.size() &&
         "querying association for a non-existent handle");
  return paramSegments[resultNumber].data() != nullptr;
}

bool transform::TransformResults::isSet(unsigned resultNumber) const {
  assert(resultNumber < paramSegments.size() &&
         "querying association for a non-existent handle");
  return paramSegments[resultNumber].data() != nullptr ||
         segments[resultNumber].data() != nullptr;
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
    if (ptr.isNull()) {
      return emitDiag() << "null result #" << res.getResultNumber()
                        << " produced";
    }
    if (ptr.is<Operation *>() &&
        !res.getType().template isa<TransformHandleTypeInterface>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce an Attribute for result #"
                        << res.getResultNumber();
    }
    if (ptr.is<Attribute>() &&
        !res.getType().template isa<TransformParamTypeInterface>()) {
      return emitDiag() << "application of " << transformOpName
                        << " expected to produce an Operation * for result #"
                        << res.getResultNumber();
    }
  }
  return success();
}

void transform::detail::setApplyToOneResults(
    Operation *transformOp, TransformResults &transformResults,
    ArrayRef<ApplyToEachResultList> results) {
  for (OpResult r : transformOp->getResults()) {
    if (r.getType().isa<TransformParamTypeInterface>()) {
      auto params = llvm::to_vector(
          llvm::map_range(results, [r](const ApplyToEachResultList &oneResult) {
            return oneResult[r.getResultNumber()].get<Attribute>();
          }));
      transformResults.setParams(r, params);
    } else {
      auto payloads = llvm::to_vector(
          llvm::map_range(results, [r](const ApplyToEachResultList &oneResult) {
            return oneResult[r.getResultNumber()].get<Operation *>();
          }));
      transformResults.set(r, payloads);
    }
  }
}

//===----------------------------------------------------------------------===//
// Utilities for PossibleTopLevelTransformOpTrait.
//===----------------------------------------------------------------------===//

LogicalResult transform::detail::mapPossibleTopLevelTransformOpBlockArguments(
    TransformState &state, Operation *op, Region &region) {
  SmallVector<Operation *> targets;
  SmallVector<SmallVector<MappedValue>> extraMappings;
  if (op->getNumOperands() != 0) {
    llvm::append_range(targets, state.getPayloadOps(op->getOperand(0)));
    for (Value operand : op->getOperands().drop_front()) {
      SmallVector<MappedValue> &mapped = extraMappings.emplace_back();
      if (operand.getType().isa<TransformHandleTypeInterface>()) {
        llvm::append_range(mapped, state.getPayloadOps(operand));
      } else {
        assert(operand.getType().isa<TransformParamTypeInterface>() &&
               "unsupported kind of transform dialect value");
        llvm::append_range(mapped, state.getParams(operand));
      }
    }
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
            .isa<TransformHandleTypeInterface, TransformParamTypeInterface>())
      continue;

    InFlightDiagnostic diag =
        op->emitOpError()
        << "expects trailing entry block arguments to be of type implementing "
           "TransformHandleTypeInterface or TransformParamTypeInterface";
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
    if (operand.getType().isa<TransformHandleTypeInterface>())
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
                           ArrayRef<ArrayRef<MappedValue>> extraMapping,
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
