//===- TransformInterfaces.cpp - Transform Dialect Interfaces -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//

constexpr const Value transform::TransformState::kTopLevelValue;

transform::TransformState::TransformState(Region &region, Operation *root)
    : topLevel(root) {
  auto result = mappings.try_emplace(&region);
  assert(result.second && "the region scope is already present");
  (void)result;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  regionStack.push_back(&region);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
}

Operation *transform::TransformState::getTopLevel() const { return topLevel; }

ArrayRef<Operation *>
transform::TransformState::getPayloadOps(Value value) const {
  const TransformOpMapping &operationMapping = getMapping(value).direct;
  auto iter = operationMapping.find(value);
  assert(iter != operationMapping.end() && "unknown handle");
  return iter->getSecond();
}

Value transform::TransformState::getHandleForPayloadOp(Operation *op) const {
  for (const Mappings &mapping : llvm::make_second_range(mappings)) {
    if (Value handle = mapping.reverse.lookup(op))
      return handle;
  }
  return Value();
}

LogicalResult transform::TransformState::tryEmplaceReverseMapping(
    Mappings &map, Operation *operation, Value handle) {
  auto insertionResult = map.reverse.insert({operation, handle});
  if (!insertionResult.second) {
    InFlightDiagnostic diag = operation->emitError()
                              << "operation tracked by two handles";
    diag.attachNote(handle.getLoc()) << "handle";
    diag.attachNote(insertionResult.first->second.getLoc()) << "handle";
    return diag;
  }
  return success();
}

LogicalResult
transform::TransformState::setPayloadOps(Value value,
                                         ArrayRef<Operation *> targets) {
  assert(value != kTopLevelValue &&
         "attempting to reset the transformation root");

  if (value.use_empty())
    return success();

  // Setting new payload for the value without cleaning it first is a misuse of
  // the API, assert here.
  SmallVector<Operation *> storedTargets(targets.begin(), targets.end());
  Mappings &mappings = getMapping(value);
  bool inserted =
      mappings.direct.insert({value, std::move(storedTargets)}).second;
  assert(inserted && "value is already associated with another list");
  (void)inserted;

  // Having multiple handles to the same operation is an error in the transform
  // expressed using the dialect and may be constructed by valid API calls from
  // valid IR. Emit an error here.
  for (Operation *op : targets) {
    if (failed(tryEmplaceReverseMapping(mappings, op, value)))
      return failure();
  }

  return success();
}

void transform::TransformState::removePayloadOps(Value value) {
  Mappings &mappings = getMapping(value);
  for (Operation *op : mappings.direct[value])
    mappings.reverse.erase(op);
  mappings.direct.erase(value);
}

LogicalResult transform::TransformState::updatePayloadOps(
    Value value, function_ref<Operation *(Operation *)> callback) {
  Mappings &mappings = getMapping(value);
  auto it = mappings.direct.find(value);
  assert(it != mappings.direct.end() && "unknown handle");
  SmallVector<Operation *> &association = it->getSecond();
  SmallVector<Operation *> updated;
  updated.reserve(association.size());

  for (Operation *op : association) {
    mappings.reverse.erase(op);
    if (Operation *updatedOp = callback(op)) {
      updated.push_back(updatedOp);
      if (failed(tryEmplaceReverseMapping(mappings, updatedOp, value)))
        return failure();
    }
  }

  std::swap(association, updated);
  return success();
}

LogicalResult
transform::TransformState::applyTransform(TransformOpInterface transform) {
  transform::TransformResults results(transform->getNumResults());
  if (failed(transform.apply(results, *this)))
    return failure();

  // Remove the mapping for the operand if it is consumed by the operation. This
  // allows us to catch use-after-free with assertions later on.
  auto memEffectInterface =
      cast<MemoryEffectOpInterface>(transform.getOperation());
  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  for (Value target : transform->getOperands()) {
    effects.clear();
    memEffectInterface.getEffectsOnValue(target, effects);
    if (llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
          return isa<transform::TransformMappingResource>(
                     effect.getResource()) &&
                 isa<MemoryEffects::Free>(effect.getEffect());
        })) {
      removePayloadOps(target);
    }
  }

  for (auto &en : llvm::enumerate(transform->getResults())) {
    assert(en.value().getDefiningOp() == transform.getOperation() &&
           "payload IR association for a value other than the result of the "
           "current transform op");
    if (failed(setPayloadOps(en.value(), results.get(en.index()))))
      return failure();
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
  return state.updatePayloadOps(state.getHandleForPayloadOp(op),
                                [&](Operation *current) {
                                  return current == op ? replacement : current;
                                });
}

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//

transform::TransformResults::TransformResults(unsigned numSegments) {
  segments.resize(numSegments,
                  ArrayRef<Operation *>(nullptr, static_cast<size_t>(0)));
}

void transform::TransformResults::set(OpResult value,
                                      ArrayRef<Operation *> ops) {
  unsigned position = value.getResultNumber();
  assert(position < segments.size() &&
         "setting results for a non-existent handle");
  assert(segments[position].data() == nullptr && "results already set");
  unsigned start = operations.size();
  llvm::append_range(operations, ops);
  segments[position] = makeArrayRef(operations).drop_front(start);
}

ArrayRef<Operation *>
transform::TransformResults::get(unsigned resultNumber) const {
  assert(resultNumber < segments.size() &&
         "querying results for a non-existent handle");
  assert(segments[resultNumber].data() != nullptr && "querying unset results");
  return segments[resultNumber];
}

//===----------------------------------------------------------------------===//
// Utilities for PossibleTopLevelTransformOpTrait.
//===----------------------------------------------------------------------===//

LogicalResult transform::detail::mapPossibleTopLevelTransformOpBlockArguments(
    TransformState &state, Operation *op) {
  SmallVector<Operation *> targets;
  if (op->getNumOperands() != 0)
    llvm::append_range(targets, state.getPayloadOps(op->getOperand(0)));
  else
    targets.push_back(state.getTopLevel());

  return state.mapBlockArguments(op->getRegion(0).front().getArgument(0),
                                 targets);
}

LogicalResult
transform::detail::verifyPossibleTopLevelTransformOpTrait(Operation *op) {
  // Attaching this trait without the interface is a misuse of the API, but it
  // cannot be caught via a static_assert because interface registration is
  // dynamic.
  assert(isa<TransformOpInterface>(op) &&
         "should implement TransformOpInterface to have "
         "PossibleTopLevelTransformOpTrait");

  if (op->getNumRegions() != 1)
    return op->emitOpError() << "expects one region";

  Region *bodyRegion = &op->getRegion(0);
  if (!llvm::hasNItems(*bodyRegion, 1))
    return op->emitOpError() << "expects a single-block region";

  Block *body = &bodyRegion->front();
  if (body->getNumArguments() != 1 ||
      !body->getArgumentTypes()[0].isa<pdl::OperationType>()) {
    return op->emitOpError()
           << "expects the entry block to have one argument of type "
           << pdl::OperationType::get(op->getContext());
  }

  if (auto *parent =
          op->getParentWithTrait<PossibleTopLevelTransformOpTrait>()) {
    if (op->getNumOperands() == 0) {
      InFlightDiagnostic diag =
          op->emitOpError()
          << "expects the root operation to be provided for a nested op";
      diag.attachNote(parent->getLoc())
          << "nested in another possible top-level op";
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Generated interface implementation.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.cpp.inc"
