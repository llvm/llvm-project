//===- TransformInterfaces.cpp - Transform Dialect Interfaces -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//

constexpr const Value transform::TransformState::kTopLevelValue;

transform::TransformState::TransformState(Operation *root) {
  operationMapping[kTopLevelValue].push_back(root);
}

Operation *transform::TransformState::getTopLevel() const {
  return operationMapping.lookup(kTopLevelValue).front();
}

ArrayRef<Operation *>
transform::TransformState::getPayloadOps(Value value) const {
  auto iter = operationMapping.find(value);
  assert(iter != operationMapping.end() && "unknown handle");
  return iter->getSecond();
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
  bool inserted =
      operationMapping.insert({value, std::move(storedTargets)}).second;
  assert(inserted && "value is already associated with another list");
  (void)inserted;

  // Having multiple handles to the same operation is an error in the transform
  // expressed using the dialect and may be constructed by valid API calls from
  // valid IR. Emit an error here.
  for (Operation *op : targets) {
    auto insertionResult = reverseMapping.insert({op, value});
    if (!insertionResult.second) {
      InFlightDiagnostic diag = op->emitError()
                                << "operation tracked by two handles";
      diag.attachNote(value.getLoc()) << "handle";
      diag.attachNote(insertionResult.first->second.getLoc()) << "handle";
      return diag;
    }
  }

  return success();
}

void transform::TransformState::removePayloadOps(Value value) {
  for (Operation *op : operationMapping[value])
    reverseMapping.erase(op);
  operationMapping.erase(value);
}

void transform::TransformState::updatePayloadOps(
    Value value, function_ref<Operation *(Operation *)> callback) {
  auto it = operationMapping.find(value);
  assert(it != operationMapping.end() && "unknown handle");
  SmallVector<Operation *> &association = it->getSecond();
  SmallVector<Operation *> updated;
  updated.reserve(association.size());

  for (Operation *op : association)
    if (Operation *updatedOp = callback(op))
      updated.push_back(updatedOp);

  std::swap(association, updated);
}

LogicalResult
transform::TransformState::applyTransform(TransformOpInterface transform) {
  transform::TransformResults results(transform->getNumResults());
  if (failed(transform.apply(results, *this)))
    return failure();

  for (Value target : transform->getOperands())
    removePayloadOps(target);

  for (auto &en : llvm::enumerate(transform->getResults()))
    if (failed(setPayloadOps(en.value(), results.get(en.index()))))
      return failure();

  return success();
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
// Generated interface implementation.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformInterfaces.cpp.inc"
