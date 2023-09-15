//===- MatchInterfaces.h - Transform Dialect Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_MATCHINTERFACES_H
#define MLIR_DIALECT_TRANSFORM_IR_MATCHINTERFACES_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace transform {
class MatchOpInterface;

template <typename OpTy>
class SingleOpMatcherOpTrait
    : public OpTrait::TraitBase<OpTy, SingleOpMatcherOpTrait> {
  template <typename T>
  using has_get_operand_handle =
      decltype(std::declval<T &>().getOperandHandle());
  template <typename T>
  using has_match_operation = decltype(std::declval<T &>().matchOperation(
      std::declval<Operation *>(), std::declval<TransformResults &>(),
      std::declval<TransformState &>()));

public:
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(llvm::is_detected<has_get_operand_handle, OpTy>::value,
                  "SingleOpMatcherOpTrait expects operation type to have the "
                  "getOperandHandle() method");
    static_assert(llvm::is_detected<has_match_operation, OpTy>::value,
                  "SingleOpMatcherOpTrait expected operation type to have the "
                  "matchOperation(Operation *, TransformResults &, "
                  "TransformState &) method");

    // This must be a dynamic assert because interface registration is dynamic.
    assert(isa<MatchOpInterface>(op) &&
           "SingleOpMatchOpTrait is only available on operations with "
           "MatchOpInterface");
    Value operandHandle = cast<OpTy>(op).getOperandHandle();
    if (!isa<TransformHandleTypeInterface>(operandHandle.getType())) {
      return op->emitError() << "SingleOpMatchOpTrait requires the op handle "
                                "to be of TransformHandleTypeInterface";
    }

    return success();
  }

  DiagnosedSilenceableFailure apply(TransformRewriter &rewriter,
                                    TransformResults &results,
                                    TransformState &state) {
    Value operandHandle = cast<OpTy>(this->getOperation()).getOperandHandle();
    auto payload = state.getPayloadOps(operandHandle);
    if (!llvm::hasSingleElement(payload)) {
      return emitDefiniteFailure(this->getOperation()->getLoc())
             << "SingleOpMatchOpTrait requires the operand handle to point to "
                "a single payload op";
    }

    return cast<OpTy>(this->getOperation())
        .matchOperation(*payload.begin(), results, state);
  }

  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    onlyReadsHandle(this->getOperation()->getOperands(), effects);
    producesHandle(this->getOperation()->getResults(), effects);
    onlyReadsPayload(effects);
  }
};

template <typename OpTy>
class SingleValueMatcherOpTrait
    : public OpTrait::TraitBase<OpTy, SingleValueMatcherOpTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // This must be a dynamic assert because interface registration is dynamic.
    assert(isa<MatchOpInterface>(op) &&
           "SingleValueMatchOpTrait is only available on operations with "
           "MatchOpInterface");

    Value operandHandle = cast<OpTy>(op).getOperandHandle();
    if (!isa<TransformValueHandleTypeInterface>(operandHandle.getType())) {
      return op->emitError() << "SingleValueMatchOpTrait requires an operand "
                                "of TransformValueHandleTypeInterface";
    }

    return success();
  }

  DiagnosedSilenceableFailure apply(TransformRewriter &rewriter,
                                    TransformResults &results,
                                    TransformState &state) {
    Value operandHandle = cast<OpTy>(this->getOperation()).getOperandHandle();
    auto payload = state.getPayloadValues(operandHandle);
    if (!llvm::hasSingleElement(payload)) {
      return emitDefiniteFailure(this->getOperation()->getLoc())
             << "SingleValueMatchOpTrait requires the value handle to point to "
                "a single payload value";
    }

    return cast<OpTy>(this->getOperation())
        .matchValue(*payload.begin(), results, state);
  }

  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    onlyReadsHandle(this->getOperation()->getOperands(), effects);
    producesHandle(this->getOperation()->getResults(), effects);
    onlyReadsPayload(effects);
  }
};

} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/MatchInterfaces.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IR_MATCHINTERFACES_H
