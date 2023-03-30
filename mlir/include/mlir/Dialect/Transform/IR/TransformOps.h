//===- TransformDialect.h - Transform dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
namespace transform {
enum class FailurePropagationMode : uint32_t;
class FailurePropagationModeAttr;

/// A builder function that populates the body of a SequenceOp.
using SequenceBodyBuilderFn = ::llvm::function_ref<void(
    ::mlir::OpBuilder &, ::mlir::Location, ::mlir::BlockArgument)>;
using SequenceBodyBuilderArgsFn =
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location,
                              ::mlir::BlockArgument, ::mlir::ValueRange)>;

/// A listener that updates a TransformState based on IR modifications. This
/// listener can be used during a greedy pattern rewrite to keep the transform
/// state up-to-date.
class TrackingListener : public RewriterBase::Listener,
                         public TransformState::Extension {
public:
  explicit TrackingListener(TransformState &state)
      : TransformState::Extension(state) {}

protected:
  /// Return a replacement payload op for the given op, which is going to be
  /// replaced with the given values. By default, if all values are defined by
  /// the same newly-created op, which also has the same type as the given op,
  /// that defining op is used as a replacement.
  virtual Operation *findReplacementOp(Operation *op,
                                       ValueRange newValues) const;

  /// This function is called when a tracked payload op is dropped because no
  /// replacement op was found. Derived classes can implement this function for
  /// custom error handling.
  virtual void notifyPayloadReplacementNotFound(Operation *op,
                                                ValueRange values) const {}

  /// Return "true" if the given op is a new op.
  bool isNewOp(Operation *op) const;

  /// Return the single op that defines all given values (if any).
  static Operation *getCommonDefiningOp(ValueRange values);

private:
  void notifyOperationInserted(Operation *op) override;

  void notifyOperationRemoved(Operation *op) override;

  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;

  /// Ops that were newly created during the transform.
  DenseMap<OperationName, DenseSet<Operation *>> newOps;
};

} // namespace transform
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
