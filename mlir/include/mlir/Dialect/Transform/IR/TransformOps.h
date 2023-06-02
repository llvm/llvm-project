//===- TransformDialect.h - Transform dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
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
class ApplyPatternsOp;

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
  /// Create a new TrackingListener for usage in the specified transform op.
  explicit TrackingListener(TransformState &state, TransformOpInterface op)
      : TransformState::Extension(state), transformOp(op) {}

protected:
  /// Return a replacement payload op for the given op, which is going to be
  /// replaced with the given values. By default, if all values are defined by
  /// the same op, which also has the same type as the given op, that defining
  /// op is used as a replacement.
  ///
  /// Example: A tracked "linalg.generic" with two results is replaced with two
  /// values defined by (another) "linalg.generic". It is reasonable to assume
  /// that the replacement "linalg.generic" represents the same "computation".
  /// Therefore, the payload op mapping is updated to the defining op of the
  /// replacement values.
  ///
  /// Counter Example: A "linalg.generic" is replaced with values defined by an
  /// "scf.for". Without further investigation, the relationship between the
  /// "linalg.generic" and the "scf.for" is unclear. They may not represent the
  /// same computation; e.g., there may be tiled "linalg.generic" inside the
  /// loop body that represents the original computation. Therefore, the
  /// TrackingListener is conservative by default: it drops the mapping and
  /// triggers the "payload replacement not found" notification.
  ///
  /// If no replacement op could be found according to the rules mentioned
  /// above, this function tries to skip over cast-like ops that implement
  /// `CastOpInterface`.
  ///
  /// Example: A tracked "linalg.generic" is replaced with "linalg.generic",
  /// wrapped in a "tensor.cast". A cast is a metadata-only operation and it is
  /// reasonable to assume that the wrapped "linalg.generic" represents the same
  /// computation as the original "linalg.generic". The mapping is updated
  /// accordingly.
  ///
  /// Certain ops (typically also metadata-only ops) are not considered casts,
  /// but should be skipped nonetheless. Such ops should implement
  /// `FindPayloadReplacementOpInterface` to specify with which operands the
  /// lookup should continue.
  ///
  /// Example: A tracked "linalg.generic" is replaced with "linalg.generic",
  /// wrapped in a "tensor.reshape". A reshape is a metadata-only operation but
  /// not cast. (Implementing `CastOpInterface` would be incorrect and cause
  /// invalid foldings.) However, due to its `FindPayloadReplacementOpInterface`
  /// implementation, the replacement op lookup continues with the wrapped
  /// "linalg.generic" and the mapping is updated accordingly.
  ///
  /// Derived classes may override `findReplacementOp` to specify custom
  /// replacement rules.
  virtual Operation *findReplacementOp(Operation *op,
                                       ValueRange newValues) const;

  /// Notify the listener that the pattern failed to match the given operation,
  /// and provide a callback to populate a diagnostic with the reason why the
  /// failure occurred.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

  /// This function is called when a tracked payload op is dropped because no
  /// replacement op was found. Derived classes can implement this function for
  /// custom error handling.
  virtual void notifyPayloadReplacementNotFound(Operation *op,
                                                ValueRange values) {}

  /// Return the single op that defines all given values (if any).
  static Operation *getCommonDefiningOp(ValueRange values);

  /// Return the transform op in which this TrackingListener is used.
  TransformOpInterface getTransformOp() const { return transformOp; }

private:
  void notifyOperationRemoved(Operation *op) override;

  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;

  /// The transform op in which this TrackingListener is used.
  TransformOpInterface transformOp;
};

/// A specialized listener that keeps track of cases in which no replacement
/// payload could be found. The error state of this listener must be checked
/// before the end of its lifetime.
class ErrorCheckingTrackingListener : public TrackingListener {
public:
  using transform::TrackingListener::TrackingListener;

  ~ErrorCheckingTrackingListener() override;

  /// Check and return the current error state of this listener. Afterwards,
  /// resets the error state to "success".
  DiagnosedSilenceableFailure checkAndResetError();

  /// Return "true" if this tracking listener had a failure.
  bool failed() const;

protected:
  void notifyPayloadReplacementNotFound(Operation *op,
                                        ValueRange values) override;

private:
  /// The error state of this listener. "Success" indicates that no error
  /// happened so far.
  DiagnosedSilenceableFailure status = DiagnosedSilenceableFailure::success();

  /// The number of errors that have been encountered.
  int64_t errorCounter = 0;
};

/// The PatternRegistry stores callbacks to functions that populate a
/// `RewritePatternSet`. Registered patterns can be applied with the
/// "transform.apply_patterns" op.
class PatternRegistry : public TransformDialectData<PatternRegistry> {
public:
  PatternRegistry(MLIRContext *ctx) : TransformDialectData(ctx), builder(ctx) {}

  /// A function that populates a `RewritePatternSet`.
  using PopulatePatternsFn = std::function<void(RewritePatternSet &)>;
  /// A function that populates a `RewritePatternSet` with a specified benefit.
  using PopulatePatternsWithBenefitFn =
      std::function<void(RewritePatternSet &, PatternBenefit)>;

  /// Registers patterns with the specified identifier. The identifier should
  /// be prefixed with the dialect to which the patterns belong.
  void registerPatterns(StringRef identifier, PopulatePatternsFn &&fn);

  /// Registers patterns with the specified identifier. The identifier should
  /// be prefixed with the dialect to which the patterns belong. The pattern
  /// benefit is currently ignored.
  void registerPatterns(StringRef identifier,
                        PopulatePatternsWithBenefitFn &&fn);

protected:
  friend class ApplyPatternsOp;

  /// Returns "true" if patterns are registered with the specified identifier.
  bool hasPatterns(StringAttr identifier) const;

  /// Populates the given pattern set with the specified patterns.
  void populatePatterns(StringAttr identifier,
                        RewritePatternSet &patternSet) const;

private:
  /// A builder for creating StringAttrs.
  Builder builder;

  DenseMap<StringAttr, PopulatePatternsFn> patterns;
};

} // namespace transform
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::transform::PatternRegistry)

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
