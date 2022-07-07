//===- TransformInterfaces.td - Transform Op interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces for transformation-related-ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_INTERFACES_TD
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_INTERFACES_TD

include "mlir/IR/OpBase.td"

def TransformOpInterface : OpInterface<"TransformOpInterface"> {
  let description = [{
    This interface is to be implemented by operations that identify
    transformations to be performed on other operations. The former are referred
    to as transform IR operations. The latter are referred to as payload IR
    operations. Such transform IR operations provide a fine-grain control
    mechanism over how transformations are applied by using and defining
    transform IR values, referred to as handles, that correspond to sets of
    operations in the payload IR. Transformations are applied starting from the
    operations identified by handles, but may affect other operations as well.
    Further restrictions may be imposed by flows that rely on transform IR
    operations to control transformations.
  }];

  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Applies the transformation represented by the current operation. This
        accepts as arguments the object that must be populated with results of
        the current transformation and a transformation state object that can be
        used for queries, e.g., to obtain the list of operations on which the
        transformation represented by the current op is targeted. Returns a
        special status object indicating whether the transformation succeeded
        or failed, and, if it failed, whether the failure is recoverable or not.
      }],
      /*returnType=*/"::mlir::DiagnosedSilenceableFailure",
      /*name=*/"apply",
      /*arguments=*/(ins
          "::mlir::transform::TransformResults &":$transformResults,
          "::mlir::transform::TransformState &":$state
    )>,
  ];

  let extraSharedClassDeclaration = [{
    /// Emits a generic transform error for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    ::mlir::LogicalResult reportUnknownTransformError(
        ::mlir::Operation *target) {
      ::mlir::InFlightDiagnostic diag = $_op->emitError() << "failed to apply";
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation.
    DiagnosedSilenceableFailure emitSilenceableError() {
      Diagnostic diag($_op->getLoc(), DiagnosticSeverity::Error);
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      Diagnostic diag($_op->getLoc(), DiagnosticSeverity::Error);
      diag << $_op->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  }];
}

def FunctionalStyleTransformOpTrait
    : NativeOpTrait<"FunctionalStyleTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def TransformEachOpTrait : NativeOpTrait<"TransformEachOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def NavigationTransformOpTrait : NativeOpTrait<"NavigationTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_INTERFACES_TD
