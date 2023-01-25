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
    InterfaceMethod<
      /*desc=*/[{
        Indicates whether the op instance allows its handle operands to be
        associated with the same payload operations.
      }],
      /*returnType=*/"bool",
      /*name=*/"allowsRepeatedHandleOperands",
      /*arguments=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];

  let extraSharedClassDeclaration = [{
    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation. Silenceable failure must be suppressed or reported
    /// explicitly at some later time.
    DiagnosedSilenceableFailure
    emitSilenceableError(const ::llvm::Twine &message = {}) {
      return ::mlir::emitSilenceableFailure($_op, message);
    }

    /// Creates the definite failure object with a diagnostic located at the
    /// current operation. Definite failure will be reported when the object
    /// is destroyed or converted.
    DiagnosedDefiniteFailure
    emitDefiniteFailure(const ::llvm::Twine &message = {}) {
      return ::mlir::emitDefiniteFailure($_op, message);
    }

    /// Emits a generic definite failure for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
        ::mlir::Operation *target) {
      auto diag = ::mlir::emitDefiniteFailure($_op, "failed to apply");
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure($_op->getLoc());
      diag << $_op->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return diag;
    }
  }];

  let verify = [{
    return ::mlir::transform::detail::verifyTransformOpInterface($_op);
  }];
}

class TransformTypeInterfaceBase<string cppClass, string cppObjectType>
    : TypeInterface<cppClass> {
  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Checks if the given associated objects (Payload IR operations or attributes)
        satisfy the conditions defined by this type. If not, produces a silenceable
        error at the specified location.
      }],
      /*returnType=*/"::mlir::DiagnosedSilenceableFailure",
      /*name=*/"checkPayload",
      /*arguments=*/(ins "::mlir::Location":$loc,
                         "::mlir::ArrayRef<" # cppObjectType # ">":$payload)
    >
  ];

  let extraSharedClassDeclaration = [{
    DiagnosedSilenceableFailure emitSilenceableError(Location loc) const {
      Diagnostic diag(loc, DiagnosticSeverity::Error);
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  }];
}

def TransformHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformHandleTypeInterface",
                                 "::mlir::Operation *"> {
  let description = [{
    Types that can be used for the Transform dialect handle values. Such types
    define the properties of Payload IR operations associated with the handle.
    A user of such a handle can assume that these properties have been verified
    for any Payload IR operation associated with it.
  }];
}

def TransformParamTypeInterface
    : TransformTypeInterfaceBase<"TransformParamTypeInterface",
                                 "::mlir::Attribute"> {
  let description = [{
    Types that can be used for the Transform dialect parameter values. Such types
    define the structure of the parameters associated with the value, e.g., their
    underlying type. A user of the value can assume that the parameter has been
    verified.
  }];
}

def Transform_AnyHandleOrParamType
  : Type<Or<[TransformParamTypeInterface.predicate,
             TransformHandleTypeInterface.predicate]>,
         "any transform handle or parameter">;

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

def ParamProducerTransformOpTrait : NativeOpTrait<"ParamProducerTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_INTERFACES_TD
