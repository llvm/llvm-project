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

        IR must be created, modified and deleted with the provided rewriter.
        implementations are responsible for setting the insertion point of the
        rewriter to the desired location.
      }],
      /*returnType=*/"::mlir::DiagnosedSilenceableFailure",
      /*name=*/"apply",
      /*arguments=*/(ins
          "::mlir::transform::TransformRewriter &":$rewriter,
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

    /// Returns all operands that are handles and being consumed by this op.
    ::llvm::SmallVector<OpOperand *> getConsumedHandleOpOperands() {
      return ::mlir::transform::detail::getConsumedHandleOpOperands($_op);
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
    Types that can be used for the Transform dialect operation handle values.
    Such types define the properties of Payload IR operations associated with
    the handle. A user of such a handle can assume that these properties have
    been verified for any Payload IR operation associated with it.
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

def TransformValueHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformValueHandleTypeInterface",
                                 "::mlir::Value"> {
  let description = [{
    Types that can be used for the Transform dialect handle values pointing to
    Payload IR values. Such types define the properties of Payload IR values
    associated with the handle. Users of such a handle can assume that these
    properties have been verified for any Payload IR value associated with it.
  }];
}

def Transform_AnyHandleOrParamType
  : Type<Or<[TransformParamTypeInterface.predicate,
             TransformHandleTypeInterface.predicate,
             TransformValueHandleTypeInterface.predicate]>,
         "any transform handle or parameter">;

def Transform_AnyHandleType
  : Type<Or<[TransformHandleTypeInterface.predicate,
             TransformValueHandleTypeInterface.predicate]>,
         "any transform handle">;

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

def ReportTrackingListenerFailuresOpTrait : NativeOpTrait<"ReportTrackingListenerFailuresOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def FindPayloadReplacementOpInterface
    : OpInterface<"FindPayloadReplacementOpInterface"> {
  let description = [{
    This interface is queried by the `TrackingListener` and can be implemented
    by payload ops to indicate that the lookup should be continue with its
    operands when looking for payload op replacements.

    Example: Consider the case where a tracked "test.foo" payload op is replaced
    with a new "test.foo" op, but wrapped in a "tensor.reshape" op. In that
    case, the mapping of the original "test.foo" op should be updated with the
    new "test.foo" op. A "tensor.reshape" is a metadata-only op that should be
    skipped when inspecting the replacement values of the original "test.foo"
    op. More details can be found at `TrackingListener` documentation.

    Note: Ops that implement `CastOpInterface` do not need to implement this
    interface. Such ops are skipped by default. This interface should be
    implemented by cast-like/metadata-only ops that cannot implement
    `CastOpInterface`.
  }];

  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the operands at which the lookup for replacement payload ops
        should continue.
      }],
      /*returnType=*/"::llvm::SmallVector<::mlir::Value>",
      /*name=*/"getNextOperands",
      /*arguments=*/(ins)
    >,
  ];

  let extraSharedClassDeclaration = [{
    /// Name of the attribute attachable to an operation, indicating that
    /// TrackingListener failures should be silenced.
    constexpr const static ::llvm::StringLiteral
        kSilenceTrackingFailuresAttrName =
            "transform.silence_tracking_failures";
  }];
}

def PatternDescriptorOpInterface : OpInterface<"PatternDescriptorOpInterface"> {
  let description = [{
    This interface should be implemented by ops that select rewrite patterns of
    a `transform.apply_patterns` op. It provides a method to populate a rewrite
    pattern set with patterns.

    Note: Conversion patterns are rewrite patterns in MLIR, but they should not
    be populated with `PatternDescriptorOpInterface` because they cannot be
    used in a greedy pattern rewrite.
  }];

  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate rewrite patterns into the given pattern set.
      }],
      /*returnType=*/"void",
      /*name=*/"populatePatterns",
      /*arguments=*/(ins "::mlir::RewritePatternSet &":$patterns)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Populate rewrite patterns into the given pattern set taking into account
        the transform state.
      }],
      /*returnType=*/"void",
      /*name=*/"populatePatternsWithState",
      /*arguments=*/(ins "::mlir::RewritePatternSet &":$patterns,
                         "::mlir::transform::TransformState &":$state),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.populatePatterns(patterns); }]
    >
  ];
}

def TypeConverterBuilderOpInterface
    : OpInterface<"TypeConverterBuilderOpInterface"> {
  let description = [{
    This interface should be implemented by ops that specify a type converter
    for a dialect conversion, or to populate a type converter with
    conversions.

    When such ops are intended to be used with "apply_conversion_patterns" or
    other operations that expect a type converter, a non-default implementation
    of `getTypeConverter` should be implemented. For use with "cast_and_call"
    like ops that construct a type converter iteratively, non-default
    `populateTypeMaterializations` should be implemented.
  }];

  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the type converter to be used with a dialect conversion.
      }],
      /*returnType=*/"std::unique_ptr<::mlir::TypeConverter>",
      /*name=*/"getTypeConverter",
      /*arguments=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::make_unique<::mlir::TypeConverter>();
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/[{
        Return the type of type converter that this `getTypeConverter` returns.
        This function is used for op verification.
      }],
      /*returnType=*/"StringRef",
      /*name=*/"getTypeConverterType",
      /*arguments=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return "TypeConverter"; }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Populate the given type converter with source/target materialization
        functions.
      }],
      /*returnType=*/"void",
      /*name=*/"populateTypeMaterializations",
      /*arguments=*/(ins "::mlir::TypeConverter &":$converter),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return; }]
    >,
  ];
}

def ConversionPatternDescriptorOpInterface
    : OpInterface<"ConversionPatternDescriptorOpInterface"> {
  let description = [{
    This interface should be implemented by ops that select conversion patterns
    of a `transform.apply_patterns` op. It provides a method to populate a
    rewrite pattern set with conversion patterns.

    Note: Non-conversion rewrite patterns should not be populated with
    `ConversionPatternDescriptorOpInterface` because it is not generally safe
    to use non-conversion rewrite patterns as part of a dialect conversion.
  }];

  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate conversion patterns into the given pattern set with the
        given type converter.
      }],
      /*returnType=*/"void",
      /*name=*/"populatePatterns",
      /*arguments=*/(ins "::mlir::TypeConverter &":$typeConverter,
                         "::mlir::RewritePatternSet &":$patterns)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Populate the ConversionTarget using the final TypeConverter. The default
        implementation is to do nothing. Overriding this method can be useful
        in order to setup the ConversionTarget for structural type conversions.
        In such a situation, an op's legality depends on using the TypeConverter
        to determine whether the op's operand and result types are legal
        (defined as converting to themselves).

      }],
      /*returnType=*/"void",
      /*name=*/"populateConversionTargetRules",
      /*arguments=*/(ins "const ::mlir::TypeConverter &":$typeConverter,
                         "::mlir::ConversionTarget &":$conversionTarget),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the type converter to be used with this pattern set. If no
        type converter is specified, the default type converter of the enclosing
        "apply_conversion_patterns" op is used.
      }],
      /*returnType=*/"std::unique_ptr<::mlir::TypeConverter>",
      /*name=*/"getTypeConverter",
      /*arguments=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Verify the default type converter that is provided by the enclosing
        "apply_conversion_patterns" op.
      }],
      /*returnType=*/"::llvm::LogicalResult",
      /*name=*/"verifyTypeConverter",
      /*arguments=*/(ins "TypeConverterBuilderOpInterface":$builder),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return success();"
    >,
  ];
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORM_INTERFACES_TD
