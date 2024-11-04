//===-- MyExtension.td - Transform dialect tutorial --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 3 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef MY_EXTENSION
#define MY_EXTENSION

include "MyExtensionTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the new operation. By convention, prefix its name with the name of the dialect 
// extension, "my.". The full operation name will be further prefixed with "transform.".
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    // Indicate that the operation implements the required TransformOpInterface and
    // MemoryEffectsOpInterface. Use the TransformEach trait to provide the 
    // implementation for TransformOpInterface.
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  // Provide a brief and a full description. It is recommended that the latter describes 
  // the effects on the operands and how the operation processes various failure modes.
  let summary = "Changes the callee of a call operation to the specified one";
  let description = [{
    For each `func.call` payload operation associated with the handle, changes its 
    callee to be the symbol whose name is provided as an attribute to this operation.

    Generates a silenceable failure if the operand is associated with payload operations 
    that are not `func.call`.
    Only reads the operand.
  }];

  // The arguments include the handle to the payload operations and the attribute that 
  // specifies the new callee. The handle must implement TransformHandleTypeInterface.   
  // We use a string attribute as the symbol may not exist in the transform IR so the 
  // verification may fail. 
  let arguments = (ins
    // Specify the type constraint on the input accepting only `func.call` payload
    // operations.
    Transform_ConcreteOpType<"func.call">:$call,
    StrAttr:$new_target);

  // The results are empty as the transformation does not produce any new payload.
  let results = (outs);

  // Provide nice syntax.
  let assemblyFormat = "$call `,` $new_target attr-dict `:` qualified(type($call))";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::CallOp call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// Define another transform operation.
def CallToOp : Op<Transform_Dialect, "my.call_to_op",
     // Indicate that the operation implements the required TransformOpInterface.
     // Use the TransformEach trait to provide implementation of this interface.
    [TransformOpInterface, TransformEachOpTrait,
     // Indicate that the operation implements the required MemoryEffectsOpInterface.
     // Use the FunctionalStyle trait to provide the implementation for this interface.
     MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
  // Summary and description omitted for brevity.

  // The argument is the handle to the payload operations.
  let arguments = (ins CallOpInterfaceHandle:$call);

  // The result is the handle to the payload operations produced during the 
  // transformation.
  let results = (outs TransformHandleTypeInterface:$transformed);

  // Provide nice syntax.
  let assemblyFormat = "$call attr-dict `:` functional-type(operands, results)";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::CallOpInterface call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // MY_EXTENSION
