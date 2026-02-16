//===-- Passes.td - TOSA pass declarations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_PASSES
#define MLIR_DIALECT_TOSA_TRANSFORMS_PASSES

include "mlir/IR/EnumAttr.td"
include "mlir/Pass/PassBase.td"

def TosaLayerwiseConstantFoldPass : Pass<"tosa-layerwise-constant-fold", "func::FuncOp"> {
  let summary = "Fold layerwise operations on constant tensors";
  let description = [{
    Pass that enables folding of full-layer operations on constant tensors.
  }];

  let options = [
      Option<"aggressiveReduceConstant", "aggressive-reduce-constant", "bool",
             /*default=*/"false",
             "Always perform the reduce constant optimization"
             "May add more tosa.const but would reduce runtime calculations">,
   ];
}

def TosaInferShapesPass : Pass<"tosa-infer-shapes", "func::FuncOp"> {
  let summary = "Propagate shapes across TOSA operations";
  let description = [{
    Pass that uses operand types and propagates shapes to TOSA operations.
    This includes legalizing rankless and dynamic shapes towards static.
  }];

  let dependentDialects = [
    "func::FuncDialect",
    "tensor::TensorDialect",
    "tosa::TosaDialect",
  ];
}

def TosaMakeBroadcastablePass
    : Pass<"tosa-make-broadcastable", "func::FuncOp"> {
  let summary = "TOSA rank Reshape to enable Broadcasting";
  let description = [{
    Pass that enables broadcast by making all input arrays have the same
    number of dimensions. Insert RESHAPE operations to prepend dimensions
    of size one until the number of dimensions is equal. Implements
    approach similar to step 1 of Numpy 4-step broadcasting:
    https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting
  }];
}

def TosaOptionalDecompositionsPass
    : Pass<"tosa-optional-decompositions", "func::FuncOp"> {
  let summary = "Applies Tosa operations optional decompositions";
  let description = [{
    Pass to apply the Tosa operations decompositions
    exposed as populate functions in include/mlir/Dialect/Tosa/Transforms/Passes.h
  }];
}

def TosaValidation : Pass<"tosa-validate", "mlir::ModuleOp"> {
  let summary = "Validates TOSA dialect";
  let description = [{
    This pass validates if input TOSA operations match the specification for given
    criteria, e.g. TOSA profile.
  }];

  let options = [
      Option<"strictOpSpecAlignment", "strict-op-spec-alignment", "bool",
             /*default=*/"false",
             "Verify if the properties of certain operations align the spec requirement">,
      Option<"allowInvalidOpDatatypeCombinations", "allow-invalid-op-datatype-combinations", "bool",
             /*default=*/"false",
             "Disable checks for operations that are determined to be invalid due to their "
             "operand/result datatypes not aligning with the 'Supported Data Types' "
             "sections of the specifciation">
   ];
}

def TosaReduceTransposes : Pass<"tosa-reduce-transposes", "func::FuncOp"> {
  let summary = "Reduce transposes through other operators";
  let description = [{
    Pass that identifies and reduces tosa.TRANSPOSE operations through chains
    of operators.

    The pass traverses dependencies of tosa.TRANSPOSE operations until they
    terminate in either a tosa.RESHAPE that we can fold the hoisted
    tosa.TRANSPOSE into, a tosa.TRANSPOSE that forms the identity with the
    hoisted one, or a tosa.CONST with a dense elements attribute. It then
    propagates the hoisted transform upward through the intervening operators
    if the support is implemented. Finally, it observes that no duplication
    will occur of both the chain that was hoisted through and the new chain
    that results, and if so, it replaces the hoisted tosa.TRANSPOSE.

    The pass has an important use-case in cleaning up the results of frameworks
    that introduce a lot of data-layout transformations when legalizing to TOSA,
    a common one being transformations between NHWC and NCHW layouts.
  }];
}

def TosaArithConstantToTosaConstPass
    : Pass<"tosa-arith-const-to-tosa-const", "func::FuncOp"> {
  let summary = "Convert tensor arith.constant operations into tosa.const";
  let description = [{
    Normalizes tensor-valued arith.constant operations into tosa.const so that
    subsequent TOSA passes operate on a consistent representation of constants.
  }];
}

def TosaConvertIntegerTypeToSignless : Pass<"tosa-convert-integer-type-to-signless", "func::FuncOp"> {
  let summary = "Convert integer types to signless";
  let description = [{
    This pass converts signed or unsigned integer types to signless. It
    currently does this greedily for all operators and can also change the
    signature of the function. Should the signature of the entrypoint
    function change, it will be the responsibility of the user to carry
    signedness information of the inputs and outputs independently.

    This can be a useful transformation for conversion to other formats
    that require strict adherence to the TOSA specification.
  }];
}

def TosaAttachTarget : Pass<"tosa-attach-target", "ModuleOp"> {
  let summary = "Attach tosa.target_env information to the given module.";

  let description = [{
    This pass allows the user to specify a TOSA target environment consisting of
    the following components: level, profiles and extensions.

    The target environment is attached to the module as an attribute, allowing other
    transformations to query the selected target and adapt their behaviour based on
    this information.
  }];

  let dependentDialects = [
    "func::FuncDialect",
    "tosa::TosaDialect",
  ];

  let options = [
    Option<"specificationVersion", "specification_version", "mlir::tosa::SpecificationVersion",
              /*default=*/"mlir::tosa::SpecificationVersion::V_1_0",
              "The specification version that TOSA operators should conform to.",
              [{::llvm::cl::values(
                clEnumValN(mlir::tosa::SpecificationVersion::V_1_0, "1.0", "TOSA Specification version 1.0"),
                clEnumValN(mlir::tosa::SpecificationVersion::V_1_1_DRAFT, "1.1.draft", "TOSA Specification version 1.1.draft")
              )}]>,
    Option<"level", "level", "mlir::tosa::Level",
              /*default=*/"mlir::tosa::Level::eightK",
              "The TOSA level that operators should conform to. A TOSA level defines "
              "operator argument ranges that an implementation shall support.",
              [{::llvm::cl::values(
                clEnumValN(mlir::tosa::Level::eightK, "8k",
                  "Ranges are expected to be sufficient for applications with frame "
                  "sizes up to 8K."),
                clEnumValN(mlir::tosa::Level::none, "none",
                  "Allows the full range of arguments specified by the operations according "
                  "to the operation data types.")
                )}]>,
    ListOption<"profiles", "profiles", "std::string",
              "The TOSA profile(s) that operators should conform to. TOSA profiles "
              "enable efficient implementation on different classes of device. Each "
              "profile is an independent set of operations and data type combinations.">,
    ListOption<"extensions", "extensions", "std::string",
            "The TOSA extension(s) that operators should conform to. TOSA profile "
            "extensions define optional operation and data type combinations.">
  ];
}

def TosaNarrowI64ToI32Pass : Pass<"tosa-narrow-i64-to-i32", "func::FuncOp"> {
  let summary = "Narrow I64 TOSA operations to I32";
  let description = [{
    This pass narrows TOSA operations with 64-bit integer tensor types to
    32-bit integer tensor types. This can be useful for backends that do not
    support the EXT-INT64 extension of TOSA.
  }];

  let options = [
    Option<"aggressiveRewrite", "aggressive-rewrite", "bool", "false",
      "If enabled, all TOSA operations are rewritten, regardless or whether the narrowing"
      "is safe. This option may lead to data loss if not used carefully.">,
    Option<"convertFunctionBoundaries", "convert-function-boundaries", "bool", "false",
      "If enabled, the pass will convert function I/O types as well. Otherwise casts will"
      "be inserted at the I/O boundaries.">
  ];

  let dependentDialects = [
    "func::FuncDialect",
    "tosa::TosaDialect",
  ];
}

def TosaNarrowF64ToF32Pass : Pass<"tosa-narrow-f64-to-f32", "func::FuncOp"> {
  let summary = "Narrow F64 TOSA operations to F32";
  let description = [{
    This pass narrows TOSA operations with 64-bit floating-point tensor types to
    32-bit floating-point tensor types. While TOSA itself has no double
    precision support, upstream conversions or frontends may still materialize
    F64 tensors temporarily, so this pass removes them before handing off to a
    TOSA backend.
  }];

  let options = [
    Option<"aggressiveRewrite", "aggressive-rewrite", "bool", "false",
      "If enabled, all TOSA operations are rewritten, regardless or whether the narrowing"
      "is safe. This option may lead to data loss if not used carefully.">,
    Option<"convertFunctionBoundaries", "convert-function-boundaries", "bool", "false",
      "If enabled, the pass will convert function I/O types as well. Otherwise casts will"
      "be inserted at the I/O boundaries.">
  ];

  let dependentDialects = [
    "func::FuncDialect",
    "tosa::TosaDialect",
  ];
}

def TosaInputShape : Pass<"tosa-experimental-input-shape", "func::FuncOp"> {
  let summary = "Override dynamic function arguments to specified static shapes.";
  let description = [{
    Pass that overrides the dynamic input shapes of function arguments to
    specified static shapes. If a specified static shape conflicts with the
    static dimensions in an original input shape, an error is reported.
  }];

  let constructor = "tosa::createTosaInputShapePass()";
  let dependentDialects = [
    "tensor::TensorDialect"
  ];
  let options = [
    ListOption<"args", "args", "std::string",
          "Comma-separated list of shape descriptions. Each description contains the "
          "argument name, a colon, and a shape with dimensions separated by x ">
  ];
}

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_PASSES
