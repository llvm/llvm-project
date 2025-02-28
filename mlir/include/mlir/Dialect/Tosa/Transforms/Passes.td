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

  let constructor = "createTosaLayerwiseConstantFoldPass()";

  let options = [
      Option<"aggressiveReduceConstant", "aggressive-reduce-constant", "bool",
             /*default=*/"false",
             "Always perform the reduce constant optimization"
             "May add more tosa.const but would reduce runtime calculations">,
   ];
}

def TosaInferShapes : Pass<"tosa-infer-shapes", "func::FuncOp"> {
  let summary = "Propagate shapes across TOSA operations";
  let description = [{
    Pass that uses operand types and propagates shapes to TOSA operations.
    This includes legalizing rankless and dynamic shapes towards static.
  }];

  let constructor = "createTosaInferShapesPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "tensor::TensorDialect",
    "tosa::TosaDialect",
  ];
}

def TosaMakeBroadcastable : Pass<"tosa-make-broadcastable", "func::FuncOp"> {
  let summary = "TOSA rank Reshape to enable Broadcasting";
  let description = [{
    Pass that enables broadcast by making all input arrays have the same
    number of dimensions. Insert RESHAPE operations to prepend dimensions
    of size one until the number of dimensions is equal. Implements
    approach similar to step 1 of Numpy 4-step broadcasting:
    https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting
  }];

  let constructor = "createTosaMakeBroadcastablePass()";
}

def TosaOptionalDecompositions
  : Pass<"tosa-optional-decompositions", "func::FuncOp"> {
  let summary = "Applies Tosa operations optional decompositions";
  let description = [{
    Pass to apply the Tosa operations decompositions
    exposed as populate functions in include/mlir/Dialect/Tosa/Transforms/Passes.h
  }];

  let constructor = "tosa::createTosaOptionalDecompositions()";
}

def TosaLevelType : I32EnumAttr<"TosaLevelEnum", "Tosa level",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"EightK", 1, "8k">,
    ]>{
  let cppNamespace = "mlir::tosa";
}

def TosaValidation : Pass<"tosa-validate", "mlir::ModuleOp"> {
  let summary = "Validates TOSA dialect";
  let description = [{
    This pass validates if input TOSA operations match the specification for given
    criteria, e.g. TOSA profile.
  }];

  let options = [
      ListOption<"profile", "profile", "std::string",
             "Validate if operations match for the given profile set">,
      ListOption<"extension", "extension", "std::string",
             "Validate if operations match for the given extension set">,
      Option<"strictOpSpecAlignment", "strict-op-spec-alignment", "bool",
             /*default=*/"false",
             "Verify if the properties of certain operations align the spec requirement">,
      Option<"level", "level", "mlir::tosa::TosaLevelEnum",
             /*default=*/"mlir::tosa::TosaLevelEnum::EightK",
             "Validate if operator parameters are within specfication for the given level",
             [{::llvm::cl::values(
               clEnumValN(mlir::tosa::TosaLevelEnum::EightK, "8k",
                "Ranges are expected to be sufficient for applications with frame sizes up to 8K."),
               clEnumValN(mlir::tosa::TosaLevelEnum::None, "none",
                "Allows the full range of arguments specified by the operations according "
                "to the operation data types.")
              )}]>
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

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_PASSES
