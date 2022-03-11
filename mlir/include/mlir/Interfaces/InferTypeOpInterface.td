//===- InferTypeOpInterface.td - Infer Type interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// related to type inference.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INFERTYPEOPINTERFACE
#define MLIR_INFERTYPEOPINTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to compute the return type of an operation. The arguments match
// those in Operation::create with the exception that the location is optional
// (if no location is provided, then the method will not emit an error on
// mismatch).
def InferTypeOpInterface : OpInterface<"InferTypeOpInterface"> {
  let description = [{
    Interface to infer the return types for an operation that could be used
    during op construction, verification or type inference.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{Infer the return types that an op would generate.

      The method takes an optional location which, if set, will be used to
      report errors on. The operands and attributes correspond to those with
      which an Operation would be created (e.g., as used in Operation::create)
      and the regions of the op. Be aware that this method is supposed to be
      called with valid arguments, e.g., operands are verified, or it may result
      in an undefined behavior.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"inferReturnTypes",
      /*args=*/(ins "::mlir::MLIRContext *":$context,
                    "::llvm::Optional<::mlir::Location>":$location,
                    "::mlir::ValueRange":$operands,
                    "::mlir::DictionaryAttr":$attributes,
                    "::mlir::RegionRange":$regions,
                    "::llvm::SmallVectorImpl<::mlir::Type>&":$inferredReturnTypes)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Returns whether two array of types are compatible result types"
               " for an op.",
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleReturnTypes",
      /*args=*/(ins "::mlir::TypeRange":$lhs, "::mlir::TypeRange":$rhs),
      /*methodBody=*/[{
        return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
      }],
      /*defaultImplementation=*/[{
        /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
      }]
    >,
  ];

  // Inferring result types may need to access the region operations.
  let verifyWithRegions = 1;
  let verify = [{
    return detail::verifyInferredResultTypes($_op);
  }];
}

def InferShapedTypeOpInterface : OpInterface<"InferShapedTypeOpInterface"> {
  let description = [{
    Interface to infer the components of a ShapedType returned by an operation
    that could be used during op construction, verification or shape inference.

    The components consists of element type, shape and raw attribute.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{Infer the components of return type of shape containter.

      The method takes an optional location which, if set, will be used to
      report errors on. The operands and attributes correspond to those with
      which an Operation would be created (e.g., as used in Operation::create)
      and the regions of the op.

      Unknown (e.g., unranked) shape and nullptrs for element type and attribute
      may be returned by this function while returning success. E.g., partial
      population of components is not error condition.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"inferReturnTypeComponents",
      /*args=*/(ins "::mlir::MLIRContext*":$context,
                    "::mlir::Optional<::mlir::Location>":$location,
                    "::mlir::ValueShapeRange":$operands,
                    "::mlir::DictionaryAttr":$attributes,
                    "::mlir::RegionRange":$regions,
                    "::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>&":
                      $inferredReturnShapes),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return ::mlir::failure(); }]
    >,
    InterfaceMethod<
      /*desc=*/[{Reify the shape computation for the operation.

      Insert operations using the given OpBuilder that computes the
      result shape. This interface is supposed to be workable during dialect
      conversion (e.g. convert from tensor world to buffer world),
      where `getOperand` may be invalid. For example, some ops (e.g.
      dynamic_reshape(input, target_shape)) may depend on their operands
      to calculate the result shape. When the `matchAndRewrite ` method
      of a conversion pattern is called, the operands of the op to convert
      may have been converted into other types, which makes it invalid to
      call the `getOperand` method of such op directly inside the
      conversion pattern.  To solve this problem, this interface follows
      the design of the conversion pattern, that is, accepting passed in
      operands to avoid calling `getOperand` directly inside the interface
      implementation.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"reifyReturnTypeShapes",
      /*args=*/(ins "::mlir::OpBuilder&":$builder,
          "::mlir::ValueRange":$operands,
          "::mlir::SmallVectorImpl<::mlir::Value> &":$reifiedReturnShapes),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return ::mlir::failure(); }]
    >
  ];
}

// Convenience class grouping together type and shaped type op interfaces for
// ops that have tensor return types.
class InferTensorType<list<string> overridenMethods = []> : TraitList<
  [
    // Op implements infer type op interface.
    InferTypeOpInterface,
    // The op will have methods implementing the ShapedType type inference
    // interface.
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface, overridenMethods>,
    // The op produces tensors and will use the ShapedType type infer interface
    // along with knowledge that it is producing Tensors to infer the type.
    NativeOpTrait<"InferTensorType">
  ]>;
def InferTensorTypeWithReify: TraitList<InferTensorType<[
    "inferReturnTypeComponents", "reifyReturnTypeShapes"]>.traits>;


def ReifyRankedShapedTypeOpInterface :
    OpInterface<"ReifyRankedShapedTypeOpInterface"> {
  let description = [{
    Interface to compute the shape of the result of an operation when
    the result is a ranked shape type, i.e. `RankedTensorType` or
    `MemRefType`.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Reify the shape of the result of an operation (typically in
        terms of shape of its operands)

        Insert operations using the given `OpBuilder` that computes
        the result shape. The `reifiedReturnShapes` is expected to be
        populated with as many vectors as the number of results of the
        op. Each of these vectors is expected to be of size equal to
        rank of the corresponding result. If the shape of a particular
        result cannot be computed it must be empty.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"reifyResultShapes",
      /*args=*/(ins "::mlir::OpBuilder &":$builder,
        "::mlir::ReifiedRankedShapedTypeDims &":$reifiedReturnShapes)
    >
  ];
}

// Op has the same operand and result type.
// TODO: Change from hard coded to utilizing type inference trait.
def SameOperandsAndResultType : NativeOpTrait<"SameOperandsAndResultType">;

#endif // MLIR_INFERTYPEOPINTERFACE
