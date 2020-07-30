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
      and the regions of the op.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"inferReturnTypes",
      /*args=*/(ins "MLIRContext*":$context,
                    "Optional<Location>":$location,
                    "ValueRange":$operands,
                    "DictionaryAttr":$attributes,
                    "RegionRange":$regions,
                    "SmallVectorImpl<Type>&":$inferredReturnTypes)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Returns whether two array of types are compatible result types"
               " for an op.",
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleReturnTypes",
      /*args=*/(ins "ArrayRef<Type>":$lhs, "ArrayRef<Type>":$rhs),
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
      /*retTy=*/"LogicalResult",
      /*methodName=*/"inferReturnTypeComponents",
      /*args=*/(ins "MLIRContext*":$context,
                    "Optional<Location>":$location,
                    "ValueRange":$operands,
                    "DictionaryAttr":$attributes,
                    "RegionRange":$regions,
                    "SmallVectorImpl<ShapedTypeComponents>&":
                      $inferredReturnShapes)
    >,
    InterfaceMethod<
      /*desc=*/[{Reify the shape computation for the operation.

      Insert operations using the given OpBuilder that computes the result
      shape.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"reifyReturnTypeShapes",
      /*args=*/(ins "OpBuilder&":$builder,
                    "SmallVectorImpl<Value>&":$reifiedReturnShapes),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return failure(); }]
    >,
  ];
}

#endif // MLIR_INFERTYPEOPINTERFACE
