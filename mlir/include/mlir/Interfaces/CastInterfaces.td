//===- CastInterfaces.td - Cast Interfaces for ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// related to cast-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CASTINTERFACES
#define MLIR_INTERFACES_CASTINTERFACES

include "mlir/IR/OpBase.td"

def CastOpInterface : OpInterface<"CastOpInterface"> {
  let description = [{
    A cast-like operation is one that converts from a set of input types to a
    set of output types. The arity of the inputs may be from 0-N, whereas the
    arity of the outputs may be anything from 1-N. Cast-like operations are
    trivially removable in cases where they produce an No-op, i.e when the
    input types and output types match 1-1.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<[{
        Returns true if the given set of input and result types are compatible
        to cast using this cast operation.
      }],
      "bool", "areCastCompatible",
      (ins "::mlir::TypeRange":$inputs, "::mlir::TypeRange":$outputs)
    >,
  ];

  let extraTraitClassDeclaration = [{
    /// Attempt to fold the given cast operation.
    static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
      return impl::foldCastInterfaceOp(op, operands, results);
    }
  }];
  let verify = [{
    return impl::verifyCastInterfaceOp($_op);
  }];
}

#endif // MLIR_INTERFACES_CASTINTERFACES
