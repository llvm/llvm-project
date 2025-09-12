//===- AggregatedOpInterface.td ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AGGREGATEDOPINTERFACE
#define MLIR_AGGREGATEDOPINTERFACE

include "mlir/IR/OpBase.td"

def AggregatedOpInterface : OpInterface<"AggregatedOpInterface"> {
  let description = [{
    This Interface is particularly useful in cases where we have an operation
    that can be lowered into a sequence of simpler operations, thus essentially
    decomposing an operation into a set of one or many simpler operations.
    The operation being decomposed need to implement this Interface by implementing
    the method `decomposeOperation` and return the set of values which would replace
    the uses of the operation being decomposed.
    Eg:
        Assume there is an operation `CustomOp_Mul_Add` that takes in an input tensor
        and a constant. It basically performs element-wise multiplication of the input
        tensor with the given constant, and then performs element-wise addition of the
        intermediate resulting tensor with the given constant.
        `CustomOp_Mul_Add` can thus essentially be decomposed by implementing this
        Interface.
        `linalg::SoftmaxOp` is one such operation which makes use of this Interface
        for implementing its decomposition.
  }];
  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Method to decompose the operation into simpler operations.

          On success, this method returns one `Value` per result in the
          original operation.
          The order of the returned values must match the order of the
          original values.
          In other words, the returned vector can be used directly with
          `RewriterBase::replaceOp(this, returnedValues)`.
        }],
        /*retType=*/"FailureOr<SmallVector<Value>>",
        /*methodName=*/"decomposeOperation",
        /*args=*/(ins
            "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return {};
        }]
      >
  ];
}

#endif // MLIR_AGGREGATEDOPINTERFACE
