//===- InferIntDivisibilityOpInterface.td - Integer Divisibility -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for divisibility analysis on scalar integers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERINTDIVISIBILITYOPINTERFACE
#define MLIR_INTERFACES_INFERINTDIVISIBILITYOPINTERFACE

include "mlir/IR/OpBase.td"

def InferIntDivisibilityOpInterface :
    OpInterface<"InferIntDivisibilityOpInterface"> {
  let description = [{
    Allows operations to participate in integer divisibility analysis.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Infer the divisibility of the results of this op given the
        divisibility of its arguments. For each result value, the method
        should call `setResultDivs` with that `Value` as an argument.
      }],
      /*retTy=*/"void",
      /*methodName=*/"inferResultDivisibility",
      /*args=*/(ins
        "::llvm::ArrayRef<::mlir::IntegerDivisibility>":$argDivs,
        "::mlir::SetIntDivisibilityFn":$setResultDivs)
    >
  ];
}

#endif // MLIR_INTERFACES_INFERINTDIVISIBILITYOPINTERFACE
