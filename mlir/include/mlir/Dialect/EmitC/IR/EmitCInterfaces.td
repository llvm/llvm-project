//===- EmitCInterfaces.td - EmitC Interfaces ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces used by EmitC.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCINTERFACES
#define MLIR_DIALECT_EMITC_IR_EMITCINTERFACES

include "mlir/IR/OpBase.td"

def CExpressionInterface : OpInterface<"CExpressionInterface"> {
  let description = [{
    Interface to mark operations that can be part of the CExpression.
  }];

  let cppNamespace = "::mlir::emitc";
  let methods = [
    InterfaceMethod<[{
      Check whether operation has side effects that may affect the expression
      evaluation.

      By default operation is marked as having side effects.

      ```c++
      class ConcreteOp ... {
      public:
        bool hasSideEffects() {
          // That way we can override the default implementation.
          return false;
        }
      };
      ```
    }],
      "bool", "hasSideEffects", (ins), /*methodBody=*/[{}],
       /*defaultImplementation=*/[{
        return true;
    }]>,
    InterfaceMethod<[{
      Check whether operation must be inlined into all its users.

      By default operation is not marked as always inlined.

      ```c++
      class ConcreteOp ... {
      public:
        bool alwaysInline() {
          // That way we can override the default implementation.
          return true;
        }
      };
      ```
    }],
    "bool", "alwaysInline", (ins), /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
        return false;
    }]>,
  ];
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCINTERFACES
