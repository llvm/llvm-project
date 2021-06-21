//===- EmitCTypes.td - EmitC types -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC types.
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_EMITC_IR_EMITCTYPES
#define MLIR_DIALECT_EMITC_IR_EMITCTYPES

include "mlir/Dialect/EmitC/IR/EmitCBase.td"

//===----------------------------------------------------------------------===//
// EmitC type definitions
//===----------------------------------------------------------------------===//

class EmitC_Type<string name, string typeMnemonic>
    : TypeDef<EmitC_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def EmitC_OpaqueType : EmitC_Type<"Opaque", "opaque"> {
  let summary = "An opaque type";

  let description = [{
    An opaque data type of which the value gets emitted as is.

    Example:

    ```mlir
    !emitc.opaque<"int">
    !emitc.opaque<"float *">
    !emitc.opaque<"std::vector<std::string>">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCTYPES
