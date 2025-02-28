//===- BytecodeOpInterface.td - Bytecode OpInterface -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface for operation interactions with the bytecode
// serialization/deserialization, in particular for properties.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEOPINTERFACES
#define MLIR_BYTECODE_BYTECODEOPINTERFACES

include "mlir/IR/OpBase.td"

// `BytecodeOpInterface`
def BytecodeOpInterface : OpInterface<"BytecodeOpInterface"> {
  let description = [{
    This interface allows operation to control the serialization of their
    properties.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<[{
      Read the properties for this operation from the bytecode and populate the state.
      }],
      "LogicalResult", "readProperties", (ins
          "::mlir::DialectBytecodeReader &":$reader,
          "::mlir::OperationState &":$state)
    >,
    InterfaceMethod<[{
      Write the properties for this operation to the bytecode.
      }],
      "void", "writeProperties", (ins "::mlir::DialectBytecodeWriter &":$writer)
    >,
  ];
}

// `FallbackBytecodeOpInterface`
def FallbackBytecodeOpInterface : OpInterface<"FallbackBytecodeOpInterface"> {
  let description = [{
    This interface allows fallback operations direct access to the bytecode
    property streams.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<[{
      Set the original name for this operation from the bytecode.
      }],
      "void", "setOriginalOperationName", (ins
          "::mlir::StringRef":$opName,
          "::mlir::OperationState &":$state)
    >,
    StaticInterfaceMethod<[{
      Read the properties blob for this operation from the bytecode and populate the state.
      }],
      "::mlir::LogicalResult", "readPropertiesBlob", (ins
          "::mlir::ArrayRef<char>":$blob,
          "::mlir::OperationState &":$state)
    >,
    InterfaceMethod<[{
      Get the properties blob for this operation to be emitted into the bytecode.
      }],
      "::mlir::ArrayRef<char>", "getPropertiesBlob", (ins)
    >,
  ];
}

#endif // MLIR_BYTECODE_BYTECODEOPINTERFACES
