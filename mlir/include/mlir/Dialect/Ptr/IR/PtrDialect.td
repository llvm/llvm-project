//===- PtrDialect.td - Pointer dialect ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_DIALECT
#define PTR_DIALECT

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pointer dialect definition.
//===----------------------------------------------------------------------===//

def Ptr_Dialect : Dialect {
  let name = "ptr";
  let summary = "Pointer dialect";
  let cppNamespace = "::mlir::ptr";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Pointer type definitions
//===----------------------------------------------------------------------===//

class Ptr_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ptr_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ptr_PtrType : Ptr_Type<"Ptr", "ptr", [
    MemRefElementTypeInterface,
    VectorElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "areCompatible", "getIndexBitwidth", "verifyEntries",
      "getPreferredAlignment"]>
  ]> {
  let summary = "pointer type";
  let description = [{
    The `ptr` type is an opaque pointer type. This type typically represents a
    handle to an object in memory or target-dependent values like `nullptr`.
    Pointers are parameterized by a memory space.

    Syntax:

    ```mlir
    pointer ::= `ptr` (`<` memory-space `>`)?
    memory-space ::= attribute-value
    ```
  }];
  let parameters = (ins "MemorySpaceAttrInterface":$memorySpace);
  let assemblyFormat = "`<` $memorySpace `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "MemorySpaceAttrInterface":$memorySpace), [{
      return $_get(memorySpace.getContext(), memorySpace);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Base address operation definition.
//===----------------------------------------------------------------------===//

class Pointer_Op<string mnemonic, list<Trait> traits = []> :
        Op<Ptr_Dialect, mnemonic, traits>;

#endif // PTR_DIALECT
