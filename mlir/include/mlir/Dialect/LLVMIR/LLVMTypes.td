//===-- LLVMOps.td - LLVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMTYPES_TD
#define LLVMTYPES_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

/// Base class for all LLVM dialect types.
class LLVMType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLVM_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// LLVMArrayType
//===----------------------------------------------------------------------===//

def LLVMArrayType : LLVMType<"LLVMArray", "array", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>,
    DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "LLVM array type";
  let description = [{
    The `!llvm.array` type represents a fixed-size array of element types.
    It is an aggregate type representing consecutive elements in memory,
    parameterized by the number of elements and the element type.

    Example:

    ```mlir
    !llvm.array<4 x i32>
    ```
  }];

  let parameters = (ins "Type":$elementType, "unsigned":$numElements);
  let assemblyFormat = [{
    `<` $numElements `x` ` ` custom<PrettyLLVMType>($elementType) `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "unsigned":$numElements)>
  ];

  let extraClassDeclaration = [{
    /// Checks if the given type can be used inside an array type.
    static bool isValidElementType(Type type);
  }];
}

#endif // LLVMTYPES_TD
