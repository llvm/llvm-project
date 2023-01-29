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
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>]> {
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
    `<` $numElements `x` custom<PrettyLLVMType>($elementType) `>`
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

//===----------------------------------------------------------------------===//
// LLVMFunctionType
//===----------------------------------------------------------------------===//

def LLVMFunctionType : LLVMType<"LLVMFunction", "func"> {
  let summary = "LLVM function type";
  let description = [{
    The `!llvm.func` is a function type. It consists of a single return type
    (unlike MLIR which can have multiple), a list of parameter types and can
    optionally be variadic.

    Example:

    ```mlir
    !llvm.func<i32 (i32)>
    ```
  }];

  let parameters = (ins "Type":$returnType, ArrayRefParameter<"Type">:$params,
                        "bool":$varArg);
  let assemblyFormat = [{
    `<` custom<PrettyLLVMType>($returnType) ` ` `(`
    custom<FunctionTypes>($params, $varArg) `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$result, "ArrayRef<Type>":$arguments,
      CArg<"bool", "false">:$isVarArg)>
  ];

  let extraClassDeclaration = [{
    /// Checks if the given type can be used an argument in a function type.
    static bool isValidArgumentType(Type type);

    /// Checks if the given type can be used as a result in a function type.
    static bool isValidResultType(Type type);

    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    /// Returns a clone of this function type with the given argument
    /// and result types.
    LLVMFunctionType clone(TypeRange inputs, TypeRange results) const;

    /// Returns the result type of the function as an ArrayRef, enabling better
    /// integration with generic MLIR utilities.
    ArrayRef<Type> getReturnTypes() const;

    /// Returns the number of arguments to the function.
    unsigned getNumParams() const { return getParams().size(); }

    /// Returns `i`-th argument of the function. Asserts on out-of-bounds.
    Type getParamType(unsigned i) { return getParams()[i]; }
  }];
}

//===----------------------------------------------------------------------===//
// LLVMPointerType
//===----------------------------------------------------------------------===//

def LLVMPointerType : LLVMType<"LLVMPointer", "ptr", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "areCompatible", "verifyEntries"]>]> {
  let summary = "LLVM pointer type";
  let description = [{
    The `!llvm.ptr` type is an LLVM pointer type. This type typically represents
    a reference to an object in memory. Pointers may be opaque or parameterized
    by the element type. Both opaque and non-opaque pointers are additionally
    parameterized by the address space.

    Example:

    ```mlir
    !llvm.ptr<i8>
    !llvm.ptr
    ```
  }];

  let parameters = (ins DefaultValuedParameter<"Type", "Type()">:$elementType,
                        DefaultValuedParameter<"unsigned", "0">:$addressSpace);
  let assemblyFormat = [{
    (`<` custom<Pointer>($elementType, $addressSpace)^ `>`)?
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                         CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilder<(ins CArg<"unsigned", "0">:$addressSpace), [{
      return $_get($_ctxt, Type(), addressSpace);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns `true` if this type is the opaque pointer type, i.e., it has no
    /// pointed-to type.
    bool isOpaque() const { return !getElementType(); }

    /// Checks if the given type can have a pointer type pointing to it.
    static bool isValidElementType(Type type);
  }];
}

//===----------------------------------------------------------------------===//
// LLVMFixedVectorType
//===----------------------------------------------------------------------===//

def LLVMFixedVectorType : LLVMType<"LLVMFixedVector", "vec"> {
  let summary = "LLVM fixed vector type";
  let description = [{
    LLVM dialect scalable vector type, represents a sequence of elements of
    unknown length that is known to be divisible by some constant. These
    elements can be processed as one in SIMD context.
  }];

  let parameters = (ins "Type":$elementType, "unsigned":$numElements);
  let assemblyFormat = [{
    `<` $numElements `x` custom<PrettyLLVMType>($elementType) `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "unsigned":$numElements)>
  ];

  let extraClassDeclaration = [{
    /// Checks if the given type can be used in a vector type.
    static bool isValidElementType(Type type);
  }];
}

//===----------------------------------------------------------------------===//
// LLVMScalableVectorType
//===----------------------------------------------------------------------===//

def LLVMScalableVectorType : LLVMType<"LLVMScalableVector", "vec"> {
  let summary = "LLVM scalable vector type";
  let description = [{
    LLVM dialect scalable vector type, represents a sequence of elements of
    unknown length that is known to be divisible by some constant. These
    elements can be processed as one in SIMD context.
  }];

  let parameters = (ins "Type":$elementType, "unsigned":$minNumElements);
  let assemblyFormat = [{
    `<` `?` `x` $minNumElements `x` ` ` custom<PrettyLLVMType>($elementType) `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "unsigned":$minNumElements)>
  ];

  let extraClassDeclaration = [{
    /// Checks if the given type can be used in a vector type.
    static bool isValidElementType(Type type);
  }];
}

#endif // LLVMTYPES_TD
