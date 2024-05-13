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
include "mlir/Interfaces/MemorySlotInterfaces.td"

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
    DeclareTypeInterfaceMethods<DestructurableTypeInterface>]> {
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
      "getIndexBitwidth", "areCompatible", "verifyEntries"]>]> {
  let summary = "LLVM pointer type";
  let description = [{
    The `!llvm.ptr` type is an LLVM pointer type. This type typically represents
    a reference to an object in memory. Pointers are optionally parameterized
    by the address space.

    Example:

    ```mlir
    !llvm.ptr
    ```
  }];

  let parameters = (ins DefaultValuedParameter<"unsigned", "0">:$addressSpace);
  let assemblyFormat = [{
    (`<` $addressSpace^ `>`)?
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins CArg<"unsigned", "0">:$addressSpace), [{
      return $_get($_ctxt, addressSpace);
    }]>
  ];
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

  let typeName = "llvm.fixed_vec";

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

  let typeName = "llvm.scalable_vec";

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

//===----------------------------------------------------------------------===//
// LLVMTargetExtType
//===----------------------------------------------------------------------===//

def LLVMTargetExtType : LLVMType<"LLVMTargetExt", "target"> {
  let summary = "LLVM target-specific extension type";
  let description = [{
    LLVM dialect target extension type, which are generally unintrospectable
    from target-independent optimizations.

    Target extension types have a string name, and optionally have type and/or
    integer parameters. The exact meaning of any parameters is dependent on the
    target.
  }];

  let parameters = (ins StringRefParameter<>:$extTypeName,
                        OptionalArrayRefParameter<"Type">:$typeParams,
                        OptionalArrayRefParameter<"unsigned int">:$intParams);

  let assemblyFormat = [{
    `<` $extTypeName (`,` custom<ExtTypeParams>($typeParams, $intParams)^ )? `>`
  }];

  let extraClassDeclaration = [{
    enum Property {
      /// zeroinitializer is valid for this target extension type.
      HasZeroInit = 1U << 0,
      /// This type may be used as the value type of a global variable.
      CanBeGlobal = 1U << 1,
    };

    bool hasProperty(Property Prop) const;
    bool supportsMemOps() const;
  }];
}

#endif // LLVMTYPES_TD
