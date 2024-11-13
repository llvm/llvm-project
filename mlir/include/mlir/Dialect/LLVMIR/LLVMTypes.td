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

  let parameters = (ins "Type":$elementType, "uint64_t":$numElements);
  let assemblyFormat = [{
    `<` $numElements `x` custom<PrettyLLVMType>($elementType) `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "uint64_t":$numElements)>
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
// LLVMStructType
//===----------------------------------------------------------------------===//

def LLVMStructType : LLVMType<"LLVMStruct", "struct", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
    ["areCompatible", "verifyEntries"]>,
  DeclareTypeInterfaceMethods<DestructurableTypeInterface>
]> {
  let summary = "LLVM struct type";

  let description = [{
    The `!llvm.struct` type is used to represent a sequential collection of
    types within memory.
    By default, all elements within the struct are aligned according to their
    type unless the `$packed` parameter is set to true.
    In that case, all elements are tightly packed with no padding inbetween
    elements.

    A struct may additionally contain a name.
    Two struct types are considered equal if their body matches and they have
    the same name.
    Note that this differs from LLVM IR where LLVM would rename structs instead
    to avoid name clashes, meaning two structs with the same name are also
    equal.

    Instead of a body, a struct may also be marked as "opaque".
    This signals that the struct definition is complete and the body of the
    struct unknown.
    An opaque struct must have a name.
  }];

  let parameters = (ins
    StringRefParameter<"struct name", [{""}]>:$name,
    OptionalArrayRefParameter<"Type">:$body,
    ImpliedBoolParameter<>:$packed,
    ImpliedBoolParameter<>:$opaque
  );

  // We do not want a builder that may set opaque to true and supply a body
  // at the same time.
  let skipDefaultBuilders = 1;

   let builders = [
     TypeBuilder<(ins
       "llvm::StringRef":$name,
       "llvm::ArrayRef<mlir::Type>":$body,
       CArg<"bool", "false">:$isPacked), [{
       return Base::get($_ctxt, name, body, isPacked, /*opaque=*/false);
     }]>
   ];

  let extraClassDeclaration = [{
    /// Checks if the given type can be contained in a structure type.
    static bool isValidElementType(Type type);

    /// Returns true if the struct has a name.
    bool isIdentified() const {
      return !getName().empty();
    }

    bool isPacked() const {
      return getPacked();
    }

    bool isOpaque() const {
      return getOpaque();
    }

    /// Destructs the struct into its indexed field types.
    std::optional<DenseMap<Attribute, Type>> getSubelementIndexMap();

    /// Returns which type is stored at a given integer index within the struct.
    Type getTypeAtIndex(Attribute index);

    /// Gets or creates a literal struct with the given body in the provided
    /// context.
    static LLVMStructType getLiteral(MLIRContext *context, ArrayRef<Type> types,
                                     bool isPacked = false);

    static LLVMStructType
    getLiteralChecked(function_ref<InFlightDiagnostic()> emitError,
                      MLIRContext *context, ArrayRef<Type> types,
                      bool isPacked = false);

    /// Gets or creates an intentionally-opaque identified struct. Such a struct
    /// cannot have its body set.
    /// Note that unlike llvm::StructType::create, this function will _NOT_
    /// rename a struct in case a struct with the same name
    /// already exists in the context. Instead, it will just return the existing
    /// struct, similarly to the rest of MLIR type ::get methods.
    static LLVMStructType getOpaque(StringRef name, MLIRContext *context);

    static LLVMStructType
    getOpaqueChecked(function_ref<InFlightDiagnostic()> emitError,
                     MLIRContext *context, StringRef name);
  }];

  let assemblyFormat = [{
    `<` (custom<OptionalName>($name)^ `,` ` `)?
        (`opaque` `` $opaque^) :
          ( (`packed` $packed^)? `(` (`)`)
            : (custom<PrettyLLVMType>($body)^ `)`)? )? `>`
  }];

  let genVerifyDecl = 1;
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

//===----------------------------------------------------------------------===//
// LLVMX86AMXType
//===----------------------------------------------------------------------===//

def LLVMX86AMXType : LLVMType<"LLVMX86AMX", "x86_amx"> {
  let summary = "LLVM x86_amx type.";
  let description = [{
    The x86_amx type represents a value held in an AMX tile register on an x86
    machine. Can only be used in AMX intrinsics calls.
  }];
}

#endif // LLVMTYPES_TD
