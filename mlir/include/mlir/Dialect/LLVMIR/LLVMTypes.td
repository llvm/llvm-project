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
include "mlir/IR/BuiltinTypeInterfaces.td"
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
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
                                ["getTypeSize", "getPreferredAlignment"]>,
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
  MutableType,
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
    ["areCompatible", "verifyEntries", "getPreferredAlignment"]>,
  DeclareTypeInterfaceMethods<DestructurableTypeInterface,
    ["getSubelementIndexMap", "getTypeAtIndex"]>
]> {
  let summary = "LLVM struct type";

  let description = [{
    LLVM dialect structure type representing a collection of different-typed
    elements manipulated together. Struct types can optionally be packed, meaning
    that their elements immediately follow each other in memory without
    accounting for potential alignment.

    Structure types can be identified (named) or literal. Literal structures
    are uniquely represented by the list of types they contain and packedness.
    Literal structure types are immutable after construction.

    Identified structures are uniquely represented by their name, a string. They
    have a mutable component, consisting of the list of types they contain,
    the packedness and the opacity bits. Identified structs can be created
    without providing the lists of element types, making them suitable to
    represent recursive, i.e. self-referring, structures. Identified structs
    without body are considered opaque. For such structs, one can set the body.
    Identified structs can be created as intentionally-opaque, implying that the
    caller does not intend to ever set the body (e.g. forward-declarations of
    structs from another module) and wants to disallow further modification of
    the body. For intentionally-opaque structs or non-opaque structs with the
    body, one is not allowed to set another body (however, one can set exactly
    the same body).

    Note that the packedness of the struct takes place in uniquing of literal
    structs, but does not in uniquing of identified structs.
  }];

  // Specify parameters for which TableGen can generate convenient getters for
  // us.
  // TODO: Other parameters such as 'packed' or 'opaque' could be added in the
  //       future iff they generate getters prefixed with 'is', instead of
  //       'get'. Until then there are no advantages in doing so.
  let parameters = (ins
    StringRefParameter<"struct name", [{""}]>:$name,
    OptionalArrayRefParameter<"mlir::Type">:$body
  );

  // A custom storage class defined in C++ is required to implement mutability.
  let storageClass = "LLVMStructTypeStorage";
  let genStorageClass = 0;

  // We want users to use the more aptly named custom builders below.
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Checks if the given type can be contained in a structure type.
    static bool isValidElementType(Type type);

    /// Gets or creates an identified struct with the given name in the provided
    /// context. Note that unlike llvm::StructType::create, this function will
    /// _NOT_ rename a struct in case a struct with the same name already exists
    /// in the context. Instead, it will just return the existing struct,
    /// similarly to the rest of MLIR type ::get methods.
    static LLVMStructType getIdentified(MLIRContext *context, StringRef name);
    static LLVMStructType
    getIdentifiedChecked(function_ref<InFlightDiagnostic()> emitError,
                         MLIRContext *context, StringRef name);

    /// Gets a new identified struct with the given body. The body _cannot_ be
    /// changed later. If a struct with the given name already exists, renames
    /// the struct by appending a `.` followed by a number to the name. Renaming
    /// happens even if the existing struct has the same body.
    static LLVMStructType getNewIdentified(MLIRContext *context, StringRef name,
                                           ArrayRef<Type> elements,
                                           bool isPacked = false);

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

    /// Set the body of an identified struct. Returns failure if the body could
    /// not be set, e.g. if the struct already has a body or if it was marked as
    /// intentionally opaque. This might happen in a multi-threaded context when a
    /// different thread modified the struct after it was created. Most callers
    /// are likely to assert this always succeeds, but it is possible to implement
    /// a local renaming scheme based on the result of this call.
    LogicalResult setBody(ArrayRef<Type> types, bool isPacked);

    /// Checks if a struct is packed.
    bool isPacked() const;

    /// Checks if a struct is identified.
    bool isIdentified() const;

    /// Checks if a struct is opaque.
    bool isOpaque() const;

    /// Checks if a struct is initialized.
    bool isInitialized();

    /// Verifies that the type about to be constructed is well-formed.
    static LogicalResult
    verifyInvariants(function_ref<InFlightDiagnostic()> emitError, StringRef,
                     bool);
    static LogicalResult
    verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                     ArrayRef<Type> types, bool);
    using Base::verifyInvariants;
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// LLVMPointerType
//===----------------------------------------------------------------------===//

def LLVMPointerType : LLVMType<"LLVMPointer", "ptr", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "getIndexBitwidth", "areCompatible", "verifyEntries",
      "getPreferredAlignment"]>,
    VectorElementTypeInterface]> {
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
    LLVM dialect vector type that supports all element types that are supported
    in LLVM vectors but that are not supported by the builtin MLIR vector type.
    E.g., LLVMFixedVectorType supports LLVM pointers as element type.
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
