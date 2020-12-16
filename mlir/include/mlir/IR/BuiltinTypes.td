//===- BuiltinTypes.td - Builtin type definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the set of builtin MLIR types, or the set of types necessary for the
// validity of and defining the IR.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_TYPES
#define BUILTIN_TYPES

include "mlir/IR/BuiltinDialect.td"

// TODO: Currently the types defined in this file are prefixed with `Builtin_`.
// This is to differentiate the types here with the ones in OpBase.td. We should
// remove the definitions in OpBase.td, and repoint users to this file instead.

// Base class for Builtin dialect types.
class Builtin_Type<string name> : TypeDef<Builtin_Dialect, name> {
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

// Base class for Builtin dialect float types.
class Builtin_FloatType<string name> : TypeDef<Builtin_Dialect, name,
                                               "::mlir::FloatType"> {
  let extraClassDeclaration = [{
    static }] # name # [{Type get(MLIRContext *context);
  }];
}

//===----------------------------------------------------------------------===//
// BFloat16Type

def Builtin_BFloat16 : Builtin_FloatType<"BFloat16"> {
  let summary = "bfloat16 floating-point type";
}

//===----------------------------------------------------------------------===//
// Float16Type

def Builtin_Float16 : Builtin_FloatType<"Float16"> {
  let summary = "16-bit floating-point type";
}

//===----------------------------------------------------------------------===//
// Float32Type

def Builtin_Float32 : Builtin_FloatType<"Float32"> {
  let summary = "32-bit floating-point type";
}

//===----------------------------------------------------------------------===//
// Float64Type

def Builtin_Float64 : Builtin_FloatType<"Float64"> {
  let summary = "64-bit floating-point type";
}

//===----------------------------------------------------------------------===//
// IndexType
//===----------------------------------------------------------------------===//

def Builtin_Index : Builtin_Type<"Index"> {
  let summary = "Integer-like type with unknown platform-dependent bit width";
  let description = [{
    Syntax:

    ```
    // Target word-sized integer.
    index-type ::= `index`
    ```

    The index type is a signless integer whose size is equal to the natural
    machine word of the target ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics) )
    and is used by the affine constructs in MLIR. Unlike fixed-size integers,
    it cannot be used as an element of vector ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/#index-type-disallowed-in-vector-types) ).

    **Rationale:** integers of platform-specific bit widths are practical to
    express sizes, dimensionalities and subscripts.
  }];
  let extraClassDeclaration = [{
    static IndexType get(MLIRContext *context);

    /// Storage bit width used for IndexType by internal compiler data
    /// structures.
    static constexpr unsigned kInternalStorageBitWidth = 64;
  }];
}

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

def Builtin_None : Builtin_Type<"None"> {
  let summary = "A unit type";
  let description = [{
    NoneType is a unit type, i.e. a type with exactly one possible value, where
    its value does not have a defined dynamic representation.
  }];
  let extraClassDeclaration = [{
    static NoneType get(MLIRContext *context);
  }];
}

#endif // BUILTIN_TYPES
