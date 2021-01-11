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
// ComplexType
//===----------------------------------------------------------------------===//

def Builtin_Complex : Builtin_Type<"Complex"> {
  let summary = "Complex number with a parameterized element type";
  let description = [{
    Syntax:

    ```
    complex-type ::= `complex` `<` type `>`
    ```

    The value of `complex` type represents a complex number with a parameterized
    element type, which is composed of a real and imaginary value of that
    element type. The element must be a floating point or integer scalar type.

    Examples:

    ```mlir
    complex<f32>
    complex<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }], [{
      return Base::getChecked($_loc, elementType);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyInvariantsDecl = 1;
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
// FunctionType
//===----------------------------------------------------------------------===//

def Builtin_Function : Builtin_Type<"Function"> {
  let summary = "Map from a list of inputs to a list of results";
  let description = [{
    Syntax:

    ```
    // Function types may have multiple results.
    function-result-type ::= type-list-parens | non-function-type
    function-type ::= type-list-parens `->` function-result-type
    ```

    The function type can be thought of as a function signature. It consists of
    a list of formal parameter types and a list of formal result types.
    ```
  }];
  let parameters = (ins "ArrayRef<Type>":$inputs, "ArrayRef<Type>":$results);
  let builders = [
    TypeBuilder<(ins CArg<"TypeRange">:$inputs, CArg<"TypeRange">:$results), [{
      return Base::get($_ctxt, inputs, results);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genStorageClass = 0;
  let extraClassDeclaration = [{
    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }

    /// Returns a new function type without the specified arguments and results.
    FunctionType getWithoutArgsAndResults(ArrayRef<unsigned> argIndices,
                                          ArrayRef<unsigned> resultIndices);
  }];
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
// IntegerType
//===----------------------------------------------------------------------===//

def Builtin_Integer : Builtin_Type<"Integer"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    Syntax:

    ```
    // Sized integers like i1, i4, i8, i16, i32.
    signed-integer-type ::= `si` [1-9][0-9]*
    unsigned-integer-type ::= `ui` [1-9][0-9]*
    signless-integer-type ::= `i` [1-9][0-9]*
    integer-type ::= signed-integer-type |
                     unsigned-integer-type |
                     signless-integer-type
    ```

    Integer types have a designated bit width and may optionally have signedness
    semantics.

    **Rationale:** low precision integers (like `i2`, `i4` etc) are useful for
    low-precision inference chips, and arbitrary precision integers are useful
    for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller
    than a 16 bit one).
  }];
  let parameters = (ins "unsigned":$width, "SignednessSemantics":$signedness);
  let builders = [
    TypeBuilder<(ins "unsigned":$width,
                     CArg<"SignednessSemantics", "Signless">:$signedness)>
  ];

  // IntegerType uses a special storage class that compacts parameters to save
  // memory.
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
  let genVerifyInvariantsDecl = 1;
  let extraClassDeclaration = [{
    /// Signedness semantics.
    enum SignednessSemantics : uint32_t {
      Signless, /// No signedness semantics
      Signed,   /// Signed integer
      Unsigned, /// Unsigned integer
    };

    /// Return true if this is a signless integer type.
    bool isSignless() const { return getSignedness() == Signless; }
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getSignedness() == Signed; }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return getSignedness() == Unsigned; }

    /// Get or create a new IntegerType with the same signedness as `this` and a
    /// bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    IntegerType scaleElementBitwidth(unsigned scale);

    /// Integer representation maximal bitwidth.
    /// Note: This is aligned with the maximum width of llvm::IntegerType.
    static constexpr unsigned kMaxWidth = (1 << 24) - 1;
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

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

def Builtin_Opaque : Builtin_Type<"Opaque"> {
  let summary = "Type of a non-registered dialect";
  let description = [{
    Syntax:

    ```
    opaque-type ::= `opaque` `<` type `>`
    ```

    Opaque types represent types of non-registered dialects. These are types
    represented in their raw string form, and can only usefully be tested for
    type equality.

    Examples:

    ```mlir
    opaque<"llvm", "struct<(i32, float)>">
    opaque<"pdl", "value">
    ```
  }];
  let parameters = (ins
    "Identifier":$dialectNamespace,
    StringRefParameter<"">:$typeData
  );
  let genVerifyInvariantsDecl = 1;
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

def Builtin_Tuple : Builtin_Type<"Tuple"> {
  let summary = "Fixed-sized collection of other types";
  let description = [{
    Syntax:

    ```
    tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
    ```

    The value of `tuple` type represents a fixed-size collection of elements,
    where each element may be of a different type.

    **Rationale:** Though this type is first class in the type system, MLIR
    provides no standard operations for operating on `tuple` types
    ([rationale](Rationale/Rationale.md#tuple-types)).

    Examples:

    ```mlir
    // Empty tuple.
    tuple<>

    // Single element
    tuple<f32>

    // Many elements.
    tuple<i32, f32, tensor<i1>, i5>
    ```
  }];
  let parameters = (ins "ArrayRef<Type>":$types);
  let builders = [
    TypeBuilder<(ins "TypeRange":$elementTypes), [{
      return Base::get($_ctxt, elementTypes);
    }]>,
    TypeBuilder<(ins), [{
      return Base::get($_ctxt, TypeRange());
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genStorageClass = 0;
  let extraClassDeclaration = [{
    /// Accumulate the types contained in this tuple and tuples nested within
    /// it. Note that this only flattens nested tuples, not any other container
    /// type, e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is
    /// flattened to (i32, tensor<i32>, f32, i64)
    void getFlattenedTypes(SmallVectorImpl<Type> &types);

    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<Type>::iterator;
    iterator begin() const { return getTypes().begin(); }
    iterator end() const { return getTypes().end(); }

    /// Return the element type at index 'index'.
    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getTypes()[index];
    }
  }];
}

#endif // BUILTIN_TYPES
