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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

// TODO: Currently the types defined in this file are prefixed with `Builtin_`.
// This is to differentiate the types here with the ones in OpBase.td. We should
// remove the definitions in OpBase.td, and repoint users to this file instead.

// Base class for Builtin dialect types.
class Builtin_Type<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Builtin_Dialect, name, traits, baseCppClass> {
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
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

// Base class for Builtin dialect float types.
class Builtin_FloatType<string name>
    : Builtin_Type<name, /*traits=*/[], "::mlir::FloatType"> {
  let extraClassDeclaration = [{
    static }] # name # [{Type get(MLIRContext *context);
  }];
}

//===----------------------------------------------------------------------===//
// Float8E5M2Type

def Builtin_Float8E5M2 : Builtin_FloatType<"Float8E5M2"> {
  let summary = "8-bit floating point with 2 bit mantissa";
  let description = [{
    An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E5M2
      * exponent bias: 15
      * infinities: supported with exponent set to all 1s and mantissa 0s
      * NaNs: supported with exponent bits set to all 1s and mantissa of
        (01, 10, or 11)
      * denormals when exponent is 0

    Described in: https://arxiv.org/abs/2209.05433
  }];
}

//===----------------------------------------------------------------------===//
// Float8E4M3FNType

def Builtin_Float8E4M3FN : Builtin_FloatType<"Float8E4M3FN"> {
  let summary = "8-bit floating point with 3 bit mantissa";
  let description = [{
    An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it follows
    similar conventions, with the exception that there are no infinity values
    and only two NaN representations. This type has the following
    characteristics:

      * bit encoding: S1E4M3
      * exponent bias: 7
      * infinities: Not supported
      * NaNs: supported with exponent bits and mantissa bits set to all 1s
      * denormals when exponent is 0

    Described in: https://arxiv.org/abs/2209.05433
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
// Float80Type

def Builtin_Float80 : Builtin_FloatType<"Float80"> {
  let summary = "80-bit floating-point type";
}

//===----------------------------------------------------------------------===//
// Float128Type

def Builtin_Float128 : Builtin_FloatType<"Float128"> {
  let summary = "128-bit floating-point type";
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
  }];
  let parameters = (ins "ArrayRef<Type>":$inputs, "ArrayRef<Type>":$results);
  let builders = [
    TypeBuilder<(ins CArg<"TypeRange">:$inputs, CArg<"TypeRange">:$results), [{
      return $_get($_ctxt, inputs, results);
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

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FunctionType clone(TypeRange inputs, TypeRange results) const;

    /// Returns a new function type with the specified arguments and results
    /// inserted.
    FunctionType getWithArgsAndResults(ArrayRef<unsigned> argIndices,
                                       TypeRange argTypes,
                                       ArrayRef<unsigned> resultIndices,
                                       TypeRange resultTypes);

    /// Returns a new function type without the specified arguments and results.
    FunctionType getWithoutArgsAndResults(const BitVector &argIndices,
                                          const BitVector &resultIndices);
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
    machine word of the target ( [rationale](../../Rationale/Rationale/#integer-signedness-semantics) )
    and is used by the affine constructs in MLIR.

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
  let genVerifyDecl = 1;
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
// MemRefType
//===----------------------------------------------------------------------===//

def Builtin_MemRef : Builtin_Type<"MemRef", [
    ShapedTypeInterface
  ], "BaseMemRefType"> {
  let summary = "Shaped reference to a region of memory";
  let description = [{
    Syntax:

    ```
    memref-type ::= `memref` `<` dimension-list-ranked type
                    (`,` layout-specification)? (`,` memory-space)? `>`
    layout-specification ::= attribute-value
    memory-space ::= attribute-value
    ```

    A `memref` type is a reference to a region of memory (similar to a buffer
    pointer, but more powerful). The buffer pointed to by a memref can be
    allocated, aliased and deallocated. A memref can be used to read and write
    data from/to the memory region which it references. Memref types use the
    same shape specifier as tensor types. Note that `memref<f32>`,
    `memref<0 x f32>`, `memref<1 x 0 x f32>`, and `memref<0 x 1 x f32>` are all
    different types.

    A `memref` is allowed to have an unknown rank (e.g. `memref<*xf32>`). The
    purpose of unranked memrefs is to allow external library functions to
    receive memref arguments of any rank without versioning the functions based
    on the rank. Other uses of this type are disallowed or will have undefined
    behavior.

    Are accepted as elements:

    - built-in integer types;
    - built-in index type;
    - built-in floating point types;
    - built-in vector types with elements of the above types;
    - another memref type;
    - any other type implementing `MemRefElementTypeInterface`.

    ##### Layout

    A memref may optionally have a layout that indicates how indices are
    transformed from the multi-dimensional form into a linear address. The
    layout must avoid internal aliasing, i.e., two distinct tuples of
    _in-bounds_ indices must be pointing to different elements in memory. The
    layout is an attribute that implements `MemRefLayoutAttrInterface`. The
    bulitin dialect offers two kinds of layouts: strided and affine map, each
    of which is available as an attribute. Other attributes may be used to
    represent the layout as long as they can be converted to a
    [semi-affine map](Affine.md/#semi-affine-maps) and implement the required
    interface. Users of memref are expected to fallback to the affine
    representation when handling unknown memref layouts. Multi-dimensional
    affine forms are interpreted in _row-major_ fashion.

    In absence of an explicit layout, a memref is considered to have a
    multi-dimensional identity affine map layout.  Identity layout maps do not
    contribute to the MemRef type identification and are discarded on
    construction. That is, a type with an explicit identity map is
    `memref<?x?xf32, (i,j)->(i,j)>` is strictly the same as the one without a
    layout, `memref<?x?xf32>`.

    ##### Affine Map Layout

    The layout may be represented directly as an affine map from the index space
    to the storage space. For example, the following figure shows an index map
    which maps a 2-dimensional index from a 2x2 index space to a 3x3 index
    space, using symbols `S0` and `S1` as offsets.

    ![Index Map Example](/includes/img/index-map.svg)

    Semi-affine maps are sufficiently flexible to represent a wide variety of
    dense storage layouts, including row- and column-major and tiled:

    ```mlir
    // MxN matrix stored in row major layout in memory:
    #layout_map_row_major = (i, j) -> (i, j)

    // MxN matrix stored in column major layout in memory:
    #layout_map_col_major = (i, j) -> (j, i)

    // MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
    #layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
    ```

    ##### Strided Layout

    Memref layout can be expressed using strides to encode the distance, in
    number of elements, in (linear) memory between successive entries along a
    particular dimension. For example, a row-major strided layout for
    `memref<2x3x4xf32>` is `strided<[12, 4, 1]>`, where the last dimension is
    contiguous as indicated by the unit stride and the remaining strides are
    products of the sizes of faster-variying dimensions. Strided layout can also
    express non-contiguity, e.g., `memref<2x3, strided<[6, 2]>>` only accesses
    even elements of the dense consecutive storage along the innermost
    dimension.

    The strided layout supports an optional _offset_ that indicates the
    distance, in the number of elements, between the beginning of the memref
    and the first accessed element. When omitted, the offset is considered to
    be zero. That is, `memref<2, strided<[2], offset: 0>>` and
    `memref<2, strided<[2]>>` are strictly the same type.

    Both offsets and strides may be _dynamic_, that is, unknown at compile time.
    This is represented by using a question mark (`?`) instead of the value in
    the textual form of the IR.

    The strided layout converts into the following canonical one-dimensional
    affine form through explicit linearization:

    ```mlir
    affine_map<(d0, ... dN)[offset, stride0, ... strideN] ->
                (offset + d0 * stride0 + ... dN * strideN)>
    ```

    Therefore, it is never subject to the implicit row-major layout
    interpretation.

    ##### Codegen of Unranked Memref

    Using unranked memref in codegen besides the case mentioned above is highly
    discouraged. Codegen is concerned with generating loop nests and specialized
    instructions for high-performance, unranked memref is concerned with hiding
    the rank and thus, the number of enclosing loops required to iterate over
    the data. However, if there is a need to code-gen unranked memref, one
    possible path is to cast into a static ranked type based on the dynamic
    rank. Another possible path is to emit a single while loop conditioned on a
    linear index and perform delinearization of the linear index to a dynamic
    array containing the (unranked) indices. While this is possible, it is
    expected to not be a good idea to perform this during codegen as the cost
    of the translations is expected to be prohibitive and optimizations at this
    level are not expected to be worthwhile. If expressiveness is the main
    concern, irrespective of performance, passing unranked memrefs to an
    external C++ library and implementing rank-agnostic logic there is expected
    to be significantly simpler.

    Unranked memrefs may provide expressiveness gains in the future and help
    bridge the gap with unranked tensors. Unranked memrefs will not be expected
    to be exposed to codegen but one may query the rank of an unranked memref
    (a special op will be needed for this purpose) and perform a switch and cast
    to a ranked memref as a prerequisite to codegen.

    Example:

    ```mlir
    // With static ranks, we need a function for each possible argument type
    %A = alloc() : memref<16x32xf32>
    %B = alloc() : memref<16x32x64xf32>
    call @helper_2D(%A) : (memref<16x32xf32>)->()
    call @helper_3D(%B) : (memref<16x32x64xf32>)->()

    // With unknown rank, the functions can be unified under one unranked type
    %A = alloc() : memref<16x32xf32>
    %B = alloc() : memref<16x32x64xf32>
    // Remove rank info
    %A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
    %B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
    // call same function with dynamic ranks
    call @helper(%A_u) : (memref<*xf32>)->()
    call @helper(%B_u) : (memref<*xf32>)->()
    ```

    The core syntax and representation of a layout specification is a
    [semi-affine map](Affine.md/#semi-affine-maps). Additionally,
    syntactic sugar is supported to make certain layout specifications more
    intuitive to read. For the moment, a `memref` supports parsing a strided
    form which is converted to a semi-affine map automatically.

    The memory space of a memref is specified by a target-specific attribute.
    It might be an integer value, string, dictionary or custom dialect attribute.
    The empty memory space (attribute is None) is target specific.

    The notionally dynamic value of a memref value includes the address of the
    buffer allocated, as well as the symbols referred to by the shape, layout
    map, and index maps.

    Examples of memref static type

    ```mlir
    // Identity index/layout map
    #identity = affine_map<(d0, d1) -> (d0, d1)>

    // Column major layout.
    #col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

    // A 2-d tiled layout with tiles of size 128 x 256.
    #tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>

    // A tiled data layout with non-constant tile sizes.
    #tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
                                 d0 mod s0, d1 mod s1)>

    // A layout that yields a padding on two at either end of the minor dimension.
    #padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>


    // The dimension list "16x32" defines the following 2D index space:
    //
    //   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
    //
    memref<16x32xf32, #identity>

    // The dimension list "16x4x?" defines the following 3D index space:
    //
    //   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
    //
    // where N is a symbol which represents the runtime value of the size of
    // the third dimension.
    //
    // %N here binds to the size of the third dimension.
    %A = alloc(%N) : memref<16x4x?xf32, #col_major>

    // A 2-d dynamic shaped memref that also has a dynamically sized tiled
    // layout. The memref index space is of size %M x %N, while %B1 and %B2
    // bind to the symbols s0, s1 respectively of the layout map #tiled_dynamic.
    // Data tiles of size %B1 x %B2 in the logical space will be stored
    // contiguously in memory. The allocation size will be
    // (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 f32 elements.
    %T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

    // A memref that has a two-element padding at either end. The allocation
    // size will fit 16 * 64 float elements of data.
    %P = alloc() : memref<16x64xf32, #padded>

    // Affine map with symbol 's0' used as offset for the first dimension.
    #imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
    // Allocate memref and bind the following symbols:
    // '%n' is bound to the dynamic second dimension of the memref type.
    // '%o' is bound to the symbol 's0' in the affine map of the memref type.
    %n = ...
    %o = ...
    %A = alloc (%n)[%o] : <16x?xf32, #imapS>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "MemRefLayoutAttrInterface":$layout,
    "Attribute":$memorySpace
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType,
      CArg<"MemRefLayoutAttrInterface", "{}">:$layout,
      CArg<"Attribute", "{}">:$memorySpace)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType,
      CArg<"AffineMap">:$map,
      CArg<"Attribute", "{}">:$memorySpace)>,
    /// [deprecated] `Attribute`-based form should be used instead.
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType,
      "AffineMap":$map,
      "unsigned":$memorySpaceInd)>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<MemRefType>::clone;
    using ShapedType::Trait<MemRefType>::getElementTypeBitWidth;
    using ShapedType::Trait<MemRefType>::getRank;
    using ShapedType::Trait<MemRefType>::getNumElements;
    using ShapedType::Trait<MemRefType>::isDynamicDim;
    using ShapedType::Trait<MemRefType>::hasStaticShape;
    using ShapedType::Trait<MemRefType>::getNumDynamicDims;
    using ShapedType::Trait<MemRefType>::getDimSize;
    using ShapedType::Trait<MemRefType>::getDynamicDimIndex;

    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// [deprecated] Returns the memory space in old raw integer representation.
    /// New `Attribute getMemorySpace()` method should be used instead.
    unsigned getMemorySpaceAsInt() const;

  }];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
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
    "StringAttr":$dialectNamespace,
    StringRefParameter<"">:$typeData
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "StringAttr":$dialectNamespace, CArg<"StringRef", "{}">:$typeData
    ), [{
      return $_get(dialectNamespace.getContext(), dialectNamespace, typeData);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

def Builtin_RankedTensor : Builtin_Type<"RankedTensor", [
    ShapedTypeInterface
  ], "TensorType"> {
  let summary = "Multi-dimensional array with a fixed number of dimensions";
  let description = [{
    Syntax:

    ```
    tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    dimension-list ::= (dimension `x`)*
    dimension ::= `?` | decimal-literal
    encoding ::= attribute-value
    ```

    Values with tensor type represents aggregate N-dimensional data values, and
    have a known element type and a fixed rank with a list of dimensions. Each
    dimension may be a static non-negative decimal constant or be dynamically
    determined (indicated by `?`).

    The runtime representation of the MLIR tensor type is intentionally
    abstracted - you cannot control layout or get a pointer to the data. For
    low level buffer access, MLIR has a [`memref` type](#memref-type). This
    abstracted runtime representation holds both the tensor data values as well
    as information about the (potentially dynamic) shape of the tensor. The
    [`dim` operation](MemRef.md/#memrefdim-mlirmemrefdimop) returns the size of a
    dimension from a value of tensor type.

    The `encoding` attribute provides additional information on the tensor.
    An empty attribute denotes a straightforward tensor without any specific
    structure. But particular properties, like sparsity or other specific
    characteristics of the data of the tensor can be encoded through this
    attribute. The semantics are defined by a type and attribute interface
    and must be respected by all passes that operate on tensor types.
    TODO: provide this interface, and document it further.

    Note: hexadecimal integer literals are not allowed in tensor type
    declarations to avoid confusion between `0xf32` and `0 x f32`. Zero sizes
    are allowed in tensors and treated as other sizes, e.g.,
    `tensor<0 x 1 x i32>` and `tensor<1 x 0 x i32>` are different types. Since
    zero sizes are not allowed in some other types, such tensors should be
    optimized away before lowering tensors to vectors.

    Examples:

    ```mlir
    // Known rank but unknown dimensions.
    tensor<? x ? x ? x ? x f32>

    // Partially known dimensions.
    tensor<? x ? x 13 x ? x f32>

    // Full static shape.
    tensor<17 x 4 x 13 x 4 x f32>

    // Tensor with rank zero. Represents a scalar.
    tensor<f32>

    // Zero-element dimensions are allowed.
    tensor<0 x 42 x f32>

    // Zero-element tensor of f32 type (hexadecimal literals not allowed here).
    tensor<0xf32>

    // Tensor with an encoding attribute (where #ENCODING is a named alias).
    tensor<?x?xf64, #ENCODING>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Attribute":$encoding
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      CArg<"Attribute", "{}">:$encoding
    ), [{
      return $_get(elementType.getContext(), shape, elementType, encoding);
    }]>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<RankedTensorType>::clone;
    using ShapedType::Trait<RankedTensorType>::getElementTypeBitWidth;
    using ShapedType::Trait<RankedTensorType>::getRank;
    using ShapedType::Trait<RankedTensorType>::getNumElements;
    using ShapedType::Trait<RankedTensorType>::isDynamicDim;
    using ShapedType::Trait<RankedTensorType>::hasStaticShape;
    using ShapedType::Trait<RankedTensorType>::getNumDynamicDims;
    using ShapedType::Trait<RankedTensorType>::getDimSize;
    using ShapedType::Trait<RankedTensorType>::getDynamicDimIndex;

    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;
  }];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
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
    ([rationale](../../Rationale/Rationale/#tuple-types)).

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
      return $_get($_ctxt, elementTypes);
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, TypeRange());
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

//===----------------------------------------------------------------------===//
// UnrankedMemRefType
//===----------------------------------------------------------------------===//

def Builtin_UnrankedMemRef : Builtin_Type<"UnrankedMemRef", [
    ShapedTypeInterface
  ], "BaseMemRefType"> {
  let summary = "Shaped reference, with unknown rank, to a region of memory";
  let description = [{
    Syntax:

    ```
    unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
    memory-space ::= attribute-value
    ```

    A `memref` type with an unknown rank (e.g. `memref<*xf32>`). The purpose of
    unranked memrefs is to allow external library functions to receive memref
    arguments of any rank without versioning the functions based on the rank.
    Other uses of this type are disallowed or will have undefined behavior.

    See [MemRefType](#builtin_memref-memreftype) for more information on
    memref types.

    Examples:

    ```mlir
    memref<*f32>

    // An unranked memref with a memory space of 10.
    memref<*f32, 10>
    ```
  }];
  let parameters = (ins "Type":$elementType, "Attribute":$memorySpace);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "Attribute":$memorySpace), [{
      // Drop default memory space value and replace it with empty attribute.
      Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
      return $_get(elementType.getContext(), elementType, nonDefaultMemorySpace);
    }]>,
    /// [deprecated] `Attribute`-based form should be used instead.
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "unsigned":$memorySpace), [{
      // Convert deprecated integer-like memory space to Attribute.
      Attribute memorySpaceAttr =
          wrapIntegerMemorySpace(memorySpace, elementType.getContext());
      return UnrankedMemRefType::get(elementType, memorySpaceAttr);
    }]>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<UnrankedMemRefType>::clone;
    using ShapedType::Trait<UnrankedMemRefType>::getElementTypeBitWidth;
    using ShapedType::Trait<UnrankedMemRefType>::getRank;
    using ShapedType::Trait<UnrankedMemRefType>::getNumElements;
    using ShapedType::Trait<UnrankedMemRefType>::isDynamicDim;
    using ShapedType::Trait<UnrankedMemRefType>::hasStaticShape;
    using ShapedType::Trait<UnrankedMemRefType>::getNumDynamicDims;
    using ShapedType::Trait<UnrankedMemRefType>::getDimSize;
    using ShapedType::Trait<UnrankedMemRefType>::getDynamicDimIndex;

    ArrayRef<int64_t> getShape() const { return std::nullopt; }

    /// [deprecated] Returns the memory space in old raw integer representation.
    /// New `Attribute getMemorySpace()` method should be used instead.
    unsigned getMemorySpaceAsInt() const;
  }];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// UnrankedTensorType
//===----------------------------------------------------------------------===//

def Builtin_UnrankedTensor : Builtin_Type<"UnrankedTensor", [
    ShapedTypeInterface
  ], "TensorType"> {
  let summary = "Multi-dimensional array with unknown dimensions";
  let description = [{
    Syntax:

    ```
    tensor-type ::= `tensor` `<` `*` `x` type `>`
    ```

    An unranked tensor is a type of tensor in which the set of dimensions have
    unknown rank. See [RankedTensorType](#builtin_rankedtensor-rankedtensortype)
    for more information on tensor types.

    Examples:

    ```mlir
    tensor<*xf32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<UnrankedTensorType>::clone;
    using ShapedType::Trait<UnrankedTensorType>::getElementTypeBitWidth;
    using ShapedType::Trait<UnrankedTensorType>::getRank;
    using ShapedType::Trait<UnrankedTensorType>::getNumElements;
    using ShapedType::Trait<UnrankedTensorType>::isDynamicDim;
    using ShapedType::Trait<UnrankedTensorType>::hasStaticShape;
    using ShapedType::Trait<UnrankedTensorType>::getNumDynamicDims;
    using ShapedType::Trait<UnrankedTensorType>::getDimSize;
    using ShapedType::Trait<UnrankedTensorType>::getDynamicDimIndex;

    ArrayRef<int64_t> getShape() const { return std::nullopt; }
  }];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

def Builtin_Vector : Builtin_Type<"Vector", [ShapedTypeInterface], "Type"> {
  let summary = "Multi-dimensional SIMD vector type";
  let description = [{
    Syntax:

    ```
    vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
    vector-element-type ::= float-type | integer-type | index-type
    vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
    static-dim-list ::= decimal-literal (`x` decimal-literal)*
    ```

    The vector type represents a SIMD style vector used by target-specific
    operation sets like AVX or SVE. While the most common use is for 1D
    vectors (e.g. vector<16 x f32>) we also support multidimensional registers
    on targets that support them (like TPUs). The dimensions of a vector type
    can be fixed-length, scalable, or a combination of the two. The scalable
    dimensions in a vector are indicated between square brackets ([ ]), and
    all fixed-length dimensions, if present, must precede the set of scalable
    dimensions. That is, a `vector<2x[4]xf32>` is valid, but `vector<[4]x2xf32>`
    is not.

    Vector shapes must be positive decimal integers. 0D vectors are allowed by
    omitting the dimension: `vector<f32>`.

    Note: hexadecimal integer literals are not allowed in vector type
    declarations, `vector<0x42xi32>` is invalid because it is interpreted as a
    2D vector with shape `(0, 42)` and zero shapes are not allowed.

    Examples:

    ```mlir
    // A 2D fixed-length vector of 3x42 i32 elements.
    vector<3x42xi32>

    // A 1D scalable-length vector that contains a multiple of 4 f32 elements.
    vector<[4]xf32>

    // A 2D scalable-length vector that contains a multiple of 2x8 i8 elements.
    vector<[2x8]xf32>

    // A 2D mixed fixed/scalable vector that contains 4 scalable vectors of 4 f32 elements.
    vector<4x[4]xf32>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "unsigned":$numScalableDims
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType,
      CArg<"unsigned", "0">:$numScalableDims
    ), [{
      return $_get(elementType.getContext(), shape, elementType,
                   numScalableDims);
    }]>
  ];
  let extraClassDeclaration = [{
    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// Returns true if the given type can be used as an element of a vector
    /// type. In particular, vectors can consist of integer, index, or float
    /// primitives.
    static bool isValidElementType(Type t) {
      return t.isa<IntegerType, IndexType, FloatType>();
    }

    /// Returns true if the vector contains scalable dimensions.
    bool isScalable() const {
      return getNumScalableDims() > 0;
    }

    /// Get or create a new VectorType with the same shape as `this` and an
    /// element type of bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    VectorType scaleElementBitwidth(unsigned scale);

    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this vector type with the given shape and element type. If the
    /// provided shape is `None`, the current shape of the type is used.
    VectorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif // BUILTIN_TYPES
