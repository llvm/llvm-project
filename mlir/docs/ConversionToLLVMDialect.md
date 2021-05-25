# Conversion to the LLVM Dialect

Conversion from several dialects that rely on
[built-in types](LangRef.md/#builtin-types) to the
[LLVM Dialect](Dialects/LLVM.md) is expected to be performed through the
[Dialect Conversion](DialectConversion.md) infrastructure.

The conversion of types and that of the overall module structure is described in
this document. Individual conversion passes provide a set of conversion patterns
for ops in different dialects, such as `-convert-std-to-llvm` for ops in the
[Standard dialect](Dialects/Standard.md) and `-convert-vector-to-llvm` in the
[Vector dialect](Dialects/Vector.md). *Note that some conversions subsume the
others.*

We use the terminology defined by the
[LLVM Dialect description](Dialects/LLVM.md) throughout this document.

[TOC]

## Type Conversion

### Scalar Types

Scalar types are converted to their LLVM counterparts if they exist. The
following conversions are currently implemented:

-   `i*` converts to `!llvm.i*`
-   `bf16` converts to `bf16`
-   `f16` converts to `f16`
-   `f32` converts to `f32`
-   `f64` converts to `f64`
-   `f80` converts to `f80`
-   `f128` converts to `f128`

### Index Type

Index type is converted to an LLVM dialect integer type with bitwidth equal to
the bitwidth of the pointer size as specified by the
[data layout](Dialects/LLVM.md/#data-layout-and-triple) of the closest module.
For example, on x86-64 CPUs it converts to `i64`. This behavior can be
overridden by the type converter configuration, which is often exposed as a pass
option by conversion passes.

### Vector Types

LLVM IR only supports *one-dimensional* vectors, unlike MLIR where vectors can
be multi-dimensional. Vector types cannot be nested in either IR. In the
one-dimensional case, MLIR vectors are converted to LLVM IR vectors of the same
size with element type converted using these conversion rules. In the
n-dimensional case, MLIR vectors are converted to (n-1)-dimensional array types
of one-dimensional vectors.

For example, `vector<4xf32>` converts to `vector<4xf32>` and `vector<4 x 8 x 16
x f32>` converts to `!llvm.array<4 x array<8 x vec<16 x f32>>>`.

### Ranked Memref Types

Memref types in MLIR have both static and dynamic information associated with
them. In the general case, the dynamic information describes dynamic sizes in
the logical indexing space and any symbols bound to the memref. This dynamic
information must be present at runtime in the LLVM dialect equivalent type.

In practice, the conversion supports two conventions:

-   the default convention for memrefs in the
    **[strided form](Dialects/Builtin.md/#strided-memref)**;
-   a "bare pointer" conversion for statically-shaped memrefs with default
    layout.

The choice between conventions is specified at type converter construction time
and is often exposed as an option by conversion passes.

Memrefs with arbitrary layouts are not supported. Instead, these layouts can be
factored out of the type and used as part of index computation for operations
that read and write into a memref with the default layout.

#### Default Convention

The dynamic information comprises the buffer pointer as well as sizes and
strides of any dynamically-sized dimensions. Memref types are normalized and
converted to a _descriptor_ that is only dependent on the rank of the memref.
The descriptor contains the following fields in order:

1.  The pointer to the data buffer as allocated, referred to as "allocated
    pointer". This is only useful for deallocating the memref.
2.  The pointer to the properly aligned data pointer that the memref indexes,
    referred to as "aligned pointer".
3.  A lowered converted `index`-type integer containing the distance in number
    of elements between the beginning of the (aligned) buffer and the first
    element to be accessed through the memref, referred to as "offset".
4.  An array containing as many converted `index`-type integers as the rank of
    the memref: the array represents the size, in number of elements, of the
    memref along the given dimension. For constant memref dimensions, the
    corresponding size entry is a constant whose runtime value must match the
    static value.
5.  A second array containing as many converted `index`-type integers as the
    rank of memref: the second array represents the "stride" (in tensor
    abstraction sense), i.e. the number of consecutive elements of the
    underlying buffer one needs to jump over to get to the next logically
    indexed element.

For constant memref dimensions, the corresponding size entry is a constant whose
runtime value matches the static value. This normalization serves as an ABI for
the memref type to interoperate with externally linked functions. In the
particular case of rank `0` memrefs, the size and stride arrays are omitted,
resulting in a struct containing two pointers + offset.

Examples:

```mlir
memref<f32> -> !llvm.struct<(ptr<f32> , ptr<f32>, i64)>
memref<1 x f32> -> !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                 array<1 x 64>, array<1 x i64>)>
memref<? x f32> -> !llvm.struct<(ptr<f32>, ptr<f32>, i64
                                 array<1 x 64>, array<1 x i64>)>
memref<10x42x42x43x123 x f32> -> !llvm.struct<(ptr<f32>, ptr<f32>, i64
                                               array<5 x 64>, array<5 x i64>)>
memref<10x?x42x?x123 x f32> -> !llvm.struct<(ptr<f32>, ptr<f32>, i64
                                             array<5 x 64>, array<5 x i64>)>

// Memref types can have vectors as element types
memref<1x? x vector<4xf32>> -> !llvm.struct<(ptr<vec<4 x f32>>,
                                             ptr<vec<4 x float>>, i64,
                                             array<1 x i64>, array<1 x i64>)>
```

#### Bare Pointer Convention

Ranked memrefs with static shape and default layout can be converted into an
LLVM dialect pointer to their element type. Only the default alignment is
supported in such cases, e.g. the `alloc` operation cannot have an alignment
attribute.

Examples:

```mlir
memref<f32> -> !llvm.ptr<f32>
memref<10x42 x f32> -> !llvm.ptr<f32>

// Memrefs with vector types are also supported.
memref<10x42 x vector<4xf32>> -> !llvm.ptr<vec<4 x f32>>
```

### Unranked Memref types

Unranked memrefs are converted to an unranked descriptor that contains:

1.  a converted `index`-typed integer representing the dynamic rank of the
    memref;
2.  a type-erased pointer (`!llvm.ptr<i8>`) to a ranked memref descriptor with
    the contents listed above.

This descriptor is primarily intended for interfacing with rank-polymorphic
library functions. The pointer to the ranked memref descriptor points to memory
_allocated on stack_ of the function in which it is used.

Note that stack allocations may be emitted at a location where the unranked
memref first appears, e.g., a cast operation, and remain live throughout the
lifetime of the function; this may lead to stack exhaustion if used in a loop.

Examples:

```mlir
// Unranked descriptor.
memref<*xf32> -> !llvm.struct<(i64, ptr<i8>)>
```

Bare pointer convention does not support unranked memrefs.

### Function Types

Function types get converted to LLVM dialect function types. The arguments are
converted individually according to these rules, except for `memref` types in
function arguments and high-order functions, which are described below. The
result types need to accommodate the fact that LLVM functions always have a
return type, which may be an `!llvm.void` type. The converted function always
has a single result type. If the original function type had no results, the
converted function will have one result of the `!llvm.void` type. If the
original function type had one result, the converted function will also have one
result converted using these rules. Otherwise, the result type will be an LLVM
dialect structure type where each element of the structure corresponds to one of
the results of the original function, converted using these rules.

Examples:

```mlir
// Zero-ary function type with no results:
() -> ()
// is converted to a zero-ary function with `void` result.
!llvm.func<void ()>

// Unary function with one result:
(i32) -> (i64)
// has its argument and result type converted, before creating the LLVM dialect
// function type.
!llvm.func<i64 (i32)>

// Binary function with one result:
(i32, f32) -> (i64)
// has its arguments handled separately
!llvm.func<i64 (i32, f32)>

// Binary function with two results:
(i32, f32) -> (i64, f64)
// has its result aggregated into a structure type.
!llvm.func<struct<(i64, f64)> (i32, f32)>
```

#### Functions as Function Arguments or Results

High-order function types, i.e. types of functions that have other functions as
arguments or results, are converted differently to accommodate the fact that
LLVM IR does not allow for function-typed values. Instead, functions are
expected to be passed into and return from other functions _by pointer_.
Therefore, function-typed function arguments are results are converted to
pointer-to-the-function type. The pointee type is converted using these rules.

Examples:

```mlir
// Function-typed arguments or results in higher-order functions:
(() -> ()) -> (() -> ())
// are converted into pointers to functions.
!llvm.func<ptr<func<void ()>> (ptr<func<void ()>>)>

// These rules apply recursively: a function type taking a function that takes
// another function
( ( (i32) -> (i64) ) -> () ) -> ()
// is converted into a function type taking a pointer-to-function that takes
// another point-to-function.
!llvm.func<void (ptr<func<void (ptr<func<i64 (i32)>>)>>)>
```

#### Memrefs as Function Arguments

When used as function arguments, both ranked and unranked memrefs are converted
into a list of arguments that represents each _scalar_ component of their
descriptor. This is intended for some compatibility with C ABI, in which
structure types would need to be passed by-pointer leading to the need for
allocations and related issues, as well as for aliasing annotations, which are
currently attached to pointer in function arguments. Having scalar components
means that each size and stride is passed as an individual value.

When used as function results, memrefs are converted as usual, i.e. each memref
is converted to a descriptor struct (default convention) or to a pointer (bare
pointer convention).

Examples:

```mlir
// A memref descriptor appearing as function argument:
(memref<f32>) -> ()
// gets converted into a list of individual scalar components of a descriptor.
!llvm.func<void (ptr<f32>, ptr<f32>, i64)>

// The list of arguments is linearized and one can freely mix memref and other
// types in this list:
(memref<f32>, f32) -> ()
// which gets converted into a flat list.
!llvm.func<void (ptr<f32>, ptr<f32>, i64, f32)>

// For nD ranked memref descriptors:
(memref<?x?xf32>) -> ()
// the converted signature will contain 2n+1 `index`-typed integer arguments,
// offset, n sizes and n strides, per memref argument type.
!llvm.func<void (ptr<f32>, ptr<f32>, i64, i64, i64, i64, i64)>

// Same rules apply to unranked descriptors:
(memref<*xf32>) -> ()
// which get converted into their components.
!llvm.func<void (i64, ptr<i8>)>

// However, returning a memref from a function is not affected:
() -> (memref<?xf32>)
// gets converted to a function returning a descriptor structure.
!llvm.func<struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<1xi64>)> ()>

// If multiple memref-typed results are returned:
() -> (memref<f32>, memref<f64>)
// their descriptor structures are additionally packed into another structure,
// potentially with other non-memref typed results.
!llvm.func<struct<(struct<(ptr<f32>, ptr<f32>, i64)>,
                   struct<(ptr<double>, ptr<double>, i64)>)> ()>
```
