# Conversion to the LLVM Dialect

Conversion from several dialects that rely on
[built-in types](LangRef.md#builtin-types) to the
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
-   `bf16` converts to `!llvm.bfloat`
-   `f16` converts to `!llvm.half`
-   `f32` converts to `!llvm.float`
-   `f64` converts to `!llvm.double`

### Index Type

Index type is converted to an LLVM dialect integer type with bitwidth equal to
the bitwidth of the pointer size as specified by the
[data layout](Dialects/LLVM.md#data-layout-and-triple) of the closest module.
For example, on x86-64 CPUs it converts to `!llvm.i64`. This behavior can be
overridden by the type converter configuration, which is often exposed as a pass
option by conversion passes.

### Vector Types

LLVM IR only supports *one-dimensional* vectors, unlike MLIR where vectors can
be multi-dimensional. Vector types cannot be nested in either IR. In the
one-dimensional case, MLIR vectors are converted to LLVM IR vectors of the same
size with element type converted using these conversion rules. In the
n-dimensional case, MLIR vectors are converted to (n-1)-dimensional array types
of one-dimensional vectors.

For example, `vector<4 x f32>` converts to `!llvm.vec<4 x float>` and `vector<4
x 8 x 16 x f32>` converts to `!llvm.array<4 x array<8 x vec<16 x float>>>`.

### Ranked Memref Types

Memref types in MLIR have both static and dynamic information associated with
them. In the general case, the dynamic information describes dynamic sizes in
the logical indexing space and any symbols bound to the memref. This dynamic
information must be present at runtime in the LLVM dialect equivalent type.

In practice, the conversion supports two conventions:

-   the default convention for memrefs in the
    **[strided form](LangRef.md#strided-memref)**;
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
memref<f32> -> !llvm.struct<(ptr<float> , ptr<float>, i64)>
memref<1 x f32> -> !llvm.struct<(ptr<float>, ptr<float>, i64,
                                 array<1 x 64>, array<1 x i64>)>
memref<? x f32> -> !llvm.struct<(ptr<float>, ptr<float>, i64
                                 array<1 x 64>, array<1 x i64>)>
memref<10x42x42x43x123 x f32> -> !llvm.struct<(ptr<float>, ptr<float>, i64
                                               array<5 x 64>, array<5 x i64>)>
memref<10x?x42x?x123 x f32> -> !llvm.struct<(ptr<float>, ptr<float>, i64
                                             array<5 x 64>, array<5 x i64>)>

// Memref types can have vectors as element types
memref<1x? x vector<4xf32>> -> !llvm.struct<(ptr<vec<4 x float>>,
                                             ptr<vec<4 x float>>, i64,
                                             array<1 x i64>, array<1 x i64>)>
```

#### Bare Pointer Convention

Ranked memrefs with static shape and default layout can be converted into an
LLVM dialect pointer to their element type. Only the default alignment is
supported in such cases, e.g. the `alloc` operation cannot have an alignemnt
attribute.

Examples:

```mlir
memref<f32> -> !llvm.ptr<float>
memref<10x42 x f32> -> !llvm.ptr<float>

// Memrefs with vector types are also supported.
memref<10x42 x vector<4xf32>> -> !llvm.ptr<vec<4 x float>>
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
!llvm.func<i64 (i32, float)>

// Binary function with two results:
(i32, f32) -> (i64, f64)
// has its result aggregated into a structure type.
!llvm.func<struct<(i64, double)> (i32, float)>
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
descriptor. This is intended for some comaptibility with C ABI, in which
structure types would need to be passed by-pointer leading to the need for
allocations and related issues, as well as for aliasing annotations, which are
currently attached to pointer in function arguments. Having scalar components
means that each size and stride is passed as an invidivual value.

When used as function results, memrefs are converted as usual, i.e. each memref
is converted to a descriptor struct (default convention) or to a pointer (bare
pointer convention).

Examples:

```mlir
// A memref descriptor appearing as function argument:
(memref<f32>) -> ()
// gets converted into a list of individual scalar components of a descriptor.
!llvm.func<void (ptr<float>, ptr<float>, i64)>

// The list of arguments is linearized and one can freely mix memref and other
// types in this list:
(memref<f32>, f32) -> ()
// which gets converted into a flat list.
!llvm.func<void (ptr<float>, ptr<float>, i64, float)>

// For nD ranked memref descriptors:
(memref<?x?xf32>) -> ()
// the converted signature will contain 2n+1 `index`-typed integer arguments,
// offset, n sizes and n strides, per memref argument type.
!llvm.func<void (ptr<float>, ptr<float>, i64, i64, i64, i64, i64)>

// Same rules apply to unranked descriptors:
(memref<*xf32>) -> ()
// which get converted into their components.
!llvm.func<void (i64, ptr<i8>)>

// However, returning a memref from a function is not affected:
() -> (memref<?xf32>)
// gets converted to a function returning a descriptor structure.
!llvm.func<struct<(ptr<float>, ptr<float>, i64, array<1xi64>, array<1xi64>)> ()>

// If multiple memref-typed results are returned:
() -> (memref<f32>, memref<f64>)
// their descriptor structures are additionally packed into another structure,
// potentially with other non-memref typed results.
!llvm.func<struct<(struct<(ptr<float>, ptr<float>, i64)>,
                   struct<(ptr<double>, ptr<double>, i64)>)> ()>
```

## Calling Convention for Standard Calls

<!-- TODO: This should be moved to a separate file, and the remaining file
     renamed decouple the description of built-in type conversion from standard
     dialect ops conversion. -->

### Result Packing

In case of multi-result functions, the returned values are inserted into a
structure-typed value before being returned and extracted from it at the call
site. This transformation is a part of the conversion and is transparent to the
defines and uses of the values being returned.

Example:

```mlir
func @foo(%arg0: i32, %arg1: i64) -> (i32, i64) {
  return %arg0, %arg1 : i32, i64
}
func @bar() {
  %0 = constant 42 : i32
  %1 = constant 17 : i64
  %2:2 = call @foo(%0, %1) : (i32, i64) -> (i32, i64)
  "use_i32"(%2#0) : (i32) -> ()
  "use_i64"(%2#1) : (i64) -> ()
}

// is transformed into

func @foo(%arg0: !llvm.i32, %arg1: !llvm.i64) -> !llvm<"{i32, i64}"> {
  // insert the vales into a structure
  %0 = llvm.mlir.undef :  !llvm<"{i32, i64}">
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{i32, i64}">
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{i32, i64}">

  // return the structure value
  llvm.return %2 : !llvm<"{i32, i64}">
}
func @bar() {
  %0 = llvm.mlir.constant(42 : i32) : !llvm.i32
  %1 = llvm.mlir.constant(17) : !llvm.i64

  // call and extract the values from the structure
  %2 = llvm.call @bar(%0, %1) : (%arg0: !llvm.i32, %arg1: !llvm.i32) -> !llvm<"{i32, i64}">
  %3 = llvm.extractvalue %2[0] : !llvm<"{i32, i64}">
  %4 = llvm.extractvalue %2[1] : !llvm<"{i32, i64}">

  // use as before
  "use_i32"(%3) : (!llvm.i32) -> ()
  "use_i64"(%4) : (!llvm.i64) -> ()
}
```

### Calling Convention for Ranked `memref`

Function _arguments_ of `memref` type, ranked or unranked, are _expanded_ into a
list of arguments of non-aggregate types that the memref descriptor defined
above comprises. That is, the outer struct type and the inner array types are
replaced with individual arguments.

This convention is implemented in the conversion of `std.func` and `std.call` to
the LLVM dialect, with the former unpacking the descriptor into a set of
individual values and the latter packing those values back into a descriptor so
as to make it transparently usable by other operations. Conversions from other
dialects should take this convention into account.

This specific convention is motivated by the necessity to specify alignment and
aliasing attributes on the raw pointers underpinning the memref.

Examples:

```mlir
func @foo(%arg0: memref<?xf32>) -> () {
  "use"(%arg0) : (memref<?xf32>) -> ()
  return
}

// Gets converted to the following.

llvm.func @foo(%arg0: !llvm<"float*">,   // Allocated pointer.
               %arg1: !llvm<"float*">,   // Aligned pointer.
               %arg2: !llvm.i64,         // Offset.
               %arg3: !llvm.i64,         // Size in dim 0.
               %arg4: !llvm.i64) {       // Stride in dim 0.
  // Populate memref descriptor structure.
  %0 = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">

  // Descriptor is now usable as a single value.
  "use"(%5) : (!llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">) -> ()
  llvm.return
}
```

```mlir
func @bar() {
  %0 = "get"() : () -> (memref<?xf32>)
  call @foo(%0) : (memref<?xf32>) -> ()
  return
}

// Gets converted to the following.

llvm.func @bar() {
  %0 = "get"() : () -> !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">

  // Unpack the memref descriptor.
  %1 = llvm.extractvalue %0[0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %2 = llvm.extractvalue %0[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %3 = llvm.extractvalue %0[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %4 = llvm.extractvalue %0[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  %5 = llvm.extractvalue %0[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">

  // Pass individual values to the callee.
  llvm.call @foo(%1, %2, %3, %4, %5) : (!llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
  llvm.return
}

```

### Calling Convention for Unranked `memref`

For unranked memrefs, the list of function arguments always contains two
elements, same as the unranked memref descriptor: an integer rank, and a
type-erased (`!llvm<"i8*">`) pointer to the ranked memref descriptor. Note that
while the _calling convention_ does not require stack allocation, _casting_ to
unranked memref does since one cannot take an address of an SSA value containing
the ranked memref. The caller is in charge of ensuring the thread safety and
eventually removing unnecessary stack allocations in cast operations.

Example

```mlir
llvm.func @foo(%arg0: memref<*xf32>) -> () {
  "use"(%arg0) : (memref<*xf32>) -> ()
  return
}

// Gets converted to the following.

llvm.func @foo(%arg0: !llvm.i64       // Rank.
               %arg1: !llvm<"i8*">) { // Type-erased pointer to descriptor.
  // Pack the unranked memref descriptor.
  %0 = llvm.mlir.undef : !llvm<"{ i64, i8* }">
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{ i64, i8* }">
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{ i64, i8* }">

  "use"(%2) : (!llvm<"{ i64, i8* }">) -> ()
  llvm.return
}
```

```mlir
llvm.func @bar() {
  %0 = "get"() : () -> (memref<*xf32>)
  call @foo(%0): (memref<*xf32>) -> ()
  return
}

// Gets converted to the following.

llvm.func @bar() {
  %0 = "get"() : () -> (!llvm<"{ i64, i8* }">)

  // Unpack the memref descriptor.
  %1 = llvm.extractvalue %0[0] : !llvm<"{ i64, i8* }">
  %2 = llvm.extractvalue %0[1] : !llvm<"{ i64, i8* }">

  // Pass individual values to the callee.
  llvm.call @foo(%1, %2) : (!llvm.i64, !llvm<"i8*">)
  llvm.return
}
```

**Lifetime.** The second element of the unranked memref descriptor points to
some memory in which the ranked memref descriptor is stored. By convention, this
memory is allocated on stack and has the lifetime of the function. (*Note:* due
to function-length lifetime, creation of multiple unranked memref descriptors,
e.g., in a loop, may lead to stack overflows.) If an unranked descriptor has to
be returned from a function, the ranked descriptor it points to is copied into
dynamically allocated memory, and the pointer in the unranked descriptor is
updated accordingly. The allocation happens immediately before returning. It is
the responsibility of the caller to free the dynamically allocated memory. The
default conversion of `std.call` and `std.call_indirect` copies the ranked
descriptor to newly allocated memory on the caller's stack. Thus, the convention
of the ranked memref descriptor pointed to by an unranked memref descriptor
being stored on stack is respected.

*This convention may or may not apply if the conversion of MemRef types is
overridden by the user.*

### C-compatible wrapper emission

In practical cases, it may be desirable to have externally-facing functions with
a single attribute corresponding to a MemRef argument. When interfacing with
LLVM IR produced from C, the code needs to respect the corresponding calling
convention. The conversion to the LLVM dialect provides an option to generate
wrapper functions that take memref descriptors as pointers-to-struct compatible
with data types produced by Clang when compiling C sources. The generation of
such wrapper functions can additionally be controlled at a function granularity
by setting the `llvm.emit_c_interface` unit attribute.

More specifically, a memref argument is converted into a pointer-to-struct
argument of type `{T*, T*, i64, i64[N], i64[N]}*` in the wrapper function, where
`T` is the converted element type and `N` is the memref rank. This type is
compatible with that produced by Clang for the following C++ structure template
instantiations or their equivalents in C.

```cpp
template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};
```

If enabled, the option will do the following. For _external_ functions declared
in the MLIR module.

1. Declare a new function `_mlir_ciface_<original name>` where memref arguments
   are converted to pointer-to-struct and the remaining arguments are converted
   as usual.
1. Add a body to the original function (making it non-external) that
   1. allocates a memref descriptor,
   1. populates it, and
   1. passes the pointer to it into the newly declared interface function, then
   1. collects the result of the call and returns it to the caller.

For (non-external) functions defined in the MLIR module.

1. Define a new function `_mlir_ciface_<original name>` where memref arguments
   are converted to pointer-to-struct and the remaining arguments are converted
   as usual.
1. Populate the body of the newly defined function with IR that
   1. loads descriptors from pointers;
   1. unpacks descriptor into individual non-aggregate values;
   1. passes these values into the original function;
   1. collects the result of the call and returns it to the caller.

Examples:

```mlir

func @qux(%arg0: memref<?x?xf32>)

// Gets converted into the following.

// Function with unpacked arguments.
llvm.func @qux(%arg0: !llvm<"float*">, %arg1: !llvm<"float*">, %arg2: !llvm.i64,
               %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64,
               %arg6: !llvm.i64) {
  // Populate memref descriptor (as per calling convention).
  %0 = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">

  // Store the descriptor in a stack-allocated space.
  %8 = llvm.mlir.constant(1 : index) : !llvm.i64
  %9 = llvm.alloca %8 x !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
                 : (!llvm.i64) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
  llvm.store %7, %9 : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">

  // Call the interface function.
  llvm.call @_mlir_ciface_qux(%9) : (!llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) -> ()

  // The stored descriptor will be freed on return.
  llvm.return
}

// Interface function.
llvm.func @_mlir_ciface_qux(!llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">)
```

```mlir
func @foo(%arg0: memref<?x?xf32>) {
  return
}

// Gets converted into the following.

// Function with unpacked arguments.
llvm.func @foo(%arg0: !llvm<"float*">, %arg1: !llvm<"float*">, %arg2: !llvm.i64,
               %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64,
               %arg6: !llvm.i64) {
  llvm.return
}

// Interface function callable from C.
llvm.func @_mlir_ciface_foo(%arg0: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) {
  // Load the descriptor.
  %0 = llvm.load %arg0 : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">

  // Unpack the descriptor as per calling convention.
  %1 = llvm.extractvalue %0[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %2 = llvm.extractvalue %0[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %3 = llvm.extractvalue %0[2] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %4 = llvm.extractvalue %0[3, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %5 = llvm.extractvalue %0[3, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %6 = llvm.extractvalue %0[4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %7 = llvm.extractvalue %0[4, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  llvm.call @foo(%1, %2, %3, %4, %5, %6, %7)
    : (!llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64,
       !llvm.i64, !llvm.i64) -> ()
  llvm.return
}
```

Rationale: Introducing auxiliary functions for C-compatible interfaces is
preferred to modifying the calling convention since it will minimize the effect
of C compatibility on intra-module calls or calls between MLIR-generated
functions. In particular, when calling external functions from an MLIR module in
a (parallel) loop, the fact of storing a memref descriptor on stack can lead to
stack exhaustion and/or concurrent access to the same address. Auxiliary
interface function serves as an allocation scope in this case. Furthermore, when
targeting accelerators with separate memory spaces such as GPUs, stack-allocated
descriptors passed by pointer would have to be transferred to the device memory,
which introduces significant overhead. In such situations, auxiliary interface
functions are executed on host and only pass the values through device function
invocation mechanism.

## Repeated Successor Removal

Since the goal of the LLVM IR dialect is to reflect LLVM IR in MLIR, the dialect
and the conversion procedure must account for the differences between block
arguments and LLVM IR PHI nodes. In particular, LLVM IR disallows PHI nodes with
different values coming from the same source. Therefore, the LLVM IR dialect
disallows operations that have identical successors accepting arguments, which
would lead to invalid PHI nodes. The conversion process resolves the potential
PHI source ambiguity by injecting dummy blocks if the same block is used more
than once as a successor in an instruction. These dummy blocks branch
unconditionally to the original successors, pass them the original operands
(available in the dummy block because it is dominated by the original block) and
are used instead of them in the original terminator operation.

Example:

```mlir
  cond_br %0, ^bb1(%1 : i32), ^bb1(%2 : i32)
^bb1(%3 : i32)
  "use"(%3) : (i32) -> ()
```

leads to a new basic block being inserted,

```mlir
  cond_br %0, ^bb1(%1 : i32), ^dummy
^bb1(%3 : i32):
  "use"(%3) : (i32) -> ()
^dummy:
  br ^bb1(%4 : i32)
```

before the conversion to the LLVM IR dialect:

```mlir
  llvm.cond_br  %0, ^bb1(%1 : !llvm.i32), ^dummy
^bb1(%3 : !llvm<"i32">):
  "use"(%3) : (!llvm.i32) -> ()
^dummy:
  llvm.br ^bb1(%2 : !llvm.i32)
```

## Default Memref Model

### Memref Descriptor

Within a converted function, a `memref`-typed value is represented by a memref
_descriptor_, the type of which is the structure type obtained by converting
from the memref type. This descriptor holds all the necessary information to
produce an address of a specific element. In particular, it holds dynamic values
for static sizes, and they are expected to match at all times.

It is created by the allocation operation and is updated by the conversion
operations that may change static dimensions into dynamic dimensions and vice versa.

**Note**: LLVM IR conversion does not support `memref`s with layouts that are
not amenable to the strided form.

### Index Linearization

Accesses to a memref element are transformed into an access to an element of the
buffer pointed to by the descriptor. The position of the element in the buffer
is calculated by linearizing memref indices in row-major order (lexically first
index is the slowest varying, similar to C, but accounting for strides). The
computation of the linear address is emitted as arithmetic operation in the LLVM
IR dialect. Strides are extracted from the memref descriptor.

Accesses to zero-dimensional memref (that are interpreted as pointers to the
elemental type) are directly converted into `llvm.load` or `llvm.store` without
any pointer manipulations.

Examples:

An access to a zero-dimensional memref is converted into a plain load:

```mlir
// before
%0 = load %m[] : memref<f32>

// after
%0 = llvm.load %m : !llvm<"float*">
```

An access to a memref with indices:

```mlir
%0 = load %m[1,2,3,4] : memref<10x?x13x?xf32>
```

is transformed into the equivalent of the following code:

```mlir
// Compute the linearized index from strides. Each block below extracts one
// stride from the descriptor, multiplies it with the index and accumulates
// the total offset.
%stride1 = llvm.extractvalue[4, 0] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx1 = llvm.mlir.constant(1 : index) !llvm.i64
%addr1 = muli %stride1, %idx1 : !llvm.i64

%stride2 = llvm.extractvalue[4, 1] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx2 = llvm.mlir.constant(2 : index) !llvm.i64
%addr2 = muli %stride2, %idx2 : !llvm.i64
%addr3 = addi %addr1, %addr2 : !llvm.i64

%stride3 = llvm.extractvalue[4, 2] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx3 = llvm.mlir.constant(3 : index) !llvm.i64
%addr4 = muli %stride3, %idx3 : !llvm.i64
%addr5 = addi %addr3, %addr4 : !llvm.i64

%stride4 = llvm.extractvalue[4, 3] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx4 = llvm.mlir.constant(4 : index) !llvm.i64
%addr6 = muli %stride4, %idx4 : !llvm.i64
%addr7 = addi %addr5, %addr6 : !llvm.i64

// Add the linear offset to the address.
%offset = llvm.extractvalue[2] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%addr8 = addi %addr7, %offset : !llvm.i64

// Obtain the aligned pointer.
%aligned = llvm.extractvalue[1] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">

// Get the address of the data pointer.
%ptr = llvm.getelementptr %aligned[%addr8]
    : !llvm<"{float*, float*, i64, i64[4], i64[4]}"> -> !llvm<"float*">

// Perform the actual load.
%0 = llvm.load %ptr : !llvm<"float*">
```

For stores, the address computation code is identical and only the actual store
operation is different.

Note: the conversion does not perform any sort of common subexpression
elimination when emitting memref accesses.
