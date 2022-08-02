# LLVM IR Target

This document describes the mechanisms of producing LLVM IR from MLIR. The
overall flow is two-stage:

1.  **conversion** of the IR to a set of dialects translatable to LLVM IR, for
    example [LLVM Dialect](Dialects/LLVM.md) or one of the hardware-specific
    dialects derived from LLVM IR intrinsics such as [AMX](Dialects/AMX.md),
    [X86Vector](Dialects/X86Vector.md) or [ArmNeon](Dialects/ArmNeon.md);
2.  **translation** of MLIR dialects to LLVM IR.

This flow allows the non-trivial transformation to be performed within MLIR
using MLIR APIs and makes the translation between MLIR and LLVM IR *simple* and
potentially bidirectional. As a corollary, dialect ops translatable to LLVM IR
are expected to closely match the corresponding LLVM IR instructions and
intrinsics. This minimizes the dependency on LLVM IR libraries in MLIR as well
as reduces the churn in case of changes.

Note that many different dialects can be lowered to LLVM but are provided as
different sets of patterns and have different passes available to mlir-opt.
However, this is primarily useful for testing and prototyping, and using the
collection of patterns together is highly recommended. One place this is
important and visible is the ControlFlow dialect's branching operations which
will fail to apply if their types mismatch with the blocks they jump to in the
parent op.

SPIR-V to LLVM dialect conversion has a
[dedicated document](SPIRVToLLVMDialectConversion.md).

[TOC]

## Conversion to the LLVM Dialect

Conversion to the LLVM dialect from other dialects is the first step to produce
LLVM IR. All non-trivial IR modifications are expected to happen at this stage
or before. The conversion is *progressive*: most passes convert one dialect to
the LLVM dialect and keep operations from other dialects intact. For example,
the `-convert-memref-to-llvm` pass will only convert operations from the
`memref` dialect but will not convert operations from other dialects even if
they use or produce `memref`-typed values.

The process relies on the [Dialect Conversion](DialectConversion.md)
infrastructure and, in particular, on the
[materialization](DialectConversion.md#type-conversion) hooks of `TypeConverter`
to support progressive lowering by injecting `unrealized_conversion_cast`
operations between converted and unconverted operations. After multiple partial
conversions to the LLVM dialect are performed, the cast operations that became
noop can be removed by the `-reconcile-unrealized-casts` pass. The latter pass
is not specific to the LLVM dialect and can remove any noop casts.

### Conversion of Built-in Types

Built-in types have a default conversion to LLVM dialect types provided by the
`LLVMTypeConverter` class. Users targeting the LLVM dialect can reuse and extend
this type converter to support other types. Extra care must be taken if the
conversion rules for built-in types are overridden: all conversion must use the
same type converter.

#### LLVM Dialect-compatible Types

The types [compatible](Dialects/LLVM.md#built-in-type-compatibility) with the
LLVM dialect are kept as is.

#### Complex Type

Complex type is converted into an LLVM dialect literal structure type with two
elements:

-   real part;
-   imaginary part.

The elemental type is converted recursively using these rules.

Example:

```mlir
  complex<f32>
  // ->
  !llvm.struct<(f32, f32)>
```

#### Index Type

Index type is converted into an LLVM dialect integer type with the bitwidth
specified by the [data layout](DataLayout.md) of the closest module. For
example, on x86-64 CPUs it converts to i64. This behavior can be overridden by
the type converter configuration, which is often exposed as a pass option by
conversion passes.

Example:

```mlir
  index
  // -> on x86_64
  i64
```

#### Ranked MemRef Types

Ranked memref types are converted into an LLVM dialect literal structure type
that contains the dynamic information associated with the memref object,
referred to as *descriptor*. Only memrefs in the
**[strided form](Dialects/Builtin.md/#strided-memref)** can be converted to the
LLVM dialect with the default descriptor format. Memrefs with other, less
trivial layouts should be converted into the strided form first, e.g., by
materializing the non-trivial address remapping due to layout as `affine.apply`
operations.

The default memref descriptor is a struct with the following fields:

1.  The pointer to the data buffer as allocated, referred to as "allocated
    pointer". This is only useful for deallocating the memref.
2.  The pointer to the properly aligned data pointer that the memref indexes,
    referred to as "aligned pointer".
3.  A lowered converted `index`-type integer containing the distance in number
    of elements between the beginning of the (aligned) buffer and the first
    element to be accessed through the memref, referred to as "offset".
4.  An array containing as many converted `index`-type integers as the rank of
    the memref: the array represents the size, in number of elements, of the
    memref along the given dimension.
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
// Assuming index is converted to i64.

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
memref<1x? x vector<4xf32>> -> !llvm.struct<(ptr<vector<4 x f32>>,
                                             ptr<vector<4 x f32>>, i64,
                                             array<2 x i64>, array<2 x i64>)>
```

#### Unranked MemRef Types

Unranked memref types are converted to LLVM dialect literal structure type that
contains the dynamic information associated with the memref object, referred to
as *unranked descriptor*. It contains:

1.  a converted `index`-typed integer representing the dynamic rank of the
    memref;
2.  a type-erased pointer (`!llvm.ptr<i8>`) to a ranked memref descriptor with
    the contents listed above.

This descriptor is primarily intended for interfacing with rank-polymorphic
library functions. The pointer to the ranked memref descriptor points to some
*allocated* memory, which may reside on stack of the current function or in
heap. Conversion patterns for operations producing unranked memrefs are expected
to manage the allocation. Note that this may lead to stack allocations
(`llvm.alloca`) being performed in a loop and not reclaimed until the end of the
current function.

#### Function Types

Function types are converted to LLVM dialect function types as follows:

-   function argument and result types are converted recursively using these
    rules;
-   if a function type has multiple results, they are wrapped into an LLVM
    dialect literal structure type since LLVM function types must have exactly
    one result;
-   if a function type has no results, the corresponding LLVM dialect function
    type will have one `!llvm.void` result since LLVM function types must have a
    result;
-   function types used in arguments of another function type are wrapped in an
    LLVM dialect pointer type to comply with LLVM IR expectations;
-   the structs corresponding to `memref` types, both ranked and unranked,
    appearing as function arguments are unbundled into individual function
    arguments to allow for specifying metadata such as aliasing information on
    individual pointers;
-   the conversion of `memref`-typed arguments is subject to
    [calling conventions](TargetLLVMIR.md#calling-conventions).
-   if a function type has boolean attribute `func.varargs` being set, the
    converted LLVM function will be variadic.

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

// If "func.varargs" attribute is set:
(i32) -> () attributes { "func.varargs" = true }
// the corresponding LLVM function will be variadic:
!llvm.func<void (i32, ...)>
```

Conversion patterns are available to convert built-in function operations and
standard call operations targeting those functions using these conversion rules.

#### Multi-dimensional Vector Types

LLVM IR only supports *one-dimensional* vectors, unlike MLIR where vectors can
be multi-dimensional. Vector types cannot be nested in either IR. In the
one-dimensional case, MLIR vectors are converted to LLVM IR vectors of the same
size with element type converted using these conversion rules. In the
n-dimensional case, MLIR vectors are converted to (n-1)-dimensional array types
of one-dimensional vectors.

Examples:

```
vector<4x8 x f32>
// ->
!llvm.array<4 x vector<8 x f32>>

memref<2 x vector<4x8 x f32>
// ->
!llvm.struct<(ptr<array<4 x vector<8xf32>>>, ptr<array<4 x vector<8xf32>>>
              i64, array<1 x i64>, array<1 x i64>)>
```

#### Tensor Types

Tensor types cannot be converted to the LLVM dialect. Operations on tensors must
be [bufferized](Bufferization.md) before being converted.

### Calling Conventions

Calling conventions provides a mechanism to customize the conversion of function
and function call operations without changing how individual types are handled
elsewhere. They are implemented simultaneously by the default type converter and
by the conversion patterns for the relevant operations.

#### Function Result Packing

In case of multi-result functions, the returned values are inserted into a
structure-typed value before being returned and extracted from it at the call
site. This transformation is a part of the conversion and is transparent to the
defines and uses of the values being returned.

Example:

```mlir
func.func @foo(%arg0: i32, %arg1: i64) -> (i32, i64) {
  return %arg0, %arg1 : i32, i64
}
func.func @bar() {
  %0 = arith.constant 42 : i32
  %1 = arith.constant 17 : i64
  %2:2 = call @foo(%0, %1) : (i32, i64) -> (i32, i64)
  "use_i32"(%2#0) : (i32) -> ()
  "use_i64"(%2#1) : (i64) -> ()
}

// is transformed into

llvm.func @foo(%arg0: i32, %arg1: i64) -> !llvm.struct<(i32, i64)> {
  // insert the vales into a structure
  %0 = llvm.mlir.undef : !llvm.struct<(i32, i64)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i64)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i32, i64)>

  // return the structure value
  llvm.return %2 : !llvm.struct<(i32, i64)>
}
llvm.func @bar() {
  %0 = llvm.mlir.constant(42 : i32) : i32
  %1 = llvm.mlir.constant(17) : i64

  // call and extract the values from the structure
  %2 = llvm.call @bar(%0, %1)
     : (i32, i32) -> !llvm.struct<(i32, i64)>
  %3 = llvm.extractvalue %2[0] : !llvm.struct<(i32, i64)>
  %4 = llvm.extractvalue %2[1] : !llvm.struct<(i32, i64)>

  // use as before
  "use_i32"(%3) : (i32) -> ()
  "use_i64"(%4) : (i64) -> ()
}
```

#### Default Calling Convention for Ranked MemRef

The default calling convention converts `memref`-typed function arguments to
LLVM dialect literal structs
[defined above](TargetLLVMIR.md#ranked-memref-types) before unbundling them into
individual scalar arguments.

Examples:

This convention is implemented in the conversion of `func.func` and `func.call` to
the LLVM dialect, with the former unpacking the descriptor into a set of
individual values and the latter packing those values back into a descriptor so
as to make it transparently usable by other operations. Conversions from other
dialects should take this convention into account.

This specific convention is motivated by the necessity to specify alignment and
aliasing attributes on the raw pointers underpinning the memref.

Examples:

```mlir
func.func @foo(%arg0: memref<?xf32>) -> () {
  "use"(%arg0) : (memref<?xf32>) -> ()
  return
}

// Gets converted to the following
// (using type alias for brevity):
!llvm.memref_1d = !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<1xi64>)>

llvm.func @foo(%arg0: !llvm.ptr<f32>,  // Allocated pointer.
               %arg1: !llvm.ptr<f32>,  // Aligned pointer.
               %arg2: i64,             // Offset.
               %arg3: i64,             // Size in dim 0.
               %arg4: i64) {           // Stride in dim 0.
  // Populate memref descriptor structure.
  %0 = llvm.mlir.undef :
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.memref_1d
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.memref_1d
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.memref_1d
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.memref_1d
  %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.memref_1d

  // Descriptor is now usable as a single value.
  "use"(%5) : (!llvm.memref_1d) -> ()
  llvm.return
}
```

```mlir
func.func @bar() {
  %0 = "get"() : () -> (memref<?xf32>)
  call @foo(%0) : (memref<?xf32>) -> ()
  return
}

// Gets converted to the following
// (using type alias for brevity):
!llvm.memref_1d = !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<1xi64>)>

llvm.func @bar() {
  %0 = "get"() : () -> !llvm.memref_1d

  // Unpack the memref descriptor.
  %1 = llvm.extractvalue %0[0] : !llvm.memref_1d
  %2 = llvm.extractvalue %0[1] : !llvm.memref_1d
  %3 = llvm.extractvalue %0[2] : !llvm.memref_1d
  %4 = llvm.extractvalue %0[3, 0] : !llvm.memref_1d
  %5 = llvm.extractvalue %0[4, 0] : !llvm.memref_1d

  // Pass individual values to the callee.
  llvm.call @foo(%1, %2, %3, %4, %5) : (!llvm.memref_1d) -> ()
  llvm.return
}
```

#### Default Calling Convention for Unranked MemRef

For unranked memrefs, the list of function arguments always contains two
elements, same as the unranked memref descriptor: an integer rank, and a
type-erased (`!llvm<"i8*">`) pointer to the ranked memref descriptor. Note that
while the *calling convention* does not require allocation, *casting* to
unranked memref does since one cannot take an address of an SSA value containing
the ranked memref, which must be stored in some memory instead. The caller is in
charge of ensuring the thread safety and management of the allocated memory, in
particular the deallocation.

Example

```mlir
llvm.func @foo(%arg0: memref<*xf32>) -> () {
  "use"(%arg0) : (memref<*xf32>) -> ()
  return
}

// Gets converted to the following.

llvm.func @foo(%arg0: i64              // Rank.
               %arg1: !llvm.ptr<i8>) { // Type-erased pointer to descriptor.
  // Pack the unranked memref descriptor.
  %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>

  "use"(%2) : (!llvm.struct<(i64, ptr<i8>)>) -> ()
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
  %0 = "get"() : () -> (!llvm.struct<(i64, ptr<i8>)>)

  // Unpack the memref descriptor.
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr<i8>)>
  %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>

  // Pass individual values to the callee.
  llvm.call @foo(%1, %2) : (i64, !llvm.ptr<i8>)
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
default conversion of `func.call` and `func.call_indirect` copies the ranked
descriptor to newly allocated memory on the caller's stack. Thus, the convention
of the ranked memref descriptor pointed to by an unranked memref descriptor
being stored on stack is respected.

#### Bare Pointer Calling Convention for Ranked MemRef

The "bare pointer" calling convention converts `memref`-typed function arguments
to a *single* pointer to the aligned data. Note that this does *not* apply to
uses of `memref` outside of function signatures, the default descriptor
structures are still used. This convention further restricts the supported cases
to the following.

-   `memref` types with default layout.
-   `memref` types with all dimensions statically known.
-   `memref` values allocated in such a way that the allocated and aligned
    pointer match. Alternatively, the same function must handle allocation and
    deallocation since only one pointer is passed to any callee.

Examples:

```
func.func @callee(memref<2x4xf32>) {

func.func @caller(%0 : memref<2x4xf32>) {
  call @callee(%0) : (memref<2x4xf32>) -> ()
}

// ->

!descriptor = !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                            array<2xi64>, array<2xi64>)>

llvm.func @callee(!llvm.ptr<f32>)

llvm.func @caller(%arg0: !llvm.ptr<f32>) {
  // A descriptor value is defined at the function entry point.
  %0 = llvm.mlir.undef : !descriptor

  // Both the allocated and aligned pointer are set up to the same value.
  %1 = llvm.insertelement %arg0, %0[0] : !descriptor
  %2 = llvm.insertelement %arg0, %1[1] : !descriptor

  // The offset is set up to zero.
  %3 = llvm.mlir.constant(0 : index) : i64
  %4 = llvm.insertelement %3, %2[2] : !descriptor

  // The sizes and strides are derived from the statically known values.
  %5 = llvm.mlir.constant(2 : index) : i64
  %6 = llvm.mlir.constant(4 : index) : i64
  %7 = llvm.insertelement %5, %4[3, 0] : !descriptor
  %8 = llvm.insertelement %6, %7[3, 1] : !descriptor
  %9 = llvm.mlir.constant(1 : index) : i64
  %10 = llvm.insertelement %9, %8[4, 0] : !descriptor
  %11 = llvm.insertelement %10, %9[4, 1] : !descriptor

  // The function call corresponds to extracting the aligned data pointer.
  %12 = llvm.extractelement %11[1] : !descriptor
  llvm.call @callee(%12) : (!llvm.ptr<f32>) -> ()
}
```

#### Bare Pointer Calling Convention For Unranked MemRef

The "bare pointer" calling convention does not support unranked memrefs as their
shape cannot be known at compile time.

### Generic alloction and deallocation functions

When converting the Memref dialect, allocations and deallocations are converted
into calls to `malloc` (`aligned_alloc` if aligned allocations are requested)
and `free`. However, it is possible to convert them to more generic functions
which can be implemented by a runtime library, thus allowing custom allocation
strategies or runtime profiling. When the conversion pass is  instructed to
perform such operation, the names of the calles are
`_mlir_memref_to_llvm_alloc`, `_mlir_memref_to_llvm_aligned_alloc` and
`_mlir_memref_to_llvm_free`. Their signatures are the same of `malloc`,
`aligned_alloc` and `free`.

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

Furthermore, we also rewrite function results to pointer parameters if the
rewritten function result has a struct type. The special result parameter is
added as the first parameter and is of pointer-to-struct type.

If enabled, the option will do the following. For *external* functions declared
in the MLIR module.

1.  Declare a new function `_mlir_ciface_<original name>` where memref arguments
    are converted to pointer-to-struct and the remaining arguments are converted
    as usual. Results are converted to a special argument if they are of struct
    type.
2.  Add a body to the original function (making it non-external) that
    1.  allocates memref descriptors,
    2.  populates them,
    3.  potentially allocates space for the result struct, and
    4.  passes the pointers to these into the newly declared interface function,
        then
    5.  collects the result of the call (potentially from the result struct),
        and
    6.  returns it to the caller.

For (non-external) functions defined in the MLIR module.

1.  Define a new function `_mlir_ciface_<original name>` where memref arguments
    are converted to pointer-to-struct and the remaining arguments are converted
    as usual. Results are converted to a special argument if they are of struct
    type.
2.  Populate the body of the newly defined function with IR that
    1.  loads descriptors from pointers;
    2.  unpacks descriptor into individual non-aggregate values;
    3.  passes these values into the original function;
    4.  collects the results of the call and
    5.  either copies the results into the result struct or returns them to the
        caller.

Examples:

```mlir

func.func @qux(%arg0: memref<?x?xf32>)

// Gets converted into the following
// (using type alias for brevity):
!llvm.memref_2d = !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2xi64>, array<2xi64>)>

// Function with unpacked arguments.
llvm.func @qux(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>,
               %arg2: i64, %arg3: i64, %arg4: i64,
               %arg5: i64, %arg6: i64) {
  // Populate memref descriptor (as per calling convention).
  %0 = llvm.mlir.undef : !llvm.memref_2d
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.memref_2d
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.memref_2d
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.memref_2d
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.memref_2d
  %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.memref_2d
  %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.memref_2d
  %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.memref_2d

  // Store the descriptor in a stack-allocated space.
  %8 = llvm.mlir.constant(1 : index) : i64
  %9 = llvm.alloca %8 x !llvm.memref_2d
     : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64,
                                        array<2xi64>, array<2xi64>)>>
  llvm.store %7, %9 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64,
                                        array<2xi64>, array<2xi64>)>>

  // Call the interface function.
  llvm.call @_mlir_ciface_qux(%9)
     : (!llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64,
                          array<2xi64>, array<2xi64>)>>) -> ()

  // The stored descriptor will be freed on return.
  llvm.return
}

// Interface function.
llvm.func @_mlir_ciface_qux(!llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64,
                                              array<2xi64>, array<2xi64>)>>)
```

```mlir
func.func @foo(%arg0: memref<?x?xf32>) {
  return
}

// Gets converted into the following
// (using type alias for brevity):
!llvm.memref_2d = !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2xi64>, array<2xi64>)>
!llvm.memref_2d_ptr = !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2xi64>, array<2xi64>)>>

// Function with unpacked arguments.
llvm.func @foo(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>,
               %arg2: i64, %arg3: i64, %arg4: i64,
               %arg5: i64, %arg6: i64) {
  llvm.return
}

// Interface function callable from C.
llvm.func @_mlir_ciface_foo(%arg0: !llvm.memref_2d_ptr) {
  // Load the descriptor.
  %0 = llvm.load %arg0 : !llvm.memref_2d_ptr

  // Unpack the descriptor as per calling convention.
  %1 = llvm.extractvalue %0[0] : !llvm.memref_2d
  %2 = llvm.extractvalue %0[1] : !llvm.memref_2d
  %3 = llvm.extractvalue %0[2] : !llvm.memref_2d
  %4 = llvm.extractvalue %0[3, 0] : !llvm.memref_2d
  %5 = llvm.extractvalue %0[3, 1] : !llvm.memref_2d
  %6 = llvm.extractvalue %0[4, 0] : !llvm.memref_2d
  %7 = llvm.extractvalue %0[4, 1] : !llvm.memref_2d
  llvm.call @foo(%1, %2, %3, %4, %5, %6, %7)
    : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64,
       i64, i64) -> ()
  llvm.return
}
```

```mlir
func.func @foo(%arg0: memref<?x?xf32>) -> memref<?x?xf32> {
  return %arg0 : memref<?x?xf32>
}

// Gets converted into the following
// (using type alias for brevity):
!llvm.memref_2d = !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2xi64>, array<2xi64>)>
!llvm.memref_2d_ptr = !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2xi64>, array<2xi64>)>>

// Function with unpacked arguments.
llvm.func @foo(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64,
               %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64)
    -> !llvm.memref_2d {
  %0 = llvm.mlir.undef : !llvm.memref_2d
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.memref_2d
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.memref_2d
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.memref_2d
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.memref_2d
  %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.memref_2d
  %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.memref_2d
  %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.memref_2d
  llvm.return %7 : !llvm.memref_2d
}

// Interface function callable from C.
llvm.func @_mlir_ciface_foo(%arg0: !llvm.memref_2d_ptr, %arg1: !llvm.memref_2d_ptr) {
  %0 = llvm.load %arg1 : !llvm.memref_2d_ptr
  %1 = llvm.extractvalue %0[0] : !llvm.memref_2d
  %2 = llvm.extractvalue %0[1] : !llvm.memref_2d
  %3 = llvm.extractvalue %0[2] : !llvm.memref_2d
  %4 = llvm.extractvalue %0[3, 0] : !llvm.memref_2d
  %5 = llvm.extractvalue %0[3, 1] : !llvm.memref_2d
  %6 = llvm.extractvalue %0[4, 0] : !llvm.memref_2d
  %7 = llvm.extractvalue %0[4, 1] : !llvm.memref_2d
  %8 = llvm.call @foo(%1, %2, %3, %4, %5, %6, %7)
    : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> !llvm.memref_2d
  llvm.store %8, %arg0 : !llvm.memref_2d_ptr
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

Limitation: Right now we cannot generate C interface for variadic functions,
regardless of being non-external or external. Because C functions are unable to
"forward" variadic arguments like this:
```c
void bar(int, ...);

void foo(int x, ...) {
  // ERROR: no way to forward variadic arguments.
  void bar(x, ...);
}
```

### Address Computation

Accesses to a memref element are transformed into an access to an element of the
buffer pointed to by the descriptor. The position of the element in the buffer
is calculated by linearizing memref indices in row-major order (lexically first
index is the slowest varying, similar to C, but accounting for strides). The
computation of the linear address is emitted as arithmetic operation in the LLVM
IR dialect. Strides are extracted from the memref descriptor.

Examples:

An access to a memref with indices:

```mlir
%0 = memref.load %m[%1,%2,%3,%4] : memref<?x?x4x8xf32, offset: ?>
```

is transformed into the equivalent of the following code:

```mlir
// Compute the linearized index from strides.
// When strides or, in absence of explicit strides, the corresponding sizes are
// dynamic, extract the stride value from the descriptor.
%stride1 = llvm.extractvalue[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                   array<4xi64>, array<4xi64>)>
%addr1 = arith.muli %stride1, %1 : i64

// When the stride or, in absence of explicit strides, the trailing sizes are
// known statically, this value is used as a constant. The natural value of
// strides is the product of all sizes following the current dimension.
%stride2 = llvm.mlir.constant(32 : index) : i64
%addr2 = arith.muli %stride2, %2 : i64
%addr3 = arith.addi %addr1, %addr2 : i64

%stride3 = llvm.mlir.constant(8 : index) : i64
%addr4 = arith.muli %stride3, %3 : i64
%addr5 = arith.addi %addr3, %addr4 : i64

// Multiplication with the known unit stride can be omitted.
%addr6 = arith.addi %addr5, %4 : i64

// If the linear offset is known to be zero, it can also be omitted. If it is
// dynamic, it is extracted from the descriptor.
%offset = llvm.extractvalue[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                               array<4xi64>, array<4xi64>)>
%addr7 = arith.addi %addr6, %offset : i64

// All accesses are based on the aligned pointer.
%aligned = llvm.extractvalue[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                array<4xi64>, array<4xi64>)>

// Get the address of the data pointer.
%ptr = llvm.getelementptr %aligned[%addr8]
     : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4xi64>, array<4xi64>)>
     -> !llvm.ptr<f32>

// Perform the actual load.
%0 = llvm.load %ptr : !llvm.ptr<f32>
```

For stores, the address computation code is identical and only the actual store
operation is different.

Note: the conversion does not perform any sort of common subexpression
elimination when emitting memref accesses.

### Utility Classes

Utility classes common to many conversions to the LLVM dialect can be found
under `lib/Conversion/LLVMCommon`. They include the following.

-   `LLVMConversionTarget` specifies all LLVM dialect operations as legal.
-   `LLVMTypeConverter` implements the default type conversion as described
    above.
-   `ConvertOpToLLVMPattern` extends the conversion pattern class with LLVM
    dialect-specific functionality.
-   `VectorConvertOpToLLVMPattern` extends the previous class to automatically
    unroll operations on higher-dimensional vectors into lists of operations on
    one-dimensional vectors before.
-   `StructBuilder` provides a convenient API for building IR that creates or
    accesses values of LLVM dialect structure types; it is derived by
    `MemRefDescriptor`, `UrankedMemrefDescriptor` and `ComplexBuilder` for the
    built-in types convertible to LLVM dialect structure types.

## Translation to LLVM IR

MLIR modules containing `llvm.func`, `llvm.mlir.global` and `llvm.metadata`
operations can be translated to LLVM IR modules using the following scheme.

-   Module-level globals are translated to LLVM IR global values.
-   Module-level metadata are translated to LLVM IR metadata, which can be later
    augmented with additional metadata defined on specific ops.
-   All functions are declared in the module so that they can be referenced.
-   Each function is then translated separately and has access to the complete
    mappings between MLIR and LLVM IR globals, metadata, and functions.
-   Within a function, blocks are traversed in topological order and translated
    to LLVM IR basic blocks. In each basic block, PHI nodes are created for each
    of the block arguments, but not connected to their source blocks.
-   Within each block, operations are translated in their order. Each operation
    has access to the same mappings as the function and additionally to the
    mapping of values between MLIR and LLVM IR, including PHI nodes. Operations
    with regions are responsible for translated the regions they contain.
-   After operations in a function are translated, the PHI nodes of blocks in
    this function are connected to their source values, which are now available.

The translation mechanism provides extension hooks for translating custom
operations to LLVM IR via a dialect interface `LLVMTranslationDialectInterface`:

-   `convertOperation` translates an operation that belongs to the current
    dialect to LLVM IR given an `IRBuilderBase` and various mappings;
-   `amendOperation` performs additional actions on an operation if it contains
    a dialect attribute that belongs to the current dialect, for example sets up
    instruction-level metadata.

Dialects containing operations or attributes that want to be translated to LLVM
IR must provide an implementation of this interface and register it with the
system. Note that registration may happen without creating the dialect, for
example, in a separate library to avoid the need for the "main" dialect library
to depend on LLVM IR libraries. The implementations of these methods may used
the
[`ModuleTranslation`](https://mlir.llvm.org/doxygen/classmlir_1_1LLVM_1_1ModuleTranslation.html)
object provided to them which holds the state of the translation and contains
numerous utilities.

Note that this extension mechanism is *intentionally restrictive*. LLVM IR has a
small, relatively stable set of instructions and types that MLIR intends to
model fully. Therefore, the extension mechanism is provided only for LLVM IR
constructs that are more often extended -- intrinsics and metadata. The primary
goal of the extension mechanism is to support sets of intrinsics, for example
those representing a particular instruction set. The extension mechanism does
not allow for customizing type or block translation, nor does it support custom
module-level operations. Such transformations should be performed within MLIR
and target the corresponding MLIR constructs.

## Translation from LLVM IR

An experimental flow allows one to import a substantially limited subset of LLVM
IR into MLIR, producing LLVM dialect operations.

```
  mlir-translate -import-llvm filename.ll
```
