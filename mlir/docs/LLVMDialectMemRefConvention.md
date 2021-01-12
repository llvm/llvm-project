# Built-in Function and MemRef Calling Convention

This documents describes the calling convention implemented in the conversion of
built-in [function operation](LangRef.md#functions), standard
[`call`](Dialects/Standard.md#stdcall-callop) operations and the handling of
[`memref`](LangRef.md#memref-type) type equivalents in the
[LLVM dialect](Dialects/LLVM.md). The conversion assumes the _default_
convention was used when converting
[built-in to the LLVM dialect types](ConversionToLLVMDialect.md).

## Function Result Packing

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

## Calling Convention for Ranked `memref`

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

// Gets converted to the following
// (using type alias for brevity):
!llvm.memref_1d = type !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                     array<1xi64>, array<1xi64>)>

llvm.func @foo(%arg0: !llvm.ptr<f32>,  // Allocated pointer.
               %arg1: !llvm.ptr<f32>,  // Aligned pointer.
               %arg2: i64,         // Offset.
               %arg3: i64,         // Size in dim 0.
               %arg4: i64) {       // Stride in dim 0.
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
func @bar() {
  %0 = "get"() : () -> (memref<?xf32>)
  call @foo(%0) : (memref<?xf32>) -> ()
  return
}

// Gets converted to the following
// (using type alias for brevity):
!llvm.memref_1d = type !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                     array<1xi64>, array<1xi64>)>

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

## Calling Convention for Unranked `memref`

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

llvm.func @foo(%arg0: i64        // Rank.
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
default conversion of `std.call` and `std.call_indirect` copies the ranked
descriptor to newly allocated memory on the caller's stack. Thus, the convention
of the ranked memref descriptor pointed to by an unranked memref descriptor
being stored on stack is respected.

*This convention may or may not apply if the conversion of MemRef types is
overridden by the user.*

## C-compatible wrapper emission

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

1.  Declare a new function `_mlir_ciface_<original name>` where memref arguments
    are converted to pointer-to-struct and the remaining arguments are converted
    as usual.
1.  Add a body to the original function (making it non-external) that
    1.  allocates a memref descriptor,
    1.  populates it, and
    1.  passes the pointer to it into the newly declared interface function,
        then
    1.  collects the result of the call and returns it to the caller.

For (non-external) functions defined in the MLIR module.

1.  Define a new function `_mlir_ciface_<original name>` where memref arguments
    are converted to pointer-to-struct and the remaining arguments are converted
    as usual.
1.  Populate the body of the newly defined function with IR that
    1.  loads descriptors from pointers;
    1.  unpacks descriptor into individual non-aggregate values;
    1.  passes these values into the original function;
    1.  collects the result of the call and returns it to the caller.

Examples:

```mlir

func @qux(%arg0: memref<?x?xf32>)

// Gets converted into the following
// (using type alias for brevity):
!llvm.memref_2d = type !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                     array<2xi64>, array<2xi64>)>

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
func @foo(%arg0: memref<?x?xf32>) {
  return
}

// Gets converted into the following
// (using type alias for brevity):
!llvm.memref_2d = type !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                     array<2xi64>, array<2xi64>)>
!llvm.memref_2d_ptr = type !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64,
                                             array<2xi64>, array<2xi64>)>>

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

## Default Memref Model

### Memref Descriptor

Within a converted function, a `memref`-typed value is represented by a memref
_descriptor_, the type of which is the structure type obtained by converting
from the memref type. This descriptor holds all the necessary information to
produce an address of a specific element. In particular, it holds dynamic values
for static sizes, and they are expected to match at all times.

It is created by the allocation operation and is updated by the conversion
operations that may change static dimensions into dynamic dimensions and vice
versa.

**Note**: LLVM IR conversion does not support `memref`s with layouts that are
not amenable to the strided form.

### Index Linearization

Accesses to a memref element are transformed into an access to an element of the
buffer pointed to by the descriptor. The position of the element in the buffer
is calculated by linearizing memref indices in row-major order (lexically first
index is the slowest varying, similar to C, but accounting for strides). The
computation of the linear address is emitted as arithmetic operation in the LLVM
IR dialect. Strides are extracted from the memref descriptor.

Examples:

An access to a memref with indices:

```mlir
%0 = load %m[%1,%2,%3,%4] : memref<?x?x4x8xf32, offset: ?>
```

is transformed into the equivalent of the following code:

```mlir
// Compute the linearized index from strides.
// When strides or, in absence of explicit strides, the corresponding sizes are
// dynamic, extract the stride value from the descriptor.
%stride1 = llvm.extractvalue[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                   array<4xi64>, array<4xi64>)>
%addr1 = muli %stride1, %1 : i64

// When the stride or, in absence of explicit strides, the trailing sizes are
// known statically, this value is used as a constant. The natural value of
// strides is the product of all sizes following the current dimension.
%stride2 = llvm.mlir.constant(32 : index) : i64
%addr2 = muli %stride2, %2 : i64
%addr3 = addi %addr1, %addr2 : i64

%stride3 = llvm.mlir.constant(8 : index) : i64
%addr4 = muli %stride3, %3 : i64
%addr5 = addi %addr3, %addr4 : i64

// Multiplication with the known unit stride can be omitted.
%addr6 = addi %addr5, %4 : i64

// If the linear offset is known to be zero, it can also be omitted. If it is
// dynamic, it is extracted from the descriptor.
%offset = llvm.extractvalue[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                               array<4xi64>, array<4xi64>)>
%addr7 = addi %addr6, %offset : i64

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
