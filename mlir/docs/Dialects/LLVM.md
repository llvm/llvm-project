# 'llvm' Dialect

This dialect maps [LLVM IR](https://llvm.org/docs/LangRef.html) into MLIR by
defining the corresponding operations and types. LLVM IR metadata is usually
represented as MLIR attributes, which offer additional structure verification.

We use "LLVM IR" to designate the
[intermediate representation of LLVM](https://llvm.org/docs/LangRef.html) and
"LLVM _dialect_" or "LLVM IR _dialect_" to refer to this MLIR dialect.

Unless explicitly stated otherwise, the semantics of the LLVM dialect operations
must correspond to the semantics of LLVM IR instructions and any divergence is
considered a bug. The dialect also contains auxiliary operations that smoothen
the differences in the IR structure, e.g., MLIR does not have `phi` operations
and LLVM IR does not have a `constant` operation. These auxiliary operations are
systematically prefixed with `mlir`, e.g. `llvm.mlir.constant` where `llvm.` is
the dialect namespace prefix.

[TOC]

## Dependency on LLVM IR

LLVM dialect is not expected to depend on any object that requires an
`LLVMContext`, such as an LLVM IR instruction or type. Instead, MLIR provides
thread-safe alternatives compatible with the rest of the infrastructure. The
dialect is allowed to depend on the LLVM IR objects that don't require a
context, such as data layout and triple description.

## Module Structure

IR modules use the built-in MLIR `ModuleOp` and support all its features. In
particular, modules can be named, nested and are subject to symbol visibility.
Modules can contain any operations, including LLVM functions and globals.

### Data Layout and Triple

An IR module may have an optional data layout and triple information attached
using MLIR attributes `llvm.data_layout` and `llvm.triple`, respectively. Both
are string attributes with the
[same syntax](https://llvm.org/docs/LangRef.html#data-layout) as in LLVM IR and
are verified to be correct. They can be defined as follows.

```mlir
module attributes {llvm.data_layout = "e",
                   llvm.target_triple = "aarch64-linux-android"} {
  // module contents
}
```

### Functions

LLVM functions are represented by a special operation, `llvm.func`, that has
syntax similar to that of the built-in function operation but supports
LLVM-related features such as linkage and variadic argument lists. See detailed
description in the operation list [below](#llvmfunc-mlirllvmllvmfuncop).

### PHI Nodes and Block Arguments

MLIR uses block arguments instead of PHI nodes to communicate values between
blocks. Therefore, the LLVM dialect has no operation directly equivalent to
`phi` in LLVM IR. Instead, all terminators can pass values as successor operands
as these values will be forwarded as block arguments when the control flow is
transferred.

For example:

```mlir
^bb1:
  %0 = llvm.addi %arg0, %cst : i32
  llvm.br ^bb2[%0: i32]

// If the control flow comes from ^bb1, %arg1 == %0.
^bb2(%arg1: i32)
  // ...
```

is equivalent to LLVM IR

```llvm
%0:
  %1 = add i32 %arg0, %cst
  br %3

%3:
  %arg1 = phi [%1, %0], //...
```

Since there is no need to use the block identifier to differentiate the source
of different values, the LLVM dialect supports terminators that transfer the
control flow to the same block with different arguments. For example:

```mlir
^bb1:
  llvm.cond_br %cond, ^bb2[%0: i32], ^bb2[%1: i32]

^bb2(%arg0: i32):
  // ...
```

### Context-Level Values

Some value kinds in LLVM IR, such as constants and undefs, are uniqued in
context and used directly in relevant operations. MLIR does not support such
values for thread-safety and concept parsimony reasons. Instead, regular values
are produced by dedicated operations that have the corresponding semantics:
[`llvm.mlir.constant`](#llvmmlirconstant-mlirllvmconstantop),
[`llvm.mlir.undef`](#llvmmlirundef-mlirllvmundefop),
[`llvm.mlir.null`](#llvmmlirnull-mlirllvmnullop). Note how these operations are
prefixed with `mlir.` to indicate that they don't belong to LLVM IR but are only
necessary to model it in MLIR. The values produced by these operations are
usable just like any other value.

Examples:

```mlir
// Create an undefined value of structure type with a 32-bit integer followed
// by a float.
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>

// Null pointer to i8.
%1 = llvm.mlir.null : !llvm.ptr<i8>

// Null pointer to a function with signature void().
%2 = llvm.mlir.null : !llvm.ptr<func<void ()>>

// Constant 42 as i32.
%3 = llvm.mlir.constant(42 : i32) : i32

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```

Note that constants list the type twice. This is an artifact of the LLVM dialect
not using built-in types, which are used for typed MLIR attributes. The syntax
will be reevaluated after considering composite constants.

### Globals

Global variables are also defined using a special operation,
[`llvm.mlir.global`](#llvmmlirglobal-mlirllvmglobalop), located at the module
level. Globals are MLIR symbols and are identified by their name.

Since functions need to be isolated-from-above, i.e. values defined outside the
function cannot be directly used inside the function, an additional operation,
[`llvm.mlir.addressof`](#llvmmliraddressof-mlirllvmaddressofop), is provided to
locally define a value containing the _address_ of a global. The actual value
can then be loaded from that pointer, or a new value can be stored into it if
the global is not declared constant. This is similar to LLVM IR where globals
are accessed through name and have a pointer type.

### Linkage

Module-level named objects in the LLVM dialect, namely functions and globals,
have an optional _linkage_ attribute derived from LLVM IR
[linkage types](https://llvm.org/docs/LangRef.html#linkage-types). Linkage is
specified by the same keyword as in LLVM IR and is located between the operation
name (`llvm.func` or `llvm.global`) and the symbol name. If no linkage keyword
is present, `external` linkage is assumed by default. Linkage is _distinct_ from
MLIR symbol visibility.

### Attribute Pass-Through

The LLVM dialect provides a mechanism to forward function-level attributes to
LLVM IR using the `passthrough` attribute. This is an array attribute containing
either string attributes or array attributes. In the former case, the value of
the string is interpreted as the name of LLVM IR function attribute. In the
latter case, the array is expected to contain exactly two string attributes, the
first corresponding to the name of LLVM IR function attribute, and the second
corresponding to its value. Note that even integer LLVM IR function attributes
have their value represented in the string form.

Example:

```mlir
llvm.func @func() attributes {
  passthrough = ["noinline",           // value-less attribute
                 ["alignstack", "4"],  // integer attribute with value
                 ["other", "attr"]]    // attribute unknown to LLVM
} {
  llvm.return
}
```

If the attribute is not known to LLVM IR, it will be attached as a string
attribute.

## Types

LLVM dialect uses built-in types whenever possible and defines a set of
complementary types, which correspond to the LLVM IR types that cannot be
directly represented with built-in types. Similarly to other MLIR context-owned
objects, the creation and manipulation of LLVM dialect types is thread-safe.

MLIR does not support module-scoped named type declarations, e.g. `%s = type
{i32, i32}` in LLVM IR. Instead, types must be fully specified at each use,
except for recursive types where only the first reference to a named type needs
to be fully specified. MLIR [type aliases](../LangRef.md/#type-aliases) can be
used to achieve more compact syntax.

The general syntax of LLVM dialect types is `!llvm.`, followed by a type kind
identifier (e.g., `ptr` for pointer or `struct` for structure) and by an
optional list of type parameters in angle brackets. The dialect follows MLIR
style for types with nested angle brackets and keyword specifiers rather than
using different bracket styles to differentiate types. Types inside the angle
brackets may omit the `!llvm.` prefix for brevity: the parser first attempts to
find a type (starting with `!` or a built-in type) and falls back to accepting a
keyword. For example, `!llvm.ptr<!llvm.ptr<i32>>` and `!llvm.ptr<ptr<i32>>` are
equivalent, with the latter being the canonical form, and denote a pointer to a
pointer to a 32-bit integer.

### Built-in Type Compatibility

LLVM dialect accepts a subset of built-in types that are referred to as _LLVM
dialect-compatible types_. The following types are compatible:

-   Signless integers - `iN` (`IntegerType`).
-   Floating point types - `bfloat`, `half`, `float`, `double` , `f80`, `f128`
    (`FloatType`).
-   1D vectors of signless integers or floating point types - `vector<NxT>`
    (`VectorType`).

Note that only a subset of types that can be represented by a given class is
compatible. For example, signed and unsigned integers are not compatible. LLVM
provides a function, `bool LLVM::isCompatibleType(Type)`, that can be used as a
compatibility check.

Each LLVM IR type corresponds to *exactly one* MLIR type, either built-in or
LLVM dialect type. For example, because `i32` is LLVM-compatible, there is no
`!llvm.i32` type. However, `!llvm.ptr<T>` is defined in the LLVM dialect as
there is no corresponding built-in type.

### Additional Simple Types

The following non-parametric types derived from the LLVM IR are available in the
LLVM dialect:

-   `!llvm.x86_mmx` (`LLVMX86MMXType`) - value held in an MMX register on x86
    machine.
-   `!llvm.ppc_fp128` (`LLVMPPCFP128Type`) - 128-bit floating-point value (two
    64 bits).
-   `!llvm.token` (`LLVMTokenType`) - a non-inspectable value associated with an
    operation.
-   `!llvm.metadata` (`LLVMMetadataType`) - LLVM IR metadata, to be used only if
    the metadata cannot be represented as structured MLIR attributes.
-   `!llvm.void` (`LLVMVoidType`) - does not represent any value; can only
    appear in function results.

These types represent a single value (or an absence thereof in case of `void`)
and correspond to their LLVM IR counterparts.

### Additional Parametric Types

These types are parameterized by the types they contain, e.g., the pointee or
the element type, which can be either compatible built-in or LLVM dialect types.

#### Pointer Types

Pointer types specify an address in memory.

Both opaque and type-parameterized pointer types are supported.
[Opaque pointers](https://llvm.org/docs/OpaquePointers.html) do not indicate the
type of the data pointed to, and are intended to simplify LLVM IR by encoding
behavior relevant to the pointee type into operations rather than into types.
Non-opaque pointer types carry the pointee type as a type parameter. Both kinds
of pointers may be additionally parameterized by an address space. The address
space is an integer, but this choice may be reconsidered if MLIR implements
named address spaces. The syntax of pointer types is as follows:

```
  llvm-ptr-type ::= `!llvm.ptr` (`<` integer-literal `>`)?
                  | `!llvm.ptr<` type (`,` integer-literal)? `>`
```

where the former case is the opaque pointer type and the latter case is the
non-opaque pointer type; the optional group containing the integer literal
corresponds to the memory space. All cases are represented by `LLVMPointerType`
internally.

#### Array Types

Array types represent sequences of elements in memory. Array elements can be
addressed with a value unknown at compile time, and can be nested. Only 1D
arrays are allowed though.

Array types are parameterized by the fixed size and the element type.
Syntactically, their representation is the following:

```
  llvm-array-type ::= `!llvm.array<` integer-literal `x` type `>`
```

and they are internally represented as `LLVMArrayType`.

#### Function Types

Function types represent the type of a function, i.e. its signature.

Function types are parameterized by the result type, the list of argument types
and by an optional "variadic" flag. Unlike built-in `FunctionType`, LLVM dialect
functions (`LLVMFunctionType`) always have single result, which may be
`!llvm.void` if the function does not return anything. The syntax is as follows:

```
  llvm-func-type ::= `!llvm.func<` type `(` type-list (`,` `...`)? `)` `>`
```

For example,

```mlir
!llvm.func<void ()>           // a function with no arguments;
!llvm.func<i32 (f32, i32)>    // a function with two arguments and a result;
!llvm.func<void (i32, ...)>   // a variadic function with at least one argument.
```

In the LLVM dialect, functions are not first-class objects and one cannot have a
value of function type. Instead, one can take the address of a function and
operate on pointers to functions.

### Vector Types

Vector types represent sequences of elements, typically when multiple data
elements are processed by a single instruction (SIMD). Vectors are thought of as
stored in registers and therefore vector elements can only be addressed through
constant indices.

Vector types are parameterized by the size, which may be either _fixed_ or a
multiple of some fixed size in case of _scalable_ vectors, and the element type.
Vectors cannot be nested and only 1D vectors are supported. Scalable vectors are
still considered 1D.

LLVM dialect uses built-in vector types for _fixed_-size vectors of built-in
types, and provides additional types for fixed-sized vectors of LLVM dialect
types (`LLVMFixedVectorType`) and scalable vectors of any types
(`LLVMScalableVectorType`). These two additional types share the following
syntax:

```
  llvm-vec-type ::= `!llvm.vec<` (`?` `x`)? integer-literal `x` type `>`
```

Note that the sets of element types supported by built-in and LLVM dialect
vector types are mutually exclusive, e.g., the built-in vector type does not
accept `!llvm.ptr<i32>` and the LLVM dialect fixed-width vector type does not
accept `i32`.

The following functions are provided to operate on any kind of the vector types
compatible with the LLVM dialect:

-   `bool LLVM::isCompatibleVectorType(Type)` - checks whether a type is a
    vector type compatible with the LLVM dialect;
-   `Type LLVM::getVectorElementType(Type)` - returns the element type of any
    vector type compatible with the LLVM dialect;
-   `llvm::ElementCount LLVM::getVectorNumElements(Type)` - returns the number
    of elements in any vector type compatible with the LLVM dialect;
-   `Type LLVM::getFixedVectorType(Type, unsigned)` - gets a fixed vector type
    with the given element type and size; the resulting type is either a
    built-in or an LLVM dialect vector type depending on which one supports the
    given element type.

#### Examples of Compatible Vector Types

```mlir
vector<42 x i32>                   // Vector of 42 32-bit integers.
!llvm.vec<42 x ptr<i32>>           // Vector of 42 pointers to 32-bit integers.
!llvm.vec<? x 4 x i32>             // Scalable vector of 32-bit integers with
                                   // size divisible by 4.
!llvm.array<2 x vector<2 x i32>>   // Array of 2 vectors of 2 32-bit integers.
!llvm.array<2 x vec<2 x ptr<i32>>> // Array of 2 vectors of 2 pointers to 32-bit
                                   // integers.
```

### Structure Types

The structure type is used to represent a collection of data members together in
memory. The elements of a structure may be any type that has a size.

Structure types are represented in a single dedicated class
mlir::LLVM::LLVMStructType. Internally, the struct type stores a (potentially
empty) name, a (potentially empty) list of contained types and a bitmask
indicating whether the struct is named, opaque, packed or uninitialized.
Structure types that don't have a name are referred to as _literal_ structs.
Such structures are uniquely identified by their contents. _Identified_ structs
on the other hand are uniquely identified by the name.

#### Identified Structure Types

Identified structure types are uniqued using their name in a given context.
Attempting to construct an identified structure with the same name a structure
that already exists in the context *will result in the existing structure being
returned*. **MLIR does not auto-rename identified structs in case of name
conflicts** because there is no naming scope equivalent to a module in LLVM IR
since MLIR modules can be arbitrarily nested.

Programmatically, identified structures can be constructed in an _uninitialized_
state. In this case, they are given a name but the body must be set up by a
later call, using MLIR's type mutation mechanism. Such uninitialized types can
be used in type construction, but must be eventually initialized for IR to be
valid. This mechanism allows for constructing _recursive_ or mutually referring
structure types: an uninitialized type can be used in its own initialization.

Once the type is initialized, its body cannot be changed anymore. Any further
attempts to modify the body will fail and return failure to the caller _unless
the type is initialized with the exact same body_. Type initialization is
thread-safe; however, if a concurrent thread initializes the type before the
current thread, the initialization may return failure.

The syntax for identified structure types is as follows.

```
llvm-ident-struct-type ::= `!llvm.struct<` string-literal, `opaque` `>`
                         | `!llvm.struct<` string-literal, `packed`?
                           `(` type-or-ref-list  `)` `>`
type-or-ref-list ::= <maybe empty comma-separated list of type-or-ref>
type-or-ref ::= <any compatible type with optional !llvm.>
              | `!llvm.`? `struct<` string-literal `>`
```

The body of the identified struct is printed in full unless the it is
transitively contained in the same struct. In the latter case, only the
identifier is printed. For example, the structure containing the pointer to
itself is represented as `!llvm.struct<"A", (ptr<"A">)>`, and the structure `A`
containing two pointers to the structure `B` containing a pointer to the
structure `A` is represented as `!llvm.struct<"A", (ptr<"B", (ptr<"A">)>,
ptr<"B", (ptr<"A">))>`. Note that the structure `B` is "unrolled" for both
elements. _A structure with the same name but different body is a syntax error._
**The user must ensure structure name uniqueness across all modules processed in
a given MLIR context.** Structure names are arbitrary string literals and may
include, e.g., spaces and keywords.

Identified structs may be _opaque_. In this case, the body is unknown but the
structure type is considered _initialized_ and is valid in the IR.

#### Literal Structure Types

Literal structures are uniqued according to the list of elements they contain,
and can optionally be packed. The syntax for such structs is as follows.

```
llvm-literal-struct-type ::= `!llvm.struct<` `packed`? `(` type-list `)` `>`
type-list ::= <maybe empty comma-separated list of types with optional !llvm.>
```

Literal structs cannot be recursive, but can contain other structs. Therefore,
they must be constructed in a single step with the entire list of contained
elements provided.

#### Examples of Structure Types

```mlir
!llvm.struct<>                  // NOT allowed
!llvm.struct<()>                // empty, literal
!llvm.struct<(i32)>             // literal
!llvm.struct<(struct<(i32)>)>   // struct containing a struct
!llvm.struct<packed (i8, i32)>  // packed struct
!llvm.struct<"a">               // recursive reference, only allowed within
                                // another struct, NOT allowed at top level
!llvm.struct<"a", ptr<struct<"a">>>  // supported example of recursive reference
!llvm.struct<"a", ()>           // empty, named (necessary to differentiate from
                                // recursive reference)
!llvm.struct<"a", opaque>       // opaque, named
!llvm.struct<"a", (i32)>        // named
!llvm.struct<"a", packed (i8, i32)>  // named, packed
```

### Unsupported Types

LLVM IR `label` type does not have a counterpart in the LLVM dialect since, in
MLIR, blocks are not values and don't need a type.

## Operations

All operations in the LLVM IR dialect have a custom form in MLIR. The mnemonic
of an operation is that used in LLVM IR prefixed with "`llvm.`".

[include "Dialects/LLVMOps.md"]
