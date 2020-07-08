# 'llvm' Dialect

This dialect wraps the LLVM IR types and instructions into MLIR types and
operations. It provides several additional operations that are necessary to
cover for the differences in the IR structure (e.g., MLIR does not have `phi`
operations and LLVM IR does not have a `constant` operation).

In this document, we use "LLVM IR" to designate the
[intermediate representation of LLVM](https://llvm.org/docs/LangRef.html) and
"LLVM IR _dialect_" to refer to the MLIR dialect reflecting LLVM instructions
and types.

[TOC]

## Context and Module Association

The LLVM IR dialect object _contains_ an LLVM Context and an LLVM Module that it
uses to define, print, parse and manage LLVM IR types. These objects can be
obtained from the dialect object using `.getLLVMContext()` and
`getLLVMModule()`. All LLVM IR objects that interact with the LLVM IR dialect
must exist in the dialect's context.

## Types

The LLVM IR dialect defines a single MLIR type, `LLVM::LLVMType`, that can wrap
any existing LLVM IR type. Its syntax is as follows

```
type ::= `!llvm<"` llvm-canonical-type `">
llvm-canonical-type ::= <canonical textual representation defined by LLVM>
```

For example, one can use primitive types `!llvm.i32`, pointer types
`!llvm<"i8*">`, vector types `!llvm<"<4 x float>">` or structure types
`!llvm<"{i32, float}">`. The parsing and printing of the canonical form are
delegated to the LLVM assembly parser and printer.

LLVM IR dialect types contain an `llvm::Type*` object that can be obtained by
calling `.getUnderlyingType()` and used in LLVM API calls directly. These
objects are allocated within the LLVM context associated with the LLVM IR
dialect and may be linked to the properties of the associated LLVM module.

LLVM IR dialect type can be constructed from any `llvm::Type*` that is
associated with the LLVM context of the dialect. In this document, we use the
term "wrapped LLVM IR type" to refer to the LLVM IR dialect type containing a
specific LLVM IR type.

## Operations

All operations in the LLVM IR dialect have a custom form in MLIR. The mnemonic
of an operation is that used in LLVM IR prefixed with "`llvm.`".

### LLVM functions

MLIR functions are defined by an operation that is not built into the IR itself.
The LLVM IR dialect provides an `llvm.func` operation to define functions
compatible with LLVM IR. These functions have wrapped LLVM IR function type but
use MLIR syntax to express it. They are required to have exactly one result
type. LLVM function operation is intended to capture additional properties of
LLVM functions, such as linkage and calling convention, that may be modeled
differently by the built-in MLIR function.

```mlir
// The type of @bar is !llvm<"i64 (i64)">
llvm.func @bar(%arg0: !llvm.i64) -> !llvm.i64 {
  llvm.return %arg0 : !llvm.i64
}

// Type type of @foo is !llvm<"void (i64)">
// !llvm.void type is omitted
llvm.func @foo(%arg0: !llvm.i64) {
  llvm.return
}

// A function with `internal` linkage.
llvm.func internal @internal_func() {
  llvm.return
}

```

#### Attribute pass-through

An LLVM IR dialect function provides a mechanism to forward function-level
attributes to LLVM IR using the `passthrough` attribute. This is an array
attribute containing either string attributes or array attributes. In the former
case, the value of the string is interpreted as the name of LLVM IR function
attribute. In the latter case, the array is expected to contain exactly two
string attributes, the first corresponding to the name of LLVM IR function
attribute, and the second corresponding to its value. Note that even integer
LLVM IR function attributes have their value represented in the string form.

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

#### Linkage

An LLVM IR dialect function has a linkage attribute derived from LLVM IR
[linkage types](https://llvm.org/docs/LangRef.html#linkage-types). Linkage is
specified by the same keyword as in LLVM IR and is located between `llvm.func`
and the symbol name. If no linkage keyword is present, `external` linkage is
assumed by default.

### LLVM IR operations

The following operations are currently supported. The semantics of these
operations corresponds to the semantics of the similarly-named LLVM IR
instructions.

#### Integer binary arithmetic operations

Take two arguments of wrapped LLVM IR integer type, produce one value of the
same type.

-   `add`
-   `sub`
-   `mul`
-   `udiv`
-   `sdiv`
-   `urem`
-   `srem`

Examples:

```mlir
// Integer addition.
%0 = llvm.add %a, %b : !llvm.i32

// Unsigned integer division.
%1 = llvm.udiv %a, %b : !llvm.i32
```

#### Floating point binary arithmetic operations

Take two arguments of wrapped LLVM IR floating point type, produce one value of
the same type.

-   `fadd`
-   `fsub`
-   `fmul`
-   `fdiv`
-   `frem`

Examples:

```mlir
// Float addition.
%0 = llvm.fadd %a, %b : !llvm.float

// Float division.
%1 = llvm.fdiv %a, %b : !llvm.float
```

#### Memory-related operations

-   `<r> = alloca <size> x <type>`
-   `<r> = getelementptr <address>[<index> (, <index>)+]`
-   `<r> = load <address>`
-   `store <value>, <address>`

In these operations, `<size>` must be a value of wrapped LLVM IR integer type,
`<address>` must be a value of wrapped LLVM IR pointer type, and `<value>` must
be a value of wrapped LLVM IR type that corresponds to the pointer type of
`<address>`.

The `index` operands are integer values whose semantics is identical to the
non-pointer arguments of LLVM IR's `getelementptr`.

Examples:

```mlir
// Allocate an array of 4 floats on stack
%c4 = llvm.mlir.constant(4) : !llvm.i64
%0 = llvm.alloca %c4 x !llvm.float : (!llvm.i64) -> !llvm<"float*">

// Get the second element of the array (note 0-based indexing).
%c1 = llvm.mlir.constant(1) : !llvm.i64
%1 = llvm.getelementptr %0[%c1] : (!llvm<"float*">, !llvm.i64)
                                   -> !llvm<"float*">

// Store a constant into this element.
%cf = llvm.mlir.constant(42.0 : f32) : !llvm.float
llvm.store %cf, %1 : !llvm<"float*">

// Load the value from this element.
%3 = llvm.load %1 : !llvm<"float*">
```

#### Operations on values of aggregate type.

-   `<value> = extractvalue <struct>[<index> (, <index>)+]`
-   `<struct> = insertvalue <value>, <struct>[<index> (, <index>)+]`

In these operations, `<struct>` must be a value of wrapped LLVM IR structure
type and `<value>` must be a value that corresponds to one of the (nested)
structure element types.

Note the use of integer literals to designate subscripts, which is made possible
by `extractvalue` and `insertvalue` must have constant subscripts. Internally,
they are modeled as array attributes.

Examples:

```mlir
// Get the value third element of the second element of a structure.
%0 = llvm.extractvalue %s[1, 2] : !llvm<"{i32, {i1, i8, i16}">

// Insert the value to the third element of the second element of a structure.
// Note that this returns a new structure-typed value.
%1 = llvm.insertvalue %0, %s[1, 2] : !llvm<"{i32, {i1, i8, i16}">
```

#### Terminator operations.

Branch operations:

-   `br [<successor>(<operands>)]`
-   `cond_br <condition> [<true-successor>(<true-operands>),`
    `<false-successor>(<false-operands>)]`

In order to comply with MLIR design, branch operations in the LLVM IR dialect
pass arguments to basic blocks. Successors must be valid block MLIR identifiers
and operand lists for each of them must have the same types as the arguments of
the respective blocks. `<condition>` must be a wrapped LLVM IR `i1` type.

Since LLVM IR uses the name of the predecessor basic block to identify the
sources of a PHI node, it is invalid for two entries of the PHI node to indicate
different values coming from the same block. Therefore, `cond_br` in the LLVM IR
dialect disallows its successors to be the same block _if_ this block has
arguments.

Examples:

```mlir
// Branch without arguments.
^bb0:
  llvm.br ^bb0

// Branch and pass arguments.
^bb1(%arg: !llvm.i32):
  llvm.br ^bb1(%arg : !llvm.i32)

// Conditionally branch and pass arguments to one of the blocks.
llvm.cond_br %cond, ^bb0, %bb1(%arg : !llvm.i32)

// It's okay to use the same block without arguments, but probably useless.
llvm.cond_br %cond, ^bb0, ^bb0

// ERROR: Passing different arguments to the same block in a conditional branch.
llvm.cond_br %cond, ^bb1(%0 : !llvm.i32), ^bb1(%1 : !llvm.i32)

```

Call operations:

-   `<r> = call(<operands>)`
-   `call(<operands>)`

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect implements
this behavior by providing a variadic `call` operation for 0- and 1-result
functions. Even though MLIR supports multi-result functions, LLVM IR dialect
disallows them.

The `call` instruction supports both direct and indirect calls. Direct calls
start with a function name (`@`-prefixed) and indirect calls start with an SSA
value (`%`-prefixed). The direct callee, if present, is stored as a function
attribute `callee`. The trailing type of the instruction is always the MLIR
function type, which may be different from the indirect callee that has the
wrapped LLVM IR function type.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (!llvm.float)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (!llvm.float) -> ()

// Indirect call with an argument and without a result.
llvm.call %1(%0) : (!llvm.float) -> ()
```

#### Miscellaneous operations.

Integer comparisons: `icmp "predicate" <lhs>, <rhs>`. The following predicate
values are supported:

-   `eq` - equality comparison;
-   `ne` - inequality comparison;
-   `slt` - signed less-than comparison
-   `sle` - signed less-than-or-equal comparison
-   `sgt` - signed greater-than comparison
-   `sge` - signed greater-than-or-equal comparison
-   `ult` - unsigned less-than comparison
-   `ule` - unsigned less-than-or-equal comparison
-   `ugt` - unsigned greater-than comparison
-   `uge` - unsigned greater-than-or-equal comparison

Bitwise reinterpretation: `bitcast <value>`.

Selection: `select <condition>, <lhs>, <rhs>`.

### Auxiliary MLIR Operations for Constants and Globals

LLVM IR has broad support for first-class constants, which is not the case for
MLIR. Instead, constants are defined in MLIR as regular SSA values produced by
operations with specific traits. The LLVM dialect provides a set of operations
that model LLVM IR constants. These operations do not correspond to LLVM IR
instructions and are therefore prefixed with `llvm.mlir`.

Inline constants can be created by `llvm.mlir.constant`, which currently
supports integer, float, string or elements attributes (constant sturcts are not
currently supported). LLVM IR constant expressions are expected to be
constructed as sequences of regular operations on SSA values produced by
`llvm.mlir.constant`. Additionally, MLIR provides semantically-charged
operations `llvm.mlir.undef` and `llvm.mlir.null` for the corresponding
constants.

LLVM IR globals can be defined using `llvm.mlir.global` at the module level,
except for functions that are defined with `llvm.func`. Globals, both variables
and functions, can be accessed by taking their address with the
`llvm.mlir.addressof` operation, which produces a pointer to the named global,
unlike the `llvm.mlir.constant` that produces the value of the same type as the
constant.

#### `llvm.mlir.addressof`

Creates an SSA value containing a pointer to a global variable or constant
defined by `llvm.mlir.global`. The global value can be defined after its first
referenced. If the global value is a constant, storing into it is not allowed.

Examples:

```mlir
func @foo() {
  // Get the address of a global variable.
  %0 = llvm.mlir.addressof @const : !llvm<"i32*">

  // Use it as a regular pointer.
  %1 = llvm.load %0 : !llvm<"i32*">

  // Get the address of a function.
  %2 = llvm.mlir.addressof @foo : !llvm<"void ()*">

  // The function address can be used for indirect calls.
  llvm.call %2() : () -> ()
}

// Define the global.
llvm.mlir.global @const(42 : i32) : !llvm.i32
```

#### `llvm.mlir.constant`

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore, all
constants must be created as SSA values before being used in other operations.
`llvm.mlir.constant` creates such values for scalars and vectors. It has a
mandatory `value` attribute, which may be an integer, floating point attribute;
dense or sparse attribute containing integers or floats. The type of the
attribute is one of the corresponding MLIR standard types. It may be omitted for
`i64` and `f64` types that are implied. The operation produces a new SSA value
of the specified LLVM IR dialect type. The type of that value _must_ correspond
to the attribute type converted to LLVM IR.

Examples:

```mlir
// Integer constant, internal i32 is mandatory
%0 = llvm.mlir.constant(42 : i32) : !llvm.i32

// It's okay to omit i64.
%1 = llvm.mlir.constant(42) : !llvm.i64

// Floating point constant.
%2 = llvm.mlir.constant(42.0 : f32) : !llvm.float

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : !llvm<"<4 x float>">
```

#### `llvm.mlir.global`

Since MLIR allows for arbitrary operations to be present at the top level,
global variables are defined using the `llvm.mlir.global` operation. Both global
constants and variables can be defined, and the value may also be initialized in
both cases.

There are two forms of initialization syntax. Simple constants that can be
represented as MLIR attributes can be given in-line:

```mlir
llvm.mlir.global @variable(32.0 : f32) : !llvm.float
```

This initialization and type syntax is similar to `llvm.mlir.constant` and may
use two types: one for MLIR attribute and another for the LLVM value. These
types must be compatible.

More complex constants that cannot be represented as MLIR attributes can be
given in an initializer region:

```mlir
// This global is initialized with the equivalent of:
//   i32* getelementptr (i32* @g2, i32 2)
llvm.mlir.global constant @int_gep() : !llvm<"i32*"> {
  %0 = llvm.mlir.addressof @g2 : !llvm<"i32*">
  %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
  %2 = llvm.getelementptr %0[%1] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
  // The initializer region must end with `llvm.return`.
  llvm.return %2 : !llvm<"i32*">
}
```

Only one of the initializer attribute or initializer region may be provided.

`llvm.mlir.global` must appear at top-level of the enclosing module. It uses an
@-identifier for its value, which will be uniqued by the module with respect to
other @-identifiers in it.

Examples:

```mlir
// Global values use @-identifiers.
llvm.mlir.global constant @cst(42 : i32) : !llvm.i32

// Non-constant values must also be initialized.
llvm.mlir.global @variable(32.0 : f32) : !llvm.float

// Strings are expected to be of wrapped LLVM i8 array type and do not
// automatically include the trailing zero.
llvm.mlir.global @string("abc") : !llvm<"[3 x i8]">

// For strings globals, the trailing type may be omitted.
llvm.mlir.global constant @no_trailing_type("foo bar")

// A complex initializer is constructed with an initializer region.
llvm.mlir.global constant @int_gep() : !llvm<"i32*"> {
  %0 = llvm.mlir.addressof @g2 : !llvm<"i32*">
  %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
  %2 = llvm.getelementptr %0[%1] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
  llvm.return %2 : !llvm<"i32*">
}
```

Similarly to functions, globals have a linkage attribute. In the custom syntax,
this attribute is placed between `llvm.mlir.global` and the optional `constant`
keyword. If the attribute is omitted, `external` linkage is assumed by default.

Examples:

```mlir
// A constant with internal linkage will not participate in linking.
llvm.mlir.global internal constant @cst(42 : i32) : !llvm.i32

// By default, "external" linkage is assumed and the global participates in
// symbol resolution at link-time.
llvm.mlir.global @glob(0 : f32) : !llvm.float
```

#### `llvm.mlir.null`

Unlike LLVM IR, MLIR does not have first-class null pointers. They must be
explicitly created as SSA values using `llvm.mlir.null`. This operation has
operands or attributes, and returns a null value of a wrapped LLVM IR pointer
type.

Examples:

```mlir
// Null pointer to i8 value.
%0 = llvm.mlir.null : !llvm<"i8*">

// Null pointer to a function with signature void() value.
%1 = llvm.mlir.null : !llvm<"void()*">
```

#### `llvm.mlir.undef`

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.mlir.undef`. This operation has no
operands or attributes. It creates an undefined value of the specified LLVM IR
dialect type wrapping an LLVM IR structure type.

Example:

```mlir
// Create a structure with a 32-bit integer followed by a float.
%0 = llvm.mlir.undef : !llvm<"{i32, float}">
```
