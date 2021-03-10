# MLIR Language Reference

MLIR (Multi-Level IR) is a compiler intermediate representation with
similarities to traditional three-address SSA representations (like
[LLVM IR](http://llvm.org/docs/LangRef.html) or
[SIL](https://github.com/apple/swift/blob/master/docs/SIL.rst)), but which
introduces notions from polyhedral loop optimization as first-class concepts.
This hybrid design is optimized to represent, analyze, and transform high level
dataflow graphs as well as target-specific code generated for high performance
data parallel systems. Beyond its representational capabilities, its single
continuous design provides a framework to lower from dataflow graphs to
high-performance target-specific code.

This document defines and describes the key concepts in MLIR, and is intended
to be a dry reference document - the [rationale
documentation](Rationale/Rationale.md),
[glossary](../getting_started/Glossary.md), and other content are hosted
elsewhere.

MLIR is designed to be used in three different forms: a human-readable textual
form suitable for debugging, an in-memory form suitable for programmatic
transformations and analysis, and a compact serialized form suitable for
storage and transport. The different forms all describe the same semantic
content. This document describes the human-readable textual form.

[TOC]

## High-Level Structure

MLIR is fundamentally based on a graph-like data structure of nodes, called
*Operations*, and edges, called *Values*. Each Value is the result of exactly
one Operation or Block Argument, and has a *Value Type* defined by the [type
system](#type-system).  [Operations](#operations) are contained in
[Blocks](#blocks) and Blocks are contained in [Regions](#regions). Operations
are also ordered within their containing block and Blocks are ordered in their
containing region, although this order may or may not be semantically
meaningful in a given [kind of region](Interfaces.md#regionkindinterfaces)).
Operations may also contain regions, enabling hierarchical structures to be
represented.

Operations can represent many different concepts, from higher-level concepts
like function definitions, function calls, buffer allocations, view or slices
of buffers, and process creation, to lower-level concepts like
target-independent arithmetic, target-specific instructions, configuration
registers, and logic gates. These different concepts are represented by
different operations in MLIR and the set of operations usable in MLIR can be
arbitrarily extended.

MLIR also provides an extensible framework for transformations on operations,
using familiar concepts of compiler [Passes](Passes.md). Enabling an arbitrary
set of passes on an arbitrary set of operations results in a significant
scaling challenge, since each transformation must potentially take into
account the semantics of any operation. MLIR addresses this complexity by
allowing operation semantics to be described abstractly using
[Traits](Traits.md) and [Interfaces](Interfaces.md), enabling transformations
to operate on operations more generically.  Traits often describe verification
constraints on valid IR, enabling complex invariants to be captured and
checked. (see [Op vs
Operation](docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations))

One obvious application of MLIR is to represent an
[SSA-based](https://en.wikipedia.org/wiki/Static_single_assignment_form) IR,
like the LLVM core IR, with appropriate choice of Operation Types to define
[Modules](#module), [Functions](#functions), Branches, Allocations, and
verification constraints to ensure the SSA Dominance property. MLIR includes a
'standard' dialect which defines just such structures. However, MLIR is
intended to be general enough to represent other compiler-like data
structures, such as Abstract Syntax Trees in a language frontend, generated
instructions in a target-specific backend, or circuits in a High-Level
Synthesis tool.

Here's an example of an MLIR module:

```mlir
// Compute A*B using an implementation of multiply kernel and print the
// result using a TensorFlow op. The dimensions of A and B are partially
// known. The shapes are assumed to match.
func @mul(%A: tensor<100x?xf32>, %B: tensor<?x50xf32>) -> (tensor<100x50xf32>) {
  // Compute the inner dimension of %A using the dim operation.
  %n = dim %A, 1 : tensor<100x?xf32>

  // Allocate addressable "buffers" and copy tensors %A and %B into them.
  %A_m = alloc(%n) : memref<100x?xf32>
  tensor_store %A to %A_m : memref<100x?xf32>

  %B_m = alloc(%n) : memref<?x50xf32>
  tensor_store %B to %B_m : memref<?x50xf32>

  // Call function @multiply passing memrefs as arguments,
  // and getting returned the result of the multiplication.
  %C_m = call @multiply(%A_m, %B_m)
          : (memref<100x?xf32>, memref<?x50xf32>) -> (memref<100x50xf32>)

  dealloc %A_m : memref<100x?xf32>
  dealloc %B_m : memref<?x50xf32>

  // Load the buffer data into a higher level "tensor" value.
  %C = tensor_load %C_m : memref<100x50xf32>
  dealloc %C_m : memref<100x50xf32>

  // Call TensorFlow built-in function to print the result tensor.
  "tf.Print"(%C){message: "mul result"}
                  : (tensor<100x50xf32) -> (tensor<100x50xf32>)

  return %C : tensor<100x50xf32>
}

// A function that multiplies two memrefs and returns the result.
func @multiply(%A: memref<100x?xf32>, %B: memref<?x50xf32>)
          -> (memref<100x50xf32>)  {
  // Compute the inner dimension of %A.
  %n = dim %A, 1 : memref<100x?xf32>

  // Allocate memory for the multiplication result.
  %C = alloc() : memref<100x50xf32>

  // Multiplication loop nest.
  affine.for %i = 0 to 100 {
     affine.for %j = 0 to 50 {
        store 0 to %C[%i, %j] : memref<100x50xf32>
        affine.for %k = 0 to %n {
           %a_v  = load %A[%i, %k] : memref<100x?xf32>
           %b_v  = load %B[%k, %j] : memref<?x50xf32>
           %prod = mulf %a_v, %b_v : f32
           %c_v  = load %C[%i, %j] : memref<100x50xf32>
           %sum  = addf %c_v, %prod : f32
           store %sum, %C[%i, %j] : memref<100x50xf32>
        }
     }
  }
  return %C : memref<100x50xf32>
}
```

## Notation

MLIR has a simple and unambiguous grammar, allowing it to reliably round-trip
through a textual form. This is important for development of the compiler -
e.g.  for understanding the state of code as it is being transformed and
writing test cases.

This document describes the grammar using
[Extended Backus-Naur Form (EBNF)](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form).

This is the EBNF grammar used in this document, presented in yellow boxes.

```
alternation ::= expr0 | expr1 | expr2  // Either expr0 or expr1 or expr2.
sequence    ::= expr0 expr1 expr2      // Sequence of expr0 expr1 expr2.
repetition0 ::= expr*  // 0 or more occurrences.
repetition1 ::= expr+  // 1 or more occurrences.
optionality ::= expr?  // 0 or 1 occurrence.
grouping    ::= (expr) // Everything inside parens is grouped together.
literal     ::= `abcd` // Matches the literal `abcd`.
```

Code examples are presented in blue boxes.

```mlir
// This is an example use of the grammar above:
// This matches things like: ba, bana, boma, banana, banoma, bomana...
example ::= `b` (`an` | `om`)* `a`
```

### Common syntax

The following core grammar productions are used in this document:

```
// TODO: Clarify the split between lexing (tokens) and parsing (grammar).
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

integer-literal ::= decimal-literal | hexadecimal-literal
decimal-literal ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
string-literal  ::= `"` [^"\n\f\v\r]* `"`   TODO: define escaping rules
```

Not listed here, but MLIR does support comments. They use standard BCPL syntax,
starting with a `//` and going until the end of the line.

### Identifiers and keywords

Syntax:

```
// Identifiers
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal)
value-id-list ::= value-id (`,` value-id)*

// Uses of value, e.g. in an operand list to an operation.
value-use ::= value-id
value-use-list ::= value-use (`,` value-use)*
```

Identifiers name entities such as values, types and functions, and are
chosen by the writer of MLIR code. Identifiers may be descriptive (e.g.
`%batch_size`, `@matmul`), or may be non-descriptive when they are
auto-generated (e.g. `%23`, `@func42`). Identifier names for values may be
used in an MLIR text file but are not persisted as part of the IR - the printer
will give them anonymous names like `%42`.

MLIR guarantees identifiers never collide with keywords by prefixing identifiers
with a sigil (e.g. `%`, `#`, `@`, `^`, `!`). In certain unambiguous contexts
(e.g. affine expressions), identifiers are not prefixed, for brevity. New
keywords may be added to future versions of MLIR without danger of collision
with existing identifiers.

Value identifiers are only [in scope](#value-scoping) for the (nested)
region in which they are defined and cannot be accessed or referenced
outside of that region. Argument identifiers in mapping functions are
in scope for the mapping body. Particular operations may further limit
which identifiers are in scope in their regions. For instance, the
scope of values in a region with [SSA control flow
semantics](#control-flow-and-ssacfg-regions) is constrained according
to the standard definition of [SSA
dominance](https://en.wikipedia.org/wiki/Dominator_\(graph_theory\)). Another
example is the [IsolatedFromAbove trait](Traits.md#isolatedfromabove),
which restricts directly accessing values defined in containing
regions.

Function identifiers and mapping identifiers are associated with
[Symbols](SymbolsAndSymbolTables) and have scoping rules dependent on
symbol attributes.

## Dialects

Dialects are the mechanism by which to engage with and extend the MLIR
ecosystem. They allow for defining new [operations](#operations), as well as
[attributes](#attributes) and [types](#type-system). Each dialect is given a
unique `namespace` that is prefixed to each defined attribute/operation/type.
For example, the [Affine dialect](Dialects/Affine.md) defines the namespace:
`affine`.

MLIR allows for multiple dialects, even those outside of the main tree, to
co-exist together within one module. Dialects are produced and consumed by
certain passes. MLIR provides a [framework](DialectConversion.md) to convert
between, and within, different dialects.

A few of the dialects supported by MLIR:

*   [Affine dialect](Dialects/Affine.md)
*   [GPU dialect](Dialects/GPU.md)
*   [LLVM dialect](Dialects/LLVM.md)
*   [SPIR-V dialect](Dialects/SPIR-V.md)
*   [Standard dialect](Dialects/Standard.md)
*   [Vector dialect](Dialects/Vector.md)

### Target specific operations

Dialects provide a modular way in which targets can expose target-specific
operations directly through to MLIR. As an example, some targets go through
LLVM. LLVM has a rich set of intrinsics for certain target-independent
operations (e.g. addition with overflow check) as well as providing access to
target-specific operations for the targets it supports (e.g. vector
permutation operations). LLVM intrinsics in MLIR are represented via
operations that start with an "llvm." name.

Example:

```mlir
// LLVM: %x = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %a, i16 %b)
%x:2 = "llvm.sadd.with.overflow.i16"(%a, %b) : (i16, i16) -> (i16, i1)
```

These operations only work when targeting LLVM as a backend (e.g. for CPUs and
GPUs), and are required to align with the LLVM definition of these intrinsics.

## Operations

Syntax:

```
operation         ::= op-result-list? (generic-operation | custom-operation)
                      trailing-location?
generic-operation ::= string-literal `(` value-use-list? `)`  successor-list?
                      (`(` region-list `)`)? dictionary-attribute? `:` function-type
custom-operation  ::= bare-id custom-operation-format
op-result-list    ::= op-result (`,` op-result)* `=`
op-result         ::= value-id (`:` integer-literal)
successor-list    ::= successor (`,` successor)*
successor         ::= caret-id (`:` bb-arg-list)?
region-list       ::= region (`,` region)*
trailing-location ::= (`loc` `(` location `)`)?
```

MLIR introduces a uniform concept called _operations_ to enable describing
many different levels of abstractions and computations. Operations in MLIR are
fully extensible (there is no fixed list of operations) and have
application-specific semantics. For example, MLIR supports [target-independent
operations](Dialects/Standard.md#memory-operations), [affine
operations](Dialects/Affine.md), and [target-specific machine
operations](#target-specific-operations).

The internal representation of an operation is simple: an operation is
identified by a unique string (e.g. `dim`, `tf.Conv2d`, `x86.repmovsb`,
`ppc.eieio`, etc), can return zero or more results, take zero or more
operands, has a dictionary of [attributes](#attributes), has zero or more
successors, and zero or more enclosed [regions](#regions). The generic printing
form includes all these elements literally, with a function type to indicate the
types of the results and operands.

Example:

```mlir
// An operation that produces two results.
// The results of %result can be accessed via the <name> `#` <opNo> syntax.
%result:2 = "foo_div"() : () -> (f32, i32)

// Pretty form that defines a unique name for each result.
%foo, %bar = "foo_div"() : () -> (f32, i32)

// Invoke a TensorFlow function called tf.scramble with two inputs
// and an attribute "fruit".
%2 = "tf.scramble"(%result#0, %bar) {fruit = "banana"} : (f32, i32) -> f32
```

In addition to the basic syntax above, dialects may register known operations.
This allows those dialects to support _custom assembly form_ for parsing and
printing operations. In the operation sets listed below, we show both forms.

### Terminator Operations

These are a special category of operations that *must* terminate a block, e.g.
[branches](Dialects/Standard.md#terminator-operations). These operations may
also have a list of successors ([blocks](#blocks) and their arguments).

Example:

```mlir
// Branch to ^bb1 or ^bb2 depending on the condition %cond.
// Pass value %v to ^bb2, but not to ^bb1.
"cond_br"(%cond)[^bb1, ^bb2(%v : index)] : (i1) -> ()
```

### Module

```
module ::= `module` symbol-ref-id? (`attributes` dictionary-attribute)? region
```

An MLIR Module represents a top-level container operation. It contains a single
[SSACFG region](#control-flow-and-ssacfg-regions) containing a single block
which can contain any operations. Operations within this region cannot
implicitly capture values defined outside the module, i.e. Modules are
[IsolatedFromAbove](Traits.md#isolatedfromabove). Modules have an optional
[symbol name](SymbolsAndSymbolTables.md) which can be used to refer to them in
operations.

### Functions

An MLIR Function is an operation with a name containing a single [SSACFG
region](#control-flow-and-ssacfg-regions).  Operations within this region
cannot implicitly capture values defined outside of the function,
i.e. Functions are [IsolatedFromAbove](Traits.md#isolatedfromabove).  All
external references must use function arguments or attributes that establish a
symbolic connection (e.g. symbols referenced by name via a string attribute
like [SymbolRefAttr](#symbol-reference-attribute)):

```
function ::= `func` function-signature function-attributes? function-body?

function-signature ::= symbol-ref-id `(` argument-list `)`
                       (`->` function-result-list)?

argument-list ::= (named-argument (`,` named-argument)*) | /*empty*/
argument-list ::= (type dictionary-attribute? (`,` type dictionary-attribute?)*)
                | /*empty*/
named-argument ::= value-id `:` type dictionary-attribute?

function-result-list ::= function-result-list-parens
                       | non-function-type
function-result-list-parens ::= `(` `)`
                              | `(` function-result-list-no-parens `)`
function-result-list-no-parens ::= function-result (`,` function-result)*
function-result ::= type dictionary-attribute?

function-attributes ::= `attributes` dictionary-attribute
function-body ::= region
```

An external function declaration (used when referring to a function declared
in some other module) has no body. While the MLIR textual form provides a nice
inline syntax for function arguments, they are internally represented as
"block arguments" to the first block in the region.

Only dialect attribute names may be specified in the attribute dictionaries
for function arguments, results, or the function itself.

Examples:

```mlir
// External function definitions.
func @abort()
func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
func @count(%x: i64) -> (i64, i64)
  attributes {fruit: "banana"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
func @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
func @example_fn_attr() attributes {dialectName.attrName = false}
```

## Blocks

Syntax:

```
block           ::= block-label operation+
block-label     ::= block-id block-arg-list? `:`
block-id        ::= caret-id
caret-id        ::= `^` suffix-id
value-id-and-type ::= value-id `:` type

// Non-empty list of names and types.
value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

block-arg-list ::= `(` value-id-and-type-list? `)`
```

A *Block* is an ordered list of operations, concluding with a single
[terminator operation](#terminator-operations). In [SSACFG
regions](#control-flow-and-ssacfg-regions), each block represents a compiler
[basic block](https://en.wikipedia.org/wiki/Basic_block) where instructions
inside the block are executed in order and terminator operations implement
control flow branches between basic blocks.

Blocks in MLIR take a list of block arguments, notated in a function-like
way. Block arguments are bound to values specified by the semantics of
individual operations. Block arguments of the entry block of a region are also
arguments to the region and the values bound to these arguments are determined
by the semantics of the containing operation. Block arguments of other blocks
are determined by the semantics of terminator operations, e.g. Branches, which
have the block as a successor. In regions with [control
flow](#control-flow-and-ssacfg-regions), MLIR leverages this structure to
implicitly represent the passage of control-flow dependent values without the
complex nuances of PHI nodes in traditional SSA representations. Note that
values which are not control-flow dependent can be referenced directly and do
not need to be passed through block arguments.

Here is a simple example function showing branches, returns, and block
arguments:

```mlir
func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cond_br %cond, ^bb1, ^bb2

^bb1:
  br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  %b = addi %a, %a : i64
  br ^bb3(%b: i64)    // Branch passes %b as the argument

// ^bb3 receives an argument, named %c, from predecessors
// and passes it on to bb4 along with %a. %a is referenced
// directly from its defining operation and is not passed through
// an argument of ^bb3.
^bb3(%c: i64):
  br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = addi %d, %e : i64
  return %0 : i64   // Return is also a terminator.
}
```

**Context:** The "block argument" representation eliminates a number
of special cases from the IR compared to traditional "PHI nodes are
operations" SSA IRs (like LLVM). For example, the [parallel copy
semantics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.524.5461&rep=rep1&type=pdf)
of SSA is immediately apparent, and function arguments are no longer a
special case: they become arguments to the entry block [[more
rationale](Rationale/Rationale.md#block-arguments-vs-phi-nodes)]. Blocks
are also a fundamental concept that cannot be represented by
operations because values defined in an operation cannot be accessed
outside the operation.

## Regions

### Definition

A region is an ordered list of MLIR [Blocks](#blocks). The semantics within a
region is not imposed by the IR. Instead, the containing operation defines the
semantics of the regions it contains. MLIR currently defines two kinds of
regions: [SSACFG regions](#control-flow-and-ssacfg-regions), which describe
control flow between blocks, and [Graph regions](#graph-regions), which do not
require control flow between block. The kinds of regions within an operation
are described using the
[RegionKindInterface](Interfaces.md#regionkindinterfaces).

Regions do not have a name or an address, only the blocks contained in a
region do. Regions must be contained within operations and have no type or
attributes. The first block in the region is a special block called the 'entry
block'. The arguments to the entry block are also the arguments of the region
itself. The entry block cannot be listed as a successor of any other
block. The syntax for a region is as follows:

```
region ::= `{` block* `}`
```

A function body is an example of a region: it consists of a CFG of blocks and
has additional semantic restrictions that other types of regions may not have.
For example, in a function body, block terminators must either branch to a
different block, or return from a function where the types of the `return`
arguments must match the result types of the function signature.  Similarly,
the function arguments must match the types and count of the region arguments.
In general, operations with regions can define these correspondances
arbitrarily.

### Value Scoping

Regions provide hierarchical encapsulation of programs: it is impossible to
reference, i.e. branch to, a block which is not in the same region as the
source of the reference, i.e. a terminator operation. Similarly, regions
provides a natural scoping for value visibility: values defined in a region
don't escape to the enclosing region, if any. By default, operations inside a
region can reference values defined outside of the region whenever it would
have been legal for operands of the enclosing operation to reference those
values, but this can be restricted using traits, such as
[OpTrait::IsolatedFromAbove](Traits.md#isolatedfromabove), or a custom
verifier.

Example:

```mlir
  "any_op"(%a) ({ // if %a is in-scope in the containing region...
	 // then %a is in-scope here too.
    %new_value = "another_op"(%a) : (i64) -> (i64)
  }) : (i64) -> (i64)
```

MLIR defines a generalized 'hierarchical dominance' concept that operates
across hierarchy and defines whether a value is 'in scope' and can be used by
a particular operation. Whether a value can be used by another operation in
the same region is defined by the kind of region. A value defined in a region
can be used by an operation which has a parent in the same region, if and only
if the parent could use the value. A value defined by an argument to a region
can always be used by any operation deeply contained in the region. A value
defined in a region can never be used outside of the region.

### Control Flow and SSACFG Regions

In MLIR, control flow semantics of a region is indicated by
[RegionKind::SSACFG](Interfaces.md#regionkindinterfaces).  Informally, these
regions support semantics where operations in a region 'execute
sequentially'. Before an operation executes, its operands have well-defined
values. After an operation executes, the operands have the same values and
results also have well-defined values. After an operation executes, the next
operation in the block executes until the operation is the terminator operation
at the end of a block, in which case some other operation will execute. The
determination of the next instruction to execute is the 'passing of control
flow'.

In general, when control flow is passed to an operation, MLIR does not
restrict when control flow enters or exits the regions contained in that
operation. However, when control flow enters a region, it always begins in the
first block of the region, called the *entry* block.  Terminator operations
ending each block represent control flow by explicitly specifying the
successor blocks of the block. Control flow can only pass to one of the
specified successor blocks as in a `branch` operation, or back to the
containing operation as in a `return` operation. Terminator operations without
successors can only pass control back to the containing operation. Within
these restrictions, the particular semantics of terminator operations is
determined by the specific dialect operations involved. Blocks (other than the
entry block) that are not listed as a successor of a terminator operation are
defined to be unreachable and can be removed without affecting the semantics
of the containing operation.

Although control flow always enters a region through the entry block, control
flow may exit a region through any block with an appropriate terminator. The
standard dialect leverages this capability to define operations with
Single-Entry-Multiple-Exit (SEME) regions, possibly flowing through different
blocks in the region and exiting through any block with a `return`
operation. This behavior is similar to that of a function body in most
programming languages. In addition, control flow may also not reach the end of
a block or region, for example if a function call does not return.

Example:

```mlir
func @accelerator_compute(i64, i1) -> i64 { // An SSACFG region
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cond_br %cond, ^bb1, ^bb2

^bb1:
  // This def for %value does not dominate ^bb2
  %value = "op.convert"(%a) : (i64) -> i64
  br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  accelerator.launch() { // An SSACFG region
    ^bb0:
      // Region of code nested under "accelerator.launch", it can reference %a but
      // not %value.
      %new_value = "accelerator.do_something"(%a) : (i64) -> ()
  }
  // %new_value cannot be referenced outside of the region

^bb3:
  ...
}
```

#### Operations with Multiple Regions

An operation containing multiple regions also completely determines the
semantics of those regions. In particular, when control flow is passed to an
operation, it may transfer control flow to any contained region. When control
flow exits a region and is returned to the containing operation, the
containing operation may pass control flow to any region in the same
operation. An operation may also pass control flow to multiple contained
regions concurrently. An operation may also pass control flow into regions
that were specified in other operations, in particular those that defined the
values or symbols the given operation uses as in a call operation. This
passage of control is generally independent of passage of control flow through
the basic blocks of the containing region.

#### Closure

Regions allow defining an operation that creates a closure, for example by
“boxing” the body of the region into a value they produce. It remains up to the
operation to define its semantics. Note that if an operation triggers
asynchronous execution of the region, it is under the responsibility of the
operation caller to wait for the region to be executed guaranteeing that any
directly used values remain live.

### Graph Regions

In MLIR, graph-like semantics in a region is indicated by
[RegionKind::Graph](Interfaces.md#regionkindinterfaces). Graph regions are
appropriate for concurrent semantics without control flow, or for modeling
generic directed graph data structures. Graph regions are appropriate for
representing cyclic relationships between coupled values where there is no
fundamental order to the relationships. For instance, operations in a graph
region may represent independent threads of control with values representing
streams of data. As usual in MLIR, the particular semantics of a region is
completely determined by its containing operation. Graph regions may only
contain a single basic block (the entry block).

**Rationale:** Currently graph regions are arbitrarily limited to a single
basic block, although there is no particular semantic reason for this
limitation. This limitation has been added to make it easier to stabilize the
pass infrastructure and commonly used passes for processing graph regions to
properly handle feedback loops. Multi-block regions may be allowed in the
future if use cases that require it arise.

In graph regions, MLIR operations naturally represent nodes, while each MLIR
value represents a multi-edge connecting a single source node and multiple
destination nodes. All values defined in the region as results of operations
are in scope within the region and can be accessed by any other operation in
the region. In graph regions, the order of operations within a block and the
order of blocks in a region is not semantically meaningful and non-terminator
operations may be freely reordered, for instance, by canonicalization. Other
kinds of graphs, such as graphs with multiple source nodes and multiple
destination nodes, can also be represented by representing graph edges as MLIR
operations.

Note that cycles can occur within a single block in a graph region, or between
basic blocks.

```mlir
"test.graph_region"() ({ // A Graph region
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 allowed here
  %2 = "test.ssacfg_region"() ({
	 %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK: %1, %2, %3, %4 all defined in the containing region
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: %4 allowed here
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
```

### Arguments and Results

The arguments of the first block of a region are treated as arguments of the
region. The source of these arguments is defined by the semantics of the parent
operation. They may correspond to some of the values the operation itself uses.

Regions produce a (possibly empty) list of values. The operation semantics
defines the relation between the region results and the operation results.

## Type System

Each value in MLIR has a type defined by the type system below. There are a
number of primitive types (like integers) and also aggregate types for tensors
and memory buffers. MLIR [builtin types](#builtin-types) do not include
structures, arrays, or dictionaries.

MLIR has an open type system (i.e. there is no fixed list of types), and types
may have application-specific semantics. For example, MLIR supports a set of
[dialect types](#dialect-types).

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
```

### Type Aliases

```
type-alias-def ::= '!' alias-name '=' 'type' type
type-alias ::= '!' alias-name
```

MLIR supports defining named aliases for types. A type alias is an identifier
that can be used in the place of the type that it defines. These aliases *must*
be defined before their uses. Alias names may not contain a '.', since those
names are reserved for [dialect types](#dialect-types).

Example:

```mlir
!avx_m128 = type vector<4 x f32>

// Using the original type.
"foo"(%x) : vector<4 x f32> -> ()

// Using the type alias.
"foo"(%x) : !avx_m128 -> ()
```

### Dialect Types

Similarly to operations, dialects may define custom extensions to the type
system.

```
dialect-namespace ::= bare-id

opaque-dialect-item ::= dialect-namespace '<' string-literal '>'

pretty-dialect-item ::= dialect-namespace '.' pretty-dialect-item-lead-ident
                                              pretty-dialect-item-body?

pretty-dialect-item-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'
pretty-dialect-item-body ::= '<' pretty-dialect-item-contents+ '>'
pretty-dialect-item-contents ::= pretty-dialect-item-body
                              | '(' pretty-dialect-item-contents+ ')'
                              | '[' pretty-dialect-item-contents+ ']'
                              | '{' pretty-dialect-item-contents+ '}'
                              | '[^[<({>\])}\0]+'

dialect-type ::= '!' opaque-dialect-item
dialect-type ::= '!' pretty-dialect-item
```

Dialect types can be specified in a verbose form, e.g. like this:

```mlir
// LLVM type that wraps around llvm IR types.
!llvm<"i32*">

// Tensor flow string type.
!tf.string

// Complex type
!foo<"something<abcd>">

// Even more complex type
!foo<"something<a%%123^^^>>>">
```

Dialect types that are simple enough can use the pretty format, which is a
lighter weight syntax that is equivalent to the above forms:

```mlir
// Tensor flow string type.
!tf.string

// Complex type
!foo.something<abcd>
```

Sufficiently complex dialect types are required to use the verbose form for
generality. For example, the more complex type shown above wouldn't be valid in
the lighter syntax: `!foo.something<a%%123^^^>>>` because it contains characters
that are not allowed in the lighter syntax, as well as unbalanced `<>`
characters.

See [here](Tutorials/DefiningAttributesAndTypes.md) to learn how to define dialect types.

### Builtin Types

Builtin types are a core set of [dialect types](#dialect-types) that are defined
in a builtin dialect and thus available to all users of MLIR.

```
builtin-type ::=      complex-type
                    | float-type
                    | function-type
                    | index-type
                    | integer-type
                    | memref-type
                    | none-type
                    | tensor-type
                    | tuple-type
                    | vector-type
```

#### Complex Type

Syntax:

```
complex-type ::= `complex` `<` type `>`
```

The value of `complex` type represents a complex number with a parameterized
element type, which is composed of a real and imaginary value of that element
type. The element must be a floating point or integer scalar type.

Examples:

```mlir
complex<f32>
complex<i32>
```

#### Floating Point Types

Syntax:

```
// Floating point.
float-type ::= `f16` | `bf16` | `f32` | `f64` | `f80` | `f128`
```

MLIR supports float types of certain widths that are widely used as indicated
above.

#### Function Type

Syntax:

```
// MLIR functions can return multiple values.
function-result-type ::= type-list-parens
                       | non-function-type

function-type ::= type-list-parens `->` function-result-type
```

MLIR supports first-class functions: for example, the
[`constant` operation](Dialects/Standard.md#stdconstant-constantop) produces the
address of a function as a value. This value may be passed to and
returned from functions, merged across control flow boundaries with
[block arguments](#blocks), and called with the
[`call_indirect` operation](Dialects/Standard.md#call-indirect-operation).

Function types are also used to indicate the arguments and results of
[operations](#operations).

#### Index Type

Syntax:

```
// Target word-sized integer.
index-type ::= `index`
```

The `index` type is a signless integer whose size is equal to the natural
machine word of the target
([rationale](Rationale/Rationale.md#integer-signedness-semantics)) and is used
by the affine constructs in MLIR. Unlike fixed-size integers, it cannot be used
as an element of vector
([rationale](Rationale/Rationale.md#index-type-disallowed-in-vector-types)).

**Rationale:** integers of platform-specific bit widths are practical to express
sizes, dimensionalities and subscripts.

#### Integer Type

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

MLIR supports arbitrary precision integer types. Integer types have a designated
width and may have signedness semantics.

**Rationale:** low precision integers (like `i2`, `i4` etc) are useful for
low-precision inference chips, and arbitrary precision integers are useful for
hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller than a 16
bit one).

TODO: Need to decide on a representation for quantized integers
([initial thoughts](Rationale/Rationale.md#quantized-integer-operations)).

#### Memref Type

Syntax:

```
memref-type ::= ranked-memref-type | unranked-memref-type

ranked-memref-type ::= `memref` `<` dimension-list-ranked type
                      (`,` layout-specification)? (`,` memory-space)? `>`

unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`

stride-list ::= `[` (dimension (`,` dimension)*)? `]`
strided-layout ::= `offset:` dimension `,` `strides: ` stride-list
semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
layout-specification ::= semi-affine-map-composition | strided-layout
memory-space ::= integer-literal /* | TODO: address-space-id */
```

A `memref` type is a reference to a region of memory (similar to a buffer
pointer, but more powerful). The buffer pointed to by a memref can be allocated,
aliased and deallocated. A memref can be used to read and write data from/to the
memory region which it references. Memref types use the same shape specifier as
tensor types. Note that `memref<f32>`, `memref<0 x f32>`, `memref<1 x 0 x f32>`,
and `memref<0 x 1 x f32>` are all different types.

A `memref` is allowed to have an unknown rank (e.g. `memref<*xf32>`). The
purpose of unranked memrefs is to allow external library functions to receive
memref arguments of any rank without versioning the functions based on the rank.
Other uses of this type are disallowed or will have undefined behavior.

##### Codegen of Unranked Memref

Using unranked memref in codegen besides the case mentioned above is highly
discouraged. Codegen is concerned with generating loop nests and specialized
instructions for high-performance, unranked memref is concerned with hiding the
rank and thus, the number of enclosing loops required to iterate over the data.
However, if there is a need to code-gen unranked memref, one possible path is to
cast into a static ranked type based on the dynamic rank. Another possible path
is to emit a single while loop conditioned on a linear index and perform
delinearization of the linear index to a dynamic array containing the (unranked)
indices. While this is possible, it is expected to not be a good idea to perform
this during codegen as the cost of the translations is expected to be
prohibitive and optimizations at this level are not expected to be worthwhile.
If expressiveness is the main concern, irrespective of performance, passing
unranked memrefs to an external C++ library and implementing rank-agnostic logic
there is expected to be significantly simpler.

Unranked memrefs may provide expressiveness gains in the future and help bridge
the gap with unranked tensors. Unranked memrefs will not be expected to be
exposed to codegen but one may query the rank of an unranked memref (a special
op will be needed for this purpose) and perform a switch and cast to a ranked
memref as a prerequisite to codegen.

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
[semi-affine map](Dialects/Affine.md#semi-affine-maps). Additionally, syntactic
sugar is supported to make certain layout specifications more intuitive to read.
For the moment, a `memref` supports parsing a strided form which is converted to
a semi-affine map automatically.

The memory space of a memref is specified by a target-specific attribute.
It might be an integer value, string, dictionary or custom dialect attribute.
The empty memory space (attribute is None) is target specific.

The notionally dynamic value of a memref value includes the address of the
buffer allocated, as well as the symbols referred to by the shape, layout map,
and index maps.

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

// A 2-d dynamic shaped memref that also has a dynamically sized tiled layout.
// The memref index space is of size %M x %N, while %B1 and %B2 bind to the
// symbols s0, s1 respectively of the layout map #tiled_dynamic. Data tiles of
// size %B1 x %B2 in the logical space will be stored contiguously in memory.
// The allocation size will be (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2
// f32 elements.
%T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

// A memref that has a two-element padding at either end. The allocation size
// will fit 16 * 64 float elements of data.
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

##### Index Space

A memref dimension list defines an index space within which the memref can be
indexed to access data.

##### Index

Data is accessed through a memref type using a multidimensional index into the
multidimensional index space defined by the memref's dimension list.

Examples

```mlir
// Allocates a memref with 2D index space:
//   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
%A = alloc() : memref<16x32xf32, #imapA>

// Loads data from memref '%A' using a 2D index: (%i, %j)
%v = load %A[%i, %j] : memref<16x32xf32, #imapA>
```

##### Index Map

An index map is a one-to-one
[semi-affine map](Dialects/Affine.md#semi-affine-maps) that transforms a
multidimensional index from one index space to another. For example, the
following figure shows an index map which maps a 2-dimensional index from a 2x2
index space to a 3x3 index space, using symbols `S0` and `S1` as offsets.

![Index Map Example](/includes/img/index-map.svg)

The number of domain dimensions and range dimensions of an index map can be
different, but must match the number of dimensions of the input and output index
spaces on which the map operates. The index space is always non-negative and
integral. In addition, an index map must specify the size of each of its range
dimensions onto which it maps. Index map symbols must be listed in order with
symbols for dynamic dimension sizes first, followed by other required symbols.

##### Layout Map

A layout map is a [semi-affine map](Dialects/Affine.md#semi-affine-maps) which
encodes logical to physical index space mapping, by mapping input dimensions to
their ordering from most-major (slowest varying) to most-minor (fastest
varying). Therefore, an identity layout map corresponds to a row-major layout.
Identity layout maps do not contribute to the MemRef type identification and are
discarded on construction. That is, a type with an explicit identity map is
`memref<?x?xf32, (i,j)->(i,j)>` is strictly the same as the one without layout
maps, `memref<?x?xf32>`.

Layout map examples:

```mlir
// MxN matrix stored in row major layout in memory:
#layout_map_row_major = (i, j) -> (i, j)

// MxN matrix stored in column major layout in memory:
#layout_map_col_major = (i, j) -> (j, i)

// MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
#layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
```

##### Affine Map Composition

A memref specifies a semi-affine map composition as part of its type. A
semi-affine map composition is a composition of semi-affine maps beginning with
zero or more index maps, and ending with a layout map. The composition must be
conformant: the number of dimensions of the range of one map, must match the
number of dimensions of the domain of the next map in the composition.

The semi-affine map composition specified in the memref type, maps from accesses
used to index the memref in load/store operations to other index spaces (i.e.
logical to physical index mapping). Each of the
[semi-affine maps](Dialects/Affine.md) and thus its composition is required to
be one-to-one.

The semi-affine map composition can be used in dependence analysis, memory
access pattern analysis, and for performance optimizations like vectorization,
copy elision and in-place updates. If an affine map composition is not specified
for the memref, the identity affine map is assumed.

##### Strided MemRef

A memref may specify strides as part of its type. A stride specification is a
list of integer values that are either static or `?` (dynamic case). Strides
encode the distance, in number of elements, in (linear) memory between
successive entries along a particular dimension. A stride specification is
syntactic sugar for an equivalent strided memref representation using
semi-affine maps. For example, `memref<42x16xf32, offset: 33, strides: [1, 64]>`
specifies a non-contiguous memory region of `42` by `16` `f32` elements such
that:

1.  the minimal size of the enclosing memory region must be `33 + 42 * 1 + 16 *
    64 = 1066` elements;
2.  the address calculation for accessing element `(i, j)` computes `33 + i +
    64 * j`
3.  the distance between two consecutive elements along the inner dimension is
    `1` element and the distance between two consecutive elements along the
    outer dimension is `64` elements.

This corresponds to a column major view of the memory region and is internally
represented as the type `memref<42x16xf32, (i, j) -> (33 + i + 64 * j)>`.

The specification of strides must not alias: given an n-D strided memref,
indices `(i1, ..., in)` and `(j1, ..., jn)` may not refer to the same memory
address unless `i1 == j1, ..., in == jn`.

Strided memrefs represent a view abstraction over preallocated data. They are
constructed with special ops, yet to be introduced. Strided memrefs are a
special subclass of memrefs with generic semi-affine map and correspond to a
normalized memref descriptor when lowering to LLVM.

#### None Type

Syntax:

```
none-type ::= `none`
```

The `none` type is a unit type, i.e. a type with exactly one possible value,
where its value does not have a defined dynamic representation.

#### Tensor Type

Syntax:

```
tensor-type ::= `tensor` `<` dimension-list type `>`

dimension-list ::= dimension-list-ranked | (`*` `x`)
dimension-list-ranked ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
```

Values with tensor type represents aggregate N-dimensional data values, and
have a known element type. It may have an unknown rank (indicated by `*`) or may
have a fixed rank with a list of dimensions. Each dimension may be a static
non-negative decimal constant or be dynamically determined (indicated by `?`).

The runtime representation of the MLIR tensor type is intentionally abstracted -
you cannot control layout or get a pointer to the data. For low level buffer
access, MLIR has a [`memref` type](#memref-type). This abstracted runtime
representation holds both the tensor data values as well as information about
the (potentially dynamic) shape of the tensor. The
[`dim` operation](Dialects/Standard.md#dim-operation) returns the size of a
dimension from a value of tensor type.

Note: hexadecimal integer literals are not allowed in tensor type declarations
to avoid confusion between `0xf32` and `0 x f32`. Zero sizes are allowed in
tensors and treated as other sizes, e.g., `tensor<0 x 1 x i32>` and `tensor<1 x
0 x i32>` are different types. Since zero sizes are not allowed in some other
types, such tensors should be optimized away before lowering tensors to vectors.

Examples:

```mlir
// Tensor with unknown rank.
tensor<* x f32>

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
```

#### Tuple Type

Syntax:

```
tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
```

The value of `tuple` type represents a fixed-size collection of elements, where
each element may be of a different type.

**Rationale:** Though this type is first class in the type system, MLIR provides
no standard operations for operating on `tuple` types
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

#### Vector Type

Syntax:

```
vector-type ::= `vector` `<` static-dimension-list vector-element-type `>`
vector-element-type ::= float-type | integer-type

static-dimension-list ::= (decimal-literal `x`)+
```

The vector type represents a SIMD style vector, used by target-specific
operation sets like AVX. While the most common use is for 1D vectors (e.g.
vector<16 x f32>) we also support multidimensional registers on targets that
support them (like TPUs).

Vector shapes must be positive decimal integers.

Note: hexadecimal integer literals are not allowed in vector type declarations,
`vector<0x42xi32>` is invalid because it is interpreted as a 2D vector with
shape `(0, 42)` and zero shapes are not allowed.

## Attributes

Syntax:

```
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

Attributes are the mechanism for specifying constant data on operations in
places where a variable is never allowed - e.g. the comparison predicate of a
[`cmpi` operation](Dialects/Standard.md#stdcmpi-cmpiop). Each operation has an
attribute dictionary, which associates a set of attribute names to attribute
values. MLIR's builtin dialect provides a rich set of
[builtin attribute values](#builtin-attribute-values) out of the box (such as
arrays, dictionaries, strings, etc.). Additionally, dialects can define their
own [dialect attribute values](#dialect-attribute-values).

The top-level attribute dictionary attached to an operation has special
semantics. The attribute entries are considered to be of two different kinds
based on whether their dictionary key has a dialect prefix:

- *inherent attributes* are inherent to the definition of an operation's
  semantics. The operation itself is expected to verify the consistency of these
  attributes. An example is the `predicate` attribute of the `std.cmpi` op.
  These attributes must have names that do not start with a dialect prefix.

- *discardable attributes* have semantics defined externally to the operation
  itself, but must be compatible with the operations's semantics. These
  attributes must have names that start with a dialect prefix. The dialect
  indicated by the dialect prefix is expected to verify these attributes. An
  example is the `gpu.container_module` attribute.

Note that attribute values are allowed to themselves be dictionary attributes,
but only the top-level dictionary attribute attached to the operation is subject
to the classification above.

### Attribute Value Aliases

```
attribute-alias-def ::= '#' alias-name '=' attribute-value
attribute-alias ::= '#' alias-name
```

MLIR supports defining named aliases for attribute values. An attribute alias is
an identifier that can be used in the place of the attribute that it defines.
These aliases *must* be defined before their uses. Alias names may not contain a
'.', since those names are reserved for
[dialect attributes](#dialect-attribute-values).

Example:

```mlir
#map = affine_map<(d0) -> (d0 + 10)>

// Using the original attribute.
%b = affine.apply affine_map<(d0) -> (d0 + 10)> (%a)

// Using the attribute alias.
%b = affine.apply #map(%a)
```

### Dialect Attribute Values

Similarly to operations, dialects may define custom attribute values. The
syntactic structure of these values is identical to custom dialect type values,
except that dialect attribute values are distinguished with a leading '#', while
dialect types are distinguished with a leading '!'.

```
dialect-attribute-value ::= '#' opaque-dialect-item
dialect-attribute-value ::= '#' pretty-dialect-item
```

Dialect attribute values can be specified in a verbose form, e.g. like this:

```mlir
// Complex attribute value.
#foo<"something<abcd>">

// Even more complex attribute value.
#foo<"something<a%%123^^^>>>">
```

Dialect attribute values that are simple enough can use the pretty format, which
is a lighter weight syntax that is equivalent to the above forms:

```mlir
// Complex attribute
#foo.something<abcd>
```

Sufficiently complex dialect attribute values are required to use the verbose
form for generality. For example, the more complex type shown above would not be
valid in the lighter syntax: `#foo.something<a%%123^^^>>>` because it contains
characters that are not allowed in the lighter syntax, as well as unbalanced
`<>` characters.

See [here](Tutorials/DefiningAttributesAndTypes.md) on how to define dialect
attribute values.

### Builtin Attribute Values

Builtin attributes are a core set of
[dialect attribute values](#dialect-attribute-values) that are defined in a
builtin dialect and thus available to all users of MLIR.

```
builtin-attribute ::=    affine-map-attribute
                       | array-attribute
                       | bool-attribute
                       | dictionary-attribute
                       | elements-attribute
                       | float-attribute
                       | integer-attribute
                       | integer-set-attribute
                       | string-attribute
                       | symbol-ref-attribute
                       | type-attribute
                       | unit-attribute
```

#### AffineMap Attribute

Syntax:

```
affine-map-attribute ::= `affine_map` `<` affine-map `>`
```

An affine-map attribute is an attribute that represents an affine-map object.

#### Array Attribute

Syntax:

```
array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
```

An array attribute is an attribute that represents a collection of attribute
values.

#### Boolean Attribute

Syntax:

```
bool-attribute ::= bool-literal
```

A boolean attribute is a literal attribute that represents a one-bit boolean
value, true or false.

#### Dictionary Attribute

Syntax:

```
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
```

A dictionary attribute is an attribute that represents a sorted collection of
named attribute values. The elements are sorted by name, and each name must be
unique within the collection.

#### Elements Attributes

Syntax:

```
elements-attribute ::= dense-elements-attribute
                     | opaque-elements-attribute
                     | sparse-elements-attribute
```

An elements attribute is a literal attribute that represents a constant
[vector](#vector-type) or [tensor](#tensor-type) value.

##### Dense Elements Attribute

Syntax:

```
dense-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                             ( tensor-type | vector-type )
```

A dense elements attribute is an elements attribute where the storage for the
constant vector or tensor value has been densely packed. The attribute supports
storing integer or floating point elements, with integer/index/floating element
types. It also support storing string elements with a custom dialect string
element type.

##### Opaque Elements Attribute

Syntax:

```
opaque-elements-attribute ::= `opaque` `<` dialect-namespace  `,`
                              hex-string-literal `>` `:`
                              ( tensor-type | vector-type )
```

An opaque elements attribute is an elements attribute where the content of the
value is opaque. The representation of the constant stored by this elements
attribute is only understood, and thus decodable, by the dialect that created
it.

Note: The parsed string literal must be in hexadecimal form.

##### Sparse Elements Attribute

Syntax:

```
sparse-elements-attribute ::= `sparse` `<` attribute-value `,` attribute-value
                              `>` `:` ( tensor-type | vector-type )
```

A sparse elements attribute is an elements attribute that represents a sparse
vector or tensor object. This is where very few of the elements are non-zero.

The attribute uses COO (coordinate list) encoding to represent the sparse
elements of the elements attribute. The indices are stored via a 2-D tensor of
64-bit integer elements with shape [N, ndims], which specifies the indices of
the elements in the sparse tensor that contains non-zero values. The element
values are stored via a 1-D tensor with shape [N], that supplies the
corresponding values for the indices.

Example:

```mlir
  sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>

// This represents the following tensor:
///  [[1, 0, 0, 0],
///   [0, 0, 5, 0],
///   [0, 0, 0, 0]]
```

#### Float Attribute

Syntax:

```
float-attribute ::= (float-literal (`:` float-type)?)
                  | (hexadecimal-literal `:` float-type)
```

A float attribute is a literal attribute that represents a floating point value
of the specified [float type](#floating-point-types). It can be represented in
the hexadecimal form where the hexadecimal value is interpreted as bits of the
underlying binary representation. This form is useful for representing infinity
and NaN floating point values. To avoid confusion with integer attributes,
hexadecimal literals _must_ be followed by a float type to define a float
attribute.

Examples:

```
42.0         // float attribute defaults to f64 type
42.0 : f32   // float attribute of f32 type
0x7C00 : f16 // positive infinity
0x7CFF : f16 // NaN (one of possible values)
42 : f32     // Error: expected integer type
```

#### Integer Attribute

Syntax:

```
integer-attribute ::= integer-literal ( `:` (index-type | integer-type) )?
```

An integer attribute is a literal attribute that represents an integral value of
the specified integer or index type. The default type for this attribute, if one
is not specified, is a 64-bit integer.

##### Integer Set Attribute

Syntax:

```
integer-set-attribute ::= `affine_set` `<` integer-set `>`
```

An integer-set attribute is an attribute that represents an integer-set object.

#### String Attribute

Syntax:

```
string-attribute ::= string-literal (`:` type)?
```

A string attribute is an attribute that represents a string literal value.

#### Symbol Reference Attribute

Syntax:

```
symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
```

A symbol reference attribute is a literal attribute that represents a named
reference to an operation that is nested within an operation with the
`OpTrait::SymbolTable` trait. As such, this reference is given meaning by the
nearest parent operation containing the `OpTrait::SymbolTable` trait. It may
optionally contain a set of nested references that further resolve to a symbol
nested within a different symbol table.

This attribute can only be held internally by
[array attributes](#array-attribute) and
[dictionary attributes](#dictionary-attribute)(including the top-level operation
attribute dictionary), i.e. no other attribute kinds such as Locations or
extended attribute kinds.

**Rationale:** Identifying accesses to global data is critical to
enabling efficient multi-threaded compilation. Restricting global
data access to occur through symbols and limiting the places that can
legally hold a symbol reference simplifies reasoning about these data
accesses.

See [`Symbols And SymbolTables`](SymbolsAndSymbolTables.md) for more
information.

#### Type Attribute

Syntax:

```
type-attribute ::= type
```

A type attribute is an attribute that represents a [type object](#type-system).

#### Unit Attribute

```
unit-attribute ::= `unit`
```

A unit attribute is an attribute that represents a value of `unit` type. The
`unit` type allows only one value forming a singleton set. This attribute value
is used to represent attributes that only have meaning from their existence.

One example of such an attribute could be the `swift.self` attribute. This
attribute indicates that a function parameter is the self/context parameter. It
could be represented as a [boolean attribute](#boolean-attribute)(true or
false), but a value of false doesn't really bring any value. The parameter
either is the self/context or it isn't.

```mlir
// A unit attribute defined with the `unit` value specifier.
func @verbose_form(i1) attributes {dialectName.unitAttr = unit}

// A unit attribute can also be defined without the value specifier.
func @simple_form(i1) attributes {dialectName.unitAttr}
```
