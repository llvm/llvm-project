# MLIR Language Reference

MLIR (Multi-Level IR) is a compiler intermediate representation with
similarities to traditional three-address SSA representations (like
[LLVM IR](http://llvm.org/docs/LangRef.html) or
[SIL](https://github.com/apple/swift/blob/main/docs/SIL.rst)), but which
introduces notions from polyhedral loop optimization as first-class concepts.
This hybrid design is optimized to represent, analyze, and transform high level
dataflow graphs as well as target-specific code generated for high performance
data parallel systems. Beyond its representational capabilities, its single
continuous design provides a framework to lower from dataflow graphs to
high-performance target-specific code.

This document defines and describes the key concepts in MLIR, and is intended to
be a dry reference document - the
[rationale documentation](Rationale/Rationale.md),
[glossary](../getting_started/Glossary.md), and other content are hosted
elsewhere.

MLIR is designed to be used in three different forms: a human-readable textual
form suitable for debugging, an in-memory form suitable for programmatic
transformations and analysis, and a compact serialized form suitable for storage
and transport. The different forms all describe the same semantic content. This
document describes the human-readable textual form.

[TOC]

## High-Level Structure

MLIR is fundamentally based on a graph-like data structure of nodes, called
*Operations*, and edges, called *Values*. Each Value is the result of exactly
one Operation or Block Argument, and has a *Value Type* defined by the
[type system](#type-system). [Operations](#operations) are contained in
[Blocks](#blocks) and Blocks are contained in [Regions](#regions). Operations
are also ordered within their containing block and Blocks are ordered in their
containing region, although this order may or may not be semantically meaningful
in a given [kind of region](Interfaces.md/#regionkindinterfaces)). Operations
may also contain regions, enabling hierarchical structures to be represented.

Operations can represent many different concepts, from higher-level concepts
like function definitions, function calls, buffer allocations, view or slices of
buffers, and process creation, to lower-level concepts like target-independent
arithmetic, target-specific instructions, configuration registers, and logic
gates. These different concepts are represented by different operations in MLIR
and the set of operations usable in MLIR can be arbitrarily extended.

MLIR also provides an extensible framework for transformations on operations,
using familiar concepts of compiler [Passes](Passes.md). Enabling an arbitrary
set of passes on an arbitrary set of operations results in a significant scaling
challenge, since each transformation must potentially take into account the
semantics of any operation. MLIR addresses this complexity by allowing operation
semantics to be described abstractly using [Traits](Traits.md) and
[Interfaces](Interfaces.md), enabling transformations to operate on operations
more generically. Traits often describe verification constraints on valid IR,
enabling complex invariants to be captured and checked. (see
[Op vs Operation](Tutorials/Toy/Ch-2.md/#op-vs-operation-using-mlir-operations))

One obvious application of MLIR is to represent an
[SSA-based](https://en.wikipedia.org/wiki/Static_single_assignment_form) IR,
like the LLVM core IR, with appropriate choice of operation types to define
Modules, Functions, Branches, Memory Allocation, and verification constraints to
ensure the SSA Dominance property. MLIR includes a collection of dialects which
defines just such structures. However, MLIR is intended to be general enough to
represent other compiler-like data structures, such as Abstract Syntax Trees in
a language frontend, generated instructions in a target-specific backend, or
circuits in a High-Level Synthesis tool.

Here's an example of an MLIR module:

```mlir
// Compute A*B using an implementation of multiply kernel and print the
// result using a TensorFlow op. The dimensions of A and B are partially
// known. The shapes are assumed to match.
func.func @mul(%A: tensor<100x?xf32>, %B: tensor<?x50xf32>) -> (tensor<100x50xf32>) {
  // Compute the inner dimension of %A using the dim operation.
  %n = memref.dim %A, 1 : tensor<100x?xf32>

  // Allocate addressable "buffers" and copy tensors %A and %B into them.
  %A_m = memref.alloc(%n) : memref<100x?xf32>
  memref.tensor_store %A to %A_m : memref<100x?xf32>

  %B_m = memref.alloc(%n) : memref<?x50xf32>
  memref.tensor_store %B to %B_m : memref<?x50xf32>

  // Call function @multiply passing memrefs as arguments,
  // and getting returned the result of the multiplication.
  %C_m = call @multiply(%A_m, %B_m)
          : (memref<100x?xf32>, memref<?x50xf32>) -> (memref<100x50xf32>)

  memref.dealloc %A_m : memref<100x?xf32>
  memref.dealloc %B_m : memref<?x50xf32>

  // Load the buffer data into a higher level "tensor" value.
  %C = memref.tensor_load %C_m : memref<100x50xf32>
  memref.dealloc %C_m : memref<100x50xf32>

  // Call TensorFlow built-in function to print the result tensor.
  "tf.Print"(%C){message: "mul result"} : (tensor<100x50xf32>) -> (tensor<100x50xf32>)

  return %C : tensor<100x50xf32>
}

// A function that multiplies two memrefs and returns the result.
func.func @multiply(%A: memref<100x?xf32>, %B: memref<?x50xf32>)
          -> (memref<100x50xf32>)  {
  // Compute the inner dimension of %A.
  %n = memref.dim %A, 1 : memref<100x?xf32>

  // Allocate memory for the multiplication result.
  %C = memref.alloc() : memref<100x50xf32>

  // Multiplication loop nest.
  affine.for %i = 0 to 100 {
     affine.for %j = 0 to 50 {
        memref.store 0 to %C[%i, %j] : memref<100x50xf32>
        affine.for %k = 0 to %n {
           %a_v  = memref.load %A[%i, %k] : memref<100x?xf32>
           %b_v  = memref.load %B[%k, %j] : memref<?x50xf32>
           %prod = arith.mulf %a_v, %b_v : f32
           %c_v  = memref.load %C[%i, %j] : memref<100x50xf32>
           %sum  = arith.addf %c_v, %prod : f32
           memref.store %sum, %C[%i, %j] : memref<100x50xf32>
        }
     }
  }
  return %C : memref<100x50xf32>
}
```

## Notation

MLIR has a simple and unambiguous grammar, allowing it to reliably round-trip
through a textual form. This is important for development of the compiler - e.g.
for understanding the state of code as it is being transformed and writing test
cases.

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

```
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


### Top level Productions

```
// Top level production
toplevel := (operation | attribute-alias-def | type-alias-def)*
```

The production `toplevel` is the top level production that is parsed by any parsing
consuming the MLIR syntax. [Operations](#operations),
[Attribute alises](#attribute-value-aliases), and [Type aliases](#type-aliases)
can be declared on the toplevel.

### Identifiers and keywords

Syntax:

```
// Identifiers
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
alias-name :: = bare-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*

// Uses of value, e.g. in an operand list to an operation.
value-use ::= value-id
value-use-list ::= value-use (`,` value-use)*
```

Identifiers name entities such as values, types and functions, and are chosen by
the writer of MLIR code. Identifiers may be descriptive (e.g. `%batch_size`,
`@matmul`), or may be non-descriptive when they are auto-generated (e.g. `%23`,
`@func42`). Identifier names for values may be used in an MLIR text file but are
not persisted as part of the IR - the printer will give them anonymous names
like `%42`.

MLIR guarantees identifiers never collide with keywords by prefixing identifiers
with a sigil (e.g. `%`, `#`, `@`, `^`, `!`). In certain unambiguous contexts
(e.g. affine expressions), identifiers are not prefixed, for brevity. New
keywords may be added to future versions of MLIR without danger of collision
with existing identifiers.

Value identifiers are only [in scope](#value-scoping) for the (nested) region in
which they are defined and cannot be accessed or referenced outside of that
region. Argument identifiers in mapping functions are in scope for the mapping
body. Particular operations may further limit which identifiers are in scope in
their regions. For instance, the scope of values in a region with
[SSA control flow semantics](#control-flow-and-ssacfg-regions) is constrained
according to the standard definition of
[SSA dominance](https://en.wikipedia.org/wiki/Dominator_\(graph_theory\)).
Another example is the [IsolatedFromAbove trait](Traits.md/#isolatedfromabove),
which restricts directly accessing values defined in containing regions.

Function identifiers and mapping identifiers are associated with
[Symbols](SymbolsAndSymbolTables.md) and have scoping rules dependent on symbol
attributes.

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
*   [Func dialect](Dialects/Func.md)
*   [GPU dialect](Dialects/GPU.md)
*   [LLVM dialect](Dialects/LLVM.md)
*   [SPIR-V dialect](Dialects/SPIR-V.md)
*   [Vector dialect](Dialects/Vector.md)

### Target specific operations

Dialects provide a modular way in which targets can expose target-specific
operations directly through to MLIR. As an example, some targets go through
LLVM. LLVM has a rich set of intrinsics for certain target-independent
operations (e.g. addition with overflow check) as well as providing access to
target-specific operations for the targets it supports (e.g. vector permutation
operations). LLVM intrinsics in MLIR are represented via operations that start
with an "llvm." name.

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
operation            ::= op-result-list? (generic-operation | custom-operation)
                         trailing-location?
generic-operation    ::= string-literal `(` value-use-list? `)`  successor-list?
                         region-list? dictionary-attribute? `:` function-type
custom-operation     ::= bare-id custom-operation-format
op-result-list       ::= op-result (`,` op-result)* `=`
op-result            ::= value-id (`:` integer-literal)
successor-list       ::= `[` successor (`,` successor)* `]`
successor            ::= caret-id (`:` block-arg-list)?
region-list          ::= `(` region (`,` region)* `)`
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location    ::= (`loc` `(` location `)`)?
```

MLIR introduces a uniform concept called *operations* to enable describing many
different levels of abstractions and computations. Operations in MLIR are fully
extensible (there is no fixed list of operations) and have application-specific
semantics. For example, MLIR supports
[target-independent operations](Dialects/MemRef.md),
[affine operations](Dialects/Affine.md), and
[target-specific machine operations](#target-specific-operations).

The internal representation of an operation is simple: an operation is
identified by a unique string (e.g. `dim`, `tf.Conv2d`, `x86.repmovsb`,
`ppc.eieio`, etc), can return zero or more results, take zero or more operands,
has a dictionary of [attributes](#attributes), has zero or more successors, and
zero or more enclosed [regions](#regions). The generic printing form includes
all these elements literally, with a function type to indicate the types of the
results and operands.

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
This allows those dialects to support *custom assembly form* for parsing and
printing operations. In the operation sets listed below, we show both forms.

### Builtin Operations

The [builtin dialect](Dialects/Builtin.md) defines a select few operations that
are widely applicable by MLIR dialects, such as a universal conversion cast
operation that simplifies inter/intra dialect conversion. This dialect also
defines a top-level `module` operation, that represents a useful IR container.

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

A *Block* is a list of operations. In
[SSACFG regions](#control-flow-and-ssacfg-regions), each block represents a
compiler [basic block](https://en.wikipedia.org/wiki/Basic_block) where
instructions inside the block are executed in order and terminator operations
implement control flow branches between basic blocks.

The last operation in a block must be a
[terminator operation](#control-flow-and-ssacfg-regions). A region with a single
block may opt out of this requirement by attaching the `NoTerminator` on the
enclosing op. The top-level `ModuleOp` is an example of such an operation which
defines this trait and whose block body does not have a terminator.

Blocks in MLIR take a list of block arguments, notated in a function-like way.
Block arguments are bound to values specified by the semantics of individual
operations. Block arguments of the entry block of a region are also arguments to
the region and the values bound to these arguments are determined by the
semantics of the containing operation. Block arguments of other blocks are
determined by the semantics of terminator operations, e.g. Branches, which have
the block as a successor. In regions with
[control flow](#control-flow-and-ssacfg-regions), MLIR leverages this structure
to implicitly represent the passage of control-flow dependent values without the
complex nuances of PHI nodes in traditional SSA representations. Note that
values which are not control-flow dependent can be referenced directly and do
not need to be passed through block arguments.

Here is a simple example function showing branches, returns, and block
arguments:

```mlir
func.func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  cf.br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  %b = arith.addi %a, %a : i64
  cf.br ^bb3(%b: i64)    // Branch passes %b as the argument

// ^bb3 receives an argument, named %c, from predecessors
// and passes it on to bb4 along with %a. %a is referenced
// directly from its defining operation and is not passed through
// an argument of ^bb3.
^bb3(%c: i64):
  cf.br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = arith.addi %d, %e : i64
  return %0 : i64   // Return is also a terminator.
}
```

**Context:** The "block argument" representation eliminates a number of special
cases from the IR compared to traditional "PHI nodes are operations" SSA IRs
(like LLVM). For example, the
[parallel copy semantics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.524.5461&rep=rep1&type=pdf)
of SSA is immediately apparent, and function arguments are no longer a special
case: they become arguments to the entry block
[[more rationale](Rationale/Rationale.md/#block-arguments-vs-phi-nodes)]. Blocks
are also a fundamental concept that cannot be represented by operations because
values defined in an operation cannot be accessed outside the operation.

## Regions

### Definition

A region is an ordered list of MLIR [Blocks](#blocks). The semantics within a
region is not imposed by the IR. Instead, the containing operation defines the
semantics of the regions it contains. MLIR currently defines two kinds of
regions: [SSACFG regions](#control-flow-and-ssacfg-regions), which describe
control flow between blocks, and [Graph regions](#graph-regions), which do not
require control flow between block. The kinds of regions within an operation are
described using the [RegionKindInterface](Interfaces.md/#regionkindinterfaces).

Regions do not have a name or an address, only the blocks contained in a region
do. Regions must be contained within operations and have no type or attributes.
The first block in the region is a special block called the 'entry block'. The
arguments to the entry block are also the arguments of the region itself. The
entry block cannot be listed as a successor of any other block. The syntax for a
region is as follows:

```
region      ::= `{` entry-block? block* `}`
entry-block ::= operation+
```

A function body is an example of a region: it consists of a CFG of blocks and
has additional semantic restrictions that other types of regions may not have.
For example, in a function body, block terminators must either branch to a
different block, or return from a function where the types of the `return`
arguments must match the result types of the function signature. Similarly, the
function arguments must match the types and count of the region arguments. In
general, operations with regions can define these correspondences arbitrarily.

An *entry block* is a block with no label and no arguments that may occur at
the beginning of a region. It enables a common pattern of using a region to
open a new scope.


### Value Scoping

Regions provide hierarchical encapsulation of programs: it is impossible to
reference, i.e. branch to, a block which is not in the same region as the source
of the reference, i.e. a terminator operation. Similarly, regions provides a
natural scoping for value visibility: values defined in a region don't escape to
the enclosing region, if any. By default, operations inside a region can
reference values defined outside of the region whenever it would have been legal
for operands of the enclosing operation to reference those values, but this can
be restricted using traits, such as
[OpTrait::IsolatedFromAbove](Traits.md/#isolatedfromabove), or a custom
verifier.

Example:

```mlir
  "any_op"(%a) ({ // if %a is in-scope in the containing region...
     // then %a is in-scope here too.
    %new_value = "another_op"(%a) : (i64) -> (i64)
  }) : (i64) -> (i64)
```

MLIR defines a generalized 'hierarchical dominance' concept that operates across
hierarchy and defines whether a value is 'in scope' and can be used by a
particular operation. Whether a value can be used by another operation in the
same region is defined by the kind of region. A value defined in a region can be
used by an operation which has a parent in the same region, if and only if the
parent could use the value. A value defined by an argument to a region can
always be used by any operation deeply contained in the region. A value defined
in a region can never be used outside of the region.

### Control Flow and SSACFG Regions

In MLIR, control flow semantics of a region is indicated by
[RegionKind::SSACFG](Interfaces.md/#regionkindinterfaces). Informally, these
regions support semantics where operations in a region 'execute sequentially'.
Before an operation executes, its operands have well-defined values. After an
operation executes, the operands have the same values and results also have
well-defined values. After an operation executes, the next operation in the
block executes until the operation is the terminator operation at the end of a
block, in which case some other operation will execute. The determination of the
next instruction to execute is the 'passing of control flow'.

In general, when control flow is passed to an operation, MLIR does not restrict
when control flow enters or exits the regions contained in that operation.
However, when control flow enters a region, it always begins in the first block
of the region, called the *entry* block. Terminator operations ending each block
represent control flow by explicitly specifying the successor blocks of the
block. Control flow can only pass to one of the specified successor blocks as in
a `branch` operation, or back to the containing operation as in a `return`
operation. Terminator operations without successors can only pass control back
to the containing operation. Within these restrictions, the particular semantics
of terminator operations is determined by the specific dialect operations
involved. Blocks (other than the entry block) that are not listed as a successor
of a terminator operation are defined to be unreachable and can be removed
without affecting the semantics of the containing operation.

Although control flow always enters a region through the entry block, control
flow may exit a region through any block with an appropriate terminator. The
standard dialect leverages this capability to define operations with
Single-Entry-Multiple-Exit (SEME) regions, possibly flowing through different
blocks in the region and exiting through any block with a `return` operation.
This behavior is similar to that of a function body in most programming
languages. In addition, control flow may also not reach the end of a block or
region, for example if a function call does not return.

Example:

```mlir
func.func @accelerator_compute(i64, i1) -> i64 { // An SSACFG region
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // This def for %value does not dominate ^bb2
  %value = "op.convert"(%a) : (i64) -> i64
  cf.br ^bb3(%a: i64)    // Branch passes %a as the argument

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
flow exits a region and is returned to the containing operation, the containing
operation may pass control flow to any region in the same operation. An
operation may also pass control flow to multiple contained regions concurrently.
An operation may also pass control flow into regions that were specified in
other operations, in particular those that defined the values or symbols the
given operation uses as in a call operation. This passage of control is
generally independent of passage of control flow through the basic blocks of the
containing region.

#### Closure

Regions allow defining an operation that creates a closure, for example by
“boxing” the body of the region into a value they produce. It remains up to the
operation to define its semantics. Note that if an operation triggers
asynchronous execution of the region, it is under the responsibility of the
operation caller to wait for the region to be executed guaranteeing that any
directly used values remain live.

### Graph Regions

In MLIR, graph-like semantics in a region is indicated by
[RegionKind::Graph](Interfaces.md/#regionkindinterfaces). Graph regions are
appropriate for concurrent semantics without control flow, or for modeling
generic directed graph data structures. Graph regions are appropriate for
representing cyclic relationships between coupled values where there is no
fundamental order to the relationships. For instance, operations in a graph
region may represent independent threads of control with values representing
streams of data. As usual in MLIR, the particular semantics of a region is
completely determined by its containing operation. Graph regions may only
contain a single basic block (the entry block).

**Rationale:** Currently graph regions are arbitrarily limited to a single basic
block, although there is no particular semantic reason for this limitation. This
limitation has been added to make it easier to stabilize the pass infrastructure
and commonly used passes for processing graph regions to properly handle
feedback loops. Multi-block regions may be allowed in the future if use cases
that require it arise.

In graph regions, MLIR operations naturally represent nodes, while each MLIR
value represents a multi-edge connecting a single source node and multiple
destination nodes. All values defined in the region as results of operations are
in scope within the region and can be accessed by any other operation in the
region. In graph regions, the order of operations within a block and the order
of blocks in a region is not semantically meaningful and non-terminator
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

Each value in MLIR has a type defined by the type system. MLIR has an open type
system (i.e. there is no fixed list of types), and types may have
application-specific semantics. MLIR dialects may define any number of types
with no restrictions on the abstractions they represent.

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
```

### Type Aliases

```
type-alias-def ::= '!' alias-name '=' type
type-alias ::= '!' alias-name
```

MLIR supports defining named aliases for types. A type alias is an identifier
that can be used in the place of the type that it defines. These aliases *must*
be defined before their uses. Alias names may not contain a '.', since those
names are reserved for [dialect types](#dialect-types).

Example:

```mlir
!avx_m128 = vector<4 x f32>

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

dialect-type ::= '!' (opaque-dialect-type | pretty-dialect-type)
opaque-dialect-type ::= dialect-namespace dialect-type-body
pretty-dialect-type ::= dialect-namespace '.' pretty-dialect-type-lead-ident
                                              dialect-type-body?
pretty-dialect-type-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'

dialect-type-body ::= '<' dialect-type-contents+ '>'
dialect-type-contents ::= dialect-type-body
                            | '(' dialect-type-contents+ ')'
                            | '[' dialect-type-contents+ ']'
                            | '{' dialect-type-contents+ '}'
                            | '[^\[<({\]>)}\0]+'
```

Dialect types are generally specified in an opaque form, where the contents
of the type are defined within a body wrapped with the dialect namespace
and `<>`. Consider the following examples:

```mlir
// A tensorflow string type.
!tf<string>

// A type with complex components.
!foo<something<abcd>>

// An even more complex type.
!foo<"a123^^^" + bar>
```

Dialect types that are simple enough may use a prettier format, which unwraps
part of the syntax into an equivalent, but lighter weight form:

```mlir
// A tensorflow string type.
!tf.string

// A type with complex components.
!foo.something<abcd>
```

See [here](AttributesAndTypes.md) to learn how to define dialect types.

### Builtin Types

The [builtin dialect](Dialects/Builtin.md) defines a set of types that are
directly usable by any other dialect in MLIR. These types cover a range from
primitive integer and floating-point types, function types, and more.

## Attributes

Syntax:

```
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

Attributes are the mechanism for specifying constant data on operations in
places where a variable is never allowed - e.g. the comparison predicate of a
[`cmpi` operation](Dialects/ArithmeticOps.md#arithcmpi-mlirarithcmpiop). Each operation has an
attribute dictionary, which associates a set of attribute names to attribute
values. MLIR's builtin dialect provides a rich set of
[builtin attribute values](#builtin-attribute-values) out of the box (such as
arrays, dictionaries, strings, etc.). Additionally, dialects can define their
own [dialect attribute values](#dialect-attribute-values).

The top-level attribute dictionary attached to an operation has special
semantics. The attribute entries are considered to be of two different kinds
based on whether their dictionary key has a dialect prefix:

-   *inherent attributes* are inherent to the definition of an operation's
    semantics. The operation itself is expected to verify the consistency of
    these attributes. An example is the `predicate` attribute of the
    `arith.cmpi` op. These attributes must have names that do not start with a
    dialect prefix.

-   *discardable attributes* have semantics defined externally to the operation
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

Similarly to operations, dialects may define custom attribute values.

```
dialect-namespace ::= bare-id

dialect-attribute ::= '#' (opaque-dialect-attribute | pretty-dialect-attribute)
opaque-dialect-attribute ::= dialect-namespace dialect-attribute-body
pretty-dialect-attribute ::= dialect-namespace '.' pretty-dialect-attribute-lead-ident
                                              dialect-attribute-body?
pretty-dialect-attribute-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'

dialect-attribute-body ::= '<' dialect-attribute-contents+ '>'
dialect-attribute-contents ::= dialect-attribute-body
                            | '(' dialect-attribute-contents+ ')'
                            | '[' dialect-attribute-contents+ ']'
                            | '{' dialect-attribute-contents+ '}'
                            | '[^\[<({\]>)}\0]+'
```

Dialect attributes are generally specified in an opaque form, where the contents
of the attribute are defined within a body wrapped with the dialect namespace
and `<>`. Consider the following examples:

```mlir
// A string attribute.
#foo<string<"">>

// A complex attribute.
#foo<"a123^^^" + bar>
```

Dialect attributes that are simple enough may use a prettier format, which unwraps
part of the syntax into an equivalent, but lighter weight form:

```mlir
// A string attribute.
#foo.string<"">
```

See [here](AttributesAndTypes.md) on how to define dialect attribute values.

### Builtin Attribute Values

The [builtin dialect](Dialects/Builtin.md) defines a set of attribute values
that are directly usable by any other dialect in MLIR. These types cover a range
from primitive integer and floating-point values, attribute dictionaries, dense
multi-dimensional arrays, and more.
