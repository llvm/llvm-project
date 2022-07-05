# 'affine' Dialect

This dialect provides a powerful abstraction for affine operations and analyses.

[TOC]

## Polyhedral Structures

MLIR uses techniques from polyhedral compilation to make dependence analysis and
loop transformations efficient and reliable. This section introduces some of the
core concepts that are used throughout the document.

### Dimensions and Symbols

Dimensions and symbols are the two kinds of identifiers that can appear in the
polyhedral structures, and are always of [`index`](Builtin.md/#indextype) type.
Dimensions are declared in parentheses and symbols are declared in square
brackets.

Examples:

```mlir
// A 2d to 3d affine mapping.
// d0/d1 are dimensions, s0 is a symbol
#affine_map2to3 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0, d1 - s0)>
```

Dimensional identifiers correspond to the dimensions of the underlying structure
being represented (a map, set, or more concretely a loop nest or a tensor); for
example, a three-dimensional loop nest has three dimensional identifiers. Symbol
identifiers represent an unknown quantity that can be treated as constant for a
region of interest.

Dimensions and symbols are bound to SSA values by various operations in MLIR and
use the same parenthesized vs square bracket list to distinguish the two.

Syntax:

```
// Uses of SSA values that are passed to dimensional identifiers.
dim-use-list ::= `(` ssa-use-list? `)`

// Uses of SSA values that are used to bind symbols.
symbol-use-list ::= `[` ssa-use-list? `]`

// Most things that bind SSA values bind dimensions and symbols.
dim-and-symbol-use-list ::= dim-use-list symbol-use-list?
```

SSA values bound to dimensions and symbols must always have 'index' type.

Example:

```mlir
#affine_map2to3 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0, d1 - s0)>
// Binds %N to the s0 symbol in affine_map2to3.
%x = memref.alloc()[%N] : memref<40x50xf32, #affine_map2to3>
```

### Restrictions on Dimensions and Symbols

The affine dialect imposes certain restrictions on dimension and symbolic
identifiers to enable powerful analysis and transformation. An SSA value's use
can be bound to a symbolic identifier if that SSA value is either 1. a region
argument for an op with trait `AffineScope` (eg. `FuncOp`), 2. a value defined
at the top level of an `AffineScope` op (i.e., immediately enclosed by the
latter), 3. a value that dominates the `AffineScope` op enclosing the value's
use, 4. the result of a
constant operation, 5. the result of an
[`affine.apply` operation](#affineapply-affineapplyop) that recursively takes as
arguments any valid symbolic identifiers, or 6. the result of a
[`dim` operation](MemRef.md/#memrefdim-mlirmemrefdimop) on either a memref that
is an argument to a `AffineScope` op or a memref where the corresponding
dimension is either static or a dynamic one in turn bound to a valid symbol.
*Note:* if the use of an SSA value is not contained in any op with the
`AffineScope` trait, only the rules 4-6 can be applied.

Note that as a result of rule (3) above, symbol validity is sensitive to the
location of the SSA use. Dimensions may be bound not only to anything that a
symbol is bound to, but also to induction variables of enclosing
[`affine.for`](#affinefor-affineforop) and
[`affine.parallel`](#affineparallel-affineparallelop) operations, and the result
of an [`affine.apply` operation](#affineapply-affineapplyop) (which recursively
may use other dimensions and symbols).

### Affine Expressions

Syntax:

```
affine-expr ::= `(` affine-expr `)`
              | affine-expr `+` affine-expr
              | affine-expr `-` affine-expr
              | `-`? integer-literal `*` affine-expr
              | affine-expr `ceildiv` integer-literal
              | affine-expr `floordiv` integer-literal
              | affine-expr `mod` integer-literal
              | `-`affine-expr
              | bare-id
              | `-`? integer-literal

multi-dim-affine-expr ::= `(` `)`
                        | `(` affine-expr (`,` affine-expr)* `)`
```

`ceildiv` is the ceiling function which maps the result of the division of its
first argument by its second argument to the smallest integer greater than or
equal to that result. `floordiv` is a function which maps the result of the
division of its first argument by its second argument to the largest integer
less than or equal to that result. `mod` is the modulo operation: since its
second argument is always positive, its results are always positive in our
usage. The `integer-literal` operand for ceildiv, floordiv, and mod is always
expected to be positive. `bare-id` is an identifier which must have type
[index](Builtin.md/#indextype). The precedence of operations in an affine
expression are ordered from highest to lowest in the order: (1)
parenthesization, (2) negation, (3) modulo, multiplication, floordiv, and
ceildiv, and (4) addition and subtraction. All of these operators associate from
left to right.

A *multidimensional affine expression* is a comma separated list of
one-dimensional affine expressions, with the entire list enclosed in
parentheses.

**Context:** An affine function, informally, is a linear function plus a
constant. More formally, a function f defined on a vector $\vec{v} \in
\mathbb{Z}^n$ is a multidimensional affine function of $\vec{v}$ if $f(\vec{v})$
can be expressed in the form $M \vec{v} + \vec{c}$ where $M$ is a constant
matrix from $\mathbb{Z}^{m \times n}$ and $\vec{c}$ is a constant vector from
$\mathbb{Z}$. $m$ is the dimensionality of such an affine function. MLIR further
extends the definition of an affine function to allow 'floordiv', 'ceildiv', and
'mod' with respect to positive integer constants. Such extensions to affine
functions have often been referred to as quasi-affine functions by the
polyhedral compiler community. MLIR uses the term 'affine map' to refer to these
multidimensional quasi-affine functions. As examples, $(i+j+1, j)$, $(i \mod 2,
j+i)$, $(j, i/4, i \mod 4)$, $(2i+1, j)$ are two-dimensional affine functions of
$(i, j)$, but $(i \cdot j, i^2)$, $(i \mod j, i/j)$ are not affine functions of
$(i, j)$.

### Affine Maps

Syntax:

```
affine-map-inline
   ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
```

The identifiers in the dimensions and symbols lists must be unique. These are
the only identifiers that may appear in 'multi-dim-affine-expr'. Affine maps
with one or more symbols in its specification are known as "symbolic affine
maps", and those with no symbols as "non-symbolic affine maps".

**Context:** Affine maps are mathematical functions that transform a list of
dimension indices and symbols into a list of results, with affine expressions
combining the indices and symbols. Affine maps distinguish between
[indices and symbols](#dimensions-and-symbols) because indices are inputs to the
affine map when the map is called (through an operation such as
[affine.apply](#affineapply-affineapplyop)), whereas symbols are bound when the
map is established (e.g. when a memref is formed, establishing a memory
[layout map](Builtin.md/#layout-map)).

Affine maps are used for various core structures in MLIR. The restrictions we
impose on their form allows powerful analysis and transformation, while keeping
the representation closed with respect to several operations of interest.

#### Named affine mappings

Syntax:

```
affine-map-id ::= `#` suffix-id

// Definitions of affine maps are at the top of the file.
affine-map-def    ::= affine-map-id `=` affine-map-inline
module-header-def ::= affine-map-def

// Uses of affine maps may use the inline form or the named form.
affine-map ::= affine-map-id | affine-map-inline
```

Affine mappings may be defined inline at the point of use, or may be hoisted to
the top of the file and given a name with an affine map definition, and used by
name.

Examples:

```mlir
// Affine map out-of-line definition and usage example.
#affine_map42 = affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>

// Use an affine mapping definition in an alloc operation, binding the
// SSA value %N to the symbol s0.
%a = memref.alloc()[%N] : memref<4x4xf32, #affine_map42>

// Same thing with an inline affine mapping definition.
%b = memref.alloc()[%N] : memref<4x4xf32, affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>>
```

### Semi-affine maps

Semi-affine maps are extensions of affine maps to allow multiplication,
`floordiv`, `ceildiv`, and `mod` with respect to symbolic identifiers.
Semi-affine maps are thus a strict superset of affine maps.

Syntax of semi-affine expressions:

```
semi-affine-expr ::= `(` semi-affine-expr `)`
                   | semi-affine-expr `+` semi-affine-expr
                   | semi-affine-expr `-` semi-affine-expr
                   | symbol-or-const `*` semi-affine-expr
                   | semi-affine-expr `ceildiv` symbol-or-const
                   | semi-affine-expr `floordiv` symbol-or-const
                   | semi-affine-expr `mod` symbol-or-const
                   | bare-id
                   | `-`? integer-literal

symbol-or-const ::= `-`? integer-literal | symbol-id

multi-dim-semi-affine-expr ::= `(` semi-affine-expr (`,` semi-affine-expr)* `)`
```

The precedence and associativity of operations in the syntax above is the same
as that for [affine expressions](#affine-expressions).

Syntax of semi-affine maps:

```
semi-affine-map-inline
   ::= dim-and-symbol-id-lists `->` multi-dim-semi-affine-expr
```

Semi-affine maps may be defined inline at the point of use, or may be hoisted to
the top of the file and given a name with a semi-affine map definition, and used
by name.

```
semi-affine-map-id ::= `#` suffix-id

// Definitions of semi-affine maps are at the top of file.
semi-affine-map-def ::= semi-affine-map-id `=` semi-affine-map-inline
module-header-def ::= semi-affine-map-def

// Uses of semi-affine maps may use the inline form or the named form.
semi-affine-map ::= semi-affine-map-id | semi-affine-map-inline
```

### Integer Sets

An integer set is a conjunction of affine constraints on a list of identifiers.
The identifiers associated with the integer set are separated out into two
classes: the set's dimension identifiers, and the set's symbolic identifiers.
The set is viewed as being parametric on its symbolic identifiers. In the
syntax, the list of set's dimension identifiers are enclosed in parentheses
while its symbols are enclosed in square brackets.

Syntax of affine constraints:

```
affine-constraint ::= affine-expr `>=` `affine-expr`
                    | affine-expr `<=` `affine-expr`
                    | affine-expr `==` `affine-expr`
affine-constraint-conjunction ::= affine-constraint (`,` affine-constraint)*
```

Integer sets may be defined inline at the point of use, or may be hoisted to the
top of the file and given a name with an integer set definition, and used by
name.

```
integer-set-id ::= `#` suffix-id

integer-set-inline
   ::= dim-and-symbol-id-lists `:` '(' affine-constraint-conjunction? ')'

// Declarations of integer sets are at the top of the file.
integer-set-decl ::= integer-set-id `=` integer-set-inline

// Uses of integer sets may use the inline form or the named form.
integer-set ::= integer-set-id | integer-set-inline
```

The dimensionality of an integer set is the number of identifiers appearing in
dimension list of the set. The affine-constraint non-terminals appearing in the
syntax above are only allowed to contain identifiers from dims and symbols. A
set with no constraints is a set that is unbounded along all of the set's
dimensions.

Example:

```mlir
// A example two-dimensional integer set with two symbols.
#set42 = affine_set<(d0, d1)[s0, s1]
   : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)>

// Inside a Region
affine.if #set42(%i, %j)[%M, %N] {
  ...
}
```

`d0` and `d1` correspond to dimensional identifiers of the set, while `s0` and
`s1` are symbol identifiers.

## Operations

[include "Dialects/AffineOps.md"]

### 'affine.load' operation

Syntax:

```
operation ::= ssa-id `=` `affine.load` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

The `affine.load` op reads an element from a memref, where the index for each
memref dimension is an affine expression of loop induction variables and
symbols. The output of 'affine.load' is a new value with the same type as the
elements of the memref. An affine expression of loop IVs and symbols must be
specified for each dimension of the memref. The keyword 'symbol' can be used to
indicate SSA identifiers which are symbolic.

Example:

```mlir

  Example 1:

    %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>

  Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

    %1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)]
      : memref<100x100xf32>

```

### 'affine.store' operation

Syntax:

```
operation ::= ssa-id `=` `affine.store` ssa-use, ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

The `affine.store` op writes an element to a memref, where the index for each
memref dimension is an affine expression of loop induction variables and
symbols. The 'affine.store' op stores a new value which is the same type as the
elements of the memref. An affine expression of loop IVs and symbols must be
specified for each dimension of the memref. The keyword 'symbol' can be used to
indicate SSA identifiers which are symbolic.

Example:

```mlir

    Example 1:

      affine.store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>

    Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

      affine.store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)]
        : memref<100x100xf32>

```

### 'affine.dma_start' operation

Syntax:

```
operation ::= `affine.dma_Start` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, ssa-use `:` memref-type
```

The `affine.dma_start` op starts a non-blocking DMA operation that transfers
data from a source memref to a destination memref. The source and destination
memref need not be of the same dimensionality, but need to have the same
elemental type. The operands include the source and destination memref's each
followed by its indices, size of the data transfer in terms of the number of
elements (of the elemental type of the memref), a tag memref with its indices,
and optionally at the end, a stride and a number_of_elements_per_stride
arguments. The tag location is used by an AffineDmaWaitOp to check for
completion. The indices of the source memref, destination memref, and the tag
memref have the same restrictions as any affine.load/store. In particular, index
for each memref dimension must be an affine expression of loop induction
variables and symbols. The optional stride arguments should be of 'index' type,
and specify a stride for the slower memory space (memory space with a lower
memory space id), transferring chunks of number_of_elements_per_stride every
stride until %num_elements are transferred. Either both or no stride arguments
should be specified. The value of 'num_elements' must be a multiple of
'number_of_elements_per_stride'.

Example:

```mlir
For example, a DmaStartOp operation that transfers 256 elements of a memref
'%src' in memory space 0 at indices [%i + 3, %j] to memref '%dst' in memory
space 1 at indices [%k + 7, %l], would be specified as follows:

  %num_elements = arith.constant 256
  %idx = arith.constant 0 : index
  %tag = memref.alloc() : memref<1xi32, 4>
  affine.dma_start %src[%i + 3, %j], %dst[%k + 7, %l], %tag[%idx],
    %num_elements :
      memref<40x128xf32, 0>, memref<2x1024xf32, 1>, memref<1xi32, 2>

  If %stride and %num_elt_per_stride are specified, the DMA is expected to
  transfer %num_elt_per_stride elements every %stride elements apart from
  memory space 0 until %num_elements are transferred.

  affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%idx], %num_elements,
    %stride, %num_elt_per_stride : ...
```

### 'affine.dma_wait' operation

Syntax:

```
operation ::= `affine.dma_Start` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, ssa-use `:` memref-type
```

The `affine.dma_start` op blocks until the completion of a DMA operation
associated with the tag element '%tag[%index]'. %tag is a memref, and %index has
to be an index with the same restrictions as any load/store index. In
particular, index for each memref dimension must be an affine expression of loop
induction variables and symbols. %num_elements is the number of elements
associated with the DMA operation. For example:

Example:

```mlir
affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%index], %num_elements :
  memref<2048xf32, 0>, memref<256xf32, 1>, memref<1xi32, 2>
...
...
affine.dma_wait %tag[%index], %num_elements : memref<1xi32, 2>
```
