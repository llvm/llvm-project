# Side Effects & Speculation

This document outlines how MLIR models side effects and how speculation works in
MLIR.

This rationale only applies to operations used in
[CFG regions](../LangRef.md/#control-flow-and-ssacfg-regions). Side effect
modeling in [graph regions](../LangRef.md/#graph-regions) is TBD.

[TOC]

## Overview

Many MLIR operations don't exhibit any behavior other than consuming and
producing SSA values. These operations can be reordered with other operations as
long as they obey SSA dominance requirements and can be eliminated or even
introduced (e.g. for
[rematerialization](https://en.wikipedia.org/wiki/Rematerialization)) as needed.

However, a subset of MLIR operations have implicit behavior than isn't reflected
in their SSA data-flow semantics. These operations need special handing, and
cannot be reordered, eliminated or introduced without additional analysis.

This doc introduces a categorization of these operations and shows how these
operations are modeled in MLIR.

## Categorization

Operations with implicit behaviors can be broadly categorized as follows:

1. Operations with memory effects. These operations read from and write to some
   mutable system resource, e.g. the heap, the stack, HW registers, the console.
   They may also interact with the heap in other ways, like by allocating and
   freeing memory. E.g. standard memory reads and writes, `printf` (which can be
   modeled as "writing" to the console and reading from the input buffers).
1. Operations with undefined behavior. These operations are not defined on
   certain inputs or in some situations -- we do not specify what happens when
   such illegal inputs are passed, and instead say that behavior is undefined
   and can assume it does not happen. In practice, in such cases these ops may
   do anything from producing garbage results to crashing the program or
   corrupting memory. E.g. integer division which has UB when dividing by zero,
   loading from a pointer that has been freed.
1. Operations that don't terminate. E.g. an `scf.while` where the condition is
   always true.
1. Operations with non-local control flow. These operations may pop their
   current frame of execution and return directly to an older frame. E.g.
   `longjmp`, operations that throw exceptions.

Finally, a given operation may have a combination of the above implicit
behaviors. The combination of implicit behaviors during the execution of the
operation may be ordered. We use 'stage' to label the order of implicit
behaviors during the execution of 'op'. Implicit behaviors with a lower stage
number happen earlier than those with a higher stage number.

## Modeling

Modeling these behaviors has to walk a fine line -- we need to empower more
complicated passes to reason about the nuances of such behaviors while
simultaneously not overburdening simple passes that only need a coarse grained
"can this op be freely moved" query.

MLIR has two op interfaces to represent these implicit behaviors:

1. The
   [`MemoryEffectsOpInterface` op interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td#L26)
   is used to track memory effects.
1. The
   [`ConditionallySpeculatable` op interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td#L105)
   is used to track undefined behavior and infinite loops.

Both of these are op interfaces which means operations can dynamically
introspect themselves (e.g. by checking input types or attributes) to infer what
memory effects they have and whether they are speculatable.

We don't have proper modeling yet to fully capture non-local control flow
semantics.

### Observable behavior and abstraction level

Side effects are modeled at the abstraction level of an operation's semantics.
A dialect defines which aspects of execution are observable and which are
implementation details outside that semantic model. For example, `memref.load`
models a read from the referenced memory, but does not model changes to
processor caches that may occur when executing the load. Such changes are
implementation details outside the operation's semantic model, so they do not
require a write effect on a cache resource.

Lowering may make implementation details explicit by introducing operations
with memory effects to implement an operation that has none at a higher
abstraction level. For example, `tensor.extract` has no memory effects because
it accesses an immutable tensor value. Bufferization lowers it to a
`memref.load`, which reports a read effect on the buffer that contains the
tensor's elements. The introduced operations must model their effects according
to their own semantics.

### Resource hierarchy and scope

Each effect is applied to a `Resource`. Resources form a hierarchy: each
resource may have a parent (`getParent()`), and the API provides
`isSubresourceOf()` and `isDisjointFrom()` so that passes can determine whether
two effects may conflict (e.g. CSE and alias analysis use disjointness to allow
more optimizations). A resource may be *addressable* (`isAddressable()`), meaning
effects on it can alias pointer-based memory; non-addressable resources (e.g.
runtime state) do not alias with any value-based memory location. The
canonical definition and API live in `mlir/Interfaces/SideEffectInterfaces.h`.

Resources can attach arbitrary attributes to their effect instances. These
attributes provide additional metadata about memory effects and can be used,
for example, to distinguish how a resource is accessed.

**Scope and limitations.** This mechanism is deliberately *not* intended for
fine-grained regions with specific addresses or sizes, or for alias classes /
offset-based disambiguation. Those concerns are out of scope for the resource
hierarchy and should be handled by alias analysis or other mechanisms.

### Guidelines for operation authors

When adding a new op, ask:

1. Does it access a resource, such as the heap or stack, in a way that is
   observable outside the operation under the dialect's semantics? It should
   probably implement `MemoryEffectsOpInterface` to report those effects.
1. Are these side effects ordered? The op should probably set the stage of
   side effects to make analyses more accurate.
1. Do these side effects act on every single value of a resource? It probably
   should set the FullEffect on effect.
1. Does it have side effects that must be preserved, like a volatile store or a
   syscall? It should probably implement `MemoryEffectsOpInterface` and model
   the effect as a read from or write to an abstract `Resource`. Please start an
   RFC if your operation has a novel side effect that cannot be adequately
   captured by `MemoryEffectsOpInterface`.
1. Is it well defined in all inputs or does it assume certain runtime
   restrictions on its inputs, e.g. the pointer operand must point to valid
   memory? It should probably implement `ConditionallySpeculatable`.
1. Can it infinitely loop on certain inputs? It should probably implement
   `ConditionallySpeculatable`.
1. Does it have non-local control flow (e.g. `longjmp`)? We don't have proper
   modeling for these yet, patches welcome!
1. Is your operation free of side effects and can be freely hoisted, introduced
   and eliminated? It should probably be marked `Pure`. (TODO: revisit this name
   since it has overloaded meanings in C++.)

## Examples

This section describes a few very simple examples that help understand how to
add side effect correctly.

### SIMD compute operation

Consider a SIMD backend dialect with a "simd.abs" operation which reads all
values from the source memref, calculates their absolute values, and writes them
to the target memref:

```mlir
  func.func @abs(%source : memref<10xf32>, %target : memref<10xf32>) {
    simd.abs(%source, %target) : memref<10xf32> to memref<10xf32>
    return
  }
```

The abs operation reads each individual value from the source resource and then
writes these values to each corresponding value in the target resource.
Therefore, we need to specify a read side effect for the source and a write side
effect for the target. The read side effect occurs before the write side effect,
so we need to mark the read stage as earlier than the write stage. Additionally,
we need to indicate that these side effects apply to each individual value in
the resource.

A typical approach is as follows:
``` mlir
  def AbsOp : SIMD_Op<"abs", [...] {
    ...

    let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the source memref",
                             [MemReadAt<0, FullEffect>]>:$source,
                         Arg<AnyRankedOrUnrankedMemRef, "the target memref",
                             [MemWriteAt<1, FullEffect>]>:$target);

    ...
  }
```

In the above example, we attach the side effect `[MemReadAt<0, FullEffect>]` to
the source, indicating that the abs operation reads each individual value from
the source during stage 0. Likewise, we attach the side effect
`[MemWriteAt<1, FullEffect>]` to the target, indicating that the abs operation
writes to each individual value within the target during stage 1 (after reading
from the source).

### Load like operation

Memref.load is a typical load like operation:
```mlir
  func.func @foo(%input : memref<10xf32>, %index : index) -> f32 {
    %result = memref.load  %input[index] : memref<10xf32>
    return %result : f32
  }
```

The load like operation reads a single value from the input memref and returns
it. Therefore, we need to specify a partial read side effect for the input
memref, indicating that not every single value is used.

A typical approach is as follows:
``` mlir
  def LoadOp : MemRef_Op<"load", [...] {
    ...

    let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                             [MemReadAt<0, PartialEffect>]>:$memref,
                         Variadic<Index>:$indices,
                         DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

    ...
  }
```

In the above example, we attach the side effect `[MemReadAt<0, PartialEffect>]` to
the source, indicating that the load operation reads parts of values from the
memref during stage 0. Since side effects typically occur at stage 0 and are
partial by default, we can abbreviate it as `[MemRead]`.

### Internal temporary memory

Consider an operation that computes a result using a temporary buffer. Its
implementation allocates the buffer, reads and writes it, and deallocates it
before returning. If the buffer does not escape and these actions have no
observable effects outside the operation under the dialect's semantics, the
operation may report no memory effects. For example, the dialect may leave
temporary storage requirements as an implementation detail. However, if the
dialect makes resource consumption or allocation failure observable, those
behaviors cannot be ignored merely because the buffer is temporary.

### Constant lookup tables

Consider an operation that computes a mathematical function using an internal,
immutable lookup table. If accessing the table has no observable effects beyond
producing the result according to the operation's semantics, the operation may
report no memory effects.

This does not mean that all reads from constant memory should always be modeled
as having no effects. For example, a dialect may model hardware that counts
accesses to a memory location or require volatile accesses to be preserved.
Such behavior must be modeled. Constant contents alone do not justify omitting
effects.
