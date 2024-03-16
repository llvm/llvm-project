# Bufferization

[TOC]

## Overview

Bufferization in MLIR is the process of converting ops with `tensor` semantics
to ops with `memref` semantics. MLIR provides an infrastructure that bufferizes
an entire program in a single pass (*One-Shot Bufferize*). This infrastructure
bufferizes all ops that implement the
[`BufferizableOpInterface`](https://github.com/llvm/llvm-project/blob/17a68065c378da74805e4e1b9a5b78cc9f83e580/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td)
can be bufferized.

MLIR has an older bufferization infrastructure built around
[dialect conversion](DialectConversion.md). Most dialect conversion
bufferization patterns have been migrated to One-Shot Bufferize, but some
functionality such as function boundary bufferization still depends on dialect
conversion and its type converter. New projects should use One-Shot Bufferize,
as the dialect conversion-based bufferization will eventually be deprecated.
Moreover, One-Shot Bufferize results in better bufferization with fewer memory
allocations and buffer copies. This documentation is mostly about One-Shot
Bufferize, but also describes how to gradually migrate a project from dialect
conversion-based bufferization to One-Shot Bufferize.

## What is One-Shot Bufferize?

One-Shot Bufferize is a new tensor bufferization pass designed for IR in
[destination-passing style](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/dps-fhpc17.pdf),
and with aggressive in-place bufferization.

One-Shot Bufferize is:

*   **Monolithic**: A single MLIR pass does the entire work, whereas the
    previous bufferization in MLIR was split across multiple passes residing in
    different dialects. In One-Shot Bufferize, `BufferizableOpInterface`
    implementations are spread across different dialects.

*   A **whole-function at a time analysis**. In-place bufferization decisions
    are made by analyzing SSA use-def chains on tensors. Op interface
    implementations not only provide the rewrite logic from tensor ops to memref
    ops, but also helper methods for One-Shot Bufferize's analysis to query
    information about an op's bufferization/memory semantics.

*   **Extensible** via an op interface: All ops that implement
    `BufferizableOpInterface` can be bufferized.

*   **2-Pass**: Bufferization is internally broken down into 2 steps: First,
    analyze the entire IR and make bufferization decisions. Then, bufferize
    (rewrite) the IR. The analysis has access to exact SSA use-def information.
    It incrementally builds alias and equivalence sets and does not rely on a
    posteriori-alias analysis from preallocated memory.

*   **Greedy**: Operations are analyzed one-by-one and it is decided on the spot
    whether a tensor OpOperand must be copied or not. Heuristics determine the
    order of analysis.

*   **Modular**: The current One-Shot Analysis can be replaced with a different
    analysis. The result of the analysis are queried by the bufferization via
    `AnalysisState`, in particular `AnalysisState::isInPlace`. Any derived class
    of `AnalysisState` that implements a small number virtual functions can
    serve as a custom analysis. It is even possible to run One-Shot Bufferize
    without any analysis (`AlwaysCopyAnalysisState`), in which case One-Shot
    Bufferize behaves exactly like the old dialect conversion-based
    bufferization (i.e., copy every buffer before writing to it).

To reduce complexity, One-Shot Bufferize should be
[run after other transformations](https://llvm.discourse.group/t/rfc-linalg-on-tensors-update-and-comprehensive-bufferization-rfc/3373),
typically as one of the last steps right before lowering memref ops. Many
transformations are easier in tensor land; e.g., tile/fuse/… on tensors first,
then bufferize the remaining IR.

From an architecture perspective, One-Shot Bufferize consists of
[BufferizableOpInterface](https://github.com/llvm/llvm-project/blob/17a68065c378da74805e4e1b9a5b78cc9f83e580/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td)
(and its implementations) and an
[analysis](https://github.com/llvm/llvm-project/blob/ae2764e835a26bad9774803eca0a6530df2a3e2d/mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h#L164)
of tensor SSA values that decides if a buffer can be used directly or must be
copied. The [bufferize] method of the op interface inspects analysis results and
rewrites tensor ops into memref ops.

## Goals of Bufferization

The high-level goal of every bufferization technique is to: 1. Use as little
memory as possible. 2. Copy as little memory as possible.

This implies reusing already allocated buffers when possible, turning
bufferization into an algorithmically complex problem with similarities to
register allocation.

Depending on the concrete use case, there may be additional bufferization
requirements. If the contents of a buffer are expensive to compute, there could
be a tradeoff between *recomputation* and *compute once and copy*. On the
contrary, it may not even be possible to allocate new buffers at runtime on some
architectures.

## Destination-Passing Style

Bufferization is an algorithmically complex problem. Given an op with a tensor
result, bufferization has to choose a memref buffer in which the result can be
stored. It is always safe to allocate a brand new buffer, but such a
bufferization strategy would be unacceptable for high-performance codegen. When
choosing an already existing buffer, we must be careful not to accidentally
overwrite data that is still needed later in the program.

To simplify this problem, One-Shot Bufferize was designed to take advantage of
*destination-passing style*. This form exists in itself independently of
bufferization and is tied to SSA semantics: many ops are “updating” part of
their input SSA variable. For example the LLVM instruction
[`insertelement`](https://llvm.org/docs/LangRef.html#insertelement-instruction)
is inserting an element inside a vector. Since SSA values are immutable, the
operation returns a copy of the input vector with the element inserted.
Another example in MLIR is `linalg.generic`, which always has an extra `outs`
operand which provides the initial values to update (for example when the
operation is doing a reduction). 

This input is referred to as "destination" in the following (quotes are
important as this operand isn't modified in place but copied) and comes into
place in the context of bufferization as a possible "anchor" for the
bufferization algorithm. This allows the user to shape the input in a form that
guarantees close to optimal bufferization result when carefully choosing the
SSA value used as "destination".

For every tensor result, a "destination-passing" style op has a corresponding
tensor operand. If there aren't any other uses of this tensor, the bufferization
can alias it with the op result and perform the operation "in-place" by reusing
the buffer allocated for this "destination" input.

As an example, consider the following op: `%0 = tensor.insert %cst into
%t[%idx] : tensor<?xf32>`

`%t` is the "destination" in this example. When choosing a buffer for the result
`%0`, denoted as `buffer(%0)`, One-Shot Bufferize considers only two options:

1.  `buffer(%0) = buffer(%t)` : alias the "destination" tensor with the
    result and perform the operation in-place.
2.  `buffer(%0)` is a newly allocated buffer.

There may be other buffers in the same function that could potentially be used
for `buffer(%0)`, but those are not considered by One-Shot Bufferize to keep the
bufferization simple. One-Shot Bufferize could be extended to consider such
buffers in the future to achieve a better quality of bufferization.

Tensor ops that are not in destination-passing style always bufferized to a
memory allocation. E.g.:

```mlir
%0 = tensor.generate %sz {
^bb0(%i : index):
  %cst = arith.constant 0.0 : f32
  tensor.yield %cst : f32
} : tensor<?xf32>
```

The result of `tensor.generate` does not have a "destination" operand, so
bufferization allocates a new buffer. This could be avoided by choosing an
op such as `linalg.generic`, which can express the same computation with a
"destination" operand, as specified behind outputs (`outs`):

```mlir
#map = affine_map<(i) -> (i)>
%0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
                    outs(%t : tensor<?xf32>) {
  ^bb0(%arg0 : f32):
    %cst = arith.constant 0.0 : f32
    linalg.yield %cst : f32
} -> tensor<?xf32>
```

At first glance, the above `linalg.generic` op may not seem very useful because
the output tensor `%t` is entirely overwritten. Why pass the tensor `%t` as an
operand in the first place? As an example, this can be useful for overwriting a
slice of a tensor:

```mlir
%t = tensor.extract_slice %s [%idx] [%sz] [1] : tensor<?xf32> to tensor<?xf32>
%0 = linalg.generic ... outs(%t) { ... } -> tensor<?xf32>
%1 = tensor.insert_slice %0 into %s [%idx] [%sz] [1]
    : tensor<?xf32> into tensor<?xf32>
```

The above example bufferizes to a `memref.subview`, followed by a
"`linalg.generic` on memrefs" that overwrites the memory of the subview, assuming
that the slice `%t` has no other user. The `tensor.insert_slice` then bufferizes
to a no-op (in the absence of RaW conflicts such as a subsequent read of `%s`).

RaW conflicts are detected with an analysis of SSA use-def chains (details
later). One-Shot Bufferize works best if there is a single SSA use-def chain,
where the result of a tensor op is the operand of the next tensor ops, e.g.:

```mlir
%0 = "my_dialect.some_op"(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
%1 = "my_dialect.another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
%2 = "my_dialect.yet_another_op"(%1) : (tensor<?xf32>) -> (tensor<?xf32>)
```

Buffer copies are likely inserted if the SSA use-def chain splits at some point,
e.g.:

```mlir
%0 = "my_dialect.some_op"(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
%1 = "my_dialect.another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
%2 = "my_dialect.yet_another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
```

One-Shot Bufferize has debug flags (`test-analysis-only print-conflicts`) that
print the results of the analysis and explain to the user why buffer copies were
inserted.

## Using One-Shot Bufferize

MLIR provides a pass
[`-one-shot-bufferize`](https://mlir.llvm.org/docs/Passes/#-one-shot-bufferize-one-shot-bufferize)
that performs an analysis and bufferizes all ops with tensor semantics that
implement `BufferizableOpInterface`. For modularity reasons, these op interface
implementations are typically external models that live in a dialect's
"Transforms" build unit. (External models are a mechanism for implementing an op
interface in a different build unit.) It is the user's responsibility to ensure
that all needed external models are registered before running One-Shot
Bufferize.

By default, One-Shot Bufferize fails when it encounters an op with tensor
semantics (i.e., tensor result or tensor operand) that is not bufferizable
(i.e., does not implement `BufferizableOpInterface`). This can be avoided with
`allow-unknown-ops`. In that case, One-Shot Bufferize inserts
`to_memref`/`to_tensor` ops around the bufferization boundary. These ops are
named versions of `unrealized_conversion_cast`. Note that One-Shot Bufferize's
analysis can currently not analyze these ops, so input IR with such ops may fail
bufferization. Therefore, running One-Shot Bufferize multiple times in a
sequence is also not supported at the moment.

One-Shot Bufferize can be configured to bufferize only ops from a set of
dialects with `dialect-filter`. This can be useful for gradually migrating from
dialect conversion-based bufferization to One-Shot Bufferize. One-Shot Bufferize
must run first in such a case, because dialect conversion-based bufferization
generates `to_tensor`/`to_memref` ops which One-Shot Bufferize cannot analyze.

One-Shot Bufferize can also be called programmatically with
[`bufferization::runOneShotBufferize`](https://github.com/llvm/llvm-project/blob/ae2764e835a26bad9774803eca0a6530df2a3e2d/mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h#L167).
Alternatively,
[`bufferization::bufferizeOp`](https://github.com/llvm/llvm-project/blob/ae2764e835a26bad9774803eca0a6530df2a3e2d/mlir/include/mlir/Dialect/Bufferization/Transforms/Bufferize.h#L78)
skips the analysis and inserts a copy on every buffer write, just like the
dialect conversion-based bufferization.

## Buffer Deallocation

**Important: this pass is deprecated, please use the ownership based buffer**
**deallocation pass instead**

One-Shot Bufferize deallocates all buffers that it allocates. This is in
contrast to the dialect conversion-based bufferization that delegates this job
to the
[`-buffer-deallocation`](https://mlir.llvm.org/docs/Passes/#-buffer-deallocation-adds-all-required-dealloc-operations-for-all-allocations-in-the-input-program)
pass. By default, One-Shot Bufferize rejects IR where a newly allocated buffer
is returned from a block. Such IR will fail bufferization.

A new buffer allocation is returned from a block when the result of an op that
is not in destination-passing style is returned. E.g.:

```mlir
%0 = scf.if %c -> (tensor<?xf32>) {
  %1 = tensor.generate ... -> tensor<?xf32>
  scf.yield %1 : tensor<?xf32>
} else {
  scf.yield %another_tensor : tensor<?xf32>
}
```

The `scf.yield` in the "else" branch is OK, but the `scf.yield` in the "then"
branch will be rejected.

Another case in which a buffer allocation may be returned is when a buffer copy
must be inserted due to a RaW conflict. E.g.:

```mlir
%0 = scf.if %c -> (tensor<?xf32>) {
  %1 = tensor.insert %cst into %another_tensor[%idx] : tensor<?xf32>
  "my_dialect.reading_tensor_op"(%another_tensor) : (tensor<?xf32>) -> ()
  ...
  scf.yield %1 : tensor<?xf32>
} else {
  scf.yield %yet_another_tensor : tensor<?xf32>
}
```

In the above example, a buffer copy of `buffer(%another_tensor)` (with `%cst`
inserted) is yielded from the "then" branch.

Note: Buffer allocations that are returned from a function are not deallocated.
It is the caller's responsibility to deallocate the buffer. For the full
function boundary ABI for MemRefs w.r.t. buffer deallocation refer to the
[*Function Boundary ABI*](#function-boundary-abi) section. In the future, this
could be automated with allocation hoisting (across function boundaries) or
reference counting.

One-Shot Bufferize leaks all memory and does not generate any buffer
deallocations. The `-buffer-deallocation-pipeline` has to be run afterwards to
insert the deallocation operations.

## Ownership-based Buffer Deallocation

Recommended compilation pipeline:
```
one-shot-bufferize
       |          it's recommended to perform all bufferization here at latest,
       |       <- any allocations inserted after this point have to be handled
       V          manually
expand-realloc
       V
ownership-based-buffer-deallocation
       V
  canonicalize <- mostly for scf.if simplifications
       V
buffer-deallocation-simplification
       V       <- from this point onwards no tensor values are allowed
lower-deallocations
       V
      CSE
       V
  canonicalize
```

One-Shot Bufferize does not deallocate any buffers that it allocates. This job
is delegated to the
[`-ownership-based-buffer-deallocation`](https://mlir.llvm.org/docs/Passes/#-ownership-based-buffer-deallocation)
pass, i.e., after running One-Shot Bufferize, the result IR may have a number of
`memref.alloc` ops, but no `memref.dealloc` ops. This pass processes operations
implementing `FunctionOpInterface` one-by-one without analysing the call-graph.
This means, that there have to be [some rules](#function-boundary-abi) on how
MemRefs are handled when being passed from one function to another. The rest of
the pass revolves heavily around the `bufferization.dealloc` operation which is
inserted at the end of each basic block with appropriate operands and should be
optimized using the Buffer Deallocation Simplification pass
(`--buffer-deallocation-simplification`) and the regular canonicalizer
(`--canonicalize`). Lowering the result of the
`-ownership-based-buffer-deallocation` pass directly using
`--convert-bufferization-to-memref` without beforehand optimization is not
recommended as it will lead to very inefficient code (the runtime-cost of
`bufferization.dealloc` is `O(|memrefs|^2+|memref|*|retained|)`).

### Function boundary ABI

The Buffer Deallocation pass operates on the level of operations implementing
the `FunctionOpInterface`. Such operations can take MemRefs as arguments, but
also return them. To ensure compatibility among all functions (including
external ones), some rules have to be enforced:
*   When a MemRef is passed as a function argument, ownership is never acquired.
    It is always the caller's responsibility to deallocate such MemRefs.
*   Returning a MemRef from a function always passes ownership to the caller,
    i.e., it is also the caller's responsibility to deallocate memrefs returned
    from a called function.
*   A function must not return a MemRef with the same allocated base buffer as
    one of its arguments (in this case a copy has to be created). Note that in
    this context two subviews of the same buffer that don't overlap are also
    considered to alias.

For external functions (e.g., library functions written externally in C), the
externally provided implementation has to adhere to these rules and they are
just assumed by the buffer deallocation pass. Functions on which the
deallocation pass is applied and the implementation is accessible are modified
by the pass such that the ABI is respected (i.e., buffer copies are inserted as
necessary).

### Inserting `bufferization.dealloc` operations

`bufferization.dealloc` operations are unconditionally inserted at the end of
each basic block (just before the terminator). The majority of the pass is about
finding the correct operands for this operation. There are three variadic
operand lists to be populated, the first contains all MemRef values that may
need to be deallocated, the second list contains their associated ownership
values (of `i1` type), and the third list contains MemRef values that are still
needed at a later point and should thus not be deallocated. This operation
allows us to deal with any kind of aliasing behavior: it lowers to runtime
aliasing checks when not enough information can be collected statically. When
enough aliasing information is statically available, operands or the entire op
may fold away.

**Ownerships**

To do so, we use a concept of ownership indicators of memrefs which materialize
as an `i1` value for any SSA value of `memref` type, indicating whether the
basic block in which it was materialized has ownership of this MemRef. Ideally,
this is a constant `true` or `false`, but might also be a non-constant SSA
value. To keep track of those ownership values without immediately materializing
them (which might require insertion of `bufferization.clone` operations or
operations checking for aliasing at runtime at positions where we don't actually
need a materialized value), we use the `Ownership` class. This class represents
the ownership in three states forming a lattice on a partial order:
```
forall X in SSA values. uninitialized < unique(X) < unknown
forall X, Y in SSA values.
  unique(X) == unique(Y) iff X and Y always evaluate to the same value
  unique(X) != unique(Y) otherwise
```
Intuitively, the states have the following meaning:
*   Uninitialized: the ownership is not initialized yet, this is the default
    state; once an operation is finished processing the ownership of all
    operation results with MemRef type should not be uninitialized anymore.
*   Unique: there is a specific SSA value that can be queried to check ownership
    without materializing any additional IR
*   Unknown: no specific SSA value is available without materializing additional
    IR, typically this is because two ownerships in 'Unique' state would have to
    be merged manually (e.g., the result of an `arith.select` either has the
    ownership of the then or else case depending on the condition value,
    inserting another `arith.select` for the ownership values can perform the
    merge and provide a 'Unique' ownership for the result), however, in the
    general case this 'Unknown' state has to be assigned.

Implied by the above partial order, the pass combines two ownerships in the
following way:

| Ownership 1   | Ownership 2   | Combined Ownership |
|:--------------|:--------------|:-------------------|
| uninitialized | uninitialized | uninitialized      |
| unique(X)     | uninitialized | unique(X)          |
| unique(X)     | unique(X)     | unique(X)          |
| unique(X)     | unique(Y)     | unknown            |
| unknown       | unique        | unknown            |
| unknown       | uninitialized | unknown            |
| <td colspan=3> + symmetric cases                   |

**Collecting the list of MemRefs that potentially need to be deallocated**

For a given block, the list of MemRefs that potentially need to be deallocated
at the end of that block is computed by keeping track of all values for which
the block potentially takes over ownership. This includes MemRefs provided as
basic block arguments, interface handlers for operations like `memref.alloc` and
`func.call`, but also liveness information in regions with multiple basic
blocks.  More concretely, it is computed by taking the MemRefs in the 'in' set
of the liveness analysis of the current basic block B, appended by the MemRef
block arguments and by the set of MemRefs allocated in B itself (determined by
the interface handlers), then subtracted (also determined by the interface
handlers) by the set of MemRefs deallocated in B.

Note that we don't have to take the intersection of the liveness 'in' set with
the 'out' set of the predecessor block because a value that is in the 'in' set
must be defined in an ancestor block that dominates all direct predecessors and
thus the 'in' set of this block is a subset of the 'out' sets of each
predecessor.

```
memrefs = filter((liveIn(block) U
  allocated(block) U arguments(block)) \ deallocated(block), isMemRef)
```

The list of conditions for the second variadic operands list of
`bufferization.dealloc` is computed by querying the stored ownership value for
each of the MemRefs collected as described above. The ownership state is updated
by the interface handlers while processing the basic block.

**Collecting the list of MemRefs to retain**

Given a basic block B, the list of MemRefs that have to be retained can be
different for each successor block S.  For the two basic blocks B and S and the
values passed via block arguments to the destination block S, we compute the
list of MemRefs that have to be retained in B by taking the MemRefs in the
successor operand list of the terminator and the MemRefs in the 'out' set of the
liveness analysis for B intersected with the 'in' set of the destination block
S.

This list of retained values makes sure that we cannot run into use-after-free
situations even if no aliasing information is present at compile-time.

```
toRetain = filter(successorOperands + (liveOut(fromBlock) insersect
  liveIn(toBlock)), isMemRef)
```

### Supported interfaces

The pass uses liveness analysis and a few interfaces:
*   `FunctionOpInterface`
*   `CallOpInterface`
*   `MemoryEffectOpInterface`
*   `RegionBranchOpInterface`
*   `RegionBranchTerminatorOpInterface`

Due to insufficient information provided by the interface, it also special-cases
on the `cf.cond_br` operation and makes some assumptions about operations
implementing the `RegionBranchOpInterface` at the moment, but improving the
interfaces would allow us to remove those dependencies in the future.

### Limitations

The Buffer Deallocation pass has some requirements and limitations on the input
IR. These are checked in the beginning of the pass and errors are emitted
accordingly:
*   The set of interfaces the pass operates on must be implemented (correctly).
    E.g., if there is an operation present with a nested region, but does not
    implement the `RegionBranchOpInterface`, an error is emitted because the
    pass cannot know the semantics of the nested region (and does not make any
    default assumptions on it).
*   No explicit control-flow loops are present. Currently, only loops using
    structural-control-flow are supported.  However, this limitation could be
    lifted in the future.
*   Deallocation operations should not be present already. The pass should
    handle them correctly already (at least in most cases), but it's not
    supported yet due to insufficient testing.
*   Terminators must implement either `RegionBranchTerminatorOpInterface` or
    `BranchOpInterface`, but not both. Terminators with more than one successor
    are not supported (except `cf.cond_br`). This is not a fundamental
    limitation, but there is no use-case justifying the more complex
    implementation at the moment.

### Example

The following example contains a few interesting cases:
*   Basic block arguments are modified to also pass along the ownership
    indicator, but not for entry blocks, where the function boundary ABI
    is applied instead.
*   The result of `arith.select` initially has 'Unknown' assigned as ownership,
    but once the `bufferization.dealloc` operation is inserted it is put in the
    'retained' list (since it has uses in a later basic block) and thus the
    'Unknown' ownership can be replaced with a 'Unique' ownership using the
    corresponding result of the dealloc operation.
*   The `cf.cond_br` operation has more than one successor and thus has to
    insert two `bufferization.dealloc` operations (one for each successor).
    While they have the same list of MemRefs to deallocate (because they perform
    the deallocations for the same block), it must be taken into account that
    some MemRefs remain *live* for one branch but not the other (thus set
    intersection is performed on the *live-out* of the current block and the
    *live-in* of the target block). Also, `cf.cond_br` supports separate
    forwarding operands for each successor. To make sure that no MemRef is
    deallocated twice (because there are two `bufferization.dealloc` operations
    with the same MemRefs to deallocate), the condition operands are adjusted to
    take the branch condition into account. While a generic lowering for such
    terminator operations could be implemented, a specialized implementation can
    take all the semantics of this particular operation into account and thus
    generate a more efficient lowering.

```mlir
func.func @example(%memref: memref<?xi8>, %select_cond: i1, %br_cond: i1) {
  %alloc = memref.alloc() : memref<?xi8>
  %alloca = memref.alloca() : memref<?xi8>
  %select = arith.select %select_cond, %alloc, %alloca : memref<?xi8>
  cf.cond_br %br_cond, ^bb1(%alloc : memref<?xi8>), ^bb1(%memref : memref<?xi8>)
^bb1(%bbarg: memref<?xi8>):
  test.copy(%bbarg, %select) : (memref<?xi8>, memref<?xi8>)
  return
}
```

After running `--ownership-based-buffer-deallocation`, it looks as follows:

```mlir
// Function boundary ABI: ownership of `%memref` will never be acquired.
func.func @example(%memref: memref<?xi8>, %select_cond: i1, %br_cond: i1) {
  %false = arith.constant false
  %true = arith.constant true

  // The ownership of a MemRef defined by the `memref.alloc` operation is always
  // assigned to be 'true'.
  %alloc = memref.alloc() : memref<?xi8>

  // The ownership of a MemRef defined by the `memref.alloca` operation is
  // always assigned to be 'false'.
  %alloca = memref.alloca() : memref<?xi8>

  // The ownership of %select will be the join of the ownership of %alloc and
  // the ownership of %alloca, i.e., of %true and %false. Because the pass does
  // not know about the semantics of the `arith.select` operation (unless a
  // custom handler is implemented), the ownership join will be 'Unknown'. If
  // the materialized ownership indicator of %select is needed, either a clone
  // has to be created for which %true is assigned as ownership or the result
  // of a `bufferization.dealloc` where %select is in the retain list has to be
  // used.
  %select = arith.select %select_cond, %alloc, %alloca : memref<?xi8>

  // We use `memref.extract_strided_metadata` to get the base memref since it is
  // not allowed to pass arbitrary memrefs to `memref.dealloc`. This property is
  // already enforced for `bufferization.dealloc`
  %base_buffer_memref, ... = memref.extract_strided_metadata %memref
    : memref<?xi8> -> memref<i8>, index, index, index
  %base_buffer_alloc, ... = memref.extract_strided_metadata %alloc
    : memref<?xi8> -> memref<i8>, index, index, index
  %base_buffer_alloca, ... = memref.extract_strided_metadata %alloca
    : memref<?xi8> -> memref<i8>, index, index, index

  // The deallocation conditions need to be adjusted to incorporate the branch
  // condition. In this example, this requires only a single negation, but might
  // also require multiple arith.andi operations.
  %not_br_cond = arith.xori %true, %br_cond : i1

  // There are two dealloc operations inserted in this basic block, one per
  // successor. Both have the same list of MemRefs to deallocate and the
  // conditions only differ by the branch condition conjunct.
  // Note, however, that the retained list differs. Here, both contain the
  // %select value because it is used in both successors (since it's the same
  // block), but the value passed via block argument differs (%memref vs.
  // %alloc).
  %10:2 = bufferization.dealloc
           (%base_buffer_memref, %base_buffer_alloc, %base_buffer_alloca
             : memref<i8>, memref<i8>, memref<i8>)
        if (%false, %br_cond, %false)
    retain (%alloc, %select : memref<?xi8>, memref<?xi8>)

  %11:2 = bufferization.dealloc
           (%base_buffer_memref, %base_buffer_alloc, %base_buffer_alloca
             : memref<i8>, memref<i8>, memref<i8>)
        if (%false, %not_br_cond, %false)
    retain (%memref, %select : memref<?xi8>, memref<?xi8>)

  // Because %select is used in ^bb1 without passing it via block argument, we
  // need to update it's ownership value here by merging the ownership values
  // returned by the dealloc operations
  %new_ownership = arith.select %br_cond, %10#1, %11#1 : i1

  // The terminator is modified to pass along the ownership indicator values
  // with each MemRef value.
  cf.cond_br %br_cond, ^bb1(%alloc, %10#0 : memref<?xi8>, i1),
                       ^bb1(%memref, %11#0 : memref<?xi8>, i1)

// All non-entry basic blocks are modified to have an additional i1 argument for
// each MemRef value in the argument list.
^bb1(%13: memref<?xi8>, %14: i1):  // 2 preds: ^bb0, ^bb0
  test.copy(%13, %select) : (memref<?xi8>, memref<?xi8>)

  %base_buffer_13, ... = memref.extract_strided_metadata %13
    : memref<?xi8> -> memref<i8>, index, index, index
  %base_buffer_select, ... = memref.extract_strided_metadata %select
    : memref<?xi8> -> memref<i8>, index, index, index

  // Here, we don't have a retained list, because the block has no successors
  // and the return has no operands.
  bufferization.dealloc (%base_buffer_13, %base_buffer_select
                          : memref<i8>, memref<i8>)
                     if (%14, %new_ownership)
  return
}
```

## Buffer Deallocation Simplification Pass

The [semantics of the `bufferization.dealloc` operation](https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationdealloc-bufferizationdeallocop)
provide a lot of opportunities for optimizations which can be conveniently split
into patterns using the greedy pattern rewriter. Some of those patterns need
access to additional analyses such as an analysis that can determine whether two
MemRef values must, may, or never originate from the same buffer allocation.
These patterns are collected in the Buffer Deallocation Simplification pass,
while patterns that don't need additional analyses are registered as part of the
regular canonicalizer pass. This pass is best run after
`--ownership-based-buffer-deallocation` followed by `--canonicalize`.

The pass applies patterns for the following simplifications:
*   Remove MemRefs from retain list when guaranteed to not alias with any value
    in the 'memref' operand list. This avoids an additional aliasing check with
    the removed value.
*   Split off values in the 'memref' list to new `bufferization.dealloc`
    operations only containing this value in the 'memref' list when it is
    guaranteed to not alias with any other value in the 'memref' list. This
    avoids at least one aliasing check at runtime and enables using a more
    efficient lowering for this new `bufferization.dealloc` operation.
*   Remove values from the 'memref' operand list when it is guaranteed to alias
    with at least one value in the 'retained' list and may not alias any other
    value in the 'retain' list.

## Lower Deallocations Pass

The `-lower-deallocations` pass transforms all `bufferization.dealloc`
operations to `memref.dealloc` operations and may also insert operations from
the `scf`, `func`, and `arith` dialects to make deallocations conditional and
check whether two MemRef values come from the same allocation at runtime (when
the `buffer-deallocation-simplification` pass wasn't able to determine it
statically).

The same lowering of the `bufferization.dealloc` operation is also part of the
`-convert-bufferization-to-memref` conversion pass which also lowers all the
other operations of the bufferization dialect.

We distinguish multiple cases in this lowering pass to provide an overall more
efficient lowering. In the general case, a library function is created to avoid
quadratic code size explosion (relative to the number of operands of the dealloc
operation). The specialized lowerings aim to avoid this library function because
it requires allocating auxiliary MemRefs of index values.

### Generic Lowering

A library function is generated to avoid code-size blow-up. On a high level, the
base-memref of all operands is extracted as an index value and stored into
specifically allocated MemRefs and passed to the library function which then
determines whether they come from the same original allocation. This information
is needed to avoid double-free situations and to correctly retain the MemRef
values in the `retained` list.

**Dealloc Operation Lowering**

This lowering supports all features the dealloc operation has to offer. It
computes the base pointer of each memref (as an index), stores it in a
new memref helper structure and passes it to the helper function generated
in `buildDeallocationLibraryFunction`. The results are stored in two lists
(represented as MemRefs) of booleans passed as arguments. The first list
stores whether the corresponding condition should be deallocated, the
second list stores the ownership of the retained values which can be used
to replace the result values of the `bufferization.dealloc` operation.

Example:
```
%0:2 = bufferization.dealloc (%m0, %m1 : memref<2xf32>, memref<5xf32>)
                          if (%cond0, %cond1)
                      retain (%r0, %r1 : memref<1xf32>, memref<2xf32>)
```
lowers to (simplified):
```
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%dealloc_base_pointer_list = memref.alloc() : memref<2xindex>
%cond_list = memref.alloc() : memref<2xi1>
%retain_base_pointer_list = memref.alloc() : memref<2xindex>
%m0_base_pointer = memref.extract_aligned_pointer_as_index %m0
memref.store %m0_base_pointer, %dealloc_base_pointer_list[%c0]
%m1_base_pointer = memref.extract_aligned_pointer_as_index %m1
memref.store %m1_base_pointer, %dealloc_base_pointer_list[%c1]
memref.store %cond0, %cond_list[%c0]
memref.store %cond1, %cond_list[%c1]
%r0_base_pointer = memref.extract_aligned_pointer_as_index %r0
memref.store %r0_base_pointer, %retain_base_pointer_list[%c0]
%r1_base_pointer = memref.extract_aligned_pointer_as_index %r1
memref.store %r1_base_pointer, %retain_base_pointer_list[%c1]
%dyn_dealloc_base_pointer_list = memref.cast %dealloc_base_pointer_list :
   memref<2xindex> to memref<?xindex>
%dyn_cond_list = memref.cast %cond_list : memref<2xi1> to memref<?xi1>
%dyn_retain_base_pointer_list = memref.cast %retain_base_pointer_list :
   memref<2xindex> to memref<?xindex>
%dealloc_cond_out = memref.alloc() : memref<2xi1>
%ownership_out = memref.alloc() : memref<2xi1>
%dyn_dealloc_cond_out = memref.cast %dealloc_cond_out :
   memref<2xi1> to memref<?xi1>
%dyn_ownership_out = memref.cast %ownership_out :
   memref<2xi1> to memref<?xi1>
call @dealloc_helper(%dyn_dealloc_base_pointer_list,
                     %dyn_retain_base_pointer_list,
                     %dyn_cond_list,
                     %dyn_dealloc_cond_out,
                     %dyn_ownership_out) : (...)
%m0_dealloc_cond = memref.load %dyn_dealloc_cond_out[%c0] : memref<2xi1>
scf.if %m0_dealloc_cond {
  memref.dealloc %m0 : memref<2xf32>
}
%m1_dealloc_cond = memref.load %dyn_dealloc_cond_out[%c1] : memref<2xi1>
scf.if %m1_dealloc_cond {
  memref.dealloc %m1 : memref<5xf32>
}
%r0_ownership = memref.load %dyn_ownership_out[%c0] : memref<2xi1>
%r1_ownership = memref.load %dyn_ownership_out[%c1] : memref<2xi1>
memref.dealloc %dealloc_base_pointer_list : memref<2xindex>
memref.dealloc %retain_base_pointer_list : memref<2xindex>
memref.dealloc %cond_list : memref<2xi1>
memref.dealloc %dealloc_cond_out : memref<2xi1>
memref.dealloc %ownership_out : memref<2xi1>
// replace %0#0 with %r0_ownership
// replace %0#1 with %r1_ownership
```

**Library function**

A library function is built per compilation unit that can be called at
bufferization dealloc sites to determine whether two MemRefs come from the same
allocation and their new ownerships.

The generated function takes two MemRefs of indices and three MemRefs of
booleans as arguments:
  * The first argument A should contain the result of the
  extract_aligned_pointer_as_index operation applied to the MemRefs to be
  deallocated
  * The second argument B should contain the result of the
  extract_aligned_pointer_as_index operation applied to the MemRefs to be
  retained
  * The third argument C should contain the conditions as passed directly
  to the deallocation operation.
  * The fourth argument D is used to pass results to the caller. Those
  represent the condition under which the MemRef at the corresponding
  position in A should be deallocated.
  * The fifth argument E is used to pass results to the caller. It
  provides the ownership value corresponding the the MemRef at the same
  position in B

This helper function is supposed to be called once for each
`bufferization.dealloc` operation to determine the deallocation need and
new ownership indicator for the retained values, but does not perform the
deallocation itself.

Generated code:
```
func.func @dealloc_helper(
    %dyn_dealloc_base_pointer_list: memref<?xindex>,
    %dyn_retain_base_pointer_list: memref<?xindex>,
    %dyn_cond_list: memref<?xi1>,
    %dyn_dealloc_cond_out: memref<?xi1>,
    %dyn_ownership_out: memref<?xi1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %false = arith.constant false
  %num_dealloc_memrefs = memref.dim %dyn_dealloc_base_pointer_list, %c0
  %num_retain_memrefs = memref.dim %dyn_retain_base_pointer_list, %c0
  // Zero initialize result buffer.
  scf.for %i = %c0 to %num_retain_memrefs step %c1 {
    memref.store %false, %dyn_ownership_out[%i] : memref<?xi1>
  }
  scf.for %i = %c0 to %num_dealloc_memrefs step %c1 {
    %dealloc_bp = memref.load %dyn_dealloc_base_pointer_list[%i]
    %cond = memref.load %dyn_cond_list[%i]
    // Check for aliasing with retained memrefs.
    %does_not_alias_retained = scf.for %j = %c0 to %num_retain_memrefs
        step %c1 iter_args(%does_not_alias_aggregated = %true) -> (i1) {
      %retain_bp = memref.load %dyn_retain_base_pointer_list[%j]
      %does_alias = arith.cmpi eq, %retain_bp, %dealloc_bp : index
      scf.if %does_alias {
        %curr_ownership = memref.load %dyn_ownership_out[%j]
        %updated_ownership = arith.ori %curr_ownership, %cond : i1
        memref.store %updated_ownership, %dyn_ownership_out[%j]
      }
      %does_not_alias = arith.cmpi ne, %retain_bp, %dealloc_bp : index
      %updated_aggregate = arith.andi %does_not_alias_aggregated,
                                      %does_not_alias : i1
      scf.yield %updated_aggregate : i1
    }
    // Check for aliasing with dealloc memrefs in the list before the
    // current one, i.e.,
    // `fix i, forall j < i: check_aliasing(%dyn_dealloc_base_pointer[j],
    // %dyn_dealloc_base_pointer[i])`
    %does_not_alias_any = scf.for %j = %c0 to %i step %c1
       iter_args(%does_not_alias_agg = %does_not_alias_retained) -> (i1) {
      %prev_dealloc_bp = memref.load %dyn_dealloc_base_pointer_list[%j]
      %does_not_alias = arith.cmpi ne, %prev_dealloc_bp, %dealloc_bp
      %updated_alias_agg = arith.andi %does_not_alias_agg, %does_not_alias
      scf.yield %updated_alias_agg : i1
    }
    %dealloc_cond = arith.andi %does_not_alias_any, %cond : i1
    memref.store %dealloc_cond, %dyn_dealloc_cond_out[%i] : memref<?xi1>
  }
  return
}
```

### Specialized Lowerings

Currently, there are two special lowerings for common cases to avoid the library
function and thus unnecessary memory load and store operations and function
calls:

**One memref, no retained**

Lower a simple case without any retained values and a single MemRef. Ideally,
static analysis can provide enough information such that the
`buffer-deallocation-simplification` pass is able to split the dealloc
operations up into this simple case as much as possible before running this
pass.

Example:
```mlir
bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
```
is lowered to
```mlir
scf.if %arg1 {
  memref.dealloc %arg0 : memref<2xf32>
}
```

In most cases, the branch condition is either constant 'true' or 'false' and can
thus be optimized away entirely by the canonicalizer pass.

**One memref, arbitrarily many retained**

A special case lowering for the deallocation operation with exactly one MemRef,
but an arbitrary number of retained values. The size of the code produced by
this lowering is linear to the number of retained values.

Example:
```mlir
%0:2 = bufferization.dealloc (%m : memref<2xf32>) if (%cond)
                      retain (%r0, %r1 : memref<1xf32>, memref<2xf32>)
return %0#0, %0#1 : i1, i1
```
is lowered to
```mlir
%m_base_pointer = memref.extract_aligned_pointer_as_index %m
%r0_base_pointer = memref.extract_aligned_pointer_as_index %r0
%r0_does_not_alias = arith.cmpi ne, %m_base_pointer, %r0_base_pointer
%r1_base_pointer = memref.extract_aligned_pointer_as_index %r1
%r1_does_not_alias = arith.cmpi ne, %m_base_pointer, %r1_base_pointer
%not_retained = arith.andi %r0_does_not_alias, %r1_does_not_alias : i1
%should_dealloc = arith.andi %not_retained, %cond : i1
scf.if %should_dealloc {
  memref.dealloc %m : memref<2xf32>
}
%true = arith.constant true
%r0_does_alias = arith.xori %r0_does_not_alias, %true : i1
%r0_ownership = arith.andi %r0_does_alias, %cond : i1
%r1_does_alias = arith.xori %r1_does_not_alias, %true : i1
%r1_ownership = arith.andi %r1_does_alias, %cond : i1
return %r0_ownership, %r1_ownership : i1, i1
```

## Memory Layouts

One-Shot Bufferize bufferizes ops from top to bottom. This works well when all
ops are bufferizable. However, when encountering a non-bufferizable tensor with
`allow-unknown-ops`, One-Shot Bufferize must insert `to_memref` ops at the
bufferization boundary and decide on a memref type. By default, One-Shot
Bufferize choose the most dynamic memref type wrt. layout maps. E.g.:

```mlir
%0 = "my_dialect.unbufferizable_op(%t) : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
%1 = tensor.extract %0[%idx1, %idx2] : tensor<?xf32>
```

When bufferizing the above IR, One-Shot Bufferize inserts a `to_memref` ops with
dynamic offset and strides:

```mlir
%0 = "my_dialect.unbufferizable_op(%t) : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
%0_m = bufferization.to_memref %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
%1 = memref.load %0_m[%idx1, %idx2] : memref<?x?xf32, strided<[?, ?], offset: ?>>
```

All users of `%0` have fully dynamic layout maps. This ensures that the
bufferized IR composes well with future bufferizations of `unbufferizable_op`
(maybe bufferized by another pass), regardless of the exact memref type of the
future bufferization. If the op turns out to be bufferized to an op with a
simpler memref type (e.g., identity layout map), we expect that canonicalization
patterns would clean up unnecessarily dynamic layout maps. (Some of these
canonicalization patterns may not be implemented yet.)

One-Shot Bufferize tries to infer the most precise memref type when bufferizing
an op. If the entire IR is bufferizable, we do not have to resort to
conservatively use fully dynamic layout maps. In that case, we also do not have
to rely on canonicalization patterns to clean up the bufferized IR.

Note: There are some bufferizable ops for which a percise layout map cannot be
inferred. E.g., a `tensor.cast` from a `tensor<*xf32>` to a `tensor<?x?xf32>`
must be bufferized to a `memref.cast` with a memref type that has a fully
dynamic layout map.

One-Shot Bufferize has an option `unknown-type-conversion` to control the
generation of layout maps when no precise layout can be inferred:

*   `fully-dynamic-layout-map` uses fully dynamic layout maps and is the default
    behavior. This composes well when IR is partially bufferized.
*   `identity-layout-map` uses static identity layout maps. This option can be
    useful for legacy code that cannot handle memref types with layout maps.
    Note that this setting can lead to additional buffer copies when folding a
    `to_tensor`/`to_memref` pair with memref types that are not cast-compatible.

Note: The `unknown-type-conversion` option does not affect layout maps of
function signatures. There is a separate `function-signature-type-conversion`
option that controls layout maps of function parameters and function results.

## Extending One-Shot Bufferize

Custom ops can be bufferized if they implement `BufferizableOpInterface`. Users
must at least implement the following interface methods.

*   `bufferizesToMemoryRead`: Return `true` if the buffer of the given tensor
    OpOperand is read.
*   `bufferizesToMemoryWrite`: Return `true` if the buffer of the given tensor
    OpOperand is written (if bufferizing in-place).
*   `getAliasingOpResult`: Return the OpResults that may share the same buffer
    as the given OpOperand. This interface method describes to
    OpOperand-to-OpResult mapping wrt. destination-passing style.
*   `bufferRelation`: Return `BufferRelation::Equivalent` if the given OpResult
    is the exact same memref as the aliasing OpOperand after bufferization (in
    case of in-place bufferization). Otherwise, (e.g., they overlap but are not
    necessarily the exact same memrefs), `BufferRelation::Unknown` should be
    returned. Additional buffer relations will be added in the future, but
    `BufferRelation::Unknown` is always safe.
*   `bufferize`: Rewrite the op with the given rewriter. Ops should be replaced
    with `bufferization::replaceOpWithBufferizedValues`.

To get a better intuition of the interface methods, we invite users to take a
look at existing implementations in MLIR, e.g., the implementation of
`tensor.insert` or `tensor.extract`.

## Debugging Buffer Copies

To get a better understanding of why One-Shot Bufferize introduced a buffer
copy, users can run the pass with `test-analysis-only print-conflicts`. Every
tensor op is then annotated with an attribute that has a boolean value for each
tensor OpOperand. `true` means that the OpOperand bufferizes in-place. `false`
means that the OpOperand bufferizes out-of-place and a buffer copy will be
inserted.

There are two reasons why a buffer copy may be inserted.

1.  Due to a RaW conflict, it is not safe to bufferize in-place. I.e., the
    overwritten data is still needed.
2.  The buffer is not writable. E.g., `memref.global` buffers that are the
    result of `arith.constant` ops are never modified.

In the first case, `print-conflicts` illustrates the conflict in the form of a
("read", "conflicting write", "last write") tuple.

## Understanding the SSA Use-Def Chain Analysis

To get a better understanding of the SSA Use-Def Chain Analysis and the RaW
conflict detection algorithm, we invite interested users to read the
[design document](https://discourse.llvm.org/uploads/short-url/5kckJ3DftYwQokG252teFgw3sYa.pdf)
and watch the corresponding [ODM talk](https://youtu.be/TXEo59CYS9A)
([slides](https://mlir.llvm.org/OpenMeetings/2022-01-13-One-Shot-Bufferization.pdf)).
can be used to bufferize a program in a single pass, as long as each op

## Migrating from Dialect Conversion-based Bufferization

Both dialect conversion-based bufferization and One-Shot Bufferize generate
`to_tensor`/`to_memref` ops at the bufferization boundary (when run with
`allow-unknown-ops`). They can be combined and run in sequence. However,
One-Shot Bufferize must run first because it cannot analyze those boundary ops.
To update existing code step-by-step, it may be useful to specify a dialect
filter for One-Shot Bufferize, so that dialects can be switched over one-by-one.

## Bufferization Function Graphs

One-Shot Bufferize does currently not support function graph bufferization.
I.e., `CallOp`, `ReturnOp` and function bbArgs are not bufferizable. Users can
run the existing `--func-bufferize` bufferization pass after One-Shot Bufferize.

Alternatively, users can try
[`ModuleBufferization`](https://github.com/llvm/llvm-project/blob/ae2764e835a26bad9774803eca0a6530df2a3e2d/mlir/include/mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h#L31),
which is an extension of One-Shot Bufferize. This bufferization is still under
development and does not support arbitrary IR. In essence, returning a tensor
from a function is not supported, unless it is equivalent to a function bbArg.
In that case, the corresponding return value can simply be dropped during
bufferization.

## Dialect Conversion-based Bufferization

Disclaimer: Most dialect conversion-based bufferization has been migrated to
One-Shot Bufferize. New users should use One-Shot Bufferize (with or without
analysis). The following documentation is only for existing users of dialect
conversion-based bufferization.

This system is a simple application of MLIR's dialect conversion infrastructure.
The bulk of the code related to bufferization is a set of ordinary
`ConversionPattern`'s that dialect authors write for converting ops that operate
on `tensor`'s to ops that operate on `memref`'s. A set of conventions and best
practices are followed that allow these patterns to be run across multiple
independent passes (rather than requiring a single huge atomic conversion pass),
which makes the compilation pipelines scalable, robust, and easy to debug.

This document is targeted at people looking to utilize MLIR's bufferization
functionality, along with people who want to extend it to cover their own ops.

<a name="the-talk">**NOTE:**</a> Before reading this document, please watch the
talk "Type Conversions the Not-So-Hard-Way: MLIR's New Bufferization
Infrastructure"
([slides](https://drive.google.com/file/d/1FVbzCXxZzS9LBLuvpPNLWJD-XDkt54ky/view?usp=sharing),
[recording](https://drive.google.com/file/d/1VfVajitgf8ZPnd-HRkJvaJiFLhBsluXN/view?usp=sharing)).
That talk gives a high-level overview of the bufferization infrastructure and
important conceptual details related to using the MLIR dialect conversion
infrastructure.

### Bufferization's place in a compilation pipeline

Bufferization itself does not free any of the buffers that have been allocated,
nor does it do anything particularly intelligent with the placement of buffers
w.r.t. control flow. Thus, a realistic compilation pipeline will usually consist
of:

1.  Bufferization
1.  Buffer optimizations such as `buffer-hoisting`, `buffer-loop-hoisting`, and
    `promote-buffers-to-stack`, which do optimizations that are only exposed
    after bufferization.
1.  Finally, running the [buffer deallocation](BufferDeallocationInternals.md)
    pass.

After buffer deallocation has been completed, the program will be quite
difficult to transform due to the presence of the deallocation ops. Thus, other
optimizations such as linalg fusion on memrefs should be done before that stage.

### General structure of the bufferization process

Bufferization consists of running multiple *partial* bufferization passes,
followed by one *finalizing* bufferization pass.

There is typically one partial bufferization pass per dialect (though other
subdivisions are possible). For example, for a dialect `X` there will typically
be a pass `X-bufferize` that knows how to bufferize all the ops in that dialect.
By running pass `X-bufferize` for each dialect `X` in the program, all the ops
in the program are incrementally bufferized.

Partial bufferization passes create programs where only some ops have been
bufferized. These passes will create *materializations* (also sometimes called
"casts") that convert between the `tensor` and `memref` type, which allows
bridging between ops that have been bufferized and ops that have not yet been
bufferized.

Finalizing bufferizations complete the bufferization process, and guarantee that
there are no tensors remaining in the program. This involves eliminating the
materializations. The pass `finalizing-bufferize` provides a minimal pass that
only eliminates materializations and issues an error if any unbufferized ops
exist in the program.

However, it is possible for a finalizing bufferization to do more than just
eliminate materializations. By adding patterns (just as a partial bufferization
would), it is possible for a finalizing bufferization pass to simultaneously
bufferize ops and eliminate materializations. This has a number of disadvantages
discussed in the talk and should generally be avoided.

### Example

As a concrete example, we will look at the bufferization pipeline from the
`mlir-npcomp` reference backend
([code](https://github.com/llvm/mlir-npcomp/blob/97d6d04d41216e73d40b89ffd79620973fc14ce3/lib/RefBackend/RefBackend.cpp#L232)).
The code, slightly simplified and annotated, is reproduced here:

```c++
  // Partial bufferization passes.
  pm.addPass(createTensorConstantBufferizePass());
  pm.addNestedPass<func::FuncOp>(createTCPBufferizePass()); // Bufferizes the downstream `tcp` dialect.
  pm.addNestedPass<func::FuncOp>(createSCFBufferizePass());
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  pm.addPass(createFuncBufferizePass());

  // Finalizing bufferization pass.
  pm.addNestedPass<func::FuncOp>(createFinalizingBufferizePass());
```

Looking first at the partial bufferization passes, we see that there are a
sequence of `FuncOp` passes (which run in parallel on functions). These function
passes are bracketed by `arith-bufferize` and `func-bufferize`, which are module
passes (and thus serialize the parallel compilation process). These two passes
must be module passes because they make changes to the top-level module.

The bulk of the bufferization work is done by the function passes. Most of these
passes are provided as part of the upstream MLIR distribution and bufferize
their respective dialects (e.g. `scf-bufferize` bufferizes the `scf` dialect).
The `tcp-bufferize` pass is an exception -- it is a partial bufferization pass
used to bufferize the downstream `tcp` dialect, and fits in perfectly with all
the other passes provided upstream.

The last pass is the finalizing bufferization pass. The `mlir-npcomp` reference
backend has arranged that all ops are bufferized by partial bufferizations, so
that the upstream `finalizing-bufferize` pass can be used as the finalizing
bufferization pass. This gives excellent diagnostics when something goes wrong
with the bufferization process, such as due to an op that wasn't handled by any
pattern.

### How to write a partial bufferization pass

The contract of a partial bufferization pass is that a subset of ops (or kinds
of ops, customizable by a ConversionTarget) get bufferized.

A partial bufferization pass is just a pass that uses the
[dialect conversion](DialectConversion.md) framework to apply
`ConversionPattern`s with a `tensor` to `memref` type conversion.

To describe how to write such a pass, we will walk through an example, the
`tensor-bufferize` pass
([code](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/lib/Dialect/Tensor/Transforms/Bufferize.cpp#L23),
[test](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/test/Dialect/Tensor/bufferize.mlir#L1))
that bufferizes the `tensor` dialect. Note that these passes have been replaced
with a `BufferizableOpInterface`-based implementation in the meantime, so we
have to take a looker at an older version of the code.

The bulk of the code in the pass will be a set of conversion patterns, with a
simple example being
[BufferizeCastOp](https://github.com/llvm/llvm-project/blob/2bf6e443e54604c7818c4d1a1837f3d091023270/mlir/lib/Dialect/Tensor/Transforms/Bufferize.cpp#L23)).

```
class BufferizeCastOp : public OpConversionPattern<tensor::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, resultType, adaptor.source());
    return success();
  }
};
```

See [the talk](#the-talk) for more details on how to write these patterns.

The
[pass itself](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/lib/Dialect/Tensor/Transforms/Bufferize.cpp#L57)
is very small, and follows the basic pattern of any dialect conversion pass.

```
void mlir::populateTensorBufferizePatterns(
    BufferizeTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<BufferizeCastOp, BufferizeExtractOp>(typeConverter,
                                                    patterns.getContext());
}

struct TensorBufferizePass : public TensorBufferizeBase<TensorBufferizePass> {
  void runOnOperation() override {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateTensorBufferizePatterns(typeConverter, patterns);
    target.addIllegalOp<tensor::CastOp, tensor::ExtractOp>();
    target.addLegalDialect<func::FuncDialect>();

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
```

The pass has all the hallmarks of a dialect conversion pass that does type
conversions: a `TypeConverter`, a `RewritePatternSet`, and a `ConversionTarget`,
and a call to `applyPartialConversion`. Note that a function
`populateTensorBufferizePatterns` is separated, so that power users can use the
patterns independently, if necessary (such as to combine multiple sets of
conversion patterns into a single conversion call, for performance).

One convenient utility provided by the MLIR bufferization infrastructure is the
`BufferizeTypeConverter`, which comes pre-loaded with the necessary conversions
and materializations between `tensor` and `memref`.

In this case, the `BufferizationOpsDialect` is marked as legal, so the
`bufferization.to_tensor` and `bufferization.to_memref` ops, which are inserted
automatically by the dialect conversion framework as materializations, are
legal. There is a helper `populateBufferizeMaterializationLegality`
([code](https://github.com/llvm/llvm-project/blob/a0b65a7bcd6065688189b3d678c42ed6af9603db/mlir/include/mlir/Transforms/Bufferize.h#L53))
which helps with this in general.

### Other partial bufferization examples

-   `scf-bufferize`
    ([code](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/lib/Dialect/SCF/Transforms/Bufferize.cpp#L1),
    [test](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/test/Dialect/SCF/bufferize.mlir#L1))

    -   Bufferizes ops from the `scf` dialect.
    -   This is an example of how to bufferize ops that implement
        `RegionBranchOpInterface` (that is, they use regions to represent
        control flow).
    -   The bulk of the work is done by
        `lib/Dialect/SCF/Transforms/StructuralTypeConversions.cpp`
        ([code](https://github.com/llvm/llvm-project/blob/daaaed6bb89044ac58a23f1bb1ccdd12342a5a58/mlir/lib/Dialect/SCF/Transforms/StructuralTypeConversions.cpp#L1)),
        which is well-commented and covers how to correctly convert ops that
        contain regions.

-   `func-bufferize`
    ([code](https://github.com/llvm/llvm-project/blob/2f5715dc78328215d51d5664c72c632a6dac1046/mlir/lib/Dialect/Func/Transforms/FuncBufferize.cpp#L1),
    [test](https://github.com/llvm/llvm-project/blob/2f5715dc78328215d51d5664c72c632a6dac1046/mlir/test/Dialect/Func/func-bufferize.mlir#L1))

    -   Bufferizes `func`, `call`, and `BranchOpInterface` ops.
    -   This is an example of how to bufferize ops that have multi-block
        regions.
    -   This is an example of a pass that is not split along dialect
        subdivisions.

### How to write a finalizing bufferization pass

The contract of a finalizing bufferization pass is that all tensors are gone
from the program.

The easiest way to write a finalizing bufferize pass is to not write one at all!
MLIR provides a pass `finalizing-bufferize` which eliminates the
`bufferization.to_tensor` / `bufferization.to_memref` materialization ops
inserted by partial bufferization passes and emits an error if that is not
sufficient to remove all tensors from the program.

This pass is sufficient when partial bufferization passes have bufferized all
the ops in the program, leaving behind only the materializations. When possible,
it is recommended to structure your pass pipeline this way, as this has the
significant advantage that if an op does not get bufferized (due to a missing
pattern, bug in the code, etc.), `finalizing-bufferize` will emit a nice clean
error, and the IR seen by `finalizing-bufferize` will only contain only one
unbufferized op.

However, before the current bufferization infrastructure was put in place,
bufferization could only be done as a single finalizing bufferization mega-pass
that used the `populate*BufferizePatterns` functions from multiple dialects to
simultaneously bufferize everything at once. Thus, one might see code in
downstream projects structured this way. This structure is not recommended in
new code. A helper, `populateEliminateBufferizeMaterializationsPatterns`
([code](https://github.com/llvm/llvm-project/blob/a0b65a7bcd6065688189b3d678c42ed6af9603db/mlir/include/mlir/Transforms/Bufferize.h#L58))
is available for such passes to provide patterns that eliminate
`bufferization.to_tensor` and `bufferization.to_memref`.

### Changes since [the talk](#the-talk)

-   `func-bufferize` was changed to be a partial conversion pass, and there is a
    new `finalizing-bufferize` which serves as a general finalizing
    bufferization pass.
-   Most partial bufferization passes have been reimplemented in terms of
    `BufferizableOpInterface`. New users should use One-Shot Bufferize instead
    of dialect conversion-based bufferization.
