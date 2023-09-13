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

To simplify this problem, One-Shot Bufferize was designed for ops that are in
*destination-passing style*. For every tensor result, such ops have a tensor
operand, whose buffer could be utilized for storing the result of the op in the
absence of other conflicts. We call such tensor operands the *destination*.

As an example, consider the following op: `%0 = tensor.insert %cst into
%t[%idx] : tensor<?xf32>`

`%t` is the destination in this example. When choosing a buffer for the result
`%0`, One-Shot Bufferize considers only two options:

1.  buffer(`%0`) = buffer(`%t`).
2.  buffer(`%0`) is a newly allocated buffer.

There may be other buffers in the same function that could potentially be used
for buffer(`%0`), but those are not considered by One-Shot Bufferize to keep the
bufferization simple. One-Shot Bufferize could be extended to consider such
buffers in the future to achieve a better quality of bufferization.

Tensor ops that are not in destination-passing style always bufferize to a
memory allocation. E.g.:

```mlir
%0 = tensor.generate %sz {
^bb0(%i : index):
  %cst = arith.constant 0.0 : f32
  tensor.yield %cst : f32
} : tensor<?xf32>
```

The result of `tensor.generate` does not have a "destination", so bufferization
allocates a new buffer. This could be avoided by choosing an op such as
`linalg.generic`, which can express the same computation with a destination
("out") tensor:

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
"`linalg.generic` on memrefs" that overwrites the memory of the subview. The
`tensor.insert_slice` bufferizes to a no-op (in the absence of RaW conflicts
such as a subsequent read of `%s`).

RaW conflicts are detected with an analysis of SSA use-def chains (details
later). One-Shot Bufferize works best if there is a single SSA use-def chain,
where the result of a tensor op is the "destination" operand of the next tensor
ops, e.g.:

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

In the above example, a buffer copy of buffer(`%another_tensor`) (with `%cst`
inserted) is yielded from the "then" branch.

In both examples, a buffer is allocated inside of a block and then yielded from
the block. Deallocation of such buffers is tricky and not currently implemented
in an efficient way. For this reason, One-Shot Bufferize must be explicitly
configured with `allow-return-allocs` to support such IR.

When running with `allow-return-allocs`, One-Shot Bufferize may introduce
allocations that cannot be deallocated by One-Shot Bufferize yet. For that
reason, `-buffer-deallocation` must be run after One-Shot Bufferize. This buffer
deallocation pass resolves yields of newly allocated buffers with copies. E.g.,
the `scf.if` example above would bufferize to IR similar to the following:

```mlir
%0 = scf.if %c -> (memref<?xf32>) {
  %1 = memref.alloc(...) : memref<?xf32>
  ...
  scf.yield %1 : memref<?xf32>
} else {
  %2 = memref.alloc(...) : memref<?xf32>
  memref.copy %another_memref, %2
  scf.yield %2 : memref<?xf32>
}
```

In the bufferized IR, both branches return a newly allocated buffer, so it does
not matter which if-branch was taken. In both cases, the resulting buffer `%0`
must be deallocated at some point after the `scf.if` (unless the `%0` is
returned/yielded from its block).

Note: Buffer allocations that are returned from a function are not deallocated,
not even with `-buffer-deallocation`. It is the caller's responsibility to
deallocate the buffer. In the future, this could be automated with allocation
hoisting (across function boundaries) or reference counting.

One-Shot Bufferize can be configured to leak all memory and not generate any
buffer deallocations with `create-deallocs=0`. This can be useful for
compatibility with legacy code that has its own method of deallocating buffers.

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
