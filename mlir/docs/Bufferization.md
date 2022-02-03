# Bufferization

[TOC]

## Overview

Bufferization in MLIR is the process of converting the `tensor` type to the
`memref` type. MLIR provides a composable system that allows dialects to
systematically bufferize a program. This system is a simple application of
MLIR's [dialect conversion](DialectConversion.md) infrastructure. The bulk of
the code related to bufferization is a set of ordinary `ConversionPattern`'s
that dialect authors write for converting ops that operate on `tensor`'s to ops
that operate on `memref`'s. A set of conventions and best practices are followed
that allow these patterns to be run across multiple independent passes (rather
than requiring a single huge atomic conversion pass), which makes the
compilation pipelines scalable, robust, and easy to debug.

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

## Bufferization's place in a compilation pipeline

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

## General structure of the bufferization process

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
  pm.addNestedPass<FuncOp>(createTCPBufferizePass()); // Bufferizes the downstream `tcp` dialect.
  pm.addNestedPass<FuncOp>(createSCFBufferizePass());
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());
  pm.addPass(createFuncBufferizePass());

  // Finalizing bufferization pass.
  pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());
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

## How to write a partial bufferization pass

The contract of a partial bufferization pass is that a subset of ops (or kinds
of ops, customizable by a ConversionTarget) get bufferized.

A partial bufferization pass is just a pass that uses the
[dialect conversion](DialectConversion.md) framework to apply
`ConversionPattern`s with a `tensor` to `memref` type conversion.

To describe how to write such a pass, we will walk through an example, the
`tensor-bufferize` pass
([code](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/lib/Dialect/Tensor/Transforms/Bufferize.cpp#L23),
[test](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/test/Dialect/Tensor/bufferize.mlir#L1))
that bufferizes the `tensor` dialect.

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
    target.addLegalDialect<StandardOpsDialect>();

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

-   `linalg-bufferize`
    ([code](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/lib/Dialect/Linalg/Transforms/Bufferize.cpp#L1),
    [test](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/test/Dialect/Linalg/bufferize.mlir#L1))

    -   Bufferizes the `linalg` dialect.
    -   This is an example of how to simultaneously bufferize all the ops that
        satisfy a certain OpInterface with a single pattern. Specifically,
        `BufferizeAnyLinalgOp`
        ([code](https://github.com/llvm/llvm-project/blob/daaaed6bb89044ac58a23f1bb1ccdd12342a5a58/mlir/lib/Dialect/Linalg/Transforms/Bufferize.cpp#L170))
        bufferizes any ops that implements the `LinalgOp` interface.

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
    ([code](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/lib/Dialect/StandardOps/Transforms/FuncBufferize.cpp#L1),
    [test](https://github.com/llvm/llvm-project/blob/bc8acf2ce8ad6e8c9b1d97b2e02d3f4ad26e1d9d/mlir/test/Dialect/Standard/func-bufferize.mlir#L1))

    -   Bufferizes `func`, `call`, and `BranchOpInterface` ops.
    -   This is an example of how to bufferize ops that have multi-block
        regions.
    -   This is an example of a pass that is not split along dialect
        subdivisions.

-   `arith-bufferize`
    ([code](https://github.com/llvm/llvm-project/blob/446425f89871aa7849c5615e6b695ebd10c9b34a/mlir/lib/Dialect/Arithmetic/Transforms/Bufferize.cpp),
    [test](https://github.com/llvm/llvm-project/blob/d1aed486efc6d35a81ca4acbabb4203c4b91cda9/mlir/test/Dialect/Arithmetic/bufferize.mlir))

    -   Bufferizes only `arith` ops of `tensor` type.
    -   This is an example of setting up the legality so that only a subset of
        `arith.constant` ops get bufferized.
    -   This is an example of a pass that is not split along dialect
        subdivisions.

## How to write a finalizing bufferization pass

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

## Changes since [the talk](#the-talk)

-   `func-bufferize` was changed to be a partial conversion pass, and there is a
    new `finalizing-bufferize` which serves as a general finalizing
    bufferization pass.
