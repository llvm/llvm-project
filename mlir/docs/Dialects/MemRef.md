# 'memref' Dialect

This dialect provides documentation for operations within the MemRef dialect.

**Please post an RFC on the [forum](https://llvm.discourse.group/c/mlir/31)
before adding or changing any operation in this dialect.**

[TOC]

## Operations

[include "Dialects/MemRefOps.md"]

### 'dma_start' operation

Syntax:

```
operation ::= `dma_start` ssa-use`[`ssa-use-list`]` `,`
               ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
               ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
              `:` memref-type `,` memref-type `,` memref-type
```

Starts a non-blocking DMA operation that transfers data from a source memref to
a destination memref. The operands include the source and destination memref's
each followed by its indices, size of the data transfer in terms of the number
of elements (of the elemental type of the memref), a tag memref with its
indices, and optionally two additional arguments corresponding to the stride (in
terms of number of elements) and the number of elements to transfer per stride.
The tag location is used by a dma_wait operation to check for completion. The
indices of the source memref, destination memref, and the tag memref have the
same restrictions as any load/store operation in an affine context (whenever DMA
operations appear in an affine context). See
[restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
in affine contexts. This allows powerful static analysis and transformations in
the presence of such DMAs including rescheduling, pipelining / overlap with
computation, and checking for matching start/end operations. The source and
destination memref need not be of the same dimensionality, but need to have the
same elemental type.

For example, a `dma_start` operation that transfers 32 vector elements from a
memref `%src` at location `[%i, %j]` to memref `%dst` at `[%k, %l]` would be
specified as shown below.

Example:

```mlir
%size = constant 32 : index
%tag = alloc() : memref<1 x i32, affine_map<(d0) -> (d0)>, 4>
%idx = constant 0 : index
dma_start %src[%i, %j], %dst[%k, %l], %size, %tag[%idx] :
     memref<40 x 8 x vector<16xf32>, affine_map<(d0, d1) -> (d0, d1)>, 0>,
     memref<2 x 4 x vector<16xf32>, affine_map<(d0, d1) -> (d0, d1)>, 2>,
     memref<1 x i32>, affine_map<(d0) -> (d0)>, 4>
```

### 'dma_wait' operation

Syntax:

```
operation ::= `dma_wait` ssa-use`[`ssa-use-list`]` `,` ssa-use `:` memref-type
```

Blocks until the completion of a DMA operation associated with the tag element
specified with a tag memref and its indices. The operands include the tag memref
followed by its indices and the number of elements associated with the DMA being
waited on. The indices of the tag memref have the same restrictions as
load/store indices.

Example:

```mlir
dma_wait %tag[%idx], %size : memref<1 x i32, affine_map<(d0) -> (d0)>, 4>
```
