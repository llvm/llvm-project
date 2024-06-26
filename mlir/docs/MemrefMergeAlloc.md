# Compile-time memref.alloc Scheduling and Merging

This document describes a compile-time optimization on `memref.alloc` to reduce
memory usage and improve memory locality.

## Current status of bufferization and memref pass pipeline
Bufferization is a process in the current MLIR of converting ops with tensor
semantics to ops with memref semantics. One-Shot Bufferize is a new tensor
bufferization pass designed for IR in destination-passing style, and with
aggressive in-place bufferization. The goal of
bufferization is to use as little memory as possible and copy as little memory
as possible, as a result, the existing focus is to determine in-place or
out-of-place among the OpOperand and OpResult of individual ops, while not
considering much about the overall memory reuse across Operators within a
sub-graph (or partition).

The current implementation of Bufferization and memref pass pipeline focuses on
copy-avoidance and in-place reusing of the memory. Consider a computation graph
of 4 layers of matmul sharing the same weight:
```mlir
func.func @mlp(%x: tensor<128x128xf32>, %y: tensor<128x128xf32>) -> tensor<128x128xf32> {
   %a0 = tensor.empty() : tensor<128x128xf32>
   %a = linalg.matmul ins(%x, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%a0: tensor<128x128xf32>) -> tensor<128x128xf32>
   %b0 = tensor.empty() : tensor<128x128xf32>
   %b = linalg.matmul ins(%a, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%b0: tensor<128x128xf32>) -> tensor<128x128xf32>
   %c0 = tensor.empty() : tensor<128x128xf32>
   %c = linalg.matmul ins(%b, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%c0: tensor<128x128xf32>) -> tensor<128x128xf32>
   %d0 = tensor.empty() : tensor<128x128xf32>
   %d = linalg.matmul ins(%c, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%d0: tensor<128x128xf32>) -> tensor<128x128xf32>
   return %d : tensor<128x128xf32>
}
```

The bufferization pass will create an `memref.alloc` for each of the tensor
`a0`, `b0` and `c0`. The bufferization result is like:

```mlir
func.func @mlp(%x: memref<128x128xf32>, %y: memref<128x128xf32>) -> memref<128x128xf32> {
   %a0 = memref.alloc() : memref<128x128xf32>
   linalg.matmul ins(%x, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%a0: memref<128x128xf32>)
   %b0 = memref.alloc() : memref<128x128xf32>
   linalg.matmul ins(%a0, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%b0: memref<128x128xf32>)
   %c0 = memref.alloc() : memref<128x128xf32>
   linalg.matmul ins(%b0, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%c0: memref<128x128xf32>)
   %d0 = memref.alloc() : memref<128x128xf32>
   linalg.matmul ins(%c0, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%d0: memref<128x128xf32>)
   return %d0 : memref<128x128xf32>
}
```

Without further optimizations, 3 temp buffers will be allocated at the runtime
for these tensors. However, as we can see in the IR, the buffer `a0` is no
longer used when buffer `c0` is allocated. So buffer `c0` can reuse the memory
buffer of buffer `a0`, to reduce the memory size footprint and improve the
locality.

An observation of the current bufferization and memref passes is that they do
not consider the memory buffer planning - to reuse the buffer/memref for less
total size and better locality.

## Merge-alloc pass
An optimization pass has been introduced to consolidate multiple allocations
(`memref.alloc` ops) into a single `memref.alloc` op and each "mergeable"
`memref.alloc` op will be transformed into a "slice" from the "single allocated
buffer" with `memref.view` and some compile-time decided `offsets`. This
optimization works on `memref` instead of `tensor` ops, so it should be executed
after bufferization pass, and before adding buffer deallocation ops.

While merging the memory allocations, the transform should consider the lifetime
of each allocated `memref`s. By lifetime, we mean the range of time when the
memory allocated from `memref.alloc` is actively used. Views (aliases) into a
"base" memref should contribute to the lifetime of the "base". A later
`memref.alloc` should consider to reuse the memory of a previously allocated
memref, if the lifetime of these two does not overlap. The transform will
perform the "reusing" of memory by setting the `offset` of the later
`memref.view` to a position within the memory range of a previous allocation's
`memref.alloc` from the `single allocated buffer`.

Below is the expected transformation result of the example IR in the above
section:

```mlir
func.func @mlp(%x: memref<256x128xf32>, %y: memref<128x128xf32>) -> memref<128x128xf32> {
   %single_buffer = memref.alloc() : memref<131072xi8> // 128*128*sizeof(f32)*2
   %a0 = memref.view %single_buffer[0][] : memref<131072xi8> to memref<128x128xf32> // a0 takes the memory from byte offset 0
   linalg.matmul ins(%x, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%a0: memref<128x128xf32>)
   %b0 = memref.view %single_buffer[65536][] : memref<131072xi8> to memref<128x128xf32> // b0 takes the memory from byte offset 128*128*sizeof(f32)
   linalg.matmul ins(%a0, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%b0: memref<128x128xf32>) 
   %c0 = memref.view %single_buffer[0][] : memref<131072xi8> to memref<128x128xf32> // c0 takes the memory from byte offset 0
   linalg.matmul ins(%b0, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%c0: memref<128x128xf32>)
   %d0 = memref.alloc() : memref<128x128xf32> // d0 is returned, do not merge
   linalg.matmul ins(%c0, %y: memref<128x128xf32>, memref<128x128xf32>) outs(%d0: memref<128x128xf32>)
   return %d0 : memref<128x128xf32>
}
```

There is one single allocation `single_buffer` for all temp buffers and `alloc`
ops for `a0`, `b0` and `c0` are removed. The returned memref `d0` is untouched.
The memrefs `a0`, `b0` and `c0` are replaced by `memref.view` on
`single_buffer`. Since `a0` and `b0`'s lifetime overlaps, the transformation
will "allocate" different memory ranges on the `single_buffer` - note that `a0`
and `b0` has different offsets `%single_buffer[0]` and `%single_buffer[65536]`
and the memory ranges does not overlap. The memref `c0` does not overlap with
`a0` in their lifetime, so that `c0` can reuse the memory range of `a0` by
setting of offset to `%single_buffer[0]`, which is the same of `a0`. The final
allocation size of temp memory buffer will be `128*128*sizeof(f32)*2` instead of
three `memref<128x128xf32>` buffers in the original IR.


## Other solutions besides merge-alloc

Another (not yet existing) approach to resolve the memory reusing issue is to
insert `memref.dealloc` as soon as the buffer is no longer used. For example, in
the above "matmul" example, `memref.dealloc` can be inserted after the last use
of `a0` at `linalg.matmul ins(%a0, %y...)`. So even without memref merging
transformation, a common runtime memory allocator will try to reuse the memory
free'd by `memref.dealloc(%a0)` when allocating buffer for `c0`. However, there
are some disadvantages of this approach comparing to the compile-time memref
merging transformation of this proposal:
1. it depends on the implementation of the runtime memory allocator.
2. the runtime memory allocator does not have full picture of the future
   allocation/deallocation patterns of the program. For example, if we change
   the above example to make buffer size `c0` greater than size of `a0`, the
   runtime memory allocator will not likely to be able to reuse the memory of
   `a0` for `c0`, becuase the free memory chunk size of `a0` does not fit
   allocation of `c0`. In contrast, the proposed optimization of this document
   has the knowledge of the allocation patterns. Thus, it can put the memory
   chunk for `a0` in a right place of the `single allocation buffer`, so that
   the allocation of `c0` can fit into it.
3. calling runtime memory allocator for each buffer introduces more run time
   overhead than a single merged allocation after allocation merging.

However, utilizing runtime memory allocator can handle the cases when the
lifetime is hard to accurately analyze at compile-time, and when the shape is
unknown at compile-time, for example, to handle memref with dynamic shapes.
These two memory optimization approaches should coexist and cowork in the pass
pipeline.

## General framework for implementation of merge-alloc

To make merge-alloc pass capable of handling different hardware architectures
and runtime requirements, the pass is implemented as a general pipeline of the
following stages:

1. Collect the memory alias via `BufferViewFlowAnalysis`
2. Collect the memory lifetime traces
3. Schedule the buffers by an allocation algorithm to compute the offsets of
   each allocations
4. Rewrite the IR to replace allocations with views of merged buffers

The steps 2, 3 and 4 can be implemented by the developers to customize the pass
for their own use cases. A tick-based pipeline of the pass is provided as the
default implementation, which will be discussed in the next section. 

The following concepts should be defined by the implementation of the pass:
 * Mergeable allocation: the memref.alloc operations that should be merged by
   the pass. Other memref.alloc operations that are not "mergeable" should be
   untouched by the pass
 * Allocation scope: for each mergeable memref.alloc operation, there should be
   one ancestor surrounding basic blocking called "allocation scope". The memory
   allocation after merge-alloc for that memref.alloc operation should be
   hoisted and merged to that basic blocking. A "allocation scope" should
   contain a single merged allocation for the mergeable allocation in it.
 * Lifetime trace: for each mergeable memref.alloc operation, the "lifetime
   trace" should be collected, indicating the "allocation scope" and the
   liveness of the buffer allocated. The contents of a "lifetime trace" is
   implementation-defined


There are some more details on each step of the pipeline above.

### Collect the memory lifetime traces

This is the first stage that a developer can customize in merge-alloc. It should
collect the lifetime traces for each of the mergable memref.alloc operation. An
implementation of the lifetime trace collector should define which allocations
are mergeable and find the allocation scopes of them. It should also implement a
data structure to hold the detailed liveness of each buffers.

This step is abstracted in a `TraceCollectorFunc` function. The merge-alloc
framework defines the abstract interfaces for lifetime trace collector and the
collected traces as below:

```c++
/// abstract base class for lifetime of buffers in the same "allocation scope".
/// It should hold the lifetime informantion of buffers that are to be merged in
/// the same allocation in an "allocation scope". TraceCollectorFunc decides
/// which buffers are put into which "allocation scope".
class LifetimeTrace {
public:
  virtual Block *getAllocScope() const = 0;
  virtual Attribute getMemorySpace() const = 0;
};

/// top level memory trace info for multiple scopes. Each element of scopeTraces
/// should contain an "allocation scope" and the implementation-defined lifetime
/// data
struct MemoryTraceScopes {
  llvm::SmallVector<std::unique_ptr<LifetimeTrace>> scopeTraces;
  MemoryTraceScopes() = default;
};

using TraceCollectorFunc = std::function<FailureOr<MemoryTraceScopes>(
    Operation *, const BufferViewFlowAnalysis &,
    const MergeAllocationOptions &)>;
```

### Memory planning and scheduling

This step is abstracted in a `MemoryPlannerFunc` function. It accepts the
`MemoryTraceScopes` collected by the previous step. For each allocation scope in
`MemoryTraceScopes`, it decides the total merged allocation size and the offsets
for each mergeable allocation inside of the allocation scope. The abstract
interfaces are shown below:

```c++
/// the memory scheduling result for allocations in the same allocation scope.
/// allocation => offset map. All Operation* in the map should be
/// memref::AllocOp which are in the same LifetimeTrace.
struct MemorySchedule {
  size_t totalSize;
  Attribute memorySpace;
  llvm::DenseMap<Operation *, int64_t> allocToOffset;
  MemorySchedule() : totalSize{0} {}
};

using MemoryPlannerFunc = std::function<FailureOr<MemorySchedule>(
    Operation *, const LifetimeTrace &, const MergeAllocationOptions &)>;
```

### Rewriting the IR

Given the `MemorySchedule` of the previous step, this step rewrites the IR to
create the merged allocation in each of the allocation scopes, to replace the
mergable memref.alloc with views on the merged allocations with the offsets
calculated in the `MemorySchedule`. This step is abstracted in a
`MemoryMergeMutatorFunc` function.

```c++
using MemoryMergeMutatorFunc = std::function<LogicalResult(
    Operation *toplevel, Block *scope, const MemorySchedule &,
    const MergeAllocationOptions &)>;
```


## Tick-based Implementation for merge-alloc

A tick-based implementation of merge-alloc in provided by default. The basic
idea of the tick-based allocation merging is that

1. Each of the operations in a function is assigned a "tick". An operation with
   a smaller tick is expected to be executed before one with a larger tick.
2. Collect the first referenced tick and the last referenced tick for each
   mergeable allocation. If a buffer is referenced in loops and branches,
   special handling is needed.
3. For each allocation scope, linearize the first referenced tick and the last
   referenced tick of mergeable allocations inside of it into a single linear
   timeline.
4. Use a "static-memory-planner" to handle the linear timeline.

Limitations of Tick-based merge-alloc:
 * only contiguous, static shaped and identical layout memrefs are considered.
   Others are disregarded
 * only `RegionBranchOpInterface` operations are
   allowed to access memref inside the operations' children regions. Other
   operaions containing regions should not access memref inside. Otherwise, a
   pass error could occur.

### Basic concepts

In the context of tick-based merge-alloc, mergeable allocation and allocation
scope are defined as follows

#### Mergeable allocation

The pass should only consider to merge a `memref.alloc` only if
 * the ownership of the memref does not escape from the function or the body of
   the loop. That is, the memref and its alias should not be returned or
   yielded by a function or a loop.
 * and the memref is "dense" in its strides (points to a contiguous range of
   memory) and it has static shape

In tick-based merge-alloc, we call these `memref.alloc` **mergeable**
allocations.

The memrefs passed by function arguments, or returned by the function will be
untouched by this optimization.

#### Allocation scopes

The transformation first needs to identify the allocation scopes, which are
single basic blocks of parent operaions which
 * implement `AutomaticAllocationScope`
 * and are not `scf.for` (allocations in an `scf.for` can be hoisted to
 parent `AutomaticAllocationScope`)

For example, below is an example IR of a function with nested `scf.forall` ops.

```mlir
func.func @mlp(...) { // <---- alloc scope 1
   scf.for(...) { // <---- NOT an alloc scope!
      // allocation inside will be merge to alloc scope 1 above
   }
   ...
   scf.forall(...) { // <---- alloc scope 2
      ...
      // allocation here will be merge to alloc scope 2
      %buf = memref.alloc() : ...
      scf.forall(...) { // <---- alloc scope 3
      }
   }
}
```

There will be three allocation scopes as marked in the comments above. An
allocation scope marks the position to insert the `single allocation buffer`
after allocation merging. After the transformation, all "mergeable"
`memref.alloc` will be merged to the `single allocation buffer` of the nearest
ancestor `alloc scope`.

### Tick-based trace collection

This section discusses how ticks are collected and how the pass consolidates the
tick to get the lifetime traces for each mergeable allocations.

Ticks are assigned on each operation in the `func.func` by a increasing counter
with pre-order recursive `walk()` of the IR, as the "execution tick" for each
operation. After walking into the IR, the pass assigns two integers for each
mergeable allocations as the analysis result: `begin_tick` and `end_tick`, to
indicate the first and last tick of the use of the allocated memref in the IR.
Note that aliasing of memref buffers is also consider during tick collection.
When an operation which is not memory-effect-free accesses memrefs via its
operands, the ticks for the referenced memrefs and the aliasing memrefs of them
should be updated. The alias analysis is performed by `BufferViewFlowAnalysis`.

The collected result for each mergeable allocations will be an integer range
`[begin_tick,end_tick]` (both boundaries are inclusive), where `begin_tick <=
end_tick`. If two tick ranges of two mergeable allocations in the same
allocation scope do not overlap, this implies that these two buffer can share
the same memory address.

There should be special handling for loop and branch ops
(`RegionBranchOpInterface` or `LoopLikeOpInterface`) which references memrefs
allocated in parent scopes, to avoid wrong reuse of buffers used in the loops or
branches.

For example, consider the code like:

```mlir
func.func @basic() {
  ...
  %e = memref.alloc() : memref<8x64xf32> // tick = 0
  %f = memref.alloc() : memref<8x64xf32> // tick = 1
  scf.for %i = %c0 to %c3 step %c1 {     // tick = 2
      "test.use"(%e)  : (memref<8x64xf32>) -> () // tick = 3
      "test.use"(%f)  : (memref<8x64xf32>) -> () // tick = 4
  }
}
```

A manual observation of the IR will see that buffers `e` and `f` have
overlapping lifetime, because the access pattern in the loop is `e f e f e f`.
Thus, buffers `e` and `f` should not share the same memory address. However, the
collected ticks for the two buffers shows that they are only accessed in tick 3
and tick 4, respectively.

To produce the correct lifetime analysis result, the tick collector will
conservatively extend the lifetime of the accessed memrefs in loop and branch
ops (`RegionBranchOpInterface` or `LoopLikeOpInterface`), to make them span at
least the begin tick and end tick of the loop or branch op.

In the above example, both of the lifetime of buffers `e` and `f` will be
extended to the tick range of the parent `scf.for` op, as `[2, 4]`.

In some special cases, when the `memref.alloc` is in the block of a loop or
branch, and the buffer is not used outside of the loop or branch, the tick
collector does not need to conservatively extend the ticks of the allocations.
For example:

```mlir
func.func @basic() {
  ...
  scf.for %i = %c0 to %c3 step %c1 {     // tick = 0
      %g = memref.alloc() : memref<8x64xf32> // tick = 1
      %h = memref.alloc() : memref<8x64xf32> // tick = 2
      "test.use"(%g)  : (memref<8x64xf32>) -> () // tick = 3
      "test.use"(%h)  : (memref<8x64xf32>) -> () // tick = 4
  }
}
```

The buffer `g` has lifetime tick range `[3,3]` and `h` has `[4,4]`, because they
are allocated within the loop. Thus `g` and `h` has non-overlapping lifetime.

The remaining part of this section will discuss how the tick collector
consolidates the lifetime trace results.

After calling `walk()` into the function operation, there will be a map of
`AllocOp => [begin_tick,end_tick]` collected for each allocation scopes. In the
view of an allocation scope, it has a timeline of first-access and last-access
events of the mergeable allocations of the scope, sorted by the ticks in
incresing order. The tick traces for buffers inside an allocation scope are then
linearized to a stream of first-access and last-access events as the lifetime
traces. For example, an allocation scope has the allocations with the ticks

```
buffer=A, tick=[1,4], size=16
buffer=B, tick=[2,3], size=64
buffer=C, tick=[5,6], size=16
```

The linearized lifetime trace will be

```
alloc(A,16)
alloc(B,64)
free(B)
free(A)
alloc(C,16)
free(C)
```

The static memory planner discussed in the next section will take the linearized
lifetime trace as input.

### Static Memory Planner

The static memory planner is a compile-time memory allocator that plans the
memory for a list of allocations. It operates on a contiguous "base-buffer"
logically. For each "alloc" event in the linearized lifetime trace (see above
section), the memory planner logically "allocates" a contiguous range in the
contiguous "buffer". The start-offset of the "allocated" contiguous range will
be the returned as the memory planning result of the mergeable allocation, for
future IR rewriting.

The implementation of static memory planner is very similar to naive chunk-based
general runtime memory allocators like `malloc`. The logical memory are managed
by memory chunks, which represents a contiguous range of the "base-buffer". A
memory-chunk may be either marked "free" or "in-use" based on its allocation
state. The memory planner reads the linearized alloc/free events in their order
in the collected traces. On an "alloc" event, the memory planner finds an
appropriate free chunk, and split the chunk into two chunks - one for the memory
range for the allocation, and another the remaining free memory range. On an
"free" event, the memory planner marks the memory chunk as "free". If the
neighbouring memory chunks are also "free", the planner will further merge the
neighbouring free chunks into a larger free chunk.

A improvement of static memory planner over runtime memory allocators is that,
if a "free" memory chunk has smaller size than an allocation size, the memory
planner is allowed to "extend" the size of the "free" memory chunk to match the
allocation. It is not possible for runtime memory allocators, because extending
a memory chunk involves moving the memory addresses of the previously allocated
memory. However, in our compile-time memory planner, all the allocations are
logical, and the offsets of the allocated memory ranges can always be adjusted
by a later allocation. This improvement helps to reduce the issue of memory
fragmentation.

On an "alloc" event, the memory planner needs to choose one candidate from all
"free" memory chunks. A memory chunk that is recently free'd is considered "hot"
in cache. In the default configuration (when `planner-options=size-first` option
is not specified to the merge-alloc pass), static memory planner considers both
cache-locality and the degree of matching of allocation size and the chunk size
for each free memory chunks, with a simple cost-model. With
`planner-options=size-first` option is specified, static memory planner will
choose the best matched free memory chunk in the chunk size.