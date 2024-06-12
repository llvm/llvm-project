# Compile-time memref.alloc Scheduling and Merging

This document describes a compile-time optimization on `memref.alloc` to reduce memory usage and improve memory locality.

## Current status of bufferization and memref pass pipeline
Bufferization is a process in the current MLIR of converting ops with tensor semantics to ops with memref semantics.
One-Shot Bufferize is a new tensor bufferization pass designed for IR in destination-passing style, and with aggressive in-place bufferization. The older/partial bufferization was built around multiple dialects. The community is trying to gradually deprecate the older bufferization and replace them with one-shot bufferization.
The goal of bufferization is to use as little memory as possible and copy as little memory as possible, as a result, the exsiting focus is to determine in-place or out-of-place among the OpOperand and OpResult of individual ops, while not considering much about the overall memory reuse across Operators within a sub-graph (or partition).

The current implementation of Bufferization and memref pass pipeline focuses on copy-avoidance and in-place reusing of the memory. Consider a computation graph of 4 layers of matmul sharing the same weight:
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

The bufferization pass will create an `memref.alloc` for each of the tensor `a0`, `b0` and `c0`. The bufferization result should be like:

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

Without further optimizations, 3 temp buffers will be allocated at the runtime for these tensors. However, as we can see in the IR, the buffer `a0` is no longer used when buffer `c0` is allocated. So buffer `c0` can reuse the memory buffer of buffer `a0`, to reduce the memory size footprint and improve the locality.

An observation of the current bufferization and memref passes is that they do not consider the memory buffer planning - to reuse the buffer/memref for less total size and better locality.

## Merge-alloc pass
An optimization pass has been introduced to consolidate multiple allocations (`memref.alloc` ops) into a single `memref.alloc` op and each static-shaped `memref.alloc` op will be transformed into a "slice" from the `single allocated buffer` with `memref.view` and some compile-time decided `offsets`. This optimization works on `memref` instead of `tensor` ops, so it should be executed after bufferization pass, and before adding buffer deallocation ops.

While merging the memory allocations, the transform should consider the lifetime of each allocated `memref`s. By lifetime, we mean the range of time when an memref allocated from `memref.alloc` is actively used. The references on `view`s of a "base" `memref` should contribute to the lifetime of the "base". A later `memref.alloc` should consider to reuse the memory of a previously allocated memref, if the lifetime of these two does not overlap. The transform will perform the "reusing" of memory by setting the `offset` of the later `memref.view` to a position within the memory range of a previous allocation's `memref.view` on the `single allocated buffer`.

Below is the expected transformation result of the example IR in the above section:

```mlir
func.func @mlp(%x: memref<128x128xf32>, %y: memref<128x128xf32>) -> memref<128x128xf32> {
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

There is one single allocation `single_buffer` for all temp buffers and `alloc` ops for `a0`, `b0` and `c0` are removed. The returned memref `d0` is untouched. The memrefs `a0`, `b0` and `c0` are replaced by `memref.view` on `single_buffer`. Since `a0` and `b0`'s lifetime overlaps, the transformation will "allocate" different memory ranges on the `single_buffer` - note that `a0` and `b0` has different offsets `%single_buffer[0]` and `%single_buffer[65536]` and the memory ranges does not overlap. The memref `c0` does not overlap with `a0` in their lifetime, so that `c0` can reuse the memory range of `a0` by setting of offset to `%single_buffer[0]`, which is the same of `a0`. The final allocation size of temp memory buffer will be `128*128*sizeof(f32)*2` instead of three `memref<128x128xf32>` buffers in the original IR.


## Other solutions besides merge-alloc

Another (not yet existing) approach to resolve the memory reusing issue is to insert `memref.dealloc` as soon as the buffer is no longer used. For example, in the above "matmul" example, `memref.dealloc` can be inserted after the last use of `a0` at `linalg.matmul ins(%a0, %y...)`. So even without memref merging transformation, a common runtime memory allocator will try to reuse the memory free'd by `memref.dealloc(%a0)` when allocating buffer for `c0`. However, there are some disadvantages of this approach comparing to the compile-time memref merging transformation of this proposal:
1. it depends on the implementation of the runtime memory allocator.
2. the runtime memory allocator does not have full picture of the future allocation/deallocation patterns of the program. For example, if we change the above example to make buffer size `c0` greater than size of `a0`, the runtime memory allocator will not likely to be able to reuse the memory of `a0` for `c0`, becuase the free memory chunk size of `a0` does not fit allocation of `c0`. In contrast, the proposed optimization of this document has the knowledge of the allocation patterns. Thus, it can put the memory chunk for `a0` in a right place of the `single allocation buffer`, so that the allocation of `c0` can fit into it.
3. calling runtime memory allocator for each buffer introduces more run time overhead than a single merged allocation after allocation merging.

However, utilizing runtime memory allocator can be viewed as a supplementary approach of the allocation merging at compile-time, for example, to handle memref with dynamic shapes. These two memory optimization approaches should coexist and cowork in the pass pipeline.

## General framework for implementation of merge-alloc

To make merge-alloc pass capable of handling different hardware architectures and runtime requirements, the pass is implemented as a general pipeline of the following stages:

1. Collect the memory alias via `BufferViewFlowAnalysis`
2. Collect the memory lifetime traces
3. Schedule the buffers by an allocation algorithm to compute the offsets of each allocations
4. Rewrite the IR to replace allocations with views of merged buffers

The steps 2, 3 and 4 can be implemented by the developers to customize the pass for their own use cases. A tick-based pipeline of the pass is provided as the default implementation, which will be discussed in the next section. 

The following concepts should be defined by the implementation of the pass:
 * Mergeable alloction: the memref.alloc operations that should be merged by the pass. Other memref.alloc operations that are not "mergeable" should be untouched by the pass
 * Allocation scope: for each mergeable memref.alloc operation, there should be one ancestor surrounding operation called "allocation scope". The memory allocation after merge-alloc for that memref.alloc operation should be hoisted and merged to the block of that "allocation scope". A "allocation scope" should contain a single merged allocation for the mergeable allocation in it.
 * Lifetime trace: for each mergeable memref.alloc operation, the "lifetime trace" should be collected, indicating the "allocation scope" and the liveness of the buffer allocated. The contents of a "lifetime trace" is implementation-defined


There are some more details on each step of the pipeline above.

### Collect the memory lifetime traces

This is the first stage that a developer can customize in merge-alloc. It should collect the lifetime traces for each of the mergable memref.alloc operation. An implementation of the lifetime trace collector should define which allocations are mergeable and find the allocation scopes of them. It should also implement a data structure to hold the detailed liveness of each buffers.

This step is abstracted in a `TraceCollectorFunc` function. The merge-alloc framework defines the abstract interfaces for lifetime trace collector and the collected traces as below:

```c++
/// abstract base class for lifetime of buffers in the same "allocation scope".
/// It should hold the lifetime informantion of buffers that are to be merged in
/// the same allocation in an "allocation scope". TraceCollectorFunc decides
/// which buffers are put into which "allocation scope".
class LifetimeTrace {
public:
  virtual ~LifetimeTrace() = default;
};

/// top level memory trace info for multiple scopes. Each key-value is the
///  "allocation scope" and the LifetimeTrace
struct MemoryTraceScopes {
  llvm::DenseMap<Operation *, std::unique_ptr<LifetimeTrace>> scopeToTraces;
  MemoryTraceScopes() = default;
};

using TraceCollectorFunc = std::function<FailureOr<MemoryTraceScopes>(
    Operation *, const BufferViewFlowAnalysis &,
    const MergeAllocationOptions &)>;
```

### Memory planning and scheduling

This step is abstracted in a `MemoryPlannerFunc` function. It accepts the `MemoryTraceScopes` collected by the previous step. For each allocation scope in `MemoryTraceScopes`, it decides the total merged allocation size and the offsets for each mergeable allocation inside of the allocation scope. The abstract interfaces are shown below:

```c++
/// the memory scheduling result for allocations in the same allocation scope.
/// allocation => offset map. All Operation* in the map should be
/// memref::AllocOp which are in the same LifetimeTrace.
struct MemorySchedule {
  size_t totalSize;
  llvm::DenseMap<Operation *, int64_t> allocToOffset;
  MemorySchedule() : totalSize{0} {}
};

using MemoryPlannerFunc = std::function<FailureOr<MemorySchedule>(
    Operation *, const LifetimeTrace &, const MergeAllocationOptions &)>;
```

### Rewriting the IR

Given the `MemorySchedule` of the previous step, this step rewrites the IR to create the merged allocation in each of the allocation scopes, to replace the mergable memref.alloc with views on the merged allocations with the offsets calculated in the `MemorySchedule`. This step is abstracted in a `MemoryMergeMutatorFunc` function.

```c++
using MemoryMergeMutatorFunc = std::function<LogicalResult(
    Operation *toplevel, Operation *scope, const MemorySchedule &,
    const MergeAllocationOptions &)>;
```


## Tick-based Implementation for merge-alloc

A tick-based implementation of merge-alloc in provided by default. The basic idea of the tick-based allocation merging is that

1. Each of the operations in a function is assigned a "tick". An operation with a smaller tick is expected to be executed before one with a larger tick
2. Collect the first referenced tick and the last referenced tick for each mergeable allocation. If a buffer is referenced in loops and branches, special handling is needed.
3. For each allocation scope, linearize the first referenced tick and the last referenced tick of mergeable allocations inside of it into a single linear timeline
4. Use a "static-memory-planner" to handle the linear timeline

### Basic concepts

In the context of tick-based merge-alloc, mergeable alloction and allocation scope are defined as follows

#### Mergeable alloction

The pass should only consider to merge a `memref.alloc` only if
 * the ownership of the memref does not escape from the function. That is, the current function is responsible to alloc and dealloc this memref
 * and, the allocated memref is contiguous and has static shape and identical layout.

In tick-based merge-alloc, we call these `memref.alloc` **mergeable** allocations.

The memrefs passed by function arguments, or returned by the function will be untouched by this optimization.

#### Allocation scopes

The transformation first needs to identify the allocation scopes, which are mlir operaions containing non-zero regions, and
 * implementing `AutomaticAllocationScope`
 * and is not `scf.for` (allocations in an `scf.for` can be hoisted to parent `AutomaticAllocationScope`)

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

There will be three allocation scopes as marked in the comments above. An allocation scope marks the position to insert the `single allocation buffer` after allocation merging. After the transformation, all "mergeable" `memref.alloc` will be merged to the `single allocation buffer` of the nearest ancestor `alloc scope`.

### Tick-based trace collection

walk()
Alias
Branch
Sort-malloc-free

### Static Memory planner


The transformantion is consist of an analysis sub-pass and a mutation sub-pass. For each `alloc scope`, the analysis sub-pass finds the lifetime of each mergeable `memref.alloc` belonging to the `alloc scope`. And given the lifetime of each allocation, a memory planning algorithm will be run to find the `single allocation buffer` size of each `alloc scope` and the `offset` for each mergeable allocation within its `single allocation buffer`. Based on the memory planning result, the mutation sub-pass transforms the IR to
1. insert `memref.alloc` at the front of `alloc scope` body for its `single allocation buffer`
2. replace mergeable `memref.alloc` with `memref.view` on its `alloc scope`'s `single allocation buffer`

Ticks are assigned on each operation in the `func.func` by a increasing counter with pre-order recursive walking of the IR, as the "execution tick" for each operation. The lifetime analysis pass will assign two integers for each mergeable allocations as the analysis result: `begin_tick` and `end_tick`, to indicate the first and last tick of the use of the allocated memref in the IR. There should be special handling for loop and branch ops (`RegionBranchOpInterface` or `LoopLikeOpInterface`) which references memrefs allocated in parent scopes, to avoid wrong reuse of buffers used in the loop.

The analysis result for each mergeable allocations will be an integer range `[begin_tick,end_tick]`, where `begin_tick <= end_tick`.

The collected ticks for each buffer will be processed by the memory planning algorithm. It should output the total size of the `single allocation buffers` for each `alloc scopes`, and the `offsets` for each individual mergeable buffers. The algorithm should also consider the locality of the buffer to use, when multiple buffer localtion candidates are available.
