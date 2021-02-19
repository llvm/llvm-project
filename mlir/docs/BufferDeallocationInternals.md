# Buffer Deallocation - Internals

This section covers the internal functionality of the BufferDeallocation
transformation. The transformation consists of several passes. The main pass
called BufferDeallocation can be applied via “-buffer-deallocation” on MLIR
programs.

## Requirements

In order to use BufferDeallocation on an arbitrary dialect, several
control-flow interfaces have to be implemented when using custom operations.
This is particularly important to understand the implicit control-flow
dependencies between different parts of the input program. Without implementing
the following interfaces, control-flow relations cannot be discovered properly
and the resulting program can become invalid:

* Branch-like terminators should implement the `BranchOpInterface` to query and
manipulate associated operands.
* Operations involving structured control flow have to implement the
`RegionBranchOpInterface` to model inter-region control flow.
* Terminators yielding values to their parent operation (in particular in the
scope of nested regions within `RegionBranchOpInterface`-based operations),
should implement the `ReturnLike` trait to represent logical “value returns”.

Example dialects that are fully compatible are the “std” and “scf” dialects
with respect to all implemented interfaces.

## Detection of Buffer Allocations

The first step of the BufferDeallocation transformation is to identify
manageable allocation operations that implement the `SideEffects` interface.
Furthermore, these ops need to apply the effect `MemoryEffects::Allocate` to a
particular result value while not using the resource
`SideEffects::AutomaticAllocationScopeResource` (since it is currently reserved
for allocations, like `Alloca` that will be automatically deallocated by a
parent scope). Allocations that have not been detected in this phase will not
be tracked internally, and thus, not deallocated automatically. However,
BufferDeallocation is fully compatible with “hybrid” setups in which tracked
and untracked allocations are mixed:

```mlir
func @mixedAllocation(%arg0: i1) {
   %0 = alloca() : memref<2xf32>  // aliases: %2
   %1 = alloc() : memref<2xf32>  // aliases: %2
   cond_br %arg0, ^bb1, ^bb2
^bb1:
  use(%0)
  br ^bb3(%0 : memref<2xf32>)
^bb2:
  use(%1)
  br ^bb3(%1 : memref<2xf32>)
^bb3(%2: memref<2xf32>):
  ...
}
```

Example of using a conditional branch with alloc and alloca. BufferDeallocation
can detect and handle the different allocation types that might be intermixed.

Note: the current version does not support allocation operations returning
multiple result buffers.

## Conversion from AllocOp to AllocaOp

The PromoteBuffersToStack-pass converts AllocOps to AllocaOps, if possible. In
some cases, it can be useful to use such stack-based buffers instead of
heap-based buffers. The conversion is restricted to several constraints like:

* Control flow
* Buffer Size
* Dynamic Size

If a buffer is leaving a block, we are not allowed to convert it into an
alloca. If the size of the buffer is large, we could convert it, but regarding
stack overflow, it makes sense to limit the size of these buffers and only
convert small ones. The size can be set via a pass option. The current default
value is 1KB. Furthermore, we can not convert buffers with dynamic size, since
the dimension is not known a priori.

## Movement and Placement of Allocations

Using the buffer hoisting pass, all buffer allocations are moved as far upwards
as possible in order to group them and make upcoming optimizations easier by
limiting the search space. Such a movement is shown in the following graphs.
In addition, we are able to statically free an alloc, if we move it into a
dominator of all of its uses. This simplifies further optimizations (e.g.
buffer fusion) in the future. However, movement of allocations is limited by
external data dependencies (in particular in the case of allocations of
dynamically shaped types). Furthermore, allocations can be moved out of nested
regions, if necessary. In order to move allocations to valid locations with
respect to their uses only, we leverage Liveness information.

The following code snippets shows a conditional branch before running the
BufferHoisting pass:

![branch_example_pre_move](/includes/img/branch_example_pre_move.svg)

```mlir
func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = alloc() : memref<2xf32>  // aliases: %1
  use(%0)
  br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):  // %1 could be %0 or %arg1
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
```

Applying the BufferHoisting pass on this program results in the following piece
of code:

![branch_example_post_move](/includes/img/branch_example_post_move.svg)

```mlir
func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>  // moved to bb0
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  br ^bb3(%arg1 : memref<2xf32>)
^bb2:
   use(%0)
   br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
```

The alloc is moved from bb2 to the beginning and it is passed as an argument to
bb3.

The following example demonstrates an allocation using dynamically shaped
types. Due to the data dependency of the allocation to %0, we cannot move the
allocation out of bb2 in this case:

```mlir
func @condBranchDynamicType(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  br ^bb3(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = alloc(%0) : memref<?xf32>   // cannot be moved upwards to the data
                                   // dependency to %0
  use(%1)
  br ^bb3(%1 : memref<?xf32>)
^bb3(%2: memref<?xf32>):
  "linalg.copy"(%2, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  return
}
```

## Introduction of Copies

In order to guarantee that all allocated buffers are freed properly, we have to
pay attention to the control flow and all potential aliases a buffer allocation
can have. Since not all allocations can be safely freed with respect to their
aliases (see the following code snippet), it is often required to introduce
copies to eliminate them. Consider the following example in which the
allocations have already been placed:

```mlir
func @branch(%arg0: i1) {
  %0 = alloc() : memref<2xf32>  // aliases: %2
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  %1 = alloc() : memref<2xf32>  // resides here for demonstration purposes
                                // aliases: %2
  br ^bb3(%1 : memref<2xf32>)
^bb2:
  use(%0)
  br ^bb3(%0 : memref<2xf32>)
^bb3(%2: memref<2xf32>):
  …
  return
}
```

The first alloc can be safely freed after the live range of its post-dominator
block (bb3). The alloc in bb1 has an alias %2 in bb3 that also keeps this
buffer alive until the end of bb3. Since we cannot determine the actual
branches that will be taken at runtime, we have to ensure that all buffers are
freed correctly in bb3 regardless of the branches we will take to reach the
exit block. This makes it necessary to introduce a copy for %2, which allows us
to free %alloc0 in bb0 and %alloc1 in bb1. Afterwards, we can continue
processing all aliases of %2 (none in this case) and we can safely free %2 at
the end of the sample program. This sample demonstrates that not all
allocations can be safely freed in their associated post-dominator blocks.
Instead, we have to pay attention to all of their aliases.

Applying the BufferDeallocation pass to the program above yields the following
result:

```mlir
func @branch(%arg0: i1) {
  %0 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  %1 = alloc() : memref<2xf32>
  %3 = alloc() : memref<2xf32>  // temp copy for %1
  "linalg.copy"(%1, %3) : (memref<2xf32>, memref<2xf32>) -> ()
  dealloc %1 : memref<2xf32> // %1 can be safely freed here
  br ^bb3(%3 : memref<2xf32>)
^bb2:
  use(%0)
  %4 = alloc() : memref<2xf32>  // temp copy for %0
  "linalg.copy"(%0, %4) : (memref<2xf32>, memref<2xf32>) -> ()
  br ^bb3(%4 : memref<2xf32>)
^bb3(%2: memref<2xf32>):
  …
  dealloc %2 : memref<2xf32> // free temp buffer %2
  dealloc %0 : memref<2xf32> // %0 can be safely freed here
  return
}
```

Note that a temporary buffer for %2 was introduced to free all allocations
properly. Note further that the unnecessary allocation of %3 can be easily
removed using one of the post-pass transformations.

Reconsider the previously introduced sample demonstrating dynamically shaped
types:

```mlir
func @condBranchDynamicType(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  br ^bb3(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = alloc(%0) : memref<?xf32>  // aliases: %2
  use(%1)
  br ^bb3(%1 : memref<?xf32>)
^bb3(%2: memref<?xf32>):
  "linalg.copy"(%2, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  return
}
```

In the presence of DSTs, we have to parameterize the allocations with
additional dimension information of the source buffers, we want to copy from.
BufferDeallocation automatically introduces all required operations to extract
dimension specifications and wires them with the associated allocations:

```mlir
func @condBranchDynamicType(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3 : index)
^bb1:
  %c0 = constant 0 : index
  %0 = dim %arg1, %c0 : memref<?xf32>   // dimension operation to parameterize
                                        // the following temp allocation
  %1 = alloc(%0) : memref<?xf32>
  "linalg.copy"(%arg1, %1) : (memref<?xf32>, memref<?xf32>) -> ()
  br ^bb3(%1 : memref<?xf32>)
^bb2(%2: index):
  %3 = alloc(%2) : memref<?xf32>
  use(%3)
  %c0_0 = constant 0 : index
  %4 = dim %3, %c0_0 : memref<?xf32>  // dimension operation to parameterize
                                      // the following temp allocation
  %5 = alloc(%4) : memref<?xf32>
  "linalg.copy"(%3, %5) : (memref<?xf32>, memref<?xf32>) -> ()
  dealloc %3 : memref<?xf32>  // %3 can be safely freed here
  br ^bb3(%5 : memref<?xf32>)
^bb3(%6: memref<?xf32>):
  "linalg.copy"(%6, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  dealloc %6 : memref<?xf32>  // %6 can be safely freed here
  return
}
```

BufferDeallocation performs a fix-point iteration taking all aliases of all
tracked allocations into account. We initialize the general iteration process
using all tracked allocations and their associated aliases. As soon as we
encounter an alias that is not properly dominated by our allocation, we mark
this alias as _critical_ (needs to be freed and tracked by the internal
fix-point iteration). The following sample demonstrates the presence of
critical and non-critical aliases:

![nested_branch_example_pre_move](/includes/img/nested_branch_example_pre_move.svg)

```mlir
func @condBranchDynamicTypeNested(
  %arg0: i1,
  %arg1: memref<?xf32>,  // aliases: %3, %4
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  br ^bb6(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = alloc(%0) : memref<?xf32>   // cannot be moved upwards due to the data
                                   // dependency to %0
                                   // aliases: %2, %3, %4
  use(%1)
  cond_br %arg0, ^bb3, ^bb4
^bb3:
  br ^bb5(%1 : memref<?xf32>)
^bb4:
  br ^bb5(%1 : memref<?xf32>)
^bb5(%2: memref<?xf32>):  // non-crit. alias of %1, since %1 dominates %2
  br ^bb6(%2 : memref<?xf32>)
^bb6(%3: memref<?xf32>):  // crit. alias of %arg1 and %2 (in other words %1)
  br ^bb7(%3 : memref<?xf32>)
^bb7(%4: memref<?xf32>):  // non-crit. alias of %3, since %3 dominates %4
  "linalg.copy"(%4, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  return
}
```

Applying BufferDeallocation yields the following output:

![nested_branch_example_post_move](/includes/img/nested_branch_example_post_move.svg)

```mlir
func @condBranchDynamicTypeNested(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cond_br %arg0, ^bb1, ^bb2(%arg3 : index)
^bb1:
  %c0 = constant 0 : index
  %d0 = dim %arg1, %c0 : memref<?xf32>
  %5 = alloc(%d0) : memref<?xf32>  // temp buffer required due to alias %3
  "linalg.copy"(%arg1, %5) : (memref<?xf32>, memref<?xf32>) -> ()
  br ^bb6(%5 : memref<?xf32>)
^bb2(%0: index):
  %1 = alloc(%0) : memref<?xf32>
  use(%1)
  cond_br %arg0, ^bb3, ^bb4
^bb3:
  br ^bb5(%1 : memref<?xf32>)
^bb4:
  br ^bb5(%1 : memref<?xf32>)
^bb5(%2: memref<?xf32>):
  %c0_0 = constant 0 : index
  %d1 = dim %2, %c0_0 : memref<?xf32>
  %6 = alloc(%d1) : memref<?xf32>  // temp buffer required due to alias %3
  "linalg.copy"(%1, %6) : (memref<?xf32>, memref<?xf32>) -> ()
  dealloc %1 : memref<?xf32>
  br ^bb6(%6 : memref<?xf32>)
^bb6(%3: memref<?xf32>):
  br ^bb7(%3 : memref<?xf32>)
^bb7(%4: memref<?xf32>):
  "linalg.copy"(%4, %arg2) : (memref<?xf32>, memref<?xf32>) -> ()
  dealloc %3 : memref<?xf32>  // free %3, since %4 is a non-crit. alias of %3
  return
}
```

Since %3 is a critical alias, BufferDeallocation introduces an additional
temporary copy in all predecessor blocks. %3 has an additional (non-critical)
alias %4 that extends the live range until the end of bb7. Therefore, we can
free %3 after its last use, while taking all aliases into account. Note that %4
 does not need to be freed, since we did not introduce a copy for it.

The actual introduction of buffer copies is done after the fix-point iteration
has been terminated and all critical aliases have been detected. A critical
alias can be either a block argument or another value that is returned by an
operation. Copies for block arguments are handled by analyzing all predecessor
blocks. This is primarily done by querying the `BranchOpInterface` of the
associated branch terminators that can jump to the current block. Consider the
following example which involves a simple branch and the critical block
argument %2:

```mlir
  custom.br ^bb1(..., %0, : ...)
  ...
  custom.br ^bb1(..., %1, : ...)
  ...
^bb1(%2: memref<2xf32>):
  ...
```

The `BranchOpInterface` allows us to determine the actual values that will be
passed to block bb1 and its argument %2 by analyzing its predecessor blocks.
Once we have resolved the values %0 and %1 (that are associated with %2 in this
sample), we can introduce a temporary buffer and clone its contents into the
new buffer. Afterwards, we rewire the branch operands to use the newly
allocated buffer instead. However, blocks can have implicitly defined
predecessors by parent ops that implement the `RegionBranchOpInterface`. This
can be the case if this block argument belongs to the entry block of a region.
In this setting, we have to identify all predecessor regions defined by the
parent operation. For every region, we need to get all terminator operations
implementing the `ReturnLike` trait, indicating that they can branch to our
current block. Finally, we can use a similar functionality as described above
to add the temporary copy. This time, we can modify the terminator operands
directly without touching a high-level interface.

Consider the following inner-region control-flow sample that uses an imaginary
“custom.region_if” operation. It either executes the “then” or “else” region
and always continues to the “join” region. The “custom.region_if_yield”
operation returns a result to the parent operation. This sample demonstrates
the use of the `RegionBranchOpInterface` to determine predecessors in order to
infer the high-level control flow:

```mlir
func @inner_region_control_flow(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = custom.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>)
   then(%arg2 : memref<?x?xf32>) {  // aliases: %arg4, %1
    custom.region_if_yield %arg2 : memref<?x?xf32>
   } else(%arg3 : memref<?x?xf32>) {  // aliases: %arg4, %1
    custom.region_if_yield %arg3 : memref<?x?xf32>
   } join(%arg4 : memref<?x?xf32>) {  // aliases: %1
    custom.region_if_yield %arg4 : memref<?x?xf32>
   }
  return %1 : memref<?x?xf32>
}
```

![region_branch_example_pre_move](/includes/img/region_branch_example_pre_move.svg)

Non-block arguments (other values) can become aliases when they are returned by
dialect-specific operations. BufferDeallocation supports this behavior via the
`RegionBranchOpInterface`. Consider the following example that uses an “scf.if”
operation to determine the value of %2 at runtime which creates an alias:

```mlir
func @nested_region_control_flow(%arg0 : index, %arg1 : index) -> memref<?x?xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>   // %2 will be an alias of %1
  } else {
    %3 = alloc(%arg0, %arg1) : memref<?x?xf32>  // nested allocation in a div.
                                                // branch
    use(%3)
    scf.yield %1 : memref<?x?xf32>   // %2 will be an alias of %1
  }
  return %2 : memref<?x?xf32>
}
```

In this example, a dealloc is inserted to release the buffer within the else
block since it cannot be accessed by the remainder of the program. Accessing
the `RegionBranchOpInterface`, allows us to infer that %2 is a non-critical
alias of %1 which does not need to be tracked.

```mlir
func @nested_region_control_flow(%arg0: index, %arg1: index) -> memref<?x?xf32> {
    %0 = cmpi "eq", %arg0, %arg1 : index
    %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
    %2 = scf.if %0 -> (memref<?x?xf32>) {
      scf.yield %1 : memref<?x?xf32>
    } else {
      %3 = alloc(%arg0, %arg1) : memref<?x?xf32>
      use(%3)
      dealloc %3 : memref<?x?xf32>  // %3 can be safely freed here
      scf.yield %1 : memref<?x?xf32>
    }
    return %2 : memref<?x?xf32>
}
```

Analogous to the previous case, we have to detect all terminator operations in
all attached regions of “scf.if” that provides a value to its parent operation
(in this sample via scf.yield). Querying the `RegionBranchOpInterface` allows
us to determine the regions that “return” a result to their parent operation.
Like before, we have to update all `ReturnLike` terminators as described above.
Reconsider a slightly adapted version of the “custom.region_if” example from
above that uses a nested allocation:

```mlir
func @inner_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = custom.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>)
   then(%arg2 : memref<?x?xf32>) {  // aliases: %arg4, %1
    custom.region_if_yield %arg2 : memref<?x?xf32>
   } else(%arg3 : memref<?x?xf32>) {
    %2 = alloc(%arg0, %arg1) : memref<?x?xf32>  // aliases: %arg4, %1
    custom.region_if_yield %2 : memref<?x?xf32>
   } join(%arg4 : memref<?x?xf32>) {  // aliases: %1
    custom.region_if_yield %arg4 : memref<?x?xf32>
   }
  return %1 : memref<?x?xf32>
}
```

Since the allocation %2 happens in a divergent branch and cannot be safely
deallocated in a post-dominator, %arg4 will be considered a critical alias.
Furthermore, %arg4 is returned to its parent operation and has an alias %1.
This causes BufferDeallocation to introduce additional copies:

```mlir
func @inner_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = custom.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>)
   then(%arg2 : memref<?x?xf32>) {
    %c0 = constant 0 : index  // determine dimension extents for temp allocation
    %2 = dim %arg2, %c0 : memref<?x?xf32>
    %c1 = constant 1 : index
    %3 = dim %arg2, %c1 : memref<?x?xf32>
    %4 = alloc(%2, %3) : memref<?x?xf32>  // temp buffer required due to critic.
                                          // alias %arg4
    linalg.copy(%arg2, %4) : memref<?x?xf32>, memref<?x?xf32>
    custom.region_if_yield %4 : memref<?x?xf32>
   } else(%arg3 : memref<?x?xf32>) {
    %2 = alloc(%arg0, %arg1) : memref<?x?xf32>
    %c0 = constant 0 : index  // determine dimension extents for temp allocation
    %3 = dim %2, %c0 : memref<?x?xf32>
    %c1 = constant 1 : index
    %4 = dim %2, %c1 : memref<?x?xf32>
    %5 = alloc(%3, %4) : memref<?x?xf32>  // temp buffer required due to critic.
                                          // alias %arg4
    linalg.copy(%2, %5) : memref<?x?xf32>, memref<?x?xf32>
    dealloc %2 : memref<?x?xf32>
    custom.region_if_yield %5 : memref<?x?xf32>
   } join(%arg4: memref<?x?xf32>) {
    %c0 = constant 0 : index  // determine dimension extents for temp allocation
    %2 = dim %arg4, %c0 : memref<?x?xf32>
    %c1 = constant 1 : index
    %3 = dim %arg4, %c1 : memref<?x?xf32>
    %4 = alloc(%2, %3) : memref<?x?xf32>  // this allocation will be removed by
                                          // applying the copy removal pass
    linalg.copy(%arg4, %4) : memref<?x?xf32>, memref<?x?xf32>
    dealloc %arg4 : memref<?x?xf32>
    custom.region_if_yield %4 : memref<?x?xf32>
   }
  dealloc %0 : memref<?x?xf32>  // %0 can be safely freed here
  return %1 : memref<?x?xf32>
}
```

## Placement of Deallocs

After introducing allocs and copies, deallocs have to be placed to free
allocated memory and avoid memory leaks. The deallocation needs to take place
after the last use of the given value. The position can be determined by
calculating the common post-dominator of all values using their remaining
non-critical aliases. A special-case is the presence of back edges: since such
edges can cause memory leaks when a newly allocated buffer flows back to
another part of the program. In these cases, we need to free the associated
buffer instances from the previous iteration by inserting additional deallocs.

Consider the following “scf.for” use case containing a nested structured
control-flow if:

```mlir
func @loop_nested_if(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %1 = cmpi "eq", %i, %ub : index
    %2 = scf.if %1 -> (memref<2xf32>) {
      %3 = alloc() : memref<2xf32>  // makes %2 a critical alias due to a
                                    // divergent allocation
      use(%3)
      scf.yield %3 : memref<2xf32>
    } else {
      scf.yield %iterBuf : memref<2xf32>
    }
    scf.yield %2 : memref<2xf32>
  }
  "linalg.copy"(%0, %res) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
```

In this example, the _then_ branch of the nested “scf.if” operation returns a
newly allocated buffer.

Since this allocation happens in the scope of a divergent branch, %2 becomes a
critical alias that needs to be handled. As before, we have to insert
additional copies to eliminate this alias using copies of %3 and %iterBuf. This
guarantees that %2 will be a newly allocated buffer that is returned in each
iteration. However, “returning” %2 to its alias %iterBuf turns %iterBuf into a
critical alias as well. In other words, we have to create a copy of %2 to pass
it to %iterBuf. Since this jump represents a back edge, and %2 will always be a
new buffer, we have to free the buffer from the previous iteration to avoid
memory leaks:

```mlir
func @loop_nested_if(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %4 = alloc() : memref<2xf32>
  "linalg.copy"(%buf, %4) : (memref<2xf32>, memref<2xf32>) -> ()
  %0 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %4) -> memref<2xf32> {
    %1 = cmpi "eq", %i, %ub : index
    %2 = scf.if %1 -> (memref<2xf32>) {
      %3 = alloc() : memref<2xf32> // makes %2 a critical alias
      use(%3)
      %5 = alloc() : memref<2xf32> // temp copy due to crit. alias %2
      "linalg.copy"(%3, %5) : memref<2xf32>, memref<2xf32>
      dealloc %3 : memref<2xf32>
      scf.yield %5 : memref<2xf32>
    } else {
      %6 = alloc() : memref<2xf32> // temp copy due to crit. alias %2
      "linalg.copy"(%iterBuf, %6) : memref<2xf32>, memref<2xf32>
      scf.yield %6 : memref<2xf32>
    }
    %7 = alloc() : memref<2xf32> // temp copy due to crit. alias %iterBuf
    "linalg.copy"(%2, %7) : memref<2xf32>, memref<2xf32>
    dealloc %2 : memref<2xf32>
    dealloc %iterBuf : memref<2xf32> // free backedge iteration variable
    scf.yield %7 : memref<2xf32>
  }
  "linalg.copy"(%0, %res) : (memref<2xf32>, memref<2xf32>) -> ()
  dealloc %0 : memref<2xf32> // free temp copy %0
  return
}
```

Example for loop-like control flow. The CFG contains back edges that have to be
handled to avoid memory leaks. The bufferization is able to free the backedge
iteration variable %iterBuf.

## Private Analyses Implementations

The BufferDeallocation transformation relies on one primary control-flow
analysis: BufferPlacementAliasAnalysis. Furthermore, we also use dominance and
liveness to place and move nodes. The liveness analysis determines the live
range of a given value. Within this range, a value is alive and can or will be
used in the course of the program. After this range, the value is dead and can
be discarded - in our case, the buffer can be freed. To place the allocs, we
need to know from which position a value will be alive. The allocs have to be
placed in front of this position. However, the most important analysis is the
alias analysis that is needed to introduce copies and to place all
deallocations.

# Post Phase

In order to limit the complexity of the BufferDeallocation transformation, some
tiny code-polishing/optimization transformations are not applied on-the-fly
during placement. Currently, there is only the CopyRemoval transformation to
remove unnecessary copy and allocation operations.

Note: further transformations might be added to the post-pass phase in the
future.

## CopyRemoval Pass

A common pattern that arises during placement is the introduction of
unnecessary temporary copies that are used instead of the original source
buffer. For this reason, there is a post-pass transformation that removes these
allocations and copies via `-copy-removal`. This pass, besides removing
unnecessary copy operations, will also remove the dead allocations and their
corresponding deallocation operations. The CopyRemoval pass can currently be
applied to operations that implement the `CopyOpInterface` in any of these two
situations which are

* reusing the source buffer of the copy operation.
* reusing the target buffer of the copy operation.

## Reusing the Source Buffer of the Copy Operation

In this case, the source of the copy operation can be used instead of target.
The unused allocation and deallocation operations that are defined for this
copy operation are also removed. Here is a working example generated by the
BufferDeallocation pass that allocates a buffer with dynamic size. A deeper
analysis of this sample reveals that the highlighted operations are redundant
and can be removed.

```mlir
func @dynamic_allocation(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %7 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %c0_0 = constant 0 : index
  %8 = dim %7, %c0_0 : memref<?x?xf32>
  %c1_1 = constant 1 : index
  %9 = dim %7, %c1_1 : memref<?x?xf32>
  %10 = alloc(%8, %9) : memref<?x?xf32>
  linalg.copy(%7, %10) : memref<?x?xf32>, memref<?x?xf32>
  dealloc %7 : memref<?x?xf32>
  return %10 : memref<?x?xf32>
}
```

Will be transformed to:

```mlir
func @dynamic_allocation(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %7 = alloc(%arg0, %arg1) : memref<?x?xf32>
  %c0_0 = constant 0 : index
  %8 = dim %7, %c0_0 : memref<?x?xf32>
  %c1_1 = constant 1 : index
  %9 = dim %7, %c1_1 : memref<?x?xf32>
  return %7 : memref<?x?xf32>
}
```

In this case, the additional copy %10 can be replaced with its original source
buffer %7. This also applies to the associated dealloc operation of %7.

To limit the complexity of this transformation, it only removes copy operations
when the following constraints are met:

* The copy operation, the defining operation for the target value, and the
deallocation of the source value lie in the same block.
* There are no users/aliases of the target value between the defining operation
of the target value and its copy operation.
* There are no users/aliases of the source value between its associated copy
operation and the deallocation of the source value.

## Reusing the Target Buffer of the Copy Operation

In this case, the target buffer of the copy operation can be used instead of
its source. The unused allocation and deallocation operations that are defined
for this copy operation are also removed.

Consider the following example where a generic linalg operation writes the
result to %temp and then copies %temp to %result. However, these two operations
can be merged into a single step. Copy removal removes the copy operation and
%temp, and replaces the uses of %temp with %result:

```mlir
func @reuseTarget(%arg0: memref<2xf32>, %result: memref<2xf32>){
  %temp = alloc() : memref<2xf32>
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %temp {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  "linalg.copy"(%temp, %result) : (memref<2xf32>, memref<2xf32>) -> ()
  dealloc %temp : memref<2xf32>
  return
}
```

Will be transformed to:

```mlir
func @reuseTarget(%arg0: memref<2xf32>, %result: memref<2xf32>){
  linalg.generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]} %arg0, %result {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }: memref<2xf32>, memref<2xf32>
  return
}
```

Like before, several constraints to use the transformation apply:

* The copy operation, the defining operation of the source value, and the
deallocation of the source value lie in the same block.
* There are no users/aliases of the target value between the defining operation
of the source value and the copy operation.
* There are no users/aliases of the source value between the copy operation and
the deallocation of the source value.

## Known Limitations

BufferDeallocation introduces additional copies using allocations from the
“memref” dialect (“memref.alloc”). Analogous, all deallocations use the
“memref” dialect-free operation “memref.dealloc”. The actual copy process is
realized using “linalg.copy”. Furthermore, buffers are essentially immutable
after their creation in a block. Another limitations are known in the case
using unstructered control flow.
