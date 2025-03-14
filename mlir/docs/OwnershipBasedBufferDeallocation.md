# Ownership-based Buffer Deallocation

[TOC]

One-Shot Bufferize does not deallocate any buffers that it allocates. After
running One-Shot Bufferize, the resulting IR may have a number of `memref.alloc`
ops, but no `memref.dealloc` ops. Buffer dellocation is delegated to the
`-ownership-based-buffer-deallocation` pass.

On a high level, buffers are "owned" by a basic block. Ownership materializes
as an `i1` SSA value and can be thought of as "responsibility to deallocate". It
is conceptually similar to `std::unique_ptr` in C++.

There are few additional preprocessing and postprocessing passes that should be
run together with the ownership-based buffer deallocation pass. The recommended
compilation pipeline is as follows:

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

The entire deallocation pipeline (excluding `-one-shot-bufferize`) is exposed
as `-buffer-deallocation-pipeline`.

The ownership-based buffer deallocation pass processes operations implementing
`FunctionOpInterface` one-by-one without analysing the call-graph.
This means that there have to be [some rules](#function-boundary-abi) on how
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

## Function boundary ABI

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
deallocation pass is applied and for which the implementation is accessible are
modified by the pass such that the ABI is respected (i.e., buffer copies are
inserted when necessary).

## Inserting `bufferization.dealloc` operations

`bufferization.dealloc` and ownership indicators are the main abstractions in
the ownership-based buffer deallocation pass. `bufferization.dealloc`
deallocates all given buffers if the respective ownership indicator is set and
there is no aliasing buffer in the retain list.

![branch_example_pre_move](/includes/img/bufferization_dealloc_op.svg)

`bufferization.dealloc` operations are unconditionally inserted at the end of
each basic block (just before the terminator). The majority of the pass is about
finding the correct operands for this operation. There are three variadic
operand lists to be populated, the first contains all MemRef values that may
need to be deallocated, the second list contains their associated ownership
values (of `i1` type), and the third list contains MemRef values that are still
needed at a later point and should thus not be deallocated (e.g., yielded or
returned buffers).

`bufferization.dealloc` allows us to deal with any kind of aliasing behavior: it
lowers to runtime aliasing checks when not enough information can be collected
statically. When enough aliasing information is statically available, operands
or the entire op may fold away.

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

## Supported interfaces

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

## Limitations

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

## Example

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

The [semantics of the `bufferization.dealloc` operation](#bufferizationdealloc-bufferizationdeallocop)
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
```mlir
%0:2 = bufferization.dealloc (%m0, %m1 : memref<2xf32>, memref<5xf32>)
                          if (%cond0, %cond1)
                      retain (%r0, %r1 : memref<1xf32>, memref<2xf32>)
```
lowers to (simplified):
```mlir
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
```mlir
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
