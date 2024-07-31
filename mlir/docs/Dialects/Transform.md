# Transform Dialect

Fine-grain transformation control dialect. See [tutorial](../Tutorials/transform) for more introductory information.

[TOC]

## Overview

This dialect provides operations that can be used to control transformation
of the IR using a different portion of the IR. It refers to the IR being
transformed as payload IR, and to the IR guiding the transformation as
transform IR.

The main use case for this dialect is orchestrating fine-grain transformations
on individual IR objects (operations or values) or sets thereof. For example, it
may involve finding loop-like operations with specific properties (e.g., large
size) in the payload IR, applying loop tiling to those and only those
operations, and then applying loop unrolling to the inner loops produced by the
previous transformations. As such, it is not intended as a replacement for the
pass infrastructure, nor for the pattern rewriting infrastructure. In the most
common case, the transform IR will be processed and applied to the payload IR by
a pass. Transformations expressed by the Transform dialect may be implemented
using the pattern infrastructure or any other relevant MLIR component.

The following IR gives a rough idea of what the operations in this dialect
may look like without using actually existing operations:

```mlir
%0 = transform.loop.find { size > 42 } : !transform.interface<tileable>
%1 = transform.compute_trailing_tile_size %0 : !transform.param<index>
%2:2 = transform.loop.tile %0 tile_sizes(1, 4, %1)
      : (!transform.interface<tileable>)
     -> (!transform.op<loop>, !transform.op<loop>)
%3 = transform.get_op_result [0] %2#0 : !transform.any_value
transform.assign_to_fast_memory %3
transform.loop.unroll %1#1 : !transform.op<loop>
```

The values used in the Transform dialect may correspond to:

  * sets of operations in the payload IR;

  * sets of values in the payload IR;

  * sets of parameters (attributes) known at the execution time of the
    transform dialect.

The former two kinds of values are also referred to as operation and value
*handles*, respectively. In the example above, `%0` corresponds to the set of
loops found in the payload IR that satisfy the condition, and `%2` correspond to
groups of outer and inner loops, respectively, produced by the tiling
transformation. `%3` corresponds to a set of values that are produced by the
outer loops after tiling. `%1` corresponds to a list of tile sizes selected for
each of the operations that `%0` corresponds to.

An operation handle such as `%0` may be associated with multiple payload
operations. This is conceptually a set of operations and no assumptions should
be made about the order of ops unless specified otherwise by the operation.
Similarly, a value handle such as `%3` may be associated with a set of payload
IR values. Transform dialect operations may take as operands and produce an
arbitrary combination of values representing handles and parameters. Most
Transform IR ops support operand values that are mapped to multiple payload
objects. They usually apply the respective transformation for every mapped
object ("batched execution"). Deviations from this convention are described in
the documentation of Transform IR ops.

Parameters, such as `%1` in the above example, have two logical roles in
transform IR. In parameter based control, they carry the values needed to
execute the explicit control defined by the transforms, for example:

```mlir
%0 = transform.match.structured.rank %linalg_op_handle : !transform.param<index>
%1 = transform.param.constant 3 : i32 -> !transform.param<index>
transform.execute_if_cmpi eq %0, %1 : !transform.param<index>, !transform.param<index>
// Some nested body of transform ops
```

Alternatively, parameters can associate with the payload IR where the specific
value at execution time has no bearing on the execution of the transform IR. In
other words, parameters can either associate with the transform IR or the
payload IR.  Note that it is generally discouraged to use parameters containing
arbitrary attributes within transform control. Parameter based control should
try to be explicitly typed when possible.

The transform IR values have transform IR types, which should implement exactly one of:

  * [TransformHandleTypeInterface](#transformhandletypeinterface-transformhandletypeinterface),

  * [TransformValueHandleTypeInterface](#transformvaluehandletypeinterface-transformvaluehandletypeinterface),

  * [TransformParamTypeInterface](#transformparamtypeinterface-transformparamtypeinterface).

The goal of these type interfaces, beyond providing a common base for accepted
types, is to verify the properties of the associated objects. For example, a
handle type interface implementation may check whether all associated payload IR
operations implement the "TileableOp" interface or have a specific "loop" kind.
Similarly, a value handle type interface implementation may check if the
associated payload IR values are block arguments or have a specific type, or a
parameter type interface may check whether the associated attributes contain
non-negative integer values. These properties are used to statically indicate
 pre- and post-conditions of a transformation connected to a Transform dialect
operation. The conditions are verified when payload objects operations are first
associated with a transform handle. By convention, Transform dialect operations
are expected to indicate narrow preconditions for their operands by enforcing
operand type constraints in the their definitions and verifiers. On the
contrary, operations are expected to have few constraints on their results.
Specific instances of a transform operation can then be created with a more
restricted result type than the constraint in the operation (e.g., the "find"
operation only constrains the result type to be a transform IR type while its
concrete instance can have a type with stricter constraints such as implementing
the "tilable" interface). The verification will then happen at transform
execution time. This approach allows one to capture payload IR operation
properties in the transform IR without resorting to excessive use of type casts
or coupling dialect extensions between themselves. It is a trade-off between
verbosity/complexity and static hardening, which can be revised in the future.

Overall, Transform IR ops are expected to be contained in a single top-level
op. Such top-level ops specify how to apply the transformations described
by the operations they contain, e.g., `transform.sequence` executes
transformations one by one and fails if any of them fails. Such ops are
expected to have the `PossibleTopLevelTransformOpTrait` and may be used
without arguments.

A program transformation expressed using the Transform dialect can be
programmatically triggered by calling:

```c++
LogicalResult transform::applyTransforms(
    Operation *payloadRoot,
    const RaggedArray<transform::MappedValue> &extraMappings,
    TransformOpInterface transform,
    const TransformOptions &options);
```

that applies the transformations specified by the top-level `transform` to
payload IR contained in `payloadRoot`. The payload root operation will be
associated with the first argument of the entry block of the top-level transform
op. This block may have additional arguments, handles or parameters. They will
be associated with values provided as `extraMappings`. The call will report an
error and return if the wrong number of mappings is provided.

## Dialect Extension Mechanism

This dialect is designed to be extensible, that is, clients of this dialect
are allowed to inject additional operations into this dialect using the
`TransformDialectExtension` mechanism. This allows the dialect to avoid a
dependency on the implementation of the transformation as well as to avoid
introducing dialect-specific transform dialects. In the example above,
the operations may have been injected by a notional `loop` dialect rather
than defined in this dialect, hence the common prefix.

It is recommended to prefix injected operations with one or several
dot-separated words that indicate which extension adds them. For
dialect-specific transformations, the prefix is naturally the name of the
dialect, e.g., `transform.affine.reschedule`. For dialect-agnostic
transformations (typically implemented using interfaces), the prefix may
be derived from the interface name or from a common concept, e.g.,
`transform.loop.tile` may apply to any loop-like operation that implements
`TileableOpInterface`. The C++ classes for the dialect extension should
include the prefix in their name, e.g., `AffineTransformDialectExtension` or
`LoopTransformDialectExtension` in the cases above. Unprefixed operation
names are reserved for ops defined directly in the Transform dialect.

Operations injected into the dialect must:

  * Implement the `TransformOpInterface` to execute the corresponding
    transformation on the payload IR.

  * Implement the `MemoryEffectsOpInterface` to annotate the effects of
    the transform IR operation on the payload IR as well as on the mapping
    between transform IR values and payload IR operations. See below for
    the description of available effects.

The presence of interface implementations is checked at runtime when the
dialect is loaded to allow for those implementations to be supplied by
separate dialect extensions if desired.

Similarly to operations, additional types can be injected into the dialect using
the same extension mechanism. The types must:

  * Implement exactly one of `TransformHandleTypeInterface`,
    `TransformValueHandleTypeInterface`, `TransformParamTypeInterface`.

## Side Effects

The Transform dialect relies on MLIR side effect modelling to enable
optimization of the transform IR. More specifically, it provides several
side effect resource objects and expects operations to describe their
effects on these resources.

  * `TransformMappingResource` - side effect resource corresponding to the
    mapping between transform IR values and payload IR operations.

    - An `Allocate` effect from this resource means creating a new mapping
      entry, it is always accompanied by a `Write` effect.

    - A `Read` effect from this resource means accessing the mapping.

    - A `Free` effect on this resource indicates the removal of the mapping
      entry, typically after a transformation that modifies the payload IR
      operations associated with one of the transform IR operation's
      operands. It is always accompanied by a `Read` effect.

  * `PayloadIRResource` - side effect resource corresponding to the payload
    IR itself.

    - A `Read` effect from this resource means accessing the payload IR.

    - A `Write` effect on this resource means mutating the payload IR. It is
      almost always accompanied by a `Read`.

The typical flow of values in the transform IR is as follows. Most
operations produce new transform IR values and immediately associate them
with a list of payload IR operations. This corresponds to `Allocate` and
`Write` effects on the `TransformMappingResource`, and often requires at
least a `Read` effect on the `PayloadIRResource`. Transform operations that
only inspect the payload IR to produce new handles are usually limited to
these effects on their operands. Transform operations that mutate the
payload IR are thought to _consume_ the handles provided as operands, that
is have the `Read` and `Free` effects on them. As with the usual memory
effects, using a value after it was freed is incorrect. In case of the
transform IR, this value is likely associated with payload IR operations
that were modified or even removed by the transformation, so it is
meaningless to refer to them. When further transformations are desired, the
transform operations can return _new_ handles that can be read or consumed
by subsequent operations.

## Execution Model

The transformation starts at the user-specified top-level transform IR
operation and applies to some user-specified payload IR scope, identified by
the payload IR op that contains the IR to transform. It is the
responsibility of the user to properly select the scope and/or to avoid the
transformations to modify the IR outside of the given scope. The top-level
transform IR operation may contain further transform operations and execute
them in the desired order.

Transformation application functions produce a tri-state status:

- success;
- recoverable (silenceable) failure;
- irrecoverable failure.

Transformation container operations may intercept recoverable failures and
perform the required recovery steps thus succeeding themselves. On
the other hand, they must propagate irrecoverable failures. For such
failures, the diagnostics are emitted immediately whereas their emission is
postponed for recoverable failures. Transformation container operations may
also fail to recover from a theoretically recoverable failure, in which case
they can either propagate it to their parent or emit the diagnostic and turn
the failure into an irrecoverable one. A recoverable failure produced by
applying the top-level transform IR operation is considered irrecoverable.

Transformation container operations are allowed to "step over" some nested
operations if the application of some previous operation produced a failure.
This can be conceptually thought of as having a global "recoverable error
register" that is read/write accessed by each transform operation as a side
effect. The transformation is skipped if the register already contains an
error description, and the control flow proceeds to the following operation.

Note that a silenceable failure, if emitted, is a compiler _error_ rather
than a warning. Transformations are expected to produce silenceable failures
if they haven't yet modified the payload IR, i.e. when reporting a
precondition failure, and an irrecoverable failure when they modified the IR
in a way that is contrary to the semantics of the transform operation or
would fail a postcondition. Some "navigation" operations that identify
payload IR targets for the following transformation may have a conceptual
"failure to match" that is considered a successful execution in the
execution model but results in handles associated with empty payload IR
operation lists.

## Handle Invalidation

The execution model of the Transform dialect allows a payload IR operation to be
associated with _multiple_ handles as well as nested payload IR operations to be
associated with different handles. Similarly, a payload IR value may be
associated with multiple transform IR value handles. When a transform IR
operation consumes a handle, it usually indicates that the corresponding payload
IR object was destroyed and should no longer be referenced. Transform IR handles
that _may_ be pointing to an erased payload IR object are _invalidated_. The
mere presence of an invalidated handle in the transform IR is not a problem, but
_using_ it results in undefined behavior. Invalidated handles can be thought of
as dangling pointers. Note that the _entire_ handle is invalidated, even if some
of the payload IR objects associated with it remain live.

The following handle invalidation rules apply.

  * When an operation handle is consumed, are invalidated:

    - operation handles associated with one of the payload operations that the
      consumed handle is associated with;

    - operation handles associated with one of the operations _nested_ in the
      payload operations described above;

    - value handles associated with any result of any operation described above;

    - value handles associated with any argument of a block contained in a
      region attached to any operation described above.

  * When a value handle is consumed, are invalidated:

    - operation handles associated with payload operations that produce as
      result any value associated with the consumed handle (when the associated
      is an operation result);

    - operation handles associated with payload operations _nested_ in the
      payload operations described above;

    - operation handles associated with payload operations (recursively)
      _contained_ in the block that defines as argument any value associated
      with the consumed handle (when the associated value is a block argument);
      note that the adjacent blocks are not affected;

    - value handles associated with any result of any operation described above,
      including all results of the operation defining as result the value
      associated with the consumed handle;

    - value handles associated with any argument of a block contained in a
      region attached to any operation described above.

More intuitively, consuming a handle invalidates any handle that may be pointing
to an object defined or contained in the payload IR subtree rooted at the
closest operation or block.

The Transform dialect infrastructure has the capability of checking whether
the transform IR op operand is invalidated before applying the
transformation. However, such a check is computationally expensive and
must be enabled explicitly through `TransformOptions`. Additionally, the
`transform-dialect-check-uses` pass emits warnings when a handle may be used
after it has been consumed, but does so abstractly, without processing the
payload IR.

Values associated with parameters (non-handles) cannot be invalidated.

## Intended Use and Integrations

The transformation control infrastructure provided by this dialect is
positioned roughly between rewrite patterns and passes. A transformation
that is executed by a transform operation is likely to be sufficiently
complex to require at least a set of patterns to be implemented. It is also
expected to be more focused than a pass: a pass typically applies identical
transformations everywhere in the IR, a transform dialect-controlled
transformation would apply to a small subset of operations selected, e.g.,
by a pattern-matching operation or generated by a previous transformation.
It is discouraged, although technically possible, to run a pass pipeline as
part of the transform op implementation.

One of the main scenarios for using this dialect is fine-grain chaining of
transformations. For example, a loop-like operation may see its iteration
domain split into two parts, implemented as separate loops (transformation
known as index-set splitting), each of which is then transformed differently
(e.g., the first loop is tiled and the second unrolled) with the necessary
enabling and cleanup patterns around the main transformation:

```mlir
// <generate %loop, e.g., by pattern-matching>
// ...
%parts:2 = transform.loop.split %loop { upper_bound_divisible_by = 8 }
transform.loop.tile %parts#0 { tile_sizes = [8] }
transform.loop.unroll %parts#1 { full }
```

This composition would have been difficult to implement as separate passes
since the hypothetical "tiling" and "unrolling" pass would need to somehow
differentiate between the parts of the loop produced by the previous pass
(both are the same operation, and it is likely undesirable to pollute the
operation with pass-specific information). Implementing passes that run the
combined transformation would have run into the combinatorial explosion
issue due to multiple possible transform compositions or into the need for
deep pass parameterization, the ultimate form of which is an ad-hoc dialect
to specify which transformations the pass should run. The transform dialect
provides a uniform, extensible mechanism for controlling transformations in
such cases.

The Transform dialect is supposed to be consumed by an "interpreter" pass
that drives the application of transformations. To ensure extensibility and
composability, this pass is not expected to actually perform the
transformations specified by the ops. Instead, the transformations are
implemented by the transform ops themselves via `TransformOpInterface`. The
pass serves as the entry point, handles the flow of transform operations and
takes care of bookkeeping. As such, the Transform dialect does not provide
the interpreter pass. Instead, it provides a set of utilities that can be
used by clients to define their own interpreter passes or as part of a more
complex pass. For example, the mapping between values in the transform IR
and operations in the payload IR, or the function that applies the
transformations specified by ops in the given block sequentially. Note that
a transform op may have regions with further transform ops in them, with
the op itself guiding how to dispatch the transformation control flow to
those regions. This approach allows clients to decide on the relative
location of the transform IR in their input (e.g., nested modules, separate
modules, optional regions to certain operations, etc.), register additional
transform operations and perform client-specific bookkeeping.

## Effects on the Infrastructure

Although scoped to a single dialect, this functionality conceptually belongs
to the MLIR infrastructure. It aims to be minimally intrusive and opt-in.

Some infrastructural components may grow extra functionality to support the
transform dialect. In particular, the pattern infrastructure may add extra
hooks to identify the "main results" of a transformation or to notify
external observers about changes made to certain operations. These are not
expected to affect the existing uses of the infrastructure.

For the sake of reusability, transformations should be implemented as
utility functions that are called from the interface methods of transform
ops rather than having the methods directly act on the payload IR.

## Type Definitions

[include "Dialects/TransformTypes.md"]

## Core Operations

[include "Dialects/TransformOps.md"]

## Affine Transform Operations

[include "Dialects/AffineLoopTransformOps.md"]

## Bufferization Transform Operations

[include "Dialects/BufferizationTransformOps.md"]

## Debug Transform Operations

[include "Dialects/DebugExtensionOps.md"]

## IRDL (extension) Transform Operations

[include "Dialects/IRDLExtensionOps.md"]

## Func Transform Operations

[include "Dialects/FuncTransformOps.md"]

## GPU Transform Operations

[include "Dialects/GPUTransformOps.md"]

## Loop (extension) Transform Operations

[include "Dialects/LoopExtensionOps.md"]

## Loop (SCF) Transform Operations

[include "Dialects/SCFLoopTransformOps.md"]

## MemRef Transform Operations

[include "Dialects/MemRefTransformOps.md"]

## PDL (extension) Transform Operations

[include "Dialects/PDLExtensionOps.md"]

## Structured (Linalg) Match Operations

[include "Dialects/LinalgStructuredMatchOps.md"]

## Structured (Linalg) Transform Operations

[include "Dialects/LinalgStructuredTransformOps.md"]

## Tensor Transform Operations

[include "Dialects/TensorTransformOps.md"]

## Vector Transform Operations

[include "Dialects/VectorTransformOps.md"]

[include "Dialects/TransformTypeInterfaces.md"]

[include "Dialects/TransformOpInterfaces.md"]
