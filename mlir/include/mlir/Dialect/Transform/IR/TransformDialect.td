//===- TransformDialect.td - Transform dialect definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT

include "mlir/IR/OpBase.td"

def Transform_Dialect : Dialect {
  let summary = "Fine-grain transformation control dialect";
  let description = [{
    ## Disclaimer

    ** Proceed with care: not ready for general use. **

    This dialect is evolving rapidly and may change on a very short notice. To
    decrease the maintenance burden and churn, only a few in-tree use cases are
    currently supported in the main tree:

      - high-level transformations on "structured ops" (i.e. ops that operate on
        chunks of data in a way that can be decomposed into operations on
        smaller chunks of data and control flow) in Linalg, Tensor and Vector
        dialects.

    *Please post a description of the intended use case on the MLIR forum and
    wait for confirmation.*

    ## Overview

    This dialect provides operations that can be used to control transformation
    of the IR using a different portion of the IR. It refers to the IR being
    transformed as payload IR, and to the IR guiding the transformation as
    transform IR.

    The main use case for this dialect is orchestrating fine-grain
    transformations on individual operations or sets thereof. For example, it
    may involve finding loop-like operations with specific properties (e.g.,
    large size) in the payload IR, applying loop tiling to those and only those
    operations, and then applying loop unrolling to the inner loops produced
    by the previous transformations. As such, it is not intended as a
    replacement for the pass infrastructure, nor for the pattern rewriting
    infrastructure. In the most common case, the transform IR will be processed
    and applied to the payload IR by a pass. Transformations expressed by the
    transform dialect may be implemented using the pattern infrastructure or any
    other relevant MLIR component.

    The following IR gives a rough idea of what the operations in this dialect
    may look like:

    ```mlir
    %0 = transform.loop.find { size > 42 }
    %1:2 = transform.loop.tile { tile_sizes = [2,3,4] }
    transform.loop.unroll %1#1
    ```

    The values defined by operations in this dialect correspond to (groups of)
    operations in the payload IR. In the example above, `%0` corresponds to the
    set of loops found in the payload IR that satisfy the condition, and `%1`
    correspond to groups of outer and inner loops, respectively, produced by
    the tiling transformation.

    Overall, Transform IR ops are expected to be contained in a single top-level
    op. Such top-level ops specify how to apply the transformations described
    by the operations they contain, e.g., `transform.sequence` executes
    transformations one by one and fails if any of them fails. Such ops are
    expected to have the `PossibleTopLevelTransformOpTrait` and may be used
    without arguments.

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

    ## Side Effects

    The Transform dialect relies on MLIR side effect modelling to enable
    optimization of the transform IR. More specifically, it provides several
    side effect resource objects and expects operations to describe their
    effects on these resources.

      * `TransformMappingResource` - side effect resource corresponding to the
        mapping between transform IR values and payload IR operations.
        
        - An `Allocate` effect from this resource means creating a new mapping
          entry, it is always accompanied by a `Write` effet.
          
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
    combined transfomration would have run into the combinatorial explosion
    issue due to multiple possible transform compositions or into the need for
    deep pass parameterization, the ultimate form of which is an ad-hoc dialect
    to specify which transformations the pass should run. The transform dialect
    provides a uniform, extensible mechanism for controlling transformations in
    such cases.

    The transform dialect is supposed to be consumed by an "interpreter" pass
    that drives the application of transformations. To ensure extensibility and
    composability, this pass is not expected to actually perform the
    transformations specified by the ops. Instead, the transformations are
    implemented by the transform ops themselves via `TransformOpInterface`. The
    pass serves as the entry point, handles the flow of transform operations and
    takes care of bookkeeping. As such, the transform dialect does not provide
    the interpreter pass. Instead, it provides a set of utilities that can be
    used by clients to define their own interpreter passes or as part of a more
    complex pass. For example, the mapping between values in the tranfsorm IR
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
  }];

  let name = "transform";
  let cppNamespace = "::mlir::transform";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;

  let dependentDialects = [
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
  ];

  let extraClassDeclaration = [{
      /// Returns the named PDL constraint functions available in the dialect
      /// as a map from their name to the function.
      const ::llvm::StringMap<::mlir::PDLConstraintFunction> &
      getPDLConstraintHooks() const;

    private:
      template <typename OpTy>
      void addOperationIfNotRegistered() {
        Optional<RegisteredOperationName> opName =
            RegisteredOperationName::lookup(OpTy::getOperationName(),
                                            getContext());
        if (!opName)
          return addOperations<OpTy>();

        if (opName->getTypeID() == TypeID::get<OpTy>())
          return;

        llvm::errs() << "error: extensible dialect operation '"
                     << OpTy::getOperationName()
                     << "' is already registered with a mismatching TypeID";
        abort();
      }

      /// Registers operations specified as template parameters with this
      /// dialect. Checks that they implement the required interfaces.
      template <typename... OpTys>
      void addOperationsChecked() {
        (void)std::initializer_list<int>{(addOperationIfNotRegistered<OpTys>(),
                                          0)...};

        #ifndef NDEBUG
        (void)std::initializer_list<int>{
          (detail::checkImplementsTransformInterface<OpTys>(getContext()),
           0)...};
        #endif // NDEBUG
      }

      template <typename, typename...>
      friend class TransformDialectExtension;

      /// Takes ownership of the named PDL constraint function from the given
      /// map and makes them available for use by the operations in the dialect.
      void mergeInPDLMatchHooks(
          ::llvm::StringMap<::mlir::PDLConstraintFunction> &&constraintFns);

      /// A container for PDL constraint function that can be used by
      /// operations in this dialect.
      PDLPatternModule pdlMatchHooks;
  }];
}

// Base class for ops that belong to the tranfsorm dialect. Ops defined in
// extensions of this dialect may also use this.
class TransformDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<Transform_Dialect, mnemonic, traits>;

// Trait for operations that may be top-level operations in Transform IR.
// Operations must have one single-block region and must be usable without
// operands. See the C++ definition of the trait for more information.
def PossibleTopLevelTransformOpTrait
    : NativeOpTrait<"PossibleTopLevelTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT
