//===- MemRefTransformOps.td - MemRef transformation ops --*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_TRANSFORM_OPS
#define MEMREF_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyExpandOpsPatternsOp : Op<Transform_Dialect,
    "apply_patterns.memref.expand_ops",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collects patterns to rewrite ops within the memref dialect.

    - Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
      AtomicRMWOpLowering pattern, e.g. with "minf" or "maxf" attributes, to
      `memref.generic_atomic_rmw` with the expanded code.
    - Converts `memref.reshape` that has a target shape of a statically-known
      size to `memref.reinterpret_cast`.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyExpandStridedMetadataPatternsOp : Op<Transform_Dialect,
    "apply_patterns.memref.expand_strided_metadata",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collects patterns for expanding memref operations that modify the metadata
    (sizes, offset, strides) of a memref into easier to analyze constructs.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyExtractAddressComputationsPatternsOp : Op<Transform_Dialect,
    "apply_patterns.memref.extract_address_computations",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collects patterns for extracting address computations from operations
    with memory accesses such that these memory accesses use only a base
    pointer.

    For instance,
    ```mlir
    memref.load %base[%off0, ...]
    ```

    Will be rewritten in:
    ```mlir
    %new_base = memref.subview %base[%off0,...][1,...][1,...]
    memref.load %new_base[%c0,...]
    ```
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyFoldMemrefAliasOpsPatternsOp : Op<Transform_Dialect,
    "apply_patterns.memref.fold_memref_alias_ops",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collects patterns for folding memref aliasing ops (memref.subview) into
    consumer load/store ops (affine.load, memref.load, nvgpu.ldmatrix,
    vector.load, vector.transfer_read, affine.store, memref.store, etc.) and
    other ops (e.g., memref.subview).
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyResolveRankedShapedTypeResultDimsPatternsOp : Op<Transform_Dialect,
    "apply_patterns.memref.resolve_ranked_shaped_type_result_dims",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collects patterns that resolve `memref.dim` operations with values that are
    defined by operations that implement the `ReifyRankedShapedTypeOpInterface`,
    in terms of shapes of its input operands.
  }];

  let assemblyFormat = "attr-dict";
}

def Transform_MemRefAllocOp : Transform_ConcreteOpType<"memref.alloc">;

def MemRefMultiBufferOp : Op<Transform_Dialect, "memref.multibuffer",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Multibuffers an allocation";
  let description = [{
     Transformation to do multi-buffering/array expansion to remove
     dependencies on the temporary allocation between consecutive loop
     iterations. This transform expands the size of an allocation by
     a given multiplicative factor and fixes up any users of the
     multibuffered allocation.
     If skip analysis is not set the transformation will only apply
     if it can prove that there is no data being carried across loop
     iterations.

     #### Return modes

     This operation returns the new allocation if multi-buffering
     succeeds, and failure otherwise.
  }];

  let arguments =
      (ins Transform_MemRefAllocOp:$target,
           ConfinedAttr<I64Attr, [IntPositive]>:$factor,
           UnitAttr:$skip_analysis);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def MemRefMakeLoopIndependentOp
    : Op<Transform_Dialect, "memref.make_loop_independent",
         [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
          TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Rewrite the targeted ops such that their index-typed operands no longer
    depend on any loop induction variable of the `num_loop` enclosing `scf.for`
    loops. I.e., compute an upper bound that is independent of any such loop IV
    for every tensor dimension. The transformed op could then be hoisted from
    the `num_loop` enclosing loops. To preserve the original semantics, place a
    `memref.subview` inside the loop.

    Currently supported operations are:
    - memref.alloca: Replaced with a new memref.alloca with upper bound sizes,
      followed by a memref.subview.

    #### Return modes

    This operation fails if at least one induction variable could not be
    eliminated. In case the targeted op is already independent of induction
    variables, this transform succeeds and returns the unmodified target op.

    Otherwise, the returned handle points to a subset of the produced ops:
    - memref.alloca: The returned handle points to the memref.subview op.

    This transform op consumes the target handle and produces a result handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target, I64Attr:$num_loops);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat =
      "$target attr-dict `:` functional-type($target, $transformed)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // MEMREF_TRANSFORM_OPS
