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
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MemrefToLLVMTypeConverterOp : Op<Transform_Dialect,
    "apply_conversion_patterns.memref.memref_to_llvm_type_converter",
    [DeclareOpInterfaceMethods<TypeConverterBuilderOpInterface,
                               ["getTypeConverter",
                                "getTypeConverterType"]>]> {
  let description = [{
    This operation provides an "LLVMTypeConverter" that lowers memref types to
    LLVM types.

    The type converter can be customized as follows:
    - `use_aligned_alloc`: Use aligned_alloc in place of malloc for heap
      allocations.
    - `index_bitwidth`: Bitwidth of the index type, "0" indicates the size of a
      machine word.
    - `use_generic_functions`: Use generic allocation and deallocation functions
      instead of the classic "malloc", "aligned_alloc" and "free" functions.
    // TODO: the following two options don't really make sense for 
    // memref_to_llvm_type_converter specifically.
    // We should have a single to_llvm_type_converter.
    - `use_bare_ptr_call_conv`: Replace FuncOp's MemRef arguments with bare 
      pointers to the MemRef element types.
    - `data-layout`: String description (LLVM format) of the data layout that is
      expected on the produced module.
  }];

  let arguments = (ins
      DefaultValuedOptionalAttr<BoolAttr, "false">:$use_aligned_alloc,
      DefaultValuedOptionalAttr<I64Attr, "64">:$index_bitwidth,
      DefaultValuedOptionalAttr<BoolAttr, "false">:$use_generic_functions,
      DefaultValuedOptionalAttr<BoolAttr, "false">:$use_bare_ptr_call_conv,
      OptionalAttr<StrAttr>:$data_layout);
  let assemblyFormat = "attr-dict";
}

def ApplyAllocToAllocaOp : Op<Transform_Dialect,
    "apply_patterns.memref.alloc_to_alloca",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface, ["populatePatternsWithState"]>]> {
  let description = [{
    Collects patterns to rewrite scoped dynamic allocation (`alloc`/`dealloc`
    pairs) into automatic allocation (`alloca`) in the same scope, for memrefs
    of static shape.

    The `size_limit` attribute controls the maximum allocated memory (in bytes,
    subject to data layout) for which the pattern applies.
  }];

  let arguments = (ins
    OptionalAttr<I64Attr>:$size_limit);
  let assemblyFormat = "(`size_limit` `(` $size_limit^ `)`)? attr-dict";
}

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
def Transform_MemRefAllocaOp : Transform_ConcreteOpType<"memref.alloca">;

def MemRefAllocaToGlobalOp :
  Op<Transform_Dialect, "memref.alloca_to_global",
     [TransformOpInterface,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Inserts a new `memref.global` for each provided `memref.alloca` into the
    nearest symbol table (e.g., a `builtin.module`) and replaces it with a
    `memref.get_global`. This is useful, for example, for allocations that
    should reside in the shared memory of a GPU, which have to be declared as
    globals.

    #### Example

    Consider the following transform op:

    ```mlir
    %get_global, %global =
        transform.memref.alloca_to_global %alloca
          : (!transform.op<"memref.alloca">)
            -> (!transform.any_op, !transform.any_op)
    ```

    and the following input payload:

    ```mlir
    module {
      func.func @func() {
        %alloca = memref.alloca() : memref<2x32xf32>
        // usages of %alloca...
      }
    }
    ```

    then applying the transform op to the payload would result in the following
    output IR:

    ```mlir
    module {
      memref.global "private" @alloc : memref<2x32xf32>
      func.func @func() {
        %alloca = memref.get_global @alloc : memref<2x32xf32>
        // usages of %alloca...
      }
    }
    ```

    #### Return modes

    Succeeds always. The returned handles refer to the `memref.get_global` and
    `memref.global` ops that were inserted by the transformation.
  }];

  let arguments = (ins Transform_MemRefAllocaOp:$alloca);
  let results = (outs TransformHandleTypeInterface:$getGlobal,
                  TransformHandleTypeInterface:$global);

  let assemblyFormat = [{
    $alloca attr-dict `:` functional-type(operands, results)
  }];
}

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

def MemRefEraseDeadAllocAndStoresOp
    : Op<Transform_Dialect, "memref.erase_dead_alloc_and_stores", [
      TransformEachOpTrait, TransformOpInterface,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      ReportTrackingListenerFailuresOpTrait
    ]> {
  let description = [{
    This applies memory optimization on memref. In particular it does store to
    load forwarding, dead store elimination and dead alloc elimination.

    #### Return modes

    This operation applies a set of memory optimization on the whole region of
    the operand.

    The transformation does not consume the target handle. It modifies the
    payload. Dead allocations, loads and stores are silently dropped from all
    mappings.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type($target, results)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
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
