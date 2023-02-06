//===- GPUTransformOps.td - GPU transform ops --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GPU_TRANSFORM_OPS
#define GPU_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MapNestedForeachToThreads :
  Op<Transform_Dialect, "gpu.map_nested_foreach_to_threads",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
      Target the `gpu.launch op` and rewrite all `scf.foreach_thread`
      nested in it to distributed `gpu.thread_id` attribute.

      The operation searches for `scf.foreach_thread` ops nested under `target`
      and maps each such op to GPU threads. Mapping is one-to-one and the
      induction variables of `scf.foreach_thread` are rewritten to
      `gpu.thread_id` according to the `mapping` attribute.

      Sibling `scf.foreach_thread` are supported in which case, the union of
      the number of threads is computed and may result in predication.

      Multiple scf.foreach_thread are supported per `gpu.launch` in which case,
      the max of all the threads is computed and taken for the global
      `gpu.thread_id`. If necessary, `scf.foreach_thread` that do not use the
      whole thread range result in predicated computations.

      Dynamic `scf.foreach_thread` trip counts are currently not supported.
      Dynamic block dim sizes are currently not supported.

      Only **bufferized** `scf.foreach_thread` are currently supported.
      Only `scf.foreach_thread` distributed to **at most 3 dimensions** are
      currently supported.

      Barriers are inserted after each scf.foreach_thread op for now.

      The operation alters the block size of the given gpu_launch using
      blockDim argument.

      #### Return modes:

      This operation ignores non-gpu_launch ops and drops them in the return.

      If any scf.foreach_thread with tensors is found, the transform definitely
      fails.

      If all the scf.foreach_thread operations with gpu.thread mapping contained
      within the LaunchOp referred to by the `target` PDLOperation lower to GPU
      properly, the transform succeeds. Otherwise the transform definitely
      fails.

      scf.foreach_thread operations with mappings other than gpu.thread are
      ignored.

      The returned handle points to the same LaunchOp operand, consuming it and
      producing a new SSA value to satisfy chaining and linearity of the IR
      properties.

      #### Example:

      ```
      gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
                 threads(%tx, %ty, %tz) in (%tx = %3, %ty = %4, %tz = %5) {
        scf.foreach_thread (%i, %j) in (7, 9) {
          ... // body 1
        } {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]}
        scf.foreach_thread (%i) in (12) {
          ... // body 2
        } {mapping = [#gpu.thread<x>]}
        gpu.terminator
      }
      ```
      is translated to:

      ```
      %bdimX = arith.constant 12 : index
      %bdimY = arith.constant 9 : index
      gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
             threads(%tx, %ty, %tz) in (%tx = %bdimX, %ty = %bdimY, %tz = %5) {
        if (threadIdx.x < 9 && threadIdx.y < 7) {
          ... // body 1
        }
        gpu.barrier
        if (threadIdx.y < 1) {
          ... // body 2
        }
        gpu.barrier
        gpu.terminator
      }
      ```
    }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$blockDim,
                   DefaultValuedAttr<BoolAttr, "true">:$syncAfterDistribute);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}


def MapForeachToBlocks :
  Op<Transform_Dialect, "gpu.map_foreach_to_blocks",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Target the gpu_launch op and rewrite the top level `scf.foreach_thread`
    to distributed gpu.block_id attribute. If `generate_gpu_launch` attribute
    is set, then first generates `gpu_launch` and moves the top level
    `scf.foreach_thread` inside.

    The operation searches top level `scf.foreach_thread` ops under
    `gpu_launch` and maps each such op to GPU blocks. Mapping is
    one-to-one and the induction variables of `scf.foreach_thread` are
    rewritten to gpu.block_id according to the `thread_dim_mapping` attribute.

    Dynamic, `scf.foreach_thread` trip counts are currently not supported.
    Dynamic block dim sizes are currently not supported.

    Only **bufferized** scf.foreach_thread are currently supported.
    Only scf.foreach_thread distributed to **at most 3 dimensions** are
    currently supported.

    The operation alters the block size of the given gpu_launch using
    gridDim argument.

    #### Return modes:

    This operation ignores non-gpu_launch ops and drops them in the return.

    If any scf.foreach_thread with tensors is found, the transform definitely
    fails.

    If all the scf.foreach_thread operations contained within the LaunchOp
    referred to by the `target` PDLOperation lower to GPU properly, the
    transform succeeds. Otherwise the transform definitely fails.

    The returned handle points to the same LaunchOp operand, consuming it and
    producing a new SSA value to satisfy chaining and linearity of the IR
    properties.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$gridDim,
                   UnitAttr:$generate_gpu_launch);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // GPU_TRANSFORM_OPS
