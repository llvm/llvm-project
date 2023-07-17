//===- NVGPUTransformOps.td - NVGPU transform ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NVGPU_TRANSFORM_OPS
#define NVGPU_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PipelineSharedMemoryCopiesOp
//===----------------------------------------------------------------------===//

def PipelineSharedMemoryCopiesOp :
  Op<Transform_Dialect, "nvgpu.pipeline_shared_memory_copies",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary =
    "Applies software pipelining to a given loop with shared memory copies";

  let description = [{
    Applies software pipelining to a given scf.for loop. The pipelining
    strategy will look for a load into shared memory and pipeline it to overlap
    it with the rest of the loop.
    
    NOTE: It is user responsibility to ensure that there are no dependency
    between `depth` iterations of the loop by using multi-buffering. It is
    also user responsibility to ensure a sufficient amount of shared memory
    is allocated to cover eventual writes by `depth-1` speculative
    iterations.

    `depth` will indicate how many stages the software pipeline should have.
    `peel_epilogue` allows to force the epilogue to be peeled out instead of
    potentially using predicated operations for the epilogue phase.

    #### Return modes

    Consumes the operand handle and produces a result handle pointing to the
    loop, which may or may not have been pipelined. Produces a definite failure
    if the loop pipeliner mutated the IR before failing to pipeline, in
    particular if `peel_epilogue` is not set and the loop body doesn't support
    predication. If failure propagation mode is set to "propagate", produces a
    silenceable failure when pipelining preconditions, e.g., loop bound being
    static, are not met or when the loop wasn't pipelined because due to the
    lack of loads into shared memory. If the failure propagation mode is set
    to "suppress" (default), succeeds in these case and associates the result
    handle with the original loop.

    TODO: the shared memory part and behavior specific to NVGPU should be
    made orthogonal to pipelining so that `transform.loop.pipeline` becomes
    usable here.
  }];

  let arguments = (ins TransformHandleTypeInterface:$for_op,
                   I64Attr:$depth,
                   UnitAttr:$peel_epilogue,
                   DefaultValuedAttr<FailurePropagationMode,
                      "::mlir::transform::FailurePropagationMode::Suppress">
                     :$failure_propagation_mode);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{ 
    `failures` `(` $failure_propagation_mode `)`
    $for_op
    attr-dict 
    `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ForOp forOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// RewriteMatmulAsMmaSyncOp
//===----------------------------------------------------------------------===//

def RewriteMatmulAsMmaSyncOp :
  Op<Transform_Dialect, "nvgpu.rewrite_matmul_as_mma_sync",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformEachOpTrait, 
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Rewrite a matmul operation on memref to an mma.sync operation on vectors.

    Memory copies with the required access patterns are automatically inserted.
    Operations that do not have a 1-1 mapping to mma.sync operations are left
    unchanged.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results) ";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::LinalgOp linalgOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // NVGPU_TRANSFORM_OPS
