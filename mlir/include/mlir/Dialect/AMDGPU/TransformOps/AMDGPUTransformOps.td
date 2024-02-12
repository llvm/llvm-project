//===- AMDGPUTransformOps.td - AMDGPU transform ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_TRANSFORM_OPS
#define AMDGPU_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ApplyOptimizeSharedMemoryReadsAndWritesOp
//===----------------------------------------------------------------------===//

def ApplyOptimizeSharedMemoryReadsAndWritesOp :
  Op<Transform_Dialect, "amdgpu.optimize_shared_memory_reads_and_writes",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Reduce shared memory bank conflicts";
  let description = [{ This opp adds a transformation and pass to the AMDGPU 
    dialect that attempts to optimize reads/writes from a memref representing 
    GPU shared memory in order to avoid bank conflicts.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::FuncOp funcOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // AMDGPU_TRANSFORM_OPS
