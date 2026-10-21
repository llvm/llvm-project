//===-- Passes.td - AGX transformation passes --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AGX_TRANSFORMS_PASSES_TD_
#define MLIR_DIALECT_AGX_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AGXExpandOpsPass : Pass<"agx-expand-ops"> {
  let summary = "Expand agx operations the target chip does not run natively";
  let description = [{
    Rewrites the agx operations that have no native lowering on the target
    chip into their portable forms (standard dialects and the basic agx
    operations): matrix fragment loads and stores into per-thread memref
    accesses, `agx.mma_tile` into fragment loads and `agx.mma`, mixed
    precision or negated `agx.mma` into conversions and a same-type
    `agx.mma`, `agx.fma_lane_shift` into a shuffle and an FMA,
    `agx.block_barrier` into `agx.threadgroup_barrier`, and
    `agx.uniform_region` into its inlined body.

    The chip comes from the `chip` option or, when it is empty, from the
    `#agxir.target` of the enclosing `gpu.module`; capabilities are read from
    `AGXChipInfo` (mlir/docs/Dialects/AGX.md).
  }];
  let dependentDialects = [
    "agx::AGXDialect", "arith::ArithDialect", "gpu::GPUDialect",
    "math::MathDialect", "memref::MemRefDialect", "vector::VectorDialect"
  ];
  let options = [
    Option<"chip", "chip", "std::string", /*default=*/"\"\"",
           "Target chip (an LLVM AGX -mcpu), overriding the gpu.module's">,
    Option<"features", "features", "std::string", /*default=*/"\"\"",
           "Target features (\"+agx-strict\"), overriding the gpu.module's">
  ];
}

#endif // MLIR_DIALECT_AGX_TRANSFORMS_PASSES_TD_
