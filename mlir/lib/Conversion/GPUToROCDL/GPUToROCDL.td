//==-- GPUToROCDL.td - GPU Ops to ROCDL Patterns -------------*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower GPU ops to ROCDL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOROCDL_TD
#define MLIR_CONVERSION_GPUTOROCDL_TD

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/GPU/GPUOps.td"
include "mlir/Dialect/LLVMIR/ROCDLOps.td"

def : Pat<(GPU_BarrierOp), (ROCDL_BarrierOp)>;

#endif // MLIR_CONVERSION_GPUTOROCDL_TD
