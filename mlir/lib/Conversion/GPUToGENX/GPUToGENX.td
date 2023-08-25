//==-- GPUToGENX.td - GPU Ops to GENX Patterns ---------------*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower GPU ops to GENX.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOGENX_TD
#define MLIR_CONVERSION_GPUTOGENX_TD

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/GPU/IR/GPUOps.td"
include "mlir/Dialect/LLVMIR/GENXOps.td"

def : Pat<(GPU_BarrierOp), (GENX_BarrierOp)>;

#endif // MLIR_CONVERSION_GPUTOGENX_TD
