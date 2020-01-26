//===-- GPUToSPIRV.td - GPU to SPIR-V Dialect Lowerings ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to lower GPU dialect ops to to SPIR-V ops.
//
//===----------------------------------------------------------------------===//


#ifndef CONVERT_GPU_TO_SPIRV
#define CONVERT_GPU_TO_SPIRV

include "mlir/Dialect/GPU/GPUOps.td"
include "mlir/Dialect/SPIRV/SPIRVStructureOps.td"

def : Pat<(GPU_ModuleEndOp), (SPV_ModuleEndOp)>;

#endif // CONVERT_GPU_TO_SPIRV
