//===-- SPIRVOps.td - MLIR SPIR-V Op Definitions Spec ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main operation definition specification file for SPIR-V
// operations.
//
//===----------------------------------------------------------------------===//

// Note that for each op in this file and the included files for specific op
// categories, we use a tool to automatically generate certain sections in its
// definition: basic structure, summary, description. So modifications to these
// sections will not be respected. Modifications to op traits, arguments,
// results, and sections after the results are retained. Besides, ops must be
// separated via the '// -----' marker.

#ifndef MLIR_DIALECT_SPIRV_IR_OPS
#define MLIR_DIALECT_SPIRV_IR_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVArithmeticOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVAtomicOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVBarrierOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVBitOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVCastOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVCompositeOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVControlFlowOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVCooperativeMatrixOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVIntelExtOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVGLOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVGroupOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVImageOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVIntegerDotProductOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVLogicalOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVMatrixOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVMemoryOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVMiscOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVNonUniformOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVCLOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVStructureOps.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

#endif // MLIR_DIALECT_SPIRV_IR_OPS
