//===- NVGPUToNVVMPass.h - Convert NVGPU to NVVM dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_NVGPUTONVVM_NVGPUTONVVMPASS_H_
#define MLIR_CONVERSION_NVGPUTONVVM_NVGPUTONVVMPASS_H_

#include <memory>

namespace mlir {

class Attribute;
class LLVMTypeConverter;
class MemRefType;
class MLIRContext;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTNVGPUTONVVMPASS
#include "mlir/Conversion/Passes.h.inc"

namespace nvgpu {
class MBarrierGroupType;

/// Returns the memory space attribute of the mbarrier object.
Attribute getMbarrierMemorySpace(MLIRContext *context,
                                 MBarrierGroupType barrierType);

/// Return the memref type that can be used to represent an mbarrier object.
MemRefType getMBarrierMemrefType(MLIRContext *context,
                                 MBarrierGroupType barrierType);
} // namespace nvgpu

void populateNVGPUToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_NVGPUTONVVM_NVGPUTONVVMPASS_H_
