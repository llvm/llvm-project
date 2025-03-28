//===- GPUToAMDGPU.h - Convert AMDGPU to ROCDL dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOAMDGPU_GPUTOAMDGPU_H_
#define MLIR_CONVERSION_GPUTOAMDGPU_GPUTOAMDGPU_H_


#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <string>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class TypeConverter;
class Pass;

#define GEN_PASS_DECL_CONVERTGPUTOAMDGPUPASS
#include "mlir/Conversion/Passes.h.inc"

void populateAMDGPUOptimizedSubgroupReducePatterns(RewritePatternSet &patterns,
                                            unsigned subgroupSize,
                                            PatternBenefit benefit);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOAMDGPU_GPUTOAMDGPU_H_