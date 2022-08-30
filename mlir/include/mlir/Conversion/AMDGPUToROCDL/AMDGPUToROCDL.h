//===- AMDGPUToROCDL.h - Convert AMDGPU to ROCDL dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_
#define MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_

#include "mlir/Conversion/AMDGPUToROCDL/Chipset.h"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTAMDGPUTOROCDLPASS
#include "mlir/Conversion/Passes.h.inc"

void populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns,
                                             amdgpu::Chipset chipset);

} // namespace mlir

#endif // MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_
