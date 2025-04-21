//===- AMDGPUToROCDL.h - Convert AMDGPU to ROCDL dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_
#define MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include <memory>
#include <string>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class TypeConverter;
class Pass;

#define GEN_PASS_DECL_CONVERTAMDGPUTOROCDLPASS
#include "mlir/Conversion/Passes.h.inc"

/// Note: This function will also add conversions for the AMDGPU-specific
/// address spaces, but those can be added separately using
/// populateAMDGPUMemorySpaceAttributeConversions().
void populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns,
                                             amdgpu::Chipset chipset);

/// Remap AMDGPU memory spaces to LLVM address spaces
/// by mapping amdgpu::AddressSpace::fat_raw_buffer to ptr addrspace(7),
/// amdgpu::AddressSpace::buffer_rsrc to ptr addrspace(8), and
/// amdgpu::AddressSpace::fat_strided_buffer to ptr addrspace(9).
void populateAMDGPUMemorySpaceAttributeConversions(
    TypeConverter &typeConverter);

} // namespace mlir

#endif // MLIR_CONVERSION_AMDGPUTOROCDL_AMDGPUTOROCDL_H_
