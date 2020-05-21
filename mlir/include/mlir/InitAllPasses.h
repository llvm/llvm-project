//===- LinkAllPassesAndDialects.h - MLIR Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITALLPASSES_H_
#define MLIR_INITALLPASSES_H_

#include "mlir/Conversion/AVX512ToLLVM/ConvertAVX512ToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "mlir/Transforms/ViewRegionGraph.h"

#include <cstdlib>

namespace mlir {

// This function may be called to register the MLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerAllPasses() {
  // Init general passes
#define GEN_PASS_REGISTRATION
#include "mlir/Transforms/Passes.h.inc"

  // Conversion passes
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/Passes.h.inc"

  // Affine
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Affine/Passes.h.inc"

  // GPU
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Passes.h.inc"

  // Linalg
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Linalg/Passes.h.inc"

  // LLVM
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

  // Loop
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SCF/Passes.h.inc"

  // Quant
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Quant/Passes.h.inc"

  // SPIR-V
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SPIRV/Passes.h.inc"

  // Standard
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/StandardOps/Transforms/Passes.h.inc"
}

} // namespace mlir

#endif // MLIR_INITALLPASSES_H_
