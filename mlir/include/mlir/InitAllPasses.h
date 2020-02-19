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

#include "mlir/Analysis/Passes.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/FxpMathOps/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LoopOps/Passes.h"
#include "mlir/Dialect/QuantOps/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Quantizer/Transforms/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

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
  // At the moment we still rely on global initializers for registering passes,
  // but we may not do it in the future.
  // We must reference the passes in such a way that compilers will not
  // delete it all as dead code, even with whole program optimization,
  // yet is effectively a NO-OP. As the compiler isn't smart enough
  // to know that getenv() never returns -1, this will do the job.
  if (std::getenv("bar") != (char *)-1)
    return;

  // Init general passes
  createCanonicalizerPass();
  createCSEPass();
  createVectorizePass({});
  createLoopUnrollPass();
  createLoopUnrollAndJamPass();
  createSimplifyAffineStructuresPass();
  createLoopFusionPass();
  createLoopInvariantCodeMotionPass();
  createAffineLoopInvariantCodeMotionPass();
  createPipelineDataTransferPass();
  createLowerAffinePass();
  createLoopTilingPass(0);
  createLoopCoalescingPass();
  createAffineDataCopyGenerationPass(0, 0);
  createMemRefDataFlowOptPass();
  createStripDebugInfoPass();
  createPrintOpStatsPass();
  createInlinerPass();
  createSymbolDCEPass();
  createLocationSnapshotPass({});

  // GPUtoRODCLPass
  createLowerGpuOpsToROCDLOpsPass();

  // FxpOpsDialect passes
  fxpmath::createLowerUniformRealMathPass();
  fxpmath::createLowerUniformCastsPass();

  // GPU
  createGpuKernelOutliningPass();
  createSimpleLoopsToGPUPass(0, 0);
  createLoopToGPUPass({}, {});

  // CUDA
  createConvertGpuLaunchFuncToCudaCallsPass();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  createConvertGPUKernelToCubinPass(
      [](const std::string &, Location, StringRef) { return nullptr; });
#endif
  createLowerGpuOpsToNVVMOpsPass();

  // Linalg
  createLinalgFusionPass();
  createLinalgTilingPass();
  createLinalgTilingToParallelLoopsPass();
  createLinalgPromotionPass(0);
  createConvertLinalgToLoopsPass();
  createConvertLinalgToParallelLoopsPass();
  createConvertLinalgToAffineLoopsPass();
  createConvertLinalgToLLVMPass();

  // LoopOps
  createParallelLoopFusionPass();

  // QuantOps
  quant::createConvertSimulatedQuantPass();
  quant::createConvertConstPass();
  quantizer::createAddDefaultStatsPass();
  quantizer::createRemoveInstrumentationPass();
  quantizer::registerInferQuantizedTypesPass();

  // SPIR-V
  spirv::createDecorateSPIRVCompositeTypeLayoutPass();
  spirv::createLowerABIAttributesPass();
  createConvertGPUToSPIRVPass();
  createConvertStandardToSPIRVPass();
  createLegalizeStdOpsForSPIRVLoweringPass();
  createLinalgToSPIRVPass();

  // Vulkan
  createConvertGpuLaunchFuncToVulkanCallsPass();
}

} // namespace mlir

#endif // MLIR_INITALLPASSES_H_
