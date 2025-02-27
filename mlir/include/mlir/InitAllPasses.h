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

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MLProgram/Transforms/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Mesh/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
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
  // General passes
  registerTransformsPasses();

  // Conversion passes
  registerConversionPasses();

  // Dialect passes
  acc::registerOpenACCPasses();
  affine::registerAffinePasses();
  amdgpu::registerAMDGPUPasses();
  registerAsyncPasses();
  arith::registerArithPasses();
  bufferization::registerBufferizationPasses();
  func::registerFuncPasses();
  registerGPUPasses();
  registerLinalgPasses();
  registerNVGPUPasses();
  registerSparseTensorPasses();
  LLVM::registerLLVMPasses();
  math::registerMathPasses();
  memref::registerMemRefPasses();
  mesh::registerMeshPasses();
  ml_program::registerMLProgramPasses();
  quant::registerQuantPasses();
  registerSCFPasses();
  registerShapePasses();
  spirv::registerSPIRVPasses();
  tensor::registerTensorPasses();
  tosa::registerTosaOptPasses();
  transform::registerTransformPasses();
  vector::registerVectorPasses();
  arm_sme::registerArmSMEPasses();
  arm_sve::registerArmSVEPasses();
  emitc::registerEmitCPasses();
  xegpu::registerXeGPUPasses();

  // Dialect pipelines
  bufferization::registerBufferizationPipelines();
  sparse_tensor::registerSparseTensorPipelines();
  tosa::registerTosaToLinalgPipelines();
  gpu::registerGPUToNVVMPipeline();
}

} // namespace mlir

#endif // MLIR_INITALLPASSES_H_
