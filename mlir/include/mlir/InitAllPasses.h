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
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
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
  registerAffinePasses();
  registerAsyncPasses();
  registerGPUPasses();
  registerGpuSerializeToCubinPass();
  registerLinalgPasses();
  LLVM::registerLLVMPasses();
  quant::registerQuantPasses();
  registerSCFPasses();
  registerShapePasses();
  spirv::registerSPIRVPasses();
  registerStandardPasses();
  tensor::registerTensorPasses();
  tosa::registerTosaOptPasses();
}

} // namespace mlir

#endif // MLIR_INITALLPASSES_H_
