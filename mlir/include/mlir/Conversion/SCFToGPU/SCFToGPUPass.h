//===- SCFToGPUPass.h - Pass converting loops to GPU kernels ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
#define MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
template <typename T>
class InterfacePass;
class Pass;

#define GEN_PASS_DECL_CONVERTAFFINEFORTOGPUPASS
#define GEN_PASS_DECL_CONVERTPARALLELLOOPTOGPUPASS
#include "mlir/Conversion/Passes.h.inc"

/// Create a pass that converts loop nests into GPU kernels.  It considers
/// top-level affine.for operations as roots of loop nests and converts them to
/// the gpu.launch operations if possible.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAffineForToGPUPass(unsigned numBlockDims, unsigned numThreadDims);
std::unique_ptr<InterfacePass<FunctionOpInterface>> createAffineForToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
