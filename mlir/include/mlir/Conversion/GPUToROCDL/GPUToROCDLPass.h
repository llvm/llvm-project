//===- GPUToROCDLPass.h - Convert GPU kernel to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
#define MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_

#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class OpBuilder;
class Location;
class RewritePatternSet;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOROCDLOPS
#include "mlir/Conversion/Passes.h.inc"

namespace amd {
/// Constant representing 32 workitems in a workgroup.
const unsigned kWaveFrontSize32 = 32;

/// Constant representing 64 workitems in a workgroup.
const unsigned kWaveFrontSize64 = 64;

/// Wavefront sizes that are supported by the GPU to ROCDL lowerings.
const unsigned kWMMASupportedWaveFrontSizes[] = {kWaveFrontSize32,
                                                 kWaveFrontSize64};

/// Generate ops to get the laneId of the current lane and return it.
Value getLaneId(ConversionPatternRewriter &rewriter, Location loc,
                unsigned indexBitwidth);

/// Return the LLVM Type corresponding to the MMAMatrixType.
Type convertWMMAToROCDLLLVMType(gpu::MMAMatrixType matrixType);
} // namespace amd

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
/// If `runtime` is Unknown, gpu.printf will not be lowered
/// The resulting pattern set should be run over a gpu.module op
void populateGpuToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          gpu::amd::Runtime runtime);

/// Configure target to convert from the GPU dialect to ROCDL.
void configureGpuToROCDLConversionLegality(ConversionTarget &target);

/// Creates a pass that lowers GPU dialect operations to ROCDL counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createLowerGpuOpsToROCDLOpsPass(
    const std::string &chipset = "gfx900",
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout,
    bool useBarePtrCallConv = false,
    gpu::amd::Runtime runtime = gpu::amd::Runtime::Unknown);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
