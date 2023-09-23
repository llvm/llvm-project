//===- GPUToAMDGPUPass.h - Convert GPU kernel to AMDGPU dialect -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOAMDGPU_GPUTOAMDGPUPASS_H_
#define MLIR_CONVERSION_GPUTOAMDGPU_GPUTOAMDGPUPASS_H_

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {
class ConversionTarget;
class OpBuilder;
class Location;
class RewritePatternSet;
class Type;
class TypeConverter;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
class MMAMatrixType;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOAMDGPUOPS
#include "mlir/Conversion/Passes.h.inc"

namespace amd {
/// Return the LLVM Type corresponding to the MMAMatrixType.
Type convertWMMAToVectorType(gpu::MMAMatrixType matrixType);

/// String to represent the `opSelect` attribute name.
constexpr char kAMDGpuOpselectAttrName[] = "opSelect";
} // namespace amd

/// Collect a set of patterns to convert from the GPU dialect to AMDGPU.
/// If `runtime` is Unknown, gpu.printf will not be lowered. The resulting
/// pattern set should be run over a gpu.module op. `chipset` is the chip we are
/// targeting. `warpSize` is the warp size to use when generating WMMA
/// intrinsics. `opSelect` is used in the lowering of f16 versions of WMMA ops
/// involving `C` operand. If `opSelect` is true upper half of the general
/// purpose 32-bit registers is used for storing the values; If false the lower
/// half is used.
void populateGpuToAMDGPUConversionPatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           llvm::StringRef chipset = "gfx1100",
                                           unsigned warpSize = 32);

/// Creates a pass that lowers GPU dialect operations to AMDGPU counterparts.
/// The index bitwidth used for the lowering of the device side index
/// computations is configurable. AMD gpus have a configurable warp size; valid
/// choices are 32 and 64. We choose 32 as the default size. `opSelect` is used
/// in the lowering of f16 versions of WMMA ops involving `C` operand. If
/// `opSelect` is true upper half of the general purpose 32-bit registers is
/// used for storing the values; If false the lower half is used.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createLowerGpuOpsToAMDGPUOpsPass(const std::string &chipset = "gfx1100",
                                 unsigned warpSize = 32);

/// Collect a set of patterns to convert WMMA ops from GPU dialect to AMDGPU.
/// `chipset` is the target chip for which the IR is being generated.
/// `warpSize` is the warp size to use when generating WMMA intrinsics.
void populateGpuWMMAToAMDGPUConversionPatterns(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               llvm::StringRef chipset,
                                               unsigned warpSize);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOAMDGPU_GPUTOAMDGPUPASS_H_
