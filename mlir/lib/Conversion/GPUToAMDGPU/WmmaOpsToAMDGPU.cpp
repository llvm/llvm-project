//===-------- WmmaOpsToAMDGPU.cpp - GPU WMMA ops to AMD GPU lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of patterns to lower GPU Subgroup MMA ops to
// AMD GPU Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAMDGPU/GPUToAMDGPUPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

namespace {

static LogicalResult areAllVectorTypes(Operation *op, ValueRange operands,
                                       ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return isa<mlir::VectorType>(value.getType());
      })) {
    return rewriter.notifyMatchFailure(
        op, "cannot convert if operands aren't of Vector type.");
  }

  return success();
}

/// Create a WMMA compute intrinsic doing the multiply-add operation as :
///
///  `cOp` = `aOp` * `bOp` + `cOp`
///
/// and return the generated op in `computeOp`.
static LogicalResult createWMMAComputeIntrinsic(Value aOp, Value bOp, Value cOp,
                                                Location loc, bool opSelect,
                                                PatternRewriter &rewriter,
                                                Value &computeOp) {
  Type aType = aOp.getType();
  Type bType = bOp.getType();
  Type cType = cOp.getType();

  // All the intrinsics present currently operate on vector types.
  auto checkVecType = [](Value value, StringRef op) {
    if (!isa<VectorType>(value.getType())) {
      return mlir::emitError(value.getLoc(), op + "should be of vector type");
    }
    return InFlightDiagnostic();
  };

  if (failed(checkVecType(aOp, "aOp")))
    return failure();
  if (failed(checkVecType(bOp, "bOp")))
    return failure();
  if (failed(checkVecType(cOp, "cOp")))
    return failure();

  auto aVecType = aType.cast<VectorType>();
  auto bVecType = bType.cast<VectorType>();
  auto cVecType = cType.cast<VectorType>();

  if (aVecType != bVecType)
    return emitError(aOp.getLoc(), "aOp and bOp must be of same type");

  Type aEltType = aVecType.getElementType();
  Type cEltType = cVecType.getElementType();

  // We support lowering for the mixed-precision and full fp16 WMMA intrinsics
  // currently.
  if (aEltType.isF16() && cEltType.isF32()) {
    // subwordOffset is always false for F32 `C` operands as they occupy all 32
    // bits in the VGPR.
    computeOp = rewriter.create<amdgpu::WMMAOp>(loc, cType, aOp, bOp, cOp,
                                                /*subwordOffset=*/false);
    return success();
  }
  if (aEltType.isF16() && cEltType.isF16()) {
    computeOp =
        rewriter.create<amdgpu::WMMAOp>(loc, cType, aOp, bOp, cOp, opSelect);
    return success();
  }

  return failure();
}

/// This class implements the conversion of GPU MMA computeOp to wmma.mma op
/// in the ROCDL dialect.
struct WmmaMmaOpToAMDGPULowering
    : public OpConversionPattern<gpu::SubgroupMmaComputeOp> {
  WmmaMmaOpToAMDGPULowering(TypeConverter &typeConverter, MLIRContext *context,
                            StringRef chip, unsigned warpSize)
      : OpConversionPattern<gpu::SubgroupMmaComputeOp>::OpConversionPattern(
            typeConverter, context),
        warpSize(warpSize), chip(chip){};

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaComputeOp subgroupMmaComputeOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllVectorTypes(subgroupMmaComputeOp.getOperation(),
                                 adaptor.getOperands(), rewriter)))
      return failure();

    std::size_t firstPos = chip.find("gfx11");
    std::size_t lastPos = chip.rfind("gfx11");
    if (firstPos != 0 || (firstPos != lastPos))
      return subgroupMmaComputeOp->emitError(
          "wmma lowering is supported for gfx11 series only");

    if (warpSize != amd::kWaveFrontSize32)
      return subgroupMmaComputeOp->emitError(
          "wavefront of size 32 only supported");

    auto aTranspose = subgroupMmaComputeOp.getATranspose();
    auto bTranspose = subgroupMmaComputeOp.getBTranspose();

    if ((aTranspose.has_value() && aTranspose.value()) ||
        (bTranspose.has_value() && bTranspose.value()))
      return subgroupMmaComputeOp->emitError(
          "lowering with transpose is not supported. Please "
          "use transpose while loading/storing the operands.");

    Location loc = subgroupMmaComputeOp->getLoc();

    gpu::MMAMatrixType aType =
        subgroupMmaComputeOp.getOpA().getType().cast<gpu::MMAMatrixType>();
    gpu::MMAMatrixType bType =
        subgroupMmaComputeOp.getOpA().getType().cast<gpu::MMAMatrixType>();
    gpu::MMAMatrixType cType =
        subgroupMmaComputeOp.getOpC().getType().cast<gpu::MMAMatrixType>();

    SmallVector<gpu::MMAMatrixType> allTypes = {aType, bType, cType};

    SmallVector<int64_t> aTypeShape(aType.getShape());
    SmallVector<int64_t> bTypeShape(bType.getShape());
    SmallVector<int64_t> cTypeShape(cType.getShape());
    SmallVector<SmallVector<int64_t>> allShapes = {aTypeShape, bTypeShape,
                                                   cTypeShape};

    if (!llvm::all_of(allShapes, [](ArrayRef<int64_t> shape) {
          return llvm::all_of(shape, [](int dim) { return dim == 16; });
        }))
      return subgroupMmaComputeOp->emitError(
          "wmma ops of shape 16x16x16 are only supported.");

    // Get the WMMA intrinsic to map to.
    bool opSelect = subgroupMmaComputeOp->hasAttrOfType<UnitAttr>(
        amd::kAMDGpuOpselectAttrName);
    Value computeOp;
    if (failed(createWMMAComputeIntrinsic(adaptor.getOpA(), adaptor.getOpB(),
                                          adaptor.getOpC(), loc, opSelect,
                                          rewriter, computeOp)))
      return rewriter.notifyMatchFailure(subgroupMmaComputeOp,
                                         "unsupported mma op variant.");

    rewriter.replaceOp(subgroupMmaComputeOp, computeOp);
    return success();
  }

  /// `warpSize` is the warp size to use when generating WMMA intrinsics.
  unsigned warpSize;

  /// The target chip for which to generate the lowering.
  std::string chip;
};

} // namespace

void mlir::populateGpuWMMAToAMDGPUConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns, StringRef chip,
    unsigned warpSize) {
  patterns.add<WmmaMmaOpToAMDGPULowering>(converter, patterns.getContext(),
                                          chip, warpSize);
}
