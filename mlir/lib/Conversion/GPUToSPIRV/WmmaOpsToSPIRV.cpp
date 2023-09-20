//===------ WmmaOpsToSPIRV.cpp - WMMA LD/ST/Compute to SPIRV lowering -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of patterns to lower GPU Subgroup MMA ops to
// SPIRV Cooperative Matrix ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <cassert>

namespace mlir {
//===----------------------------------------------------------------------===//
// Patterns and helpers used by both the KHR and the NV lowering paths.
//===----------------------------------------------------------------------===//

/// Creates a SPIR-V op to replace the given GPU subgroup mma elementwise op
/// when the elementwise op directly supports with cooperative matrix type.
/// Returns false if cannot.
///
/// See SPV_NV_cooperative_matrix for supported elementwise ops.
static bool createElementwiseOp(ConversionPatternRewriter &builder,
                                gpu::SubgroupMmaElementwiseOp op, Type coopType,
                                ValueRange operands) {
  assert((isa<spirv::CooperativeMatrixType, spirv::CooperativeMatrixNVType>(
      coopType)));

  switch (op.getOpType()) {
  case gpu::MMAElementwiseOp::ADDF:
    builder.replaceOpWithNewOp<spirv::FAddOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::ADDI:
    builder.replaceOpWithNewOp<spirv::IAddOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::SUBF:
    builder.replaceOpWithNewOp<spirv::FSubOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::SUBI:
    builder.replaceOpWithNewOp<spirv::ISubOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::DIVF:
    builder.replaceOpWithNewOp<spirv::FDivOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::DIVS:
    builder.replaceOpWithNewOp<spirv::SDivOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::DIVU:
    builder.replaceOpWithNewOp<spirv::UDivOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::NEGATEF:
    builder.replaceOpWithNewOp<spirv::FNegateOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::NEGATES:
    builder.replaceOpWithNewOp<spirv::SNegateOp>(op, coopType, operands);
    return true;
  case gpu::MMAElementwiseOp::EXTF:
    builder.replaceOpWithNewOp<spirv::FConvertOp>(op, coopType, operands);
    return true;
  default:
    break;
  }
  return false;
}

bool allOperandsHaveSameCoopMatrixType(ValueRange operands) {
  assert(!operands.empty());
  if (!llvm::all_equal(
          llvm::map_range(operands, [](Value v) { return v.getType(); })))
    return false;

  return isa<spirv::CooperativeMatrixType, spirv::CooperativeMatrixNVType>(
      operands.front().getType());
}

namespace {
/// Converts GPU MMA ConstantMatrixOp to constant SPIR-V KHR/NV cooperative
/// matrix ops.
struct WmmaConstantOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaConstantMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaConstantMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    Value cst = adaptor.getOperands().front();
    auto coopType = getTypeConverter()->convertType(op.getType());
    if (!coopType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, coopType, cst);
    return success();
  }
};

/// Converts elementwise ops to SPIR-V cooperative matrix elementwise ops for
/// the default case.
struct WmmaElementwiseOpToSPIRVDefaultLowering final
    : OpConversionPattern<gpu::SubgroupMmaElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // All operands should be of cooperative matrix types.
    if (!allOperandsHaveSameCoopMatrixType(adaptor.getOperands())) {
      return rewriter.notifyMatchFailure(op,
                                         "not all operands are coop matrices");
    }

    auto coopType = getTypeConverter()->convertType(op.getType());
    if (!coopType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    return success(
        createElementwiseOp(rewriter, op, coopType, adaptor.getOperands()));
  }
};

/// Converts elementwise ops to SPIR-V cooperative matrix elementwise ops for
/// matrix times scalar case.
struct WmmaElementwiseOpToSPIRVScalarMulLowering final
    : OpConversionPattern<gpu::SubgroupMmaElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 2)
      return failure();

    // All operands should be of cooperative matrix types.
    if (!allOperandsHaveSameCoopMatrixType(adaptor.getOperands())) {
      return rewriter.notifyMatchFailure(op,
                                         "not all operands are coop matrices");
    }

    if (op.getOpType() != gpu::MMAElementwiseOp::MULF)
      return failure();

    // Use the original operands to check whether one of the operands is a splat
    // scalar value.
    Value lhs = op.getOperands().front();
    Value rhs = op.getOperands().back();
    Value splat = nullptr;
    Value matrix = nullptr;
    if (lhs.getDefiningOp<gpu::SubgroupMmaConstantMatrixOp>()) {
      splat = adaptor.getOperands().front();
      matrix = adaptor.getOperands().back();
    } else if (rhs.getDefiningOp<gpu::SubgroupMmaConstantMatrixOp>()) {
      matrix = adaptor.getOperands().front();
      splat = adaptor.getOperands().back();
    }
    if (!splat || !matrix)
      return rewriter.notifyMatchFailure(op, "no splat operand");

    // Constant MMA matrix ops are converted to `spirv.CompositeConstruct` ops.
    Value scalar;
    auto cc = splat.getDefiningOp<spirv::CompositeConstructOp>();
    if (!cc) {
      return rewriter.notifyMatchFailure(op,
                                         "splat is not a composite construct");
    }

    assert(cc.getConstituents().size() == 1);
    scalar = cc.getConstituents().front();

    auto coopType = getTypeConverter()->convertType(op.getType());
    if (!coopType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<spirv::MatrixTimesScalarOp>(
        op, coopType, ValueRange{matrix, scalar});
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SPV_KHR_cooperative_matrix
//===----------------------------------------------------------------------===//

namespace khr {
namespace {

/// Converts the GPU MMA loadOp to KHRCooperativeMatrixLoad op in the SPIRV
/// dialect.
struct WmmaLoadOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    Location loc = op->getLoc();

    auto retType = cast<gpu::MMAMatrixType>(op.getRes().getType());
    MemRefType memrefType = op.getSrcMemref().getType();
    Value bufferPtr =
        spirv::getElementPtr(typeConverter, memrefType, adaptor.getSrcMemref(),
                             adaptor.getIndices(), loc, rewriter);

    auto coopType =
        typeConverter.convertType<spirv::CooperativeMatrixType>(retType);
    if (!coopType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    int64_t stride = op.getLeadDimension().getSExtValue();
    IntegerType i32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, IntegerAttr::get(i32Type, stride));

    bool isColMajor = op.getTranspose().value_or(false);
    auto layout = isColMajor ? spirv::CooperativeMatrixLayoutKHR::ColumnMajor
                             : spirv::CooperativeMatrixLayoutKHR::RowMajor;

    rewriter.replaceOpWithNewOp<spirv::KHRCooperativeMatrixLoadOp>(
        op, coopType, bufferPtr, strideValue, layout);
    return success();
  }
};

/// Converts the GPU MMA StoreOp to KHRCooperativeMatrixStore op in the SPIRV
/// dialect.
struct WmmaStoreOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaStoreMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    Location loc = op->getLoc();

    auto memrefType = cast<MemRefType>(op.getDstMemref().getType());
    Value bufferPtr =
        spirv::getElementPtr(typeConverter, memrefType, adaptor.getDstMemref(),
                             adaptor.getIndices(), loc, rewriter);

    int64_t stride = op.getLeadDimension().getSExtValue();
    IntegerType i32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, IntegerAttr::get(i32Type, stride));

    bool isColMajor = op.getTranspose().value_or(false);
    auto layout = isColMajor ? spirv::CooperativeMatrixLayoutKHR::ColumnMajor
                             : spirv::CooperativeMatrixLayoutKHR::RowMajor;

    rewriter.replaceOpWithNewOp<spirv::KHRCooperativeMatrixStoreOp>(
        op, bufferPtr, adaptor.getSrc(), strideValue, layout);
    return success();
  }
};

/// Converts GPU MMA Compute to KHRCooperativeMatrixMulAdd op in the SPIRV
/// dialect.
struct WmmaMmaOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaComputeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaComputeOp subgroupMmaComputeOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::KHRCooperativeMatrixMulAddOp>(
        subgroupMmaComputeOp, adaptor.getOpA(), adaptor.getOpB(),
        adaptor.getOpC());
    return success();
  }
};

} // namespace
} // namespace khr

//===----------------------------------------------------------------------===//
// SPV_NV_cooperative_matrix
//===----------------------------------------------------------------------===//

namespace nv {
namespace {

/// Converts the GPU MMA loadOp to NVCooperativeMatrixLoad op in the SPIRV
/// dialect.
struct WmmaLoadOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp subgroupMmaLoadMatrixOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = subgroupMmaLoadMatrixOp->getLoc();
    gpu::MMAMatrixType retType =
        cast<gpu::MMAMatrixType>(subgroupMmaLoadMatrixOp.getRes().getType());
    auto memrefType =
        cast<MemRefType>(subgroupMmaLoadMatrixOp.getSrcMemref().getType());
    Value bufferPtr = spirv::getElementPtr(
        *getTypeConverter<const SPIRVTypeConverter>(), memrefType,
        adaptor.getSrcMemref(), adaptor.getIndices(), loc, rewriter);
    auto coopType = convertMMAToSPIRVCoopMatrixNVType(retType);
    int64_t stride = subgroupMmaLoadMatrixOp.getLeadDimension().getSExtValue();
    auto i32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, IntegerAttr::get(i32Type, stride));
    bool isColMajor = static_cast<bool>(subgroupMmaLoadMatrixOp.getTranspose());
    auto columnMajor = rewriter.create<spirv::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(isColMajor));
    rewriter.replaceOpWithNewOp<spirv::NVCooperativeMatrixLoadOp>(
        subgroupMmaLoadMatrixOp, coopType, bufferPtr, strideValue, columnMajor,
        spirv::MemoryAccessAttr());
    return success();
  }
};

/// Converts the GPU MMA StoreOp to NVCooperativeMatrixStore op in the SPIRV
/// dialect.
struct WmmaStoreOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaStoreMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp subgroupMmaStoreMatrixOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = subgroupMmaStoreMatrixOp->getLoc();
    auto memrefType =
        cast<MemRefType>(subgroupMmaStoreMatrixOp.getDstMemref().getType());
    Value bufferPtr = spirv::getElementPtr(
        *getTypeConverter<const SPIRVTypeConverter>(), memrefType,
        adaptor.getDstMemref(), adaptor.getIndices(), loc, rewriter);
    int64_t stride = subgroupMmaStoreMatrixOp.getLeadDimension().getSExtValue();
    auto i32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, IntegerAttr::get(i32Type, stride));
    bool useColMajor =
        static_cast<bool>(subgroupMmaStoreMatrixOp.getTranspose());
    auto columnMajor = rewriter.create<spirv::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(useColMajor));
    rewriter.replaceOpWithNewOp<spirv::NVCooperativeMatrixStoreOp>(
        subgroupMmaStoreMatrixOp, bufferPtr, adaptor.getSrc(), strideValue,
        columnMajor, spirv::MemoryAccessAttr());
    return success();
  }
};

/// Converts GPU MMA Compute to
/// NVCooperativeMatrixMulAdd op in the SPIRV dialect.
struct WmmaMmaOpToSPIRVLowering final
    : OpConversionPattern<gpu::SubgroupMmaComputeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaComputeOp subgroupMmaComputeOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::NVCooperativeMatrixMulAddOp>(
        subgroupMmaComputeOp, adaptor.getOpC().getType(), adaptor.getOpA(),
        adaptor.getOpB(), adaptor.getOpC());
    return success();
  }
};

} // namespace
} // namespace nv
} // namespace mlir

mlir::spirv::CooperativeMatrixNVType
mlir::convertMMAToSPIRVCoopMatrixNVType(gpu::MMAMatrixType type) {
  ArrayRef<int64_t> retTypeShape = type.getShape();
  Type elementType = type.getElementType();
  return spirv::CooperativeMatrixNVType::get(
      elementType, spirv::Scope::Subgroup, retTypeShape[0], retTypeShape[1]);
}

mlir::spirv::CooperativeMatrixType
mlir::convertMMAToSPIRVCoopMatrixType(gpu::MMAMatrixType type) {
  ArrayRef<int64_t> retTypeShape = type.getShape();
  Type elementType = type.getElementType();

  auto use =
      llvm::StringSwitch<spirv::CooperativeMatrixUseKHR>(type.getOperand())
          .Case("AOp", spirv::CooperativeMatrixUseKHR::MatrixA)
          .Case("BOp", spirv::CooperativeMatrixUseKHR::MatrixB)
          .Default(spirv::CooperativeMatrixUseKHR::MatrixAcc);

  return spirv::CooperativeMatrixType::get(elementType, retTypeShape[0],
                                           retTypeShape[1],
                                           spirv::Scope::Subgroup, use);
}

void mlir::populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(
    SPIRVTypeConverter &converter, RewritePatternSet &patterns) {
  using namespace mlir;
  MLIRContext *context = patterns.getContext();
  patterns.add<khr::WmmaLoadOpToSPIRVLowering, khr::WmmaMmaOpToSPIRVLowering,
               khr::WmmaStoreOpToSPIRVLowering, WmmaConstantOpToSPIRVLowering,
               WmmaElementwiseOpToSPIRVDefaultLowering>(converter, context);
  // Give the following patterns higher benefit to prevail over the default one.
  patterns.add<WmmaElementwiseOpToSPIRVScalarMulLowering>(converter, context,
                                                          /*benefit=*/2);
}

void mlir::populateGpuWMMAToSPIRVCoopMatrixNVConversionPatterns(
    SPIRVTypeConverter &converter, RewritePatternSet &patterns) {
  using namespace mlir;
  MLIRContext *context = patterns.getContext();
  patterns.add<nv::WmmaLoadOpToSPIRVLowering, nv::WmmaMmaOpToSPIRVLowering,
               nv::WmmaStoreOpToSPIRVLowering, WmmaConstantOpToSPIRVLowering,
               WmmaElementwiseOpToSPIRVDefaultLowering>(converter, context);
  // Give the following patterns higher benefit to prevail over the default one.
  patterns.add<WmmaElementwiseOpToSPIRVScalarMulLowering>(converter, context,
                                                          /*benefit=*/2);
}
