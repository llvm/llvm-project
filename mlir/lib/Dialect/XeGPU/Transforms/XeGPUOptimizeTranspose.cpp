//===- XeGPUOptimizeTranspose.cpp - XeGPU optimize transpose ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUOPTIMIZETRANSPOSE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-optimize-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

struct TransposableBlockRange {
  int minWidth, maxWidth, minHeight, maxHeight;
};

// TODO: Use uArch to get supported block ranges.
static TransposableBlockRange getBlockRange(int bitWidth) {
  switch (bitWidth) {
  case 32:
    return {/**min width**/ 1, /**max width**/ 8, /**min height**/ 1,
            /**max height**/ 32};
  default:
    llvm_unreachable("Add support for other element bitwidths");
  }
}

namespace {

static std::optional<SmallVector<int64_t>>
get2DLaneData(xegpu::TensorDescType tdescType) {
  auto layout = tdescType.getLayoutAttr();
  if (!layout)
    return std::nullopt;
  auto laneData = layout.getEffectiveLaneDataAsInt();
  if (laneData.size() != 2)
    return std::nullopt;
  return laneData;
}

static xegpu::TensorDescType
getModifiedTensorDescType(xegpu::TensorDescType tdescType) {
  auto optionalLaneData = get2DLaneData(tdescType);
  if (!optionalLaneData)
    return tdescType;
  auto laneData = optionalLaneData.value();
  int64_t innerLaneData = laneData[1];
  if (laneData[0] == 1 && innerLaneData != 1) {
    int elementTyBitwidth = tdescType.getElementType().getIntOrFloatBitWidth();
    assert(elementTyBitwidth < 32 &&
           "Expected element type bitwidth < 32 with laneData[1] != 1");
    SmallVector<int64_t> newShape(tdescType.getShape());
    newShape.back() = newShape.back() / innerLaneData;
    Type newElemTy = IntegerType::get(tdescType.getContext(),
                                      elementTyBitwidth * innerLaneData);
    xegpu::LayoutAttr newLayout = xegpu::LayoutAttr::get(
        tdescType.getContext(),
        tdescType.getLayoutAttr().getLaneLayout().asArrayRef(), {1, 1});
    return xegpu::TensorDescType::get(
        newShape, newElemTy, tdescType.getArrayLength(),
        tdescType.getBoundaryCheck(), tdescType.getMemorySpace(), newLayout);
  }
  return tdescType;
}

static Value convertToValue(ConversionPatternRewriter &rewriter, Location loc,
                            OpFoldResult ofr) {
  std::optional<int64_t> mayBeInt = getConstantIntValue(ofr);
  if (mayBeInt)
    return arith::ConstantIndexOp::create(rewriter, loc, *mayBeInt);
  return llvm::cast<Value>(ofr);
}

static Value divideByConstant(ConversionPatternRewriter &rewriter, Location loc,
                              Value val, int64_t constant) {
  auto constantOp = arith::ConstantIndexOp::create(rewriter, loc, constant);
  return arith::DivUIOp::create(rewriter, loc, val, constantOp.getResult())
      .getResult();
}

class XeGPUCreateNdDescOpPattern final
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
public:
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp createNdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tdescTy = createNdOp.getType();
    auto convertType = getModifiedTensorDescType(tdescTy);
    if (convertType == tdescTy)
      return failure();
    auto strides = createNdOp.getMixedStrides();
    auto maybeConstInnerStride = getConstantIntValue(strides.back());
    // Only row-major memrefs are expected for now.
    if (!maybeConstInnerStride || *maybeConstInnerStride != 1)
      return failure();
    Value source = createNdOp.getSource();
    auto optionalLaneData = get2DLaneData(tdescTy);
    assert(optionalLaneData && "Expected 2D lane data");
    auto laneData = optionalLaneData.value();
    int64_t innerLaneData = laneData[1];
    auto memrefType = dyn_cast<MemRefType>(source.getType());
    // Inner dimension of the shape must be adjusted based on innerLaneData.
    SmallVector<OpFoldResult> modifiedShape(createNdOp.getMixedSizes());
    modifiedShape.back() = divideByConstant(
        rewriter, createNdOp.getLoc(),
        convertToValue(rewriter, createNdOp.getLoc(), modifiedShape.back()),
        innerLaneData);
    // Similarly, second to last stride must be adjusted.
    assert(strides.size() >= 2 &&
           "Expected at least 2 strides for CreateNdDescOp");
    SmallVector<OpFoldResult> modifiedStrides(strides);
    modifiedStrides[modifiedStrides.size() - 2] = divideByConstant(
        rewriter, createNdOp.getLoc(),
        convertToValue(rewriter, createNdOp.getLoc(),
                       modifiedStrides[modifiedStrides.size() - 2]),
        innerLaneData);

    // If the source is a static memref, we need to extract the pointer to
    // base address.
    if (memrefType && memrefType.hasStaticShape()) {
      auto extractOp = memref::ExtractAlignedPointerAsIndexOp::create(
          rewriter, createNdOp.getLoc(), source);
      source = arith::IndexCastOp::create(
                   rewriter, createNdOp.getLoc(),
                   IntegerType::get(rewriter.getContext(), 64),
                   extractOp.getResult())
                   .getResult();
    }
    // Create a new CreateNdDescOp with the modified shape and converted type.
    auto newCreateNdDescOp = xegpu::CreateNdDescOp::create(
        rewriter, createNdOp.getLoc(), convertType, source, modifiedShape,
        modifiedStrides);
    rewriter.replaceOp(createNdOp, newCreateNdDescOp.getResult());
    return success();
  }
};
} // namespace

class XeGPULoadNdDescOpPattern final
    : public OpConversionPattern<xegpu::LoadNdOp> {
public:
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp loadNdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origTensorDescType = loadNdOp.getTensorDescType();
    auto adaptorType =
        cast<xegpu::TensorDescType>(adaptor.getTensorDesc().getType());
    if (adaptorType == origTensorDescType)
      return failure();
    // Offsets must be adjusted based on innerLaneData.
    auto optionalLaneData = get2DLaneData(loadNdOp.getTensorDescType());
    assert(optionalLaneData && "Expected 2D lane data");
    int64_t innerLaneData = optionalLaneData.value()[1];
    auto offsets = loadNdOp.getMixedOffsets();
    if (offsets.empty())
      return rewriter.notifyMatchFailure(loadNdOp,
                                         "Expecting offsets in LoadNd");
    SmallVector<OpFoldResult> modifiedOffsets(offsets);
    modifiedOffsets.back() = divideByConstant(
        rewriter, loadNdOp.getLoc(),
        convertToValue(rewriter, loadNdOp.getLoc(), modifiedOffsets.back()),
        innerLaneData);
    VectorType modifiedType =
        VectorType::get(adaptorType.getShape(), adaptorType.getElementType());
    // Create a new LoadNdOp with modified offsets and type.
    auto newLoadNdOp = xegpu::LoadNdOp::create(
        rewriter, loadNdOp->getLoc(), modifiedType, adaptor.getTensorDesc(),
        modifiedOffsets, loadNdOp.getPackedAttr(), loadNdOp.getTransposeAttr(),
        loadNdOp.getL1HintAttr(), loadNdOp.getL2HintAttr(),
        loadNdOp.getL3HintAttr());
    // Bitcast back to the original type.
    auto castOp = vector::BitCastOp::create(rewriter, loadNdOp->getLoc(),
                                            loadNdOp.getType(), newLoadNdOp);
    // Cast op must have the same layout as the original LoadNdOp result.
    xegpu::setDistributeLayoutAttr(
        castOp->getOpResult(0),
        xegpu::getDistributeLayoutAttr(loadNdOp.getResult()));
    rewriter.replaceOp(loadNdOp, castOp.getResult());
    return success();
  }
};

void xegpu::populateXeGPUOptimizeTransposePatterns(
    RewritePatternSet &patterns) {
  patterns.add<XeGPUCreateNdDescOpPattern, XeGPULoadNdDescOpPattern>(
      patterns.getContext());
}

namespace {

struct XeGPUOptimizeTransposePass final
    : public xegpu::impl::XeGPUOptimizeTransposeBase<
          XeGPUOptimizeTransposePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter converter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    auto checkValidInnerLaneData =
        [](std::optional<SmallVector<int64_t>> optionalLaneData) -> bool {
      if (!optionalLaneData)
        return true;
      auto laneData = optionalLaneData.value();
      return laneData[0] != 1 || laneData[1] == 1;
    };

    target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
        [&](xegpu::CreateNdDescOp createNdOp) {
          auto optionalLaneData = get2DLaneData(createNdOp.getType());
          return checkValidInnerLaneData(optionalLaneData);
        });
    target.addDynamicallyLegalOp<xegpu::LoadNdOp>(
        [&](xegpu::LoadNdOp loadNdOp) {
          auto optionalLaneData = get2DLaneData(loadNdOp.getTensorDescType());
          return checkValidInnerLaneData(optionalLaneData);
        });
    converter.addConversion([](Type type) { return type; });

    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           vector::VectorDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    xegpu::populateXeGPUOptimizeTransposePatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      DBGS() << "Optimize transpose pass failed.\n";
      return signalPassFailure();
    }
  }
};

} // namespace
