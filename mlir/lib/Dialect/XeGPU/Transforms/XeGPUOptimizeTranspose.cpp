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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
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

namespace {

struct Allowed2DShapeRange {
  int64_t minWidth, maxWidth, minHeight, maxHeight;
};

/// Helper to get the size range of a 2D block that can be transposed by HW.
/// TODO: Use uArch to get supported block ranges.
static Allowed2DShapeRange getTransposableBlockRange(int bitWidth) {
  switch (bitWidth) {
  case 32:
    return {/**min width**/ 1, /**max width**/ 8, /**min height**/ 1,
            /**max height**/ 32};
  default:
    llvm_unreachable("Add support for other element bitwidths");
  }
}

/// Get the 2D lane data from a tensor desc type if it exists.
static std::optional<SmallVector<int64_t>>
getMaybeLaneData(xegpu::TensorDescType tdescType) {
  auto layout = tdescType.getLayoutAttr();
  if (!layout)
    return std::nullopt;
  auto laneData = layout.getEffectiveLaneDataAsInt();
  if (laneData.size() != 2)
    return std::nullopt;
  return laneData;
}

/// Get the 2D lane layout from a tensor desc type if it exists.
static std::optional<SmallVector<int64_t>>
getMaybeLaneLayout(xegpu::TensorDescType tdescType) {
  auto layout = tdescType.getLayoutAttr();
  if (!layout)
    return std::nullopt;
  auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
  if (laneLayout.size() != 2)
    return std::nullopt;
  return laneLayout;
}

/// A layout can be optimized if its lane layout is transposed (lane[0] != 1 &&
/// lane[1] == 1), but inner lane data is not equal to [1, 1].
static bool canBeOptimized(ArrayRef<int64_t> laneLayout,
                           ArrayRef<int64_t> laneData) {
  if (laneLayout.size() != 2 || laneData.size() != 2)
    return false;
  if (laneLayout[0] == 1 || laneLayout[1] != 1)
    return false;
  if (laneData[0] != 1 || laneData[1] == 1)
    return false;
  return true;
}

/// A tensor desc type can be optimized if its element type is less than 32 bits
/// and its layout can be optimized.
static bool canBeOptimized(xegpu::TensorDescType tdescType) {
  // If the dtype is greater or equal to 32 bits, layout must be valid.
  int elementTyBitwidth = tdescType.getElementType().getIntOrFloatBitWidth();
  if (elementTyBitwidth >= 32)
    return false;
  auto maybeLaneLayout = getMaybeLaneLayout(tdescType);
  auto maybeLaneData = getMaybeLaneData(tdescType);
  if (!maybeLaneData || !maybeLaneLayout)
    return false;
  return canBeOptimized(*maybeLaneLayout, *maybeLaneData);
}

/// Check if a tensor desc type can be optimized for transpose, if so return the
/// new optimized tensor desc type with a valid transpose layout.
static xegpu::TensorDescType tryOptimize(xegpu::TensorDescType tdescType) {
  if (!canBeOptimized(tdescType))
    return tdescType;
  auto laneData = getMaybeLaneData(tdescType).value();
  int64_t innerLaneData = laneData[1];
  int elementTyBitwidth = tdescType.getElementType().getIntOrFloatBitWidth();
  // Required shape is total shape of the vector result that this tensor desc
  // must eventually load after adjusting for the new bitwidth and array
  // length.
  SmallVector<int64_t> requiredShape(tdescType.getShape());
  requiredShape.back() =
      requiredShape.back() * tdescType.getArrayLength() / innerLaneData;
  int newBitWidth = elementTyBitwidth * innerLaneData;
  Type newElemTy = IntegerType::get(tdescType.getContext(), newBitWidth);
  // Supported shape is the max transpose shape that can be supported by
  // hardware that is less than or equal to required shape.
  auto supportedHeight = std::min(
      requiredShape[0], getTransposableBlockRange(newBitWidth).maxHeight);
  auto supportedWidth = std::min(
      requiredShape[1], getTransposableBlockRange(newBitWidth).maxWidth);
  SmallVector<int64_t> supportedShape = {supportedHeight, supportedWidth};

  // Required shape must be multiple of supported shape. Otherwise, we can not
  // optimize it.
  // TODO: Supported shape can be adjusted to handle non-multiple cases.
  if (requiredShape[0] % supportedShape[0] != 0 ||
      requiredShape[1] % supportedShape[1] != 0)
    return tdescType;

  xegpu::LayoutAttr newLayout = xegpu::LayoutAttr::get(
      tdescType.getContext(),
      tdescType.getLayoutAttr().getLaneLayout().asArrayRef(), {1, 1});
  // Array length can not be larger than 1 for transpose case.
  return xegpu::TensorDescType::get(
      supportedShape, newElemTy, /**array length**/ 1,
      tdescType.getBoundaryCheck(), tdescType.getMemorySpace(), newLayout);
}

/// Helper to create a constant index value.
static Value createConstantIndex(ConversionPatternRewriter &rewriter,
                                 Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(rewriter, loc, value).getResult();
}

/// Helper to convert an OpFoldResult to Value.
static Value convertToValue(ConversionPatternRewriter &rewriter, Location loc,
                            OpFoldResult ofr) {
  std::optional<int64_t> mayBeInt = getConstantIntValue(ofr);
  if (mayBeInt)
    return createConstantIndex(rewriter, loc, *mayBeInt);
  return llvm::cast<Value>(ofr);
}

/// Helper to divide a Value by a constant integer.
static Value divideByConstant(ConversionPatternRewriter &rewriter, Location loc,
                              Value val, int64_t constant) {
  // If the constant is a power of 2, use right shift for division.
  if (llvm::isPowerOf2_64(constant)) {
    int64_t shiftAmount = llvm::Log2_64(constant);
    return arith::ShRUIOp::create(
               rewriter, loc, val,
               createConstantIndex(rewriter, loc, shiftAmount))
        .getResult();
  }
  auto constantOp = createConstantIndex(rewriter, loc, constant);
  return arith::DivUIOp::create(rewriter, loc, val, constantOp).getResult();
}

/// This function takes a larger register block `data` and generates multiple
/// smaller loads (size given by `newTensorDesc`) to fill in the `data` block
/// starting from `offsets`.
static Value generateLoads(ConversionPatternRewriter &rewriter,
                           TypedValue<VectorType> data,
                           SmallVector<OpFoldResult> offsets,
                           TypedValue<xegpu::TensorDescType> newTensorDesc,
                           xegpu::LoadNdOp origLoadOp) {
  Location loc = data.getLoc();
  assert(offsets.size() >= 2 && "Expecting at least 2 offsets for 2D LoadNdOp");
  Value offsetX = convertToValue(rewriter, loc, offsets[offsets.size() - 2]);
  Value offsetY = convertToValue(rewriter, loc, offsets[offsets.size() - 1]);
  SmallVector<int64_t> supportedShape(newTensorDesc.getType().getShape());
  // Compute the ratio between original shape and supported shape. We need to
  // generate loads in this ratio arrangement.
  auto shapeRatio = computeShapeRatio(data.getType().getShape(),
                                      supportedShape)
                        .value(); // `ratio` must be defined if we reach here.
  for (int64_t h = 0; h < shapeRatio[0]; ++h) {
    for (int64_t w = 0; w < shapeRatio[1]; ++w) {
      int64_t localOffsetX = h * supportedShape[0];
      int64_t localOffsetY = w * supportedShape[1];
      Value loadOffsetX = arith::AddIOp::create(
          rewriter, loc, offsetX,
          createConstantIndex(rewriter, loc, localOffsetX));
      Value loadOffsetY = arith::AddIOp::create(
          rewriter, loc, offsetY,
          createConstantIndex(rewriter, loc, localOffsetY));
      auto loadOp = xegpu::LoadNdOp::create(
          rewriter, loc,
          VectorType::get(supportedShape, data.getType().getElementType()),
          newTensorDesc, ArrayRef<OpFoldResult>{loadOffsetX, loadOffsetY},
          origLoadOp.getPackedAttr(), origLoadOp.getTransposeAttr(),
          origLoadOp.getL1HintAttr(), origLoadOp.getL2HintAttr(),
          origLoadOp.getL3HintAttr());
      // Set the layout for the loadOp.
      auto layoutAttr = newTensorDesc.getType().getLayoutAttr();
      xegpu::setDistributeLayoutAttr(loadOp->getOpResult(0), layoutAttr);
      // Insert the loaded block into the right position in data.
      auto insertOp = vector::InsertStridedSliceOp::create(
          rewriter, loc, loadOp.getResult(), data,
          ArrayRef<int64_t>{localOffsetX, localOffsetY},
          ArrayRef<int64_t>{1, 1});
      // InsertOp must have the same layout as newTensorDesc.
      xegpu::setDistributeLayoutAttr(insertOp->getOpResult(0), layoutAttr);
      data = insertOp.getResult();
    }
  }
  return data;
}

/// Checks is a CreateNdDescOp can be optimized for transpose, if so creates a
/// new CreateNdDescOp with optimized tensor desc type. This involves extracting
/// the base pointer from the original memory source and adjusting the shape and
/// strides of the tensor desc to fit with the new optimized transpose layout.
class XeGPUCreateNdDescOpPattern final
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
public:
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp createNdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tdescTy = createNdOp.getType();
    auto convertType = tryOptimize(tdescTy);
    if (convertType == tdescTy)
      return failure();
    auto strides = createNdOp.getMixedStrides();
    auto maybeConstInnerStride = getConstantIntValue(strides.back());
    // Only row-major memrefs are expected for now.
    if (!maybeConstInnerStride || *maybeConstInnerStride != 1)
      return failure();
    Value source = createNdOp.getSource();
    auto optionalLaneData = getMaybeLaneData(tdescTy);
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

/// Checks if a LoadNdOp consumes a tensor desc type that was rewritten for
/// tranpose optimization. If so, rewrites the LoadNdOp to to align with the
/// adjusted tensor desc type. This can result in multiple LoadNdOps being
/// generated to fill in the original load shape.
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
    auto laneData = getMaybeLaneData(loadNdOp.getTensorDescType()).value();
    int64_t innerLaneData = laneData[1];
    auto offsets = loadNdOp.getMixedOffsets();
    if (offsets.empty())
      return rewriter.notifyMatchFailure(loadNdOp,
                                         "Expecting offsets in LoadNd");
    SmallVector<OpFoldResult> modifiedOffsets(offsets);
    modifiedOffsets.back() = divideByConstant(
        rewriter, loadNdOp.getLoc(),
        convertToValue(rewriter, loadNdOp.getLoc(), modifiedOffsets.back()),
        innerLaneData);
    // Get the 2D data shape of this loadNdOp in its original type including
    // array length.
    SmallVector<int64_t> origDataShape(origTensorDescType.getShape());
    // Adjust the data shape based on innerLaneData.
    origDataShape.back() /= innerLaneData;
    // HW supported shape is the new tensor desc shape after conversion.
    SmallVector<int64_t> hwSupportedShape(adaptorType.getShape());
    VectorType origVectorType =
        VectorType::get(origDataShape, adaptorType.getElementType());
    Value data;
    // Orig data shape is 3D for the array length case.
    if (origTensorDescType.getArrayLength() > 1) {
      SmallVector<Value> arraySlices;
      for (int64_t i = 0; i < origTensorDescType.getArrayLength(); ++i) {
        Value slice = arith::ConstantOp::create(
            rewriter, loadNdOp->getLoc(), origVectorType,
            rewriter.getZeroAttr(origVectorType));
        // Increase the Y offset for each array slice.
        Value offsetY = convertToValue(rewriter, loadNdOp->getLoc(),
                                       modifiedOffsets.back());
        modifiedOffsets.back() =
            arith::AddIOp::create(rewriter, loadNdOp->getLoc(), offsetY,
                                  createConstantIndex(rewriter,
                                                      loadNdOp->getLoc(),
                                                      i * origDataShape[1]))
                .getResult();
        slice = generateLoads(
            rewriter, cast<TypedValue<VectorType>>(slice), modifiedOffsets,
            cast<TypedValue<xegpu::TensorDescType>>(adaptor.getTensorDesc()),
            loadNdOp);
        // BitCast back to original load shape without array length.
        auto bitcastType = VectorType::get(origTensorDescType.getShape(),
                                           origTensorDescType.getElementType());
        auto bitCastOp = vector::BitCastOp::create(rewriter, loadNdOp->getLoc(),
                                                   bitcastType, slice);
        // BitCastOp must have the same layout as the original loadNdOp.
        xegpu::setDistributeLayoutAttr(bitCastOp->getOpResult(0),
                                       origTensorDescType.getLayoutAttr());
        arraySlices.push_back(bitCastOp.getResult());
      }
      rewriter.replaceOpWithMultiple(loadNdOp, {arraySlices});
      return success();
    }
    data = arith::ConstantOp::create(
        rewriter, loadNdOp->getLoc(),
        VectorType::get(origDataShape, adaptorType.getElementType()),
        rewriter.getZeroAttr(origVectorType));
    data = generateLoads(
        rewriter, cast<TypedValue<VectorType>>(data), modifiedOffsets,
        cast<TypedValue<xegpu::TensorDescType>>(adaptor.getTensorDesc()),
        loadNdOp);
    auto bitCastOp = vector::BitCastOp::create(rewriter, loadNdOp->getLoc(),
                                               loadNdOp.getType(), data);
    // BitCastOp must have the same layout as the original loadNdOp.
    xegpu::setDistributeLayoutAttr(bitCastOp->getOpResult(0),
                                   origTensorDescType.getLayoutAttr());
    rewriter.replaceOp(loadNdOp, bitCastOp);
    return success();
  }
};

/// Vector ExtractOp must be processed if the original tensor desc type has
/// array length greater than 1. In this case, the LoadNdOp is replaced with
/// multiple LoadNdOps for each array slice making the extraction unnecessary.
/// In this case, we simply remove the ExtractOp.
class VectorExtractOpPattern final
    : public OpConversionPattern<vector::ExtractOp> {
public:
  using OpConversionPattern<vector::ExtractOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the source of the extraction is split to multiple values.
    if (adaptor.getSource().size() == 1)
      return failure();
    auto mixedPos = extractOp.getMixedPosition();
    if (mixedPos.size() != 1)
      return failure();
    auto mayBeInt = getConstantIntValue(mixedPos[0]);
    if (!mayBeInt)
      return failure();
    rewriter.replaceOp(extractOp, adaptor.getSource()[*mayBeInt]);
    return success();
  }
};

} // namespace

void xegpu::populateXeGPUOptimizeTransposePatterns(
    RewritePatternSet &patterns) {
  patterns.add<XeGPUCreateNdDescOpPattern, XeGPULoadNdDescOpPattern,
               VectorExtractOpPattern>(patterns.getContext());
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

    // CreateNdDescOp and LoadNdOp with optimizable tensor desc types must be
    // converted.
    target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
        [&](xegpu::CreateNdDescOp createNdOp) {
          return !canBeOptimized(createNdOp.getType());
        });
    target.addDynamicallyLegalOp<xegpu::LoadNdOp>(
        [&](xegpu::LoadNdOp loadNdOp) {
          return !canBeOptimized(loadNdOp.getTensorDescType());
        });
    // Vector ExtractOps can have optimizable layouts if they extract from
    // LoadNdOps with array length greater than 1. These ExtractOps must be
    // converted.
    target.addDynamicallyLegalOp<vector::ExtractOp>(
        [&](vector::ExtractOp extractOp) {
          auto layout = xegpu::getDistributeLayoutAttr(extractOp.getResult());
          if (!layout)
            return true;
          auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
          auto laneData = layout.getEffectiveLaneDataAsInt();
          return !canBeOptimized(laneLayout, laneData);
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
