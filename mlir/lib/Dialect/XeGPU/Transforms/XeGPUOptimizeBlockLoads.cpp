//===- XeGPUOptimizeBlockLoads.cpp - XeGPU optimize block loads -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUOPTIMIZEBLOCKLOADS
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-optimize-block-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

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
/// Example:
///     !xegpu.tensor_desc<16x16xf16,
///         #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>>
/// In this case, lane layout is transposed (from the usual [1, SG_SIZE] form)
/// indicating that this is a load that requires transpose effect. However,
/// lane data is [1, 2], meaning that each lane must grab 2 f16 elements from
/// the inner dimension. We convert this to a optimized form by converting the
/// tensor_desc to i32 type such that lane data becomes [1, 1]. This makes the
/// later lowering easily use the load with transpose instruction.
static bool canBeOptimizedForTranspose(ArrayRef<int64_t> laneLayout,
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
static bool canBeOptimizedForTranspose(xegpu::TensorDescType tdescType) {
  // If the dtype is greater or equal to 32 bits, layout must be valid.
  int elementTyBitwidth = tdescType.getElementType().getIntOrFloatBitWidth();
  if (elementTyBitwidth >= 32)
    return false;
  auto maybeLaneLayout = getMaybeLaneLayout(tdescType);
  auto maybeLaneData = getMaybeLaneData(tdescType);
  if (!maybeLaneData || !maybeLaneLayout)
    return false;
  return canBeOptimizedForTranspose(*maybeLaneLayout, *maybeLaneData);
}

/// Check if a tensor desc type can be optimized for transpose, if so return the
/// new optimized tensor desc type with a valid transpose layout.
static xegpu::TensorDescType tryOptimize(xegpu::TensorDescType tdescType,
                                         const uArch *targetuArch) {
  if (!canBeOptimizedForTranspose(tdescType))
    return tdescType;
  auto laneData = getMaybeLaneData(tdescType)
                      .value(); // Lane data must exist if we reach here.
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
  auto *blockLoadTarget = dyn_cast<Subgroup2DBlockLoadInstruction>(
      targetuArch->getInstruction(InstructionKind::Subgroup2DBlockLoad));
  auto maybeHWParams = blockLoadTarget->getBlockWidthHeightCount(
      newElemTy, /** has transform */ false, /** has transpose */ true);
  // If no HW params found, return the original type.
  if (!maybeHWParams)
    return tdescType;
  auto [widths, heights, counts] = maybeHWParams.value();
  // TODO: Currently we expect array length to be 1 for transpose case.
  if (counts.size() != 1 || counts[0] != 1)
    return tdescType;
  int arrayLen = counts[0];
  int supportedHeight =
      xegpu::getLargestDivisor(static_cast<int>(requiredShape[0]), heights);
  int supportedWidth =
      xegpu::getLargestDivisor(static_cast<int>(requiredShape[1]), widths);
  // If no supported height or width found, return the original type.
  if (supportedHeight == -1 || supportedWidth == -1)
    return tdescType;

  SmallVector<int64_t> supportedShape = {supportedHeight, supportedWidth};
  xegpu::LayoutAttr newLayout = xegpu::LayoutAttr::get(
      tdescType.getContext(),
      tdescType.getLayoutAttr().getLaneLayout().asArrayRef(), {1, 1});
  // Array length can not be larger than 1 for transpose case.
  return xegpu::TensorDescType::get(supportedShape, newElemTy, arrayLen,
                                    tdescType.getBoundaryCheck(),
                                    tdescType.getMemorySpace(), newLayout);
}

/// Helper to convert an OpFoldResult to Value.
static Value convertToValue(ConversionPatternRewriter &rewriter, Location loc,
                            OpFoldResult ofr) {
  std::optional<int64_t> mayBeInt = getConstantIntValue(ofr);
  if (mayBeInt)
    return arith::ConstantIndexOp::create(rewriter, loc, *mayBeInt).getResult();
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
               arith::ConstantIndexOp::create(rewriter, loc, shiftAmount)
                   .getResult())
        .getResult();
  }
  auto constantOp =
      arith::ConstantIndexOp::create(rewriter, loc, constant).getResult();
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
  Value offsetDim0 = convertToValue(rewriter, loc, offsets[offsets.size() - 2]);
  Value offsetDim1 = convertToValue(rewriter, loc, offsets[offsets.size() - 1]);
  SmallVector<int64_t> supportedShape(newTensorDesc.getType().getShape());
  // Compute the ratio between original shape and supported shape. We need to
  // generate loads in this ratio arrangement.
  auto shapeRatio = computeShapeRatio(data.getType().getShape(),
                                      supportedShape)
                        .value(); // `ratio` must be defined if we reach here.
  for (int64_t h = 0; h < shapeRatio[0]; ++h) {
    for (int64_t w = 0; w < shapeRatio[1]; ++w) {
      int64_t localOffsetDim0 = h * supportedShape[0];
      int64_t localOffsetDim1 = w * supportedShape[1];
      Value loadOffsetX = arith::AddIOp::create(
          rewriter, loc, offsetDim0,
          arith::ConstantIndexOp::create(rewriter, loc, localOffsetDim0)
              .getResult());
      Value loadOffsetY = arith::AddIOp::create(
          rewriter, loc, offsetDim1,
          arith::ConstantIndexOp::create(rewriter, loc, localOffsetDim1)
              .getResult());
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
          ArrayRef<int64_t>{localOffsetDim0, localOffsetDim1},
          ArrayRef<int64_t>{1, 1});
      // InsertOp must have the same layout as newTensorDesc.
      xegpu::setDistributeLayoutAttr(insertOp->getOpResult(0), layoutAttr);
      data = insertOp.getResult();
    }
  }
  return data;
}

/// Checks if a CreateNdDescOp can be optimized for transpose, if so creates a
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
    // Get the target uArch info.
    auto chipStr = xegpu::getChipStr(createNdOp);
    // Check if the chip is supported.
    assert(
        chipStr && (chipStr.value() == "pvc" || chipStr.value() == "bmg") &&
        "Expecting target chip to be pvc or bmg for transpose optimization.");
    const uArch *targetuArch = xegpu::uArch::getUArch(chipStr.value());

    auto convertType = tryOptimize(tdescTy, targetuArch);
    if (convertType == tdescTy)
      return failure();
    auto strides = createNdOp.getMixedStrides();
    auto maybeConstInnerStride = getConstantIntValue(strides.back());
    // Only row-major memrefs are expected for now.
    if (!maybeConstInnerStride || *maybeConstInnerStride != 1)
      return rewriter.notifyMatchFailure(
          createNdOp, "Expecting row-major memref for transpose optimization.");
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
      source = arith::IndexCastOp::create(rewriter, createNdOp.getLoc(),
                                          rewriter.getI64Type(),
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
            arith::AddIOp::create(
                rewriter, loadNdOp->getLoc(), offsetY,
                arith::ConstantIndexOp::create(rewriter, loadNdOp->getLoc(),
                                               i * origDataShape[1])
                    .getResult())
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

void xegpu::populateXeGPUOptimizeBlockLoadsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<XeGPUCreateNdDescOpPattern, XeGPULoadNdDescOpPattern,
               VectorExtractOpPattern>(patterns.getContext());
}

namespace {

struct XeGPUOptimizeBlockLoadsPass final
    : public xegpu::impl::XeGPUOptimizeBlockLoadsBase<
          XeGPUOptimizeBlockLoadsPass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter converter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    // This pass is only meant for PVC and BMG targets. If unsupported target
    // is found, exit early.
    bool isTargetSupported = false;
    getOperation()->walk([&](gpu::GPUFuncOp funcOp) {
      auto chipStr = xegpu::getChipStr(funcOp);
      if (chipStr && (chipStr.value() == "pvc" || chipStr.value() == "bmg"))
        isTargetSupported = true;
    });

    if (!isTargetSupported) {
      DBGS() << "XeGPUOptimizeBlockLoadsPass only supports PVC and BMG targets."
             << "\n";
      return;
    }

    // CreateNdDescOp and LoadNdOp with optimizable tensor desc types must be
    // converted.
    target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
        [&](xegpu::CreateNdDescOp createNdOp) {
          return !canBeOptimizedForTranspose(createNdOp.getType());
        });
    target.addDynamicallyLegalOp<xegpu::LoadNdOp>(
        [&](xegpu::LoadNdOp loadNdOp) {
          return !canBeOptimizedForTranspose(loadNdOp.getTensorDescType());
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
          return !canBeOptimizedForTranspose(laneLayout, laneData);
        });
    converter.addConversion([](Type type) { return type; });

    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           vector::VectorDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    xegpu::populateXeGPUOptimizeBlockLoadsPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      DBGS() << "Optimize block loads pass failed.\n";
      return signalPassFailure();
    }
  }
};

} // namespace
