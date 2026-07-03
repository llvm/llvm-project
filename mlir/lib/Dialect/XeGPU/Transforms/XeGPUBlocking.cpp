//===---- XeGPUBlocking.cpp ---- XeGPU Blocking Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUBLOCKING
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-blocking"

using namespace mlir;

namespace {

//===------------------------------------------------------------------------===//
// The XeGPUBlockingPass leverages the unroll patterns for XeGPU and Vector ops
// to partition operations that process large shapes into multiple operations on
// smaller shapes, as specified by the inst_data in the layout attribute. This
// enables each resulting operation to be efficiently mapped to a hardware
// instruction.
//===------------------------------------------------------------------------===//

class XeGPUBlockingPass final
    : public xegpu::impl::XeGPUBlockingBase<XeGPUBlockingPass> {
public:
  void runOnOperation() override;

private:
  // Get the tile shape for a given OpOperand or OpResult by examining the
  // corresponding layout attribute. If layout is not present or is not a
  // subgroup level layout, it returns std::nullopt.
  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                        std::is_same_v<T, OpResult>>>
  std::optional<SmallVector<int64_t>>
  getTileShape(const T &operandOrResult) const;

  // Get the tile shape for a given operation.
  std::optional<SmallVector<int64_t>> getTileShape(Operation *op) const;

  // Determine if the operation requires unrolling. Return false if all operands
  // and results have tile shapes identical to their original types. Otherwise,
  // return true.
  bool needsUnroll(Operation *op) const;
};
} // namespace

template <typename T, typename>
std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(const T &operandOrResult) const {
  Value value;
  if constexpr (std::is_same_v<T, OpOperand>) {
    value = operandOrResult.get();
  } else {
    value = (Value)operandOrResult;
  }

  xegpu::DistributeLayoutAttr layout =
      xegpu::getDistributeLayoutAttr(operandOrResult);
  if (layout && layout.isForSubgroup()) {
    if (!layout.getEffectiveInstDataAsInt().empty()) {
      SmallVector<int64_t> instData = layout.getEffectiveInstDataAsInt();
      return instData;
    }
    if (auto type = dyn_cast<ShapedType>(value.getType()))
      return llvm::to_vector(type.getShape());
  }
  LDBG() << "failed to getTileShape for: " << value;
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(Operation *op) const {
  if (isa<xegpu::CreateNdDescOp, xegpu::LoadMatrixOp>(op))
    return getTileShape(op->getOpResult(0));
  if (isa<xegpu::PrefetchNdOp, xegpu::LoadNdOp, xegpu::PrefetchOp,
          xegpu::StoreMatrixOp>(op))
    return getTileShape(op->getOpOperand(0));
  if (isa<xegpu::StoreNdOp>(op))
    return getTileShape(op->getOpOperand(1));

  if (isa<xegpu::LoadGatherOp>(op))
    return getTileShape(op->getOpResult(0));

  if (auto convertLayoutOp = dyn_cast<xegpu::ConvertLayoutOp>(op)) {
    auto inputInstData =
        convertLayoutOp.getInputLayout().getEffectiveInstDataAsInt();
    auto targetInstData =
        convertLayoutOp.getTargetLayout().getEffectiveInstDataAsInt();
    // return the one with larger size
    if (computeProduct(inputInstData) >= computeProduct(targetInstData))
      return inputInstData;
    else
      return targetInstData;
  }

  if (isa<xegpu::StoreScatterOp>(op))
    return getTileShape(op->getOpOperand(0));

  // Helper lambda to validate and get A/B tiles
  auto validateABTiles = [&](Operation *op)
      -> std::optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> {
    std::optional<SmallVector<int64_t>> aTile =
        getTileShape(op->getOpOperand(0));
    std::optional<SmallVector<int64_t>> bTile =
        getTileShape(op->getOpOperand(1));

    if (!aTile || aTile->size() < 2 || !bTile || bTile->size() < 2)
      return std::nullopt;

    // Both must have the same number of batch dimensions.
    int64_t aBatchRank = aTile->size() - 2;
    int64_t bBatchRank = bTile->size() - 2;
    if (aBatchRank != bBatchRank)
      return std::nullopt;

    // Batch dimensions must match.
    for (int64_t i = 0; i < aBatchRank; ++i) {
      if ((*aTile)[i] != (*bTile)[i])
        return std::nullopt;
    }

    // Semantic check for A and B: K dimension must match.
    // A[..., M, K] x B[..., K, N]
    if ((*aTile).back() != (*bTile)[bBatchRank])
      return std::nullopt;

    return std::make_pair(*aTile, *bTile);
  };

  // Helper lambda to validate C tile
  auto validateCTile = [&](Operation *op, unsigned cOperandIdx,
                           const SmallVector<int64_t> &aTile,
                           const SmallVector<int64_t> &bTile) -> bool {
    if (op->getNumOperands() <= cOperandIdx)
      return true;

    std::optional<SmallVector<int64_t>> cTile =
        getTileShape(op->getOpOperand(cOperandIdx));
    if (!cTile)
      return false;
    // Expected C tile: batch dims from A + [M, N]
    int64_t aBatchRank = aTile.size() - 2;
    SmallVector<int64_t> expectedCTile(aTile.begin(),
                                       aTile.begin() + aBatchRank);
    expectedCTile.push_back(aTile[aBatchRank]); // M from A
    expectedCTile.push_back(bTile.back());      // N from B
    if (!llvm::equal(*cTile, expectedCTile))
      return false;
    return true;
  };

  // Helper lambda to validate scale A tile for DpasMxOp
  auto validateScaleATile =
      [&](Operation *op, unsigned scaleAOperandIdx,
          const SmallVector<int64_t> &aTile) -> std::optional<int64_t> {
    std::optional<SmallVector<int64_t>> aScaleTile =
        getTileShape(op->getOpOperand(scaleAOperandIdx));

    if (!aScaleTile || aScaleTile->size() < 2)
      return std::nullopt;

    // Validate scale_a tile: [batch..., M_tile, K_scale]
    // M dimension (second-to-last) must match A's M dimension
    int64_t scaleRank = aScaleTile->size();
    int64_t aBatchRank = aTile.size() - 2;
    if ((*aScaleTile)[scaleRank - 2] != aTile[aBatchRank])
      return std::nullopt;

    // Return the K scale factor (last dim)
    return aScaleTile->back();
  };

  // Helper lambda to validate scale B tile for DpasMxOp
  auto validateScaleBTile =
      [&](Operation *op, unsigned scaleBOperandIdx,
          const SmallVector<int64_t> &bTile) -> std::optional<int64_t> {
    std::optional<SmallVector<int64_t>> bScaleTile =
        getTileShape(op->getOpOperand(scaleBOperandIdx));

    if (!bScaleTile || bScaleTile->size() < 2)
      return std::nullopt;

    // Validate scale_b tile: [batch..., K_scale, N_tile]
    // N dimension (last) must match B's N dimension (last)
    if (bScaleTile->back() != bTile.back())
      return std::nullopt;

    // Return the K scale factor (second-to-last dim)
    int64_t scaleRank = bScaleTile->size();
    return (*bScaleTile)[scaleRank - 2];
  };

  if (isa<xegpu::DpasOp>(op)) {
    auto abTiles = validateABTiles(op);
    if (!abTiles)
      return std::nullopt;

    auto [aTile, bTile] = *abTiles;

    // Semantic check for C.
    if (!validateCTile(op, 2, aTile, bTile))
      return std::nullopt;

    // Return [batch..., M, K, N] as the target shape for unrolling.
    int64_t aBatchRank = aTile.size() - 2;
    SmallVector<int64_t> tileShape(aTile.begin(), aTile.begin() + aBatchRank);
    tileShape.push_back(aTile[aBatchRank]);     // M
    tileShape.push_back(aTile[aBatchRank + 1]); // K
    tileShape.push_back(bTile.back());          // N
    return tileShape;
  }

  if (auto dpasMxOp = dyn_cast<xegpu::DpasMxOp>(op)) {
    auto abTiles = validateABTiles(op);
    if (!abTiles)
      return std::nullopt;

    auto [aTile, bTile] = *abTiles;

    // Validate C tile if present using op-specific accessor
    if (dpasMxOp.getAcc()) {
      unsigned accOperandIdx = 2; // acc is the 3rd operand
      if (!validateCTile(op, accOperandIdx, aTile, bTile))
        return std::nullopt;
    }

    // Validate scale tiles if present using op-specific accessors
    int64_t kScaleFactor = 1;
    std::optional<int64_t> scaleAFactor;
    std::optional<int64_t> scaleBFactor;

    if (dpasMxOp.getScaleA()) {
      unsigned scaleAOperandIdx = 2 + (dpasMxOp.getAcc() ? 1 : 0);
      scaleAFactor = validateScaleATile(op, scaleAOperandIdx, aTile);
      if (!scaleAFactor)
        return std::nullopt;
    }

    if (dpasMxOp.getScaleB()) {
      unsigned scaleBOperandIdx =
          2 + (dpasMxOp.getAcc() ? 1 : 0) + (dpasMxOp.getScaleA() ? 1 : 0);
      scaleBFactor = validateScaleBTile(op, scaleBOperandIdx, bTile);
      if (!scaleBFactor)
        return std::nullopt;
    }

    // If both scales are present, their K dimensions must match
    if (scaleAFactor && scaleBFactor) {
      if (*scaleAFactor != *scaleBFactor)
        return std::nullopt;
      kScaleFactor = *scaleAFactor;
    } else if (scaleAFactor) {
      kScaleFactor = *scaleAFactor;
    } else if (scaleBFactor) {
      kScaleFactor = *scaleBFactor;
    }

    // Return [batch..., M, K, N, S] as the target shape for unrolling.
    int64_t aBatchRank = aTile.size() - 2;
    SmallVector<int64_t> tileShape(aTile.begin(), aTile.begin() + aBatchRank);
    tileShape.push_back(aTile[aBatchRank]);     // M
    tileShape.push_back(aTile[aBatchRank + 1]); // K
    tileShape.push_back(bTile.back());          // N
    tileShape.push_back(kScaleFactor);          // S
    return tileShape;
  }

  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)
    return getTileShape(op->getOpResult(0));

  if (isa<vector::MultiDimReductionOp>(op))
    return getTileShape(op->getOpOperand(0));

  if (isa<vector::TransposeOp, vector::BroadcastOp, vector::StepOp,
          vector::ShapeCastOp, vector::ConstantMaskOp, vector::CreateMaskOp,
          vector::BitCastOp, vector::InterleaveOp, vector::DeinterleaveOp>(op))
    return getTileShape(op->getOpResult(0));

  return std::nullopt;
}

bool XeGPUBlockingPass::needsUnroll(Operation *op) const {
  // skip the op if any of its operands or results has workgroup level layouts
  bool hasWgLayoutOperands =
      llvm::any_of(op->getOpOperands(), [](OpOperand &opr) {
        xegpu::DistributeLayoutAttr layout =
            xegpu::getDistributeLayoutAttr(opr);
        return layout && layout.isForWorkgroup();
      });
  bool hasWgLayoutResults =
      llvm::any_of(op->getOpResults(), [](OpResult result) {
        xegpu::DistributeLayoutAttr layout =
            xegpu::getDistributeLayoutAttr(result);
        return layout && layout.isForWorkgroup();
      });
  if (hasWgLayoutOperands || hasWgLayoutResults) {
    LDBG() << "skip unrolling for op with workgroup level layout: " << *op;
    return false;
  }

  auto isUnrollable = [](Value value, ArrayRef<int64_t> tileShape) {
    Type valTy = value.getType();
    if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(valTy)) {
      xegpu::DistributeLayoutAttr layout = tdescTy.getLayoutAttr();
      return layout && !layout.getEffectiveInstDataAsInt().empty();
    }
    auto shapedType = dyn_cast<ShapedType>(valTy);
    return shapedType && !llvm::equal(tileShape, shapedType.getShape());
  };

  bool hasUnrollableOperands =
      llvm::any_of(op->getOpOperands(), [&](OpOperand &opr) {
        std::optional<SmallVector<int64_t>> tileShape = getTileShape(opr);
        return tileShape.has_value() && isUnrollable(opr.get(), *tileShape);
      });
  bool hasUnrollableResults =
      llvm::any_of(op->getOpResults(), [&](OpResult result) {
        std::optional<SmallVector<int64_t>> tileShape = getTileShape(result);
        return tileShape.has_value() && isUnrollable(result, *tileShape);
      });
  // ConvertLayoutOp must be processed to drop the inst_data in the layout
  bool isConvertLayoutWithInstData = false;
  if (auto convertLayoutOp = dyn_cast<xegpu::ConvertLayoutOp>(op)) {
    auto targettLayout = convertLayoutOp.getTargetLayout();
    if (targettLayout && !targettLayout.getEffectiveInstDataAsInt().empty()) {
      isConvertLayoutWithInstData = true;
    }
  }
  return hasUnrollableOperands || hasUnrollableResults ||
         isConvertLayoutWithInstData;
}

void XeGPUBlockingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *op = getOperation();

  if (!xegpu::recoverTemporaryLayouts(op)) {
    signalPassFailure();
    return;
  }

  auto getTileShapeAndCount = [](llvm::ArrayRef<int64_t> shape,
                                 xegpu::DistributeLayoutAttr layout) {
    int count = 1;
    SmallVector<int64_t> tileShape(shape);
    if (layout && !layout.getEffectiveInstDataAsInt().empty()) {
      tileShape = layout.getEffectiveInstDataAsInt();
      count = computeProduct(shape) / computeProduct(tileShape);
    }
    assert(count >= 1 && "count must be at least 1");
    return std::make_pair(tileShape, count);
  };

  // Perform context-aware type conversion for SCF structural ops.
  // Inspects Values to find inst_data layout information for 1:N conversion.
  llvm::SmallSetVector<UnrealizedConversionCastOp, 8> existingCasts;
  op->walk(
      [&](UnrealizedConversionCastOp castOp) { existingCasts.insert(castOp); });

  {
    TypeConverter converter;
    converter.addConversion([](Type type) -> Type { return type; });

    // TensorDescType 1:N converter (type-based, layout is in the type).
    converter.addConversion(
        [&](xegpu::TensorDescType type,
            SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
          Type elemTy = type.getElementType();
          ArrayRef<int64_t> shape = type.getShape();

          xegpu::DistributeLayoutAttr layout = type.getLayoutAttr();
          if (layout && layout.isForWorkgroup())
            return failure();

          int count;
          SmallVector<int64_t> subShape;
          std::tie(subShape, count) = getTileShapeAndCount(shape, layout);

          if (layout)
            layout = layout.dropInstData();

          auto newTy = xegpu::TensorDescType::get(
              type.getContext(), subShape, elemTy, type.getEncoding(), layout);
          result.append(count, newTy);
          return success();
        });

    // Context-aware VectorType conversion based on inst_data (1:1
    // shape-changing or 1:N).
    auto getSubShapeAndCount = [&](VectorType vecTy,
                                   xegpu::DistributeLayoutAttr layout)
        -> std::pair<SmallVector<int64_t>, int> {
      return getTileShapeAndCount(vecTy.getShape(), layout);
    };
    auto loopArgTypes =
        xegpu::precomputeLoopBlockArgTypes(op, getSubShapeAndCount);
    xegpu::addVectorTypeConversion(converter, getSubShapeAndCount,
                                   std::move(loopArgTypes));

    // Loop-carried types are now in the converter's map, so the transient
    // per-position layout attrs on SCF loop ops are no longer needed. Strip
    // them before converting: the SCF converters copy old attrs onto the new
    // op (ConvertForOpTypes::setAttrs), and after 1:N result expansion a stale
    // `layout_result_N` lands on the wrong (renumbered) result, corrupting the
    // count invariant and leaving the loop illegal.
    op->walk([](Operation *loopOp) {
      if (!isa<scf::ForOp, scf::WhileOp, scf::ConditionOp, scf::IfOp>(loopOp))
        return;
      SmallVector<StringRef> toRemove;
      for (const NamedAttribute &attr : loopOp->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name.starts_with("layout_operand_") ||
            name.starts_with("layout_result_"))
          toRemove.push_back(name);
      }
      for (StringRef name : toRemove)
        loopOp->removeAttr(name);
    });

    // Source (N:1) and target (1:1) materializations using
    // UnrealizedConversionCastOp.
    auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(materializeCast);
    converter.addTargetMaterialization(materializeCast);
    // Blocking runs SCF conversion separately (not combined with XeGPU
    // patterns), so it also needs a 1:N target materialization.
    converter.addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::TypeRange types,
           mlir::ValueRange inputs, mlir::Location loc) -> SmallVector<Value> {
          auto castOp =
              UnrealizedConversionCastOp::create(builder, loc, types, inputs);
          return SmallVector<Value>(castOp.getResults());
        });

    ConversionTarget target(*ctx);
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet scfPatterns(ctx);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, scfPatterns,
                                                         target);
    if (failed(applyPartialConversion(op, target, std::move(scfPatterns))))
      return signalPassFailure();

    // Fold cancelling cast chains and erase dead casts.
    xegpu::cleanupUnrealizedConversionCasts(op, existingCasts);
  }

  xegpu::UnrollOptions options;
  options.setFilterConstraint(
      [&](Operation *op) -> LogicalResult { return success(needsUnroll(op)); });

  options.setNativeShapeFn([&](Operation *op) { return getTileShape(op); });

  options.setUnrolledTypesFn([&](ShapedType type, ArrayRef<int64_t> tileShape) {
    Type elemTy = type.getElementType();

    if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type)) {

      Attribute encoding = tdescTy.getEncoding();

      xegpu::TensorDescType newTy =
          xegpu::TensorDescType::get(ctx, tileShape, elemTy, encoding,
                                     tdescTy.getLayoutAttr().dropInstData());
      // Compute the product of batch (higher) dimensions.
      ArrayRef<int64_t> shape = type.getShape();
      int64_t batchCount =
          shape.size() > 2 ? computeProduct(shape.drop_back(2)) : 1;
      return SmallVector<Type>(batchCount, newTy);
    }
    Type newTy = VectorType::get(tileShape, elemTy);

    std::optional<SmallVector<int64_t>> ratio =
        computeShapeRatio(type.getShape(), tileShape);
    assert(ratio && "The shape of the type must be a multiple of tileShape.");
    return SmallVector<Type>(computeProduct(*ratio), newTy);
  });

  RewritePatternSet patterns(ctx);
  vector::UnrollVectorOptions vectorOptions;
  vectorOptions.setNativeShapeFn(options.nativeShape);

  populateXeGPUUnrollPatterns(patterns, options);
  vector::populateVectorUnrollPatterns(patterns, vectorOptions);

  // Note: The pattern driver does op folding as well and clean up.
  // But intermediate insert/extract strided slice ops with
  // unrealized conversion cast ops in the middle does not get
  // cleaned up in this step. One more round of folding is needed
  // after the walk to resolve those unrealized conversion cast ops.
  (void)applyPatternsGreedily(op, std::move(patterns));

  op->walk([](Operation *op) {
    // Remove the layout attributes cached per operands.
    for (OpOperand &opr : op->getOpOperands()) {
      std::string name = xegpu::getTemporaryLayoutName(opr);
      if (op->hasAttrOfType<xegpu::DistributeLayoutAttr>(name))
        op->removeAttr(name);
    }

    // Update the layout attributes per result.
    for (OpResult result : op->getOpResults()) {
      std::string name = xegpu::getTemporaryLayoutName(result);
      if (auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(name)) {
        op->removeAttr(name);
        if (!isa<LoopLikeOpInterface>(op))
          xegpu::setDistributeLayoutAttr(result, layout.dropInstData());
      }
    }

    // Drop left-over inst_data if the unroll pattern does not being applied,
    // say, inst_data just matches their shape.
    SmallVector<NamedAttribute> newAttrs =
        xegpu::dropInstDataOnAttrs(op->getAttrs());
    op->setAttrs(newAttrs);
  });

  // Resolve UnrealizedConversionCastOps generated by SCF structural type
  // conversion and by XeGPU/Vector unrolling (cancelling cast chains and
  // unpaired pack/unpack casts).
  xegpu::cleanupUnrealizedConversionCasts(op, existingCasts);

  // One more round of folding to clean up the intermediate
  // insert/extract strided slice ops.
  RewritePatternSet emptyPatterns(ctx);
  (void)applyPatternsGreedily(op, std::move(emptyPatterns));
}
