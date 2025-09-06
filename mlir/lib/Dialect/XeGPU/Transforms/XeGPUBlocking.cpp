//===---- XeGPUBlocking.cpp ---- XeGPU Blocking Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
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

// reslove the unrealized conversion cast ops generated when doing SCF
// Structural Type Conversion. It will have two formats, N:1 vector
// cast and 1:N vector cast. vector::insert_strided_slice ops will be
// used for the first case, and vector::extract_strided_slice ops will be
// used for the second case.
static void
resolveUnrealizedConversionCastOp(UnrealizedConversionCastOp castOp) {
  ValueRange inputs = castOp.getInputs();
  ValueRange outputs = castOp.getOutputs();

  auto hasIdenticalVectorTypes = [](ValueRange values) {
    auto types = values.getTypes();
    return llvm::all_of(types, [&](Type type) {
      return isa<VectorType>(type) && type == types.front();
    });
  };

  // We only interest in the case where all inputs and outputs have the
  // identical VectorTypes
  if (!hasIdenticalVectorTypes(inputs) || !hasIdenticalVectorTypes(outputs)) {
    LDBG() << "skip unrealized conversion cast op not emulating pack/unpack.";
    return;
  }

  VectorType outputTy = dyn_cast<VectorType>(outputs[0].getType());
  OpBuilder builder(castOp);
  if (inputs.size() > 1 && outputs.size() == 1) {
    // the castOp is emulating an unpack op
    ArrayRef<int64_t> shape = outputTy.getShape();
    Value result = xegpu::createVectorWithShapeFromValues(
        builder, castOp.getLoc(), inputs, shape);
    castOp->replaceAllUsesWith(ValueRange(result));
    castOp->erase();
  } else if (castOp.getNumResults() > 1 && castOp.getNumOperands() == 1) {
    // the castOp is emulating a pack op
    ArrayRef<int64_t> tileShape = outputTy.getShape();
    SmallVector<Value> results = xegpu::extractVectorsWithShapeFromValue(
        builder, castOp.getLoc(), inputs[0], tileShape);
    castOp->replaceAllUsesWith(results);
    castOp->erase();
  }
}

// This pattern lowers ConvertLayoutOp by removing the inst_data field from the
// layout attributes. Since both producer and consumer operations handle data
// partitioning based on their own inst_data, while maintaining original input
// and output shape, ConvertLayoutOp does not need to manage inst_data.
struct ConvertLayoutOpPattern
    : public OpRewritePattern<xegpu::ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(xegpu::ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr input_layout = op.getInputLayoutAttr();
    xegpu::DistributeLayoutAttr target_layout = op.getTargetLayoutAttr();
    if (input_layout.getInstDataAsInt().empty() ||
        target_layout.getInstDataAsInt().empty())
      return rewriter.notifyMatchFailure(op, "Not a target ConvertLayoutOp.");

    input_layout = input_layout.dropInstData();
    target_layout = target_layout.dropInstData();
    auto newOp = rewriter.createOrFold<xegpu::ConvertLayoutOp>(
        op.getLoc(), op.getType(), op.getSource(), input_layout, target_layout);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

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
  if constexpr (std::is_same_v<T, OpOperand>)
    value = operandOrResult.get();
  else
    value = (Value)operandOrResult;

  xegpu::DistributeLayoutAttr layout =
      xegpu::getDistributeLayoutAttr(operandOrResult);
  if (layout && layout.isForSubgroup()) {
    if (!layout.getInstDataAsInt().empty())
      return layout.getInstDataAsInt();

    if (auto type = dyn_cast<ShapedType>(value.getType()))
      return llvm::to_vector(type.getShape());
  }
  LDBG() << "failed to getTileShape for: " << value;
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(Operation *op) const {
  if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp, xegpu::CreateDescOp,
          xegpu::UpdateOffsetOp, xegpu::LoadMatrixOp>(op))
    return getTileShape(op->getOpResult(0));
  if (isa<xegpu::PrefetchNdOp, xegpu::LoadNdOp, xegpu::PrefetchOp,
          xegpu::LoadGatherOp, xegpu::StoreMatrixOp>(op))
    return getTileShape(op->getOpOperand(0));
  if (isa<xegpu::StoreNdOp, xegpu::StoreScatterOp>(op))
    return getTileShape(op->getOpOperand(1));

  if (isa<xegpu::DpasOp>(op)) {
    std::optional<SmallVector<int64_t>> aTile =
        getTileShape(op->getOpOperand(0));
    std::optional<SmallVector<int64_t>> bTile =
        getTileShape(op->getOpOperand(1));

    if (!aTile || aTile->size() != 2 || !bTile || bTile->size() != 2)
      return std::nullopt;

    // semantic check for A and B
    if ((*aTile)[1] != (*bTile)[0])
      return std::nullopt;

    // semantic check for C
    if (op->getNumOperands() == 3) {
      std::optional<SmallVector<int64_t>> cTile =
          getTileShape(op->getOpOperand(2));
      int64_t expectedCTile[2] = {(*aTile)[0], (*bTile)[1]};
      if (!cTile || !llvm::equal(*cTile, expectedCTile))
        return std::nullopt;
    }

    return SmallVector<int64_t>({(*aTile)[0], (*aTile)[1], (*bTile)[1]});
  }

  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)
    return getTileShape(op->getOpResult(0));

  if (isa<vector::MultiDimReductionOp>(op))
    return getTileShape(op->getOpOperand(0));

  if (isa<vector::TransposeOp, vector::BroadcastOp>(op))
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
      return layout && !layout.getInstDataAsInt().empty();
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
  return hasUnrollableOperands || hasUnrollableResults;
}

void XeGPUBlockingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *op = getOperation();

  // Preserve the LayoutAttr for each operand to the owner's DictionaryAttr.
  // This ensures that the LayoutAttr remains accessible even if the defining
  // operation is replaced.
  xegpu::setDistributeLayoutAttrs(
      op, [](Value v) { return xegpu::getDistributeLayoutAttr(v); });

  auto getTileShapeAndCount = [](llvm::ArrayRef<int64_t> shape,
                                 xegpu::LayoutAttr layout) {
    int count = 1;
    SmallVector<int64_t> tileShape(shape);
    if (layout && layout.getInstData()) {
      DenseI32ArrayAttr instData = layout.getInstData();
      tileShape = llvm::to_vector_of<int64_t>(instData.asArrayRef());
      count = computeProduct(shape) / computeProduct(tileShape);
    }
    return std::make_pair(tileShape, count);
  };

  // Perform type conversion for SCF control folow ops
  TypeConverter converter;
  converter.addConversion([](Type type) -> Type { return type; });
  converter.addConversion(
      [&](RankedTensorType type,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        Type elemTy = type.getElementType();
        ArrayRef<int64_t> shape = type.getShape();

        auto layout =
            llvm::dyn_cast_if_present<xegpu::LayoutAttr>(type.getEncoding());
        if (layout && layout.isForWorkgroup())
          return failure();

        int count;
        SmallVector<int64_t> subShape;
        std::tie(subShape, count) = getTileShapeAndCount(shape, layout);
        auto newTy = VectorType::get(subShape, elemTy);
        result.append(count, newTy);
        return success();
      });
  converter.addConversion(
      [&](xegpu::TensorDescType type,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        Type elemTy = type.getElementType();
        ArrayRef<int64_t> shape = type.getShape();

        xegpu::LayoutAttr layout = type.getLayoutAttr();
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

  xegpu::doSCFStructuralTypeConversionWithTensorType(op, converter);

  xegpu::UnrollOptions options;
  options.setFilterConstraint(
      [&](Operation *op) -> LogicalResult { return success(needsUnroll(op)); });

  options.setNativeShapeFn([&](Operation *op) { return getTileShape(op); });

  options.setUnrolledTypesFn([&](ShapedType type, ArrayRef<int64_t> tileShape) {
    Type elemTy = type.getElementType();
    Type newTy;

    if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type)) {

      Attribute encoding = tdescTy.getEncoding();
      // If the encoding is a ScatterTensorDescAttr, we need to
      // potentially adjust the chunk size based on the inst_data.
      if (tdescTy.isScattered()) {
        int64_t chunkSize = tdescTy.getChunkSizeAsInt();

        if (chunkSize > 1) {
          int64_t blockedChunkSize = chunkSize;
          auto instData = tdescTy.getLayoutAttr().getInstData();
          if (!instData.empty())
            blockedChunkSize = instData.asArrayRef().back();

          // To create a new attribute with a different chunk_size:
          auto newEncoding = xegpu::ScatterTensorDescAttr::get(
              ctx, tdescTy.getMemorySpace(), blockedChunkSize);

          encoding = newEncoding;
        }
      }

      newTy =
          xegpu::TensorDescType::get(ctx, tileShape, elemTy, encoding,
                                     tdescTy.getLayoutAttr().dropInstData());
    } else {
      newTy = type.clone(tileShape, elemTy);
    }

    std::optional<SmallVector<int64_t>> ratio =
        computeShapeRatio(type.getShape(), tileShape);
    assert(ratio && "The shape of the type must be a multiple of tileShape.");
    return SmallVector<Type>(computeProduct(*ratio), newTy);
  });

  RewritePatternSet patterns(ctx);
  patterns.add<ConvertLayoutOpPattern>(ctx);

  vector::UnrollVectorOptions vectorOptions;
  vectorOptions.setNativeShapeFn(options.nativeShape);

  populateXeGPUUnrollPatterns(patterns, options);
  vector::populateVectorUnrollPatterns(patterns, vectorOptions);

  (void)applyPatternsGreedily(op, std::move(patterns));

  op->walk([](Operation *op) {
    // Remove the layout attributes cached per operands.
    for (OpOperand &opr : op->getOpOperands()) {
      std::string name = xegpu::getLayoutName(opr);
      if (op->hasAttrOfType<xegpu::LayoutAttr>(name))
        op->removeAttr(name);
    }

    // Update the layout attributes per result.
    for (OpResult result : op->getOpResults()) {
      std::string name = xegpu::getLayoutName(result);
      if (auto layout = op->getAttrOfType<xegpu::LayoutAttr>(name)) {
        op->removeAttr(name);
        if (!isa<LoopLikeOpInterface>(op))
          xegpu::setDistributeLayoutAttr(result, layout.dropInstData());
      }
    }

    // Resolve unrealized conversion cast ops emulating pack/unpack
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op))
      resolveUnrealizedConversionCastOp(castOp);
  });
}
