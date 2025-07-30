//===- ShapeToStandard.cpp - conversion from Shape to Standard dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSHAPETOSTANDARDPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::shape;
using namespace mlir::scf;

/// Conversion patterns.
namespace {
class AnyOpConversion : public OpConversionPattern<AnyOp> {
public:
  using OpConversionPattern<AnyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AnyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
AnyOpConversion::matchAndRewrite(AnyOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  // Replace `any` with its first operand.
  // Any operand would be a valid substitution.
  rewriter.replaceOp(op, {adaptor.getInputs().front()});
  return success();
}

namespace {
template <typename SrcOpTy, typename DstOpTy>
class BinaryOpConversion : public OpConversionPattern<SrcOpTy> {
public:
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now, only error-free types are supported by this lowering.
    if (isa<SizeType>(op.getType()))
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(op, adaptor.getLhs(),
                                         adaptor.getRhs());
    return success();
  }
};
} // namespace

namespace {
struct BroadcastOpConverter : public OpConversionPattern<BroadcastOp> {
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Get the resulting extent in a given dimension. This is computed with any
// number of extent tensors and shifted offsets into them.
Value getBroadcastedDim(ImplicitLocOpBuilder lb, ValueRange extentTensors,
                        ValueRange rankDiffs, Value outputDimension) {
  Value one = arith::ConstantIndexOp::create(lb, 1);
  Value broadcastedDim = one;
  for (auto tup : llvm::zip(extentTensors, rankDiffs)) {
    Value shape = std::get<0>(tup);
    Value rankDiff = std::get<1>(tup);
    Value outOfBounds = arith::CmpIOp::create(lb, arith::CmpIPredicate::ult,
                                              outputDimension, rankDiff);
    Type indexTy = lb.getIndexType();
    broadcastedDim =
        IfOp::create(
            lb, outOfBounds,
            [&](OpBuilder &b, Location loc) {
              scf::YieldOp::create(b, loc, broadcastedDim);
            },
            [&](OpBuilder &b, Location loc) {
              // The broadcasting logic is:
              // - if one extent (here we arbitrarily choose the
              // extent from the greater-rank operand) is equal to 1,
              // then take the extent from the other operand
              // - otherwise, take the extent as-is.
              // Note that this logic remains correct in the presence
              // of dimensions of zero extent.
              Value lesserRankOperandDimension = arith::SubIOp::create(
                  b, loc, indexTy, outputDimension, rankDiff);
              Value lesserRankOperandExtent = tensor::ExtractOp::create(
                  b, loc, shape, ValueRange{lesserRankOperandDimension});

              Value dimIsOne =
                  arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                        lesserRankOperandExtent, one);
              Value dim = arith::SelectOp::create(
                  b, loc, dimIsOne, broadcastedDim, lesserRankOperandExtent);
              scf::YieldOp::create(b, loc, dim);
            })
            .getResult(0);
  }
  return broadcastedDim;
}
} // namespace

LogicalResult BroadcastOpConverter::matchAndRewrite(
    BroadcastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  if (isa<ShapeType>(op.getType()))
    return failure();

  auto loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);

  Value zero = arith::ConstantIndexOp::create(lb, 0);
  Type indexTy = lb.getIndexType();

  // Save all the ranks for bounds checking. Because this is a tensor
  // representing the shape extents, the rank is the extent of the only
  // dimension in the tensor.
  SmallVector<Value> ranks, rankDiffs;
  llvm::append_range(ranks, llvm::map_range(adaptor.getShapes(), [&](Value v) {
                       return tensor::DimOp::create(lb, v, zero);
                     }));

  // Find the maximum rank
  Value maxRank = ranks.front();
  for (Value v : llvm::drop_begin(ranks, 1)) {
    maxRank = arith::MaxUIOp::create(lb, v, maxRank);
  }

  // Calculate the difference of ranks and the maximum rank for later offsets.
  llvm::append_range(rankDiffs, llvm::map_range(ranks, [&](Value v) {
                       return arith::SubIOp::create(lb, indexTy, maxRank, v);
                     }));

  Value replacement = tensor::GenerateOp::create(
      lb, getExtentTensorType(lb.getContext()), ValueRange{maxRank},
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value broadcastedDim =
            getBroadcastedDim(ImplicitLocOpBuilder(loc, b), adaptor.getShapes(),
                              rankDiffs, args[0]);

        tensor::YieldOp::create(b, loc, broadcastedDim);
      });
  if (replacement.getType() != op.getType())
    replacement = tensor::CastOp::create(lb, op.getType(), replacement);
  rewriter.replaceOp(op, replacement);
  return success();
}

namespace {
class ConstShapeOpConverter : public OpConversionPattern<ConstShapeOp> {
public:
  using OpConversionPattern<ConstShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ConstShapeOpConverter::matchAndRewrite(
    ConstShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // For now, this lowering supports only extent tensors, not `shape.shape`
  // types.
  if (isa<ShapeType>(op.getType()))
    return failure();

  auto loc = op.getLoc();
  SmallVector<Value, 4> extentOperands;
  for (auto extent : op.getShape()) {
    extentOperands.push_back(arith::ConstantIndexOp::create(
        rewriter, loc, extent.getLimitedValue()));
  }
  Type resultTy =
      RankedTensorType::get({op.getShape().size()}, rewriter.getIndexType());
  Value tensor =
      tensor::FromElementsOp::create(rewriter, loc, resultTy, extentOperands);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultTy, tensor);
  return success();
}

namespace {
class ConstSizeOpConversion : public OpConversionPattern<ConstSizeOp> {
public:
  using OpConversionPattern<ConstSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ConstSizeOpConversion::matchAndRewrite(
    ConstSizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
      op, op.getValue().getSExtValue());
  return success();
}

namespace {
struct IsBroadcastableOpConverter
    : public OpConversionPattern<IsBroadcastableOp> {
  using OpConversionPattern<IsBroadcastableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsBroadcastableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult IsBroadcastableOpConverter::matchAndRewrite(
    IsBroadcastableOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  if (!llvm::all_of(op.getShapes(),
                    [](Value v) { return !isa<ShapeType>(v.getType()); }))
    return failure();

  auto loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);
  Value zero = arith::ConstantIndexOp::create(lb, 0);
  Value one = arith::ConstantIndexOp::create(lb, 1);
  Type indexTy = lb.getIndexType();

  // Save all the ranks for bounds checking. Because this is a tensor
  // representing the shape extents, the rank is the extent of the only
  // dimension in the tensor.
  SmallVector<Value> ranks, rankDiffs;
  llvm::append_range(ranks, llvm::map_range(adaptor.getShapes(), [&](Value v) {
                       return tensor::DimOp::create(lb, v, zero);
                     }));

  // Find the maximum rank
  Value maxRank = ranks.front();
  for (Value v : llvm::drop_begin(ranks, 1)) {
    maxRank = arith::MaxUIOp::create(lb, v, maxRank);
  }

  // Calculate the difference of ranks and the maximum rank for later offsets.
  llvm::append_range(rankDiffs, llvm::map_range(ranks, [&](Value v) {
                       return arith::SubIOp::create(lb, indexTy, maxRank, v);
                     }));

  Type i1Ty = rewriter.getI1Type();
  Value trueVal = arith::ConstantOp::create(rewriter, loc, i1Ty,
                                            rewriter.getBoolAttr(true));

  auto reduceResult = ForOp::create(
      lb, loc, zero, maxRank, one, ValueRange{trueVal},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        // Find a non-1 dim, if it exists. Note that the first part of this
        // could reuse the Broadcast lowering entirely, but we redo the work
        // here to make optimizations easier between the two loops.
        Value broadcastedDim = getBroadcastedDim(
            ImplicitLocOpBuilder(loc, b), adaptor.getShapes(), rankDiffs, iv);

        Value broadcastable = iterArgs[0];
        for (auto tup : llvm::zip(adaptor.getShapes(), rankDiffs)) {
          Value shape, rankDiff;
          std::tie(shape, rankDiff) = tup;
          Value outOfBounds = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ult, iv, rankDiff);
          broadcastable =
              IfOp::create(
                  b, loc, outOfBounds,
                  [&](OpBuilder &b, Location loc) {
                    // Non existent dimensions are always broadcastable
                    scf::YieldOp::create(b, loc, broadcastable);
                  },
                  [&](OpBuilder &b, Location loc) {
                    // Every value needs to be either 1, or the same non-1
                    // value to be broadcastable in this dim.
                    Value operandDimension =
                        arith::SubIOp::create(b, loc, indexTy, iv, rankDiff);
                    Value dimensionExtent = tensor::ExtractOp::create(
                        b, loc, shape, ValueRange{operandDimension});

                    Value equalOne = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::eq, dimensionExtent, one);
                    Value equalBroadcasted =
                        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                              dimensionExtent, broadcastedDim);
                    Value result = arith::AndIOp::create(
                        b, loc, broadcastable,
                        arith::OrIOp::create(b, loc, equalOne,
                                             equalBroadcasted));
                    scf::YieldOp::create(b, loc, result);
                  })
                  .getResult(0);
        }

        scf::YieldOp::create(b, loc, broadcastable);
      });

  rewriter.replaceOp(op, reduceResult.getResults().front());
  return success();
}

namespace {
class DimOpConverter : public OpConversionPattern<DimOp> {
  using OpConversionPattern<DimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
DimOpConverter::matchAndRewrite(DimOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // Lower to dim(X, i) to get_extent(shape_of(X), i) and rely on further
  // lowerings. This can be further optimized if needed to avoid intermediate
  // steps.
  auto shapeOf = shape::ShapeOfOp::create(rewriter, op.getLoc(), op.getValue());
  rewriter.replaceOpWithNewOp<shape::GetExtentOp>(op, op.getType(), shapeOf,
                                                  op.getIndex());
  return success();
}

namespace {
class GetExtentOpConverter : public OpConversionPattern<GetExtentOp> {
  using OpConversionPattern<GetExtentOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetExtentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult GetExtentOpConverter::matchAndRewrite(
    GetExtentOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // For now, only error-free types are supported by this lowering.
  if (isa<SizeType>(op.getType()))
    return failure();

  // Derive shape extent directly from shape origin if possible. This
  // circumvents the necessity to materialize the shape in memory.
  if (auto shapeOfOp = op.getShape().getDefiningOp<ShapeOfOp>()) {
    if (isa<ShapedType>(shapeOfOp.getArg().getType())) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, shapeOfOp.getArg(),
                                                 adaptor.getDim());
      return success();
    }
  }

  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, rewriter.getIndexType(),
                                                 adaptor.getShape(),
                                                 ValueRange{adaptor.getDim()});
  return success();
}

namespace {
class RankOpConverter : public OpConversionPattern<shape::RankOp> {
public:
  using OpConversionPattern<shape::RankOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::RankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
RankOpConverter::matchAndRewrite(shape::RankOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  // For now, this lowering supports only error-free types.
  if (isa<SizeType>(op.getType()))
    return failure();

  rewriter.replaceOpWithNewOp<tensor::DimOp>(op, adaptor.getShape(), 0);
  return success();
}

namespace {
/// Converts `shape.reduce` to `scf.for`.
struct ReduceOpConverter : public OpConversionPattern<shape::ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
ReduceOpConverter::matchAndRewrite(shape::ReduceOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands.
  if (isa<ShapeType>(op.getShape().getType()))
    return failure();

  auto loc = op.getLoc();

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Type indexTy = rewriter.getIndexType();
  Value rank =
      tensor::DimOp::create(rewriter, loc, indexTy, adaptor.getShape(), zero);

  auto loop = scf::ForOp::create(
      rewriter, loc, zero, rank, one, op.getInitVals(),
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value extent =
            tensor::ExtractOp::create(b, loc, adaptor.getShape(), iv);

        SmallVector<Value, 2> mappedValues{iv, extent};
        mappedValues.append(args.begin(), args.end());

        IRMapping mapping;
        Block *reduceBody = op.getBody();
        mapping.map(reduceBody->getArguments(), mappedValues);
        for (auto &nested : reduceBody->without_terminator())
          b.clone(nested, mapping);

        SmallVector<Value, 2> mappedResults;
        for (auto result : reduceBody->getTerminator()->getOperands())
          mappedResults.push_back(mapping.lookup(result));
        scf::YieldOp::create(b, loc, mappedResults);
      });

  rewriter.replaceOp(op, loop.getResults());
  return success();
}

namespace {
/// Converts `shape.shape_eq` to an `scf.for` loop. For now, the lowering is
/// only defined on `tensor<?xindex>` operands. The test for equality first
/// compares their size and, if equal, checks every extent for equality.
///
/// Example:
///
/// %result = shape.shape_eq %a, %b : tensor<?xindex>, tensor<?xindex>
///
/// becomes
///
/// %c0 = arith.constant 0 : index
/// %0 = dim %arg0, %c0 : tensor<?xindex>
/// %1 = dim %arg1, %c0 : tensor<?xindex>
/// %2 = arith.cmpi "eq", %0, %1 : index
/// %result = scf.if %2 -> (i1) {
///   %c1 = arith.constant 1 : index
///   %true = arith.constant true
///   %4 = scf.for %arg2 = %c0 to %0 step %c1 iter_args(%arg3 = %true) -> (i1) {
///     %5 = tensor.extract %arg0[%arg2] : tensor<?xindex>
///     %6 = tensor.extract %arg1[%arg2] : tensor<?xindex>
///     %7 = arith.cmpi "eq", %5, %6 : index
///     %8 = arith.andi %arg3, %7 : i1
///     scf.yield %8 : i1
///   }
///   scf.yield %4 : i1
/// } else {
///   %false = arith.constant false
///   scf.yield %false : i1
/// }
///
struct ShapeEqOpConverter : public OpConversionPattern<ShapeEqOp> {
  using OpConversionPattern<ShapeEqOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeEqOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ShapeEqOpConverter::matchAndRewrite(ShapeEqOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  if (!llvm::all_of(op.getShapes(),
                    [](Value v) { return !isa<ShapeType>(v.getType()); }))
    return failure();

  Type i1Ty = rewriter.getI1Type();
  if (op.getShapes().size() <= 1) {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, i1Ty,
                                                   rewriter.getBoolAttr(true));
    return success();
  }

  auto loc = op.getLoc();
  Type indexTy = rewriter.getIndexType();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value firstShape = adaptor.getShapes().front();
  Value firstRank =
      tensor::DimOp::create(rewriter, loc, indexTy, firstShape, zero);
  Value result = nullptr;
  // Generate a linear sequence of compares, all with firstShape as lhs.
  for (Value shape : adaptor.getShapes().drop_front(1)) {
    Value rank = tensor::DimOp::create(rewriter, loc, indexTy, shape, zero);
    Value eqRank = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, firstRank, rank);
    auto same = IfOp::create(
        rewriter, loc, eqRank,
        [&](OpBuilder &b, Location loc) {
          Value one = arith::ConstantIndexOp::create(b, loc, 1);
          Value init =
              arith::ConstantOp::create(b, loc, i1Ty, b.getBoolAttr(true));
          auto loop = scf::ForOp::create(
              b, loc, zero, firstRank, one, ValueRange{init},
              [&](OpBuilder &b, Location nestedLoc, Value iv, ValueRange args) {
                Value conj = args[0];
                Value lhsExtent =
                    tensor::ExtractOp::create(b, loc, firstShape, iv);
                Value rhsExtent = tensor::ExtractOp::create(b, loc, shape, iv);
                Value eqExtent = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::eq, lhsExtent, rhsExtent);
                Value conjNext = arith::AndIOp::create(b, loc, conj, eqExtent);
                scf::YieldOp::create(b, loc, ValueRange({conjNext}));
              });
          scf::YieldOp::create(b, loc, loop.getResults());
        },
        [&](OpBuilder &b, Location loc) {
          Value result =
              arith::ConstantOp::create(b, loc, i1Ty, b.getBoolAttr(false));
          scf::YieldOp::create(b, loc, result);
        });
    result = !result ? same.getResult(0)
                     : arith::AndIOp::create(rewriter, loc, result,
                                             same.getResult(0));
  }
  rewriter.replaceOp(op, result);
  return success();
}

namespace {
class ShapeOfOpConversion : public OpConversionPattern<ShapeOfOp> {
public:
  using OpConversionPattern<ShapeOfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ShapeOfOpConversion::matchAndRewrite(
    ShapeOfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // For now, only error-free types are supported by this lowering.
  if (isa<ShapeType>(op.getType()))
    return failure();

  // For ranked tensor arguments, lower to `tensor.from_elements`.
  auto loc = op.getLoc();
  Value tensor = adaptor.getArg();
  Type tensorTy = tensor.getType();
  if (isa<RankedTensorType>(tensorTy)) {

    // Build values for individual extents.
    SmallVector<Value, 8> extentValues;
    RankedTensorType rankedTensorTy = cast<RankedTensorType>(tensorTy);
    int64_t rank = rankedTensorTy.getRank();
    for (int64_t i = 0; i < rank; i++) {
      if (rankedTensorTy.isDynamicDim(i)) {
        Value extent = tensor::DimOp::create(rewriter, loc, tensor, i);
        extentValues.push_back(extent);
      } else {
        Value extent = arith::ConstantIndexOp::create(
            rewriter, loc, rankedTensorTy.getDimSize(i));
        extentValues.push_back(extent);
      }
    }

    // Materialize extent tensor.
    Value staticExtentTensor = tensor::FromElementsOp::create(
        rewriter, loc, RankedTensorType::get({rank}, rewriter.getIndexType()),
        extentValues);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                staticExtentTensor);
    return success();
  }

  // Lower to `tensor.generate` otherwise.
  auto *ctx = rewriter.getContext();
  Value rank = tensor::RankOp::create(rewriter, loc, tensor);
  rewriter.replaceOpWithNewOp<tensor::GenerateOp>(
      op, getExtentTensorType(ctx), ValueRange{rank},
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value dim = args.front();
        Value extent = tensor::DimOp::create(b, loc, tensor, dim);
        tensor::YieldOp::create(b, loc, extent);
      });

  return success();
}

namespace {
class SplitAtOpConversion : public OpConversionPattern<SplitAtOp> {
public:
  using OpConversionPattern<SplitAtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplitAtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult SplitAtOpConversion::matchAndRewrite(
    SplitAtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Error conditions are not implemented, only lower if all operands and
  // results are extent tensors.
  if (llvm::any_of(ValueRange{op.getOperand(), op.getHead(), op.getTail()},
                   [](Value v) { return isa<ShapeType>(v.getType()); }))
    return failure();

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Value zero = arith::ConstantIndexOp::create(b, 0);
  Value rank = tensor::DimOp::create(b, adaptor.getOperand(), zero);

  // index < 0 ? index + rank : index
  Value originalIndex = adaptor.getIndex();
  Value add = arith::AddIOp::create(b, originalIndex, rank);
  Value indexIsNegative =
      arith::CmpIOp::create(b, arith::CmpIPredicate::slt, originalIndex, zero);
  Value index = arith::SelectOp::create(b, indexIsNegative, add, originalIndex);

  Value one = arith::ConstantIndexOp::create(b, 1);
  Value head =
      tensor::ExtractSliceOp::create(b, adaptor.getOperand(), zero, index, one);
  Value tailSize = arith::SubIOp::create(b, rank, index);
  Value tail = tensor::ExtractSliceOp::create(b, adaptor.getOperand(), index,
                                              tailSize, one);
  rewriter.replaceOp(op, {head, tail});
  return success();
}

namespace {
class ToExtentTensorOpConversion
    : public OpConversionPattern<ToExtentTensorOp> {
public:
  using OpConversionPattern<ToExtentTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToExtentTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(adaptor.getInput().getType()))
      return rewriter.notifyMatchFailure(op, "input needs to be a tensor");

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                adaptor.getInput());
    return success();
  }
};
} // namespace

namespace {
/// Import the Shape Ops to Std Patterns.
#include "ShapeToStandard.cpp.inc"
} // namespace

namespace {
/// Conversion pass.
class ConvertShapeToStandardPass
    : public impl::ConvertShapeToStandardPassBase<ConvertShapeToStandardPass> {

  void runOnOperation() override;
};
} // namespace

void ConvertShapeToStandardPass::runOnOperation() {
  // Setup target legality.
  MLIRContext &ctx = getContext();
  ConversionTarget target(ctx);
  target.addLegalDialect<arith::ArithDialect, SCFDialect,
                         tensor::TensorDialect>();
  target.addLegalOp<CstrRequireOp, func::FuncOp, ModuleOp>();

  // Setup conversion patterns.
  RewritePatternSet patterns(&ctx);
  populateShapeToStandardConversionPatterns(patterns);

  // Apply conversion.
  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void mlir::populateShapeToStandardConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  populateWithGenerated(patterns);
  patterns.add<
      AnyOpConversion,
      BinaryOpConversion<AddOp, arith::AddIOp>,
      BinaryOpConversion<MulOp, arith::MulIOp>,
      BroadcastOpConverter,
      ConstShapeOpConverter,
      ConstSizeOpConversion,
      DimOpConverter,
      IsBroadcastableOpConverter,
      GetExtentOpConverter,
      RankOpConverter,
      ReduceOpConverter,
      ShapeEqOpConverter,
      ShapeOfOpConversion,
      SplitAtOpConversion,
      ToExtentTensorOpConversion>(patterns.getContext());
  // clang-format on
}
