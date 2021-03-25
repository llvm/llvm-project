//===- ShapeToStandard.cpp - conversion from Shape to Standard dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::shape;
using namespace mlir::scf;

/// Conversion patterns.
namespace {
class AnyOpConversion : public OpConversionPattern<AnyOp> {
public:
  using OpConversionPattern<AnyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AnyOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
AnyOpConversion::matchAndRewrite(AnyOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  AnyOp::Adaptor transformed(operands);

  // Replace `any` with its first operand.
  // Any operand would be a valid substitution.
  rewriter.replaceOp(op, {transformed.inputs().front()});
  return success();
}

namespace {
template <typename SrcOpTy, typename DstOpTy>
class BinaryOpConversion : public OpConversionPattern<SrcOpTy> {
public:
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor transformed(operands);

    // For now, only error-free types are supported by this lowering.
    if (op.getType().template isa<SizeType>())
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(op, transformed.lhs(),
                                         transformed.rhs());
    return success();
  }
};
} // namespace

namespace {
struct BroadcastOpConverter : public OpConversionPattern<BroadcastOp> {
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

// Get the resulting extent in a given dimension. This is computed with any
// number of extent tensors and shifted offsets into them.
Value getBroadcastedDim(ImplicitLocOpBuilder lb, ValueRange extentTensors,
                        ValueRange rankDiffs, Value outputDimension) {
  Value one = lb.create<ConstantIndexOp>(1);
  Value broadcastedDim = one;
  for (auto tup : llvm::zip(extentTensors, rankDiffs)) {
    Value shape = std::get<0>(tup);
    Value rankDiff = std::get<1>(tup);
    Value outOfBounds =
        lb.create<CmpIOp>(CmpIPredicate::ult, outputDimension, rankDiff);
    Type indexTy = lb.getIndexType();
    broadcastedDim =
        lb.create<IfOp>(
              TypeRange{indexTy}, outOfBounds,
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(loc, broadcastedDim);
              },
              [&](OpBuilder &b, Location loc) {
                // The broadcasting logic is:
                // - if one extent (here we arbitrarily choose the
                // extent from the greater-rank operand) is equal to 1,
                // then take the extent from the other operand
                // - otherwise, take the extent as-is.
                // Note that this logic remains correct in the presence
                // of dimensions of zero extent.
                Value lesserRankOperandDimension =
                    b.create<SubIOp>(loc, indexTy, outputDimension, rankDiff);
                Value lesserRankOperandExtent = b.create<tensor::ExtractOp>(
                    loc, shape, ValueRange{lesserRankOperandDimension});

                Value dimIsOne = b.create<CmpIOp>(loc, CmpIPredicate::eq,
                                                  lesserRankOperandExtent, one);
                Value dim = b.create<SelectOp>(loc, dimIsOne, broadcastedDim,
                                               lesserRankOperandExtent);
                b.create<scf::YieldOp>(loc, dim);
              })
            .getResult(0);
  }
  return broadcastedDim;
}
} // namespace

LogicalResult BroadcastOpConverter::matchAndRewrite(
    BroadcastOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  if (op.getType().isa<ShapeType>())
    return failure();

  auto loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);
  BroadcastOp::Adaptor transformed(operands);

  Value zero = lb.create<ConstantIndexOp>(0);
  Type indexTy = lb.getIndexType();

  // Save all the ranks for bounds checking. Because this is a tensor
  // representing the shape extents, the rank is the extent of the only
  // dimension in the tensor.
  SmallVector<Value> ranks, rankDiffs;
  llvm::append_range(ranks, llvm::map_range(transformed.shapes(), [&](Value v) {
                       return lb.create<memref::DimOp>(v, zero);
                     }));

  // Find the maximum rank
  Value maxRank = ranks.front();
  for (Value v : llvm::drop_begin(ranks, 1)) {
    Value rankIsGreater = lb.create<CmpIOp>(CmpIPredicate::ugt, v, maxRank);
    maxRank = lb.create<SelectOp>(rankIsGreater, v, maxRank);
  }

  // Calculate the difference of ranks and the maximum rank for later offsets.
  llvm::append_range(rankDiffs, llvm::map_range(ranks, [&](Value v) {
                       return lb.create<SubIOp>(indexTy, maxRank, v);
                     }));

  rewriter.replaceOp(
      op, lb.create<tensor::GenerateOp>(
                getExtentTensorType(lb.getContext()), ValueRange{maxRank},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value broadcastedDim = getBroadcastedDim(
                      ImplicitLocOpBuilder(loc, b), transformed.shapes(),
                      rankDiffs, args[0]);

                  b.create<tensor::YieldOp>(loc, broadcastedDim);
                })
              ->getResults());
  return success();
}

namespace {
class ConstShapeOpConverter : public OpConversionPattern<ConstShapeOp> {
public:
  using OpConversionPattern<ConstShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ConstShapeOpConverter::matchAndRewrite(
    ConstShapeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // For now, this lowering supports only extent tensors, not `shape.shape`
  // types.
  if (op.getType().isa<ShapeType>())
    return failure();

  auto loc = op.getLoc();
  SmallVector<Value, 4> extentOperands;
  for (auto extent : op.shape()) {
    extentOperands.push_back(
        rewriter.create<ConstantIndexOp>(loc, extent.getLimitedValue()));
  }
  Type indexTy = rewriter.getIndexType();
  Value tensor =
      rewriter.create<tensor::FromElementsOp>(loc, indexTy, extentOperands);
  Type resultTy = RankedTensorType::get({ShapedType::kDynamicSize}, indexTy);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultTy, tensor);
  return success();
}

namespace {
class ConstSizeOpConversion : public OpConversionPattern<ConstSizeOp> {
public:
  using OpConversionPattern<ConstSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstSizeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ConstSizeOpConversion::matchAndRewrite(
    ConstSizeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<ConstantIndexOp>(op, op.value().getSExtValue());
  return success();
}

namespace {
struct IsBroadcastableOpConverter
    : public OpConversionPattern<IsBroadcastableOp> {
  using OpConversionPattern<IsBroadcastableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsBroadcastableOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult IsBroadcastableOpConverter::matchAndRewrite(
    IsBroadcastableOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  IsBroadcastableOp::Adaptor transformed(operands);
  if (!llvm::all_of(op.shapes(),
                    [](Value v) { return !v.getType().isa<ShapeType>(); }))
    return failure();

  auto loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);
  Value zero = lb.create<ConstantIndexOp>(0);
  Value one = lb.create<ConstantIndexOp>(1);
  Type indexTy = lb.getIndexType();

  // Save all the ranks for bounds checking. Because this is a tensor
  // representing the shape extents, the rank is the extent of the only
  // dimension in the tensor.
  SmallVector<Value> ranks, rankDiffs;
  llvm::append_range(ranks, llvm::map_range(transformed.shapes(), [&](Value v) {
                       return lb.create<memref::DimOp>(v, zero);
                     }));

  // Find the maximum rank
  Value maxRank = ranks.front();
  for (Value v : llvm::drop_begin(ranks, 1)) {
    Value rankIsGreater = lb.create<CmpIOp>(CmpIPredicate::ugt, v, maxRank);
    maxRank = lb.create<SelectOp>(rankIsGreater, v, maxRank);
  }

  // Calculate the difference of ranks and the maximum rank for later offsets.
  llvm::append_range(rankDiffs, llvm::map_range(ranks, [&](Value v) {
                       return lb.create<SubIOp>(indexTy, maxRank, v);
                     }));

  Type i1Ty = rewriter.getI1Type();
  Value trueVal =
      rewriter.create<ConstantOp>(loc, i1Ty, rewriter.getBoolAttr(true));

  auto reduceResult = lb.create<ForOp>(
      loc, zero, maxRank, one, ValueRange{trueVal},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        // Find a non-1 dim, if it exists. Note that the first part of this
        // could reuse the Broadcast lowering entirely, but we redo the work
        // here to make optimizations easier between the two loops.
        Value broadcastedDim = getBroadcastedDim(
            ImplicitLocOpBuilder(loc, b), transformed.shapes(), rankDiffs, iv);

        Value broadcastable = iterArgs[0];
        for (auto tup : llvm::zip(transformed.shapes(), rankDiffs)) {
          Value shape, rankDiff;
          std::tie(shape, rankDiff) = tup;
          Value outOfBounds =
              b.create<CmpIOp>(loc, CmpIPredicate::ult, iv, rankDiff);
          broadcastable =
              b.create<IfOp>(
                   loc, TypeRange{i1Ty}, outOfBounds,
                   [&](OpBuilder &b, Location loc) {
                     // Non existent dimensions are always broadcastable
                     b.create<scf::YieldOp>(loc, broadcastable);
                   },
                   [&](OpBuilder &b, Location loc) {
                     // Every value needs to be either 1, or the same non-1
                     // value to be broadcastable in this dim.
                     Value operandDimension =
                         b.create<SubIOp>(loc, indexTy, iv, rankDiff);
                     Value dimensionExtent = b.create<tensor::ExtractOp>(
                         loc, shape, ValueRange{operandDimension});

                     Value equalOne = b.create<CmpIOp>(loc, CmpIPredicate::eq,
                                                       dimensionExtent, one);
                     Value equalBroadcasted =
                         b.create<CmpIOp>(loc, CmpIPredicate::eq,
                                          dimensionExtent, broadcastedDim);
                     Value result = b.create<AndOp>(
                         loc, broadcastable,
                         b.create<OrOp>(loc, equalOne, equalBroadcasted));
                     b.create<scf::YieldOp>(loc, result);
                   })
                  .getResult(0);
        }

        b.create<scf::YieldOp>(loc, broadcastable);
      });

  rewriter.replaceOp(op, reduceResult.results().front());
  return success();
}

namespace {
class GetExtentOpConverter : public OpConversionPattern<GetExtentOp> {
  using OpConversionPattern<GetExtentOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetExtentOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult GetExtentOpConverter::matchAndRewrite(
    GetExtentOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  GetExtentOp::Adaptor transformed(operands);

  // For now, only error-free types are supported by this lowering.
  if (op.getType().isa<SizeType>())
    return failure();

  // Derive shape extent directly from shape origin if possible. This
  // circumvents the necessity to materialize the shape in memory.
  if (auto shapeOfOp = op.shape().getDefiningOp<ShapeOfOp>()) {
    if (shapeOfOp.arg().getType().isa<ShapedType>()) {
      rewriter.replaceOpWithNewOp<memref::DimOp>(op, shapeOfOp.arg(),
                                                 transformed.dim());
      return success();
    }
  }

  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, rewriter.getIndexType(),
                                                 transformed.shape(),
                                                 ValueRange{transformed.dim()});
  return success();
}

namespace {
class RankOpConverter : public OpConversionPattern<shape::RankOp> {
public:
  using OpConversionPattern<shape::RankOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::RankOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
RankOpConverter::matchAndRewrite(shape::RankOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // For now, this lowering supports only error-free types.
  if (op.getType().isa<SizeType>())
    return failure();

  shape::RankOp::Adaptor transformed(operands);
  rewriter.replaceOpWithNewOp<memref::DimOp>(op, transformed.shape(), 0);
  return success();
}

namespace {
/// Converts `shape.reduce` to `scf.for`.
struct ReduceOpConverter : public OpConversionPattern<shape::ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::ReduceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
ReduceOpConverter::matchAndRewrite(shape::ReduceOp op, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands.
  if (op.shape().getType().isa<ShapeType>())
    return failure();

  auto loc = op.getLoc();
  shape::ReduceOp::Adaptor transformed(operands);

  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  Type indexTy = rewriter.getIndexType();
  Value rank =
      rewriter.create<memref::DimOp>(loc, indexTy, transformed.shape(), zero);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, rank, one, op.initVals(),
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value extent =
            b.create<tensor::ExtractOp>(loc, transformed.shape(), iv);

        SmallVector<Value, 2> mappedValues{iv, extent};
        mappedValues.append(args.begin(), args.end());

        BlockAndValueMapping mapping;
        Block *reduceBody = op.getBody();
        mapping.map(reduceBody->getArguments(), mappedValues);
        for (auto &nested : reduceBody->without_terminator())
          b.clone(nested, mapping);

        SmallVector<Value, 2> mappedResults;
        for (auto result : reduceBody->getTerminator()->getOperands())
          mappedResults.push_back(mapping.lookup(result));
        b.create<scf::YieldOp>(loc, mappedResults);
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
/// %c0 = constant 0 : index
/// %0 = dim %arg0, %c0 : tensor<?xindex>
/// %1 = dim %arg1, %c0 : tensor<?xindex>
/// %2 = cmpi "eq", %0, %1 : index
/// %result = scf.if %2 -> (i1) {
///   %c1 = constant 1 : index
///   %true = constant true
///   %4 = scf.for %arg2 = %c0 to %0 step %c1 iter_args(%arg3 = %true) -> (i1) {
///     %5 = tensor.extract %arg0[%arg2] : tensor<?xindex>
///     %6 = tensor.extract %arg1[%arg2] : tensor<?xindex>
///     %7 = cmpi "eq", %5, %6 : index
///     %8 = and %arg3, %7 : i1
///     scf.yield %8 : i1
///   }
///   scf.yield %4 : i1
/// } else {
///   %false = constant false
///   scf.yield %false : i1
/// }
///
struct ShapeEqOpConverter : public OpConversionPattern<ShapeEqOp> {
  using OpConversionPattern<ShapeEqOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeEqOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ShapeEqOpConverter::matchAndRewrite(ShapeEqOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  if (!llvm::all_of(op.shapes(),
                    [](Value v) { return !v.getType().isa<ShapeType>(); }))
    return failure();

  Type i1Ty = rewriter.getI1Type();
  if (op.shapes().size() <= 1) {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, i1Ty,
                                            rewriter.getBoolAttr(true));
    return success();
  }

  ShapeEqOp::Adaptor transformed(operands);
  auto loc = op.getLoc();
  Type indexTy = rewriter.getIndexType();
  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value firstShape = transformed.shapes().front();
  Value firstRank =
      rewriter.create<memref::DimOp>(loc, indexTy, firstShape, zero);
  Value result = nullptr;
  // Generate a linear sequence of compares, all with firstShape as lhs.
  for (Value shape : transformed.shapes().drop_front(1)) {
    Value rank = rewriter.create<memref::DimOp>(loc, indexTy, shape, zero);
    Value eqRank =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, firstRank, rank);
    auto same = rewriter.create<IfOp>(
        loc, i1Ty, eqRank,
        [&](OpBuilder &b, Location loc) {
          Value one = b.create<ConstantIndexOp>(loc, 1);
          Value init = b.create<ConstantOp>(loc, i1Ty, b.getBoolAttr(true));
          auto loop = b.create<scf::ForOp>(
              loc, zero, firstRank, one, ValueRange{init},
              [&](OpBuilder &b, Location nestedLoc, Value iv, ValueRange args) {
                Value conj = args[0];
                Value lhsExtent =
                    b.create<tensor::ExtractOp>(loc, firstShape, iv);
                Value rhsExtent = b.create<tensor::ExtractOp>(loc, shape, iv);
                Value eqExtent = b.create<CmpIOp>(loc, CmpIPredicate::eq,
                                                  lhsExtent, rhsExtent);
                Value conjNext = b.create<AndOp>(loc, conj, eqExtent);
                b.create<scf::YieldOp>(loc, ValueRange({conjNext}));
              });
          b.create<scf::YieldOp>(loc, loop.getResults());
        },
        [&](OpBuilder &b, Location loc) {
          Value result = b.create<ConstantOp>(loc, i1Ty, b.getBoolAttr(false));
          b.create<scf::YieldOp>(loc, result);
        });
    result = !result ? same.getResult(0)
                     : rewriter.create<AndOp>(loc, result, same.getResult(0));
  }
  rewriter.replaceOp(op, result);
  return success();
}

namespace {
class ShapeOfOpConversion : public OpConversionPattern<ShapeOfOp> {
public:
  using OpConversionPattern<ShapeOfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeOfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ShapeOfOpConversion::matchAndRewrite(
    ShapeOfOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // For now, only error-free types are supported by this lowering.
  if (op.getType().isa<ShapeType>())
    return failure();

  // For ranked tensor arguments, lower to `tensor.from_elements`.
  auto loc = op.getLoc();
  ShapeOfOp::Adaptor transformed(operands);
  Value tensor = transformed.arg();
  Type tensorTy = tensor.getType();
  if (tensorTy.isa<RankedTensorType>()) {

    // Build values for individual extents.
    SmallVector<Value, 8> extentValues;
    RankedTensorType rankedTensorTy = tensorTy.cast<RankedTensorType>();
    int64_t rank = rankedTensorTy.getRank();
    for (int64_t i = 0; i < rank; i++) {
      if (rankedTensorTy.isDynamicDim(i)) {
        Value extent = rewriter.create<memref::DimOp>(loc, tensor, i);
        extentValues.push_back(extent);
      } else {
        Value extent =
            rewriter.create<ConstantIndexOp>(loc, rankedTensorTy.getDimSize(i));
        extentValues.push_back(extent);
      }
    }

    // Materialize extent tensor.
    Value staticExtentTensor = rewriter.create<tensor::FromElementsOp>(
        loc, rewriter.getIndexType(), extentValues);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                staticExtentTensor);
    return success();
  }

  // Lower to `tensor.generate` otherwise.
  auto *ctx = rewriter.getContext();
  Value rank = rewriter.create<mlir::RankOp>(loc, tensor);
  rewriter.replaceOpWithNewOp<tensor::GenerateOp>(
      op, getExtentTensorType(ctx), ValueRange{rank},
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value dim = args.front();
        Value extent = b.create<memref::DimOp>(loc, tensor, dim);
        b.create<tensor::YieldOp>(loc, extent);
      });

  return success();
}

namespace {
class SplitAtOpConversion : public OpConversionPattern<SplitAtOp> {
public:
  using OpConversionPattern<SplitAtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplitAtOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult SplitAtOpConversion::matchAndRewrite(
    SplitAtOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Error conditions are not implemented, only lower if all operands and
  // results are extent tensors.
  if (llvm::any_of(ValueRange{op.operand(), op.head(), op.tail()},
                   [](Value v) { return v.getType().isa<ShapeType>(); }))
    return failure();

  SplitAtOp::Adaptor transformed(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Value zero = b.create<ConstantIndexOp>(0);
  Value rank = b.create<memref::DimOp>(transformed.operand(), zero);

  // index < 0 ? index + rank : index
  Value originalIndex = transformed.index();
  Value add = b.create<AddIOp>(originalIndex, rank);
  Value indexIsNegative =
      b.create<CmpIOp>(CmpIPredicate::slt, originalIndex, zero);
  Value index = b.create<SelectOp>(indexIsNegative, add, originalIndex);

  Value one = b.create<ConstantIndexOp>(1);
  Value head = b.create<SubTensorOp>(transformed.operand(), zero, index, one);
  Value tailSize = b.create<SubIOp>(rank, index);
  Value tail =
      b.create<SubTensorOp>(transformed.operand(), index, tailSize, one);
  rewriter.replaceOp(op, {head, tail});
  return success();
}

namespace {
class ToExtentTensorOpConversion
    : public OpConversionPattern<ToExtentTensorOp> {
public:
  using OpConversionPattern<ToExtentTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToExtentTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ToExtentTensorOpAdaptor adaptor(operands);

    if (!adaptor.input().getType().isa<RankedTensorType>())
      return rewriter.notifyMatchFailure(op, "input needs to be a tensor");

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                adaptor.input());
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
    : public ConvertShapeToStandardBase<ConvertShapeToStandardPass> {

  void runOnOperation() override;
};
} // namespace

void ConvertShapeToStandardPass::runOnOperation() {
  // Setup target legality.
  MLIRContext &ctx = getContext();
  ConversionTarget target(ctx);
  target.addLegalDialect<memref::MemRefDialect, StandardOpsDialect, SCFDialect,
                         tensor::TensorDialect>();
  target.addLegalOp<CstrRequireOp, FuncOp, ModuleOp>();

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
      BinaryOpConversion<AddOp, AddIOp>,
      BinaryOpConversion<MulOp, MulIOp>,
      BroadcastOpConverter,
      ConstShapeOpConverter,
      ConstSizeOpConversion,
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

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertShapeToStandardPass() {
  return std::make_unique<ConvertShapeToStandardPass>();
}
