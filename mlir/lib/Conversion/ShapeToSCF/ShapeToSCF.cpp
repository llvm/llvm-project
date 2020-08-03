//===- ShapeToSCF.cpp - conversion from Shape to SCF dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToSCF/ShapeToSCF.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::shape;
using namespace mlir::scf;

namespace {
struct BroadcastOpConverter : public OpConversionPattern<BroadcastOp> {
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult BroadcastOpConverter::matchAndRewrite(
    BroadcastOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  if (op.getType().isa<ShapeType>())
    return failure();

  assert(!op.lhs().getType().isa<ShapeType>() &&
         !op.rhs().getType().isa<ShapeType>());
  auto loc = op.getLoc();
  BroadcastOp::Adaptor transformed(operands);
  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);

  // Find smaller and greater rank and extent tensor.
  Value lhsRank = rewriter.create<DimOp>(loc, transformed.lhs(), zero);
  Value rhsRank = rewriter.create<DimOp>(loc, transformed.rhs(), zero);
  Value lhsSmaller =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, lhsRank, rhsRank);
  Type indexTy = rewriter.getIndexType();
  Type extentTensorTy = op.getType();
  auto ifOp = rewriter.create<IfOp>(
      loc, TypeRange{indexTy, extentTensorTy, indexTy, extentTensorTy},
      lhsSmaller,
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, ValueRange{lhsRank, transformed.lhs(),
                                               rhsRank, transformed.rhs()});
      },
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, ValueRange{rhsRank, transformed.rhs(),
                                               lhsRank, transformed.lhs()});
      });
  Value smallerRank = ifOp.getResult(0);
  Value smallerOperand = ifOp.getResult(1);
  Value greaterRank = ifOp.getResult(2);
  Value greaterOperand = ifOp.getResult(3);

  // Allocate stack memory for the broadcasted extent tensor.
  Type memTy = MemRefType::get({ShapedType::kDynamicSize}, indexTy);
  Value mem = rewriter.create<AllocaOp>(loc, memTy, ValueRange{greaterRank});

  // Copy extents from greater operand that are not challenged.
  Value rankDiff =
      rewriter.create<SubIOp>(loc, indexTy, greaterRank, smallerRank);
  rewriter.create<ForOp>(loc, zero, rankDiff, one, llvm::None,
                         [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
                           Value extent = b.create<ExtractElementOp>(
                               loc, greaterOperand, ValueRange{iv});
                           b.create<StoreOp>(loc, extent, mem, ValueRange{iv});
                           b.create<scf::YieldOp>(loc);
                         });

  // Determine remaining broadcasted extents.
  rewriter.create<ForOp>(
      loc, rankDiff, greaterRank, one, llvm::None,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
        Value greaterOperandExtent =
            b.create<ExtractElementOp>(loc, greaterOperand, ValueRange{iv});
        Value greaterOperandExtentIsOne =
            b.create<CmpIOp>(loc, CmpIPredicate::eq, greaterOperandExtent, one);
        auto ifOp = b.create<IfOp>(
            loc, TypeRange{indexTy}, greaterOperandExtentIsOne,
            [&](OpBuilder &b, Location loc) {
              Value ivShifted = b.create<SubIOp>(loc, indexTy, iv, rankDiff);
              Value smallerOperandExtent = b.create<ExtractElementOp>(
                  loc, smallerOperand, ValueRange{ivShifted});
              b.create<scf::YieldOp>(loc, smallerOperandExtent);
            },
            [&](OpBuilder &b, Location loc) {
              b.create<scf::YieldOp>(loc, greaterOperandExtent);
            });
        Value extent = ifOp.getResult(0);
        b.create<StoreOp>(loc, extent, mem, ValueRange{iv});
        b.create<scf::YieldOp>(loc);
      });

  // Load broadcasted shape as an extent tensor.
  rewriter.replaceOpWithNewOp<TensorLoadOp>(op, mem);
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
///     %5 = extract_element %arg0[%arg2] : tensor<?xindex>
///     %6 = extract_element %arg1[%arg2] : tensor<?xindex>
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
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  if (op.lhs().getType().isa<ShapeType>() ||
      op.rhs().getType().isa<ShapeType>()) {
    return failure();
  }

  ShapeEqOp::Adaptor transformed(operands);
  auto loc = op.getLoc();
  Type indexTy = rewriter.getIndexType();
  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value lhsRank = rewriter.create<DimOp>(loc, indexTy, transformed.lhs(), zero);
  Value rhsRank = rewriter.create<DimOp>(loc, indexTy, transformed.rhs(), zero);
  Value eqRank =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, lhsRank, rhsRank);
  Type i1Ty = rewriter.getI1Type();
  rewriter.replaceOpWithNewOp<IfOp>(
      op, i1Ty, eqRank,
      [&](OpBuilder &b, Location loc) {
        Value one = b.create<ConstantIndexOp>(loc, 1);
        Value init = b.create<ConstantOp>(loc, i1Ty, b.getBoolAttr(true));
        auto loop = b.create<scf::ForOp>(
            loc, zero, lhsRank, one, ValueRange{init},
            [&](OpBuilder &b, Location nestedLoc, Value iv, ValueRange args) {
              Value conj = args[0];
              Value lhsExtent =
                  b.create<ExtractElementOp>(loc, transformed.lhs(), iv);
              Value rhsExtent =
                  b.create<ExtractElementOp>(loc, transformed.rhs(), iv);
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
  Value rank = rewriter.create<DimOp>(loc, indexTy, transformed.shape(), zero);

  auto loop = rewriter.create<scf::ForOp>(
      loc, zero, rank, one, op.initVals(),
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value extent = b.create<ExtractElementOp>(loc, transformed.shape(), iv);

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
/// Converts `shape_of` to for loop for unranked tensors.
class ShapeOfOpConverter : public OpConversionPattern<ShapeOfOp> {
public:
  using OpConversionPattern<ShapeOfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeOfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ShapeOfOpConverter::matchAndRewrite(ShapeOfOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  // For now, this lowering supports only error-free arguments.
  if (op.getType().isa<ShapeType>())
    return failure();

  // For ranked tensors `shape_of` lowers to `std` and the pattern can be
  // found in the corresponding pass.
  ShapeOfOp::Adaptor transformed(operands);
  Value arg = transformed.arg();
  Type argTy = arg.getType();
  if (argTy.isa<RankedTensorType>())
    return failure();

  // Allocate stack memory.
  auto loc = op.getLoc();
  Value rank = rewriter.create<mlir::RankOp>(loc, arg);
  Type indexTy = rewriter.getIndexType();
  Type memTy = MemRefType::get({ShapedType::kDynamicSize}, indexTy);
  Value mem = rewriter.create<AllocaOp>(loc, memTy, ValueRange{rank});

  // Copy shape extents to stack-allocated memory.
  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  rewriter.create<scf::ForOp>(
      loc, zero, rank, one, llvm::None,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Value dim = rewriter.create<DimOp>(loc, arg, iv);
        rewriter.create<StoreOp>(loc, dim, mem, ValueRange{iv});
        rewriter.create<scf::YieldOp>(loc);
      });

  // Load extents to tensor value.
  rewriter.replaceOpWithNewOp<TensorLoadOp>(op.getOperation(), mem);
  return success();
}

namespace {
struct ConvertShapeToSCFPass
    : public ConvertShapeToSCFBase<ConvertShapeToSCFPass> {
  void runOnFunction() override;
};
} // namespace

void ConvertShapeToSCFPass::runOnFunction() {
  MLIRContext &ctx = getContext();

  // Populate conversion patterns.
  OwningRewritePatternList patterns;
  populateShapeToSCFConversionPatterns(patterns, &ctx);

  // Setup target legality.
  ConversionTarget target(getContext());
  target.addLegalDialect<SCFDialect, StandardOpsDialect>();

  // Apply conversion.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

void mlir::populateShapeToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
      BroadcastOpConverter,
      ShapeEqOpConverter,
      ReduceOpConverter,
      ShapeOfOpConverter>(ctx);
  // clang-format on
}

std::unique_ptr<FunctionPass> mlir::createConvertShapeToSCFPass() {
  return std::make_unique<ConvertShapeToSCFPass>();
}
