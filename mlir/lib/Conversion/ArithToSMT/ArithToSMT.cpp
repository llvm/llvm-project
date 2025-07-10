//===- ArithToSMT.cpp
//------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToSMT/ArithToSMT.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOSMT
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower a arith::CmpIOp operation to a smt::BVCmpOp, smt::EqOp or
/// smt::DistinctOp
///
struct CmpIOpConversion : OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getPredicate() == arith::CmpIPredicate::eq) {
      rewriter.replaceOpWithNewOp<smt::EqOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
      return success();
    }

    if (adaptor.getPredicate() == arith::CmpIPredicate::ne) {
      rewriter.replaceOpWithNewOp<smt::DistinctOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
      return success();
    }

    smt::BVCmpPredicate pred;
    switch (adaptor.getPredicate()) {
    case arith::CmpIPredicate::sge:
      pred = smt::BVCmpPredicate::sge;
      break;
    case arith::CmpIPredicate::sgt:
      pred = smt::BVCmpPredicate::sgt;
      break;
    case arith::CmpIPredicate::sle:
      pred = smt::BVCmpPredicate::sle;
      break;
    case arith::CmpIPredicate::slt:
      pred = smt::BVCmpPredicate::slt;
      break;
    case arith::CmpIPredicate::uge:
      pred = smt::BVCmpPredicate::uge;
      break;
    case arith::CmpIPredicate::ugt:
      pred = smt::BVCmpPredicate::ugt;
      break;
    case arith::CmpIPredicate::ule:
      pred = smt::BVCmpPredicate::ule;
      break;
    case arith::CmpIPredicate::ult:
      pred = smt::BVCmpPredicate::ult;
      break;
    default:
      llvm_unreachable("all cases handled above");
    }

    rewriter.replaceOpWithNewOp<smt::BVCmpOp>(op, pred, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

/// Lower a arith::SubOp operation to an smt::BVNegOp + smt::BVAddOp
struct SubOpConversion : OpConversionPattern<arith::SubIOp> {
  using OpConversionPattern<arith::SubIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SubIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value negRhs = rewriter.create<smt::BVNegOp>(op.getLoc(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<smt::BVAddOp>(op, adaptor.getLhs(), negRhs);
    return success();
  }
};

/// Lower the SourceOp to the TargetOp one-to-one.
template <typename SourceOp, typename TargetOp>
struct OneToOneOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(
        op,
        OpConversionPattern<SourceOp>::typeConverter->convertType(
            op.getResult().getType()),
        adaptor.getOperands());
    return success();
  }
};

struct CeilDivSIOpConversion : OpConversionPattern<arith::CeilDivSIOp> {
  using OpConversionPattern<arith::CeilDivSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto numPlusDenom = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    auto bitWidth =
        llvm::cast<IntegerType>(getElementTypeOrSelf(adaptor.getLhs()))
            .getWidth();
    auto one = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, bitWidth);
    auto numPlusDenomMinusOne =
        rewriter.createOrFold<arith::SubIOp>(op.getLoc(), numPlusDenom, one);
    rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, numPlusDenomMinusOne,
                                                adaptor.getRhs());
    return success();
  }
};

/// Lower the SourceOp to the TargetOp special-casing if the second operand is
/// zero to return a new symbolic value.
template <typename SourceOp, typename TargetOp>
struct DivisionOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = dyn_cast<smt::BitVectorType>(adaptor.getRhs().getType());
    if (!type)
      return failure();

    auto resultType = OpConversionPattern<SourceOp>::typeConverter->convertType(
        op.getResult().getType());
    Value zero =
        rewriter.create<smt::BVConstantOp>(loc, APInt(type.getWidth(), 0));
    Value isZero = rewriter.create<smt::EqOp>(loc, adaptor.getRhs(), zero);
    Value symbolicVal = rewriter.create<smt::DeclareFunOp>(loc, resultType);
    Value division =
        rewriter.create<TargetOp>(loc, resultType, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, isZero, symbolicVal, division);
    return success();
  }
};

/// Converts an operation with a variadic number of operands to a chain of
/// binary operations assuming left-associativity of the operation.
template <typename SourceOp, typename TargetOp>
struct VariadicToBinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    if (operands.size() < 2)
      return failure();

    Value runner = operands[0];
    for (Value operand : operands.drop_front())
      runner = rewriter.create<TargetOp>(op.getLoc(), runner, operand);

    rewriter.replaceOp(op, runner);
    return success();
  }
};

/// Lower a arith::ConstantOp operation to smt::BVConstantOp
struct ArithConstantIntOpConversion
    : OpConversionPattern<arith::ConstantIntOp> {
  using OpConversionPattern<arith::ConstantIntOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto v = llvm::cast<IntegerAttr>(adaptor.getValue());
    if (v.getValue().getBitWidth() < 1)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "0-bit constants not supported");
    // TODO(max): signed/unsigned/signless semenatics
    rewriter.replaceOpWithNewOp<smt::BVConstantOp>(op, v.getValue());
    return success();
  }
};

} // namespace

void populateArithToSMTTypeConverter(TypeConverter &converter) {
  // The semantics of the builtin integer at the MLIR core level is currently
  // not very well defined. It is used for two-valued, four-valued, and possible
  // other multi-valued logic. Here, we interpret it as two-valued for now.
  // From a formal perspective, MLIR would ideally define its own types for
  // two-valued, four-valued, nine-valued (etc.) logic each. In MLIR upstream
  // the integer type also carries poison information (which we don't have in
  // MLIR?).
  converter.addConversion([](IntegerType type) -> std::optional<Type> {
    if (type.getWidth() <= 0)
      return std::nullopt;
    return smt::BitVectorType::get(type.getContext(), type.getWidth());
  });

  // Default target materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        ->getResult(0);
  });

  // Convert a 'smt.bool'-typed value to a 'smt.bv<N>'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, smt::BitVectorType resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1)
          return Value();

        if (!isa<smt::BoolType>(inputs[0].getType()))
          return Value();

        unsigned width = resultType.getWidth();
        Value constZero = builder.create<smt::BVConstantOp>(loc, 0, width);
        Value constOne = builder.create<smt::BVConstantOp>(loc, 1, width);
        return builder.create<smt::IteOp>(loc, inputs[0], constOne, constZero);
      });

  // Convert an unrealized conversion cast from 'smt.bool' to i1
  // into a direct conversion from 'smt.bool' to 'smt.bv<1>'.
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, smt::BitVectorType resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1 || resultType.getWidth() != 1)
          return Value();

        auto intType = dyn_cast<IntegerType>(inputs[0].getType());
        if (!intType || intType.getWidth() != 1)
          return Value();

        auto castOp =
            inputs[0].getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          return Value();

        if (!isa<smt::BoolType>(castOp.getInputs()[0].getType()))
          return Value();

        Value constZero = builder.create<smt::BVConstantOp>(loc, 0, 1);
        Value constOne = builder.create<smt::BVConstantOp>(loc, 1, 1);
        return builder.create<smt::IteOp>(loc, castOp.getInputs()[0], constOne,
                                          constZero);
      });

  // Convert a 'smt.bv<1>'-typed value to a 'smt.bool'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, smt::BoolType resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1)
          return Value();

        auto bvType = dyn_cast<smt::BitVectorType>(inputs[0].getType());
        if (!bvType || bvType.getWidth() != 1)
          return Value();

        Value constOne = builder.create<smt::BVConstantOp>(loc, 1, 1);
        return builder.create<smt::EqOp>(loc, inputs[0], constOne);
      });

  // Default source materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> Value {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        ->getResult(0);
  });
}

namespace {
struct ConvertArithToSMT
    : public impl::ConvertArithToSMTBase<ConvertArithToSMT> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CeilDivSIOpConversion>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    ConversionTarget target(getContext());
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalDialect<smt::SMTDialect>();

    TypeConverter converter;
    populateArithToSMTTypeConverter(converter);
    patterns.clear();
    arith::populateArithToSMTConversionPatterns(converter, patterns);

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir::arith {
void populateArithToSMTConversionPatterns(TypeConverter &converter,
                                          RewritePatternSet &patterns) {
  patterns.add<ArithConstantIntOpConversion, CmpIOpConversion, SubOpConversion,
               OneToOneOpConversion<arith::ShLIOp, smt::BVShlOp>,
               OneToOneOpConversion<arith::ShRUIOp, smt::BVLShrOp>,
               OneToOneOpConversion<arith::ShRSIOp, smt::BVAShrOp>,
               DivisionOpConversion<arith::DivSIOp, smt::BVSDivOp>,
               DivisionOpConversion<arith::DivUIOp, smt::BVUDivOp>,
               DivisionOpConversion<arith::RemSIOp, smt::BVSRemOp>,
               DivisionOpConversion<arith::RemUIOp, smt::BVURemOp>,
               VariadicToBinaryOpConversion<arith::AddIOp, smt::BVAddOp>,
               VariadicToBinaryOpConversion<arith::MulIOp, smt::BVMulOp>,
               VariadicToBinaryOpConversion<arith::AndIOp, smt::BVAndOp>,
               VariadicToBinaryOpConversion<arith::OrIOp, smt::BVOrOp>,
               VariadicToBinaryOpConversion<arith::XOrIOp, smt::BVXOrOp>>(
      converter, patterns.getContext());
}
} // namespace mlir::arith