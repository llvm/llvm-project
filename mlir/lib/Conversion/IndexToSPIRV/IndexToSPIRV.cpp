//===- IndexToSPIRV.cpp - Index to SPIRV dialect conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace index;

namespace {

//===----------------------------------------------------------------------===//
// Trivial Conversions
//===----------------------------------------------------------------------===//

using ConvertIndexAdd = spirv::ElementwiseOpPattern<AddOp, spirv::IAddOp>;
using ConvertIndexSub = spirv::ElementwiseOpPattern<SubOp, spirv::ISubOp>;
using ConvertIndexMul = spirv::ElementwiseOpPattern<MulOp, spirv::IMulOp>;
using ConvertIndexDivS = spirv::ElementwiseOpPattern<DivSOp, spirv::SDivOp>;
using ConvertIndexDivU = spirv::ElementwiseOpPattern<DivUOp, spirv::UDivOp>;
using ConvertIndexRemS = spirv::ElementwiseOpPattern<RemSOp, spirv::SRemOp>;
using ConvertIndexRemU = spirv::ElementwiseOpPattern<RemUOp, spirv::UModOp>;
using ConvertIndexMaxS = spirv::ElementwiseOpPattern<MaxSOp, spirv::GLSMaxOp>;
using ConvertIndexMaxU = spirv::ElementwiseOpPattern<MaxUOp, spirv::GLUMaxOp>;
using ConvertIndexMinS = spirv::ElementwiseOpPattern<MinSOp, spirv::GLSMinOp>;
using ConvertIndexMinU = spirv::ElementwiseOpPattern<MinUOp, spirv::GLUMinOp>;

using ConvertIndexShl =
    spirv::ElementwiseOpPattern<ShlOp, spirv::ShiftLeftLogicalOp>;
using ConvertIndexShrS =
    spirv::ElementwiseOpPattern<ShrSOp, spirv::ShiftRightArithmeticOp>;
using ConvertIndexShrU =
    spirv::ElementwiseOpPattern<ShrUOp, spirv::ShiftRightLogicalOp>;

/// It is the case that when we convert bitwise operations to SPIR-V operations
/// we must take into account the special pattern in SPIR-V that if the
/// operands are boolean values, then SPIR-V uses `SPIRVLogicalOp`. Otherwise,
/// for non-boolean operands, SPIR-V should use `SPIRVBitwiseOp`. However,
/// index.add is never a boolean operation so we can directly convert it to the
/// Bitwise[And|Or]Op.
using ConvertIndexAnd = spirv::ElementwiseOpPattern<AndOp, spirv::BitwiseAndOp>;
using ConvertIndexOr = spirv::ElementwiseOpPattern<OrOp, spirv::BitwiseOrOp>;
using ConvertIndexXor = spirv::ElementwiseOpPattern<XOrOp, spirv::BitwiseXorOp>;

//===----------------------------------------------------------------------===//
// ConvertConstantBool
//===----------------------------------------------------------------------===//

// Converts index.bool.constant operation to spirv.Constant.
struct ConvertIndexConstantBoolOpPattern final
    : OpConversionPattern<BoolConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BoolConstantOp op, BoolConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(op, op.getType(),
                                                   op.getValueAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertConstant
//===----------------------------------------------------------------------===//

// Converts index.constant op to spirv.Constant. Will truncate from i64 to i32
// when required.
struct ConvertIndexConstantOpPattern final : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
    Type indexType = typeConverter->getIndexType();

    APInt value = op.getValue().trunc(typeConverter->getIndexTypeBitwidth());
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        op, indexType, IntegerAttr::get(indexType, value));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertIndexCeilDivS
//===----------------------------------------------------------------------===//

/// Convert `ceildivs(n, m)` into `x = m > 0 ? -1 : 1` and then
/// `n*m > 0 ? (n+x)/m + 1 : -(-n/m)`. Formula taken from the equivalent
/// conversion in IndexToLLVM.
struct ConvertIndexCeilDivSPattern final : OpConversionPattern<CeilDivSOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CeilDivSOp op, CeilDivSOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value n = adaptor.getLhs();
    Type n_type = n.getType();
    Value m = adaptor.getRhs();

    // Define the constants
    Value zero = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, 0));
    Value posOne = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, 1));
    Value negOne = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, -1));

    // Compute `x`.
    Value mPos = rewriter.create<spirv::SGreaterThanOp>(loc, m, zero);
    Value x = rewriter.create<spirv::SelectOp>(loc, mPos, negOne, posOne);

    // Compute the positive result.
    Value nPlusX = rewriter.create<spirv::IAddOp>(loc, n, x);
    Value nPlusXDivM = rewriter.create<spirv::SDivOp>(loc, nPlusX, m);
    Value posRes = rewriter.create<spirv::IAddOp>(loc, nPlusXDivM, posOne);

    // Compute the negative result.
    Value negN = rewriter.create<spirv::ISubOp>(loc, zero, n);
    Value negNDivM = rewriter.create<spirv::SDivOp>(loc, negN, m);
    Value negRes = rewriter.create<spirv::ISubOp>(loc, zero, negNDivM);

    // Pick the positive result if `n` and `m` have the same sign and `n` is
    // non-zero, i.e. `(n > 0) == (m > 0) && n != 0`.
    Value nPos = rewriter.create<spirv::SGreaterThanOp>(loc, n, zero);
    Value sameSign = rewriter.create<spirv::LogicalEqualOp>(loc, nPos, mPos);
    Value nNonZero = rewriter.create<spirv::INotEqualOp>(loc, n, zero);
    Value cmp = rewriter.create<spirv::LogicalAndOp>(loc, sameSign, nNonZero);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, cmp, posRes, negRes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertIndexCeilDivU
//===----------------------------------------------------------------------===//

/// Convert `ceildivu(n, m)` into `n == 0 ? 0 : (n-1)/m + 1`. Formula taken
/// from the equivalent conversion in IndexToLLVM.
struct ConvertIndexCeilDivUPattern final : OpConversionPattern<CeilDivUOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CeilDivUOp op, CeilDivUOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value n = adaptor.getLhs();
    Type n_type = n.getType();
    Value m = adaptor.getRhs();

    // Define the constants
    Value zero = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, 0));
    Value one = rewriter.create<spirv::ConstantOp>(loc, n_type,
                                                   IntegerAttr::get(n_type, 1));

    // Compute the non-zero result.
    Value minusOne = rewriter.create<spirv::ISubOp>(loc, n, one);
    Value quotient = rewriter.create<spirv::UDivOp>(loc, minusOne, m);
    Value plusOne = rewriter.create<spirv::IAddOp>(loc, quotient, one);

    // Pick the result
    Value cmp = rewriter.create<spirv::IEqualOp>(loc, n, zero);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, cmp, zero, plusOne);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertIndexFloorDivS
//===----------------------------------------------------------------------===//

/// Convert `floordivs(n, m)` into `x = m < 0 ? 1 : -1` and then
/// `n*m < 0 ? -1 - (x-n)/m : n/m`. Formula taken from the equivalent conversion
/// in IndexToLLVM.
struct ConvertIndexFloorDivSPattern final : OpConversionPattern<FloorDivSOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FloorDivSOp op, FloorDivSOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value n = adaptor.getLhs();
    Type n_type = n.getType();
    Value m = adaptor.getRhs();

    // Define the constants
    Value zero = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, 0));
    Value posOne = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, 1));
    Value negOne = rewriter.create<spirv::ConstantOp>(
        loc, n_type, IntegerAttr::get(n_type, -1));

    // Compute `x`.
    Value mNeg = rewriter.create<spirv::SLessThanOp>(loc, m, zero);
    Value x = rewriter.create<spirv::SelectOp>(loc, mNeg, posOne, negOne);

    // Compute the negative result
    Value xMinusN = rewriter.create<spirv::ISubOp>(loc, x, n);
    Value xMinusNDivM = rewriter.create<spirv::SDivOp>(loc, xMinusN, m);
    Value negRes = rewriter.create<spirv::ISubOp>(loc, negOne, xMinusNDivM);

    // Compute the positive result.
    Value posRes = rewriter.create<spirv::SDivOp>(loc, n, m);

    // Pick the negative result if `n` and `m` have different signs and `n` is
    // non-zero, i.e. `(n < 0) != (m < 0) && n != 0`.
    Value nNeg = rewriter.create<spirv::SLessThanOp>(loc, n, zero);
    Value diffSign = rewriter.create<spirv::LogicalNotEqualOp>(loc, nNeg, mNeg);
    Value nNonZero = rewriter.create<spirv::INotEqualOp>(loc, n, zero);

    Value cmp = rewriter.create<spirv::LogicalAndOp>(loc, diffSign, nNonZero);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, cmp, posRes, negRes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertIndexCast
//===----------------------------------------------------------------------===//

/// Convert a cast op. If the materialized index type is the same as the other
/// type, fold away the op. Otherwise, use the Convert SPIR-V operation.
/// Signed casts sign extend when the result bitwidth is larger. Unsigned casts
/// zero extend when the result bitwidth is larger.
template <typename CastOp, typename ConvertOp>
struct ConvertIndexCast final : OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
    Type indexType = typeConverter->getIndexType();

    Type srcType = adaptor.getInput().getType();
    Type dstType = op.getType();
    if (isa<IndexType>(srcType)) {
      srcType = indexType;
    }
    if (isa<IndexType>(dstType)) {
      dstType = indexType;
    }

    if (srcType == dstType) {
      rewriter.replaceOp(op, adaptor.getInput());
    } else {
      rewriter.template replaceOpWithNewOp<ConvertOp>(op, dstType,
                                                      adaptor.getOperands());
    }
    return success();
  }
};

using ConvertIndexCastS = ConvertIndexCast<CastSOp, spirv::SConvertOp>;
using ConvertIndexCastU = ConvertIndexCast<CastUOp, spirv::UConvertOp>;

//===----------------------------------------------------------------------===//
// ConvertIndexCmp
//===----------------------------------------------------------------------===//

// Helper template to replace the operation
template <typename ICmpOp>
static LogicalResult rewriteCmpOp(CmpOp op, CmpOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<ICmpOp>(op, adaptor.getLhs(), adaptor.getRhs());
  return success();
}

struct ConvertIndexCmpPattern final : OpConversionPattern<CmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, CmpOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We must convert the predicates to the corresponding int comparions.
    switch (op.getPred()) {
    case IndexCmpPredicate::EQ:
      return rewriteCmpOp<spirv::IEqualOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::NE:
      return rewriteCmpOp<spirv::INotEqualOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::SGE:
      return rewriteCmpOp<spirv::SGreaterThanEqualOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::SGT:
      return rewriteCmpOp<spirv::SGreaterThanOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::SLE:
      return rewriteCmpOp<spirv::SLessThanEqualOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::SLT:
      return rewriteCmpOp<spirv::SLessThanOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::UGE:
      return rewriteCmpOp<spirv::UGreaterThanEqualOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::UGT:
      return rewriteCmpOp<spirv::UGreaterThanOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::ULE:
      return rewriteCmpOp<spirv::ULessThanEqualOp>(op, adaptor, rewriter);
    case IndexCmpPredicate::ULT:
      return rewriteCmpOp<spirv::ULessThanOp>(op, adaptor, rewriter);
    }
  }
};

//===----------------------------------------------------------------------===//
// ConvertIndexSizeOf
//===----------------------------------------------------------------------===//

/// Lower `index.sizeof` to a constant with the value of the index bitwidth.
struct ConvertIndexSizeOf final : OpConversionPattern<SizeOfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SizeOfOp op, SizeOfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
    Type indexType = typeConverter->getIndexType();
    unsigned bitwidth = typeConverter->getIndexTypeBitwidth();
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        op, indexType, IntegerAttr::get(indexType, bitwidth));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void index::populateIndexToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
    ConvertIndexAdd,
    ConvertIndexSub,
    ConvertIndexMul,
    ConvertIndexDivS,
    ConvertIndexDivU,
    ConvertIndexRemS,
    ConvertIndexRemU,
    ConvertIndexMaxS,
    ConvertIndexMaxU,
    ConvertIndexMinS,
    ConvertIndexMinU,
    ConvertIndexShl,
    ConvertIndexShrS,
    ConvertIndexShrU,
    ConvertIndexAnd,
    ConvertIndexOr,
    ConvertIndexXor,
    ConvertIndexConstantBoolOpPattern,
    ConvertIndexConstantOpPattern,
    ConvertIndexCeilDivSPattern,
    ConvertIndexCeilDivUPattern,
    ConvertIndexFloorDivSPattern,
    ConvertIndexCastS,
    ConvertIndexCastU,
    ConvertIndexCmpPattern,
    ConvertIndexSizeOf
  >(typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
#define GEN_PASS_DEF_CONVERTINDEXTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertIndexToSPIRVPass
    : public impl::ConvertIndexToSPIRVPassBase<ConvertIndexToSPIRVPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<SPIRVConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    options.use64bitIndex = this->use64bitIndex;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    // Use UnrealizedConversionCast as the bridge so that we don't need to pull
    // in patterns for other dialects.
    target->addLegalOp<UnrealizedConversionCastOp>();

    // Allow the spirv operations we are converting to
    target->addLegalDialect<spirv::SPIRVDialect>();
    // Fail hard when there are any remaining 'index' ops.
    target->addIllegalDialect<index::IndexDialect>();

    RewritePatternSet patterns(&getContext());
    index::populateIndexToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
