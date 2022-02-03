//===- ArithmeticToLLVM.cpp - Arithmetic to LLVM dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

using AddIOpLowering = VectorConvertToLLVMPattern<arith::AddIOp, LLVM::AddOp>;
using SubIOpLowering = VectorConvertToLLVMPattern<arith::SubIOp, LLVM::SubOp>;
using MulIOpLowering = VectorConvertToLLVMPattern<arith::MulIOp, LLVM::MulOp>;
using DivUIOpLowering =
    VectorConvertToLLVMPattern<arith::DivUIOp, LLVM::UDivOp>;
using DivSIOpLowering =
    VectorConvertToLLVMPattern<arith::DivSIOp, LLVM::SDivOp>;
using RemUIOpLowering =
    VectorConvertToLLVMPattern<arith::RemUIOp, LLVM::URemOp>;
using RemSIOpLowering =
    VectorConvertToLLVMPattern<arith::RemSIOp, LLVM::SRemOp>;
using AndIOpLowering = VectorConvertToLLVMPattern<arith::AndIOp, LLVM::AndOp>;
using OrIOpLowering = VectorConvertToLLVMPattern<arith::OrIOp, LLVM::OrOp>;
using XOrIOpLowering = VectorConvertToLLVMPattern<arith::XOrIOp, LLVM::XOrOp>;
using ShLIOpLowering = VectorConvertToLLVMPattern<arith::ShLIOp, LLVM::ShlOp>;
using ShRUIOpLowering =
    VectorConvertToLLVMPattern<arith::ShRUIOp, LLVM::LShrOp>;
using ShRSIOpLowering =
    VectorConvertToLLVMPattern<arith::ShRSIOp, LLVM::AShrOp>;
using NegFOpLowering = VectorConvertToLLVMPattern<arith::NegFOp, LLVM::FNegOp>;
using AddFOpLowering = VectorConvertToLLVMPattern<arith::AddFOp, LLVM::FAddOp>;
using SubFOpLowering = VectorConvertToLLVMPattern<arith::SubFOp, LLVM::FSubOp>;
using MulFOpLowering = VectorConvertToLLVMPattern<arith::MulFOp, LLVM::FMulOp>;
using DivFOpLowering = VectorConvertToLLVMPattern<arith::DivFOp, LLVM::FDivOp>;
using RemFOpLowering = VectorConvertToLLVMPattern<arith::RemFOp, LLVM::FRemOp>;
using ExtUIOpLowering =
    VectorConvertToLLVMPattern<arith::ExtUIOp, LLVM::ZExtOp>;
using ExtSIOpLowering =
    VectorConvertToLLVMPattern<arith::ExtSIOp, LLVM::SExtOp>;
using ExtFOpLowering = VectorConvertToLLVMPattern<arith::ExtFOp, LLVM::FPExtOp>;
using TruncIOpLowering =
    VectorConvertToLLVMPattern<arith::TruncIOp, LLVM::TruncOp>;
using TruncFOpLowering =
    VectorConvertToLLVMPattern<arith::TruncFOp, LLVM::FPTruncOp>;
using UIToFPOpLowering =
    VectorConvertToLLVMPattern<arith::UIToFPOp, LLVM::UIToFPOp>;
using SIToFPOpLowering =
    VectorConvertToLLVMPattern<arith::SIToFPOp, LLVM::SIToFPOp>;
using FPToUIOpLowering =
    VectorConvertToLLVMPattern<arith::FPToUIOp, LLVM::FPToUIOp>;
using FPToSIOpLowering =
    VectorConvertToLLVMPattern<arith::FPToSIOp, LLVM::FPToSIOp>;
using BitcastOpLowering =
    VectorConvertToLLVMPattern<arith::BitcastOp, LLVM::BitcastOp>;
using SelectOpLowering =
    VectorConvertToLLVMPattern<arith::SelectOp, LLVM::SelectOp>;

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

/// Directly lower to LLVM op.
struct ConstantOpLowering : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
struct IndexCastOpLowering : public ConvertOpToLLVMPattern<arith::IndexCastOp> {
  using ConvertOpToLLVMPattern<arith::IndexCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CmpIOpLowering : public ConvertOpToLLVMPattern<arith::CmpIOp> {
  using ConvertOpToLLVMPattern<arith::CmpIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CmpFOpLowering : public ConvertOpToLLVMPattern<arith::CmpFOp> {
  using ConvertOpToLLVMPattern<arith::CmpFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOpLowering::matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  return LLVM::detail::oneToOneRewrite(op, LLVM::ConstantOp::getOperationName(),
                                       adaptor.getOperands(),
                                       *getTypeConverter(), rewriter);
}

//===----------------------------------------------------------------------===//
// IndexCastOpLowering
//===----------------------------------------------------------------------===//

LogicalResult IndexCastOpLowering::matchAndRewrite(
    arith::IndexCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto targetType = typeConverter->convertType(op.getResult().getType());
  auto targetElementType =
      typeConverter->convertType(getElementTypeOrSelf(op.getResult()))
          .cast<IntegerType>();
  auto sourceElementType =
      getElementTypeOrSelf(adaptor.getIn()).cast<IntegerType>();
  unsigned targetBits = targetElementType.getWidth();
  unsigned sourceBits = sourceElementType.getWidth();

  if (targetBits == sourceBits)
    rewriter.replaceOp(op, adaptor.getIn());
  else if (targetBits < sourceBits)
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, targetType, adaptor.getIn());
  else
    rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, targetType, adaptor.getIn());
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOpLowering
//===----------------------------------------------------------------------===//

// Convert arith.cmp predicate into the LLVM dialect CmpPredicate. The two enums
// share numerical values so just cast.
template <typename LLVMPredType, typename PredType>
static LLVMPredType convertCmpPredicate(PredType pred) {
  return static_cast<LLVMPredType>(pred);
}

LogicalResult
CmpIOpLowering::matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto operandType = adaptor.getLhs().getType();
  auto resultType = op.getResult().getType();

  // Handle the scalar and 1D vector cases.
  if (!operandType.isa<LLVM::LLVMArrayType>()) {
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }

  auto vectorType = resultType.dyn_cast<VectorType>();
  if (!vectorType)
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return LLVM::detail::handleMultidimensionalVectors(
      op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
      [&](Type llvm1DVectorTy, ValueRange operands) {
        OpAdaptor adaptor(operands);
        return rewriter.create<LLVM::ICmpOp>(
            op.getLoc(), llvm1DVectorTy,
            convertCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
            adaptor.getLhs(), adaptor.getRhs());
      },
      rewriter);
}

//===----------------------------------------------------------------------===//
// CmpFOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOpLowering::matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto operandType = adaptor.getLhs().getType();
  auto resultType = op.getResult().getType();

  // Handle the scalar and 1D vector cases.
  if (!operandType.isa<LLVM::LLVMArrayType>()) {
    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate<LLVM::FCmpPredicate>(op.getPredicate()),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }

  auto vectorType = resultType.dyn_cast<VectorType>();
  if (!vectorType)
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return LLVM::detail::handleMultidimensionalVectors(
      op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
      [&](Type llvm1DVectorTy, ValueRange operands) {
        OpAdaptor adaptor(operands);
        return rewriter.create<LLVM::FCmpOp>(
            op.getLoc(), llvm1DVectorTy,
            convertCmpPredicate<LLVM::FCmpPredicate>(op.getPredicate()),
            adaptor.getLhs(), adaptor.getRhs());
      },
      rewriter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertArithmeticToLLVMPass
    : public ConvertArithmeticToLLVMBase<ConvertArithmeticToLLVMPass> {
  ConvertArithmeticToLLVMPass() = default;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(&getContext(), options);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(converter,
                                                            patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithmeticToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    ConstantOpLowering,
    AddIOpLowering,
    SubIOpLowering,
    MulIOpLowering,
    DivUIOpLowering,
    DivSIOpLowering,
    RemUIOpLowering,
    RemSIOpLowering,
    AndIOpLowering,
    OrIOpLowering,
    XOrIOpLowering,
    ShLIOpLowering,
    ShRUIOpLowering,
    ShRSIOpLowering,
    NegFOpLowering,
    AddFOpLowering,
    SubFOpLowering,
    MulFOpLowering,
    DivFOpLowering,
    RemFOpLowering,
    ExtUIOpLowering,
    ExtSIOpLowering,
    ExtFOpLowering,
    TruncIOpLowering,
    TruncFOpLowering,
    UIToFPOpLowering,
    SIToFPOpLowering,
    FPToUIOpLowering,
    FPToSIOpLowering,
    IndexCastOpLowering,
    BitcastOpLowering,
    CmpIOpLowering,
    CmpFOpLowering,
    SelectOpLowering
  >(converter);
  // clang-format on
}

std::unique_ptr<Pass> mlir::arith::createConvertArithmeticToLLVMPass() {
  return std::make_unique<ConvertArithmeticToLLVMPass>();
}
