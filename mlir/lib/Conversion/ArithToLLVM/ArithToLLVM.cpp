//===- ArithToLLVM.cpp - Arithmetic to LLVM dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// Map arithmetic fastmath enum values to LLVMIR enum values.
static LLVM::FastmathFlags
convertArithFastMathFlagsToLLVM(arith::FastMathFlags arithFMF) {
  LLVM::FastmathFlags llvmFMF{};
  const std::pair<arith::FastMathFlags, LLVM::FastmathFlags> flags[] = {
      {arith::FastMathFlags::nnan, LLVM::FastmathFlags::nnan},
      {arith::FastMathFlags::ninf, LLVM::FastmathFlags::ninf},
      {arith::FastMathFlags::nsz, LLVM::FastmathFlags::nsz},
      {arith::FastMathFlags::arcp, LLVM::FastmathFlags::arcp},
      {arith::FastMathFlags::contract, LLVM::FastmathFlags::contract},
      {arith::FastMathFlags::afn, LLVM::FastmathFlags::afn},
      {arith::FastMathFlags::reassoc, LLVM::FastmathFlags::reassoc}};
  for (auto fmfMap : flags) {
    if (bitEnumContainsAny(arithFMF, fmfMap.first))
      llvmFMF = llvmFMF | fmfMap.second;
  }
  return llvmFMF;
}

// Create an LLVM fastmath attribute from a given arithmetic fastmath attribute.
static LLVM::FastmathFlagsAttr
convertArithFastMathAttr(arith::FastMathFlagsAttr fmfAttr) {
  arith::FastMathFlags arithFMF = fmfAttr.getValue();
  return LLVM::FastmathFlagsAttr::get(
      fmfAttr.getContext(), convertArithFastMathFlagsToLLVM(arithFMF));
}

// Attribute converter that populates a NamedAttrList by removing the fastmath
// attribute from the source operation attributes, and replacing it with an
// equivalent LLVM fastmath attribute.
template <typename SourceOp, typename TargetOp>
class AttrConvertFastMath {
public:
  AttrConvertFastMath(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith fastmath attribute.
    llvm::StringRef arithFMFAttrName = SourceOp::getFastMathAttrName();
    // Remove the source fastmath attribute.
    auto arithFMFAttr = convertedAttr.erase(arithFMFAttrName)
                            .dyn_cast_or_null<arith::FastMathFlagsAttr>();
    if (arithFMFAttr) {
      llvm::StringRef targetAttrName = TargetOp::getFastmathAttrName();
      convertedAttr.set(targetAttrName, convertArithFastMathAttr(arithFMFAttr));
    }
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};

// Attribute converter that populates a NamedAttrList by removing the fastmath
// attribute from the source operation attributes. This may be useful for
// target operations that do not require the fastmath attribute, or for targets
// that do not yet support the LLVM fastmath attribute.
template <typename SourceOp, typename TargetOp>
class AttrDropFastMath {
public:
  AttrDropFastMath(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith fastmath attribute.
    llvm::StringRef arithFMFAttrName = SourceOp::getFastMathAttrName();
    // Remove the source fastmath attribute.
    convertedAttr.erase(arithFMFAttrName);
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

using AddFOpLowering = VectorConvertToLLVMPattern<arith::AddFOp, LLVM::FAddOp,
                                                  AttrConvertFastMath>;
using AddIOpLowering = VectorConvertToLLVMPattern<arith::AddIOp, LLVM::AddOp>;
using AndIOpLowering = VectorConvertToLLVMPattern<arith::AndIOp, LLVM::AndOp>;
using BitcastOpLowering =
    VectorConvertToLLVMPattern<arith::BitcastOp, LLVM::BitcastOp>;
using DivFOpLowering = VectorConvertToLLVMPattern<arith::DivFOp, LLVM::FDivOp,
                                                  AttrConvertFastMath>;
using DivSIOpLowering =
    VectorConvertToLLVMPattern<arith::DivSIOp, LLVM::SDivOp>;
using DivUIOpLowering =
    VectorConvertToLLVMPattern<arith::DivUIOp, LLVM::UDivOp>;
using ExtFOpLowering = VectorConvertToLLVMPattern<arith::ExtFOp, LLVM::FPExtOp>;
using ExtSIOpLowering =
    VectorConvertToLLVMPattern<arith::ExtSIOp, LLVM::SExtOp>;
using ExtUIOpLowering =
    VectorConvertToLLVMPattern<arith::ExtUIOp, LLVM::ZExtOp>;
using FPToSIOpLowering =
    VectorConvertToLLVMPattern<arith::FPToSIOp, LLVM::FPToSIOp>;
using FPToUIOpLowering =
    VectorConvertToLLVMPattern<arith::FPToUIOp, LLVM::FPToUIOp>;
// TODO: Add LLVM intrinsic support for fastmath
using MaxFOpLowering =
    VectorConvertToLLVMPattern<arith::MaxFOp, LLVM::MaxNumOp, AttrDropFastMath>;
using MaxSIOpLowering =
    VectorConvertToLLVMPattern<arith::MaxSIOp, LLVM::SMaxOp>;
using MaxUIOpLowering =
    VectorConvertToLLVMPattern<arith::MaxUIOp, LLVM::UMaxOp>;
// TODO: Add LLVM intrinsic support for fastmath
using MinFOpLowering =
    VectorConvertToLLVMPattern<arith::MinFOp, LLVM::MinNumOp, AttrDropFastMath>;
using MinSIOpLowering =
    VectorConvertToLLVMPattern<arith::MinSIOp, LLVM::SMinOp>;
using MinUIOpLowering =
    VectorConvertToLLVMPattern<arith::MinUIOp, LLVM::UMinOp>;
using MulFOpLowering = VectorConvertToLLVMPattern<arith::MulFOp, LLVM::FMulOp,
                                                  AttrConvertFastMath>;
using MulIOpLowering = VectorConvertToLLVMPattern<arith::MulIOp, LLVM::MulOp>;
using NegFOpLowering = VectorConvertToLLVMPattern<arith::NegFOp, LLVM::FNegOp,
                                                  AttrConvertFastMath>;
using OrIOpLowering = VectorConvertToLLVMPattern<arith::OrIOp, LLVM::OrOp>;
// TODO: Add LLVM intrinsic support for fastmath
using RemFOpLowering =
    VectorConvertToLLVMPattern<arith::RemFOp, LLVM::FRemOp, AttrDropFastMath>;
using RemSIOpLowering =
    VectorConvertToLLVMPattern<arith::RemSIOp, LLVM::SRemOp>;
using RemUIOpLowering =
    VectorConvertToLLVMPattern<arith::RemUIOp, LLVM::URemOp>;
using SelectOpLowering =
    VectorConvertToLLVMPattern<arith::SelectOp, LLVM::SelectOp>;
using ShLIOpLowering = VectorConvertToLLVMPattern<arith::ShLIOp, LLVM::ShlOp>;
using ShRSIOpLowering =
    VectorConvertToLLVMPattern<arith::ShRSIOp, LLVM::AShrOp>;
using ShRUIOpLowering =
    VectorConvertToLLVMPattern<arith::ShRUIOp, LLVM::LShrOp>;
using SIToFPOpLowering =
    VectorConvertToLLVMPattern<arith::SIToFPOp, LLVM::SIToFPOp>;
using SubFOpLowering = VectorConvertToLLVMPattern<arith::SubFOp, LLVM::FSubOp,
                                                  AttrConvertFastMath>;
using SubIOpLowering = VectorConvertToLLVMPattern<arith::SubIOp, LLVM::SubOp>;
using TruncFOpLowering =
    VectorConvertToLLVMPattern<arith::TruncFOp, LLVM::FPTruncOp>;
using TruncIOpLowering =
    VectorConvertToLLVMPattern<arith::TruncIOp, LLVM::TruncOp>;
using UIToFPOpLowering =
    VectorConvertToLLVMPattern<arith::UIToFPOp, LLVM::UIToFPOp>;
using XOrIOpLowering = VectorConvertToLLVMPattern<arith::XOrIOp, LLVM::XOrOp>;

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

/// Directly lower to LLVM op.
struct ConstantOpLowering : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
template <typename OpTy, typename ExtCastTy>
struct IndexCastOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

using IndexCastOpSILowering =
    IndexCastOpLowering<arith::IndexCastOp, LLVM::SExtOp>;
using IndexCastOpUILowering =
    IndexCastOpLowering<arith::IndexCastUIOp, LLVM::ZExtOp>;

struct AddUICarryOpLowering
    : public ConvertOpToLLVMPattern<arith::AddUICarryOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::AddUICarryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CmpIOpLowering : public ConvertOpToLLVMPattern<arith::CmpIOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CmpFOpLowering : public ConvertOpToLLVMPattern<arith::CmpFOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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
                                       adaptor.getOperands(), op->getAttrs(),
                                       *getTypeConverter(), rewriter);
}

//===----------------------------------------------------------------------===//
// IndexCastOpLowering
//===----------------------------------------------------------------------===//

template <typename OpTy, typename ExtCastTy>
LogicalResult IndexCastOpLowering<OpTy, ExtCastTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type resultType = op.getResult().getType();
  Type targetElementType =
      this->typeConverter->convertType(getElementTypeOrSelf(resultType));
  Type sourceElementType =
      this->typeConverter->convertType(getElementTypeOrSelf(op.getIn()));
  unsigned targetBits = targetElementType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceElementType.getIntOrFloatBitWidth();

  if (targetBits == sourceBits) {
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }

  // Handle the scalar and 1D vector cases.
  Type operandType = adaptor.getIn().getType();
  if (!operandType.isa<LLVM::LLVMArrayType>()) {
    Type targetType = this->typeConverter->convertType(resultType);
    if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, targetType,
                                                 adaptor.getIn());
    else
      rewriter.replaceOpWithNewOp<ExtCastTy>(op, targetType, adaptor.getIn());
    return success();
  }

  if (!resultType.isa<VectorType>())
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return LLVM::detail::handleMultidimensionalVectors(
      op.getOperation(), adaptor.getOperands(), *(this->getTypeConverter()),
      [&](Type llvm1DVectorTy, ValueRange operands) -> Value {
        typename OpTy::Adaptor adaptor(operands);
        if (targetBits < sourceBits) {
          return rewriter.create<LLVM::TruncOp>(op.getLoc(), llvm1DVectorTy,
                                                adaptor.getIn());
        }
        return rewriter.create<ExtCastTy>(op.getLoc(), llvm1DVectorTy,
                                          adaptor.getIn());
      },
      rewriter);
}

//===----------------------------------------------------------------------===//
// AddUICarryOpLowering
//===----------------------------------------------------------------------===//

LogicalResult AddUICarryOpLowering::matchAndRewrite(
    arith::AddUICarryOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type operandType = adaptor.getLhs().getType();
  Type sumResultType = op.getSum().getType();
  Type carryResultType = op.getCarry().getType();

  if (!LLVM::isCompatibleType(operandType))
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();

  // Handle the scalar and 1D vector cases.
  if (!operandType.isa<LLVM::LLVMArrayType>()) {
    Type newCarryType = typeConverter->convertType(carryResultType);
    Type structType =
        LLVM::LLVMStructType::getLiteral(ctx, {sumResultType, newCarryType});
    Value addOverflow = rewriter.create<LLVM::UAddWithOverflowOp>(
        loc, structType, adaptor.getLhs(), adaptor.getRhs());
    Value sumExtracted =
        rewriter.create<LLVM::ExtractValueOp>(loc, addOverflow, 0);
    Value carryExtracted =
        rewriter.create<LLVM::ExtractValueOp>(loc, addOverflow, 1);
    rewriter.replaceOp(op, {sumExtracted, carryExtracted});
    return success();
  }

  if (!sumResultType.isa<VectorType>())
    return rewriter.notifyMatchFailure(loc, "expected vector result types");

  return rewriter.notifyMatchFailure(loc,
                                     "ND vector types are not supported yet");
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
  Type operandType = adaptor.getLhs().getType();
  Type resultType = op.getResult().getType();

  // Handle the scalar and 1D vector cases.
  if (!operandType.isa<LLVM::LLVMArrayType>()) {
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }

  if (!resultType.isa<VectorType>())
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
  Type operandType = adaptor.getLhs().getType();
  Type resultType = op.getResult().getType();

  // Handle the scalar and 1D vector cases.
  if (!operandType.isa<LLVM::LLVMArrayType>()) {
    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate<LLVM::FCmpPredicate>(op.getPredicate()),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }

  if (!resultType.isa<VectorType>())
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
struct ArithToLLVMConversionPass
    : public impl::ArithToLLVMConversionPassBase<ArithToLLVMConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(&getContext(), options);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    AddFOpLowering,
    AddIOpLowering,
    AndIOpLowering,
    AddUICarryOpLowering,
    BitcastOpLowering,
    ConstantOpLowering,
    CmpFOpLowering,
    CmpIOpLowering,
    DivFOpLowering,
    DivSIOpLowering,
    DivUIOpLowering,
    ExtFOpLowering,
    ExtSIOpLowering,
    ExtUIOpLowering,
    FPToSIOpLowering,
    FPToUIOpLowering,
    IndexCastOpSILowering,
    IndexCastOpUILowering,
    MaxFOpLowering,
    MaxSIOpLowering,
    MaxUIOpLowering,
    MinFOpLowering,
    MinSIOpLowering,
    MinUIOpLowering,
    MulFOpLowering,
    MulIOpLowering,
    NegFOpLowering,
    OrIOpLowering,
    RemFOpLowering,
    RemSIOpLowering,
    RemUIOpLowering,
    SelectOpLowering,
    ShLIOpLowering,
    ShRSIOpLowering,
    ShRUIOpLowering,
    SIToFPOpLowering,
    SubFOpLowering,
    SubIOpLowering,
    TruncFOpLowering,
    TruncIOpLowering,
    UIToFPOpLowering,
    XOrIOpLowering
  >(converter);
  // clang-format on
}
