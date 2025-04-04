//===- LegalizeForLLVMExport.cpp - Prepare X86Vector for LLVM translation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/Transforms.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::x86vector;

namespace {

// Replaces an operation with a call to an LLVM intrinsic.
LogicalResult intrinsicRewrite(Operation *op, StringAttr intrinsic,
                               ValueRange operands,
                               const LLVMTypeConverter &typeConverter,
                               ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();

  unsigned numResults = op->getNumResults();
  Type resType;
  if (numResults != 0)
    resType = typeConverter.packOperationResults(op->getResultTypes());

  auto callIntrOp =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, resType, intrinsic, operands);
  // Propagate attributes.
  callIntrOp->setAttrs(op->getAttrDictionary());

  if (numResults <= 1) {
    // Directly replace the original op.
    rewriter.replaceOp(op, callIntrOp);
  } else {
    // Extract individual results from packed structure and use them as
    // replacements.
    SmallVector<Value, 4> results;
    results.reserve(numResults);
    Value intrRes = callIntrOp.getResults();
    for (unsigned i = 0; i < numResults; ++i) {
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(loc, intrRes, i));
    }
    rewriter.replaceOp(op, results);
  }

  return success();
}

template <typename OpTy>
struct CallIntrinsic : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return intrinsicRewrite(op, rewriter.getStringAttr(op.getIntrinsicName()),
                            adaptor.getOperands(), *this->getTypeConverter(),
                            rewriter);
  }
};

struct MaskCompressOpConversion
    : public ConvertOpToLLVMPattern<MaskCompressOp> {
  using ConvertOpToLLVMPattern<MaskCompressOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MaskCompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = adaptor.getA().getType();

    Value src;
    if (op.getSrc()) {
      src = adaptor.getSrc();
    } else if (op.getConstantSrc()) {
      src = rewriter.create<LLVM::ConstantOp>(op.getLoc(), opType,
                                              op.getConstantSrcAttr());
    } else {
      auto zeroAttr = rewriter.getZeroAttr(opType);
      src = rewriter.create<LLVM::ConstantOp>(op->getLoc(), opType, zeroAttr);
    }
    SmallVector<Value> operands = {adaptor.getA(), src, adaptor.getK()};

    return intrinsicRewrite(op, rewriter.getStringAttr(op.getIntrinsicName()),
                            operands, *this->getTypeConverter(), rewriter);
  }
};

struct DotOpConversion : public ConvertOpToLLVMPattern<DotOp> {
  using ConvertOpToLLVMPattern<DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type llvmIntType = rewriter.getIntegerType(8);
    // Dot product of all elements, broadcasted to all elements.
    auto attr = rewriter.getI8IntegerAttr(static_cast<int8_t>(0xff));
    Value scale =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), llvmIntType, attr);

    SmallVector<Value> operands(adaptor.getOperands());
    operands.push_back(scale);

    return intrinsicRewrite(op, rewriter.getStringAttr(op.getIntrinsicName()),
                            operands, *this->getTypeConverter(), rewriter);
  }
};

} // namespace

/// Populate the given list with patterns that convert from X86Vector to LLVM.
void mlir::populateX86VectorLegalizeForLLVMExportPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MaskCompressOpConversion, CallIntrinsic<MaskRndScaleOp>,
               CallIntrinsic<MaskScaleFOp>, CallIntrinsic<Vp2IntersectOp>,
               CallIntrinsic<DotBF16Op>, CallIntrinsic<CvtPackedF32ToBF16Op>,
               CallIntrinsic<RsqrtOp>, DotOpConversion>(converter);
}

void mlir::configureX86VectorLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addIllegalOp<MaskCompressOp, MaskRndScaleOp, MaskScaleFOp,
                      Vp2IntersectOp, DotBF16Op, CvtPackedF32ToBF16Op, RsqrtOp,
                      DotOp>();
}
