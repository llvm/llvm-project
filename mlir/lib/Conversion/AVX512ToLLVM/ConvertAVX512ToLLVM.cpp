//===- ConvertAVX512ToLLVM.cpp - Convert AVX512 to the LLVM dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AVX512ToLLVM/ConvertAVX512ToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::avx512;

template <typename OpTy>
static Type getSrcVectorElementType(OpTy op) {
  return op.src().getType().template cast<VectorType>().getElementType();
}

// TODO: Code is currently copy-pasted and adapted from the code
// 1-1 LLVM conversion. It would better if it were properly exposed in core and
// reusable.
/// Basic lowering implementation for one-to-one rewriting from AVX512 Ops to
/// LLVM Dialect Ops. Convert the type of the result to an LLVM type, pass
/// operands as is, preserve attributes.
template <typename SourceOp, typename TargetOp>
static LogicalResult
matchAndRewriteOneToOne(const ConvertToLLVMPattern &lowering,
                        LLVMTypeConverter &typeConverter, Operation *op,
                        ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  Type packedType;
  if (numResults != 0) {
    packedType = typeConverter.packFunctionResults(op->getResultTypes());
    if (!packedType)
      return failure();
  }

  auto newOp = rewriter.create<TargetOp>(op->getLoc(), packedType, operands,
                                         op->getAttrs());

  // If the operation produced 0 or 1 result, return them immediately.
  if (numResults == 0)
    return rewriter.eraseOp(op), success();
  if (numResults == 1)
    return rewriter.replaceOp(op, newOp.getOperation()->getResult(0)),
           success();

  // Otherwise, it had been converted to an operation producing a structure.
  // Extract individual results from the structure and return them as list.
  SmallVector<Value, 4> results;
  results.reserve(numResults);
  for (unsigned i = 0; i < numResults; ++i) {
    auto type = typeConverter.convertType(op->getResult(i).getType());
    results.push_back(rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), type, newOp.getOperation()->getResult(0),
        rewriter.getI64ArrayAttr(i)));
  }
  rewriter.replaceOp(op, results);
  return success();
}

namespace {
// TODO: Patterns are too verbose due to the fact that we have 1 op (e.g.
// MaskRndScaleOp) and different possible target ops. It would be better to take
// a Functor so that all these conversions become 1-liners.
struct MaskRndScaleOpPS512Conversion : public ConvertToLLVMPattern {
  explicit MaskRndScaleOpPS512Conversion(MLIRContext *context,
                                         LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(MaskRndScaleOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!getSrcVectorElementType(cast<MaskRndScaleOp>(op)).isF32())
      return failure();
    return matchAndRewriteOneToOne<MaskRndScaleOp,
                                   LLVM::x86_avx512_mask_rndscale_ps_512>(
        *this, this->typeConverter, op, operands, rewriter);
  }
};

struct MaskRndScaleOpPD512Conversion : public ConvertToLLVMPattern {
  explicit MaskRndScaleOpPD512Conversion(MLIRContext *context,
                                         LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(MaskRndScaleOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!getSrcVectorElementType(cast<MaskRndScaleOp>(op)).isF64())
      return failure();
    return matchAndRewriteOneToOne<MaskRndScaleOp,
                                   LLVM::x86_avx512_mask_rndscale_pd_512>(
        *this, this->typeConverter, op, operands, rewriter);
  }
};

struct ScaleFOpPS512Conversion : public ConvertToLLVMPattern {
  explicit ScaleFOpPS512Conversion(MLIRContext *context,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(MaskScaleFOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!getSrcVectorElementType(cast<MaskScaleFOp>(op)).isF32())
      return failure();
    return matchAndRewriteOneToOne<MaskScaleFOp,
                                   LLVM::x86_avx512_mask_scalef_ps_512>(
        *this, this->typeConverter, op, operands, rewriter);
  }
};

struct ScaleFOpPD512Conversion : public ConvertToLLVMPattern {
  explicit ScaleFOpPD512Conversion(MLIRContext *context,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(MaskScaleFOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!getSrcVectorElementType(cast<MaskScaleFOp>(op)).isF64())
      return failure();
    return matchAndRewriteOneToOne<MaskScaleFOp,
                                   LLVM::x86_avx512_mask_scalef_pd_512>(
        *this, this->typeConverter, op, operands, rewriter);
  }
};
} // namespace

/// Populate the given list with patterns that convert from AVX512 to LLVM.
void mlir::populateAVX512ToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  // clang-format off
  patterns.insert<MaskRndScaleOpPS512Conversion,
                  MaskRndScaleOpPD512Conversion,
                  ScaleFOpPS512Conversion,
                  ScaleFOpPD512Conversion>(ctx, converter);
  // clang-format on
}

namespace {
struct ConvertAVX512ToLLVMPass
    : public ConvertAVX512ToLLVMBase<ConvertAVX512ToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAVX512ToLLVMPass::runOnOperation() {
  // Convert to the LLVM IR dialect.
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateAVX512ToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<LLVM::LLVMAVX512Dialect>();
  target.addIllegalDialect<avx512::AVX512Dialect>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertAVX512ToLLVMPass() {
  return std::make_unique<ConvertAVX512ToLLVMPass>();
}
