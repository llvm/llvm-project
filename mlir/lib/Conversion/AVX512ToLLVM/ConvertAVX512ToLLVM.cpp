//===- ConvertAVX512ToLLVM.cpp - Convert AVX512 to the LLVM dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AVX512ToLLVM/ConvertAVX512ToLLVM.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::avx512;

template <typename OpTy>
static Type getSrcVectorElementType(Operation *op) {
  return cast<OpTy>(op)
      .src()
      .getType()
      .template cast<VectorType>()
      .getElementType();
}

namespace {

// TODO: turn these into simpler declarative templated patterns when we've had
// enough.
struct MaskRndScaleOp512Conversion : public ConvertToLLVMPattern {
  explicit MaskRndScaleOp512Conversion(MLIRContext *context,
                                       LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(MaskRndScaleOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getSrcVectorElementType<MaskRndScaleOp>(op);
    if (elementType.isF32())
      return LLVM::detail::oneToOneRewrite(
          op, LLVM::x86_avx512_mask_rndscale_ps_512::getOperationName(),
          operands, *getTypeConverter(), rewriter);
    if (elementType.isF64())
      return LLVM::detail::oneToOneRewrite(
          op, LLVM::x86_avx512_mask_rndscale_pd_512::getOperationName(),
          operands, *getTypeConverter(), rewriter);
    return failure();
  }
};

struct ScaleFOp512Conversion : public ConvertToLLVMPattern {
  explicit ScaleFOp512Conversion(MLIRContext *context,
                                 LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(MaskScaleFOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getSrcVectorElementType<MaskScaleFOp>(op);
    if (elementType.isF32())
      return LLVM::detail::oneToOneRewrite(
          op, LLVM::x86_avx512_mask_scalef_ps_512::getOperationName(), operands,
          *getTypeConverter(), rewriter);
    if (elementType.isF64())
      return LLVM::detail::oneToOneRewrite(
          op, LLVM::x86_avx512_mask_scalef_pd_512::getOperationName(), operands,
          *getTypeConverter(), rewriter);
    return failure();
  }
};
} // namespace

/// Populate the given list with patterns that convert from AVX512 to LLVM.
void mlir::populateAVX512ToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<MaskRndScaleOp512Conversion,
                  ScaleFOp512Conversion>(&converter.getContext(), converter);
  // clang-format on
}
