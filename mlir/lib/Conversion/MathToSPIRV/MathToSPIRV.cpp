//===- MathToSPIRV.cpp - Math to SPIR-V Patterns --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Math dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "math-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts unary and binary standard operations to SPIR-V operations.
template <typename StdOp, typename SPIRVOp>
class UnaryAndBinaryOpPattern final : public OpConversionPattern<StdOp> {
public:
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() <= 2);
    auto dstType = this->getTypeConverter()->convertType(operation.getType());
    if (!dstType)
      return failure();
    if (SPIRVOp::template hasTrait<OpTrait::spirv::UnsignedOp>() &&
        dstType != operation.getType()) {
      return operation.emitError(
          "bitwidth emulation is not implemented yet on unsigned op");
    }
    rewriter.template replaceOpWithNewOp<SPIRVOp>(operation, dstType, operands);
    return success();
  }
};

/// Converts math.log1p to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for log(1+x). Explicitly lower to
/// these operations.
class Log1pOpPattern final : public OpConversionPattern<math::Log1pOp> {
public:
  using OpConversionPattern<math::Log1pOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    Location loc = operation.getLoc();
    auto type =
        this->getTypeConverter()->convertType(operation.operand().getType());
    auto one = spirv::ConstantOp::getOne(type, operation.getLoc(), rewriter);
    auto onePlus = rewriter.create<spirv::FAddOp>(loc, one, operands[0]);
    rewriter.replaceOpWithNewOp<spirv::GLSLLogOp>(operation, type, onePlus);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateMathToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  patterns.add<Log1pOpPattern,
               UnaryAndBinaryOpPattern<math::CosOp, spirv::GLSLCosOp>,
               UnaryAndBinaryOpPattern<math::ExpOp, spirv::GLSLExpOp>,
               UnaryAndBinaryOpPattern<math::LogOp, spirv::GLSLLogOp>,
               UnaryAndBinaryOpPattern<math::RsqrtOp, spirv::GLSLInverseSqrtOp>,
               UnaryAndBinaryOpPattern<math::PowFOp, spirv::GLSLPowOp>,
               UnaryAndBinaryOpPattern<math::SinOp, spirv::GLSLSinOp>,
               UnaryAndBinaryOpPattern<math::SqrtOp, spirv::GLSLSqrtOp>,
               UnaryAndBinaryOpPattern<math::TanhOp, spirv::GLSLTanhOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir
