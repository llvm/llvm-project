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

#include "../SPIRVCommon/Pattern.h"
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
/// Converts math.expm1 to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for exp(x)-1. Explicitly lower to
/// these operations.
template <typename ExpOp>
class ExpM1OpPattern final : public OpConversionPattern<math::ExpM1Op> {
public:
  using OpConversionPattern<math::ExpM1Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    Location loc = operation.getLoc();
    auto type = this->getTypeConverter()->convertType(operation.getType());
    auto exp = rewriter.create<ExpOp>(loc, type, adaptor.getOperand());
    auto one = spirv::ConstantOp::getOne(type, loc, rewriter);
    rewriter.replaceOpWithNewOp<spirv::FSubOp>(operation, exp, one);
    return success();
  }
};

/// Converts math.log1p to SPIR-V ops.
///
/// SPIR-V does not have a direct operations for log(1+x). Explicitly lower to
/// these operations.
template <typename LogOp>
class Log1pOpPattern final : public OpConversionPattern<math::Log1pOp> {
public:
  using OpConversionPattern<math::Log1pOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 1);
    Location loc = operation.getLoc();
    auto type = this->getTypeConverter()->convertType(operation.getType());
    auto one = spirv::ConstantOp::getOne(type, operation.getLoc(), rewriter);
    auto onePlus =
        rewriter.create<spirv::FAddOp>(loc, one, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<LogOp>(operation, type, onePlus);
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

  // GLSL patterns
  patterns
      .add<Log1pOpPattern<spirv::GLSLLogOp>, ExpM1OpPattern<spirv::GLSLExpOp>,
           spirv::ElementwiseOpPattern<math::AbsOp, spirv::GLSLFAbsOp>,
           spirv::ElementwiseOpPattern<math::CeilOp, spirv::GLSLCeilOp>,
           spirv::ElementwiseOpPattern<math::CosOp, spirv::GLSLCosOp>,
           spirv::ElementwiseOpPattern<math::ExpOp, spirv::GLSLExpOp>,
           spirv::ElementwiseOpPattern<math::FloorOp, spirv::GLSLFloorOp>,
           spirv::ElementwiseOpPattern<math::LogOp, spirv::GLSLLogOp>,
           spirv::ElementwiseOpPattern<math::PowFOp, spirv::GLSLPowOp>,
           spirv::ElementwiseOpPattern<math::RsqrtOp, spirv::GLSLInverseSqrtOp>,
           spirv::ElementwiseOpPattern<math::SinOp, spirv::GLSLSinOp>,
           spirv::ElementwiseOpPattern<math::SqrtOp, spirv::GLSLSqrtOp>,
           spirv::ElementwiseOpPattern<math::TanhOp, spirv::GLSLTanhOp>,
           spirv::ElementwiseOpPattern<math::FmaOp, spirv::GLSLFmaOp>>(
          typeConverter, patterns.getContext());

  // OpenCL patterns
  patterns.add<Log1pOpPattern<spirv::OCLLogOp>, ExpM1OpPattern<spirv::OCLExpOp>,
               spirv::ElementwiseOpPattern<math::AbsOp, spirv::OCLFAbsOp>,
               spirv::ElementwiseOpPattern<math::CeilOp, spirv::OCLCeilOp>,
               spirv::ElementwiseOpPattern<math::CosOp, spirv::OCLCosOp>,
               spirv::ElementwiseOpPattern<math::ErfOp, spirv::OCLErfOp>,
               spirv::ElementwiseOpPattern<math::ExpOp, spirv::OCLExpOp>,
               spirv::ElementwiseOpPattern<math::FloorOp, spirv::OCLFloorOp>,
               spirv::ElementwiseOpPattern<math::LogOp, spirv::OCLLogOp>,
               spirv::ElementwiseOpPattern<math::PowFOp, spirv::OCLPowOp>,
               spirv::ElementwiseOpPattern<math::RsqrtOp, spirv::OCLRsqrtOp>,
               spirv::ElementwiseOpPattern<math::SinOp, spirv::OCLSinOp>,
               spirv::ElementwiseOpPattern<math::SqrtOp, spirv::OCLSqrtOp>,
               spirv::ElementwiseOpPattern<math::TanhOp, spirv::OCLTanhOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir
