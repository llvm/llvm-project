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
    auto type =
        this->getTypeConverter()->convertType(operation.getOperand().getType());
    auto one = spirv::ConstantOp::getOne(type, operation.getLoc(), rewriter);
    auto onePlus =
        rewriter.create<spirv::FAddOp>(loc, one, adaptor.getOperands()[0]);
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
  patterns.add<
      Log1pOpPattern<spirv::GLSLLogOp>,
      spirv::UnaryAndBinaryOpPattern<math::AbsOp, spirv::GLSLFAbsOp>,
      spirv::UnaryAndBinaryOpPattern<math::CeilOp, spirv::GLSLCeilOp>,
      spirv::UnaryAndBinaryOpPattern<math::CosOp, spirv::GLSLCosOp>,
      spirv::UnaryAndBinaryOpPattern<math::ExpOp, spirv::GLSLExpOp>,
      spirv::UnaryAndBinaryOpPattern<math::FloorOp, spirv::GLSLFloorOp>,
      spirv::UnaryAndBinaryOpPattern<math::LogOp, spirv::GLSLLogOp>,
      spirv::UnaryAndBinaryOpPattern<math::PowFOp, spirv::GLSLPowOp>,
      spirv::UnaryAndBinaryOpPattern<math::RsqrtOp, spirv::GLSLInverseSqrtOp>,
      spirv::UnaryAndBinaryOpPattern<math::SinOp, spirv::GLSLSinOp>,
      spirv::UnaryAndBinaryOpPattern<math::SqrtOp, spirv::GLSLSqrtOp>,
      spirv::UnaryAndBinaryOpPattern<math::TanhOp, spirv::GLSLTanhOp>>(
      typeConverter, patterns.getContext());

  // OpenCL patterns
  patterns.add<Log1pOpPattern<spirv::OCLLogOp>,
               spirv::UnaryAndBinaryOpPattern<math::AbsOp, spirv::OCLFAbsOp>,
               spirv::UnaryAndBinaryOpPattern<math::CeilOp, spirv::OCLCeilOp>,
               spirv::UnaryAndBinaryOpPattern<math::CosOp, spirv::OCLCosOp>,
               spirv::UnaryAndBinaryOpPattern<math::ErfOp, spirv::OCLErfOp>,
               spirv::UnaryAndBinaryOpPattern<math::ExpOp, spirv::OCLExpOp>,
               spirv::UnaryAndBinaryOpPattern<math::FloorOp, spirv::OCLFloorOp>,
               spirv::UnaryAndBinaryOpPattern<math::LogOp, spirv::OCLLogOp>,
               spirv::UnaryAndBinaryOpPattern<math::PowFOp, spirv::OCLPowOp>,
               spirv::UnaryAndBinaryOpPattern<math::RsqrtOp, spirv::OCLRsqrtOp>,
               spirv::UnaryAndBinaryOpPattern<math::SinOp, spirv::OCLSinOp>,
               spirv::UnaryAndBinaryOpPattern<math::SqrtOp, spirv::OCLSqrtOp>,
               spirv::UnaryAndBinaryOpPattern<math::TanhOp, spirv::OCLTanhOp>>(
      typeConverter, patterns.getContext());
}

} // namespace mlir
