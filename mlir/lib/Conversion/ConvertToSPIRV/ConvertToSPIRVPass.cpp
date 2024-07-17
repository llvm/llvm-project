//===- ConvertToSPIRVPass.cpp - MLIR SPIR-V Conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToSPIRV/ConvertToSPIRVPass.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#define DEBUG_TYPE "convert-to-spirv"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// A pass to perform the SPIR-V conversion.
struct ConvertToSPIRVPass final
    : impl::ConvertToSPIRVPassBase<ConvertToSPIRVPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    SPIRVTypeConverter typeConverter(targetAttr);

    RewritePatternSet patterns(context);
    ScfToSPIRVContext scfToSPIRVContext;

    // Populate patterns.
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);
    index::populateIndexToSPIRVPatterns(typeConverter, patterns);
    populateVectorToSPIRVPatterns(typeConverter, patterns);
    populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);
    ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
