//===------- VectorToSPIRV.cpp - Vector to SPIRV lowering passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate SPIRV operations for Vector
// operations.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/VectorToSPIRV/ConvertVectorToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/ConvertVectorToSPIRVPass.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct VectorBroadcastConvert final
    : public SPIRVOpLowering<vector::BroadcastOp> {
  using SPIRVOpLowering<vector::BroadcastOp>::SPIRVOpLowering;
  LogicalResult
  matchAndRewrite(vector::BroadcastOp broadcastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (broadcastOp.source().getType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(broadcastOp.getVectorType()))
      return failure();
    vector::BroadcastOp::Adaptor adaptor(operands);
    SmallVector<Value, 4> source(broadcastOp.getVectorType().getNumElements(),
                                 adaptor.source());
    Value construct = rewriter.create<spirv::CompositeConstructOp>(
        broadcastOp.getLoc(), broadcastOp.getVectorType(), source);
    rewriter.replaceOp(broadcastOp, construct);
    return success();
  }
};

struct VectorExtractOpConvert final
    : public SPIRVOpLowering<vector::ExtractOp> {
  using SPIRVOpLowering<vector::ExtractOp>::SPIRVOpLowering;
  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (extractOp.getType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(extractOp.getVectorType()))
      return failure();
    vector::ExtractOp::Adaptor adaptor(operands);
    int32_t id = extractOp.position().begin()->cast<IntegerAttr>().getInt();
    Value newExtract = rewriter.create<spirv::CompositeExtractOp>(
        extractOp.getLoc(), adaptor.vector(), id);
    rewriter.replaceOp(extractOp, newExtract);
    return success();
  }
};

struct VectorInsertOpConvert final : public SPIRVOpLowering<vector::InsertOp> {
  using SPIRVOpLowering<vector::InsertOp>::SPIRVOpLowering;
  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (insertOp.getSourceType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(insertOp.getDestVectorType())) 
      return failure();
    vector::InsertOp::Adaptor adaptor(operands);
    int32_t id = insertOp.position().begin()->cast<IntegerAttr>().getInt();
    Value newInsert = rewriter.create<spirv::CompositeInsertOp>(
        insertOp.getLoc(), adaptor.source(), adaptor.dest(), id);
    rewriter.replaceOp(insertOp, newInsert);
    return success();
  }
};
} // namespace

void mlir::populateVectorToSPIRVPatterns(MLIRContext *context,
                                         SPIRVTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<VectorBroadcastConvert, VectorExtractOpConvert,
                  VectorInsertOpConvert>(context, typeConverter);
}

namespace {
struct LowerVectorToSPIRVPass
    : public ConvertVectorToSPIRVBase<LowerVectorToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void LowerVectorToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  OwningRewritePatternList patterns;
  populateVectorToSPIRVPatterns(context, typeConverter, patterns);

  target->addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target->addLegalOp<FuncOp>();

  if (failed(applyFullConversion(module, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertVectorToSPIRVPass() {
  return std::make_unique<LowerVectorToSPIRVPass>();
}
