//===- TensorToSPIRV.cpp - Tensor to SPIR-V Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Tensor dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Converts tensor.extract into loading using access chains from SPIR-V local
/// variables.
class TensorExtractPattern final
    : public OpConversionPattern<tensor::ExtractOp> {
public:
  TensorExtractPattern(TypeConverter &typeConverter, MLIRContext *context,
                       int64_t threshold, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        byteCountThreshold(threshold) {}

  LogicalResult
  matchAndRewrite(tensor::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(extractOp.getTensor().getType());

    if (!tensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(extractOp, "non-static tensor");

    if (tensorType.getNumElements() * tensorType.getElementTypeBitWidth() >
        byteCountThreshold * 8)
      return rewriter.notifyMatchFailure(extractOp,
                                         "exceeding byte count threshold");

    Location loc = extractOp.getLoc();

    int64_t rank = tensorType.getRank();
    SmallVector<int64_t, 4> strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * tensorType.getDimSize(i + 1);
    }

    Type varType = spirv::PointerType::get(adaptor.getTensor().getType(),
                                           spirv::StorageClass::Function);

    spirv::VariableOp varOp;
    if (adaptor.getTensor().getDefiningOp<spirv::ConstantOp>()) {
      // We could use the initializer directly; but certain driver compilers
      // have bugs dealing with that. So for now, use spirv.Store for
      // initialization.
      varOp = rewriter.create<spirv::VariableOp>(loc, varType,
                                                 spirv::StorageClass::Function,
                                                 /*initializer=*/nullptr);
      rewriter.create<spirv::StoreOp>(loc, varOp, adaptor.getTensor());
    } else {
      // Need to store the value to the local variable. It's questionable
      // whether we want to support such case though.
      return failure();
    }

    auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter.getIndexType();

    Value index = spirv::linearizeIndex(adaptor.getIndices(), strides,
                                        /*offset=*/0, indexType, loc, rewriter);
    auto acOp = rewriter.create<spirv::AccessChainOp>(loc, varOp, index);

    rewriter.replaceOpWithNewOp<spirv::LoadOp>(extractOp, acOp);

    return success();
  }

private:
  int64_t byteCountThreshold;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateTensorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                         int64_t byteCountThreshold,
                                         RewritePatternSet &patterns) {
  patterns.add<TensorExtractPattern>(typeConverter, patterns.getContext(),
                                     byteCountThreshold);
}
