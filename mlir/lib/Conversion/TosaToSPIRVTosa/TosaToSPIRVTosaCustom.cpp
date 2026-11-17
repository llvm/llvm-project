//===- TosaToSPIRVTosaCustom.cpp - TOSA to SPIR-V Graph/TOSA patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert TOSA IR to SPIR-V Graph/TOSA.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#define DEBUG_TYPE "tosa-to-spirv-tosa-custom-pattern"

namespace mlir::tosa {
namespace {

Value encodeStringAsI8Array(StringRef value, Location loc,
                            ConversionPatternRewriter &rewriter) {
  Type i8Type = rewriter.getIntegerType(8);
  // Empty strings are encoded as a single NULL byte because SPIR-V array
  // types require at least one element.
  StringRef encodedValue = value.empty() ? StringRef("\0", 1) : value;

  SmallVector<Attribute> bytes;
  bytes.reserve(encodedValue.size());
  llvm::transform(
      encodedValue, std::back_inserter(bytes),
      [&](unsigned char byte) { return IntegerAttr::get(i8Type, byte); });

  auto arrayType =
      spirv::ArrayType::get(i8Type, static_cast<unsigned>(bytes.size()));
  auto arrayValue = ArrayAttr::get(rewriter.getContext(), bytes);
  return spirv::ConstantOp::create(rewriter, loc, arrayType, arrayValue);
}

struct TosaCustomOpConvert final : public OpConversionPattern<tosa::CustomOp> {
  TosaCustomOpConvert(const TypeConverter &typeConverter, MLIRContext *context,
                      llvm::StringMap<int32_t> domainToOpcode)
      : OpConversionPattern<tosa::CustomOp>(typeConverter, context),
        domainToOpcode(std::move(domainToOpcode)) {}

  LogicalResult
  matchAndRewrite(tosa::CustomOp op, tosa::CustomOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opCode = domainToOpcode.find(op.getDomainName());
    if (opCode == domainToOpcode.end())
      return failure();

    if (op->getResultTypes().empty())
      return op.emitOpError("with mapped domain requires at least one result");

    SmallVector<Type> types;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      types)))
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Type resultType =
        types.size() == 1 ? types.front() : spirv::StructType::get(types);

    Value operatorName =
        encodeStringAsI8Array(op.getOperatorName(), op.getLoc(), rewriter);
    Value implementationAttrsBlob = encodeStringAsI8Array(
        op.getImplementationAttrs(), op.getLoc(), rewriter);

    SmallVector<Value> inputs = {operatorName, implementationAttrsBlob};
    inputs.append(adaptor.getInputList().begin(), adaptor.getInputList().end());

    Value result = spirv::ExperimentalMLCallOp::create(
        rewriter, op.getLoc(), resultType,
        rewriter.getI32IntegerAttr(opCode->second), inputs);

    if (types.size() == 1) {
      rewriter.replaceOp(op, result);
      return success();
    }

    SmallVector<Value> results;
    for (auto index : llvm::seq<int32_t>(0, types.size())) {
      results.push_back(spirv::CompositeExtractOp::create(rewriter, op.getLoc(),
                                                          result, {index}));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  llvm::StringMap<int32_t> domainToOpcode;
};

} // namespace

void populateTosaToSPIRVTosaCustomConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns,
    llvm::StringMap<int32_t> domainToOpcode) {
  patterns.add<TosaCustomOpConvert>(typeConverter, patterns.getContext(),
                                    std::move(domainToOpcode));
}

} // namespace mlir::tosa
