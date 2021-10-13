//===- StandardToSPIRV.cpp - Standard to SPIR-V Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert standard dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "std-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Converts the given `srcAttr` into a boolean attribute if it holds an
/// integral value. Returns null attribute if conversion fails.
static BoolAttr convertBoolAttr(Attribute srcAttr, Builder builder) {
  if (auto boolAttr = srcAttr.dyn_cast<BoolAttr>())
    return boolAttr;
  if (auto intAttr = srcAttr.dyn_cast<IntegerAttr>())
    return builder.getBoolAttr(intAttr.getValue().getBoolValue());
  return BoolAttr();
}

/// Converts the given `srcAttr` to a new attribute of the given `dstType`.
/// Returns null attribute if conversion fails.
static IntegerAttr convertIntegerAttr(IntegerAttr srcAttr, IntegerType dstType,
                                      Builder builder) {
  // If the source number uses less active bits than the target bitwidth, then
  // it should be safe to convert.
  if (srcAttr.getValue().isIntN(dstType.getWidth()))
    return builder.getIntegerAttr(dstType, srcAttr.getInt());

  // XXX: Try again by interpreting the source number as a signed value.
  // Although integers in the standard dialect are signless, they can represent
  // a signed number. It's the operation decides how to interpret. This is
  // dangerous, but it seems there is no good way of handling this if we still
  // want to change the bitwidth. Emit a message at least.
  if (srcAttr.getValue().isSignedIntN(dstType.getWidth())) {
    auto dstAttr = builder.getIntegerAttr(dstType, srcAttr.getInt());
    LLVM_DEBUG(llvm::dbgs() << "attribute '" << srcAttr << "' converted to '"
                            << dstAttr << "' for type '" << dstType << "'\n");
    return dstAttr;
  }

  LLVM_DEBUG(llvm::dbgs() << "attribute '" << srcAttr
                          << "' illegal: cannot fit into target type '"
                          << dstType << "'\n");
  return IntegerAttr();
}

/// Converts the given `srcAttr` to a new attribute of the given `dstType`.
/// Returns null attribute if `dstType` is not 32-bit or conversion fails.
static FloatAttr convertFloatAttr(FloatAttr srcAttr, FloatType dstType,
                                  Builder builder) {
  // Only support converting to float for now.
  if (!dstType.isF32())
    return FloatAttr();

  // Try to convert the source floating-point number to single precision.
  APFloat dstVal = srcAttr.getValue();
  bool losesInfo = false;
  APFloat::opStatus status =
      dstVal.convert(APFloat::IEEEsingle(), APFloat::rmTowardZero, &losesInfo);
  if (status != APFloat::opOK || losesInfo) {
    LLVM_DEBUG(llvm::dbgs()
               << srcAttr << " illegal: cannot fit into converted type '"
               << dstType << "'\n");
    return FloatAttr();
  }

  return builder.getF32FloatAttr(dstVal.convertToFloat());
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts composite std.constant operation to spv.Constant.
class ConstantCompositeOpPattern final
    : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts scalar std.constant operation to spv.Constant.
class ConstantScalarOpPattern final : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.return to spv.Return.
class ReturnOpPattern final : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.select to spv.Select.
class SelectOpPattern final : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.splat to spv.CompositeConstruct.
class SplatPattern final : public OpConversionPattern<SplatOp> {
public:
  using OpConversionPattern<SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

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
    TensorType tensorType = extractOp.tensor().getType().cast<TensorType>();

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

    Type varType = spirv::PointerType::get(adaptor.tensor().getType(),
                                           spirv::StorageClass::Function);

    spirv::VariableOp varOp;
    if (adaptor.tensor().getDefiningOp<spirv::ConstantOp>()) {
      varOp = rewriter.create<spirv::VariableOp>(
          loc, varType, spirv::StorageClass::Function,
          /*initializer=*/adaptor.tensor());
    } else {
      // Need to store the value to the local variable. It's questionable
      // whether we want to support such case though.
      return failure();
    }

    auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter.getIndexType();

    Value index = spirv::linearizeIndex(adaptor.indices(), strides,
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
// ConstantOp with composite type.
//===----------------------------------------------------------------------===//

// TODO: This probably should be split into the vector case and tensor case,
// so that the tensor case can be moved to TensorToSPIRV conversion. But,
// std.constant is for the standard dialect though.
LogicalResult ConstantCompositeOpPattern::matchAndRewrite(
    ConstantOp constOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto srcType = constOp.getType().dyn_cast<ShapedType>();
  if (!srcType)
    return failure();

  // std.constant should only have vector or tenor types.
  assert((srcType.isa<VectorType, RankedTensorType>()));

  auto dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  auto dstElementsAttr = constOp.value().dyn_cast<DenseElementsAttr>();
  ShapedType dstAttrType = dstElementsAttr.getType();
  if (!dstElementsAttr)
    return failure();

  // If the composite type has more than one dimensions, perform linearization.
  if (srcType.getRank() > 1) {
    if (srcType.isa<RankedTensorType>()) {
      dstAttrType = RankedTensorType::get(srcType.getNumElements(),
                                          srcType.getElementType());
      dstElementsAttr = dstElementsAttr.reshape(dstAttrType);
    } else {
      // TODO: add support for large vectors.
      return failure();
    }
  }

  Type srcElemType = srcType.getElementType();
  Type dstElemType;
  // Tensor types are converted to SPIR-V array types; vector types are
  // converted to SPIR-V vector/array types.
  if (auto arrayType = dstType.dyn_cast<spirv::ArrayType>())
    dstElemType = arrayType.getElementType();
  else
    dstElemType = dstType.cast<VectorType>().getElementType();

  // If the source and destination element types are different, perform
  // attribute conversion.
  if (srcElemType != dstElemType) {
    SmallVector<Attribute, 8> elements;
    if (srcElemType.isa<FloatType>()) {
      for (FloatAttr srcAttr : dstElementsAttr.getValues<FloatAttr>()) {
        FloatAttr dstAttr =
            convertFloatAttr(srcAttr, dstElemType.cast<FloatType>(), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    } else if (srcElemType.isInteger(1)) {
      return failure();
    } else {
      for (IntegerAttr srcAttr : dstElementsAttr.getValues<IntegerAttr>()) {
        IntegerAttr dstAttr = convertIntegerAttr(
            srcAttr, dstElemType.cast<IntegerType>(), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    }

    // Unfortunately, we cannot use dialect-specific types for element
    // attributes; element attributes only works with builtin types. So we need
    // to prepare another converted builtin types for the destination elements
    // attribute.
    if (dstAttrType.isa<RankedTensorType>())
      dstAttrType = RankedTensorType::get(dstAttrType.getShape(), dstElemType);
    else
      dstAttrType = VectorType::get(dstAttrType.getShape(), dstElemType);

    dstElementsAttr = DenseElementsAttr::get(dstAttrType, elements);
  }

  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType,
                                                 dstElementsAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp with scalar type.
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOpPattern::matchAndRewrite(
    ConstantOp constOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type srcType = constOp.getType();
  if (!srcType.isIntOrIndexOrFloat())
    return failure();

  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  // Floating-point types.
  if (srcType.isa<FloatType>()) {
    auto srcAttr = constOp.value().cast<FloatAttr>();
    auto dstAttr = srcAttr;

    // Floating-point types not supported in the target environment are all
    // converted to float type.
    if (srcType != dstType) {
      dstAttr = convertFloatAttr(srcAttr, dstType.cast<FloatType>(), rewriter);
      if (!dstAttr)
        return failure();
    }

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
    return success();
  }

  // Bool type.
  if (srcType.isInteger(1)) {
    // std.constant can use 0/1 instead of true/false for i1 values. We need to
    // handle that here.
    auto dstAttr = convertBoolAttr(constOp.value(), rewriter);
    if (!dstAttr)
      return failure();
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
    return success();
  }

  // IndexType or IntegerType. Index values are converted to 32-bit integer
  // values when converting to SPIR-V.
  auto srcAttr = constOp.value().cast<IntegerAttr>();
  auto dstAttr =
      convertIntegerAttr(srcAttr, dstType.cast<IntegerType>(), rewriter);
  if (!dstAttr)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(ReturnOp returnOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  if (returnOp.getNumOperands() > 1)
    return failure();

  if (returnOp.getNumOperands() == 1) {
    rewriter.replaceOpWithNewOp<spirv::ReturnValueOp>(returnOp,
                                                      adaptor.getOperands()[0]);
  } else {
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(
      op, adaptor.condition(), adaptor.true_value(), adaptor.false_value());
  return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

LogicalResult
SplatPattern::matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  auto dstVecType = op.getType().dyn_cast<VectorType>();
  if (!dstVecType || !spirv::CompositeType::isValid(dstVecType))
    return failure();
  SmallVector<Value, 4> source(dstVecType.getNumElements(), adaptor.input());
  rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, dstVecType,
                                                           source);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateStandardToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<
      // Unary and binary patterns
      spirv::UnaryAndBinaryOpPattern<MaxFOp, spirv::GLSLFMaxOp>,
      spirv::UnaryAndBinaryOpPattern<MaxSIOp, spirv::GLSLSMaxOp>,
      spirv::UnaryAndBinaryOpPattern<MaxUIOp, spirv::GLSLUMaxOp>,
      spirv::UnaryAndBinaryOpPattern<MinFOp, spirv::GLSLFMinOp>,
      spirv::UnaryAndBinaryOpPattern<MinSIOp, spirv::GLSLSMinOp>,
      spirv::UnaryAndBinaryOpPattern<MinUIOp, spirv::GLSLUMinOp>,

      // Constant patterns
      ConstantCompositeOpPattern, ConstantScalarOpPattern,

      ReturnOpPattern, SelectOpPattern, SplatPattern>(typeConverter, context);
}

void populateTensorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   int64_t byteCountThreshold,
                                   RewritePatternSet &patterns) {
  patterns.add<TensorExtractPattern>(typeConverter, patterns.getContext(),
                                     byteCountThreshold);
}

} // namespace mlir
