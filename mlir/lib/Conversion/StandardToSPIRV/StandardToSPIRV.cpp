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

/// Returns true if the given `type` is a boolean scalar or vector type.
static bool isBoolScalarOrVector(Type type) {
  if (type.isInteger(1))
    return true;
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getElementType().isInteger(1);
  return false;
}

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

/// Returns signed remainder for `lhs` and `rhs` and lets the result follow
/// the sign of `signOperand`.
///
/// Note that this is needed for Vulkan. Per the Vulkan's SPIR-V environment
/// spec, "for the OpSRem and OpSMod instructions, if either operand is negative
/// the result is undefined."  So we cannot directly use spv.SRem/spv.SMod
/// if either operand can be negative. Emulate it via spv.UMod.
static Value emulateSignedRemainder(Location loc, Value lhs, Value rhs,
                                    Value signOperand, OpBuilder &builder) {
  assert(lhs.getType() == rhs.getType());
  assert(lhs == signOperand || rhs == signOperand);

  Type type = lhs.getType();

  // Calculate the remainder with spv.UMod.
  Value lhsAbs = builder.create<spirv::GLSLSAbsOp>(loc, type, lhs);
  Value rhsAbs = builder.create<spirv::GLSLSAbsOp>(loc, type, rhs);
  Value abs = builder.create<spirv::UModOp>(loc, lhsAbs, rhsAbs);

  // Fix the sign.
  Value isPositive;
  if (lhs == signOperand)
    isPositive = builder.create<spirv::IEqualOp>(loc, lhs, lhsAbs);
  else
    isPositive = builder.create<spirv::IEqualOp>(loc, rhs, rhsAbs);
  Value absNegate = builder.create<spirv::SNegateOp>(loc, type, abs);
  return builder.create<spirv::SelectOp>(loc, type, isPositive, abs, absNegate);
}

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

/// Converts std.remi_signed to SPIR-V ops.
///
/// This cannot be merged into the template unary/binary pattern due to
/// Vulkan restrictions over spv.SRem and spv.SMod.
class SignedRemIOpPattern final : public OpConversionPattern<SignedRemIOp> {
public:
  using OpConversionPattern<SignedRemIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SignedRemIOp remOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts bitwise standard operations to SPIR-V operations. This is a special
/// pattern other than the BinaryOpPatternPattern because if the operands are
/// boolean values, SPIR-V uses different operations (`SPIRVLogicalOp`). For
/// non-boolean operands, SPIR-V should use `SPIRVBitwiseOp`.
template <typename StdOp, typename SPIRVLogicalOp, typename SPIRVBitwiseOp>
class BitwiseOpPattern final : public OpConversionPattern<StdOp> {
public:
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 2);
    auto dstType =
        this->getTypeConverter()->convertType(operation.getResult().getType());
    if (!dstType)
      return failure();
    if (isBoolScalarOrVector(operands.front().getType())) {
      rewriter.template replaceOpWithNewOp<SPIRVLogicalOp>(operation, dstType,
                                                           operands);
    } else {
      rewriter.template replaceOpWithNewOp<SPIRVBitwiseOp>(operation, dstType,
                                                           operands);
    }
    return success();
  }
};

/// Converts composite std.constant operation to spv.Constant.
class ConstantCompositeOpPattern final
    : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts scalar std.constant operation to spv.Constant.
class ConstantScalarOpPattern final : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating-point comparison operations to SPIR-V ops.
class CmpFOpPattern final : public OpConversionPattern<CmpFOp> {
public:
  using OpConversionPattern<CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern requires
/// Kernel capability.
class CmpFOpNanKernelPattern final : public OpConversionPattern<CmpFOp> {
public:
  using OpConversionPattern<CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern does not
/// require additional capability.
class CmpFOpNanNonePattern final : public OpConversionPattern<CmpFOp> {
public:
  using OpConversionPattern<CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation on i1 type operands to SPIR-V ops.
class BoolCmpIOpPattern final : public OpConversionPattern<CmpIOp> {
public:
  using OpConversionPattern<CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation to SPIR-V ops.
class CmpIOpPattern final : public OpConversionPattern<CmpIOp> {
public:
  using OpConversionPattern<CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.return to spv.Return.
class ReturnOpPattern final : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.select to spv.Select.
class SelectOpPattern final : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.splat to spv.CompositeConstruct.
class SplatPattern final : public OpConversionPattern<SplatOp> {
public:
  using OpConversionPattern<SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.zexti to spv.Select if the type of source is i1 or vector of
/// i1.
class ZeroExtendI1Pattern final : public OpConversionPattern<ZeroExtendIOp> {
public:
  using OpConversionPattern<ZeroExtendIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZeroExtendIOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = operands.front().getType();
    if (!isBoolScalarOrVector(srcType))
      return failure();

    auto dstType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Location loc = op.getLoc();
    Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
    Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.template replaceOpWithNewOp<spirv::SelectOp>(
        op, dstType, operands.front(), one, zero);
    return success();
  }
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
  matchAndRewrite(tensor::ExtractOp extractOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorType tensorType = extractOp.tensor().getType().cast<TensorType>();

    if (!tensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(extractOp, "non-static tensor");

    if (tensorType.getNumElements() * tensorType.getElementTypeBitWidth() >
        byteCountThreshold * 8)
      return rewriter.notifyMatchFailure(extractOp,
                                         "exceeding byte count threshold");

    Location loc = extractOp.getLoc();
    tensor::ExtractOp::Adaptor adaptor(operands);

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

    Value index = spirv::linearizeIndex(adaptor.indices(), strides,
                                        /*offset=*/0, loc, rewriter);
    auto acOp = rewriter.create<spirv::AccessChainOp>(loc, varOp, index);

    rewriter.replaceOpWithNewOp<spirv::LoadOp>(extractOp, acOp);

    return success();
  }

private:
  int64_t byteCountThreshold;
};

/// Converts std.trunci to spv.Select if the type of result is i1 or vector of
/// i1.
class TruncI1Pattern final : public OpConversionPattern<TruncateIOp> {
public:
  using OpConversionPattern<TruncateIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncateIOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!isBoolScalarOrVector(dstType))
      return failure();

    Location loc = op.getLoc();
    auto srcType = operands.front().getType();
    // Check if (x & 1) == 1.
    Value mask = spirv::ConstantOp::getOne(srcType, loc, rewriter);
    Value maskedSrc =
        rewriter.create<spirv::BitwiseAndOp>(loc, srcType, operands[0], mask);
    Value isOne = rewriter.create<spirv::IEqualOp>(loc, maskedSrc, mask);

    Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
    Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, dstType, isOne, one, zero);
    return success();
  }
};

/// Converts std.uitofp to spv.Select if the type of source is i1 or vector of
/// i1.
class UIToFPI1Pattern final : public OpConversionPattern<UIToFPOp> {
public:
  using OpConversionPattern<UIToFPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UIToFPOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = operands.front().getType();
    if (!isBoolScalarOrVector(srcType))
      return failure();

    auto dstType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Location loc = op.getLoc();
    Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
    Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.template replaceOpWithNewOp<spirv::SelectOp>(
        op, dstType, operands.front(), one, zero);
    return success();
  }
};

/// Converts type-casting standard operations to SPIR-V operations.
template <typename StdOp, typename SPIRVOp>
class TypeCastingOpPattern final : public OpConversionPattern<StdOp> {
public:
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    auto srcType = operands.front().getType();
    auto dstType =
        this->getTypeConverter()->convertType(operation.getResult().getType());
    if (isBoolScalarOrVector(srcType) || isBoolScalarOrVector(dstType))
      return failure();
    if (dstType == srcType) {
      // Due to type conversion, we are seeing the same source and target type.
      // Then we can just erase this operation by forwarding its operand.
      rewriter.replaceOp(operation, operands.front());
    } else {
      rewriter.template replaceOpWithNewOp<SPIRVOp>(operation, dstType,
                                                    operands);
    }
    return success();
  }
};

/// Converts std.xor to SPIR-V operations.
class XOrOpPattern final : public OpConversionPattern<XOrOp> {
public:
  using OpConversionPattern<XOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.xor to SPIR-V operations if the type of source is i1 or vector
/// of i1.
class BoolXOrOpPattern final : public OpConversionPattern<XOrOp> {
public:
  using OpConversionPattern<XOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// SignedRemIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult SignedRemIOpPattern::matchAndRewrite(
    SignedRemIOp remOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Value result = emulateSignedRemainder(remOp.getLoc(), operands[0],
                                        operands[1], operands[0], rewriter);
  rewriter.replaceOp(remOp, result);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp with composite type.
//===----------------------------------------------------------------------===//

// TODO: This probably should be split into the vector case and tensor case,
// so that the tensor case can be moved to TensorToSPIRV conversion. But,
// std.constant is for the standard dialect though.
LogicalResult ConstantCompositeOpPattern::matchAndRewrite(
    ConstantOp constOp, ArrayRef<Value> operands,
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
      for (Attribute srcAttr : dstElementsAttr.getAttributeValues()) {
        FloatAttr dstAttr = convertFloatAttr(
            srcAttr.cast<FloatAttr>(), dstElemType.cast<FloatType>(), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    } else if (srcElemType.isInteger(1)) {
      return failure();
    } else {
      for (Attribute srcAttr : dstElementsAttr.getAttributeValues()) {
        IntegerAttr dstAttr =
            convertIntegerAttr(srcAttr.cast<IntegerAttr>(),
                               dstElemType.cast<IntegerType>(), rewriter);
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
    ConstantOp constOp, ArrayRef<Value> operands,
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
// CmpFOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOpPattern::matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  CmpFOpAdaptor cmpFOpOperands(operands);

  switch (cmpFOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpFOp, cmpFOp.getResult().getType(), \
                                         cmpFOpOperands.lhs(),                 \
                                         cmpFOpOperands.rhs());                \
    return success();

    // Ordered.
    DISPATCH(CmpFPredicate::OEQ, spirv::FOrdEqualOp);
    DISPATCH(CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
    DISPATCH(CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::OLT, spirv::FOrdLessThanOp);
    DISPATCH(CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
    DISPATCH(CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
    // Unordered.
    DISPATCH(CmpFPredicate::UEQ, spirv::FUnordEqualOp);
    DISPATCH(CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
    DISPATCH(CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::ULT, spirv::FUnordLessThanOp);
    DISPATCH(CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
    DISPATCH(CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

  default:
    break;
  }
  return failure();
}

LogicalResult CmpFOpNanKernelPattern::matchAndRewrite(
    CmpFOp cmpFOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  CmpFOpAdaptor cmpFOpOperands(operands);

  if (cmpFOp.getPredicate() == CmpFPredicate::ORD) {
    rewriter.replaceOpWithNewOp<spirv::OrderedOp>(cmpFOp, cmpFOpOperands.lhs(),
                                                  cmpFOpOperands.rhs());
    return success();
  }

  if (cmpFOp.getPredicate() == CmpFPredicate::UNO) {
    rewriter.replaceOpWithNewOp<spirv::UnorderedOp>(
        cmpFOp, cmpFOpOperands.lhs(), cmpFOpOperands.rhs());
    return success();
  }

  return failure();
}

LogicalResult CmpFOpNanNonePattern::matchAndRewrite(
    CmpFOp cmpFOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (cmpFOp.getPredicate() != CmpFPredicate::ORD &&
      cmpFOp.getPredicate() != CmpFPredicate::UNO)
    return failure();

  CmpFOpAdaptor cmpFOpOperands(operands);
  Location loc = cmpFOp.getLoc();

  Value lhsIsNan = rewriter.create<spirv::IsNanOp>(loc, cmpFOpOperands.lhs());
  Value rhsIsNan = rewriter.create<spirv::IsNanOp>(loc, cmpFOpOperands.rhs());

  Value replace = rewriter.create<spirv::LogicalOrOp>(loc, lhsIsNan, rhsIsNan);
  if (cmpFOp.getPredicate() == CmpFPredicate::ORD)
    replace = rewriter.create<spirv::LogicalNotOp>(loc, replace);

  rewriter.replaceOp(cmpFOp, replace);
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

LogicalResult
BoolCmpIOpPattern::matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  CmpIOpAdaptor cmpIOpOperands(operands);

  Type operandType = cmpIOp.lhs().getType();
  if (!isBoolScalarOrVector(operandType))
    return failure();

  switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpIOp, cmpIOp.getResult().getType(), \
                                         cmpIOpOperands.lhs(),                 \
                                         cmpIOpOperands.rhs());                \
    return success();

    DISPATCH(CmpIPredicate::eq, spirv::LogicalEqualOp);
    DISPATCH(CmpIPredicate::ne, spirv::LogicalNotEqualOp);

#undef DISPATCH
  default:;
  }
  return failure();
}

LogicalResult
CmpIOpPattern::matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  CmpIOpAdaptor cmpIOpOperands(operands);

  Type operandType = cmpIOp.lhs().getType();
  if (isBoolScalarOrVector(operandType))
    return failure();

  switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    if (spirvOp::template hasTrait<OpTrait::spirv::UnsignedOp>() &&            \
        operandType != this->getTypeConverter()->convertType(operandType)) {   \
      return cmpIOp.emitError(                                                 \
          "bitwidth emulation is not implemented yet on unsigned op");         \
    }                                                                          \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpIOp, cmpIOp.getResult().getType(), \
                                         cmpIOpOperands.lhs(),                 \
                                         cmpIOpOperands.rhs());                \
    return success();

    DISPATCH(CmpIPredicate::eq, spirv::IEqualOp);
    DISPATCH(CmpIPredicate::ne, spirv::INotEqualOp);
    DISPATCH(CmpIPredicate::slt, spirv::SLessThanOp);
    DISPATCH(CmpIPredicate::sle, spirv::SLessThanEqualOp);
    DISPATCH(CmpIPredicate::sgt, spirv::SGreaterThanOp);
    DISPATCH(CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
    DISPATCH(CmpIPredicate::ult, spirv::ULessThanOp);
    DISPATCH(CmpIPredicate::ule, spirv::ULessThanEqualOp);
    DISPATCH(CmpIPredicate::ugt, spirv::UGreaterThanOp);
    DISPATCH(CmpIPredicate::uge, spirv::UGreaterThanEqualOp);

#undef DISPATCH
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  if (returnOp.getNumOperands() > 1)
    return failure();

  if (returnOp.getNumOperands() == 1) {
    rewriter.replaceOpWithNewOp<spirv::ReturnValueOp>(returnOp, operands[0]);
  } else {
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  SelectOpAdaptor selectOperands(operands);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, selectOperands.condition(),
                                               selectOperands.true_value(),
                                               selectOperands.false_value());
  return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

LogicalResult
SplatPattern::matchAndRewrite(SplatOp op, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) const {
  auto dstVecType = op.getType().dyn_cast<VectorType>();
  if (!dstVecType || !spirv::CompositeType::isValid(dstVecType))
    return failure();
  SplatOp::Adaptor adaptor(operands);
  SmallVector<Value, 4> source(dstVecType.getNumElements(), adaptor.input());
  rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, dstVecType,
                                                           source);
  return success();
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

LogicalResult
XOrOpPattern::matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) const {
  assert(operands.size() == 2);

  if (isBoolScalarOrVector(operands.front().getType()))
    return failure();

  auto dstType = getTypeConverter()->convertType(xorOp.getType());
  if (!dstType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::BitwiseXorOp>(xorOp, dstType, operands);

  return success();
}

LogicalResult
BoolXOrOpPattern::matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  assert(operands.size() == 2);

  if (!isBoolScalarOrVector(operands.front().getType()))
    return failure();

  auto dstType = getTypeConverter()->convertType(xorOp.getType());
  if (!dstType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::LogicalNotEqualOp>(xorOp, dstType,
                                                        operands);
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
      BitwiseOpPattern<AndOp, spirv::LogicalAndOp, spirv::BitwiseAndOp>,
      BitwiseOpPattern<OrOp, spirv::LogicalOrOp, spirv::BitwiseOrOp>,
      UnaryAndBinaryOpPattern<AbsFOp, spirv::GLSLFAbsOp>,
      UnaryAndBinaryOpPattern<AddFOp, spirv::FAddOp>,
      UnaryAndBinaryOpPattern<AddIOp, spirv::IAddOp>,
      UnaryAndBinaryOpPattern<CeilFOp, spirv::GLSLCeilOp>,
      UnaryAndBinaryOpPattern<DivFOp, spirv::FDivOp>,
      UnaryAndBinaryOpPattern<FloorFOp, spirv::GLSLFloorOp>,
      UnaryAndBinaryOpPattern<MulFOp, spirv::FMulOp>,
      UnaryAndBinaryOpPattern<MulIOp, spirv::IMulOp>,
      UnaryAndBinaryOpPattern<NegFOp, spirv::FNegateOp>,
      UnaryAndBinaryOpPattern<RemFOp, spirv::FRemOp>,
      UnaryAndBinaryOpPattern<ShiftLeftOp, spirv::ShiftLeftLogicalOp>,
      UnaryAndBinaryOpPattern<SignedDivIOp, spirv::SDivOp>,
      UnaryAndBinaryOpPattern<SignedShiftRightOp,
                              spirv::ShiftRightArithmeticOp>,
      UnaryAndBinaryOpPattern<SubIOp, spirv::ISubOp>,
      UnaryAndBinaryOpPattern<SubFOp, spirv::FSubOp>,
      UnaryAndBinaryOpPattern<UnsignedDivIOp, spirv::UDivOp>,
      UnaryAndBinaryOpPattern<UnsignedRemIOp, spirv::UModOp>,
      UnaryAndBinaryOpPattern<UnsignedShiftRightOp, spirv::ShiftRightLogicalOp>,
      SignedRemIOpPattern, XOrOpPattern, BoolXOrOpPattern,

      // Comparison patterns
      BoolCmpIOpPattern, CmpFOpPattern, CmpFOpNanNonePattern, CmpIOpPattern,

      // Constant patterns
      ConstantCompositeOpPattern, ConstantScalarOpPattern,

      ReturnOpPattern, SelectOpPattern, SplatPattern,

      // Type cast patterns
      UIToFPI1Pattern, ZeroExtendI1Pattern, TruncI1Pattern,
      TypeCastingOpPattern<IndexCastOp, spirv::SConvertOp>,
      TypeCastingOpPattern<SIToFPOp, spirv::ConvertSToFOp>,
      TypeCastingOpPattern<UIToFPOp, spirv::ConvertUToFOp>,
      TypeCastingOpPattern<SignExtendIOp, spirv::SConvertOp>,
      TypeCastingOpPattern<ZeroExtendIOp, spirv::UConvertOp>,
      TypeCastingOpPattern<TruncateIOp, spirv::SConvertOp>,
      TypeCastingOpPattern<FPToSIOp, spirv::ConvertFToSOp>,
      TypeCastingOpPattern<FPExtOp, spirv::FConvertOp>,
      TypeCastingOpPattern<FPTruncOp, spirv::FConvertOp>>(typeConverter,
                                                          context);

  // Give CmpFOpNanKernelPattern a higher benefit so it can prevail when Kernel
  // capability is available.
  patterns.add<CmpFOpNanKernelPattern>(typeConverter, context,
                                       /*benefit=*/2);
}

void populateTensorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   int64_t byteCountThreshold,
                                   RewritePatternSet &patterns) {
  patterns.add<TensorExtractPattern>(typeConverter, patterns.getContext(),
                                     byteCountThreshold);
}

} // namespace mlir
