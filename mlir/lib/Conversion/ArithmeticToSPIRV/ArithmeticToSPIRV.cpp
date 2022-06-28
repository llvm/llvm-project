//===- ArithmeticToSPIRV.cpp - Arithmetic to SPIRV dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h"
#include "../PassDetail.h"
#include "../SPIRVCommon/Pattern.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arith-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation Conversion
//===----------------------------------------------------------------------===//

namespace {

/// Converts composite arith.constant operation to spv.Constant.
struct ConstantCompositeOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts scalar arith.constant operation to spv.Constant.
struct ConstantScalarOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.remsi to GLSL SPIR-V ops.
///
/// This cannot be merged into the template unary/binary pattern due to Vulkan
/// restrictions over spv.SRem and spv.SMod.
struct RemSIOpGLSLPattern final : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.remsi to OpenCL SPIR-V ops.
struct RemSIOpOCLPattern final : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts bitwise operations to SPIR-V operations. This is a special pattern
/// other than the BinaryOpPatternPattern because if the operands are boolean
/// values, SPIR-V uses different operations (`SPIRVLogicalOp`). For
/// non-boolean operands, SPIR-V should use `SPIRVBitwiseOp`.
template <typename Op, typename SPIRVLogicalOp, typename SPIRVBitwiseOp>
struct BitwiseOpPattern final : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.xori to SPIR-V operations.
struct XOrIOpLogicalPattern final : public OpConversionPattern<arith::XOrIOp> {
  using OpConversionPattern<arith::XOrIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.xori to SPIR-V operations if the type of source is i1 or
/// vector of i1.
struct XOrIOpBooleanPattern final : public OpConversionPattern<arith::XOrIOp> {
  using OpConversionPattern<arith::XOrIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.uitofp to spv.Select if the type of source is i1 or vector of
/// i1.
struct UIToFPI1Pattern final : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern<arith::UIToFPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.extui to spv.Select if the type of source is i1 or vector of
/// i1.
struct ExtUII1Pattern final : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern<arith::ExtUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.trunci to spv.Select if the type of result is i1 or vector of
/// i1.
struct TruncII1Pattern final : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts type-casting standard operations to SPIR-V operations.
template <typename Op, typename SPIRVOp>
struct TypeCastingOpPattern final : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation on i1 type operands to SPIR-V ops.
class CmpIOpBooleanPattern final : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation to SPIR-V ops.
class CmpIOpPattern final : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating-point comparison operations to SPIR-V ops.
class CmpFOpPattern final : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern requires
/// Kernel capability.
class CmpFOpNanKernelPattern final : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern does not
/// require additional capability.
class CmpFOpNanNonePattern final : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.select to spv.Select.
class SelectOpPattern final : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Helpers
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

/// Returns true if the given `type` is a boolean scalar or vector type.
static bool isBoolScalarOrVector(Type type) {
  if (type.isInteger(1))
    return true;
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getElementType().isInteger(1);
  return false;
}

/// Returns true if scalar/vector type `a` and `b` have the same number of
/// bitwidth.
static bool hasSameBitwidth(Type a, Type b) {
  auto getNumBitwidth = [](Type type) {
    unsigned bw = 0;
    if (type.isIntOrFloat())
      bw = type.getIntOrFloatBitWidth();
    else if (auto vecType = type.dyn_cast<VectorType>())
      bw = vecType.getElementTypeBitWidth() * vecType.getNumElements();
    return bw;
  };
  unsigned aBW = getNumBitwidth(a);
  unsigned bBW = getNumBitwidth(b);
  return aBW != 0 && bBW != 0 && aBW == bBW;
}

//===----------------------------------------------------------------------===//
// ConstantOp with composite type
//===----------------------------------------------------------------------===//

LogicalResult ConstantCompositeOpPattern::matchAndRewrite(
    arith::ConstantOp constOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto srcType = constOp.getType().dyn_cast<ShapedType>();
  if (!srcType || srcType.getNumElements() == 1)
    return failure();

  // arith.constant should only have vector or tenor types.
  assert((srcType.isa<VectorType, RankedTensorType>()));

  auto dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  auto dstElementsAttr = constOp.getValue().dyn_cast<DenseElementsAttr>();
  if (!dstElementsAttr)
    return failure();

  ShapedType dstAttrType = dstElementsAttr.getType();

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
// ConstantOp with scalar type
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOpPattern::matchAndRewrite(
    arith::ConstantOp constOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type srcType = constOp.getType();
  if (auto shapedType = srcType.dyn_cast<ShapedType>()) {
    if (shapedType.getNumElements() != 1)
      return failure();
    srcType = shapedType.getElementType();
  }
  if (!srcType.isIntOrIndexOrFloat())
    return failure();

  Attribute cstAttr = constOp.getValue();
  if (cstAttr.getType().isa<ShapedType>())
    cstAttr = cstAttr.cast<DenseElementsAttr>().getSplatValue<Attribute>();

  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  // Floating-point types.
  if (srcType.isa<FloatType>()) {
    auto srcAttr = cstAttr.cast<FloatAttr>();
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
    // arith.constant can use 0/1 instead of true/false for i1 values. We need
    // to handle that here.
    auto dstAttr = convertBoolAttr(cstAttr, rewriter);
    if (!dstAttr)
      return failure();
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
    return success();
  }

  // IndexType or IntegerType. Index values are converted to 32-bit integer
  // values when converting to SPIR-V.
  auto srcAttr = cstAttr.cast<IntegerAttr>();
  auto dstAttr =
      convertIntegerAttr(srcAttr, dstType.cast<IntegerType>(), rewriter);
  if (!dstAttr)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// RemSIOpGLSLPattern
//===----------------------------------------------------------------------===//

/// Returns signed remainder for `lhs` and `rhs` and lets the result follow
/// the sign of `signOperand`.
///
/// Note that this is needed for Vulkan. Per the Vulkan's SPIR-V environment
/// spec, "for the OpSRem and OpSMod instructions, if either operand is negative
/// the result is undefined."  So we cannot directly use spv.SRem/spv.SMod
/// if either operand can be negative. Emulate it via spv.UMod.
template <typename SignedAbsOp>
static Value emulateSignedRemainder(Location loc, Value lhs, Value rhs,
                                    Value signOperand, OpBuilder &builder) {
  assert(lhs.getType() == rhs.getType());
  assert(lhs == signOperand || rhs == signOperand);

  Type type = lhs.getType();

  // Calculate the remainder with spv.UMod.
  Value lhsAbs = builder.create<SignedAbsOp>(loc, type, lhs);
  Value rhsAbs = builder.create<SignedAbsOp>(loc, type, rhs);
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

LogicalResult
RemSIOpGLSLPattern::matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Value result = emulateSignedRemainder<spirv::GLSLSAbsOp>(
      op.getLoc(), adaptor.getOperands()[0], adaptor.getOperands()[1],
      adaptor.getOperands()[0], rewriter);
  rewriter.replaceOp(op, result);

  return success();
}

//===----------------------------------------------------------------------===//
// RemSIOpOCLPattern
//===----------------------------------------------------------------------===//

LogicalResult
RemSIOpOCLPattern::matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Value result = emulateSignedRemainder<spirv::OCLSAbsOp>(
      op.getLoc(), adaptor.getOperands()[0], adaptor.getOperands()[1],
      adaptor.getOperands()[0], rewriter);
  rewriter.replaceOp(op, result);

  return success();
}

//===----------------------------------------------------------------------===//
// BitwiseOpPattern
//===----------------------------------------------------------------------===//

template <typename Op, typename SPIRVLogicalOp, typename SPIRVBitwiseOp>
LogicalResult
BitwiseOpPattern<Op, SPIRVLogicalOp, SPIRVBitwiseOp>::matchAndRewrite(
    Op op, typename Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  assert(adaptor.getOperands().size() == 2);
  auto dstType =
      this->getTypeConverter()->convertType(op.getResult().getType());
  if (!dstType)
    return failure();
  if (isBoolScalarOrVector(adaptor.getOperands().front().getType())) {
    rewriter.template replaceOpWithNewOp<SPIRVLogicalOp>(op, dstType,
                                                         adaptor.getOperands());
  } else {
    rewriter.template replaceOpWithNewOp<SPIRVBitwiseOp>(op, dstType,
                                                         adaptor.getOperands());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XOrIOpLogicalPattern
//===----------------------------------------------------------------------===//

LogicalResult XOrIOpLogicalPattern::matchAndRewrite(
    arith::XOrIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  assert(adaptor.getOperands().size() == 2);

  if (isBoolScalarOrVector(adaptor.getOperands().front().getType()))
    return failure();

  auto dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::BitwiseXorOp>(op, dstType,
                                                   adaptor.getOperands());

  return success();
}

//===----------------------------------------------------------------------===//
// XOrIOpBooleanPattern
//===----------------------------------------------------------------------===//

LogicalResult XOrIOpBooleanPattern::matchAndRewrite(
    arith::XOrIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  assert(adaptor.getOperands().size() == 2);

  if (!isBoolScalarOrVector(adaptor.getOperands().front().getType()))
    return failure();

  auto dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::LogicalNotEqualOp>(op, dstType,
                                                        adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// UIToFPI1Pattern
//===----------------------------------------------------------------------===//

LogicalResult
UIToFPI1Pattern::matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto srcType = adaptor.getOperands().front().getType();
  if (!isBoolScalarOrVector(srcType))
    return failure();

  auto dstType =
      this->getTypeConverter()->convertType(op.getResult().getType());
  Location loc = op.getLoc();
  Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
  Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
  rewriter.template replaceOpWithNewOp<spirv::SelectOp>(
      op, dstType, adaptor.getOperands().front(), one, zero);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtUII1Pattern
//===----------------------------------------------------------------------===//

LogicalResult
ExtUII1Pattern::matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto srcType = adaptor.getOperands().front().getType();
  if (!isBoolScalarOrVector(srcType))
    return failure();

  auto dstType =
      this->getTypeConverter()->convertType(op.getResult().getType());
  Location loc = op.getLoc();
  Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
  Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
  rewriter.template replaceOpWithNewOp<spirv::SelectOp>(
      op, dstType, adaptor.getOperands().front(), one, zero);
  return success();
}

//===----------------------------------------------------------------------===//
// TruncII1Pattern
//===----------------------------------------------------------------------===//

LogicalResult
TruncII1Pattern::matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto dstType =
      this->getTypeConverter()->convertType(op.getResult().getType());
  if (!isBoolScalarOrVector(dstType))
    return failure();

  Location loc = op.getLoc();
  auto srcType = adaptor.getOperands().front().getType();
  // Check if (x & 1) == 1.
  Value mask = spirv::ConstantOp::getOne(srcType, loc, rewriter);
  Value maskedSrc = rewriter.create<spirv::BitwiseAndOp>(
      loc, srcType, adaptor.getOperands()[0], mask);
  Value isOne = rewriter.create<spirv::IEqualOp>(loc, maskedSrc, mask);

  Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
  Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, dstType, isOne, one, zero);
  return success();
}

//===----------------------------------------------------------------------===//
// TypeCastingOpPattern
//===----------------------------------------------------------------------===//

template <typename Op, typename SPIRVOp>
LogicalResult TypeCastingOpPattern<Op, SPIRVOp>::matchAndRewrite(
    Op op, typename Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  assert(adaptor.getOperands().size() == 1);
  auto srcType = adaptor.getOperands().front().getType();
  auto dstType =
      this->getTypeConverter()->convertType(op.getResult().getType());
  if (isBoolScalarOrVector(srcType) || isBoolScalarOrVector(dstType))
    return failure();
  if (dstType == srcType) {
    // Due to type conversion, we are seeing the same source and target type.
    // Then we can just erase this operation by forwarding its operand.
    rewriter.replaceOp(op, adaptor.getOperands().front());
  } else {
    rewriter.template replaceOpWithNewOp<SPIRVOp>(op, dstType,
                                                  adaptor.getOperands());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOpBooleanPattern
//===----------------------------------------------------------------------===//

LogicalResult CmpIOpBooleanPattern::matchAndRewrite(
    arith::CmpIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type srcType = op.getLhs().getType();
  if (!isBoolScalarOrVector(srcType))
    return failure();
  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq: {
    rewriter.replaceOpWithNewOp<spirv::LogicalEqualOp>(op, adaptor.getLhs(),
                                                       adaptor.getRhs());
    return success();
  }
  case arith::CmpIPredicate::ne: {
    rewriter.replaceOpWithNewOp<spirv::LogicalNotEqualOp>(op, adaptor.getLhs(),
                                                          adaptor.getRhs());
    return success();
  }
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ult: {
    // There are no direct corresponding instructions in SPIR-V for such cases.
    // Extend them to 32-bit and do comparision then.
    Type type = rewriter.getI32Type();
    if (auto vectorType = dstType.dyn_cast<VectorType>())
      type = VectorType::get(vectorType.getShape(), type);
    auto extLhs =
        rewriter.create<arith::ExtUIOp>(op.getLoc(), type, adaptor.getLhs());
    auto extRhs =
        rewriter.create<arith::ExtUIOp>(op.getLoc(), type, adaptor.getRhs());

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, op.getPredicate(), extLhs,
                                               extRhs);
    return success();
  }
  default:
    break;
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// CmpIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
CmpIOpPattern::matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Type srcType = op.getLhs().getType();
  if (isBoolScalarOrVector(srcType))
    return failure();
  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  switch (op.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    if (spirvOp::template hasTrait<OpTrait::spirv::UnsignedOp>() &&            \
        srcType != dstType && !hasSameBitwidth(srcType, dstType)) {            \
      return op.emitError(                                                     \
          "bitwidth emulation is not implemented yet on unsigned op");         \
    }                                                                          \
    rewriter.replaceOpWithNewOp<spirvOp>(op, adaptor.getLhs(),                 \
                                         adaptor.getRhs());                    \
    return success();

    DISPATCH(arith::CmpIPredicate::eq, spirv::IEqualOp);
    DISPATCH(arith::CmpIPredicate::ne, spirv::INotEqualOp);
    DISPATCH(arith::CmpIPredicate::slt, spirv::SLessThanOp);
    DISPATCH(arith::CmpIPredicate::sle, spirv::SLessThanEqualOp);
    DISPATCH(arith::CmpIPredicate::sgt, spirv::SGreaterThanOp);
    DISPATCH(arith::CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
    DISPATCH(arith::CmpIPredicate::ult, spirv::ULessThanOp);
    DISPATCH(arith::CmpIPredicate::ule, spirv::ULessThanEqualOp);
    DISPATCH(arith::CmpIPredicate::ugt, spirv::UGreaterThanOp);
    DISPATCH(arith::CmpIPredicate::uge, spirv::UGreaterThanEqualOp);

#undef DISPATCH
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// CmpFOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOpPattern::matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  switch (op.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(op, adaptor.getLhs(),                 \
                                         adaptor.getRhs());                    \
    return success();

    // Ordered.
    DISPATCH(arith::CmpFPredicate::OEQ, spirv::FOrdEqualOp);
    DISPATCH(arith::CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
    DISPATCH(arith::CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
    DISPATCH(arith::CmpFPredicate::OLT, spirv::FOrdLessThanOp);
    DISPATCH(arith::CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
    DISPATCH(arith::CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
    // Unordered.
    DISPATCH(arith::CmpFPredicate::UEQ, spirv::FUnordEqualOp);
    DISPATCH(arith::CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
    DISPATCH(arith::CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
    DISPATCH(arith::CmpFPredicate::ULT, spirv::FUnordLessThanOp);
    DISPATCH(arith::CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
    DISPATCH(arith::CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

  default:
    break;
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// CmpFOpNanKernelPattern
//===----------------------------------------------------------------------===//

LogicalResult CmpFOpNanKernelPattern::matchAndRewrite(
    arith::CmpFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (op.getPredicate() == arith::CmpFPredicate::ORD) {
    rewriter.replaceOpWithNewOp<spirv::OrderedOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    return success();
  }

  if (op.getPredicate() == arith::CmpFPredicate::UNO) {
    rewriter.replaceOpWithNewOp<spirv::UnorderedOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CmpFOpNanNonePattern
//===----------------------------------------------------------------------===//

LogicalResult CmpFOpNanNonePattern::matchAndRewrite(
    arith::CmpFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (op.getPredicate() != arith::CmpFPredicate::ORD &&
      op.getPredicate() != arith::CmpFPredicate::UNO)
    return failure();

  Location loc = op.getLoc();

  Value lhsIsNan = rewriter.create<spirv::IsNanOp>(loc, adaptor.getLhs());
  Value rhsIsNan = rewriter.create<spirv::IsNanOp>(loc, adaptor.getRhs());

  Value replace = rewriter.create<spirv::LogicalOrOp>(loc, lhsIsNan, rhsIsNan);
  if (op.getPredicate() == arith::CmpFPredicate::ORD)
    replace = rewriter.create<spirv::LogicalNotOp>(loc, replace);

  rewriter.replaceOp(op, replace);
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, adaptor.getCondition(),
                                               adaptor.getTrueValue(),
                                               adaptor.getFalseValue());
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithmeticToSPIRVPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    ConstantCompositeOpPattern,
    ConstantScalarOpPattern,
    spirv::ElementwiseOpPattern<arith::AddIOp, spirv::IAddOp>,
    spirv::ElementwiseOpPattern<arith::SubIOp, spirv::ISubOp>,
    spirv::ElementwiseOpPattern<arith::MulIOp, spirv::IMulOp>,
    spirv::ElementwiseOpPattern<arith::DivUIOp, spirv::UDivOp>,
    spirv::ElementwiseOpPattern<arith::DivSIOp, spirv::SDivOp>,
    spirv::ElementwiseOpPattern<arith::RemUIOp, spirv::UModOp>,
    RemSIOpGLSLPattern, RemSIOpOCLPattern,
    BitwiseOpPattern<arith::AndIOp, spirv::LogicalAndOp, spirv::BitwiseAndOp>,
    BitwiseOpPattern<arith::OrIOp, spirv::LogicalOrOp, spirv::BitwiseOrOp>,
    XOrIOpLogicalPattern, XOrIOpBooleanPattern,
    spirv::ElementwiseOpPattern<arith::ShLIOp, spirv::ShiftLeftLogicalOp>,
    spirv::ElementwiseOpPattern<arith::ShRUIOp, spirv::ShiftRightLogicalOp>,
    spirv::ElementwiseOpPattern<arith::ShRSIOp, spirv::ShiftRightArithmeticOp>,
    spirv::ElementwiseOpPattern<arith::NegFOp, spirv::FNegateOp>,
    spirv::ElementwiseOpPattern<arith::AddFOp, spirv::FAddOp>,
    spirv::ElementwiseOpPattern<arith::SubFOp, spirv::FSubOp>,
    spirv::ElementwiseOpPattern<arith::MulFOp, spirv::FMulOp>,
    spirv::ElementwiseOpPattern<arith::DivFOp, spirv::FDivOp>,
    spirv::ElementwiseOpPattern<arith::RemFOp, spirv::FRemOp>,
    TypeCastingOpPattern<arith::ExtUIOp, spirv::UConvertOp>, ExtUII1Pattern,
    TypeCastingOpPattern<arith::ExtSIOp, spirv::SConvertOp>,
    TypeCastingOpPattern<arith::ExtFOp, spirv::FConvertOp>,
    TypeCastingOpPattern<arith::TruncIOp, spirv::SConvertOp>, TruncII1Pattern,
    TypeCastingOpPattern<arith::TruncFOp, spirv::FConvertOp>,
    TypeCastingOpPattern<arith::UIToFPOp, spirv::ConvertUToFOp>, UIToFPI1Pattern,
    TypeCastingOpPattern<arith::SIToFPOp, spirv::ConvertSToFOp>,
    TypeCastingOpPattern<arith::FPToSIOp, spirv::ConvertFToSOp>,
    TypeCastingOpPattern<arith::IndexCastOp, spirv::SConvertOp>,
    TypeCastingOpPattern<arith::BitcastOp, spirv::BitcastOp>,
    CmpIOpBooleanPattern, CmpIOpPattern,
    CmpFOpNanNonePattern, CmpFOpPattern,
    SelectOpPattern,

    spirv::ElementwiseOpPattern<arith::MaxFOp, spirv::GLSLFMaxOp>,
    spirv::ElementwiseOpPattern<arith::MaxSIOp, spirv::GLSLSMaxOp>,
    spirv::ElementwiseOpPattern<arith::MaxUIOp, spirv::GLSLUMaxOp>,
    spirv::ElementwiseOpPattern<arith::MinFOp, spirv::GLSLFMinOp>,
    spirv::ElementwiseOpPattern<arith::MinSIOp, spirv::GLSLSMinOp>,
    spirv::ElementwiseOpPattern<arith::MinUIOp, spirv::GLSLUMinOp>
  >(typeConverter, patterns.getContext());
  // clang-format on

  // Give CmpFOpNanKernelPattern a higher benefit so it can prevail when Kernel
  // capability is available.
  patterns.add<CmpFOpNanKernelPattern>(typeConverter, patterns.getContext(),
                                       /*benefit=*/2);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertArithmeticToSPIRVPass
    : public ConvertArithmeticToSPIRVBase<ConvertArithmeticToSPIRVPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
    auto target = SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter::Options options;
    options.emulateNon32BitScalarTypes = this->emulateNon32BitScalarTypes;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    // Use UnrealizedConversionCast as the bridge so that we don't need to pull
    // in patterns for other dialects.
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);
    target->addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(&getContext());
    arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arith::createConvertArithmeticToSPIRVPass() {
  return std::make_unique<ConvertArithmeticToSPIRVPass>();
}
