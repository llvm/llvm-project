//===- ArithToSPIRV.cpp - Arithmetic to SPIRV dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"

#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOSPIRV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "arith-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation Conversion
//===----------------------------------------------------------------------===//

namespace {

/// Converts composite arith.constant operation to spirv.Constant.
struct ConstantCompositeOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts scalar arith.constant operation to spirv.Constant.
struct ConstantScalarOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.remsi to GLSL SPIR-V ops.
///
/// This cannot be merged into the template unary/binary pattern due to Vulkan
/// restrictions over spirv.SRem and spirv.SMod.
struct RemSIOpGLPattern final : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.remsi to OpenCL SPIR-V ops.
struct RemSIOpCLPattern final : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern::OpConversionPattern;

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
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.xori to SPIR-V operations if the type of source is i1 or
/// vector of i1.
struct XOrIOpBooleanPattern final : public OpConversionPattern<arith::XOrIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.uitofp to spirv.Select if the type of source is i1 or vector
/// of i1.
struct UIToFPI1Pattern final : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.extsi to spirv.Select if the type of source is i1 or vector
/// of i1.
struct ExtSII1Pattern final : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.extui to spirv.Select if the type of source is i1 or vector
/// of i1.
struct ExtUII1Pattern final : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.trunci to spirv.Select if the type of result is i1 or vector
/// of i1.
struct TruncII1Pattern final : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

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
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation to SPIR-V ops.
class CmpIOpPattern final : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating-point comparison operations to SPIR-V ops.
class CmpFOpPattern final : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern requires
/// Kernel capability.
class CmpFOpNanKernelPattern final : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

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

/// Converts arith.addui_extended to spirv.IAddCarry.
class AddUIExtendedOpPattern final
    : public OpConversionPattern<arith::AddUIExtendedOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.mul*i_extended to spirv.*MulExtended.
template <typename ArithMulOp, typename SPIRVMulOp>
class MulIExtendedOpPattern final : public OpConversionPattern<ArithMulOp> {
public:
  using OpConversionPattern<ArithMulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ArithMulOp op, typename ArithMulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.select to spirv.Select.
class SelectOpPattern final : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts arith.maxf to spirv.GL.FMax or spirv.CL.fmax.
template <typename Op, typename SPIRVOp>
class MinMaxFOpPattern final : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Helpers
//===----------------------------------------------------------------------===//

/// Converts the given `srcAttr` into a boolean attribute if it holds an
/// integral value. Returns null attribute if conversion fails.
static BoolAttr convertBoolAttr(Attribute srcAttr, Builder builder) {
  if (auto boolAttr = dyn_cast<BoolAttr>(srcAttr))
    return boolAttr;
  if (auto intAttr = dyn_cast<IntegerAttr>(srcAttr))
    return builder.getBoolAttr(intAttr.getValue().getBoolValue());
  return {};
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
  return {};
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
  assert(type && "Not a valid type");
  if (type.isInteger(1))
    return true;

  if (auto vecType = dyn_cast<VectorType>(type))
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
    else if (auto vecType = dyn_cast<VectorType>(type))
      bw = vecType.getElementTypeBitWidth() * vecType.getNumElements();
    return bw;
  };
  unsigned aBW = getNumBitwidth(a);
  unsigned bBW = getNumBitwidth(b);
  return aBW != 0 && bBW != 0 && aBW == bBW;
}

/// Returns a source type conversion failure for `srcType` and operation `op`.
static LogicalResult
getTypeConversionFailure(ConversionPatternRewriter &rewriter, Operation *op,
                         Type srcType) {
  return rewriter.notifyMatchFailure(
      op->getLoc(),
      llvm::formatv("failed to convert source type '{0}'", srcType));
}

/// Returns a source type conversion failure for the result type of `op`.
static LogicalResult
getTypeConversionFailure(ConversionPatternRewriter &rewriter, Operation *op) {
  assert(op->getNumResults() == 1);
  return getTypeConversionFailure(rewriter, op, op->getResultTypes().front());
}

//===----------------------------------------------------------------------===//
// ConstantOp with composite type
//===----------------------------------------------------------------------===//

LogicalResult ConstantCompositeOpPattern::matchAndRewrite(
    arith::ConstantOp constOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto srcType = dyn_cast<ShapedType>(constOp.getType());
  if (!srcType || srcType.getNumElements() == 1)
    return failure();

  // arith.constant should only have vector or tenor types.
  assert((isa<VectorType, RankedTensorType>(srcType)));

  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  auto dstElementsAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!dstElementsAttr)
    return failure();

  ShapedType dstAttrType = dstElementsAttr.getType();

  // If the composite type has more than one dimensions, perform linearization.
  if (srcType.getRank() > 1) {
    if (isa<RankedTensorType>(srcType)) {
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
  if (auto arrayType = dyn_cast<spirv::ArrayType>(dstType))
    dstElemType = arrayType.getElementType();
  else
    dstElemType = cast<VectorType>(dstType).getElementType();

  // If the source and destination element types are different, perform
  // attribute conversion.
  if (srcElemType != dstElemType) {
    SmallVector<Attribute, 8> elements;
    if (isa<FloatType>(srcElemType)) {
      for (FloatAttr srcAttr : dstElementsAttr.getValues<FloatAttr>()) {
        FloatAttr dstAttr =
            convertFloatAttr(srcAttr, cast<FloatType>(dstElemType), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    } else if (srcElemType.isInteger(1)) {
      return failure();
    } else {
      for (IntegerAttr srcAttr : dstElementsAttr.getValues<IntegerAttr>()) {
        IntegerAttr dstAttr = convertIntegerAttr(
            srcAttr, cast<IntegerType>(dstElemType), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    }

    // Unfortunately, we cannot use dialect-specific types for element
    // attributes; element attributes only works with builtin types. So we need
    // to prepare another converted builtin types for the destination elements
    // attribute.
    if (isa<RankedTensorType>(dstAttrType))
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
  if (auto shapedType = dyn_cast<ShapedType>(srcType)) {
    if (shapedType.getNumElements() != 1)
      return failure();
    srcType = shapedType.getElementType();
  }
  if (!srcType.isIntOrIndexOrFloat())
    return failure();

  Attribute cstAttr = constOp.getValue();
  if (auto elementsAttr = dyn_cast<DenseElementsAttr>(cstAttr))
    cstAttr = elementsAttr.getSplatValue<Attribute>();

  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  // Floating-point types.
  if (isa<FloatType>(srcType)) {
    auto srcAttr = cast<FloatAttr>(cstAttr);
    auto dstAttr = srcAttr;

    // Floating-point types not supported in the target environment are all
    // converted to float type.
    if (srcType != dstType) {
      dstAttr = convertFloatAttr(srcAttr, cast<FloatType>(dstType), rewriter);
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
  auto srcAttr = cast<IntegerAttr>(cstAttr);
  IntegerAttr dstAttr =
      convertIntegerAttr(srcAttr, cast<IntegerType>(dstType), rewriter);
  if (!dstAttr)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// RemSIOpGLPattern
//===----------------------------------------------------------------------===//

/// Returns signed remainder for `lhs` and `rhs` and lets the result follow
/// the sign of `signOperand`.
///
/// Note that this is needed for Vulkan. Per the Vulkan's SPIR-V environment
/// spec, "for the OpSRem and OpSMod instructions, if either operand is negative
/// the result is undefined."  So we cannot directly use spirv.SRem/spirv.SMod
/// if either operand can be negative. Emulate it via spirv.UMod.
template <typename SignedAbsOp>
static Value emulateSignedRemainder(Location loc, Value lhs, Value rhs,
                                    Value signOperand, OpBuilder &builder) {
  assert(lhs.getType() == rhs.getType());
  assert(lhs == signOperand || rhs == signOperand);

  Type type = lhs.getType();

  // Calculate the remainder with spirv.UMod.
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
RemSIOpGLPattern::matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Value result = emulateSignedRemainder<spirv::GLSAbsOp>(
      op.getLoc(), adaptor.getOperands()[0], adaptor.getOperands()[1],
      adaptor.getOperands()[0], rewriter);
  rewriter.replaceOp(op, result);

  return success();
}

//===----------------------------------------------------------------------===//
// RemSIOpCLPattern
//===----------------------------------------------------------------------===//

LogicalResult
RemSIOpCLPattern::matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Value result = emulateSignedRemainder<spirv::CLSAbsOp>(
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
  Type dstType = this->getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

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

  Type dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

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

  Type dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

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
  Type srcType = adaptor.getOperands().front().getType();
  if (!isBoolScalarOrVector(srcType))
    return failure();

  Type dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

  Location loc = op.getLoc();
  Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
  Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(
      op, dstType, adaptor.getOperands().front(), one, zero);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtSII1Pattern
//===----------------------------------------------------------------------===//

LogicalResult
ExtSII1Pattern::matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Value operand = adaptor.getIn();
  if (!isBoolScalarOrVector(operand.getType()))
    return failure();

  Location loc = op.getLoc();
  Type dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

  Value allOnes;
  if (auto intTy = dyn_cast<IntegerType>(dstType)) {
    unsigned componentBitwidth = intTy.getWidth();
    allOnes = rewriter.create<spirv::ConstantOp>(
        loc, intTy,
        rewriter.getIntegerAttr(intTy, APInt::getAllOnes(componentBitwidth)));
  } else if (auto vectorTy = dyn_cast<VectorType>(dstType)) {
    unsigned componentBitwidth = vectorTy.getElementTypeBitWidth();
    allOnes = rewriter.create<spirv::ConstantOp>(
        loc, vectorTy,
        SplatElementsAttr::get(vectorTy, APInt::getAllOnes(componentBitwidth)));
  } else {
    return rewriter.notifyMatchFailure(
        loc, llvm::formatv("unhandled type: {0}", dstType));
  }

  Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, dstType, operand, allOnes,
                                               zero);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtUII1Pattern
//===----------------------------------------------------------------------===//

LogicalResult
ExtUII1Pattern::matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Type srcType = adaptor.getOperands().front().getType();
  if (!isBoolScalarOrVector(srcType))
    return failure();

  Type dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

  Location loc = op.getLoc();
  Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
  Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(
      op, dstType, adaptor.getOperands().front(), one, zero);
  return success();
}

//===----------------------------------------------------------------------===//
// TruncII1Pattern
//===----------------------------------------------------------------------===//

LogicalResult
TruncII1Pattern::matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Type dstType = getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

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
  Type srcType = adaptor.getOperands().front().getType();
  Type dstType = this->getTypeConverter()->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

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
    return getTypeConversionFailure(rewriter, op, srcType);

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
    if (auto vectorType = dyn_cast<VectorType>(dstType))
      type = VectorType::get(vectorType.getShape(), type);
    Value extLhs =
        rewriter.create<arith::ExtUIOp>(op.getLoc(), type, adaptor.getLhs());
    Value extRhs =
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
    return getTypeConversionFailure(rewriter, op, srcType);

  switch (op.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    if (spirvOp::template hasTrait<OpTrait::spirv::UnsignedOp>() &&            \
        !getElementTypeOrSelf(srcType).isIndex() && srcType != dstType &&      \
        !hasSameBitwidth(srcType, dstType)) {                                  \
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
  auto *converter = getTypeConverter<SPIRVTypeConverter>();

  Value replace;
  if (converter->getOptions().enableFastMathMode) {
    if (op.getPredicate() == arith::CmpFPredicate::ORD) {
      // Ordered comparsion checks if neither operand is NaN.
      replace = spirv::ConstantOp::getOne(op.getType(), loc, rewriter);
    } else {
      // Unordered comparsion checks if either operand is NaN.
      replace = spirv::ConstantOp::getZero(op.getType(), loc, rewriter);
    }
  } else {
    Value lhsIsNan = rewriter.create<spirv::IsNanOp>(loc, adaptor.getLhs());
    Value rhsIsNan = rewriter.create<spirv::IsNanOp>(loc, adaptor.getRhs());

    replace = rewriter.create<spirv::LogicalOrOp>(loc, lhsIsNan, rhsIsNan);
    if (op.getPredicate() == arith::CmpFPredicate::ORD)
      replace = rewriter.create<spirv::LogicalNotOp>(loc, replace);
  }

  rewriter.replaceOp(op, replace);
  return success();
}

//===----------------------------------------------------------------------===//
// AddUIExtendedOpPattern
//===----------------------------------------------------------------------===//

LogicalResult AddUIExtendedOpPattern::matchAndRewrite(
    arith::AddUIExtendedOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type dstElemTy = adaptor.getLhs().getType();
  Location loc = op->getLoc();
  Value result = rewriter.create<spirv::IAddCarryOp>(loc, adaptor.getLhs(),
                                                     adaptor.getRhs());

  Value sumResult = rewriter.create<spirv::CompositeExtractOp>(
      loc, result, llvm::ArrayRef(0));
  Value carryValue = rewriter.create<spirv::CompositeExtractOp>(
      loc, result, llvm::ArrayRef(1));

  // Convert the carry value to boolean.
  Value one = spirv::ConstantOp::getOne(dstElemTy, loc, rewriter);
  Value carryResult = rewriter.create<spirv::IEqualOp>(loc, carryValue, one);

  rewriter.replaceOp(op, {sumResult, carryResult});
  return success();
}

//===----------------------------------------------------------------------===//
// MulIExtendedOpPattern
//===----------------------------------------------------------------------===//

template <typename ArithMulOp, typename SPIRVMulOp>
LogicalResult MulIExtendedOpPattern<ArithMulOp, SPIRVMulOp>::matchAndRewrite(
    ArithMulOp op, typename ArithMulOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value result =
      rewriter.create<SPIRVMulOp>(loc, adaptor.getLhs(), adaptor.getRhs());

  Value low = rewriter.create<spirv::CompositeExtractOp>(loc, result,
                                                         llvm::ArrayRef(0));
  Value high = rewriter.create<spirv::CompositeExtractOp>(loc, result,
                                                          llvm::ArrayRef(1));

  rewriter.replaceOp(op, {low, high});
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
// MaxFOpPattern
//===----------------------------------------------------------------------===//

template <typename Op, typename SPIRVOp>
LogicalResult MinMaxFOpPattern<Op, SPIRVOp>::matchAndRewrite(
    Op op, typename Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto *converter = this->template getTypeConverter<SPIRVTypeConverter>();
  Type dstType = converter->convertType(op.getType());
  if (!dstType)
    return getTypeConversionFailure(rewriter, op);

  // arith.maxf/minf:
  //   "if one of the arguments is NaN, then the result is also NaN."
  // spirv.GL.FMax/FMin
  //   "which operand is the result is undefined if one of the operands
  //   is a NaN."
  // spirv.CL.fmax/fmin:
  //   "If one argument is a NaN, Fmin returns the other argument."

  Location loc = op.getLoc();
  Value spirvOp = rewriter.create<SPIRVOp>(loc, dstType, adaptor.getOperands());

  if (converter->getOptions().enableFastMathMode) {
    rewriter.replaceOp(op, spirvOp);
    return success();
  }

  Value lhsIsNan = rewriter.create<spirv::IsNanOp>(loc, adaptor.getLhs());
  Value rhsIsNan = rewriter.create<spirv::IsNanOp>(loc, adaptor.getRhs());

  Value select1 = rewriter.create<spirv::SelectOp>(loc, dstType, lhsIsNan,
                                                   adaptor.getLhs(), spirvOp);
  Value select2 = rewriter.create<spirv::SelectOp>(loc, dstType, rhsIsNan,
                                                   adaptor.getRhs(), select1);

  rewriter.replaceOp(op, select2);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithToSPIRVPatterns(
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
    RemSIOpGLPattern, RemSIOpCLPattern,
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
    TypeCastingOpPattern<arith::ExtSIOp, spirv::SConvertOp>, ExtSII1Pattern,
    TypeCastingOpPattern<arith::ExtFOp, spirv::FConvertOp>,
    TypeCastingOpPattern<arith::TruncIOp, spirv::SConvertOp>, TruncII1Pattern,
    TypeCastingOpPattern<arith::TruncFOp, spirv::FConvertOp>,
    TypeCastingOpPattern<arith::UIToFPOp, spirv::ConvertUToFOp>, UIToFPI1Pattern,
    TypeCastingOpPattern<arith::SIToFPOp, spirv::ConvertSToFOp>,
    TypeCastingOpPattern<arith::FPToUIOp, spirv::ConvertFToUOp>,
    TypeCastingOpPattern<arith::FPToSIOp, spirv::ConvertFToSOp>,
    TypeCastingOpPattern<arith::IndexCastOp, spirv::SConvertOp>,
    TypeCastingOpPattern<arith::IndexCastUIOp, spirv::UConvertOp>,
    TypeCastingOpPattern<arith::BitcastOp, spirv::BitcastOp>,
    CmpIOpBooleanPattern, CmpIOpPattern,
    CmpFOpNanNonePattern, CmpFOpPattern,
    AddUIExtendedOpPattern,
    MulIExtendedOpPattern<arith::MulSIExtendedOp, spirv::SMulExtendedOp>,
    MulIExtendedOpPattern<arith::MulUIExtendedOp, spirv::UMulExtendedOp>,
    SelectOpPattern,

    MinMaxFOpPattern<arith::MaxFOp, spirv::GLFMaxOp>,
    MinMaxFOpPattern<arith::MinFOp, spirv::GLFMinOp>,
    spirv::ElementwiseOpPattern<arith::MaxSIOp, spirv::GLSMaxOp>,
    spirv::ElementwiseOpPattern<arith::MaxUIOp, spirv::GLUMaxOp>,
    spirv::ElementwiseOpPattern<arith::MinSIOp, spirv::GLSMinOp>,
    spirv::ElementwiseOpPattern<arith::MinUIOp, spirv::GLUMinOp>,

    MinMaxFOpPattern<arith::MaxFOp, spirv::CLFMaxOp>,
    MinMaxFOpPattern<arith::MinFOp, spirv::CLFMinOp>,
    spirv::ElementwiseOpPattern<arith::MaxSIOp, spirv::CLSMaxOp>,
    spirv::ElementwiseOpPattern<arith::MaxUIOp, spirv::CLUMaxOp>,
    spirv::ElementwiseOpPattern<arith::MinSIOp, spirv::CLSMinOp>,
    spirv::ElementwiseOpPattern<arith::MinUIOp, spirv::CLUMinOp>
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
struct ConvertArithToSPIRVPass
    : public impl::ConvertArithToSPIRVBase<ConvertArithToSPIRVPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<SPIRVConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    options.emulateLT32BitScalarTypes = this->emulateLT32BitScalarTypes;
    options.enableFastMathMode = this->enableFastMath;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    // Use UnrealizedConversionCast as the bridge so that we don't need to pull
    // in patterns for other dialects.
    target->addLegalOp<UnrealizedConversionCastOp>();

    // Fail hard when there are any remaining 'arith' ops.
    target->addIllegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    arith::populateArithToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<>> mlir::arith::createConvertArithToSPIRVPass() {
  return std::make_unique<ConvertArithToSPIRVPass>();
}
