//===- LegalizeForLLVMExport.cpp - Prepare VCIX for LLVM translation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/VCIX/Transforms.h"
#include "mlir/Dialect/VCIX/VCIXDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include <string>

using namespace mlir;

static constexpr char kVCIXTargetFeaturesAttr[] = "vcix.target_features";

// Get integer value from an attribute and zext it to unsigned integer
static unsigned getInteger(Attribute attr) {
  auto intAttr = cast<IntegerAttr>(attr);
  unsigned value = intAttr.getInt();
  return value & ((1 << intAttr.getType().getIntOrFloatBitWidth()) - 1);
}

template <typename SourceOp, typename DestOp>
struct OneToOneWithPromotionBase : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  StringAttr getTargetFeatures(Operation *op) const {
    Operation *func = op;
    while (func) {
      func = func->getParentOp();
      if (isa<FunctionOpInterface>(func))
        break;
    }
    if (!func)
      llvm_unreachable("Cannot find function-like operation in parents");

    const DictionaryAttr dictAttr = func->getAttrDictionary();
    if (auto targetFeatures = dictAttr.getNamed(kVCIXTargetFeaturesAttr))
      return targetFeatures->getValue().cast<StringAttr>();
    return nullptr;
  }

  unsigned getXLen(Operation *op) const {
    StringAttr targetFeatures = getTargetFeatures(op);
    if (!targetFeatures)
      return 64;

    if (targetFeatures.getValue().contains("+32bit"))
      return 32;

    if (targetFeatures.getValue().contains("+64bit"))
      return 64;

    llvm_unreachable("Unsupported RISC-V target");
  }

  explicit OneToOneWithPromotionBase(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<SourceOp>(converter) {}

  /// Return new IntegerAttr with a value promoted to xlen if necessary
  IntegerAttr promoteIntAttr(ConversionPatternRewriter &rewriter,
                             Attribute attr, const unsigned xlen) const {
    Type xlenType = rewriter.getIntegerType(xlen);
    return rewriter.getIntegerAttr(xlenType, getInteger(attr));
  }

  /// Convert all operands to required type for correct legalization
  FailureOr<SmallVector<Value>>
  convertOperands(ConversionPatternRewriter &rewriter, ValueRange operands,
                  const unsigned xlen) const {
    SmallVector<Value> res(operands);
    Value op1 = operands.front();
    if (auto intType = op1.getType().template dyn_cast<IntegerType>())
      if (intType.getWidth() < xlen) {
        Value zext = rewriter.create<LLVM::ZExtOp>(
            op1.getLoc(), rewriter.getIntegerType(xlen), op1);
        res[0] = zext;
      }
    return res;
  }
};

/// Convert vcix operation into intrinsic version with promotion of opcode, rd
/// rs2 to Xlen
template <typename SourceOp, typename DestOp>
struct OneToOneUnaryROWithPromotion
    : public OneToOneWithPromotionBase<SourceOp, DestOp> {
  using OneToOneWithPromotionBase<SourceOp, DestOp>::OneToOneWithPromotionBase;

  explicit OneToOneUnaryROWithPromotion(LLVMTypeConverter &converter)
      : OneToOneWithPromotionBase<SourceOp, DestOp>(converter) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const unsigned xlen = this->getXLen(op);
    FailureOr<SmallVector<Value>> operands =
        this->convertOperands(rewriter, adaptor.getOperands(), xlen);
    if (failed(operands))
      return failure();

    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(DestOp::getOperationName()),
        *operands, {}, op->getAttrs());
    DestOp dstOp = dyn_cast<DestOp>(newOp);
    Type xlenType = rewriter.getIntegerType(xlen);
    dstOp.setOpcodeAttr(
        rewriter.getIntegerAttr(xlenType, getInteger(dstOp.getOpcodeAttr())));
    dstOp.setRs2Attr(
        rewriter.getIntegerAttr(xlenType, getInteger(dstOp.getRs2Attr())));
    dstOp.setRdAttr(
        rewriter.getIntegerAttr(xlenType, getInteger(dstOp.getRdAttr())));
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert vcix operation into intrinsic version with promotion of opcode and
/// rs2 to Xlen
template <typename SourceOp, typename DestOp>
struct OneToOneUnaryWithPromotion
    : public OneToOneWithPromotionBase<SourceOp, DestOp> {
  using OneToOneWithPromotionBase<SourceOp, DestOp>::OneToOneWithPromotionBase;

  explicit OneToOneUnaryWithPromotion(LLVMTypeConverter &converter)
      : OneToOneWithPromotionBase<SourceOp, DestOp>(converter) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const unsigned xlen = this->getXLen(op);
    FailureOr<SmallVector<Value>> operands =
        this->convertOperands(rewriter, adaptor.getOperands(), xlen);
    if (failed(operands))
      return failure();

    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(DestOp::getOperationName()),
        *operands, op->getResultTypes(), op->getAttrs());
    DestOp dstOp = dyn_cast<DestOp>(newOp);
    dstOp.setOpcodeAttr(
        this->promoteIntAttr(rewriter, dstOp.getOpcodeAttr(), xlen));
    dstOp.setRs2Attr(this->promoteIntAttr(rewriter, dstOp.getRs2Attr(), xlen));
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Convert vcix operation into intrinsic version with promotion of opcode and
/// rd to Xlen
template <typename SourceOp, typename DestOp>
struct OneToOneBinaryROWithPromotion
    : public OneToOneWithPromotionBase<SourceOp, DestOp> {
  using OneToOneWithPromotionBase<SourceOp, DestOp>::OneToOneWithPromotionBase;

  explicit OneToOneBinaryROWithPromotion(LLVMTypeConverter &converter)
      : OneToOneWithPromotionBase<SourceOp, DestOp>(converter) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const unsigned xlen = this->getXLen(op);
    FailureOr<SmallVector<Value>> operands =
        this->convertOperands(rewriter, adaptor.getOperands(), xlen);
    if (failed(operands))
      return failure();

    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(DestOp::getOperationName()),
        *operands, {}, op->getAttrs());
    DestOp dstOp = dyn_cast<DestOp>(newOp);
    dstOp.setOpcodeAttr(
        this->promoteIntAttr(rewriter, dstOp.getOpcodeAttr(), xlen));
    dstOp.setRdAttr(this->promoteIntAttr(rewriter, dstOp.getRdAttr(), xlen));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert vcix operation into intrinsic version with promotion of opcode to
/// Xlen
template <typename SourceOp, typename DestOp>
struct OneToOneWithPromotion
    : public OneToOneWithPromotionBase<SourceOp, DestOp> {
  using OneToOneWithPromotionBase<SourceOp, DestOp>::OneToOneWithPromotionBase;

  explicit OneToOneWithPromotion(LLVMTypeConverter &converter)
      : OneToOneWithPromotionBase<SourceOp, DestOp>(converter) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const unsigned xlen = this->getXLen(op);
    FailureOr<SmallVector<Value>> operands =
        this->convertOperands(rewriter, adaptor.getOperands(), xlen);
    if (failed(operands))
      return failure();

    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(DestOp::getOperationName()),
        *operands, op->getResultTypes(), op->getAttrs());
    DestOp dstOp = dyn_cast<DestOp>(newOp);
    dstOp.setOpcodeAttr(
        this->promoteIntAttr(rewriter, dstOp.getOpcodeAttr(), xlen));

    if (op->getResultTypes().empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, newOp);

    return success();
  }
};

/// Populate the given list with patterns that convert from VCIX to LLVM.
void mlir::populateVCIXLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Populate conversion patterns
  patterns.add<
      OneToOneUnaryWithPromotion<vcix::UnaryOp, vcix::UnaryIntrinOp>,
      OneToOneUnaryROWithPromotion<vcix::UnaryROOp, vcix::UnaryIntrinROOp>,
      OneToOneBinaryROWithPromotion<vcix::BinaryROOp, vcix::BinaryIntrinROOp>,
      OneToOneWithPromotion<vcix::BinaryOp, vcix::BinaryIntrinOp>,
      OneToOneWithPromotion<vcix::TernaryOp, vcix::TernaryIntrinOp>,
      OneToOneWithPromotion<vcix::TernaryROOp, vcix::TernaryIntrinROOp>,
      OneToOneWithPromotion<vcix::WideTernaryOp, vcix::WideTernaryIntrinOp>,
      OneToOneWithPromotion<vcix::WideTernaryROOp,
                            vcix::WideTernaryIntrinROOp>>(converter);
}

void mlir::configureVCIXLegalizeForExportTarget(LLVMConversionTarget &target) {
  // During legalization some operation may zext operands to simplify conversion
  // to LLVM IR later
  // clang-format off
  target.addLegalOp<LLVM::ZExtOp,
                    LLVM::UndefOp,
                    LLVM::vector_extract,
                    LLVM::vector_insert,
                    LLVM::BitcastOp>();
  target.addLegalOp<vcix::UnaryIntrinOp,
                    vcix::UnaryIntrinROOp,
                    vcix::BinaryIntrinOp,
                    vcix::BinaryIntrinROOp,
                    vcix::TernaryIntrinOp,
                    vcix::TernaryIntrinROOp,
                    vcix::WideTernaryIntrinOp,
                    vcix::WideTernaryIntrinROOp>();
  target.addIllegalOp<vcix::UnaryOp,
                      vcix::UnaryROOp,
                      vcix::BinaryOp,
                      vcix::BinaryROOp,
                      vcix::TernaryOp,
                      vcix::TernaryROOp,
                      vcix::WideTernaryOp,
                      vcix::WideTernaryROOp>();
  // clang-format on
}
