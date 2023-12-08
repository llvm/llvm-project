//===- ComplexOps.cpp - MLIR Complex Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::complex;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  if (auto arrAttr = llvm::dyn_cast<ArrayAttr>(value)) {
    auto complexTy = llvm::dyn_cast<ComplexType>(type);
    if (!complexTy || arrAttr.size() != 2)
      return false;
    auto complexEltTy = complexTy.getElementType();
    if (auto fre = llvm::dyn_cast<FloatAttr>(arrAttr[0])) {
      auto im = llvm::dyn_cast<FloatAttr>(arrAttr[1]);
      return im && fre.getType() == complexEltTy &&
             im.getType() == complexEltTy;
    }
    if (auto ire = llvm::dyn_cast<IntegerAttr>(arrAttr[0])) {
      auto im = llvm::dyn_cast<IntegerAttr>(arrAttr[1]);
      return im && ire.getType() == complexEltTy &&
             im.getType() == complexEltTy;
    }
  }
  return false;
}

LogicalResult ConstantOp::verify() {
  ArrayAttr arrayAttr = getValue();
  if (arrayAttr.size() != 2) {
    return emitOpError(
        "requires 'value' to be a complex constant, represented as array of "
        "two values");
  }

  auto complexEltTy = getType().getElementType();
  if (!isa<FloatAttr, IntegerAttr>(arrayAttr[0]) ||
      !isa<FloatAttr, IntegerAttr>(arrayAttr[1]))
    return emitOpError(
        "requires attribute's elements to be float or integer attributes");
  auto re = llvm::dyn_cast<TypedAttr>(arrayAttr[0]);
  auto im = llvm::dyn_cast<TypedAttr>(arrayAttr[1]);
  if (complexEltTy != re.getType() || complexEltTy != im.getType()) {
    return emitOpError()
           << "requires attribute's element types (" << re.getType() << ", "
           << im.getType()
           << ") to match the element type of the op's return type ("
           << complexEltTy << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(FoldAdaptor bitcast) {
  if (getOperand().getType() == getType())
    return getOperand();

  return {};
}

LogicalResult BitcastOp::verify() {
  auto operandType = getOperand().getType();
  auto resultType = getType();

  // We allow this to be legal as it can be folded away.
  if (operandType == resultType)
    return success();

  if (!operandType.isIntOrFloat() && !isa<ComplexType>(operandType)) {
    return emitOpError("operand must be int/float/complex");
  }

  if (!resultType.isIntOrFloat() && !isa<ComplexType>(resultType)) {
    return emitOpError("result must be int/float/complex");
  }

  if (isa<ComplexType>(operandType) == isa<ComplexType>(resultType)) {
    return emitOpError(
        "requires that either input or output has a complex type");
  }

  if (isa<ComplexType>(resultType))
    std::swap(operandType, resultType);

  int32_t operandBitwidth = dyn_cast<ComplexType>(operandType)
                                .getElementType()
                                .getIntOrFloatBitWidth() *
                            2;
  int32_t resultBitwidth = resultType.getIntOrFloatBitWidth();

  if (operandBitwidth != resultBitwidth) {
    return emitOpError("casting bitwidths do not match");
  }

  return success();
}

struct MergeComplexBitcast final : OpRewritePattern<BitcastOp> {
  using OpRewritePattern<BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitcastOp op,
                                PatternRewriter &rewriter) const override {
    if (auto defining = op.getOperand().getDefiningOp<BitcastOp>()) {
      if (isa<ComplexType>(op.getType()) ||
          isa<ComplexType>(defining.getOperand().getType())) {
        // complex.bitcast requires that input or output is complex.
        rewriter.replaceOpWithNewOp<BitcastOp>(op, op.getType(),
                                               defining.getOperand());
      } else {
        rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, op.getType(),
                                                      defining.getOperand());
      }
      return success();
    }

    if (auto defining = op.getOperand().getDefiningOp<arith::BitcastOp>()) {
      rewriter.replaceOpWithNewOp<BitcastOp>(op, op.getType(),
                                             defining.getOperand());
      return success();
    }

    return failure();
  }
};

struct MergeArithBitcast final : OpRewritePattern<arith::BitcastOp> {
  using OpRewritePattern<arith::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    if (auto defining = op.getOperand().getDefiningOp<complex::BitcastOp>()) {
      rewriter.replaceOpWithNewOp<complex::BitcastOp>(op, op.getType(),
                                                      defining.getOperand());
      return success();
    }

    return failure();
  }
};

void BitcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<MergeComplexBitcast, MergeArithBitcast>(context);
}

//===----------------------------------------------------------------------===//
// CreateOp
//===----------------------------------------------------------------------===//

OpFoldResult CreateOp::fold(FoldAdaptor adaptor) {
  // Fold complex.create(complex.re(op), complex.im(op)).
  if (auto reOp = getOperand(0).getDefiningOp<ReOp>()) {
    if (auto imOp = getOperand(1).getDefiningOp<ImOp>()) {
      if (reOp.getOperand() == imOp.getOperand()) {
        return reOp.getOperand();
      }
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ImOp
//===----------------------------------------------------------------------===//

OpFoldResult ImOp::fold(FoldAdaptor adaptor) {
  ArrayAttr arrayAttr =
      llvm::dyn_cast_if_present<ArrayAttr>(adaptor.getComplex());
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[1];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(1);
  return {};
}

namespace {
template <typename OpKind, int ComponentIndex>
struct FoldComponentNeg final : OpRewritePattern<OpKind> {
  using OpRewritePattern<OpKind>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpKind op,
                                PatternRewriter &rewriter) const override {
    auto negOp = op.getOperand().template getDefiningOp<NegOp>();
    if (!negOp)
      return failure();

    auto createOp = negOp.getComplex().template getDefiningOp<CreateOp>();
    if (!createOp)
      return failure();

    Type elementType = createOp.getType().getElementType();
    assert(isa<FloatType>(elementType));

    rewriter.replaceOpWithNewOp<arith::NegFOp>(
        op, elementType, createOp.getOperand(ComponentIndex));
    return success();
  }
};
} // namespace

void ImOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<FoldComponentNeg<ImOp, 1>>(context);
}

//===----------------------------------------------------------------------===//
// ReOp
//===----------------------------------------------------------------------===//

OpFoldResult ReOp::fold(FoldAdaptor adaptor) {
  ArrayAttr arrayAttr =
      llvm::dyn_cast_if_present<ArrayAttr>(adaptor.getComplex());
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[0];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(0);
  return {};
}

void ReOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<FoldComponentNeg<ReOp, 0>>(context);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // complex.add(complex.sub(a, b), b) -> a
  if (auto sub = getLhs().getDefiningOp<SubOp>())
    if (getRhs() == sub.getRhs())
      return sub.getLhs();

  // complex.add(b, complex.sub(a, b)) -> a
  if (auto sub = getRhs().getDefiningOp<SubOp>())
    if (getLhs() == sub.getRhs())
      return sub.getLhs();

  // complex.add(a, complex.constant<0.0, 0.0>) -> a
  if (auto constantOp = getRhs().getDefiningOp<ConstantOp>()) {
    auto arrayAttr = constantOp.getValue();
    if (llvm::cast<FloatAttr>(arrayAttr[0]).getValue().isZero() &&
        llvm::cast<FloatAttr>(arrayAttr[1]).getValue().isZero()) {
      return getLhs();
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  // complex.sub(complex.add(a, b), b) -> a
  if (auto add = getLhs().getDefiningOp<AddOp>())
    if (getRhs() == add.getRhs())
      return add.getLhs();

  // complex.sub(a, complex.constant<0.0, 0.0>) -> a
  if (auto constantOp = getRhs().getDefiningOp<ConstantOp>()) {
    auto arrayAttr = constantOp.getValue();
    if (llvm::cast<FloatAttr>(arrayAttr[0]).getValue().isZero() &&
        llvm::cast<FloatAttr>(arrayAttr[1]).getValue().isZero()) {
      return getLhs();
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  // complex.neg(complex.neg(a)) -> a
  if (auto negOp = getOperand().getDefiningOp<NegOp>())
    return negOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

OpFoldResult LogOp::fold(FoldAdaptor adaptor) {
  // complex.log(complex.exp(a)) -> a
  if (auto expOp = getOperand().getDefiningOp<ExpOp>())
    return expOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

OpFoldResult ExpOp::fold(FoldAdaptor adaptor) {
  // complex.exp(complex.log(a)) -> a
  if (auto logOp = getOperand().getDefiningOp<LogOp>())
    return logOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// ConjOp
//===----------------------------------------------------------------------===//

OpFoldResult ConjOp::fold(FoldAdaptor adaptor) {
  // complex.conj(complex.conj(a)) -> a
  if (auto conjOp = getOperand().getDefiningOp<ConjOp>())
    return conjOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto constant = getRhs().getDefiningOp<ConstantOp>();
  if (!constant)
    return {};

  ArrayAttr arrayAttr = constant.getValue();
  APFloat real = cast<FloatAttr>(arrayAttr[0]).getValue();
  APFloat imag = cast<FloatAttr>(arrayAttr[1]).getValue();

  if (!imag.isZero())
    return {};

  // complex.mul(a, complex.constant<1.0, 0.0>) -> a
  if (real == APFloat(real.getSemantics(), 1))
    return getLhs();

  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"
