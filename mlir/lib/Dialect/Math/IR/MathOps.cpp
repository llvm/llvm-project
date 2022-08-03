//===- MathOps.cpp - MLIR operations for math implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::math;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"

//===----------------------------------------------------------------------===//
// AbsOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::AbsOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<FloatAttr>(operands, [](const APFloat &a) {
    const APFloat &result(a);
    return abs(result);
  });
}

//===----------------------------------------------------------------------===//
// AtanOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::AtanOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(atan(a.convertToDouble()));
        case 32:
          return APFloat(atanf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// CeilOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CeilOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<FloatAttr>(operands, [](const APFloat &a) {
    APFloat result(a);
    result.roundToIntegral(llvm::RoundingMode::TowardPositive);
    return result;
  });
}

//===----------------------------------------------------------------------===//
// CopySignOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CopySignOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(operands,
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat result(a);
                                        result.copySign(b);
                                        return result;
                                      });
}

//===----------------------------------------------------------------------===//
// CountLeadingZerosOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CountLeadingZerosOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](const APInt &a) {
    return APInt(a.getBitWidth(), a.countLeadingZeros());
  });
}

//===----------------------------------------------------------------------===//
// CountTrailingZerosOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CountTrailingZerosOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](const APInt &a) {
    return APInt(a.getBitWidth(), a.countTrailingZeros());
  });
}

//===----------------------------------------------------------------------===//
// CtPopOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CtPopOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](const APInt &a) {
    return APInt(a.getBitWidth(), a.countPopulation());
  });
}

//===----------------------------------------------------------------------===//
// LogOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::LogOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        if (a.isNegative())
          return {};

        if (a.getSizeInBits(a.getSemantics()) == 64)
          return APFloat(log(a.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32)
          return APFloat(logf(a.convertToFloat()));

        return {};
      });
}

//===----------------------------------------------------------------------===//
// Log2Op folder
//===----------------------------------------------------------------------===//

OpFoldResult math::Log2Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        if (a.isNegative())
          return {};

        if (a.getSizeInBits(a.getSemantics()) == 64)
          return APFloat(log2(a.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32)
          return APFloat(log2f(a.convertToFloat()));

        return {};
      });
}

//===----------------------------------------------------------------------===//
// Log10Op folder
//===----------------------------------------------------------------------===//

OpFoldResult math::Log10Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        if (a.isNegative())
          return {};

        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(log10(a.convertToDouble()));
        case 32:
          return APFloat(log10f(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// Log1pOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::Log1pOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          if ((a + APFloat(1.0)).isNegative())
            return {};
          return APFloat(log1p(a.convertToDouble()));
        case 32:
          if ((a + APFloat(1.0f)).isNegative())
            return {};
          return APFloat(log1pf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// PowFOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::PowFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) -> Optional<APFloat> {
        if (a.getSizeInBits(a.getSemantics()) == 64 &&
            b.getSizeInBits(b.getSemantics()) == 64)
          return APFloat(pow(a.convertToDouble(), b.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32 &&
            b.getSizeInBits(b.getSemantics()) == 32)
          return APFloat(powf(a.convertToFloat(), b.convertToFloat()));

        return {};
      });
}

//===----------------------------------------------------------------------===//
// SqrtOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::SqrtOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        if (a.isNegative())
          return {};

        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(sqrt(a.convertToDouble()));
        case 32:
          return APFloat(sqrtf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// ExpOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::ExpOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(exp(a.convertToDouble()));
        case 32:
          return APFloat(expf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// Exp2Op folder
//===----------------------------------------------------------------------===//

OpFoldResult math::Exp2Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(exp2(a.convertToDouble()));
        case 32:
          return APFloat(exp2f(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// ExpM1Op folder
//===----------------------------------------------------------------------===//

OpFoldResult math::ExpM1Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(expm1(a.convertToDouble()));
        case 32:
          return APFloat(expm1f(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// TanOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::TanOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(tan(a.convertToDouble()));
        case 32:
          return APFloat(tanf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

//===----------------------------------------------------------------------===//
// TanhOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::TanhOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOpConditional<FloatAttr>(
      operands, [](const APFloat &a) -> Optional<APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return APFloat(tanh(a.convertToDouble()));
        case 32:
          return APFloat(tanhf(a.convertToFloat()));
        default:
          return {};
        }
      });
}

/// Materialize an integer or floating point constant.
Operation *math::MathDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<arith::ConstantOp>(loc, value, type);
}
