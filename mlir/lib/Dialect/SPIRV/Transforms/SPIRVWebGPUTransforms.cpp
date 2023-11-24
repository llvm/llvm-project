//===- SPIRVWebGPUTransforms.cpp - WebGPU-specific transforms -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SPIR-V transforms used when targetting WebGPU.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/SPIRVWebGPUTransforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <cstdint>

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVWEBGPUPREPAREPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
namespace {
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
Attribute getScalarOrSplatAttr(Type type, int64_t value) {
  APInt sizedValue(getElementTypeOrSelf(type).getIntOrFloatBitWidth(), value);
  if (auto intTy = dyn_cast<IntegerType>(type))
    return IntegerAttr::get(intTy, sizedValue);

  return SplatElementsAttr::get(cast<ShapedType>(type), sizedValue);
}

Value lowerExtendedMultiplication(Operation *mulOp, PatternRewriter &rewriter,
                                  Value lhs, Value rhs,
                                  bool signExtendArguments) {
  Location loc = mulOp->getLoc();
  Type argTy = lhs.getType();
  // Emulate 64-bit multiplication by splitting each input element of type i32
  // into 2 16-bit digits of type i32. This is so that the intermediate
  // multiplications and additions do not overflow. We extract these 16-bit
  // digits from i32 vector elements by masking (low digit) and shifting right
  // (high digit).
  //
  // The multiplication algorithm used is the standard (long) multiplication.
  // Multiplying two i32 integers produces 64 bits of result, i.e., 4 16-bit
  // digits.
  //   - With zero-extended arguments, we end up emitting only 4 multiplications
  //     and 4 additions after constant folding.
  //   - With sign-extended arguments, we end up emitting 8 multiplications and
  //     and 12 additions after CSE.
  Value cstLowMask = rewriter.create<ConstantOp>(
      loc, lhs.getType(), getScalarOrSplatAttr(argTy, (1 << 16) - 1));
  auto getLowDigit = [&rewriter, loc, cstLowMask](Value val) {
    return rewriter.create<BitwiseAndOp>(loc, val, cstLowMask);
  };

  Value cst16 = rewriter.create<ConstantOp>(loc, lhs.getType(),
                                            getScalarOrSplatAttr(argTy, 16));
  auto getHighDigit = [&rewriter, loc, cst16](Value val) {
    return rewriter.create<ShiftRightLogicalOp>(loc, val, cst16);
  };

  auto getSignDigit = [&rewriter, loc, cst16, &getHighDigit](Value val) {
    // We only need to shift arithmetically by 15, but the extra
    // sign-extension bit will be truncated by the logical shift, so this is
    // fine. We do not have to introduce an extra constant since any
    // value in [15, 32) would do.
    return getHighDigit(
        rewriter.create<ShiftRightArithmeticOp>(loc, val, cst16));
  };

  Value cst0 = rewriter.create<ConstantOp>(loc, lhs.getType(),
                                           getScalarOrSplatAttr(argTy, 0));

  Value lhsLow = getLowDigit(lhs);
  Value lhsHigh = getHighDigit(lhs);
  Value lhsExt = signExtendArguments ? getSignDigit(lhs) : cst0;
  Value rhsLow = getLowDigit(rhs);
  Value rhsHigh = getHighDigit(rhs);
  Value rhsExt = signExtendArguments ? getSignDigit(rhs) : cst0;

  std::array<Value, 4> lhsDigits = {lhsLow, lhsHigh, lhsExt, lhsExt};
  std::array<Value, 4> rhsDigits = {rhsLow, rhsHigh, rhsExt, rhsExt};
  std::array<Value, 4> resultDigits = {cst0, cst0, cst0, cst0};

  for (auto [i, lhsDigit] : llvm::enumerate(lhsDigits)) {
    for (auto [j, rhsDigit] : llvm::enumerate(rhsDigits)) {
      if (i + j >= resultDigits.size())
        continue;

      if (lhsDigit == cst0 || rhsDigit == cst0)
        continue;

      Value &thisResDigit = resultDigits[i + j];
      Value mul = rewriter.create<IMulOp>(loc, lhsDigit, rhsDigit);
      Value current = rewriter.createOrFold<IAddOp>(loc, thisResDigit, mul);
      thisResDigit = getLowDigit(current);

      if (i + j + 1 != resultDigits.size()) {
        Value &nextResDigit = resultDigits[i + j + 1];
        Value carry = rewriter.createOrFold<IAddOp>(loc, nextResDigit,
                                                    getHighDigit(current));
        nextResDigit = carry;
      }
    }
  }

  auto combineDigits = [loc, cst16, &rewriter](Value low, Value high) {
    Value highBits = rewriter.create<ShiftLeftLogicalOp>(loc, high, cst16);
    return rewriter.create<BitwiseOrOp>(loc, low, highBits);
  };
  Value low = combineDigits(resultDigits[0], resultDigits[1]);
  Value high = combineDigits(resultDigits[2], resultDigits[3]);

  return rewriter.create<CompositeConstructOp>(
      loc, mulOp->getResultTypes().front(), llvm::ArrayRef({low, high}));
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

template <typename MulExtendedOp, bool SignExtendArguments>
struct ExpandMulExtendedPattern final : OpRewritePattern<MulExtendedOp> {
  using OpRewritePattern<MulExtendedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();

    // Currently, WGSL only supports 32-bit integer types. Any other integer
    // types should already have been promoted/demoted to i32.
    auto elemTy = cast<IntegerType>(getElementTypeOrSelf(lhs.getType()));
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(
          loc,
          llvm::formatv("Unexpected integer type for WebGPU: '{0}'", elemTy));

    Value mul = lowerExtendedMultiplication(op, rewriter, lhs, rhs,
                                            SignExtendArguments);
    rewriter.replaceOp(op, mul);
    return success();
  }
};

using ExpandSMulExtendedPattern =
    ExpandMulExtendedPattern<SMulExtendedOp, true>;
using ExpandUMulExtendedPattern =
    ExpandMulExtendedPattern<UMulExtendedOp, false>;

struct ExpandAddCarryPattern final : OpRewritePattern<IAddCarryOp> {
  using OpRewritePattern<IAddCarryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IAddCarryOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();

    // Currently, WGSL only supports 32-bit integer types. Any other integer
    // types should already have been promoted/demoted to i32.
    Type argTy = lhs.getType();
    auto elemTy = cast<IntegerType>(getElementTypeOrSelf(argTy));
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(
          loc,
          llvm::formatv("Unexpected integer type for WebGPU: '{0}'", elemTy));

    Value one =
        rewriter.create<ConstantOp>(loc, argTy, getScalarOrSplatAttr(argTy, 1));
    Value zero =
        rewriter.create<ConstantOp>(loc, argTy, getScalarOrSplatAttr(argTy, 0));

    // Calculate the carry by checking if the addition resulted in an overflow.
    Value out = rewriter.create<IAddOp>(loc, lhs, rhs);
    Value cmp = rewriter.create<ULessThanOp>(loc, out, lhs);
    Value carry = rewriter.create<SelectOp>(loc, cmp, one, zero);

    Value add = rewriter.create<CompositeConstructOp>(
        loc, op->getResultTypes().front(), llvm::ArrayRef({out, carry}));

    rewriter.replaceOp(op, add);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//
class WebGPUPreparePass
    : public impl::SPIRVWebGPUPreparePassBase<WebGPUPreparePass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateSPIRVExpandExtendedMultiplicationPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//
void populateSPIRVExpandExtendedMultiplicationPatterns(
    RewritePatternSet &patterns) {
  // WGSL currently does not support extended multiplication ops, see:
  // https://github.com/gpuweb/gpuweb/issues/1565.
  patterns.add<
      // clang-format off
    ExpandSMulExtendedPattern,
    ExpandUMulExtendedPattern,
    ExpandAddCarryPattern
  >(patterns.getContext());
}
} // namespace spirv
} // namespace mlir
