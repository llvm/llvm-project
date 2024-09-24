//===- EmulateUnsupportedFloats.cpp - Promote small floats --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass promotes small floats (of some unsupported types T) to a supported
// type U by wrapping all float operations on Ts with expansion to and
// truncation from U, then operating on U.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHEMULATEUNSUPPORTEDFLOATS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;

namespace {
struct EmulateUnsupportedFloatsPass
    : arith::impl::ArithEmulateUnsupportedFloatsBase<
          EmulateUnsupportedFloatsPass> {
  using arith::impl::ArithEmulateUnsupportedFloatsBase<
      EmulateUnsupportedFloatsPass>::ArithEmulateUnsupportedFloatsBase;

  void runOnOperation() override;
};

struct EmulateFloatPattern final : ConversionPattern {
  EmulateFloatPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult match(Operation *op) const override;
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

/// Map strings to float types. This function is here because no one else needs
/// it yet, feel free to abstract it out.
static std::optional<FloatType> parseFloatType(MLIRContext *ctx,
                                               StringRef name) {
  Builder b(ctx);
  return llvm::StringSwitch<std::optional<FloatType>>(name)
      .Case("f4E2M1FN", b.getFloat4E2M1FNType())
      .Case("f6E2M3FN", b.getFloat6E2M3FNType())
      .Case("f6E3M2FN", b.getFloat6E3M2FNType())
      .Case("f8E5M2", b.getFloat8E5M2Type())
      .Case("f8E4M3", b.getFloat8E4M3Type())
      .Case("f8E4M3FN", b.getFloat8E4M3FNType())
      .Case("f8E5M2FNUZ", b.getFloat8E5M2FNUZType())
      .Case("f8E4M3FNUZ", b.getFloat8E4M3FNUZType())
      .Case("f8E3M4", b.getFloat8E3M4Type())
      .Case("bf16", b.getBF16Type())
      .Case("f16", b.getF16Type())
      .Case("f32", b.getF32Type())
      .Case("f64", b.getF64Type())
      .Case("f80", b.getF80Type())
      .Case("f128", b.getF128Type())
      .Default(std::nullopt);
}

LogicalResult EmulateFloatPattern::match(Operation *op) const {
  if (getTypeConverter()->isLegal(op))
    return failure();
  // The rewrite doesn't handle cloning regions.
  if (op->getNumRegions() != 0)
    return failure();
  return success();
}

void EmulateFloatPattern::rewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  const TypeConverter *converter = getTypeConverter();
  SmallVector<Type> resultTypes;
  if (failed(converter->convertTypes(op->getResultTypes(), resultTypes))) {
    // Note to anyone looking for this error message: this is a "can't happen".
    // If you're seeing it, there's a bug.
    op->emitOpError("type conversion failed in float emulation");
    return;
  }
  Operation *expandedOp =
      rewriter.create(loc, op->getName().getIdentifier(), operands, resultTypes,
                      op->getAttrs(), op->getSuccessors(), /*regions=*/{});
  SmallVector<Value> newResults(expandedOp->getResults());
  for (auto [res, oldType, newType] : llvm::zip_equal(
           MutableArrayRef{newResults}, op->getResultTypes(), resultTypes)) {
    if (oldType != newType) {
      auto truncFOp = rewriter.create<arith::TruncFOp>(loc, oldType, res);
      truncFOp.setFastmath(arith::FastMathFlags::contract);
      res = truncFOp.getResult();
    }
  }
  rewriter.replaceOp(op, newResults);
}

void mlir::arith::populateEmulateUnsupportedFloatsConversions(
    TypeConverter &converter, ArrayRef<Type> sourceTypes, Type targetType) {
  converter.addConversion([sourceTypes = SmallVector<Type>(sourceTypes),
                           targetType](Type type) -> std::optional<Type> {
    if (llvm::is_contained(sourceTypes, type))
      return targetType;
    if (auto shaped = dyn_cast<ShapedType>(type))
      if (llvm::is_contained(sourceTypes, shaped.getElementType()))
        return shaped.clone(targetType);
    // All other types legal
    return type;
  });
  converter.addTargetMaterialization(
      [](OpBuilder &b, Type target, ValueRange input, Location loc) {
        auto extFOp = b.create<arith::ExtFOp>(loc, target, input);
        extFOp.setFastmath(arith::FastMathFlags::contract);
        return extFOp;
      });
}

void mlir::arith::populateEmulateUnsupportedFloatsPatterns(
    RewritePatternSet &patterns, TypeConverter &converter) {
  patterns.add<EmulateFloatPattern>(converter, patterns.getContext());
}

void mlir::arith::populateEmulateUnsupportedFloatsLegality(
    ConversionTarget &target, TypeConverter &converter) {
  // Don't try to legalize functions and other ops that don't need expansion.
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  target.addDynamicallyLegalDialect<arith::ArithDialect>(
      [&](Operation *op) -> std::optional<bool> {
        return converter.isLegal(op);
      });
  // Manually mark arithmetic-performing vector instructions.
  target.addDynamicallyLegalOp<
      vector::ContractionOp, vector::ReductionOp, vector::MultiDimReductionOp,
      vector::FMAOp, vector::OuterProductOp, vector::MatmulOp, vector::ScanOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addLegalOp<arith::BitcastOp, arith::ExtFOp, arith::TruncFOp,
                    arith::ConstantOp, vector::SplatOp>();
}

void EmulateUnsupportedFloatsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *op = getOperation();
  SmallVector<Type> sourceTypes;
  Type targetType;

  std::optional<FloatType> maybeTargetType = parseFloatType(ctx, targetTypeStr);
  if (!maybeTargetType) {
    emitError(UnknownLoc::get(ctx), "could not map target type '" +
                                        targetTypeStr +
                                        "' to a known floating-point type");
    return signalPassFailure();
  }
  targetType = *maybeTargetType;
  for (StringRef sourceTypeStr : sourceTypeStrs) {
    std::optional<FloatType> maybeSourceType =
        parseFloatType(ctx, sourceTypeStr);
    if (!maybeSourceType) {
      emitError(UnknownLoc::get(ctx), "could not map source type '" +
                                          sourceTypeStr +
                                          "' to a known floating-point type");
      return signalPassFailure();
    }
    sourceTypes.push_back(*maybeSourceType);
  }
  if (sourceTypes.empty())
    (void)emitOptionalWarning(
        std::nullopt,
        "no source types specified, float emulation will do nothing");

  if (llvm::is_contained(sourceTypes, targetType)) {
    emitError(UnknownLoc::get(ctx),
              "target type cannot be an unsupported source type");
    return signalPassFailure();
  }
  TypeConverter converter;
  arith::populateEmulateUnsupportedFloatsConversions(converter, sourceTypes,
                                                     targetType);
  RewritePatternSet patterns(ctx);
  arith::populateEmulateUnsupportedFloatsPatterns(patterns, converter);
  ConversionTarget target(getContext());
  arith::populateEmulateUnsupportedFloatsLegality(target, converter);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}
