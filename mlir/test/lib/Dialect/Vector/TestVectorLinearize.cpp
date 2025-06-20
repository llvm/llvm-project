//===- TestVectorLinearize.cpp - Test Vector linearization ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math//IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorLinearize.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::vector;

namespace {

struct TestVectorLinearize final
    : public PassWrapper<TestVectorLinearize, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorLinearize)

  TestVectorLinearize() = default;

  StringRef getArgument() const override { return "test-vector-linearize"; }
  StringRef getDescription() const override {
    return "Linearizes ND vectors for N >= 2 into 1D vectors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect, arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter converter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    initializeForVectorLinearize(converter);
    populateForFullVectorLinearize(converter, target, patterns);

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

struct TestVectorBitWidthLinearize final
    : public PassWrapper<TestVectorBitWidthLinearize, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorBitWidthLinearize)

  TestVectorBitWidthLinearize() = default;
  TestVectorBitWidthLinearize(const TestVectorBitWidthLinearize &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const override {
    return "test-bit-width-constrained-vector-linearize";
  }
  StringRef getDescription() const override {
    return "Linearizes ND vectors for N >= 2 into 1D vectors, with constraints "
           "on inner-most dimension's bit width. If the inner-most dimension "
           "exceded a threshold, the op is not linearized.";
  }
  Option<unsigned> targetVectorBitwidth{
      *this, "target-vector-bitwidth",
      llvm::cl::desc(
          "Minimum vector bitwidth to enable the flattening transformation"),
      llvm::cl::init(std::numeric_limits<unsigned>::max())};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter typeConverter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    populateWithBitWidthConstraints(typeConverter, target, patterns,
                                    targetVectorBitwidth);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

private:
  /// If `type` is VectorType with trailing dimension of (bit) size greater than
  /// or equal to `targetBitWidth`, its defining op is considered legal.
  static bool
  isNotLinearizableBecauseLargeInnerDimension(Type type,
                                              unsigned targetBitWidth) {

    VectorType vecType = dyn_cast<VectorType>(type);

    // Not linearizable for reasons other than what this function checks.
    if (!vecType || vecType.getRank() == 0)
      return false;

    // The width of the type 'index' is unbounded (and therefore potentially
    // above the target width).
    if (vecType.getElementType().isIndex())
      return true;

    unsigned finalDimSize = vecType.getShape().back();
    unsigned nbBitsPerElm = vecType.getElementTypeBitWidth();
    unsigned trailingVecDimBitWidth = finalDimSize * nbBitsPerElm;
    return trailingVecDimBitWidth >= targetBitWidth;
  }

  static bool
  isNotLinearizableBecauseLargeInnerDimension(Operation *op,
                                              unsigned targetBitWidth) {
    // Check on bitwidths.
    SmallVector<std::pair<Type, unsigned>> toCheck =
        getTypeBitWidthBoundPairs(op, targetBitWidth);
    return std::any_of(toCheck.begin(), toCheck.end(),
                       [&](std::pair<Type, unsigned> typeWidth) {
                         return isNotLinearizableBecauseLargeInnerDimension(
                             typeWidth.first, typeWidth.second);
                       });
  }

  static void populateWithBitWidthConstraints(TypeConverter &typeConverter,
                                              ConversionTarget &target,
                                              RewritePatternSet &patterns,
                                              unsigned targetBitWidth) {

    initializeForVectorLinearize(typeConverter);
    populateForFullVectorLinearize(typeConverter, target, patterns);

    // Extend the set of legal ops to include those with large inner-most
    // dimensions on selected operands/results.
    target.markUnknownOpDynamicallyLegal(
        [=](Operation *op) -> std::optional<bool> {
          if (isNotLinearizableBecauseLargeInnerDimension(op, targetBitWidth)) {
            return true;
          }
          return {};
        });
  }

  /// Get the set of operand/result types to check for sufficiently
  /// small inner-most dimension size.
  static SmallVector<std::pair<Type, unsigned>>
  getTypeBitWidthBoundPairs(Operation *op, unsigned targetBitWidth) {

    if (auto insertOp = dyn_cast<InsertOp>(op)) {
      unsigned w = targetBitWidth < std::numeric_limits<unsigned>::max()
                       ? targetBitWidth + 1
                       : targetBitWidth;
      return {{insertOp.getValueToStoreType(), w}};
    }

    auto resultTypes = op->getResultTypes();
    SmallVector<std::pair<Type, unsigned>> resultsWithBitWidth;
    resultsWithBitWidth.reserve(resultTypes.size());
    for (Type type : resultTypes) {
      resultsWithBitWidth.push_back({type, targetBitWidth});
    }
    return resultsWithBitWidth;
  }
};

} // namespace

namespace mlir {
namespace test {
extern void registerTestVectorLinearize() {
  PassRegistration<TestVectorLinearize>();
  PassRegistration<TestVectorBitWidthLinearize>();
}
} // namespace test
} // namespace mlir
