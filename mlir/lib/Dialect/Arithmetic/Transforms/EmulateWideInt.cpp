//===- EmulateWideInt.cpp - Wide integer operation emulation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/Transforms/WideIntEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHMETICEMULATEWIDEINT
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;

namespace {
struct EmulateWideIntPass final
    : arith::impl::ArithmeticEmulateWideIntBase<EmulateWideIntPass> {
  using ArithmeticEmulateWideIntBase::ArithmeticEmulateWideIntBase;

  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(widestIntSupported)) {
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    arith::WideIntEmulationConverter typeConverter(widestIntSupported);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(ctx);
    arith::populateWideIntEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

arith::WideIntEmulationConverter::WideIntEmulationConverter(
    unsigned widestIntSupportedByTarget)
    : maxIntWidth(widestIntSupportedByTarget) {
  assert(llvm::isPowerOf2_32(widestIntSupportedByTarget) &&
         "Only power-of-two integers are supported");

  // Scalar case.
  addConversion([this](IntegerType ty) -> Optional<Type> {
    unsigned width = ty.getWidth();
    if (width <= maxIntWidth)
      return ty;

    // i2N --> vector<2xiN>
    if (width == 2 * maxIntWidth)
      return VectorType::get(2, IntegerType::get(ty.getContext(), maxIntWidth));

    return None;
  });

  // Vector case.
  addConversion([this](VectorType ty) -> Optional<Type> {
    auto intTy = ty.getElementType().dyn_cast<IntegerType>();
    if (!intTy)
      return ty;

    unsigned width = intTy.getWidth();
    if (width <= maxIntWidth)
      return ty;

    // vector<...xi2N> --> vector<...x2xiN>
    if (width == 2 * maxIntWidth) {
      auto newShape = to_vector(ty.getShape());
      newShape.push_back(2);
      return VectorType::get(newShape,
                             IntegerType::get(ty.getContext(), maxIntWidth));
    }

    return None;
  });

  // Function case.
  addConversion([this](FunctionType ty) -> Optional<Type> {
    // Convert inputs and results, e.g.:
    //   (i2N, i2N) -> i2N --> (vector<2xiN>, vector<2xiN>) -> vector<2xiN>
    SmallVector<Type> inputs;
    if (failed(convertTypes(ty.getInputs(), inputs)))
      return None;

    SmallVector<Type> results;
    if (failed(convertTypes(ty.getResults(), results)))
      return None;

    return FunctionType::get(ty.getContext(), inputs, results);
  });
}

void arith::populateWideIntEmulationPatterns(
    WideIntEmulationConverter &typeConverter, RewritePatternSet &patterns) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}
