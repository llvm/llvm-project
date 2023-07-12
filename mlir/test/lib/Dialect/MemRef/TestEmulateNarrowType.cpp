//===- TestEmulateNarrowType.cpp - Test Narrow Type Emulation  ------*- c++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct TestEmulateNarrowTypePass
    : public PassWrapper<TestEmulateNarrowTypePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEmulateNarrowTypePass)

  TestEmulateNarrowTypePass() = default;
  TestEmulateNarrowTypePass(const TestEmulateNarrowTypePass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                vector::VectorDialect, affine::AffineDialect>();
  }
  StringRef getArgument() const final { return "test-emulate-narrow-int"; }
  StringRef getDescription() const final {
    return "Function pass to test Narrow Integer Emulation";
  }

  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(loadStoreEmulateBitwidth) ||
        loadStoreEmulateBitwidth < 8) {
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    arith::NarrowTypeEmulationConverter typeConverter(loadStoreEmulateBitwidth);

    // Convert scalar type.
    typeConverter.addConversion([this](IntegerType ty) -> std::optional<Type> {
      unsigned width = ty.getWidth();
      if (width >= arithComputeBitwidth)
        return ty;

      return IntegerType::get(ty.getContext(), arithComputeBitwidth);
    });

    // Convert vector type.
    typeConverter.addConversion([this](VectorType ty) -> std::optional<Type> {
      auto intTy = dyn_cast<IntegerType>(ty.getElementType());
      if (!intTy)
        return ty;

      unsigned width = intTy.getWidth();
      if (width >= arithComputeBitwidth)
        return ty;

      return VectorType::get(
          to_vector(ty.getShape()),
          IntegerType::get(ty.getContext(), arithComputeBitwidth));
    });

    memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
        affine::AffineDialect>(
        [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(ctx);

    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
    vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<unsigned> loadStoreEmulateBitwidth{
      *this, "memref-load-bitwidth",
      llvm::cl::desc("memref load/store emulation bit width"),
      llvm::cl::init(8)};

  Option<unsigned> arithComputeBitwidth{
      *this, "arith-compute-bitwidth",
      llvm::cl::desc("arith computation bit width"), llvm::cl::init(4)};
};
} // namespace

namespace mlir::test {
void registerTestEmulateNarrowTypePass() {
  PassRegistration<TestEmulateNarrowTypePass>();
}
} // namespace mlir::test
