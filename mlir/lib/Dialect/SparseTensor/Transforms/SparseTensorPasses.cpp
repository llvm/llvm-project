//===- SparseTensorPasses.cpp - Pass for autogen sparse tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SPARSIFICATIONPASS
#define GEN_PASS_DEF_SPARSETENSORCONVERSIONPASS
#define GEN_PASS_DEF_SPARSETENSORCODEGEN
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

struct SparsificationPass
    : public impl::SparsificationPassBase<SparsificationPass> {

  SparsificationPass() = default;
  SparsificationPass(const SparsificationPass &pass) = default;
  SparsificationPass(const SparsificationOptions &options) {
    parallelization = options.parallelizationStrategy;
    vectorization = options.vectorizationStrategy;
    vectorLength = options.vectorLength;
    enableSIMDIndex32 = options.enableSIMDIndex32;
    enableVLAVectorization = options.enableVLAVectorization;
    enableRuntimeLibrary = options.enableRuntimeLibrary;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet prePatterns(ctx);
    // Translate strategy flags to strategy options.
    SparsificationOptions options(parallelization, vectorization, vectorLength,
                                  enableSIMDIndex32, enableVLAVectorization,
                                  enableRuntimeLibrary);
    // Apply pre-rewriting.
    populateSparseTensorRewriting(prePatterns, options.enableRuntimeLibrary);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(prePatterns));
    // Apply sparsification and vector cleanup rewriting.
    RewritePatternSet patterns(ctx);
    populateSparsificationPatterns(patterns, options);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct SparseTensorConversionPass
    : public impl::SparseTensorConversionPassBase<SparseTensorConversionPass> {

  SparseTensorConversionPass() = default;
  SparseTensorConversionPass(const SparseTensorConversionPass &pass) = default;
  SparseTensorConversionPass(const SparseTensorConversionOptions &options) {
    sparseToSparse = static_cast<int32_t>(options.sparseToSparseStrategy);
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeToPtrConverter converter;
    ConversionTarget target(*ctx);
    // Everything in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    // All dynamic rules below accept new function, call, return, and various
    // tensor and bufferization operations as legal output of the rewriting
    // provided that all sparse tensor types have been fully rewritten.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<tensor::DimOp>([&](tensor::DimOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<tensor::CastOp>([&](tensor::CastOp op) {
      return converter.isLegal(op.getSource().getType()) &&
             converter.isLegal(op.getDest().getType());
    });
    target.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
        [&](tensor::ExpandShapeOp op) {
          return converter.isLegal(op.getSrc().getType()) &&
                 converter.isLegal(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
        [&](tensor::CollapseShapeOp op) {
          return converter.isLegal(op.getSrc().getType()) &&
                 converter.isLegal(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
        [&](bufferization::AllocTensorOp op) {
          return converter.isLegal(op.getType());
        });
    target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
        [&](bufferization::DeallocTensorOp op) {
          return converter.isLegal(op.getTensor().getType());
        });
    // The following operations and dialects may be introduced by the
    // rewriting rules, and are therefore marked as legal.
    target.addLegalOp<complex::ConstantOp, complex::NotEqualOp, linalg::FillOp,
                      linalg::YieldOp, tensor::ExtractOp>();
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();
    // Translate strategy flags to strategy options.
    SparseTensorConversionOptions options(
        sparseToSparseConversionStrategy(sparseToSparse));
    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateSparseTensorConversionPatterns(converter, patterns, options);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct SparseTensorCodegenPass
    : public impl::SparseTensorCodegenBase<SparseTensorCodegenPass> {

  SparseTensorCodegenPass() = default;
  SparseTensorCodegenPass(const SparseTensorCodegenPass &pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeToBufferConverter converter;
    ConversionTarget target(*ctx);
    // Everything in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    // All dynamic rules below accept new function, call, return, and various
    // tensor and bufferization operations as legal output of the rewriting
    // provided that all sparse tensor types have been fully rewritten.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
        [&](bufferization::AllocTensorOp op) {
          return converter.isLegal(op.getType());
        });
    target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
        [&](bufferization::DeallocTensorOp op) {
          return converter.isLegal(op.getTensor().getType());
        });
    // The following operations and dialects may be introduced by the
    // codegen rules, and are therefore marked as legal.
    target.addLegalOp<linalg::FillOp>();
    target.addLegalDialect<arith::ArithmeticDialect,
                           bufferization::BufferizationDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateSparseTensorCodegenPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Strategy flag methods.
//===----------------------------------------------------------------------===//

SparseToSparseConversionStrategy
mlir::sparseToSparseConversionStrategy(int32_t flag) {
  switch (flag) {
  default:
    return SparseToSparseConversionStrategy::kAuto;
  case 1:
    return SparseToSparseConversionStrategy::kViaCOO;
  case 2:
    return SparseToSparseConversionStrategy::kDirect;
  }
}

//===----------------------------------------------------------------------===//
// Pass creation methods.
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createSparsificationPass() {
  return std::make_unique<SparsificationPass>();
}

std::unique_ptr<Pass>
mlir::createSparsificationPass(const SparsificationOptions &options) {
  return std::make_unique<SparsificationPass>(options);
}

std::unique_ptr<Pass> mlir::createSparseTensorConversionPass() {
  return std::make_unique<SparseTensorConversionPass>();
}

std::unique_ptr<Pass> mlir::createSparseTensorConversionPass(
    const SparseTensorConversionOptions &options) {
  return std::make_unique<SparseTensorConversionPass>(options);
}

std::unique_ptr<Pass> mlir::createSparseTensorCodegenPass() {
  return std::make_unique<SparseTensorCodegenPass>();
}
