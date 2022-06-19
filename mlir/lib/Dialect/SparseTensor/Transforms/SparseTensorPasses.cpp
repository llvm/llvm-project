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
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Passes declaration.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

struct SparsificationPass : public SparsificationBase<SparsificationPass> {

  SparsificationPass() = default;
  SparsificationPass(const SparsificationPass &pass) = default;
  SparsificationPass(const SparsificationOptions &options) {
    parallelization = static_cast<int32_t>(options.parallelizationStrategy);
    vectorization = static_cast<int32_t>(options.vectorizationStrategy);
    vectorLength = options.vectorLength;
    enableSIMDIndex32 = options.enableSIMDIndex32;
    enableVLAVectorization = options.enableVLAVectorization;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // Translate strategy flags to strategy options.
    SparsificationOptions options(
        sparseParallelizationStrategy(parallelization),
        sparseVectorizationStrategy(vectorization), vectorLength,
        enableSIMDIndex32, enableVLAVectorization);
    // Apply rewriting.
    populateSparsificationPatterns(patterns, options);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

class SparseTensorTypeConverter : public TypeConverter {
public:
  SparseTensorTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertSparseTensorTypes);
  }
  // Maps each sparse tensor type to an opaque pointer.
  static Optional<Type> convertSparseTensorTypes(Type type) {
    if (getSparseTensorEncoding(type) != nullptr)
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
    return llvm::None;
  }
};

struct SparseTensorConversionPass
    : public SparseTensorConversionBase<SparseTensorConversionPass> {

  SparseTensorConversionPass() = default;
  SparseTensorConversionPass(const SparseTensorConversionPass &pass) = default;
  SparseTensorConversionPass(const SparseTensorConversionOptions &options) {
    sparseToSparse = static_cast<int32_t>(options.sparseToSparseStrategy);
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeConverter converter;
    ConversionTarget target(*ctx);
    // Everything in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    // All dynamic rules below accept new function, call, return, and tensor
    // dim and cast operations as legal output of the rewriting provided that
    // all sparse tensor types have been fully rewritten.
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
      return converter.isLegal(op.getOperand().getType());
    });
    // The following operations and dialects may be introduced by the
    // rewriting rules, and are therefore marked as legal.
    target.addLegalOp<arith::CmpFOp, arith::CmpIOp, arith::ConstantOp,
                      arith::IndexCastOp, complex::ConstantOp,
                      complex::NotEqualOp, linalg::FillOp, linalg::YieldOp,
                      tensor::ExtractOp>();
    target
        .addLegalDialect<bufferization::BufferizationDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
    target.addIllegalOp<bufferization::AllocTensorOp>();
    // Translate strategy flags to strategy options.
    SparseTensorConversionOptions options(
        sparseToSparseConversionStrategy(sparseToSparse));
    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateSparseTensorConversionPatterns(converter, patterns, options);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

SparseParallelizationStrategy
mlir::sparseParallelizationStrategy(int32_t flag) {
  switch (flag) {
  default:
    return SparseParallelizationStrategy::kNone;
  case 1:
    return SparseParallelizationStrategy::kDenseOuterLoop;
  case 2:
    return SparseParallelizationStrategy::kAnyStorageOuterLoop;
  case 3:
    return SparseParallelizationStrategy::kDenseAnyLoop;
  case 4:
    return SparseParallelizationStrategy::kAnyStorageAnyLoop;
  }
}

SparseVectorizationStrategy mlir::sparseVectorizationStrategy(int32_t flag) {
  switch (flag) {
  default:
    return SparseVectorizationStrategy::kNone;
  case 1:
    return SparseVectorizationStrategy::kDenseInnerLoop;
  case 2:
    return SparseVectorizationStrategy::kAnyStorageInnerLoop;
  }
}

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
