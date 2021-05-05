//===- SparsificationPass.cpp - Pass for autogen spares tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

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
  SparsificationPass(const SparsificationPass &pass) {}

  Option<int32_t> parallelization{
      *this, "parallelization-strategy",
      llvm::cl::desc("Set the parallelization strategy"), llvm::cl::init(0)};

  Option<int32_t> vectorization{
      *this, "vectorization-strategy",
      llvm::cl::desc("Set the vectorization strategy"), llvm::cl::init(0)};

  Option<int32_t> vectorLength{
      *this, "vl", llvm::cl::desc("Set the vector length"), llvm::cl::init(1)};

  Option<int32_t> ptrType{*this, "ptr-type",
                          llvm::cl::desc("Set the pointer type"),
                          llvm::cl::init(0)};

  Option<int32_t> indType{*this, "ind-type",
                          llvm::cl::desc("Set the index type"),
                          llvm::cl::init(0)};

  Option<bool> fastOutput{*this, "fast-output",
                          llvm::cl::desc("Allows fast output buffers"),
                          llvm::cl::init(false)};

  /// Returns parallelization strategy given on command line.
  SparseParallelizationStrategy parallelOption() {
    switch (parallelization) {
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

  /// Returns vectorization strategy given on command line.
  SparseVectorizationStrategy vectorOption() {
    switch (vectorization) {
    default:
      return SparseVectorizationStrategy::kNone;
    case 1:
      return SparseVectorizationStrategy::kDenseInnerLoop;
    case 2:
      return SparseVectorizationStrategy::kAnyStorageInnerLoop;
    }
  }

  /// Returns the requested integer type.
  SparseIntType typeOption(int32_t option) {
    switch (option) {
    default:
      return SparseIntType::kNative;
    case 1:
      return SparseIntType::kI64;
    case 2:
      return SparseIntType::kI32;
    case 3:
      return SparseIntType::kI16;
    case 4:
      return SparseIntType::kI8;
    }
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // Translate strategy flags to strategy options.
    SparsificationOptions options(parallelOption(), vectorOption(),
                                  vectorLength, typeOption(ptrType),
                                  typeOption(indType), fastOutput);
    // Apply rewriting.
    populateSparsificationPatterns(patterns, options);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct SparseTensorConversionPass
    : public SparseTensorConversionBase<SparseTensorConversionPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet conversionPatterns(ctx);
    ConversionTarget target(*ctx);
    target
        .addIllegalOp<sparse_tensor::FromPointerOp, sparse_tensor::ToPointersOp,
                      sparse_tensor::ToIndicesOp, sparse_tensor::ToValuesOp>();
    target.addLegalOp<CallOp>();
    populateSparseTensorConversionPatterns(conversionPatterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(conversionPatterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createSparsificationPass() {
  return std::make_unique<SparsificationPass>();
}

std::unique_ptr<Pass> mlir::createSparseTensorConversionPass() {
  return std::make_unique<SparseTensorConversionPass>();
}
