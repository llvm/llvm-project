//===- TestSparsification.cpp - Test sparsification of tensors ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestSparsification
    : public PassWrapper<TestSparsification, OperationPass<ModuleOp>> {

  TestSparsification() = default;
  TestSparsification(const TestSparsification &pass) {}

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

  Option<bool> lower{*this, "lower", llvm::cl::desc("Lower sparse primitives"),
                     llvm::cl::init(false)};

  /// Registers all dialects required by testing.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect, LLVM::LLVMDialect>();
  }

  /// Returns parallelization strategy given on command line.
  linalg::SparseParallelizationStrategy parallelOption() {
    switch (parallelization) {
    default:
      return linalg::SparseParallelizationStrategy::kNone;
    case 1:
      return linalg::SparseParallelizationStrategy::kDenseOuterLoop;
    case 2:
      return linalg::SparseParallelizationStrategy::kAnyStorageOuterLoop;
    case 3:
      return linalg::SparseParallelizationStrategy::kDenseAnyLoop;
    case 4:
      return linalg::SparseParallelizationStrategy::kAnyStorageAnyLoop;
    }
  }

  /// Returns vectorization strategy given on command line.
  linalg::SparseVectorizationStrategy vectorOption() {
    switch (vectorization) {
    default:
      return linalg::SparseVectorizationStrategy::kNone;
    case 1:
      return linalg::SparseVectorizationStrategy::kDenseInnerLoop;
    case 2:
      return linalg::SparseVectorizationStrategy::kAnyStorageInnerLoop;
    }
  }

  /// Returns the requested integer type.
  linalg::SparseIntType typeOption(int32_t option) {
    switch (option) {
    default:
      return linalg::SparseIntType::kNative;
    case 1:
      return linalg::SparseIntType::kI64;
    case 2:
      return linalg::SparseIntType::kI32;
    case 3:
      return linalg::SparseIntType::kI16;
    case 4:
      return linalg::SparseIntType::kI8;
    }
  }

  /// Runs the test on a function.
  void runOnOperation() override {
    auto *ctx = &getContext();
    OwningRewritePatternList patterns;
    // Translate strategy flags to strategy options.
    linalg::SparsificationOptions options(parallelOption(), vectorOption(),
                                          vectorLength, typeOption(ptrType),
                                          typeOption(indType), fastOutput);
    // Apply rewriting.
    linalg::populateSparsificationPatterns(ctx, patterns, options);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    // Lower sparse primitives to calls into runtime support library.
    if (lower) {
      OwningRewritePatternList conversionPatterns;
      ConversionTarget target(*ctx);
      target.addIllegalOp<linalg::SparseTensorFromPointerOp,
                          linalg::SparseTensorToPointersMemRefOp,
                          linalg::SparseTensorToIndicesMemRefOp,
                          linalg::SparseTensorToValuesMemRefOp>();
      target.addLegalOp<CallOp>();
      linalg::populateSparsificationConversionPatterns(ctx, conversionPatterns);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(conversionPatterns))))
        signalPassFailure();
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {

void registerTestSparsification() {
  PassRegistration<TestSparsification> sparsificationPass(
      "test-sparsification", "Test automatic generation of sparse tensor code");
}

} // namespace test
} // namespace mlir
