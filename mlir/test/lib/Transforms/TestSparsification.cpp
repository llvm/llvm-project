//===- TestSparsification.cpp - Test sparsification of tensors ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestSparsification
    : public PassWrapper<TestSparsification, FunctionPass> {

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

  /// Registers all dialects required by testing.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
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
    }
  }

  /// Runs the test on a function.
  void runOnFunction() override {
    auto *ctx = &getContext();
    OwningRewritePatternList patterns;
    // Translate strategy flags to strategy options.
    linalg::SparsificationOptions options(parallelOption(), vectorOption(),
                                          vectorLength, typeOption(ptrType),
                                          typeOption(indType));
    // Apply rewriting.
    linalg::populateSparsificationPatterns(ctx, patterns, options);
    applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {

void registerTestSparsification() {
  PassRegistration<TestSparsification> sparsificationPass(
      "test-sparsification",
      "Test automatic geneneration of sparse tensor code");
}

} // namespace test
} // namespace mlir
