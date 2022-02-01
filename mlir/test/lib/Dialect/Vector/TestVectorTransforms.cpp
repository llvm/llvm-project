//===- TestVectorTransforms.cpp - Test Vector transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;

namespace {

struct TestVectorToVectorLowering
    : public PassWrapper<TestVectorToVectorLowering, OperationPass<FuncOp>> {
  TestVectorToVectorLowering() = default;
  TestVectorToVectorLowering(const TestVectorToVectorLowering &pass)
      : PassWrapper(pass) {}
  StringRef getArgument() const final {
    return "test-vector-to-vector-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns between ops in the vector dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  Option<bool> unroll{*this, "unroll", llvm::cl::desc("Include unrolling"),
                      llvm::cl::init(false)};

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    if (unroll) {
      populateVectorUnrollPatterns(
          patterns,
          UnrollVectorOptions().setNativeShapeFn(getShape).setFilterConstraint(
              filter));
    }
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateBubbleVectorBitCastOpPatterns(patterns);
    populateCastAwayVectorLeadingOneDimPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  // Return the target shape based on op type.
  static Optional<SmallVector<int64_t, 4>> getShape(Operation *op) {
    if (isa<arith::AddFOp, SelectOp, arith::CmpFOp>(op))
      return SmallVector<int64_t, 4>(2, 2);
    if (isa<vector::ContractionOp>(op))
      return SmallVector<int64_t, 4>(3, 2);
    // For transfer ops, just propagate the shape coming from
    // InsertStridedSlices/ExtractStridedSlices.
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      VectorType dstVec;
      for (Operation *users : readOp->getUsers()) {
        auto extract = dyn_cast<ExtractStridedSliceOp>(users);
        if (!extract)
          return llvm::None;
        auto vecType = extract.getResult().getType().cast<VectorType>();
        if (dstVec && dstVec != vecType)
          return llvm::None;
        dstVec = vecType;
      }
      return SmallVector<int64_t, 4>(dstVec.getShape().begin(),
                                     dstVec.getShape().end());
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      auto insert = writeOp.vector().getDefiningOp<InsertStridedSliceOp>();
      if (!insert)
        return llvm::None;
      ArrayRef<int64_t> shape = insert.getSourceVectorType().getShape();
      return SmallVector<int64_t, 4>(shape.begin(), shape.end());
    }
    return llvm::None;
  }

  static LogicalResult filter(Operation *op) {
    return success(isa<arith::AddFOp, SelectOp, arith::CmpFOp, ContractionOp,
                       TransferReadOp, TransferWriteOp>(op));
  }
};

struct TestVectorContractionLowering
    : public PassWrapper<TestVectorContractionLowering, OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-contraction-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns that lower contract ops in the vector "
           "dialect";
  }
  TestVectorContractionLowering() = default;
  TestVectorContractionLowering(const TestVectorContractionLowering &pass)
      : PassWrapper(pass) {}

  Option<bool> lowerToFlatMatrix{
      *this, "vector-lower-matrix-intrinsics",
      llvm::cl::desc("Lower vector.contract to llvm.intr.matrix.multiply"),
      llvm::cl::init(false)};
  Option<bool> lowerToOuterProduct{
      *this, "vector-outerproduct",
      llvm::cl::desc("Lower vector.contract to vector.outerproduct"),
      llvm::cl::init(false)};
  Option<bool> lowerToFilterOuterProduct{
      *this, "vector-filter-outerproduct",
      llvm::cl::desc("Lower vector.contract to vector.outerproduct but not for "
                     "vectors of size 4."),
      llvm::cl::init(false)};

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    // Test on one pattern in isolation.
    if (lowerToOuterProduct) {
      VectorContractLowering lowering = VectorContractLowering::OuterProduct;
      VectorTransformsOptions options{lowering};
      patterns.add<ContractionOpToOuterProductOpLowering>(options,
                                                          &getContext());
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
      return;
    }

    // Test on one pattern in isolation.
    if (lowerToFilterOuterProduct) {
      VectorContractLowering lowering = VectorContractLowering::OuterProduct;
      VectorTransformsOptions options{lowering};
      patterns.add<ContractionOpToOuterProductOpLowering>(
          options, &getContext(), [](vector::ContractionOp op) {
            // Only lowers vector.contract where the lhs as a type vector<MxNx?>
            // where M is not 4.
            if (op.getRhsType().getShape()[0] == 4)
              return failure();
            return success();
          });
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
      return;
    }

    // Test on all contract lowering patterns.
    VectorContractLowering contractLowering = VectorContractLowering::Dot;
    if (lowerToFlatMatrix)
      contractLowering = VectorContractLowering::Matmul;
    VectorMultiReductionLowering vectorMultiReductionLowering =
        VectorMultiReductionLowering::InnerParallel;
    VectorTransformsOptions options{contractLowering,
                                    vectorMultiReductionLowering,
                                    VectorTransposeLowering()};
    populateVectorBroadcastLoweringPatterns(patterns);
    populateVectorContractLoweringPatterns(patterns, options);
    populateVectorMaskOpLoweringPatterns(patterns);
    populateVectorShapeCastLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransposeLowering
    : public PassWrapper<TestVectorTransposeLowering, OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-transpose-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns that lower contract ops in the vector "
           "dialect";
  }
  TestVectorTransposeLowering() = default;
  TestVectorTransposeLowering(const TestVectorTransposeLowering &pass)
      : PassWrapper(pass) {}

  Option<bool> lowerToEltwise{
      *this, "eltwise",
      llvm::cl::desc("Lower 2-D vector.transpose to eltwise insert/extract"),
      llvm::cl::init(false)};
  Option<bool> lowerToFlatTranspose{
      *this, "flat",
      llvm::cl::desc("Lower 2-D vector.transpose to vector.flat_transpose"),
      llvm::cl::init(false)};
  Option<bool> lowerToShuffleTranspose{
      *this, "shuffle",
      llvm::cl::desc("Lower 2-D vector.transpose to shape_cast + shuffle"),
      llvm::cl::init(false)};
  Option<bool> lowerToAvx2{
      *this, "avx2",
      llvm::cl::desc("Lower vector.transpose to avx2-specific patterns"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    // Test on one pattern in isolation.
    // Explicitly disable shape_cast lowering.
    LinalgVectorLoweringOptions options = LinalgVectorLoweringOptions()
                                              .enableVectorTransposeLowering()
                                              .enableShapeCastLowering(false);
    if (lowerToEltwise) {
      options = options.setVectorTransformsOptions(
          VectorTransformsOptions().setVectorTransposeLowering(
              VectorTransposeLowering::EltWise));
    }
    if (lowerToFlatTranspose) {
      options = options.setVectorTransformsOptions(
          VectorTransformsOptions().setVectorTransposeLowering(
              VectorTransposeLowering::Flat));
    }
    if (lowerToShuffleTranspose) {
      options = options.setVectorTransformsOptions(
          VectorTransformsOptions().setVectorTransposeLowering(
              VectorTransposeLowering::Shuffle));
    }
    if (lowerToAvx2) {
      options = options.enableAVX2Lowering().setAVX2LoweringOptions(
          x86vector::avx2::LoweringOptions().setTransposeOptions(
              x86vector::avx2::TransposeLoweringOptions()
                  .lower4x8xf32()
                  .lower8x8xf32()));
    }

    OpPassManager dynamicPM("builtin.func");
    dynamicPM.addPass(createLinalgStrategyLowerVectorsPass(options));
    if (failed(runPipeline(dynamicPM, getOperation())))
      return signalPassFailure();
  }
};

struct TestVectorUnrollingPatterns
    : public PassWrapper<TestVectorUnrollingPatterns, OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-unrolling-patterns";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns to unroll contract ops in the vector "
           "dialect";
  }
  TestVectorUnrollingPatterns() = default;
  TestVectorUnrollingPatterns(const TestVectorUnrollingPatterns &pass)
      : PassWrapper(pass) {}
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateVectorUnrollPatterns(
        patterns, UnrollVectorOptions()
                      .setNativeShape(ArrayRef<int64_t>{2, 2})
                      .setFilterConstraint([](Operation *op) {
                        return success(isa<arith::AddFOp, vector::FMAOp>(op));
                      }));

    if (unrollBasedOnType) {
      UnrollVectorOptions::NativeShapeFnType nativeShapeFn =
          [](Operation *op) -> Optional<SmallVector<int64_t, 4>> {
        vector::ContractionOp contractOp = cast<vector::ContractionOp>(op);
        SmallVector<int64_t, 4> nativeShape = {4, 4, 2};
        if (auto floatType = contractOp.getLhsType()
                                 .getElementType()
                                 .dyn_cast<FloatType>()) {
          if (floatType.getWidth() == 16) {
            nativeShape[2] = 4;
          }
        }
        return nativeShape;
      };
      populateVectorUnrollPatterns(patterns,
                                   UnrollVectorOptions()
                                       .setNativeShapeFn(nativeShapeFn)
                                       .setFilterConstraint([](Operation *op) {
                                         return success(isa<ContractionOp>(op));
                                       }));
    } else {
      populateVectorUnrollPatterns(
          patterns, UnrollVectorOptions()
                        .setNativeShape(ArrayRef<int64_t>{2, 2, 2})
                        .setFilterConstraint([](Operation *op) {
                          return success(isa<ContractionOp>(op));
                        }));
    }
    populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  Option<bool> unrollBasedOnType{
      *this, "unroll-based-on-type",
      llvm::cl::desc("Set the unroll factor based on type of the operation"),
      llvm::cl::init(false)};
};

struct TestVectorDistributePatterns
    : public PassWrapper<TestVectorDistributePatterns, OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-distribute-patterns";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns to distribute vector ops in the vector "
           "dialect";
  }
  TestVectorDistributePatterns() = default;
  TestVectorDistributePatterns(const TestVectorDistributePatterns &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
    registry.insert<AffineDialect>();
  }
  ListOption<int32_t> multiplicity{
      *this, "distribution-multiplicity", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Set the multiplicity used for distributing vector")};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    FuncOp func = getOperation();
    func.walk([&](arith::AddFOp op) {
      OpBuilder builder(op);
      if (auto vecType = op.getType().dyn_cast<VectorType>()) {
        SmallVector<int64_t, 2> mul;
        SmallVector<AffineExpr, 2> perm;
        SmallVector<Value, 2> ids;
        unsigned count = 0;
        // Remove the multiplicity of 1 and calculate the affine map based on
        // the multiplicity.
        SmallVector<int32_t, 4> m(multiplicity.begin(), multiplicity.end());
        for (unsigned i = 0, e = vecType.getRank(); i < e; i++) {
          if (i < m.size() && m[i] != 1 && vecType.getDimSize(i) % m[i] == 0) {
            mul.push_back(m[i]);
            ids.push_back(func.getArgument(count++));
            perm.push_back(getAffineDimExpr(i, ctx));
          }
        }
        auto map = AffineMap::get(op.getType().cast<VectorType>().getRank(), 0,
                                  perm, ctx);
        Optional<mlir::vector::DistributeOps> ops = distributPointwiseVectorOp(
            builder, op.getOperation(), ids, mul, map);
        if (ops.hasValue()) {
          SmallPtrSet<Operation *, 1> extractOp({ops->extract, ops->insert});
          op.getResult().replaceAllUsesExcept(ops->insert.getResult(),
                                              extractOp);
        }
      }
    });
    populatePropagateVectorDistributionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorToLoopPatterns
    : public PassWrapper<TestVectorToLoopPatterns, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-vector-to-forloop"; }
  StringRef getDescription() const final {
    return "Test lowering patterns to break up a vector op into a for loop";
  }
  TestVectorToLoopPatterns() = default;
  TestVectorToLoopPatterns(const TestVectorToLoopPatterns &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
    registry.insert<AffineDialect>();
  }
  Option<int32_t> multiplicity{
      *this, "distribution-multiplicity",
      llvm::cl::desc("Set the multiplicity used for distributing vector"),
      llvm::cl::init(32)};
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    FuncOp func = getOperation();
    func.walk([&](arith::AddFOp op) {
      // Check that the operation type can be broken down into a loop.
      VectorType type = op.getType().dyn_cast<VectorType>();
      if (!type || type.getRank() != 1 ||
          type.getNumElements() % multiplicity != 0)
        return mlir::WalkResult::advance();
      auto filterAlloc = [](Operation *op) {
        return !isa<arith::ConstantOp, memref::AllocOp, CallOp>(op);
      };
      auto dependentOps = getSlice(op, filterAlloc);
      // Create a loop and move instructions from the Op slice into the loop.
      OpBuilder builder(op);
      auto zero = builder.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      auto one = builder.create<arith::ConstantIndexOp>(op.getLoc(), 1);
      auto numIter =
          builder.create<arith::ConstantIndexOp>(op.getLoc(), multiplicity);
      auto forOp = builder.create<scf::ForOp>(op.getLoc(), zero, numIter, one);
      for (Operation *it : dependentOps) {
        it->moveBefore(forOp.getBody()->getTerminator());
      }
      auto map = AffineMap::getMultiDimIdentityMap(1, ctx);
      // break up the original op and let the patterns propagate.
      Optional<mlir::vector::DistributeOps> ops = distributPointwiseVectorOp(
          builder, op.getOperation(), {forOp.getInductionVar()}, {multiplicity},
          map);
      if (ops.hasValue()) {
        SmallPtrSet<Operation *, 1> extractOp({ops->extract, ops->insert});
        op.getResult().replaceAllUsesExcept(ops->insert.getResult(), extractOp);
      }
      return mlir::WalkResult::interrupt();
    });
    populatePropagateVectorDistributionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransferUnrollingPatterns
    : public PassWrapper<TestVectorTransferUnrollingPatterns,
                         OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }
  StringRef getArgument() const final {
    return "test-vector-transfer-unrolling-patterns";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns to unroll transfer ops in the vector "
           "dialect";
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateVectorUnrollPatterns(
        patterns,
        UnrollVectorOptions()
            .setNativeShape(ArrayRef<int64_t>{2, 2})
            .setFilterConstraint([](Operation *op) {
              return success(
                  isa<vector::TransferReadOp, vector::TransferWriteOp>(op));
            }));
    populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransferFullPartialSplitPatterns
    : public PassWrapper<TestVectorTransferFullPartialSplitPatterns,
                         OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-transfer-full-partial-split";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns to split "
           "transfer ops via scf.if + linalg ops";
  }
  TestVectorTransferFullPartialSplitPatterns() = default;
  TestVectorTransferFullPartialSplitPatterns(
      const TestVectorTransferFullPartialSplitPatterns &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  Option<bool> useLinalgOps{
      *this, "use-linalg-copy",
      llvm::cl::desc("Split using a unmasked vector.transfer + linalg.fill + "
                     "linalg.copy operations."),
      llvm::cl::init(false)};
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    VectorTransformsOptions options;
    if (useLinalgOps)
      options.setVectorTransferSplit(VectorTransferSplit::LinalgCopy);
    else
      options.setVectorTransferSplit(VectorTransferSplit::VectorTransfer);
    patterns.add<VectorTransferFullPartialRewriter>(ctx, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransferOpt
    : public PassWrapper<TestVectorTransferOpt, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-vector-transferop-opt"; }
  StringRef getDescription() const final {
    return "Test optimization transformations for transfer ops";
  }
  void runOnOperation() override { transferOpflowOpt(getOperation()); }
};

struct TestVectorTransferLoweringPatterns
    : public PassWrapper<TestVectorTransferLoweringPatterns,
                         OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, memref::MemRefDialect>();
  }
  StringRef getArgument() const final {
    return "test-vector-transfer-lowering-patterns";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns to lower transfer ops to other vector ops";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorTransferLoweringPatterns(patterns);
    populateVectorTransferPermutationMapLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorMultiReductionLoweringPatterns
    : public PassWrapper<TestVectorMultiReductionLoweringPatterns,
                         OperationPass<FuncOp>> {
  TestVectorMultiReductionLoweringPatterns() = default;
  TestVectorMultiReductionLoweringPatterns(
      const TestVectorMultiReductionLoweringPatterns &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  StringRef getArgument() const final {
    return "test-vector-multi-reduction-lowering-patterns";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns to lower vector.multi_reduction to other "
           "vector ops";
  }
  Option<bool> useOuterReductions{
      *this, "use-outer-reductions",
      llvm::cl::desc("Move reductions to outer most dimensions"),
      llvm::cl::init(false)};
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorMultiReductionLoweringPatterns(
        patterns, useOuterReductions
                      ? vector::VectorMultiReductionLowering::InnerParallel
                      : vector::VectorMultiReductionLowering::InnerReduction);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransferCollapseInnerMostContiguousDims
    : public PassWrapper<TestVectorTransferCollapseInnerMostContiguousDims,
                         OperationPass<FuncOp>> {
  TestVectorTransferCollapseInnerMostContiguousDims() = default;
  TestVectorTransferCollapseInnerMostContiguousDims(
      const TestVectorTransferCollapseInnerMostContiguousDims &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, AffineDialect>();
  }

  StringRef getArgument() const final {
    return "test-vector-transfer-collapse-inner-most-dims";
  }

  StringRef getDescription() const final {
    return "Test lowering patterns that reducedes the rank of the vector "
           "transfer memory and vector operands.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorTransferCollapseInnerMostContiguousDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorReduceToContractPatternsPatterns
    : public PassWrapper<TestVectorReduceToContractPatternsPatterns,
                         OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-reduction-to-contract-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to convert multireduce op to contract and combine "
           "broadcast/transpose to contract";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorReductionToContractPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransferDropUnitDimsPatterns
    : public PassWrapper<TestVectorTransferDropUnitDimsPatterns,
                         OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-transfer-drop-unit-dims-patterns";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorTransferDropUnitDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestFlattenVectorTransferPatterns
    : public PassWrapper<TestFlattenVectorTransferPatterns,
                         OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-vector-transfer-flatten-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to rewrite contiguous row-major N-dimensional "
           "vector.transfer_{read,write} ops into 1D transfers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFlattenVectorTransferPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorScanLowering
    : public PassWrapper<TestVectorScanLowering, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-vector-scan-lowering"; }
  StringRef getDescription() const final {
    return "Test lowering patterns that lower the scan op in the vector "
           "dialect";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorScanLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestVectorLowerings() {
  PassRegistration<TestVectorToVectorLowering>();

  PassRegistration<TestVectorContractionLowering>();

  PassRegistration<TestVectorTransposeLowering>();

  PassRegistration<TestVectorUnrollingPatterns>();

  PassRegistration<TestVectorTransferUnrollingPatterns>();

  PassRegistration<TestVectorTransferFullPartialSplitPatterns>();

  PassRegistration<TestVectorDistributePatterns>();

  PassRegistration<TestVectorToLoopPatterns>();

  PassRegistration<TestVectorTransferOpt>();

  PassRegistration<TestVectorTransferLoweringPatterns>();

  PassRegistration<TestVectorMultiReductionLoweringPatterns>();

  PassRegistration<TestVectorTransferCollapseInnerMostContiguousDims>();

  PassRegistration<TestVectorReduceToContractPatternsPatterns>();

  PassRegistration<TestVectorTransferDropUnitDimsPatterns>();

  PassRegistration<TestFlattenVectorTransferPatterns>();

  PassRegistration<TestVectorScanLowering>();
}
} // namespace test
} // namespace mlir
