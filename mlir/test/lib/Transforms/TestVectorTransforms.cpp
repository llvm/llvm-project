//===- TestVectorToVectorConversion.cpp - Test VectorTransfers lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
namespace {

struct TestVectorToVectorConversion
    : public PassWrapper<TestVectorToVectorConversion, FunctionPass> {
  TestVectorToVectorConversion() = default;
  TestVectorToVectorConversion(const TestVectorToVectorConversion &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  Option<bool> unroll{*this, "unroll", llvm::cl::desc("Include unrolling"),
                      llvm::cl::init(false)};

  void runOnFunction() override {
    auto *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    if (unroll) {
      patterns.insert<UnrollVectorPattern>(
          ctx,
          UnrollVectorOptions().setNativeShapeFn(getShape).setFilterConstraint(
              filter));
    }
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorToVectorTransformationPatterns(patterns);
    populateBubbleVectorBitCastOpPatterns(patterns);
    populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populateSplitVectorTransferPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

private:
  // Return the target shape based on op type.
  static Optional<SmallVector<int64_t, 4>> getShape(Operation *op) {
    if (isa<AddFOp, SelectOp, CmpFOp>(op))
      return SmallVector<int64_t, 4>(2, 2);
    if (isa<vector::ContractionOp>(op))
      return SmallVector<int64_t, 4>(3, 2);
    return llvm::None;
  }

  static LogicalResult filter(Operation *op) {
    return success(isa<AddFOp, SelectOp, CmpFOp, ContractionOp>(op));
  }
};

struct TestVectorSlicesConversion
    : public PassWrapper<TestVectorSlicesConversion, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    populateVectorSlicesLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

struct TestVectorContractionConversion
    : public PassWrapper<TestVectorContractionConversion, FunctionPass> {
  TestVectorContractionConversion() = default;
  TestVectorContractionConversion(const TestVectorContractionConversion &pass) {
  }

  Option<bool> lowerToFlatMatrix{
      *this, "vector-lower-matrix-intrinsics",
      llvm::cl::desc("Lower vector.contract to llvm.intr.matrix.multiply"),
      llvm::cl::init(false)};
  Option<bool> lowerToFlatTranspose{
      *this, "vector-flat-transpose",
      llvm::cl::desc("Lower 2-D vector.transpose to vector.flat_transpose"),
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

  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());

    // Test on one pattern in isolation.
    if (lowerToOuterProduct) {
      VectorContractLowering lowering = VectorContractLowering::OuterProduct;
      VectorTransformsOptions options{lowering};
      patterns.insert<ContractionOpToOuterProductOpLowering>(options,
                                                             &getContext());
      (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
      return;
    }

    // Test on one pattern in isolation.
    if (lowerToFilterOuterProduct) {
      VectorContractLowering lowering = VectorContractLowering::OuterProduct;
      VectorTransformsOptions options{lowering};
      patterns.insert<ContractionOpToOuterProductOpLowering>(
          options, &getContext(), [](vector::ContractionOp op) {
            // Only lowers vector.contract where the lhs as a type vector<MxNx?>
            // where M is not 4.
            if (op.getRhsType().getShape()[0] == 4)
              return failure();
            return success();
          });
      (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
      return;
    }

    // Test on all contract lowering patterns.
    VectorContractLowering contractLowering = VectorContractLowering::Dot;
    if (lowerToFlatMatrix)
      contractLowering = VectorContractLowering::Matmul;
    VectorTransposeLowering transposeLowering =
        VectorTransposeLowering::EltWise;
    if (lowerToFlatTranspose)
      transposeLowering = VectorTransposeLowering::Flat;
    VectorTransformsOptions options{contractLowering, transposeLowering};
    populateVectorContractLoweringPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

struct TestVectorUnrollingPatterns
    : public PassWrapper<TestVectorUnrollingPatterns, FunctionPass> {
  TestVectorUnrollingPatterns() = default;
  TestVectorUnrollingPatterns(const TestVectorUnrollingPatterns &pass) {}
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    patterns.insert<UnrollVectorPattern>(
        ctx, UnrollVectorOptions()
                 .setNativeShape(ArrayRef<int64_t>{2, 2})
                 .setFilterConstraint([](Operation *op) {
                   return success(isa<AddFOp, vector::FMAOp>(op));
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
      patterns.insert<UnrollVectorPattern>(
          ctx, UnrollVectorOptions()
                   .setNativeShapeFn(nativeShapeFn)
                   .setFilterConstraint([](Operation *op) {
                     return success(isa<ContractionOp>(op));
                   }));
    } else {
      patterns.insert<UnrollVectorPattern>(
          ctx, UnrollVectorOptions()
                   .setNativeShape(ArrayRef<int64_t>{2, 2, 2})
                   .setFilterConstraint([](Operation *op) {
                     return success(isa<ContractionOp>(op));
                   }));
    }
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorToVectorTransformationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

  Option<bool> unrollBasedOnType{
      *this, "unroll-based-on-type",
      llvm::cl::desc("Set the unroll factor based on type of the operation"),
      llvm::cl::init(false)};
};

struct TestVectorDistributePatterns
    : public PassWrapper<TestVectorDistributePatterns, FunctionPass> {
  TestVectorDistributePatterns() = default;
  TestVectorDistributePatterns(const TestVectorDistributePatterns &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
    registry.insert<AffineDialect>();
  }
  ListOption<int32_t> multiplicity{
      *this, "distribution-multiplicity", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Set the multiplicity used for distributing vector")};

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    FuncOp func = getFunction();
    func.walk([&](AddFOp op) {
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
    patterns.insert<PointwiseExtractPattern>(ctx);
    populateVectorToVectorTransformationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

struct TestVectorToLoopPatterns
    : public PassWrapper<TestVectorToLoopPatterns, FunctionPass> {
  TestVectorToLoopPatterns() = default;
  TestVectorToLoopPatterns(const TestVectorToLoopPatterns &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
    registry.insert<AffineDialect>();
  }
  Option<int32_t> multiplicity{
      *this, "distribution-multiplicity",
      llvm::cl::desc("Set the multiplicity used for distributing vector"),
      llvm::cl::init(32)};
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    FuncOp func = getFunction();
    func.walk([&](AddFOp op) {
      // Check that the operation type can be broken down into a loop.
      VectorType type = op.getType().dyn_cast<VectorType>();
      if (!type || type.getRank() != 1 ||
          type.getNumElements() % multiplicity != 0)
        return mlir::WalkResult::advance();
      auto filterAlloc = [](Operation *op) {
        if (isa<ConstantOp, memref::AllocOp, CallOp>(op))
          return false;
        return true;
      };
      auto dependentOps = getSlice(op, filterAlloc);
      // Create a loop and move instructions from the Op slice into the loop.
      OpBuilder builder(op);
      auto zero = builder.create<ConstantOp>(
          op.getLoc(), builder.getIndexType(),
          builder.getIntegerAttr(builder.getIndexType(), 0));
      auto one = builder.create<ConstantOp>(
          op.getLoc(), builder.getIndexType(),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      auto numIter = builder.create<ConstantOp>(
          op.getLoc(), builder.getIndexType(),
          builder.getIntegerAttr(builder.getIndexType(), multiplicity));
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
    patterns.insert<PointwiseExtractPattern>(ctx);
    populateVectorToVectorTransformationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

struct TestVectorTransferUnrollingPatterns
    : public PassWrapper<TestVectorTransferUnrollingPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    patterns.insert<UnrollVectorPattern>(
        ctx,
        UnrollVectorOptions()
            .setNativeShape(ArrayRef<int64_t>{2, 2})
            .setFilterConstraint([](Operation *op) {
              return success(
                  isa<vector::TransferReadOp, vector::TransferWriteOp>(op));
            }));
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorToVectorTransformationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

struct TestVectorTransferFullPartialSplitPatterns
    : public PassWrapper<TestVectorTransferFullPartialSplitPatterns,
                         FunctionPass> {
  TestVectorTransferFullPartialSplitPatterns() = default;
  TestVectorTransferFullPartialSplitPatterns(
      const TestVectorTransferFullPartialSplitPatterns &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  Option<bool> useLinalgOps{
      *this, "use-linalg-copy",
      llvm::cl::desc("Split using a unmasked vector.transfer + linalg.fill + "
                     "linalg.copy operations."),
      llvm::cl::init(false)};
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    OwningRewritePatternList patterns(ctx);
    VectorTransformsOptions options;
    if (useLinalgOps)
      options.setVectorTransferSplit(VectorTransferSplit::LinalgCopy);
    else
      options.setVectorTransferSplit(VectorTransferSplit::VectorTransfer);
    patterns.insert<VectorTransferFullPartialRewriter>(ctx, options);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

struct TestVectorTransferOpt
    : public PassWrapper<TestVectorTransferOpt, FunctionPass> {
  void runOnFunction() override { transferOpflowOpt(getFunction()); }
};

struct TestVectorTransferLoweringPatterns
    : public PassWrapper<TestVectorTransferLoweringPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    populateVectorTransferLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestVectorConversions() {
  PassRegistration<TestVectorToVectorConversion> vectorToVectorPass(
      "test-vector-to-vector-conversion",
      "Test conversion patterns between ops in the vector dialect");

  PassRegistration<TestVectorSlicesConversion> slicesPass(
      "test-vector-slices-conversion",
      "Test conversion patterns that lower slices ops in the vector dialect");

  PassRegistration<TestVectorContractionConversion> contractionPass(
      "test-vector-contraction-conversion",
      "Test conversion patterns that lower contract ops in the vector dialect");

  PassRegistration<TestVectorUnrollingPatterns> contractionUnrollingPass(
      "test-vector-unrolling-patterns",
      "Test conversion patterns to unroll contract ops in the vector dialect");

  PassRegistration<TestVectorTransferUnrollingPatterns> transferOpUnrollingPass(
      "test-vector-transfer-unrolling-patterns",
      "Test conversion patterns to unroll transfer ops in the vector dialect");

  PassRegistration<TestVectorTransferFullPartialSplitPatterns>
      vectorTransformFullPartialPass("test-vector-transfer-full-partial-split",
                                     "Test conversion patterns to split "
                                     "transfer ops via scf.if + linalg ops");

  PassRegistration<TestVectorDistributePatterns> distributePass(
      "test-vector-distribute-patterns",
      "Test conversion patterns to distribute vector ops in the vector "
      "dialect");

  PassRegistration<TestVectorToLoopPatterns> vectorToForLoop(
      "test-vector-to-forloop",
      "Test conversion patterns to break up a vector op into a for loop");

  PassRegistration<TestVectorTransferOpt> transferOpOpt(
      "test-vector-transferop-opt",
      "Test optimization transformations for transfer ops");

  PassRegistration<TestVectorTransferLoweringPatterns> transferOpLoweringPass(
      "test-vector-transfer-lowering-patterns",
      "Test conversion patterns to lower transfer ops to other vector ops");
}
} // namespace test
} // namespace mlir
