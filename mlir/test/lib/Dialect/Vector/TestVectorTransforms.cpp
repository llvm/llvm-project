//===- TestVectorTransforms.cpp - Test Vector transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;

namespace {

struct TestVectorToVectorLowering
    : public PassWrapper<TestVectorToVectorLowering,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorToVectorLowering)

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
    registry.insert<affine::AffineDialect>();
    registry.insert<vector::VectorDialect>();
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
  static std::optional<SmallVector<int64_t>> getShape(Operation *op) {
    if (isa<arith::AddFOp, arith::SelectOp, arith::CmpFOp>(op))
      return SmallVector<int64_t>(2, 2);
    if (isa<vector::ContractionOp>(op))
      return SmallVector<int64_t>(3, 2);
    // For transfer ops, just propagate the shape coming from
    // InsertStridedSlices/ExtractStridedSlices.
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      VectorType dstVec;
      for (Operation *users : readOp->getUsers()) {
        auto extract = dyn_cast<ExtractStridedSliceOp>(users);
        if (!extract)
          return std::nullopt;
        auto vecType = cast<VectorType>(extract.getResult().getType());
        if (dstVec && dstVec != vecType)
          return std::nullopt;
        dstVec = vecType;
      }
      return SmallVector<int64_t>(dstVec.getShape().begin(),
                                  dstVec.getShape().end());
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      auto insert = writeOp.getVector().getDefiningOp<InsertStridedSliceOp>();
      if (!insert)
        return std::nullopt;
      ArrayRef<int64_t> shape = insert.getSourceVectorType().getShape();
      return SmallVector<int64_t>(shape.begin(), shape.end());
    }
    return std::nullopt;
  }

  static LogicalResult filter(Operation *op) {
    return success(isa<arith::AddFOp, arith::SelectOp, arith::CmpFOp,
                       ContractionOp, TransferReadOp, TransferWriteOp>(op));
  }
};

struct TestVectorContractionPrepareForMMTLowering
    : public PassWrapper<TestVectorContractionPrepareForMMTLowering,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorContractionPrepareForMMTLowering)

  StringRef getArgument() const final {
    return "test-vector-contraction-prepare-for-mmt-lowering";
  }
  StringRef getDescription() const final {
    return "Test vector.contraction matmul canonicalization for MMT lowering.";
  }
  TestVectorContractionPrepareForMMTLowering() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    vector::populateVectorContractCanonicalizeMatmulToMMT(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorUnrollingPatterns
    : public PassWrapper<TestVectorUnrollingPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorUnrollingPatterns)

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
                        return success(isa<arith::AddFOp, vector::FMAOp,
                                           vector::MultiDimReductionOp>(op));
                      }));
    populateVectorUnrollPatterns(
        patterns, UnrollVectorOptions()
                      .setNativeShape(ArrayRef<int64_t>{2})
                      .setFilterConstraint([](Operation *op) {
                        return success(isa<vector::ReductionOp>(op));
                      }));
    populateVectorUnrollPatterns(
        patterns, UnrollVectorOptions()
                      .setNativeShape(ArrayRef<int64_t>{1, 3, 4, 2})
                      .setFilterConstraint([](Operation *op) {
                        return success(isa<vector::TransposeOp>(op));
                      }));

    if (unrollBasedOnType) {
      UnrollVectorOptions::NativeShapeFnType nativeShapeFn =
          [](Operation *op) -> std::optional<SmallVector<int64_t>> {
        vector::ContractionOp contractOp = cast<vector::ContractionOp>(op);
        SmallVector<int64_t> nativeShape(contractOp.getIteratorTypes().size(),
                                         4);
        Type lhsType = contractOp.getLhsType().getElementType();
        nativeShape[nativeShape.size() - 1] = lhsType.isF16() ? 4 : 2;
        return nativeShape;
      };

      UnrollVectorOptions opts;
      opts.setNativeShapeFn(nativeShapeFn)
          .setFilterConstraint(
              [](Operation *op) { return success(isa<ContractionOp>(op)); });

      if (!unrollOrder.empty()) {
        opts.setUnrollTraversalOrderFn(
            [this](Operation *op) -> std::optional<SmallVector<int64_t>> {
              vector::ContractionOp contractOp =
                  cast<vector::ContractionOp>(op);
              if (contractOp.getIteratorTypes().size() == unrollOrder.size())
                return SmallVector<int64_t>(unrollOrder.begin(),
                                            unrollOrder.end());
              return std::nullopt;
            });
      }
      populateVectorUnrollPatterns(patterns, opts);
    } else {
      auto nativeShapeFn =
          [](Operation *op) -> std::optional<SmallVector<int64_t>> {
        auto contractOp = dyn_cast<ContractionOp>(op);
        if (!contractOp)
          return std::nullopt;
        return SmallVector<int64_t>(contractOp.getIteratorTypes().size(), 2);
      };
      populateVectorUnrollPatterns(patterns,
                                   UnrollVectorOptions()
                                       .setNativeShapeFn(nativeShapeFn)
                                       .setFilterConstraint([](Operation *op) {
                                         return success(isa<ContractionOp>(op));
                                       }));
    }
    populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  ListOption<int64_t> unrollOrder{*this, "unroll-order",
                                  llvm::cl::desc("set the unroll order")};

  Option<bool> unrollBasedOnType{
      *this, "unroll-based-on-type",
      llvm::cl::desc("Set the unroll factor based on type of the operation"),
      llvm::cl::init(false)};
};

struct TestVectorTransferUnrollingPatterns
    : public PassWrapper<TestVectorTransferUnrollingPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorTransferUnrollingPatterns)

  TestVectorTransferUnrollingPatterns() = default;
  TestVectorTransferUnrollingPatterns(
      const TestVectorTransferUnrollingPatterns &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
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
    UnrollVectorOptions opts;
    opts.setNativeShape(ArrayRef<int64_t>{2, 2})
        .setFilterConstraint([](Operation *op) {
          return success(isa<vector::TransferReadOp, vector::TransferWriteOp,
                             vector::GatherOp>(op));
        });
    if (reverseUnrollOrder.getValue()) {
      opts.setUnrollTraversalOrderFn(
          [](Operation *op) -> std::optional<SmallVector<int64_t>> {
            int64_t numLoops = 0;
            if (auto readOp = dyn_cast<vector::TransferReadOp>(op))
              numLoops = readOp.getVectorType().getRank();
            else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op))
              numLoops = writeOp.getVectorType().getRank();
            else if (auto gatherOp = dyn_cast<vector::GatherOp>(op))
              numLoops = gatherOp.getVectorType().getRank();
            else
              return std::nullopt;
            auto order = llvm::reverse(llvm::seq<int64_t>(0, numLoops));
            return llvm::to_vector(order);
          });
    }
    populateVectorUnrollPatterns(patterns, opts);
    populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  Option<bool> reverseUnrollOrder{
      *this, "reverse-unroll-order",
      llvm::cl::desc(
          "reverse the order of unrolling of vector transfer operations"),
      llvm::cl::init(false)};
};

struct TestScalarVectorTransferLoweringPatterns
    : public PassWrapper<TestScalarVectorTransferLoweringPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestScalarVectorTransferLoweringPatterns)

  TestScalarVectorTransferLoweringPatterns() = default;
  TestScalarVectorTransferLoweringPatterns(
      const TestScalarVectorTransferLoweringPatterns &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-scalar-vector-transfer-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering of scalar vector transfers to memref loads/stores.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  Option<bool> allowMultipleUses{
      *this, "allow-multiple-uses",
      llvm::cl::desc("Fold transfer operations with multiple uses"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    vector::populateScalarVectorTransferLoweringPatterns(
        patterns, /*benefit=*/1, allowMultipleUses.getValue());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorTransferOpt
    : public PassWrapper<TestVectorTransferOpt, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorTransferOpt)

  StringRef getArgument() const final { return "test-vector-transferop-opt"; }
  StringRef getDescription() const final {
    return "Test optimization transformations for transfer ops";
  }
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    transferOpflowOpt(rewriter, getOperation());
  }
};

struct TestVectorTransferCollapseInnerMostContiguousDims
    : public PassWrapper<TestVectorTransferCollapseInnerMostContiguousDims,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorTransferCollapseInnerMostContiguousDims)

  TestVectorTransferCollapseInnerMostContiguousDims() = default;
  TestVectorTransferCollapseInnerMostContiguousDims(
      const TestVectorTransferCollapseInnerMostContiguousDims &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, affine::AffineDialect>();
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

struct TestSinkVectorBroadcast
    : public PassWrapper<TestSinkVectorBroadcast, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSinkVectorBroadcast)

  TestSinkVectorBroadcast() = default;
  TestSinkVectorBroadcast(const TestSinkVectorBroadcast &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, affine::AffineDialect>();
  }

  StringRef getArgument() const final { return "test-sink-vector-broadcast"; }

  StringRef getDescription() const final {
    return "Test lowering patterns that eliminate redundant brodacast "
           "operations.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateSinkVectorBroadcastPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorReduceToContractPatternsPatterns
    : public PassWrapper<TestVectorReduceToContractPatternsPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorReduceToContractPatternsPatterns)

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

struct TestVectorChainedReductionFoldingPatterns
    : public PassWrapper<TestVectorChainedReductionFoldingPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorChainedReductionFoldingPatterns)

  StringRef getArgument() const final {
    return "test-vector-chained-reduction-folding-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to fold chained vector reductions";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateChainedVectorReductionFoldingPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorBreakDownReductionPatterns
    : public PassWrapper<TestVectorBreakDownReductionPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorBreakDownReductionPatterns)

  StringRef getArgument() const final {
    return "test-vector-break-down-reduction-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to break down vector reductions into arith "
           "reductions";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBreakDownVectorReductionPatterns(patterns,
                                             /*maxNumElementsToExtract=*/2);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestFlattenVectorTransferPatterns
    : public PassWrapper<TestFlattenVectorTransferPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestFlattenVectorTransferPatterns)

  TestFlattenVectorTransferPatterns() = default;
  TestFlattenVectorTransferPatterns(
      const TestFlattenVectorTransferPatterns &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-vector-transfer-flatten-patterns";
  }

  StringRef getDescription() const final {
    return "Test patterns to rewrite contiguous row-major N-dimensional "
           "vector.transfer_{read,write} ops into 1D transfers";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<vector::VectorDialect>();
  }

  Option<unsigned> targetVectorBitwidth{
      *this, "target-vector-bitwidth",
      llvm::cl::desc(
          "Minimum vector bitwidth to enable the flattening transformation"),
      llvm::cl::init(std::numeric_limits<unsigned>::max())};

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFlattenVectorTransferPatterns(patterns, targetVectorBitwidth);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorScanLowering
    : public PassWrapper<TestVectorScanLowering, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorScanLowering)

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

/// Allocate shared memory for a single warp to test lowering of
/// WarpExecuteOnLane0Op.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  static constexpr int64_t kSharedMemorySpace = 3;
  // Compute type of shared memory buffer.
  MemRefType memrefType;
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(), {},
                        kSharedMemorySpace);
  } else {
    memrefType = MemRefType::get({1}, type, {}, kSharedMemorySpace);
  }

  // Get symbol table holding all shared memory globals.
  ModuleOp moduleOp = warpOp->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(moduleOp);

  // Create a pretty name.
  SmallString<64> buf;
  llvm::raw_svector_ostream os(buf);
  interleave(memrefType.getShape(), os, "x");
  os << "x" << memrefType.getElementType();
  std::string symbolName = (Twine("__shared_") + os.str()).str();

  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPoint(moduleOp);
  auto global = builder.create<memref::GlobalOp>(
      loc,
      /*sym_name=*/symbolName,
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/memrefType,
      /*initial_value=*/Attribute(),
      /*constant=*/false,
      /*alignment=*/IntegerAttr());
  symbolTable.insert(global);
  // The symbol table inserts at the end of the module, but globals are a bit
  // nicer if they are at the beginning.
  global->moveBefore(&moduleOp.front());

  builder.restoreInsertionPoint(ip);
  return builder.create<memref::GetGlobalOp>(loc, memrefType, symbolName);
}

static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           CombiningKind kind, uint32_t size) {
  // First reduce on a single thread to get per lane reduction value.
  Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, laneVal, i,
                                                 /*width=*/size,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  return laneVal;
}

struct TestVectorDistribution
    : public PassWrapper<TestVectorDistribution, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorDistribution)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, gpu::GPUDialect,
                    affine::AffineDialect>();
  }

  StringRef getArgument() const final { return "test-vector-warp-distribute"; }
  StringRef getDescription() const final {
    return "Test vector warp distribute transformation and lowering patterns";
  }
  TestVectorDistribution() = default;
  TestVectorDistribution(const TestVectorDistribution &pass)
      : PassWrapper(pass) {}

  Option<bool> warpOpToSCF{
      *this, "rewrite-warp-ops-to-scf-if",
      llvm::cl::desc("Lower vector.warp_execute_on_lane0 to scf.if op"),
      llvm::cl::init(false)};

  Option<bool> distributeTransferWriteOps{
      *this, "distribute-transfer-write",
      llvm::cl::desc("Test distribution of transfer write"),
      llvm::cl::init(false)};

  Option<unsigned> maxTransferWriteElements{
      *this, "max-transfer-write-elements",
      llvm::cl::desc("Maximum number of transfer write elements to distribute"),
      llvm::cl::init(1)};

  Option<bool> hoistUniform{*this, "hoist-uniform",
                            llvm::cl::desc("Test hoist uniform"),
                            llvm::cl::init(false)};

  Option<bool> propagateDistribution{
      *this, "propagate-distribution",
      llvm::cl::desc("Test distribution propgation"), llvm::cl::init(false)};

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    getOperation().walk([&](Operation *op) {
      if (auto warpOp = dyn_cast<WarpExecuteOnLane0Op>(op)) {
        if (hoistUniform) {
          moveScalarUniformCode(warpOp);
        }
        WalkResult::interrupt();
      }
    });
    MLIRContext *ctx = &getContext();
    auto distributionFn = [](Value val) {
      // Create an identity dim map of the same rank as the vector.
      VectorType vecType = dyn_cast<VectorType>(val.getType());
      int64_t vecRank = vecType ? vecType.getRank() : 0;
      OpBuilder builder(val.getContext());
      if (vecRank == 0)
        return AffineMap::get(val.getContext());
      return AffineMap::getMultiDimIdentityMap(vecRank, val.getContext());
    };
    auto shuffleFn = [](Location loc, OpBuilder &builder, Value val,
                        Value srcIdx, int64_t warpSz) {
      assert((val.getType().isF32() || val.getType().isInteger(32)) &&
             "unsupported shuffle type");
      Type i32Type = builder.getIntegerType(32);
      Value srcIdxI32 =
          builder.create<arith::IndexCastOp>(loc, i32Type, srcIdx);
      Value warpSzI32 = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(i32Type, warpSz));
      Value result = builder
                         .create<gpu::ShuffleOp>(loc, val, srcIdxI32, warpSzI32,
                                                 gpu::ShuffleMode::IDX)
                         .getResult(0);
      return result;
    };
    if (distributeTransferWriteOps && propagateDistribution) {
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, shuffleFn, /*benefit=*/1,
          /*readBenefit=*/0);
      vector::populateDistributeReduction(patterns, warpReduction, 1);
      populateDistributeTransferWriteOpPatterns(patterns, distributionFn, 2);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    } else if (distributeTransferWriteOps) {
      RewritePatternSet patterns(ctx);
      populateDistributeTransferWriteOpPatterns(patterns, distributionFn,
                                                maxTransferWriteElements);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    } else if (propagateDistribution) {
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, shuffleFn);
      vector::populateDistributeReduction(patterns, warpReduction);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
    WarpExecuteOnLane0LoweringOptions options;
    options.warpAllocationFn = allocateGlobalSharedMemory;
    options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                      WarpExecuteOnLane0Op warpOp) {
      builder.create<gpu::BarrierOp>(loc);
    };
    // Test on one pattern in isolation.
    if (warpOpToSCF) {
      populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
      return;
    }
  }
};

struct TestVectorExtractStridedSliceLowering
    : public PassWrapper<TestVectorExtractStridedSliceLowering,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestVectorExtractStridedSliceLowering)

  StringRef getArgument() const final {
    return "test-vector-extract-strided-slice-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering patterns that converts vector.extract_strided_slice "
           "into a chain of vector.extract and vector.insert ops";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorExtractStridedSliceToExtractInsertChainPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorBreakDownBitCast
    : public PassWrapper<TestVectorBreakDownBitCast,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorBreakDownBitCast)

  StringRef getArgument() const final {
    return "test-vector-break-down-bitcast";
  }
  StringRef getDescription() const final {
    return "Test pattern that breaks down vector.bitcast ops ";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBreakDownVectorBitCastOpPatterns(patterns, [](BitCastOp op) {
      return op.getSourceVectorType().getShape().back() > 4;
    });
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestCreateVectorBroadcast
    : public PassWrapper<TestCreateVectorBroadcast,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCreateVectorBroadcast)

  StringRef getArgument() const final { return "test-create-vector-broadcast"; }
  StringRef getDescription() const final {
    return "Test optimization transformations for transfer ops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (op->getName().getStringRef() != "test_create_broadcast")
        return;
      auto targetShape =
          cast<VectorType>(op->getResult(0).getType()).getShape();
      auto arrayAttr =
          cast<DenseI64ArrayAttr>(op->getDiscardableAttr("broadcast_dims"))
              .asArrayRef();
      llvm::SetVector<int64_t> broadcastedDims;
      broadcastedDims.insert(arrayAttr.begin(), arrayAttr.end());
      OpBuilder b(op);
      Value bcast = vector::BroadcastOp::createOrFoldBroadcastOp(
          b, op->getOperand(0), targetShape, broadcastedDims);
      op->getResult(0).replaceAllUsesWith(bcast);
      op->erase();
    });
  }
};

struct TestVectorGatherLowering
    : public PassWrapper<TestVectorGatherLowering,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorGatherLowering)

  StringRef getArgument() const final { return "test-vector-gather-lowering"; }
  StringRef getDescription() const final {
    return "Test patterns that lower the gather op in the vector conditional "
           "loads";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorGatherLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestFoldArithExtensionIntoVectorContractPatterns
    : public PassWrapper<TestFoldArithExtensionIntoVectorContractPatterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestFoldArithExtensionIntoVectorContractPatterns)

  StringRef getArgument() const final {
    return "test-fold-arith-extf-into-vector-contract-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns that fold arithmetic extension ops into vector "
           "contract ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, nvgpu::NVGPUDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFoldArithExtensionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorEmulateMaskedLoadStore final
    : public PassWrapper<TestVectorEmulateMaskedLoadStore,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorEmulateMaskedLoadStore)

  StringRef getArgument() const override {
    return "test-vector-emulate-masked-load-store";
  }
  StringRef getDescription() const override {
    return "Test patterns that emulate the maskedload/maskedstore op by "
           " memref.load/store and scf.if";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorMaskedLoadStoreEmulationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct TestVectorLinearize final
    : public PassWrapper<TestVectorLinearize, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorLinearize)

  TestVectorLinearize() = default;
  TestVectorLinearize(const TestVectorLinearize &pass) : PassWrapper(pass) {}

  StringRef getArgument() const override { return "test-vector-linearize"; }
  StringRef getDescription() const override {
    return "Linearizes ND vectors for N >= 2 into 1D vectors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  Option<unsigned> targetVectorBitwidth{
      *this, "target-vector-bitwidth",
      llvm::cl::desc(
          "Minimum vector bitwidth to enable the flattening transformation"),
      llvm::cl::init(std::numeric_limits<unsigned>::max())};
  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    vector::populateVectorLinearizeTypeConversionsAndLegality(
        typeConverter, patterns, target, targetVectorBitwidth);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestVectorLowerings() {
  PassRegistration<TestVectorToVectorLowering>();

  PassRegistration<TestVectorContractionPrepareForMMTLowering>();

  PassRegistration<TestVectorUnrollingPatterns>();

  PassRegistration<TestVectorTransferUnrollingPatterns>();

  PassRegistration<TestScalarVectorTransferLoweringPatterns>();

  PassRegistration<TestVectorTransferOpt>();

  PassRegistration<TestVectorTransferCollapseInnerMostContiguousDims>();

  PassRegistration<TestSinkVectorBroadcast>();

  PassRegistration<TestVectorReduceToContractPatternsPatterns>();

  PassRegistration<TestVectorChainedReductionFoldingPatterns>();

  PassRegistration<TestVectorBreakDownReductionPatterns>();

  PassRegistration<TestFlattenVectorTransferPatterns>();

  PassRegistration<TestVectorScanLowering>();

  PassRegistration<TestVectorDistribution>();

  PassRegistration<TestVectorExtractStridedSliceLowering>();

  PassRegistration<TestVectorBreakDownBitCast>();

  PassRegistration<TestCreateVectorBroadcast>();

  PassRegistration<TestVectorGatherLowering>();

  PassRegistration<TestFoldArithExtensionIntoVectorContractPatterns>();

  PassRegistration<TestVectorEmulateMaskedLoadStore>();

  PassRegistration<TestVectorLinearize>();
}
} // namespace test
} // namespace mlir
