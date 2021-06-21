//===- VectorToGPU.cpp - Convert vector to GPU dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector operations to GPU dialect ops.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"

#include "../PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

// Return true if the contract op can be convert to MMA matmul.
static bool contractSupportsMMAMatrixType(vector::ContractionOp contract) {
  if (llvm::size(contract.masks()) != 0)
    return false;

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(contract.getContext(), m, n, k);
  auto iteratorTypes = contract.iterator_types().getValue();
  if (!(isParallelIterator(iteratorTypes[0]) &&
        isParallelIterator(iteratorTypes[1]) &&
        isReductionIterator(iteratorTypes[2])))
    return false;

  // The contract needs to represent a matmul to be able to convert to
  // MMAMatrix matmul.
  if (contract.getIndexingMaps() != infer({{m, k}, {k, n}, {m, n}}))
    return false;

  // Check that the size matches what is natively supported.
  VectorType lhsType = contract.lhs().getType().cast<VectorType>();
  VectorType rhsType = contract.rhs().getType().cast<VectorType>();
  VectorType accType = contract.acc().getType().cast<VectorType>();

  std::tuple<int, int, int> dim(lhsType.getDimSize(0), rhsType.getDimSize(1),
                                lhsType.getDimSize(1));
  if (lhsType.getElementType().isInteger(8) &&
      rhsType.getElementType().isInteger(8) &&
      accType.getElementType().isInteger(32) &&
      (dim == std::make_tuple(8, 8, 32) || dim == std::make_tuple(16, 16, 32) ||
       dim == std::make_tuple(16, 8, 32)))
    return true;

  if (lhsType.getElementType().isF16() && rhsType.getElementType().isF16() &&
      (accType.getElementType().isF16() || accType.getElementType().isF32()) &&
      (dim == std::make_tuple(8, 8, 16) || dim == std::make_tuple(16, 16, 16) ||
       dim == std::make_tuple(16, 8, 16)))
    return true;
  return false;
}

// Return the stide for the dimension 0 of |type| if it is a memref and has a
// constant stride.
static llvm::Optional<int64_t>
getMemrefConstantHorizontalStride(ShapedType type) {
  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType)
    return false;
  int64_t offset = 0;
  SmallVector<int64_t, 2> strides;
  if (failed(getStridesAndOffset(memrefType, strides, offset)))
    return llvm::None;
  if (strides[0] == ShapedType::kDynamicStrideOrOffset)
    return llvm::None;
  return strides[0];
}

// Return true if the transfer op can be converted to a MMA matrix load.
static bool transferReadSupportsMMAMatrixType(vector::TransferReadOp readOp) {
  if (readOp.mask() || readOp.hasOutOfBoundsDim() ||
      readOp.getVectorType().getRank() != 2)
    return false;
  if (!getMemrefConstantHorizontalStride(readOp.getShapedType()))
    return false;
  // TODO: Support transpose once it is added to GPU dialect ops.
  if (!readOp.permutation_map().isMinorIdentity())
    return false;
  return true;
}

// Return true if the transfer op can be converted to a MMA matrix store.
static bool
transferWriteSupportsMMAMatrixType(vector::TransferWriteOp writeOp) {
  if (writeOp.mask() || writeOp.hasOutOfBoundsDim() ||
      writeOp.getVectorType().getRank() != 2)
    return false;
  if (!getMemrefConstantHorizontalStride(writeOp.getShapedType()))
    return false;
  // TODO: Support transpose once it is added to GPU dialect ops.
  if (!writeOp.permutation_map().isMinorIdentity())
    return false;
  return true;
}

static bool supportsMMaMatrixType(Operation *op) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op))
    return transferReadSupportsMMAMatrixType(transferRead);
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteSupportsMMAMatrixType(transferWrite);
  if (auto contract = dyn_cast<vector::ContractionOp>(op))
    return contractSupportsMMAMatrixType(contract);
  return false;
}

// Analyze slice of operations based on convert op to figure out if the whole
// slice can be converted to MMA operations.
static SetVector<Operation *> getOpToConvert(mlir::Operation *op) {
  auto hasVectorDest = [](Operation *op) {
    return op->getNumResults() == 0 ||
           llvm::any_of(op->getResultTypes(),
                        [](Type t) { return t.isa<VectorType>(); });
  };
  SetVector<Operation *> opToConvert;
  op->walk([&](vector::ContractionOp contract) {
    if (opToConvert.contains(contract.getOperation()))
      return;
    SetVector<Operation *> dependentOps =
        getSlice(contract, hasVectorDest, hasVectorDest);
    // If any instruction cannot use MMA matrix type drop the whole
    // chaine. MMA matrix are stored in an opaque type so they cannot be used
    // by all operations.
    if (llvm::any_of(dependentOps,
                     [](Operation *op) { return !supportsMMaMatrixType(op); }))
      return;
    opToConvert.insert(dependentOps.begin(), dependentOps.end());
  });
  return opToConvert;
}

namespace {
// Transform contract into (m, k)x(k, n)x(m, n) form so that it can be converted
// to MMA matmul.
struct PrepareContractToGPUMMA
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.lhs(), rhs = op.rhs(), res = op.acc();

    // Set up the parallel/reduction structure in right form.
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto iteratorTypes = op.iterator_types().getValue();
    SmallVector<AffineMap, 4> maps = op.getIndexingMaps();
    if (!(isParallelIterator(iteratorTypes[0]) &&
          isParallelIterator(iteratorTypes[1]) &&
          isReductionIterator(iteratorTypes[2])))
      return failure();
    //
    // Two outer parallel, one inner reduction (matmat flavor).
    //
    if (maps == infer({{m, k}, {k, n}, {m, n}})) {
      // This is the classical row-major matmul, nothing to do.
      return failure();
    }
    if (maps == infer({{m, k}, {n, k}, {m, n}})) {
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      std::swap(lhs, rhs);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      std::swap(lhs, rhs);
    } else {
      return failure();
    }
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, lhs, rhs, res,
        rewriter.getAffineMapArrayAttr(infer({{m, k}, {k, n}, {m, n}})),
        op.iterator_types());
    return success();
  }
};

// Merge transpose op into the transfer read op. Transpose are not supported on
// MMA types but MMA load can transpose the matrix when loading.
struct CombineTransferReadOpTranspose final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto transferReadOp = op.vector().getDefiningOp<vector::TransferReadOp>();
    if (!transferReadOp)
      return failure();
    if (transferReadOp.mask() || transferReadOp.hasOutOfBoundsDim())
      return failure();
    SmallVector<int64_t, 2> perm;
    op.getTransp(perm);
    SmallVector<unsigned, 2> permU;
    for (int64_t o : perm)
      permU.push_back(unsigned(o));
    AffineMap permutationMap =
        AffineMap::getPermutationMap(permU, op.getContext());
    AffineMap newMap = permutationMap.compose(transferReadOp.permutation_map());
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), transferReadOp.source(), transferReadOp.indices(),
        newMap, transferReadOp.padding(), transferReadOp.mask(),
        transferReadOp.in_boundsAttr());
    return success();
  }
};

} // namespace

// MMA types have different layout based on how they are used in matmul ops.
// Figure the right layout to use by looking at Transfer op uses.
// TODO: Change the GPU dialect to abstract the layout at the this level and
// only care about it during lowering to NVVM.
static const char *inferFragType(vector::TransferReadOp op) {
  for (Operation *users : op->getUsers()) {
    auto contract = dyn_cast<vector::ContractionOp>(users);
    if (!contract)
      continue;
    if (contract.lhs() == op.getResult())
      return "AOp";
    if (contract.rhs() == op.getResult())
      return "BOp";
  }
  return "COp";
}

static void convertTransferReadOp(vector::TransferReadOp op,
                                  llvm::DenseMap<Value, Value> &valueMapping) {
  assert(transferReadSupportsMMAMatrixType(op));
  Optional<int64_t> stride =
      getMemrefConstantHorizontalStride(op.getShapedType());
  assert(stride);
  const char *fragType = inferFragType(op);
  gpu::MMAMatrixType type =
      gpu::MMAMatrixType::get(op.getVectorType().getShape(),
                              op.getVectorType().getElementType(), fragType);
  OpBuilder b(op);
  Value load = b.create<gpu::SubgroupMmaLoadMatrixOp>(
      op.getLoc(), type, op.source(), op.indices(), b.getIndexAttr(*stride));
  valueMapping[op.getResult()] = load;
}

static void convertTransferWriteOp(vector::TransferWriteOp op,
                                   llvm::DenseMap<Value, Value> &valueMapping) {
  assert(transferWriteSupportsMMAMatrixType(op));
  Optional<int64_t> stride =
      getMemrefConstantHorizontalStride(op.getShapedType());
  assert(stride);
  OpBuilder b(op);
  Value matrix = valueMapping.find(op.vector())->second;
  b.create<gpu::SubgroupMmaStoreMatrixOp>(
      op.getLoc(), matrix, op.source(), op.indices(), b.getIndexAttr(*stride));
  op.erase();
}

static void convertContractOp(vector::ContractionOp op,
                              llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  Value opA = valueMapping.find(op.lhs())->second;
  Value opB = valueMapping.find(op.rhs())->second;
  Value opC = valueMapping.find(op.acc())->second;
  Value matmul = b.create<gpu::SubgroupMmaComputeOp>(op.getLoc(), opC.getType(),
                                                     opA, opB, opC);
  valueMapping[op.getResult()] = matmul;
}

namespace mlir {

void populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns) {
  patterns.add<PrepareContractToGPUMMA, CombineTransferReadOpTranspose>(
      patterns.getContext());
}

void convertVectorToMMAOps(FuncOp funcOp) {
  SetVector<Operation *> ops = getOpToConvert(funcOp);
  llvm::DenseMap<Value, Value> valueMapping;
  for (Operation *op : ops) {
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
      convertTransferReadOp(transferRead, valueMapping);
    } else if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
      convertTransferWriteOp(transferWrite, valueMapping);
    } else if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      convertContractOp(contractOp, valueMapping);
    }
  }
}

} // namespace mlir
namespace {

struct ConvertVectorToGPUPass
    : public ConvertVectorToGPUBase<ConvertVectorToGPUPass> {
  void runOnFunction() override {
    RewritePatternSet patterns(getFunction().getContext());
    populatePrepareVectorToMMAPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

    convertVectorToMMAOps(getFunction());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertVectorToGPUPass() {
  return std::make_unique<ConvertVectorToGPUPass>();
}
