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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
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

/// Return true if the constant is a splat to a 2D vector so that it can be
/// converted to a MMA constant matrix op.
static bool constantSupportsMMAMatrixType(arith::ConstantOp constantOp) {
  auto vecType = constantOp.getType().dyn_cast<VectorType>();
  if (!vecType || vecType.getRank() != 2)
    return false;
  return constantOp.getValue().isa<SplatElementsAttr>();
}

/// Return true if this is a broadcast from scalar to a 2D vector.
static bool broadcastSupportsMMAMatrixType(vector::BroadcastOp broadcastOp) {
  return broadcastOp.getVectorType().getRank() == 2 &&
         broadcastOp.source().getType().isa<FloatType>();
}

static bool supportsMMaMatrixType(Operation *op) {
  if (isa<scf::ForOp, scf::YieldOp>(op))
    return true;
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op))
    return transferReadSupportsMMAMatrixType(transferRead);
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteSupportsMMAMatrixType(transferWrite);
  if (auto contract = dyn_cast<vector::ContractionOp>(op))
    return contractSupportsMMAMatrixType(contract);
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return constantSupportsMMAMatrixType(constant);
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op))
    return broadcastSupportsMMAMatrixType(broadcast);
  return false;
}

// Analyze slice of operations based on convert op to figure out if the whole
// slice can be converted to MMA operations.
static SetVector<Operation *> getOpToConvert(mlir::Operation *op) {
  auto hasVectorDest = [](Operation *op) {
    return llvm::any_of(op->getResultTypes(),
                        [](Type t) { return t.isa<VectorType>(); });
  };
  auto hasVectorSrc = [](Operation *op) {
    return llvm::any_of(op->getOperandTypes(),
                        [](Type t) { return t.isa<VectorType>(); });
  };
  SetVector<Operation *> opToConvert;
  op->walk([&](vector::ContractionOp contract) {
    if (opToConvert.contains(contract.getOperation()))
      return;
    SetVector<Operation *> dependentOps =
        getSlice(contract, hasVectorDest, hasVectorSrc);
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
// Figure the right layout to use by looking at op uses.
// TODO: Change the GPU dialect to abstract the layout at the this level and
// only care about it during lowering to NVVM.
template <typename OpTy>
static const char *inferFragType(OpTy op) {
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

/// Convert a 2D splat ConstantOp to a SubgroupMmaConstantMatrix op.
static void convertConstantOp(arith::ConstantOp op,
                              llvm::DenseMap<Value, Value> &valueMapping) {
  assert(constantSupportsMMAMatrixType(op));
  OpBuilder b(op);
  Attribute splat = op.getValue().cast<SplatElementsAttr>().getSplatValue();
  auto scalarConstant =
      b.create<arith::ConstantOp>(op.getLoc(), splat.getType(), splat);
  const char *fragType = inferFragType(op);
  auto vecType = op.getType().cast<VectorType>();
  gpu::MMAMatrixType type = gpu::MMAMatrixType::get(
      vecType.getShape(), vecType.getElementType(), llvm::StringRef(fragType));
  auto matrix = b.create<gpu::SubgroupMmaConstantMatrixOp>(op.getLoc(), type,
                                                           scalarConstant);
  valueMapping[op.getResult()] = matrix;
}

/// Convert a vector.broadcast from scalar to a SubgroupMmaConstantMatrix op.
static void convertBroadcastOp(vector::BroadcastOp op,
                               llvm::DenseMap<Value, Value> &valueMapping) {
  assert(broadcastSupportsMMAMatrixType(op));
  OpBuilder b(op);
  const char *fragType = inferFragType(op);
  auto vecType = op.getVectorType();
  gpu::MMAMatrixType type = gpu::MMAMatrixType::get(
      vecType.getShape(), vecType.getElementType(), llvm::StringRef(fragType));
  auto matrix = b.create<gpu::SubgroupMmaConstantMatrixOp>(op.getLoc(), type,
                                                           op.source());
  valueMapping[op.getResult()] = matrix;
}

// Replace ForOp with a new ForOp with extra operands. The YieldOp is not
// updated and needs to be updated separatly for the loop to be correct.
static scf::ForOp replaceForOpWithNewSignature(OpBuilder &b, scf::ForOp loop,
                                               ValueRange newIterOperands) {
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getIterOperands());
  operands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop =
      b.create<scf::ForOp>(loop.getLoc(), loop.lowerBound(), loop.upperBound(),
                           loop.step(), operands);
  newLoop.getBody()->erase();
  newLoop.getLoopBody().getBlocks().splice(
      newLoop.getLoopBody().getBlocks().begin(),
      loop.getLoopBody().getBlocks());
  for (auto operand : newIterOperands)
    newLoop.getBody()->addArgument(operand.getType());

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  loop.erase();
  return newLoop;
}

static void convertForOp(scf::ForOp op,
                         llvm::DenseMap<Value, Value> &valueMapping) {
  SmallVector<Value> newOperands;
  SmallVector<std::pair<size_t, size_t>> argMapping;
  for (auto operand : llvm::enumerate(op.getIterOperands())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end())
      continue;
    argMapping.push_back(std::make_pair(
        operand.index(), op.getNumIterOperands() + newOperands.size()));
    newOperands.push_back(it->second);
  }
  OpBuilder b(op);
  scf::ForOp newForOp = replaceForOpWithNewSignature(b, op, newOperands);
  Block &loopBody = *newForOp.getBody();
  for (auto mapping : argMapping) {
    valueMapping[newForOp.getResult(mapping.first)] =
        newForOp.getResult(mapping.second);
    valueMapping[loopBody.getArgument(mapping.first +
                                      newForOp.getNumInductionVars())] =
        loopBody.getArgument(mapping.second + newForOp.getNumInductionVars());
  }
}

static void convertYieldOp(scf::YieldOp op,
                           llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  auto loop = cast<scf::ForOp>(op->getParentOp());
  auto yieldOperands = llvm::to_vector<4>(op.getOperands());
  for (auto operand : llvm::enumerate(op.getOperands())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end())
      continue;
    // Replace the yield of old value with the for op argument to make it easier
    // to remove the dead code.
    yieldOperands[operand.index()] = loop.getIterOperands()[operand.index()];
    yieldOperands.push_back(it->second);
  }
  b.create<scf::YieldOp>(op.getLoc(), yieldOperands);
  op.erase();
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
    } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      convertConstantOp(constantOp, valueMapping);
    } else if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
      convertBroadcastOp(broadcastOp, valueMapping);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      convertForOp(forOp, valueMapping);
    } else if (auto yiledOp = dyn_cast<scf::YieldOp>(op)) {
      convertYieldOp(yiledOp, valueMapping);
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
