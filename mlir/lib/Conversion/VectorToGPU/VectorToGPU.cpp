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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

// Return true if the contract op can be convert to MMA matmul.
static bool contractSupportsMMAMatrixType(vector::ContractionOp contract) {
  if (llvm::size(contract.getMasks()) != 0)
    return false;

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(contract.getContext(), m, n, k);
  auto iteratorTypes = contract.getIteratorTypes().getValue();
  if (!(isParallelIterator(iteratorTypes[0]) &&
        isParallelIterator(iteratorTypes[1]) &&
        isReductionIterator(iteratorTypes[2])))
    return false;

  // The contract needs to represent a matmul to be able to convert to
  // MMAMatrix matmul.
  if (contract.getIndexingMaps() != infer({{m, k}, {k, n}, {m, n}}))
    return false;

  return true;
}

// Return the stide for the dimension 0 of |type| if it is a memref and has a
// constant stride.
static llvm::Optional<int64_t>
getMemrefConstantHorizontalStride(ShapedType type) {
  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType)
    return false;
  // If the memref is 0 or 1D the horizontal stride is 0.
  if(memrefType.getRank() < 2)
    return 0;
  int64_t offset = 0;
  SmallVector<int64_t, 2> strides;
  if (failed(getStridesAndOffset(memrefType, strides, offset)) ||
      strides.back() != 1)
    return llvm::None;
  int64_t stride = strides[strides.size() - 2];
  if (stride == ShapedType::kDynamicStrideOrOffset)
    return llvm::None;
  return stride;
}

// Return true if the transfer op can be converted to a MMA matrix load.
static bool transferReadSupportsMMAMatrixType(vector::TransferReadOp readOp) {
  if (readOp.getMask() || readOp.hasOutOfBoundsDim() ||
      readOp.getVectorType().getRank() != 2)
    return false;
  if (!getMemrefConstantHorizontalStride(readOp.getShapedType()))
    return false;
  AffineMap map = readOp.getPermutationMap();
  OpBuilder b(readOp.getContext());
  AffineExpr innerDim = b.getAffineDimExpr(map.getNumDims() - 1);
  AffineExpr zero = b.getAffineConstantExpr(0);
  auto broadcastInnerDim = AffineMap::get(map.getNumDims(), 0, {zero, innerDim},
                                          readOp.getContext());
  // TODO: Support transpose once it is added to GPU dialect ops.
  // For now we only support (d0, d1) -> (d0, d1) and (d0, d1) -> (0, d1).
  return !(!map.isMinorIdentity() && map != broadcastInnerDim);
}

// Return true if the transfer op can be converted to a MMA matrix store.
static bool
transferWriteSupportsMMAMatrixType(vector::TransferWriteOp writeOp) {
  // TODO: support 0-d corner case.
  if (writeOp.getTransferRank() == 0)
    return false;

  if (writeOp.getMask() || writeOp.hasOutOfBoundsDim() ||
      writeOp.getVectorType().getRank() != 2)
    return false;
  if (!getMemrefConstantHorizontalStride(writeOp.getShapedType()))
    return false;
  // TODO: Support transpose once it is added to GPU dialect ops.
  if (!writeOp.getPermutationMap().isMinorIdentity())
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
         broadcastOp.getSource().getType().isa<FloatType>();
}

/// Return the MMA elementwise enum associated with `op` if it is supported.
/// Return `llvm::None` otherwise.
static llvm::Optional<gpu::MMAElementwiseOp>
convertElementwiseOpToMMA(Operation *op) {
  if (isa<arith::AddFOp>(op))
    return gpu::MMAElementwiseOp::ADDF;
  if (isa<arith::MulFOp>(op))
    return gpu::MMAElementwiseOp::MULF;
  if (isa<arith::MaxFOp>(op))
    return gpu::MMAElementwiseOp::MAXF;
  if (isa<arith::MinFOp>(op))
    return gpu::MMAElementwiseOp::MINF;
  if (isa<arith::DivFOp>(op))
    return gpu::MMAElementwiseOp::DIVF;
  return llvm::None;
}

/// Return true if the op is supported as elementwise op on MMAMatrix type.
static bool elementwiseSupportsMMAMatrixType(Operation *op) {
  return convertElementwiseOpToMMA(op).hasValue();
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
  return elementwiseSupportsMMAMatrixType(op);
}

/// Return an unsorted slice handling scf.for region differently than
/// `getSlice`. In scf.for we only want to include as part of the slice elements
/// that are part of the use/def chain.
static SetVector<Operation *> getSliceContract(Operation *op,
                                               TransitiveFilter backwardFilter,
                                               TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);
  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    getBackwardSlice(currentOp, &backwardSlice, backwardFilter);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    // Special case for ForOp, we don't want to include the whole region but
    // only the value using the region arguments.
    // TODO: We should refine this to only care about the region arguments being
    // converted to matrix type.
    if (auto forOp = dyn_cast<scf::ForOp>(currentOp)) {
      for (Value forOpResult : forOp.getResults())
        getForwardSlice(forOpResult, &forwardSlice, forwardFilter);
      for (BlockArgument &arg : forOp.getRegionIterArgs())
        getForwardSlice(arg, &forwardSlice, forwardFilter);
    } else {
      getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    }
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return slice;
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
        getSliceContract(contract, hasVectorDest, hasVectorSrc);
    // If any instruction cannot use MMA matrix type drop the whole
    // chain. MMA matrix are stored in an opaque type so they cannot be used
    // by all operations.
    if (llvm::any_of(dependentOps,
                     [](Operation *op) { return !supportsMMaMatrixType(op); }))
      return;
    opToConvert.insert(dependentOps.begin(), dependentOps.end());
  });
  // Sort the operations so that we can convert them in topological order.
  return topologicalSort(opToConvert);
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
    Value lhs = op.getLhs(), rhs = op.getRhs(), res = op.getAcc();

    // Set up the parallel/reduction structure in right form.
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto iteratorTypes = op.getIteratorTypes().getValue();
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
        op.getIteratorTypes());
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
    auto transferReadOp =
        op.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!transferReadOp)
      return failure();

    // TODO: support 0-d corner case.
    if (transferReadOp.getTransferRank() == 0)
      return failure();

    if (transferReadOp.getMask() || transferReadOp.hasOutOfBoundsDim())
      return failure();
    SmallVector<int64_t, 2> perm;
    op.getTransp(perm);
    SmallVector<unsigned, 2> permU;
    for (int64_t o : perm)
      permU.push_back(unsigned(o));
    AffineMap permutationMap =
        AffineMap::getPermutationMap(permU, op.getContext());
    AffineMap newMap =
        permutationMap.compose(transferReadOp.getPermutationMap());
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), transferReadOp.getSource(),
        transferReadOp.getIndices(), AffineMapAttr::get(newMap),
        transferReadOp.getPadding(), transferReadOp.getMask(),
        transferReadOp.getInBoundsAttr());
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
    if (contract.getLhs() == op.getResult())
      return "AOp";
    if (contract.getRhs() == op.getResult())
      return "BOp";
  }
  return "COp";
}

static void convertTransferReadOp(vector::TransferReadOp op,
                                  llvm::DenseMap<Value, Value> &valueMapping) {
  assert(op.getTransferRank() > 0 && "unexpected 0-d transfer");
  assert(transferReadSupportsMMAMatrixType(op));
  Optional<int64_t> stride =
      getMemrefConstantHorizontalStride(op.getShapedType());
  AffineMap map = op.getPermutationMap();
  // Handle broadcast by setting the stride to 0.
  if (map.getResult(0).isa<AffineConstantExpr>()) {
    assert(map.getResult(0).cast<AffineConstantExpr>().getValue() == 0);
    stride = 0;
  }
  assert(stride);
  const char *fragType = inferFragType(op);
  gpu::MMAMatrixType type =
      gpu::MMAMatrixType::get(op.getVectorType().getShape(),
                              op.getVectorType().getElementType(), fragType);
  OpBuilder b(op);
  Value load = b.create<gpu::SubgroupMmaLoadMatrixOp>(
      op.getLoc(), type, op.getSource(), op.getIndices(),
      b.getIndexAttr(*stride));
  valueMapping[op.getResult()] = load;
}

static void convertTransferWriteOp(vector::TransferWriteOp op,
                                   llvm::DenseMap<Value, Value> &valueMapping) {
  assert(transferWriteSupportsMMAMatrixType(op));
  Optional<int64_t> stride =
      getMemrefConstantHorizontalStride(op.getShapedType());
  assert(stride);
  OpBuilder b(op);
  Value matrix = valueMapping.find(op.getVector())->second;
  b.create<gpu::SubgroupMmaStoreMatrixOp>(op.getLoc(), matrix, op.getSource(),
                                          op.getIndices(),
                                          b.getIndexAttr(*stride));
  op.erase();
}

static void convertContractOp(vector::ContractionOp op,
                              llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  Value opA = valueMapping.find(op.getLhs())->second;
  Value opB = valueMapping.find(op.getRhs())->second;
  Value opC = valueMapping.find(op.getAcc())->second;
  Value matmul = b.create<gpu::SubgroupMmaComputeOp>(op.getLoc(), opC.getType(),
                                                     opA, opB, opC);
  valueMapping[op.getResult()] = matmul;
}

/// Convert a 2D splat ConstantOp to a SubgroupMmaConstantMatrix op.
static void convertConstantOp(arith::ConstantOp op,
                              llvm::DenseMap<Value, Value> &valueMapping) {
  assert(constantSupportsMMAMatrixType(op));
  OpBuilder b(op);
  Attribute splat =
      op.getValue().cast<SplatElementsAttr>().getSplatValue<Attribute>();
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
                                                           op.getSource());
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
      b.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                           loop.getUpperBound(), loop.getStep(), operands);
  newLoop.getBody()->erase();
  newLoop.getLoopBody().getBlocks().splice(
      newLoop.getLoopBody().getBlocks().begin(),
      loop.getLoopBody().getBlocks());
  for (Value operand : newIterOperands)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

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
  for (const auto &operand : llvm::enumerate(op.getIterOperands())) {
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
  for (const auto &operand : llvm::enumerate(op.getOperands())) {
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

/// Convert an elementwise op to the equivalent elementwise op on MMA matrix.
static void convertElementwiseOp(Operation *op, gpu::MMAElementwiseOp opType,
                                 llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  SmallVector<Value> matrixOperands;
  for (Value operand : op->getOperands())
    matrixOperands.push_back(valueMapping.find(operand)->second);
  Value newOp = b.create<gpu::SubgroupMmaElementwiseOp>(
      op->getLoc(), matrixOperands[0].getType(), matrixOperands, opType);
  valueMapping[op->getResult(0)] = newOp;
}

void mlir::populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns) {
  patterns.add<PrepareContractToGPUMMA, CombineTransferReadOpTranspose>(
      patterns.getContext());
}

void mlir::convertVectorToMMAOps(Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp);
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
    } else if (auto elementwiseType = convertElementwiseOpToMMA(op)) {
      convertElementwiseOp(op, *elementwiseType, valueMapping);
    }
  }
}

namespace {

struct ConvertVectorToGPUPass
    : public ConvertVectorToGPUBase<ConvertVectorToGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePrepareVectorToMMAPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    convertVectorToMMAOps(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertVectorToGPUPass() {
  return std::make_unique<ConvertVectorToGPUPass>();
}
