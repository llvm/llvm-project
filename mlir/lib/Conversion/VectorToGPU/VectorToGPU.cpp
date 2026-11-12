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

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/VectorToGPUCommon/VectorToGPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "vector-to-gpu"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOGPUPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

// Return true if the contract op can be convert to MMA matmul.
static bool contractSupportsMMAMatrixType(vector::ContractionOp contract) {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [&](MapList m) {
    return AffineMap::inferFromExprList(m, contract.getContext());
  };
  AffineExpr m, n, k;
  bindDims(contract.getContext(), m, n, k);
  auto iteratorTypes = contract.getIteratorTypes().getValue();
  if (!(vector::isParallelIterator(iteratorTypes[0]) &&
        vector::isParallelIterator(iteratorTypes[1]) &&
        vector::isReductionIterator(iteratorTypes[2])))
    return false;

  // The contract needs to represent a matmul to be able to convert to
  // MMAMatrix matmul.
  if (contract.getIndexingMapsArray() != infer({{m, k}, {k, n}, {m, n}}))
    return false;

  return true;
}

// Test whether the permutation map's first result corresponds to its last
// dimension.
//
// In contexts where we only accept maps that have the last (most minor)
// dimension as exactly one of the two results, this is sufficient to classify
// whether it represents a transpose.
static bool isFirstResultLastMapDimension(AffineMap permutationMap) {
  MLIRContext *ctx = permutationMap.getContext();
  const unsigned nDim = permutationMap.getNumDims();
  if (0 == nDim || permutationMap.getResults().empty())
    return false;
  return permutationMap.getResult(0) == getAffineDimExpr(nDim - 1, ctx);
}

// Return the `leadDimension` (row stride) implied by |permutationMap| for
// |type|, if |type| is a memref with a statically-known layout.
//
// The `leadDimension` is the stride (in elements) between consecutive rows in
// the 2D view described by |permutationMap|. This helper supports the subset
// of maps permitted by vector.transfer_read:
// - Exactly 2 results.
// - Each result is either an affine dimension or the constant 0 (broadcast).
//
// Constraints:
// - Requires the most minor memref stride to be 1.
//
// Broadcast:
// - If either result is constant 0, the implied `leadDimension` is 0.
static std::optional<int64_t>
getStaticallyKnownRowStride(ShapedType type, AffineMap permutationMap) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return std::nullopt;
  // If the memref is 0 or 1D the horizontal stride is 0.
  if (memrefType.getRank() < 2)
    return 0;
  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(memrefType.getStridesAndOffset(strides, offset)) ||
      strides.back() != 1)
    return std::nullopt;

  if (permutationMap.getNumResults() != 2)
    return std::nullopt;

  unsigned strideIndex = strides.size();

  for (AffineExpr result : permutationMap.getResults()) {
    if (auto cst = dyn_cast<AffineConstantExpr>(result)) {
      // Constant value must be zero.
      if (0 != cst.getValue())
        return std::nullopt;
      // A broadcast result forces row stride to 0.
      return 0;
    }
    auto dim = dyn_cast<AffineDimExpr>(result);
    // Only Dim & Const results are supported.
    if (!dim)
      return std::nullopt;
    strideIndex = std::min(strideIndex, dim.getPosition());
  }

  // Structural validity check: ensure that the map selects at least one
  // dimension more major than the most minor dimension. This also excludes
  // degenerate cases where both results map to the most minor dimension.
  if (strideIndex + 1 >= strides.size())
    return std::nullopt;

  const int64_t stride = strides[strideIndex];
  if (stride == ShapedType::kDynamic)
    return std::nullopt;
  return stride;
}

// Return true if the transfer op can be converted to a MMA matrix load.
static bool transferReadSupportsMMAMatrixType(vector::TransferReadOp readOp) {
  if (readOp.getMask() || readOp.hasOutOfBoundsDim() ||
      readOp.getVectorType().getRank() != 2)
    return false;

  AffineMap permutationMap = readOp.getPermutationMap();
  if (!getStaticallyKnownRowStride(readOp.getShapedType(), permutationMap))
    return false;

  // Only allow integer types if the signedness can be inferred.
  if (readOp.getVectorType().getElementType().isInteger(8))
    if (!readOp->hasOneUse() || (!isa<arith::ExtSIOp>(*readOp->user_begin()) &&
                                 !isa<arith::ExtUIOp>(*readOp->user_begin())))
      return false;

  MLIRContext *ctx = readOp.getContext();
  AffineExpr innerDim = getAffineDimExpr(permutationMap.getNumDims() - 1, ctx);
  return llvm::is_contained(permutationMap.getResults(), innerDim);
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

  AffineMap permutationMap = writeOp.getPermutationMap();
  std::optional<int64_t> stride =
      getStaticallyKnownRowStride(writeOp.getShapedType(), permutationMap);
  // Stride of zero means broadcast which is not permitted for writes.
  if (!stride.has_value() || stride.value() == 0)
    return false;

  MLIRContext *ctx = writeOp.getContext();
  AffineExpr innerDim = getAffineDimExpr(permutationMap.getNumDims() - 1, ctx);
  return llvm::is_contained(permutationMap.getResults(), innerDim);
}

static bool supportsMMaMatrixType(Operation *op) {
  if (isa<scf::ForOp, scf::YieldOp>(op))
    return true;
  // MMA ops cannot honor a mask. An op masked by an enclosing vector.mask has
  // no mask operand, so the checks below would miss it; rewriting it in place
  // would leave non-maskable ops inside the vector.mask region.
  auto maskableOp = dyn_cast<vector::MaskableOpInterface>(op);
  if (maskableOp && maskableOp.isMasked())
    return false;
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
  if (auto signedExtend = dyn_cast<arith::ExtSIOp>(op))
    return integerExtendSupportsMMAMatrixType<arith::ExtSIOp>(signedExtend);
  if (auto unsignedExtend = dyn_cast<arith::ExtUIOp>(op))
    return integerExtendSupportsMMAMatrixType<arith::ExtUIOp>(unsignedExtend);
  return elementwiseSupportsMMAMatrixType(op);
}

// Analyze slice of operations based on convert op to figure out if the whole
// slice can be converted to MMA operations.
static SetVector<Operation *> getOpToConvert(mlir::Operation *op) {
  auto hasVectorDest = [](Operation *op) {
    return llvm::any_of(op->getResultTypes(), llvm::IsaPred<VectorType>);
  };
  BackwardSliceOptions backwardSliceOptions;
  backwardSliceOptions.filter = hasVectorDest;

  auto hasVectorSrc = [](Operation *op) {
    return llvm::any_of(op->getOperandTypes(), llvm::IsaPred<VectorType>);
  };
  ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.filter = hasVectorSrc;

  DenseMap<Operation *, SmallVector<Operation *>> backwardSliceCache;
  DenseMap<Operation *, SmallVector<Operation *>> forwardSliceCache;
  DenseMap<Operation *, bool> supportsMMAMatrixTypeCache;

  auto getCachedBackwardSlice =
      [&](Operation *currentOp) -> ArrayRef<Operation *> {
    auto [it, inserted] = backwardSliceCache.try_emplace(currentOp);
    if (!inserted)
      return it->second;

    SetVector<Operation *> backwardSlice;
    LogicalResult result =
        getBackwardSlice(currentOp, &backwardSlice, backwardSliceOptions);
    assert(result.succeeded() && "expected a backward slice");
    (void)result;
    it->second = backwardSlice.takeVector();
    return it->second;
  };

  auto getCachedForwardSlice =
      [&](Operation *currentOp) -> ArrayRef<Operation *> {
    auto [it, inserted] = forwardSliceCache.try_emplace(currentOp);
    if (!inserted)
      return it->second;

    SetVector<Operation *> forwardSlice;
    // Special case for ForOp, we don't want to include the whole region but
    // only the value using the region arguments.
    // TODO: We should refine this to only care about the region arguments being
    // converted to matrix type.
    if (auto forOp = dyn_cast<scf::ForOp>(currentOp)) {
      for (Value forOpResult : forOp.getResults())
        getForwardSlice(forOpResult, &forwardSlice, forwardSliceOptions);
      for (BlockArgument &arg : forOp.getRegionIterArgs())
        getForwardSlice(arg, &forwardSlice, forwardSliceOptions);
    } else {
      getForwardSlice(currentOp, &forwardSlice, forwardSliceOptions);
    }
    it->second = forwardSlice.takeVector();
    return it->second;
  };

  auto cachedSupportsMMAMatrixType = [&](Operation *currentOp) {
    auto [it, inserted] =
        supportsMMAMatrixTypeCache.try_emplace(currentOp, false);
    if (inserted)
      it->second = supportsMMaMatrixType(currentOp);
    return it->second;
  };

  SetVector<Operation *> opToConvert;
  op->walk([&](Operation *nestedOp) {
    if (!isa<vector::ContractionOp>(nestedOp) &&
        !elementwiseSupportsMMAMatrixType(nestedOp))
      return;
    if (backwardSliceCache.contains(nestedOp))
      return;

    SetVector<Operation *> dependentOps;
    dependentOps.insert(nestedOp);
    unsigned currentIndex = 0;
    while (currentIndex != dependentOps.size()) {
      Operation *currentOp = dependentOps[currentIndex++];
      dependentOps.insert_range(getCachedBackwardSlice(currentOp));
      dependentOps.insert_range(getCachedForwardSlice(currentOp));
    }
    // If any instruction cannot use MMA matrix type drop the whole
    // chain. MMA matrix are stored in an opaque type so they cannot be used
    // by all operations.
    if (llvm::any_of(dependentOps, [&](Operation *op) {
          if (!cachedSupportsMMAMatrixType(op)) {
            LDBG() << "cannot convert op: " << *op;
            return true;
          }
          return false;
        }))
      return;

    opToConvert.insert_range(dependentOps);
  });
  // Sort the operations so that we can convert them in topological order.
  return topologicalSort(opToConvert);
}

namespace {
// Transform contract into (m, k)x(k, n)x(m, n) form so that it can be converted
// to MMA matmul.
struct PrepareContractToGPUMMA
    : public OpRewritePattern<vector::ContractionOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // The mask is not updated: the rewrite may create ops inside the
    // vector.mask region or swap the m and n iteration dimensions.
    if (op.isMasked())
      return rewriter.notifyMatchFailure(op, "masked contraction");

    Location loc = op.getLoc();
    Value lhs = op.getLhs(), rhs = op.getRhs(), res = op.getAcc();

    // Set up the parallel/reduction structure in right form.
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, op.getContext());
    };
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto iteratorTypes = op.getIteratorTypes().getValue();
    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (!(vector::isParallelIterator(iteratorTypes[0]) &&
          vector::isParallelIterator(iteratorTypes[1]) &&
          vector::isReductionIterator(iteratorTypes[2])))
      return rewriter.notifyMatchFailure(op, "not a gemm contraction");
    //
    // Two outer parallel, one inner reduction (matmat flavor).
    //
    // This is the classical row-major matmul, nothing to do.
    if (maps == infer({{m, k}, {k, n}, {m, n}}))
      return rewriter.notifyMatchFailure(op, "contraction already prepared");
    if (maps == infer({{m, k}, {n, k}, {m, n}})) {
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, perm);
    } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, perm);
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, perm);
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, perm);
    } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, perm);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      std::swap(lhs, rhs);
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      std::swap(lhs, rhs);
    } else {
      // TODO: llvm_unreachable ?
      return rewriter.notifyMatchFailure(op, "unexpected contraction case");
    }
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, lhs, rhs, res,
        rewriter.getAffineMapArrayAttr(infer({{m, k}, {k, n}, {m, n}})),
        op.getIteratorTypes(), op.getKind(), op.getFastmath());
    return success();
  }
};
} // namespace

// MMA types have different layout based on how they are used in matmul ops.
// Figure the right layout to use by looking at op uses.
// TODO: Change the GPU dialect to abstract the layout at the this level and
// only care about it during lowering to NVVM.
static const char *inferFragType(Operation *op) {
  // We can have arith.ext ops before reaching contract ops. See through them
  // and other kinds of elementwise ops.
  if (op->hasOneUse()) {
    Operation *userOp = *op->user_begin();
    if (userOp->hasTrait<OpTrait::Elementwise>())
      return inferFragType(userOp);
  }

  for (auto contract :
       llvm::make_isa_range<vector::ContractionOp>(op->getUsers())) {
    assert(op->getNumResults() == 1);
    if (contract.getLhs() == op->getResult(0))
      return "AOp";
    if (contract.getRhs() == op->getResult(0))
      return "BOp";
  }
  return "COp";
}

static LogicalResult
convertTransferReadOp(RewriterBase &rewriter, vector::TransferReadOp op,
                      llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  assert(op.getTransferRank() > 0 && "unexpected 0-d transfer");
  assert(transferReadSupportsMMAMatrixType(op) &&
         "expected convertible operation");

  AffineMap permutationMap = op.getPermutationMap();
  std::optional<int64_t> stride =
      getStaticallyKnownRowStride(op.getShapedType(), permutationMap);
  if (!stride.has_value()) {
    LDBG() << "no stride";
    return rewriter.notifyMatchFailure(op, "no stride");
  }

  // transferReadSupportsMMAMatrixType ensures that either of the map results is
  // the most minor dimension. Under this constraint, whether the map represents
  // a transposed view can be inferred from whether the first result is the most
  // minor memref dimension.
  const bool isTranspose = isFirstResultLastMapDimension(permutationMap);

  Value mappingResult = op.getResult();
  auto elType = op.getVectorType().getElementType();
  const char *fragType = inferFragType(op);
  if (op->hasOneUse()) {
    auto *user = *op->user_begin();
    // Infer the signedness of the mma type from the integer extend.
    if (isa<arith::ExtSIOp, arith::ExtUIOp>(user)) {
      elType = IntegerType::get(
          op.getContext(), cast<IntegerType>(elType).getWidth(),
          isa<arith::ExtSIOp>(user) ? IntegerType::Signed
                                    : IntegerType::Unsigned);
      mappingResult = user->getResult(0);
    }
  }
  gpu::MMAMatrixType type =
      gpu::MMAMatrixType::get(op.getVectorType().getShape(), elType, fragType);
  Value load = gpu::SubgroupMmaLoadMatrixOp::create(
      rewriter, op.getLoc(), type, op.getBase(), op.getIndices(),
      rewriter.getIndexAttr(*stride),
      isTranspose ? rewriter.getUnitAttr() : UnitAttr());
  valueMapping[mappingResult] = load;

  LDBG() << "transfer read to: " << load;
  return success();
}

static LogicalResult
convertTransferWriteOp(RewriterBase &rewriter, vector::TransferWriteOp op,
                       llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  assert(transferWriteSupportsMMAMatrixType(op));
  AffineMap permutationMap = op.getPermutationMap();
  std::optional<int64_t> stride =
      getStaticallyKnownRowStride(op.getShapedType(), permutationMap);
  if (!stride.has_value()) {
    LDBG() << "no stride";
    return rewriter.notifyMatchFailure(op, "no stride");
  }

  // As for transfer_read, transferWriteSupportsMMAMatrixType ensures that
  // either of the map results is the most minor dimension, so the first result
  // being that dimension means a transposed store.
  const bool isTranspose = isFirstResultLastMapDimension(permutationMap);

  auto it = valueMapping.find(op.getVector());
  if (it == valueMapping.end()) {
    LDBG() << "no mapping";
    return rewriter.notifyMatchFailure(op, "no mapping");
  }

  Value matrix = it->second;
  auto store = gpu::SubgroupMmaStoreMatrixOp::create(
      rewriter, op.getLoc(), matrix, op.getBase(), op.getIndices(),
      rewriter.getIndexAttr(*stride),
      isTranspose ? rewriter.getUnitAttr() : UnitAttr());
  (void)store;

  LDBG() << "transfer write to: " << store;

  LDBG() << "erase: " << op;
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult
convertContractOp(RewriterBase &rewriter, vector::ContractionOp op,
                  llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto itA = valueMapping.find(op.getLhs());
  auto itB = valueMapping.find(op.getRhs());
  auto itC = valueMapping.find(op.getAcc());
  if (itA == valueMapping.end() || itB == valueMapping.end() ||
      itC == valueMapping.end())
    return rewriter.notifyMatchFailure(op, "no mapping");
  Value opA = itA->second, opB = itB->second, opC = itC->second;
  Value matmul = gpu::SubgroupMmaComputeOp::create(rewriter, op.getLoc(),
                                                   opC.getType(), opA, opB, opC,
                                                   /*a_transpose=*/UnitAttr(),
                                                   /*b_transpose=*/UnitAttr());
  valueMapping[op.getResult()] = matmul;
  return success();
}

/// Convert a 2D splat ConstantOp to a SubgroupMmaConstantMatrix op.
static LogicalResult
convertConstantOp(RewriterBase &rewriter, arith::ConstantOp op,
                  llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  assert(constantSupportsMMAMatrixType(op));

  auto splat =
      cast<SplatElementsAttr>(op.getValue()).getSplatValue<TypedAttr>();
  auto scalarConstant =
      arith::ConstantOp::create(rewriter, op.getLoc(), splat.getType(), splat);
  const char *fragType = inferFragType(op);
  auto vecType = cast<VectorType>(op.getType());
  gpu::MMAMatrixType type = gpu::MMAMatrixType::get(
      vecType.getShape(), vecType.getElementType(), llvm::StringRef(fragType));
  auto matrix = gpu::SubgroupMmaConstantMatrixOp::create(rewriter, op.getLoc(),
                                                         type, scalarConstant);
  valueMapping[op.getResult()] = matrix;
  return success();
}

/// Convert a vector.broadcast from scalar to a SubgroupMmaConstantMatrix op.
static LogicalResult
convertBroadcastOp(RewriterBase &rewriter, vector::BroadcastOp op,
                   llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  assert(broadcastSupportsMMAMatrixType(op));

  const char *fragType = inferFragType(op);
  auto vecType = op.getResultVectorType();
  gpu::MMAMatrixType type = gpu::MMAMatrixType::get(
      vecType.getShape(), vecType.getElementType(), llvm::StringRef(fragType));
  auto matrix = gpu::SubgroupMmaConstantMatrixOp::create(rewriter, op.getLoc(),
                                                         type, op.getSource());
  valueMapping[op.getResult()] = matrix;
  return success();
}

/// Convert an elementwise op to the equivalent elementwise op on MMA matrix.
static LogicalResult
convertElementwiseOp(RewriterBase &rewriter, Operation *op,
                     gpu::MMAElementwiseOp opType,
                     llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  SmallVector<Value> matrixOperands;
  for (Value operand : op->getOperands()) {
    auto it = valueMapping.find(operand);
    if (it == valueMapping.end())
      return rewriter.notifyMatchFailure(op, "no mapping");
    matrixOperands.push_back(it->second);
  }
  auto resultType = cast<gpu::MMAMatrixType>(matrixOperands[0].getType());
  if (opType == gpu::MMAElementwiseOp::EXTF ||
      opType == gpu::MMAElementwiseOp::TRUNCF) {
    // The floating point extension and truncation has a different result type.
    auto vectorType = cast<VectorType>(op->getResultTypes()[0]);
    resultType = gpu::MMAMatrixType::get(resultType.getShape(),
                                         vectorType.getElementType(),
                                         resultType.getOperand());
  }

  Value newOp = gpu::SubgroupMmaElementwiseOp::create(
      rewriter, op->getLoc(), resultType, matrixOperands, opType);
  valueMapping[op->getResult(0)] = newOp;
  return success();
}

void mlir::populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns) {
  patterns.add<PrepareContractToGPUMMA>(patterns.getContext());
  populateVectorTransferReadTransposePatterns(patterns);
}

LogicalResult mlir::convertVectorToMMAOps(RewriterBase &rewriter,
                                          Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp);
  llvm::DenseMap<Value, Value> valueMapping;

  auto globalRes = LogicalResult::success();
  for (Operation *op : ops) {
    LDBG() << "Process op: " << *op;
    // Apparently callers do not want to early exit on failure here.
    auto res = LogicalResult::success();
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
      res = convertTransferReadOp(rewriter, transferRead, valueMapping);
    } else if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
      res = convertTransferWriteOp(rewriter, transferWrite, valueMapping);
    } else if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      res = convertContractOp(rewriter, contractOp, valueMapping);
    } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      res = convertConstantOp(rewriter, constantOp, valueMapping);
    } else if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
      res = convertBroadcastOp(rewriter, broadcastOp, valueMapping);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      res = convertForOpForMMA(rewriter, forOp, valueMapping);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      res = convertYieldOpForMMA(rewriter, yieldOp, valueMapping);
    } else if (auto elementwiseType = convertElementwiseOpToMMA(op)) {
      res = convertElementwiseOp(rewriter, op, *elementwiseType, valueMapping);
    }
    if (failed(res))
      globalRes = failure();
  }
  return globalRes;
}

namespace {
struct ConvertVectorToGPUPass
    : public impl::ConvertVectorToGPUPassBase<ConvertVectorToGPUPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePrepareVectorToMMAPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    (void)convertVectorToMMAOps(rewriter, getOperation());
  }
};
} // namespace
