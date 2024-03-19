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

#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "vector-to-gpu"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOGPU
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// For a vector TransferOpType `xferOp`, an empty `indices` vector, and an
/// AffineMap representing offsets to apply to indices, the function fills
/// `indices` with the original indices plus the offsets. The offsets are
/// applied by taking into account the permutation map of the transfer op. If
/// the `offsetMap` has dimension placeholders, those should be provided in
/// `dimValues`.
template <typename TransferOpType>
static void getXferIndices(RewriterBase &rewriter, TransferOpType xferOp,
                           AffineMap offsetMap, ArrayRef<Value> dimValues,
                           SmallVector<Value, 4> &indices) {
  indices.append(xferOp.getIndices().begin(), xferOp.getIndices().end());
  Location loc = xferOp.getLoc();
  unsigned offsetsIdx = 0;
  for (auto expr : xferOp.getPermutationMap().getResults()) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      Value prevIdx = indices[dim.getPosition()];
      SmallVector<OpFoldResult, 3> dims(dimValues.begin(), dimValues.end());
      dims.push_back(prevIdx);
      AffineExpr d0 = rewriter.getAffineDimExpr(offsetMap.getNumDims());
      indices[dim.getPosition()] = affine::makeComposedAffineApply(
          rewriter, loc, d0 + offsetMap.getResult(offsetsIdx++), dims);
      continue;
    }
  }
}

// Return true if the contract op can be convert to MMA matmul.
static bool contractSupportsMMAMatrixType(vector::ContractionOp contract,
                                          bool useNvGpu) {
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
  if (!useNvGpu &&
      contract.getIndexingMapsArray() != infer({{m, k}, {k, n}, {m, n}}))
    return false;
  if (useNvGpu &&
      contract.getIndexingMapsArray() != infer({{m, k}, {n, k}, {m, n}}))
    return false;

  return true;
}

// Return true if the given map represents a transposed matrix load,
// i.e. (d0, d1, ...) -> (dn-1, dn-2).
static bool isTransposeMatrixLoadMap(AffineMap permutationMap) {
  MLIRContext *ctx = permutationMap.getContext();
  // Local OpBuilder is fine here, we just build attributes.
  OpBuilder b(ctx);
  auto nDim = permutationMap.getNumDims();
  AffineExpr zero = b.getAffineConstantExpr(0);
  if (nDim < 2) {
    // Support transposed+broadcasted cases: affine_map<(d0) -> (d0, 0)>.
    AffineExpr dim0 = b.getAffineDimExpr(0);
    return permutationMap == AffineMap::get(1, 0, {dim0, zero}, ctx);
  }

  AffineExpr innerDim = b.getAffineDimExpr(nDim - 1);
  AffineExpr outerDim = b.getAffineDimExpr(nDim - 2);
  // Support both transposed and transposed+broadcasted cases.
  return permutationMap == AffineMap::get(nDim, 0, {innerDim, outerDim}, ctx) ||
         permutationMap == AffineMap::get(nDim, 0, {innerDim, zero}, ctx);
}

// Return the stide for the second-to-last dimension of |type| if it is a memref
// and has a constant stride.
static std::optional<int64_t> getStaticallyKnownRowStride(ShapedType type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return false;
  // If the memref is 0 or 1D the horizontal stride is 0.
  if (memrefType.getRank() < 2)
    return 0;
  int64_t offset = 0;
  SmallVector<int64_t, 2> strides;
  if (failed(getStridesAndOffset(memrefType, strides, offset)) ||
      strides.back() != 1)
    return std::nullopt;
  int64_t stride = strides[strides.size() - 2];
  if (stride == ShapedType::kDynamic)
    return std::nullopt;
  return stride;
}

// Return true if the transfer op can be converted to a MMA matrix load.
static bool transferReadSupportsMMAMatrixType(vector::TransferReadOp readOp) {
  if (readOp.getMask() || readOp.hasOutOfBoundsDim() ||
      readOp.getVectorType().getRank() != 2)
    return false;
  if (!getStaticallyKnownRowStride(readOp.getShapedType()))
    return false;

  // Only allow integer types if the signedness can be inferred.
  if (readOp.getVectorType().getElementType().isInteger(8))
    if (!readOp->hasOneUse() || (!isa<arith::ExtSIOp>(*readOp->user_begin()) &&
                                 !isa<arith::ExtUIOp>(*readOp->user_begin())))
      return false;

  AffineMap map = readOp.getPermutationMap();
  MLIRContext *ctx = readOp.getContext();
  AffineExpr innerDim = getAffineDimExpr(map.getNumDims() - 1, ctx);
  AffineExpr zero = getAffineConstantExpr(0, ctx);
  auto broadcastInnerDim =
      AffineMap::get(map.getNumDims(), 0, {zero, innerDim}, ctx);
  return map.isMinorIdentity() || map == broadcastInnerDim ||
         isTransposeMatrixLoadMap(map);
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
  if (!getStaticallyKnownRowStride(writeOp.getShapedType()))
    return false;
  // TODO: Support transpose once it is added to GPU dialect ops.
  if (!writeOp.getPermutationMap().isMinorIdentity())
    return false;
  return true;
}

/// Return true if the constant is a splat to a 2D vector so that it can be
/// converted to a MMA constant matrix op.
static bool constantSupportsMMAMatrixType(arith::ConstantOp constantOp) {
  auto vecType = dyn_cast<VectorType>(constantOp.getType());
  if (!vecType || vecType.getRank() != 2)
    return false;
  return isa<SplatElementsAttr>(constantOp.getValue());
}

/// Return true if this is a broadcast from scalar to a 2D vector.
static bool broadcastSupportsMMAMatrixType(vector::BroadcastOp broadcastOp) {
  return broadcastOp.getResultVectorType().getRank() == 2;
}

/// Return true if this integer extend op can be folded into a contract op.
template <typename ExtOpTy>
static bool integerExtendSupportsMMAMatrixType(ExtOpTy extOp) {
  if (!isa<vector::TransferReadOp>(extOp.getOperand().getDefiningOp()))
    return false;
  return llvm::all_of(extOp->getUsers(), [](Operation *user) {
    return isa<vector::ContractionOp>(user);
  });
}

static bool fpExtendSupportsMMAMatrixType(arith::ExtFOp extOp) { return true; }

/// Return the MMA elementwise enum associated with `op` if it is supported.
/// Return `std::nullopt` otherwise.
static std::optional<gpu::MMAElementwiseOp>
convertElementwiseOpToMMA(Operation *op) {
  if (isa<arith::AddFOp>(op))
    return gpu::MMAElementwiseOp::ADDF;
  if (isa<arith::MulFOp>(op))
    return gpu::MMAElementwiseOp::MULF;
  if (isa<arith::SubFOp>(op))
    return gpu::MMAElementwiseOp::SUBF;
  if (isa<arith::MaximumFOp>(op))
    return gpu::MMAElementwiseOp::MAXF;
  if (isa<arith::MinimumFOp>(op))
    return gpu::MMAElementwiseOp::MINF;
  if (isa<arith::DivFOp>(op))
    return gpu::MMAElementwiseOp::DIVF;
  if (isa<arith::AddIOp>(op))
    return gpu::MMAElementwiseOp::ADDI;
  if (isa<arith::MulIOp>(op))
    return gpu::MMAElementwiseOp::MULI;
  if (isa<arith::SubIOp>(op))
    return gpu::MMAElementwiseOp::SUBI;
  if (isa<arith::DivSIOp>(op))
    return gpu::MMAElementwiseOp::DIVS;
  if (isa<arith::DivUIOp>(op))
    return gpu::MMAElementwiseOp::DIVU;
  if (isa<arith::NegFOp>(op))
    return gpu::MMAElementwiseOp::NEGATEF;
  if (isa<arith::ExtFOp>(op))
    return gpu::MMAElementwiseOp::EXTF;
  return std::nullopt;
}

/// Return true if the op is supported as elementwise op on MMAMatrix type.
static bool elementwiseSupportsMMAMatrixType(Operation *op) {
  return convertElementwiseOpToMMA(op).has_value();
}

/// Returns true if the extract strided slice op is supported with `mma.sync`
/// path.
static bool
extractStridedSliceSupportsMMAMatrixType(vector::ExtractStridedSliceOp op) {

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return false;

  FailureOr<vector::ContractionOp> contractOp = nvgpu::getUserContract(op);
  if (failed(contractOp))
    return false;

  // Handle vector.extract_strided_slice on registers containing
  // matrixB and matrixC operands. vector.extract_strided_slice op
  // is not supported on registers containing matrixA operands.
  if (warpMatrixInfo->operandRole == nvgpu::MatMulOperandRole::B)
    return (cast<VectorType>(op->getResult(0).getType()) ==
            cast<VectorType>((*contractOp).getRhs().getType()));
  if (warpMatrixInfo->operandRole == nvgpu::MatMulOperandRole::C)
    return (cast<VectorType>(op->getResult(0).getType()) ==
            cast<VectorType>((*contractOp).getAcc().getType()));

  return false;
}

static bool supportsMMaMatrixType(Operation *op, bool useNvGpu) {
  if (isa<scf::ForOp, scf::YieldOp>(op))
    return true;
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op))
    return useNvGpu ? nvgpu::canLowerToWarpMatrixOperation(transferRead)
                    : transferReadSupportsMMAMatrixType(transferRead);
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return useNvGpu ? nvgpu::canLowerToWarpMatrixOperation(transferWrite)
                    : transferWriteSupportsMMAMatrixType(transferWrite);
  if (auto extractStridedSlice = dyn_cast<vector::ExtractStridedSliceOp>(op))
    return useNvGpu &&
           extractStridedSliceSupportsMMAMatrixType(extractStridedSlice);
  if (auto contract = dyn_cast<vector::ContractionOp>(op))
    return contractSupportsMMAMatrixType(contract, useNvGpu);
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return constantSupportsMMAMatrixType(constant);
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op))
    return broadcastSupportsMMAMatrixType(broadcast);
  if (auto signedExtend = dyn_cast<arith::ExtSIOp>(op))
    return integerExtendSupportsMMAMatrixType<arith::ExtSIOp>(signedExtend);
  if (auto unsignedExtend = dyn_cast<arith::ExtUIOp>(op))
    return integerExtendSupportsMMAMatrixType<arith::ExtUIOp>(unsignedExtend);
  if (auto fpExtend = dyn_cast<arith::ExtFOp>(op))
    return fpExtendSupportsMMAMatrixType(fpExtend);
  return elementwiseSupportsMMAMatrixType(op);
}

/// Return an unsorted slice handling scf.for region differently than
/// `getSlice`. In scf.for we only want to include as part of the slice elements
/// that are part of the use/def chain.
static SetVector<Operation *>
getSliceContract(Operation *op,
                 const BackwardSliceOptions &backwardSliceOptions,
                 const ForwardSliceOptions &forwardSliceOptions) {
  SetVector<Operation *> slice;
  slice.insert(op);
  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    getBackwardSlice(currentOp, &backwardSlice, backwardSliceOptions);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
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
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return slice;
}

// Analyze slice of operations based on convert op to figure out if the whole
// slice can be converted to MMA operations.
static SetVector<Operation *> getOpToConvert(mlir::Operation *op,
                                             bool useNvGpu) {
  auto hasVectorDest = [](Operation *op) {
    return llvm::any_of(op->getResultTypes(),
                        [](Type t) { return isa<VectorType>(t); });
  };
  BackwardSliceOptions backwardSliceOptions;
  backwardSliceOptions.filter = hasVectorDest;

  auto hasVectorSrc = [](Operation *op) {
    return llvm::any_of(op->getOperandTypes(),
                        [](Type t) { return isa<VectorType>(t); });
  };
  ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.filter = hasVectorSrc;

  SetVector<Operation *> opToConvert;
  op->walk([&](vector::ContractionOp contract) {
    if (opToConvert.contains(contract.getOperation()))
      return;
    SetVector<Operation *> dependentOps =
        getSliceContract(contract, backwardSliceOptions, forwardSliceOptions);
    // If any instruction cannot use MMA matrix type drop the whole
    // chain. MMA matrix are stored in an opaque type so they cannot be used
    // by all operations.
    if (llvm::any_of(dependentOps, [useNvGpu](Operation *op) {
          if (!supportsMMaMatrixType(op, useNvGpu)) {
            LLVM_DEBUG(DBGS() << "cannot convert op: " << *op << "\n");
            return true;
          }
          return false;
        }))
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
      // TODO: llvm_unreachable ?
      return rewriter.notifyMatchFailure(op, "unexpected contraction case");
    }
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, lhs, rhs, res,
        rewriter.getAffineMapArrayAttr(infer({{m, k}, {k, n}, {m, n}})),
        op.getIteratorTypes());
    return success();
  }
};

// Fold transpose op into the transfer read op. Nvgpu mma.sync op only supports
// row-, column-, and row-major layout for matrixA, matrixB, and matrixC,
// respectively. We can fold the transpose operation when loading the data from
// Shared Memory to registers.
struct CombineTransferReadOpTranspose final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Look through integer extend ops.
    Value source = op.getVector();
    Type resultType = op.getType();
    Operation *extOp;
    if ((extOp = source.getDefiningOp<arith::ExtSIOp>()) ||
        (extOp = source.getDefiningOp<arith::ExtUIOp>()) ||
        (extOp = source.getDefiningOp<arith::ExtFOp>())) {
      source = extOp->getOperand(0);
      resultType =
          VectorType::get(cast<VectorType>(resultType).getShape(),
                          cast<VectorType>(source.getType()).getElementType());
    }

    auto transferReadOp = source.getDefiningOp<vector::TransferReadOp>();
    if (!transferReadOp)
      return rewriter.notifyMatchFailure(op, "no transfer read");

    // TODO: support 0-d corner case.
    if (transferReadOp.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-D transfer read");

    if (transferReadOp.getMask() || transferReadOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(op, "not inbounds transfer read");

    AffineMap permutationMap =
        AffineMap::getPermutationMap(op.getPermutation(), op.getContext());
    AffineMap newMap =
        permutationMap.compose(transferReadOp.getPermutationMap());

    auto loc = op.getLoc();
    Value result =
        rewriter
            .create<vector::TransferReadOp>(
                loc, resultType, transferReadOp.getSource(),
                transferReadOp.getIndices(), AffineMapAttr::get(newMap),
                transferReadOp.getPadding(), transferReadOp.getMask(),
                transferReadOp.getInBoundsAttr())
            .getResult();

    // Fuse through the integer extend op.
    if (extOp) {
      if (isa<arith::ExtSIOp>(extOp))
        result = rewriter.create<arith::ExtSIOp>(loc, op.getType(), result)
                     .getResult();
      else if (isa<arith::ExtUIOp>(extOp))
        result = rewriter.create<arith::ExtUIOp>(loc, op.getType(), result)
                     .getResult();
      else
        result = rewriter.create<arith::ExtFOp>(loc, op.getType(), result)
                     .getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

// MMA types have different layout based on how they are used in matmul ops.
// Figure the right layout to use by looking at op uses.
// TODO: Change the GPU dialect to abstract the layout at the this level and
// only care about it during lowering to NVVM.
static const char *inferFragType(Operation *op) {
  for (Operation *users : op->getUsers()) {
    auto contract = dyn_cast<vector::ContractionOp>(users);
    if (!contract)
      continue;
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

  std::optional<int64_t> stride =
      getStaticallyKnownRowStride(op.getShapedType());
  if (!stride.has_value()) {
    LLVM_DEBUG(DBGS() << "no stride\n");
    return rewriter.notifyMatchFailure(op, "no stride");
  }

  AffineMap map = op.getPermutationMap();
  bool isTranspose = isTransposeMatrixLoadMap(map);

  // Handle broadcast by setting the stride to 0.
  if (auto cstExpr = dyn_cast<AffineConstantExpr>(map.getResult(isTranspose))) {
    assert(cstExpr.getValue() == 0);
    stride = 0;
  }

  Value mappingResult = op.getResult();
  auto elType = op.getVectorType().getElementType();
  const char *fragType = inferFragType(op);
  if (op->hasOneUse()) {
    auto *user = *op->user_begin();
    // Infer the signedness of the mma type from the integer extend.
    bool isSignedExtend = isa<arith::ExtSIOp>(user);
    if (isSignedExtend || isa<arith::ExtUIOp>(user)) {
      elType = IntegerType::get(
          op.getContext(), cast<IntegerType>(elType).getWidth(),
          isSignedExtend ? IntegerType::Signed : IntegerType::Unsigned);
      mappingResult = user->getResult(0);
      fragType = inferFragType(user);
    }
  }
  gpu::MMAMatrixType type =
      gpu::MMAMatrixType::get(op.getVectorType().getShape(), elType, fragType);
  Value load = rewriter.create<gpu::SubgroupMmaLoadMatrixOp>(
      op.getLoc(), type, op.getSource(), op.getIndices(),
      rewriter.getIndexAttr(*stride),
      isTranspose ? rewriter.getUnitAttr() : UnitAttr());
  valueMapping[mappingResult] = load;

  LLVM_DEBUG(DBGS() << "transfer read to: " << load << "\n");
  return success();
}

static LogicalResult
convertTransferWriteOp(RewriterBase &rewriter, vector::TransferWriteOp op,
                       llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  assert(transferWriteSupportsMMAMatrixType(op));
  std::optional<int64_t> stride =
      getStaticallyKnownRowStride(op.getShapedType());
  if (!stride.has_value()) {
    LLVM_DEBUG(DBGS() << "no stride\n");
    return rewriter.notifyMatchFailure(op, "no stride");
  }

  auto it = valueMapping.find(op.getVector());
  if (it == valueMapping.end()) {
    LLVM_DEBUG(DBGS() << "no mapping\n");
    return rewriter.notifyMatchFailure(op, "no mapping");
  }

  Value matrix = it->second;
  auto store = rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(
      op.getLoc(), matrix, op.getSource(), op.getIndices(),
      rewriter.getIndexAttr(*stride), /*transpose=*/UnitAttr());
  (void)store;

  LLVM_DEBUG(DBGS() << "transfer write to: " << store << "\n");

  LLVM_DEBUG(DBGS() << "erase: " << op << "\n");
  rewriter.eraseOp(op);
  return success();
}

/// Returns the vector type which represents a matrix fragment.
static VectorType
getMmaSyncVectorOperandType(const nvgpu::FragmentElementInfo &regInfo) {
  SmallVector<int64_t> shape{regInfo.numRegistersPerFragment,
                             regInfo.elementsPerRegister};
  Type elType = regInfo.registerLLVMType;
  if (auto vecType = dyn_cast<VectorType>(elType))
    elType = vecType.getElementType();
  return VectorType::get(shape, elType);
}

/// Convert a 2D splat ConstantOp to a SubgroupMmaConstantMatrix op.
static LogicalResult
convertConstantOpMmaSync(RewriterBase &rewriter, arith::ConstantOp op,
                         llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo)) {
    LLVM_DEBUG(DBGS() << "no warpMatrixInfo\n");
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");
  }

  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo)) {
    LLVM_DEBUG(DBGS() << "not mma sync reg info\n");
    return rewriter.notifyMatchFailure(op, "not mma sync reg info");
  }

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);
  auto dense = dyn_cast<SplatElementsAttr>(op.getValue());
  if (!dense) {
    LLVM_DEBUG(DBGS() << "not a splat\n");
    return rewriter.notifyMatchFailure(op, "not a splat");
  }

  Value result = rewriter.create<arith::ConstantOp>(
      op.getLoc(), vectorType,
      DenseElementsAttr::get(vectorType, dense.getSplatValue<Attribute>()));
  valueMapping[op.getResult()] = result;
  return success();
}

/// Check if the loaded matrix operand requires transposed.
/// Transposed Map Example:
/// Example 1   : (..., d0, d1) -> (d1 * 1, d0 * 2)
/// Example 2   : (d0, d1, d2, d3) -> (d3, d2)
/// The code below checks if the output 2D is transposed using a generalized
/// version     : (d0, d1, dn, ..., dm, ...) -> (dm, dn)
/// Returns     : true; if m > n, false o.w.
static FailureOr<bool> isTransposed(vector::TransferReadOp op) {
  mlir::AffineMap map = op.getPermutationMap();

  if (map.getNumResults() != 2) {
    LLVM_DEBUG(DBGS() << "Failed because the result of `vector.transfer_read` "
                         "is not a 2d operand\n");
    return failure();
  }

  // Output 2D matrix dimensions in the order of d0, d1.
  mlir::AffineExpr dM = map.getResult(0);
  mlir::AffineExpr dN = map.getResult(1);

  //  Find the position of these expressions in the input.
  auto exprM = dyn_cast<AffineDimExpr>(dM);
  auto exprN = dyn_cast<AffineDimExpr>(dN);

  if (!exprM || !exprN) {
    LLVM_DEBUG(DBGS() << "Failed because expressions are not affine dim "
                         "expressions, then transpose cannot be determined.\n");
    return failure();
  }

  return exprM.getPosition() > exprN.getPosition();
}

static LogicalResult
creatLdMatrixCompatibleLoads(RewriterBase &rewriter, vector::TransferReadOp op,
                             llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  Location loc = op->getLoc();

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo)) {
    LLVM_DEBUG(DBGS() << "no warpMatrixInfo\n");
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");
  }

  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo)) {
    LLVM_DEBUG(DBGS() << "not mma sync reg info\n");
    return rewriter.notifyMatchFailure(op, "not mma sync reg info");
  }

  FailureOr<bool> transpose = isTransposed(op);
  if (failed(transpose)) {
    LLVM_DEBUG(DBGS() << "failed to determine the transpose\n");
    return rewriter.notifyMatchFailure(
        op, "Op should likely not be converted to a nvgpu.ldmatrix call.");
  }

  FailureOr<nvgpu::LdMatrixParams> params =
      nvgpu::getLdMatrixParams(*warpMatrixInfo, *transpose);

  if (failed(params)) {
    LLVM_DEBUG(
        DBGS()
        << "failed to convert vector.transfer_read to ldmatrix. "
        << "Op should likely not be converted to a nvgpu.ldmatrix call.\n");
    return rewriter.notifyMatchFailure(
        op, "failed to convert vector.transfer_read to ldmatrix; this op "
            "likely should not be converted to a nvgpu.ldmatrix call.");
  }

  // Adjust the load offset.
  auto laneId = rewriter.create<gpu::LaneIdOp>(loc);
  FailureOr<AffineMap> offsets =
      nvgpu::getLaneIdToLdMatrixMatrixCoord(rewriter, loc, *params);
  if (failed(offsets)) {
    LLVM_DEBUG(DBGS() << "no offsets\n");
    return rewriter.notifyMatchFailure(op, "no offsets");
  }

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);

  SmallVector<Value, 4> indices;
  getXferIndices<vector::TransferReadOp>(rewriter, op, *offsets, {laneId},
                                         indices);

  nvgpu::LdMatrixOp newOp = rewriter.create<nvgpu::LdMatrixOp>(
      loc, vectorType, op.getSource(), indices, *transpose, params->numTiles);
  valueMapping[op] = newOp->getResult(0);
  return success();
}

static LogicalResult
createNonLdMatrixLoads(RewriterBase &rewriter, vector::TransferReadOp op,
                       llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Location loc = op.getLoc();
  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");
  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo)) {
    return rewriter.notifyMatchFailure(
        op, "Failed to deduce register fragment type during "
            "conversion to distributed non-ldmatrix compatible load");
  }

  Value laneId = rewriter.create<gpu::LaneIdOp>(loc);
  SmallVector<Value, 4> elements;

  // This is the individual element type.
  Type loadedElType = regInfo->registerLLVMType;
  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);

  Value fill = rewriter.create<arith::ConstantOp>(
      op.getLoc(), vectorType.getElementType(),
      rewriter.getZeroAttr(vectorType.getElementType()));
  Value result =
      rewriter.create<vector::SplatOp>(op.getLoc(), fill, vectorType);

  bool isTransposeLoad = !op.getPermutationMap().isMinorIdentity();

  // If we are not transposing, then we can use vectorized loads. Otherwise, we
  // must load each element individually.
  if (!isTransposeLoad) {
    if (!isa<VectorType>(loadedElType)) {
      loadedElType = VectorType::get({1}, loadedElType);
    }

    for (int i = 0; i < vectorType.getShape()[0]; i++) {
      FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
          rewriter, op.getLoc(), *warpMatrixInfo);
      if (failed(coords))
        return rewriter.notifyMatchFailure(op, "no coords");

      Value logicalValueId = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIndexAttr(i * regInfo->elementsPerRegister));
      SmallVector<Value, 4> newIndices;
      getXferIndices<vector::TransferReadOp>(
          rewriter, op, *coords, {laneId, logicalValueId}, newIndices);

      Value el = rewriter.create<vector::LoadOp>(loc, loadedElType,
                                                 op.getSource(), newIndices);
      result = rewriter.create<vector::InsertOp>(loc, el, result, i);
    }
  } else {
    if (auto vecType = dyn_cast<VectorType>(loadedElType)) {
      loadedElType = vecType.getElementType();
    }
    for (int i = 0; i < vectorType.getShape()[0]; i++) {
      for (unsigned innerIdx = 0; innerIdx < vectorType.getShape()[1];
           innerIdx++) {

        Value logicalValueId = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexType(),
            rewriter.getIndexAttr(i * regInfo->elementsPerRegister + innerIdx));
        FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
            rewriter, op.getLoc(), *warpMatrixInfo);
        if (failed(coords))
          return rewriter.notifyMatchFailure(op, "no coords");

        SmallVector<Value, 4> newIndices;
        getXferIndices<vector::TransferReadOp>(
            rewriter, op, *coords, {laneId, logicalValueId}, newIndices);
        Value el = rewriter.create<memref::LoadOp>(op.getLoc(), loadedElType,
                                                   op.getSource(), newIndices);
        result = rewriter.create<vector::InsertOp>(
            op.getLoc(), el, result, ArrayRef<int64_t>{i, innerIdx});
      }
    }
  }

  valueMapping[op.getResult()] = result;
  return success();
}

/// Return true if this is a shared memory memref type.
static bool isSharedMemory(MemRefType type) {
  auto addressSpace =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace());
  return addressSpace &&
         addressSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Converts a `vector.transfer_read` operation directly to either a
/// `vector.load` or a `nvgpu.ldmatrix` operation. This function should only be
/// used when converting to `nvgpu.mma.sync` operations.
static LogicalResult
convertTransferReadToLoads(RewriterBase &rewriter, vector::TransferReadOp op,
                           llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");

  bool isLdMatrixCompatible =
      isSharedMemory(cast<MemRefType>(op.getSource().getType())) &&
      nvgpu::inferTileWidthInBits(*warpMatrixInfo) == 128;

  VectorType vecTy = op.getVectorType();
  int64_t bitWidth = vecTy.getElementType().getIntOrFloatBitWidth();

  // When we are transposing the B operand, ldmatrix will only work if we have
  // at least 8 rows to read and the width to read for the transpose is 128
  // bits.
  if (!op.getPermutationMap().isMinorIdentity() &&
      (bitWidth != 16 || vecTy.getDimSize(1) < 8 ||
       vecTy.getDimSize(0) * bitWidth < 128))
    isLdMatrixCompatible = false;

  if (!isLdMatrixCompatible)
    return createNonLdMatrixLoads(rewriter, op, valueMapping);

  return creatLdMatrixCompatibleLoads(rewriter, op, valueMapping);
}

static LogicalResult
convertTransferWriteToStores(RewriterBase &rewriter, vector::TransferWriteOp op,
                             llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Location loc = op->getLoc();
  auto it = valueMapping.find(op.getVector());
  if (it == valueMapping.end())
    return rewriter.notifyMatchFailure(op, "no mapping");
  Value matrix = it->second;

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");
  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo))
    return rewriter.notifyMatchFailure(op, "not mma sync reg info");

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc);

  for (unsigned i = 0; i < vectorType.getShape()[0]; i++) {
    Value logicalValueId = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIndexAttr(i * regInfo->elementsPerRegister));
    FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
        rewriter, op.getLoc(), *warpMatrixInfo);
    if (failed(coords))
      return rewriter.notifyMatchFailure(op, "no coords");

    Value el =
        rewriter.create<vector::ExtractOp>(loc, matrix, ArrayRef<int64_t>{i});
    SmallVector<Value, 4> newIndices;
    getXferIndices<vector::TransferWriteOp>(
        rewriter, op, *coords, {laneId, logicalValueId}, newIndices);
    rewriter.create<vector::StoreOp>(loc, el, op.getSource(), newIndices);
  }

  LLVM_DEBUG(DBGS() << "erase: " << op << "\n");
  rewriter.eraseOp(op);
  return success();
}

static void populateFromInt64AttrArray(ArrayAttr arrayAttr,
                                       SmallVectorImpl<int64_t> &results) {
  for (auto attr : arrayAttr)
    results.push_back(cast<IntegerAttr>(attr).getInt());
}

static LogicalResult
convertExtractStridedSlice(RewriterBase &rewriter,
                           vector::ExtractStridedSliceOp op,
                           llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Location loc = op->getLoc();

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");

  FailureOr<nvgpu::FragmentElementInfo> mmaSyncFragmentInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(mmaSyncFragmentInfo))
    return rewriter.notifyMatchFailure(op, "no mmaSyncFragmentInfo");

  // Find the vector.transer_read whose result vector is being sliced.
  auto transferReadOp = op.getVector().getDefiningOp<vector::TransferReadOp>();
  if (!transferReadOp)
    return rewriter.notifyMatchFailure(op, "no transfer read");

  warpMatrixInfo = nvgpu::getWarpMatrixInfo(transferReadOp);
  if (failed(warpMatrixInfo))
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");

  FailureOr<nvgpu::FragmentElementInfo> ldFragmentInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(ldFragmentInfo))
    return rewriter.notifyMatchFailure(op, "no ldFragmentInfo");

  assert(
      (mmaSyncFragmentInfo->elementsPerRegister ==
       ldFragmentInfo->elementsPerRegister) &&
      "Number of elements per register should be same for load and mma.sync");

  // Create vector.extract_strided_slice op for thread-owned fragments.
  std::array<int64_t, 2> strides = {1,
                                    1}; // stride for extract slice is always 1.
  std::array<int64_t, 2> sliceShape = {
      mmaSyncFragmentInfo->numRegistersPerFragment,
      mmaSyncFragmentInfo->elementsPerRegister};
  auto it = valueMapping.find(transferReadOp);
  if (it == valueMapping.end())
    return rewriter.notifyMatchFailure(op, "no mapping");
  auto sourceVector = it->second;

  // offset and sizes at warp-level of onwership.
  SmallVector<int64_t> offsets;
  populateFromInt64AttrArray(op.getOffsets(), offsets);

  SmallVector<int64_t> sizes;
  populateFromInt64AttrArray(op.getSizes(), sizes);
  ArrayRef<int64_t> warpVectorShape = op.getSourceVectorType().getShape();

  // Compute offset in vector registers. Note that the mma.sync vector registers
  // are shaped as numberOfFragments x numberOfRegistersPerfFragment. The vector
  // registers can only be sliced along numberOfFragments, i.e., sliceOffset[0].
  std::array<int64_t, 2> sliceOffset = {0, 0};

  if (offsets[0] && offsets[1])
    return op->emitError() << "Slicing fragments in 2D is not supported. ";
  if (offsets[0])
    sliceOffset[0] = (warpVectorShape[0] / offsets[0]);
  else if (offsets[1])
    sliceOffset[0] = (warpVectorShape[1] / offsets[1]);

  Value newOp = rewriter.create<vector::ExtractStridedSliceOp>(
      loc, sourceVector, sliceOffset, sliceShape, strides);

  valueMapping[op] = newOp;
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
  Value matmul = rewriter.create<gpu::SubgroupMmaComputeOp>(
      op.getLoc(), opC.getType(), opA, opB, opC, /*a_transpose=*/UnitAttr(),
      /*b_transpose=*/UnitAttr());
  valueMapping[op.getResult()] = matmul;
  return success();
}

static LogicalResult
convertContractOpToMmaSync(RewriterBase &rewriter, vector::ContractionOp op,
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
  int64_t m = cast<VectorType>(op.getLhs().getType()).getShape()[0];
  int64_t n = cast<VectorType>(op.getRhs().getType()).getShape()[0];
  int64_t k = cast<VectorType>(op.getLhs().getType()).getShape()[1];
  Value matmul = rewriter.create<nvgpu::MmaSyncOp>(
      op.getLoc(), opA, opB, opC, rewriter.getI64ArrayAttr({m, n, k}));
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
      rewriter.create<arith::ConstantOp>(op.getLoc(), splat.getType(), splat);
  const char *fragType = inferFragType(op);
  auto vecType = cast<VectorType>(op.getType());
  gpu::MMAMatrixType type = gpu::MMAMatrixType::get(
      vecType.getShape(), vecType.getElementType(), llvm::StringRef(fragType));
  auto matrix = rewriter.create<gpu::SubgroupMmaConstantMatrixOp>(
      op.getLoc(), type, scalarConstant);
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
  auto matrix = rewriter.create<gpu::SubgroupMmaConstantMatrixOp>(
      op.getLoc(), type, op.getSource());
  valueMapping[op.getResult()] = matrix;
  return success();
}

// Replace ForOp with a new ForOp with extra operands. The YieldOp is not
// updated and needs to be updated separately for the loop to be correct.
static scf::ForOp replaceForOpWithNewSignature(RewriterBase &rewriter,
                                               scf::ForOp loop,
                                               ValueRange newInitArgs) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  // Create a new loop before the existing one, with the extra operands.
  rewriter.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getInitArgs());
  llvm::append_range(operands, newInitArgs);
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      operands);
  rewriter.eraseBlock(newLoop.getBody());

  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());
  for (Value operand : newInitArgs)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    rewriter.replaceAllUsesWith(std::get<0>(it), std::get<1>(it));

  LLVM_DEBUG(DBGS() << "newLoop now: " << newLoop << "\n");
  LLVM_DEBUG(DBGS() << "stripped scf.for: " << loop << "\n");
  LLVM_DEBUG(DBGS() << "erase: " << loop);

  rewriter.eraseOp(loop);
  return newLoop;
}

static LogicalResult convertForOp(RewriterBase &rewriter, scf::ForOp op,
                                  llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  SmallVector<Value> newOperands;
  SmallVector<std::pair<size_t, size_t>> argMapping;
  for (const auto &operand : llvm::enumerate(op.getInitArgs())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end()) {
      LLVM_DEBUG(DBGS() << "no value mapping for: " << operand.value() << "\n");
      continue;
    }
    argMapping.push_back(std::make_pair(
        operand.index(), op.getInitArgs().size() + newOperands.size()));
    newOperands.push_back(it->second);
  }

  scf::ForOp newForOp = replaceForOpWithNewSignature(rewriter, op, newOperands);
  Block &loopBody = *newForOp.getBody();
  for (auto mapping : argMapping) {
    valueMapping[newForOp.getResult(mapping.first)] =
        newForOp.getResult(mapping.second);
    valueMapping[loopBody.getArgument(mapping.first +
                                      newForOp.getNumInductionVars())] =
        loopBody.getArgument(mapping.second + newForOp.getNumInductionVars());
  }

  LLVM_DEBUG(DBGS() << "scf.for to: " << newForOp << "\n");
  return success();
}

static LogicalResult
convertYieldOp(RewriterBase &rewriter, scf::YieldOp op,
               llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto loop = cast<scf::ForOp>(op->getParentOp());
  auto yieldOperands = llvm::to_vector<4>(op.getOperands());
  for (const auto &operand : llvm::enumerate(op.getOperands())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end())
      continue;
    // Replace the yield of old value with the for op argument to make it easier
    // to remove the dead code.
    yieldOperands[operand.index()] = loop.getInitArgs()[operand.index()];
    yieldOperands.push_back(it->second);
  }
  rewriter.create<scf::YieldOp>(op.getLoc(), yieldOperands);

  LLVM_DEBUG(DBGS() << "erase: " << op << "\n");
  rewriter.eraseOp(op);
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
  auto resultType = matrixOperands[0].getType().cast<gpu::MMAMatrixType>();
  if (opType == gpu::MMAElementwiseOp::EXTF) {
    // The floating point extension case has a different result type.
    auto vectorType = op->getResultTypes()[0].cast<VectorType>();
    resultType = gpu::MMAMatrixType::get(resultType.getShape(),
                                         vectorType.getElementType(),
                                         resultType.getOperand());
  }

  Value newOp = rewriter.create<gpu::SubgroupMmaElementwiseOp>(
      op->getLoc(), resultType, matrixOperands, opType);
  valueMapping[op->getResult(0)] = newOp;
  return success();
}

void mlir::populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns,
                                              bool useNvGpu) {
  if (!useNvGpu) {
    patterns.add<PrepareContractToGPUMMA, CombineTransferReadOpTranspose>(
        patterns.getContext());
    return;
  }
  vector::populateVectorContractCanonicalizeMatmulToMMT(patterns);
  patterns.add<CombineTransferReadOpTranspose>(patterns.getContext());
}

LogicalResult mlir::convertVectorToMMAOps(RewriterBase &rewriter,
                                          Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp, /*useNvGpu=*/false);
  llvm::DenseMap<Value, Value> valueMapping;

  auto globalRes = LogicalResult::success();
  for (Operation *op : ops) {
    LLVM_DEBUG(DBGS() << "Process op: " << *op << "\n");
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
      res = convertForOp(rewriter, forOp, valueMapping);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      res = convertYieldOp(rewriter, yieldOp, valueMapping);
    } else if (auto elementwiseType = convertElementwiseOpToMMA(op)) {
      res = convertElementwiseOp(rewriter, op, *elementwiseType, valueMapping);
    }
    if (failed(res))
      globalRes = failure();
  }
  return globalRes;
}

LogicalResult mlir::convertVectorToNVVMCompatibleMMASync(RewriterBase &rewriter,
                                                         Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp, /*useNvGpu=*/true);
  llvm::DenseMap<Value, Value> valueMapping;
  for (Operation *op : ops) {
    if (llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](vector::TransferReadOp transferReadOp) {
              return convertTransferReadToLoads(rewriter, transferReadOp,
                                                valueMapping);
            })
            .Case([&](vector::TransferWriteOp transferWriteOp) {
              return convertTransferWriteToStores(rewriter, transferWriteOp,
                                                  valueMapping);
            })
            .Case([&](vector::ExtractStridedSliceOp extractStridedSliceOp) {
              return convertExtractStridedSlice(rewriter, extractStridedSliceOp,
                                                valueMapping);
            })
            .Case([&](vector::ContractionOp contractionOp) {
              return convertContractOpToMmaSync(rewriter, contractionOp,
                                                valueMapping);
            })
            .Case([&](scf::ForOp forOp) {
              return convertForOp(rewriter, forOp, valueMapping);
            })
            .Case([&](scf::YieldOp yieldOp) {
              return convertYieldOp(rewriter, yieldOp, valueMapping);
            })
            .Case([&](arith::ConstantOp constOp) {
              return convertConstantOpMmaSync(rewriter, constOp, valueMapping);
            })
            .Default([&](Operation *op) {
              return op->emitError() << "unhandled vector to mma type: " << *op;
            })
            .failed()) {
      return op->emitOpError()
             << "failed to convert op during vector-to-nvgpu conversion";
    }
  }
  return success();
}

namespace {

struct ConvertVectorToGPUPass
    : public impl::ConvertVectorToGPUBase<ConvertVectorToGPUPass> {

  explicit ConvertVectorToGPUPass(bool useNvGpu_) {
    useNvGpu.setValue(useNvGpu_);
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePrepareVectorToMMAPatterns(patterns, useNvGpu.getValue());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    if (useNvGpu) {
      if (failed(
              convertVectorToNVVMCompatibleMMASync(rewriter, getOperation())))
        return signalPassFailure();
      return;
    }
    (void)convertVectorToMMAOps(rewriter, getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertVectorToGPUPass(bool useNvGpu) {
  return std::make_unique<ConvertVectorToGPUPass>(useNvGpu);
}
