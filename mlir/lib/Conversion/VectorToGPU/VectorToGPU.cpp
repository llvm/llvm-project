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

#include "NvGpuSupport.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

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
static void getXferIndices(OpBuilder &b, TransferOpType xferOp,
                           AffineMap offsetMap, ArrayRef<Value> dimValues,
                           SmallVector<Value, 4> &indices) {
  indices.append(xferOp.getIndices().begin(), xferOp.getIndices().end());
  Location loc = xferOp.getLoc();
  unsigned offsetsIdx = 0;
  for (auto expr : xferOp.getPermutationMap().getResults()) {
    if (auto dim = expr.template dyn_cast<AffineDimExpr>()) {
      Value prevIdx = indices[dim.getPosition()];
      SmallVector<Value, 3> dims(dimValues.begin(), dimValues.end());
      dims.push_back(prevIdx);
      AffineExpr d0 = b.getAffineDimExpr(offsetMap.getNumDims());
      indices[dim.getPosition()] = makeComposedAffineApply(
          b, loc, d0 + offsetMap.getResult(offsetsIdx++), dims);
      continue;
    }
  }
}

// Return true if the contract op can be convert to MMA matmul.
static bool contractSupportsMMAMatrixType(vector::ContractionOp contract,
                                          bool useNvGpu) {
  if (!contract.getMasks().empty())
    return false;

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
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

// Return the stide for the dimension 0 of |type| if it is a memref and has a
// constant stride.
static llvm::Optional<int64_t>
getMemrefConstantHorizontalStride(ShapedType type) {
  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType)
    return false;
  // If the memref is 0 or 1D the horizontal stride is 0.
  if (memrefType.getRank() < 2)
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
static bool transferReadSupportsMMAMatrixType(vector::TransferReadOp readOp,
                                              bool useNvGpu) {
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

  if (!useNvGpu) {
    // TODO: Support transpose once it is added to GPU dialect ops.
    // For now we only support (d0, d1) -> (d0, d1) and (d0, d1) -> (0, d1).
    return map.isMinorIdentity() || map == broadcastInnerDim;
  }

  return true;
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
  return convertElementwiseOpToMMA(op).has_value();
}

static bool supportsMMaMatrixType(Operation *op, bool useNvGpu) {
  if (isa<scf::ForOp, scf::YieldOp>(op))
    return true;
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op))
    return transferReadSupportsMMAMatrixType(transferRead, useNvGpu);
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteSupportsMMAMatrixType(transferWrite);
  if (auto contract = dyn_cast<vector::ContractionOp>(op))
    return contractSupportsMMAMatrixType(contract, useNvGpu);
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
static SetVector<Operation *> getOpToConvert(mlir::Operation *op,
                                             bool useNvGpu) {
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
    if (llvm::any_of(dependentOps, [useNvGpu](Operation *op) {
          return !supportsMMaMatrixType(op, useNvGpu);
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
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto iteratorTypes = op.getIteratorTypes().getValue();
    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (!(vector::isParallelIterator(iteratorTypes[0]) &&
          vector::isParallelIterator(iteratorTypes[1]) &&
          vector::isReductionIterator(iteratorTypes[2])))
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
  assert(transferReadSupportsMMAMatrixType(op, /*useNvGpu=*/false));
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

/// Returns the vector type which represents a matrix fragment.
static VectorType
getMmaSyncVectorOperandType(const nvgpu::FragmentElementInfo &regInfo) {
  SmallVector<int64_t> shape{regInfo.numRegistersPerFragment,
                             regInfo.elementsPerRegister};
  Type elType = regInfo.registerLLVMType;
  if (auto vecType = elType.dyn_cast<VectorType>())
    elType = vecType.getElementType();
  return VectorType::get(shape, elType);
}

/// Convert a 2D splat ConstantOp to a SubgroupMmaConstantMatrix op.
static LogicalResult
convertConstantOpMmaSync(arith::ConstantOp op,
                         llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return failure();

  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo))
    return failure();

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);
  auto dense = op.getValue().dyn_cast<SplatElementsAttr>();
  if (!dense)
    return failure();
  Value result = b.create<arith::ConstantOp>(
      op.getLoc(), vectorType,
      DenseElementsAttr::get(vectorType, dense.getSplatValue<Attribute>()));
  valueMapping[op.getResult()] = result;
  return success();
}

static LogicalResult
creatLdMatrixCompatibleLoads(vector::TransferReadOp op, OpBuilder &builder,
                             llvm::DenseMap<Value, Value> &valueMapping) {
  Location loc = op->getLoc();

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return failure();

  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo))
    return failure();

  FailureOr<nvgpu::LdMatrixParams> params = nvgpu::getLdMatrixParams(
      *warpMatrixInfo,
      /*transpose=*/!op.getPermutationMap().isMinorIdentity());
  if (failed(params)) {
    return op->emitError()
           << "failed to convert vector.transfer_read to ldmatrix; this op "
              "likely "
              "should not be converted to a nvgpu.ldmatrix call.";
  }

  // Adjust the load offset.
  auto laneId = builder.create<gpu::LaneIdOp>(loc);
  FailureOr<AffineMap> offsets =
      nvgpu::getLaneIdToLdMatrixMatrixCoord(loc, builder, *params);
  if (failed(offsets))
    return failure();

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);

  SmallVector<Value, 4> indices;
  getXferIndices<vector::TransferReadOp>(builder, op, *offsets, {laneId},
                                         indices);
  nvgpu::LdMatrixOp newOp = builder.create<nvgpu::LdMatrixOp>(
      loc, vectorType, op.getSource(), indices,
      !op.getPermutationMap().isMinorIdentity(), params->numTiles);
  valueMapping[op] = newOp->getResult(0);
  return success();
}

static LogicalResult
createNonLdMatrixLoads(vector::TransferReadOp op, OpBuilder &builder,
                       llvm::DenseMap<Value, Value> &valueMapping) {
  Location loc = op.getLoc();
  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return failure();
  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo)) {
    op->emitError() << "Failed to deduce register fragment type during "
                       "conversion to distributed non-ldmatrix compatible load";
    return failure();
  }

  Value laneId = builder.create<gpu::LaneIdOp>(loc);
  SmallVector<Value, 4> elements;

  // This is the individual element type.
  Type loadedElType = regInfo->registerLLVMType;
  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);

  Value fill = builder.create<arith::ConstantOp>(
      op.getLoc(), vectorType.getElementType(),
      builder.getZeroAttr(vectorType.getElementType()));
  Value result = builder.create<vector::SplatOp>(op.getLoc(), fill, vectorType);

  bool isTransposeLoad = !op.getPermutationMap().isMinorIdentity();

  // If we are not transposing, then we can use vectorized loads. Otherwise, we
  // must load each element individually.
  if (!isTransposeLoad) {
    if (!loadedElType.isa<VectorType>()) {
      loadedElType = VectorType::get({1}, loadedElType);
    }

    for (int i = 0; i < vectorType.getShape()[0]; i++) {
      FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
          op.getLoc(), builder, *warpMatrixInfo);
      if (failed(coords))
        return failure();
      Value logicalValueId = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(),
          builder.getIndexAttr(i * regInfo->elementsPerRegister));
      SmallVector<Value, 4> newIndices;
      getXferIndices<vector::TransferReadOp>(
          builder, op, *coords, {laneId, logicalValueId}, newIndices);

      Value el = builder.create<vector::LoadOp>(loc, loadedElType,
                                                op.getSource(), newIndices);
      result = builder.create<vector::InsertOp>(loc, el, result,
                                                builder.getI64ArrayAttr(i));
    }
  } else {
    if (auto vecType = loadedElType.dyn_cast<VectorType>()) {
      loadedElType = vecType.getElementType();
    }
    for (int i = 0; i < vectorType.getShape()[0]; i++) {
      for (unsigned innerIdx = 0; innerIdx < vectorType.getShape()[1];
           innerIdx++) {

        Value logicalValueId = builder.create<arith::ConstantOp>(
            loc, builder.getIndexType(),
            builder.getIndexAttr(i * regInfo->elementsPerRegister + innerIdx));
        FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
            op.getLoc(), builder, *warpMatrixInfo);
        if (failed(coords))
          return failure();

        SmallVector<Value, 4> newIndices;
        getXferIndices<vector::TransferReadOp>(
            builder, op, *coords, {laneId, logicalValueId}, newIndices);
        Value el = builder.create<memref::LoadOp>(op.getLoc(), loadedElType,
                                                  op.getSource(), newIndices);
        result = builder.create<vector::InsertOp>(
            op.getLoc(), el, result, builder.getI64ArrayAttr({i, innerIdx}));
      }
    }
  }

  valueMapping[op.getResult()] = result;
  return success();
}

/// Converts a `vector.transfer_read` operation directly to either a
/// `vector.load` or a `nvgpu.ldmatrix` operation. This function should only be
/// used when converting to `nvgpu.mma.sync` operations.
static LogicalResult
convertTransferReadToLoads(vector::TransferReadOp op,
                           llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return failure();

  bool isLdMatrixCompatible =
      op.getSource().getType().cast<MemRefType>().getMemorySpaceAsInt() == 3 &&
      nvgpu::inferTileWidthInBits(*warpMatrixInfo) == 128;

  VectorType vecTy = op.getVectorType();
  int64_t bitWidth = vecTy.getElementType().getIntOrFloatBitWidth();

  // When we are transposing the B operand, ldmatrix will only work if we have
  // at least 8 rows to read and  the width to read for the transpose is 128
  // bits.
  if (!op.getPermutationMap().isMinorIdentity() &&
      (bitWidth != 16 || vecTy.getDimSize(1) < 8 ||
       vecTy.getDimSize(0) * bitWidth < 128))
    isLdMatrixCompatible = false;

  if (!isLdMatrixCompatible)
    return createNonLdMatrixLoads(op, b, valueMapping);

  return creatLdMatrixCompatibleLoads(op, b, valueMapping);
}

static LogicalResult
convertTransferWriteToStores(vector::TransferWriteOp op,
                             llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Value matrix = valueMapping.find(op.getVector())->second;

  FailureOr<nvgpu::WarpMatrixInfo> warpMatrixInfo =
      nvgpu::getWarpMatrixInfo(op);
  if (failed(warpMatrixInfo))
    return failure();
  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo))
    return failure();

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);
  Value laneId = b.create<gpu::LaneIdOp>(loc);

  for (unsigned i = 0; i < vectorType.getShape()[0]; i++) {
    Value logicalValueId = b.create<arith::ConstantOp>(
        loc, b.getIndexType(),
        b.getIndexAttr(i * regInfo->elementsPerRegister));
    FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
        op.getLoc(), b, *warpMatrixInfo);
    if (failed(coords))
      return failure();

    Value el = b.create<vector::ExtractOp>(loc, matrix, ArrayRef<int64_t>{i});
    SmallVector<Value, 4> newIndices;
    getXferIndices<vector::TransferWriteOp>(
        b, op, *coords, {laneId, logicalValueId}, newIndices);
    b.create<vector::StoreOp>(loc, el, op.getSource(), newIndices);
  }
  op->erase();
  return success();
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

static LogicalResult
convertContractOpToMmaSync(vector::ContractionOp op,
                           llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder b(op);
  Value opA = valueMapping.find(op.getLhs())->second;
  Value opB = valueMapping.find(op.getRhs())->second;
  Value opC = valueMapping.find(op.getAcc())->second;
  int64_t m = op.getLhs().getType().cast<VectorType>().getShape()[0];
  int64_t n = op.getRhs().getType().cast<VectorType>().getShape()[0];
  int64_t k = op.getLhs().getType().cast<VectorType>().getShape()[1];
  Value matmul = b.create<nvgpu::MmaSyncOp>(op.getLoc(), opA, opB, opC,
                                            b.getI64ArrayAttr({m, n, k}));
  valueMapping[op.getResult()] = matmul;
  return success();
}

/// Convert a 2D splat ConstantOp to a SubgroupMmaConstantMatrix op.
static void convertConstantOp(arith::ConstantOp op,
                              llvm::DenseMap<Value, Value> &valueMapping) {
  assert(constantSupportsMMAMatrixType(op));
  OpBuilder b(op);
  auto splat =
      op.getValue().cast<SplatElementsAttr>().getSplatValue<TypedAttr>();
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

void mlir::populatePrepareVectorToMMAPatterns(RewritePatternSet &patterns,
                                              bool useNvGpu) {
  if (!useNvGpu) {
    patterns.add<PrepareContractToGPUMMA, CombineTransferReadOpTranspose>(
        patterns.getContext());
    return;
  }
  patterns
      .add<nvgpu::PrepareContractToGPUMMASync, CombineTransferReadOpTranspose>(
          patterns.getContext());
}

void mlir::convertVectorToMMAOps(Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp, /*useNvGpu=*/false);
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

LogicalResult mlir::convertVectorToNVVMCompatibleMMASync(Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp, /*useNvGpu=*/true);
  llvm::DenseMap<Value, Value> valueMapping;
  for (Operation *op : ops) {
    if (llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](vector::TransferReadOp transferReadOp) {
              return convertTransferReadToLoads(transferReadOp, valueMapping);
            })
            .Case([&](vector::TransferWriteOp transferWriteOp) {
              return convertTransferWriteToStores(transferWriteOp,
                                                  valueMapping);
            })
            .Case([&](vector::ContractionOp contractionOp) {
              return convertContractOpToMmaSync(contractionOp, valueMapping);
            })
            .Case([&](scf::ForOp forOp) {
              convertForOp(forOp, valueMapping);
              return success();
            })
            .Case([&](scf::YieldOp yieldOp) {
              convertYieldOp(yieldOp, valueMapping);
              return success();
            })
            .Case([&](arith::ConstantOp constOp) {
              return convertConstantOpMmaSync(constOp, valueMapping);
            })
            .Default([&](Operation *op) {
              op->emitError() << "unhandled vector to mma type: " << *op;
              return failure();
            })
            .failed()) {
      op->emitError() << "Failed to convert op " << *op;
      return failure();
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

    if (useNvGpu.getValue()) {
      if (failed(convertVectorToNVVMCompatibleMMASync(getOperation())))
        return signalPassFailure();
    }

    (void)convertVectorToMMAOps(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertVectorToGPUPass(bool useNvGpu) {
  return std::make_unique<ConvertVectorToGPUPass>(useNvGpu);
}
