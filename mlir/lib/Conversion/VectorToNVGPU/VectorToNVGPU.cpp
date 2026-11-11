//===- VectorToNVGPU.cpp - Convert vector to NVGPU dialect ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector operations to NVGPU dialect ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToNVGPU/VectorToNVGPU.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/VectorToGPUCommon/VectorToGPUUtils.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "vector-to-nvgpu"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTONVGPUPASS
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
      SmallVector<OpFoldResult, 3> dims(dimValues);
      dims.push_back(prevIdx);
      AffineExpr d0 = rewriter.getAffineDimExpr(offsetMap.getNumDims());
      indices[dim.getPosition()] = affine::makeComposedAffineApply(
          rewriter, loc, d0 + offsetMap.getResult(offsetsIdx++), dims);
      continue;
    }
  }
}

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
  if (contract.getIndexingMapsArray() != infer({{m, k}, {n, k}, {m, n}}))
    return false;

  return true;
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
    return nvgpu::canLowerToWarpMatrixOperation(transferRead);
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return nvgpu::canLowerToWarpMatrixOperation(transferWrite);
  if (auto extractStridedSlice = dyn_cast<vector::ExtractStridedSliceOp>(op))
    return extractStridedSliceSupportsMMAMatrixType(extractStridedSlice);
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
    LDBG() << "no warpMatrixInfo";
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");
  }

  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo)) {
    LDBG() << "not mma sync reg info";
    return rewriter.notifyMatchFailure(op, "not mma sync reg info");
  }

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);
  auto dense = dyn_cast<SplatElementsAttr>(op.getValue());
  if (!dense) {
    LDBG() << "not a splat";
    return rewriter.notifyMatchFailure(op, "not a splat");
  }

  Value result = arith::ConstantOp::create(
      rewriter, op.getLoc(), vectorType,
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
    LDBG() << "Failed because the result of `vector.transfer_read` "
              "is not a 2d operand";
    return failure();
  }

  // Output 2D matrix dimensions in the order of d0, d1.
  mlir::AffineExpr dM = map.getResult(0);
  mlir::AffineExpr dN = map.getResult(1);

  //  Find the position of these expressions in the input.
  auto exprM = dyn_cast<AffineDimExpr>(dM);
  auto exprN = dyn_cast<AffineDimExpr>(dN);

  if (!exprM || !exprN) {
    LDBG() << "Failed because expressions are not affine dim "
              "expressions, then transpose cannot be determined.";
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
    LDBG() << "no warpMatrixInfo";
    return rewriter.notifyMatchFailure(op, "no warpMatrixInfo");
  }

  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      nvgpu::getMmaSyncRegisterType(*warpMatrixInfo);
  if (failed(regInfo)) {
    LDBG() << "not mma sync reg info";
    return rewriter.notifyMatchFailure(op, "not mma sync reg info");
  }

  FailureOr<bool> transpose = isTransposed(op);
  if (failed(transpose)) {
    LDBG() << "failed to determine the transpose";
    return rewriter.notifyMatchFailure(
        op, "Op should likely not be converted to a nvgpu.ldmatrix call.");
  }

  FailureOr<nvgpu::LdMatrixParams> params =
      nvgpu::getLdMatrixParams(*warpMatrixInfo, *transpose);

  if (failed(params)) {
    LDBG() << "failed to convert vector.transfer_read to ldmatrix. "
           << "Op should likely not be converted to a nvgpu.ldmatrix call.";
    return rewriter.notifyMatchFailure(
        op, "failed to convert vector.transfer_read to ldmatrix; this op "
            "likely should not be converted to a nvgpu.ldmatrix call.");
  }

  // Adjust the load offset.
  auto laneId = gpu::LaneIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
  FailureOr<AffineMap> offsets =
      nvgpu::getLaneIdToLdMatrixMatrixCoord(rewriter, loc, *params);
  if (failed(offsets)) {
    LDBG() << "no offsets";
    return rewriter.notifyMatchFailure(op, "no offsets");
  }

  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);

  SmallVector<Value, 4> indices;
  getXferIndices<vector::TransferReadOp>(rewriter, op, *offsets, {laneId},
                                         indices);

  nvgpu::LdMatrixOp newOp =
      nvgpu::LdMatrixOp::create(rewriter, loc, vectorType, op.getBase(),
                                indices, *transpose, params->numTiles);
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

  Value laneId = gpu::LaneIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);

  // This is the individual element type.
  Type loadedElType = regInfo->registerLLVMType;
  VectorType vectorType = getMmaSyncVectorOperandType(*regInfo);

  Value fill = arith::ConstantOp::create(
      rewriter, op.getLoc(), vectorType.getElementType(),
      rewriter.getZeroAttr(vectorType.getElementType()));
  Value result =
      vector::BroadcastOp::create(rewriter, op.getLoc(), vectorType, fill);

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

      Value logicalValueId = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(),
          rewriter.getIndexAttr(i * regInfo->elementsPerRegister));
      SmallVector<Value, 4> newIndices;
      getXferIndices<vector::TransferReadOp>(
          rewriter, op, *coords, {laneId, logicalValueId}, newIndices);

      Value el = vector::LoadOp::create(rewriter, loc, loadedElType,
                                        op.getBase(), newIndices);
      result = vector::InsertOp::create(rewriter, loc, el, result, i);
    }
  } else {
    if (auto vecType = dyn_cast<VectorType>(loadedElType)) {
      loadedElType = vecType.getElementType();
    }
    for (int i = 0; i < vectorType.getShape()[0]; i++) {
      for (unsigned innerIdx = 0; innerIdx < vectorType.getShape()[1];
           innerIdx++) {

        Value logicalValueId = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIndexType(),
            rewriter.getIndexAttr(i * regInfo->elementsPerRegister + innerIdx));
        FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
            rewriter, op.getLoc(), *warpMatrixInfo);
        if (failed(coords))
          return rewriter.notifyMatchFailure(op, "no coords");

        SmallVector<Value, 4> newIndices;
        getXferIndices<vector::TransferReadOp>(
            rewriter, op, *coords, {laneId, logicalValueId}, newIndices);
        Value el = memref::LoadOp::create(rewriter, op.getLoc(), loadedElType,
                                          op.getBase(), newIndices);
        result = vector::InsertOp::create(rewriter, op.getLoc(), el, result,
                                          ArrayRef<int64_t>{i, innerIdx});
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
      isSharedMemory(cast<MemRefType>(op.getBase().getType())) &&
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
  Value laneId = gpu::LaneIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);

  for (unsigned i = 0; i < vectorType.getShape()[0]; i++) {
    Value logicalValueId = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexType(),
        rewriter.getIndexAttr(i * regInfo->elementsPerRegister));
    FailureOr<AffineMap> coords = nvgpu::getLaneIdAndValueIdToOperandCoord(
        rewriter, op.getLoc(), *warpMatrixInfo);
    if (failed(coords))
      return rewriter.notifyMatchFailure(op, "no coords");

    Value el =
        vector::ExtractOp::create(rewriter, loc, matrix, ArrayRef<int64_t>{i});
    SmallVector<Value, 4> newIndices;
    getXferIndices<vector::TransferWriteOp>(
        rewriter, op, *coords, {laneId, logicalValueId}, newIndices);
    vector::StoreOp::create(rewriter, loc, el, op.getBase(), newIndices);
  }

  LDBG() << "erase: " << op;
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
  auto transferReadOp = op.getSource().getDefiningOp<vector::TransferReadOp>();
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

  Value newOp = vector::ExtractStridedSliceOp::create(
      rewriter, loc, sourceVector, sliceOffset, sliceShape, strides);

  valueMapping[op] = newOp;
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
  Value matmul = nvgpu::MmaSyncOp::create(rewriter, op.getLoc(), opA, opB, opC,
                                          rewriter.getI64ArrayAttr({m, n, k}));
  valueMapping[op.getResult()] = matmul;
  return success();
}

void mlir::populatePrepareVectorToNVMMAPatterns(RewritePatternSet &patterns) {
  vector::populateVectorContractCanonicalizeMatmulToMMT(patterns);
  populateVectorTransferReadTransposePatterns(patterns);
}

LogicalResult mlir::convertVectorToNVVMCompatibleMMASync(RewriterBase &rewriter,
                                                         Operation *rootOp) {
  SetVector<Operation *> ops = getOpToConvert(rootOp);
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
              return convertForOpForMMA(rewriter, forOp, valueMapping);
            })
            .Case([&](scf::YieldOp yieldOp) {
              return convertYieldOpForMMA(rewriter, yieldOp, valueMapping);
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
struct ConvertVectorToNVGPUPass
    : public impl::ConvertVectorToNVGPUPassBase<ConvertVectorToNVGPUPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePrepareVectorToNVMMAPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, getOperation())))
      return signalPassFailure();
  }
};
} // namespace
