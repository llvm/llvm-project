//===- LoopEmitter.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopEmitter.h"
#include "CodegenUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// File local shorthand macros
//===----------------------------------------------------------------------===//

#define CMPI(p, l, r)                                                          \
  (builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::p, (l), (r))       \
       .getResult())

#define C_IDX(v) (constantIndex(builder, loc, (v)))
#define YIELD(vs) (builder.create<scf::YieldOp>(loc, (vs)))
#define ADDI(lhs, rhs) (builder.create<arith::AddIOp>(loc, (lhs), (rhs)))
#define ANDI(lhs, rhs) (builder.create<arith::AndIOp>(loc, (lhs), (rhs)))
#define SUBI(lhs, rhs) (builder.create<arith::SubIOp>(loc, (lhs), (rhs)))
#define MULI(lhs, rhs) (builder.create<arith::MulIOp>(loc, (lhs), (rhs)))
#define REMUI(lhs, rhs) (builder.create<arith::RemUIOp>(loc, (lhs), (rhs)))
#define DIVUI(lhs, rhs) (builder.create<arith::DivUIOp>(loc, (lhs), (rhs)))
#define SELECT(c, l, r) (builder.create<arith::SelectOp>(loc, (c), (l), (r)))

//===----------------------------------------------------------------------===//
// Debugging utils
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
LLVM_ATTRIBUTE_UNUSED static void dumpIndexMemRef(OpBuilder &builder,
                                                  Location loc, Value memref) {
  memref = builder.create<memref::CastOp>(
      loc, UnrankedMemRefType::get(builder.getIndexType(), 0), memref);
  createFuncCall(builder, loc, "printMemrefInd", TypeRange{},
                 ValueRange{memref}, EmitCInterface::On);
}
#endif

//===----------------------------------------------------------------------===//
// File local helper functions.
//===----------------------------------------------------------------------===//

// For index reduction loops, since the tensor are sliced into non-continuous
// fragments, we need a triple [pLo, pHi, pPtr], in which the pair (pLo, pHi)
// specifies the range of the fragment, and pPtr specifies the index of the
// corresponding fragment in the child level (i.e., a pointer to the sliced
// position array).
static constexpr unsigned kSliceIterWidth = 3;

static Value genSliceOffset(OpBuilder &builder, Location loc, Value tensor,
                            Level lvl) {
  auto enc = getSparseTensorEncoding(tensor.getType());
  // FIXME: `toOrigDim` is deprecated
  return createOrFoldSliceOffsetOp(builder, loc, tensor, toOrigDim(enc, lvl));
}

static Value genSliceStride(OpBuilder &builder, Location loc, Value tensor,
                            Level lvl) {
  auto enc = getSparseTensorEncoding(tensor.getType());
  // FIXME: `toOrigDim` is deprecated
  return createOrFoldSliceStrideOp(builder, loc, tensor, toOrigDim(enc, lvl));
}

/// Converts a coordinate relative to the slice to the coordinate relative
/// to the underlying tensor.
// FIXME: that description says "sliceCrd -> tensorCrd"; but the function
// name suggests it should be "tensorCrd -> sliceCrd".
static Value toSliceCrd(OpBuilder &builder, Location loc, Value crd,
                        Value offset, Value stride, Value tensor, Level lvl) {
  // tensorCrd = sliceCrd * stride + offset
  return ADDI(MULI(crd, stride), offset);
}

/// Generates code to compute the *absolute* offset of the slice based on the
/// provide minimum coordinates in the slice.
/// E.g., when reducing d0 + d1 + d2, we need two slices to fully reduced the
/// expression, i,e, s1 = slice(T, d0), s2 = slice(s1, d1). The *absolute*
/// offset is the offset computed relative to the initial tensors T.
///
/// When isNonEmpty == true, the computed offset is meaningless and should not
/// be used during runtime, the method generates code to return 0 currently in
/// that case.
///
/// offset = isNonEmpty && minCrd >= size ? minCrd - size + 1 : 0;
static Value offsetFromMinCoord(OpBuilder &builder, Location loc, Value minCrd,
                                Value size, Value isNonEmpty) {
  Value geSize = CMPI(uge, minCrd, size);
  Value pred = ANDI(isNonEmpty, geSize);
  // Computes minCrd - size + 1
  Value mms = SUBI(ADDI(minCrd, C_IDX(1)), size);
  // This is the absolute offset related to the underly tensor.
  return SELECT(pred, mms, C_IDX(0));
}

/// Converts a coordinate relative to the underlying tensor to the coordinate
/// relative to the slice, returns a extra reminder value
// FIXME: that description says "tensorCrd -> sliceCrd"; but the function
// name suggests it should be "sliceCrd -> tensorCrd".
static std::pair<Value, Value> fromSliceCrd(OpBuilder &builder, Location loc,
                                            Value crd, Value offset,
                                            Value stride, Value tensor,
                                            Level lvl) {
  // sliceCrd = (tensorCrd - offset) / stride
  crd = SUBI(crd, offset);
  Value rem = REMUI(crd, stride);
  crd = DIVUI(crd, stride);
  return std::make_pair(crd, rem);
}

// Generates a bool value for while loop condition that tries to iterate over a
// fully reduced level with affine index expression.
static Value genSparseReducedAffineCond(OpBuilder &builder, Location loc,
                                        Value crdBuf, Value crdHi, Value posit,
                                        Value posHi) {
  Value inBound = CMPI(ult, posit, posHi);
  auto ifOp =
      builder.create<scf::IfOp>(loc, builder.getI1Type(), inBound, true);
  // if (inbound)
  //   yield coord < crdHi
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value crd = genIndexLoad(builder, loc, crdBuf, posit);
  YIELD(CMPI(ult, crd, crdHi));
  // else
  //   yield false
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  YIELD(constantI1(builder, loc, false));

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

// Helper functions that load/store into the position buffer for slice-driven
// loops.
// The sliced pointer buffer is orgnized as:
// [size, curPtr] (two metadata) + [[pLo, pHi, pNext], ...] (list of tuples)
static Value loadSlicePosPtr(OpBuilder &builder, Location loc, Value sPosBuf) {
  // Load curPtr.
  // TODO: We should use SSA value for it.
  return genIndexLoad(builder, loc, sPosBuf, C_IDX(1));
}
static void updateSlicePosPtr(OpBuilder &builder, Location loc, Value sPosBuf,
                              Value pPtr) {
  // Set curPtr.
  // TODO: We should use SSA value for it.
  builder.create<memref::StoreOp>(loc, pPtr, sPosBuf, C_IDX(1));
}
static Value loadSliceNextPosPtrStart(OpBuilder &builder, Location loc,
                                      Value sPosBuf, Value tupleIdx) {
  // load the pNext in the current tuple specified by `tupleIdx`.
  // 4 = 2 (two metadata) + 2 (pNext == tuple[2])
  return genIndexLoad(builder, loc, sPosBuf, ADDI(tupleIdx, C_IDX(4)));
}

std::pair<Value, Value>
LoopEmitter::genSliceLegitPredicate(OpBuilder &builder, Location loc, Value crd,
                                    TensorId tid, Level lvl) {
  assert(isSparseSlices[tid]);
  Value slice = tensors[tid];
  Value offset = sliceOffsets[tid][lvl];
  Value stride = sliceStrides[tid][lvl];
  auto enc = getSparseTensorEncoding(slice.getType());

  const auto [newCrd, crdRem] =
      fromSliceCrd(builder, loc, crd, offset, stride, slice, lvl);

  SmallVector<Value, 3> conds; // at most 3 conditions

  // First, coord >= offset (skip the check if offset is known to be 0).
  if (auto staticOffset = enc.getStaticLvlSliceOffset(lvl);
      !(staticOffset.has_value() && *staticOffset == 0)) {
    auto geOffset = CMPI(uge, crd, offset);
    conds.push_back(geOffset);
  }

  // Second, coord_in_slice < length
  auto ltLength = CMPI(ult, newCrd, lvlSizes[tid][lvl]);
  conds.push_back(ltLength);

  // Third, rem == 0 (skip the check if stride is known to be 1).
  if (auto staticStride = enc.getStaticLvlSliceStride(lvl);
      !(staticStride.has_value() && *staticStride == 1)) {
    auto fitStride = CMPI(eq, crdRem, C_IDX(0));
    conds.push_back(fitStride);
  }

  // Must meet all condition to be a valid coordinate in slice.
  auto pred = conds.front();
  for (auto cond : ValueRange(conds).drop_front())
    pred = ANDI(pred, cond);

  return {newCrd, pred};
}

//===----------------------------------------------------------------------===//
// Sparse tensor loop emitter class implementations
//===----------------------------------------------------------------------===//

Value LoopEmitter::genAddress(OpBuilder &builder, Location loc, TensorId tid,
                              Level lvl, Value crd) {
  Value pos = lvl == 0 ? C_IDX(0) : posits[tid][lvl - 1];
  Value mul = MULI(highs[tid][lvl], pos);
  if (isSparseSlices[tid])
    crd = toSliceCrd(builder, loc, crd, sliceOffsets[tid][lvl],
                     sliceStrides[tid][lvl], tensors[tid], lvl);
  Value add = ADDI(mul, crd);
  return add;
}

Value LoopEmitter::genSegmentHigh(OpBuilder &builder, Location loc,
                                  TensorId tid, Level lvl, Value pLo,
                                  Value pHi) {
  const auto coordinates = coordinatesBuffers[tid][lvl];
  const auto sameCrd = genIndexLoad(builder, loc, coordinates, pLo);
  auto whileOp = builder.create<scf::WhileOp>(
      loc, builder.getIndexType(), pLo,
      /*beforeBuilder=*/
      [pHi, coordinates, sameCrd](OpBuilder &builder, Location loc,
                                  ValueRange ivs) {
        const auto pos = ivs[0];
        Value inBound = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, pos, pHi);
        auto ifInBound =
            builder.create<scf::IfOp>(loc, builder.getI1Type(), inBound, true);
        {
          OpBuilder::InsertionGuard guard(builder);
          // Load the next coordinates only when inbound (to avoid OOB
          // acccesses).
          builder.setInsertionPointToStart(ifInBound.thenBlock());
          Value crd = genIndexLoad(builder, loc, coordinates, pos);
          Value isSameCrd = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, crd, sameCrd);
          YIELD(isSameCrd);
          // Else, the position is out of bound, yield false to terminate the
          // loop.
          builder.setInsertionPointToStart(ifInBound.elseBlock());
          YIELD(constantI1(builder, loc, false));
        }
        builder.create<scf::ConditionOp>(loc, ifInBound.getResults()[0], ivs);
      },
      /*afterBuilder=*/
      [](OpBuilder &builder, Location loc, ValueRange ivs) {
        // pos ++
        Value nextPos = ADDI(ivs[0], C_IDX(1));
        YIELD(nextPos);
      });
  // Return the segment high.
  return whileOp.getResult(0);
}

Value LoopEmitter::genSparseCrd(OpBuilder &builder, Location loc, TensorId tid,
                                Level lvl) {
  // A load on the coordinates array yields the coordinate.
  const Value mem = coordinatesBuffers[tid][lvl];
  /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
  const Value pos = posits[tid][lvl];
  const Value crd = genIndexLoad(builder, loc, mem, pos);
  return crd;
}

LoopEmitter::LoopEmitter(ValueRange tensors, StringAttr loopTag, bool hasOutput,
                         bool isSparseOut, ArrayRef<LoopId> topSort,
                         DependentLvlGetter dimGetter) {
  initialize(tensors, loopTag, hasOutput, isSparseOut, topSort, dimGetter);
}

void LoopEmitter::initialize(ValueRange ts, StringAttr loopTag, bool hasOutput,
                             bool isSparseOut, ArrayRef<LoopId> topSort,
                             DependentLvlGetter dimGetter) {
  // First initialize the top-level type of the fields.
  this->loopTag = loopTag;
  this->hasOutput = hasOutput;
  this->isSparseOut = isSparseOut;

  const unsigned numManifestTensors = ts.size();
  const unsigned synTensorId = numManifestTensors;
  const unsigned numTensors = numManifestTensors + 1;
  // tensors array (len == numManifestTensor).
  this->tensors.assign(ts.begin(), ts.end());
  // Arrays with len == numTensor.
  this->lvlTypes.assign(numTensors, std::vector<DimLevelType>());
  this->lvlSizes.assign(numTensors, std::vector<Value>());
  this->highs.assign(numTensors, std::vector<Value>());
  this->segHi.assign(numTensors, std::vector<Value>());
  this->posits.assign(numTensors, std::vector<Value>());
  this->coords.assign(numTensors, std::vector<Value>());
  this->positionsBuffers.assign(numTensors, std::vector<Value>());
  this->coordinatesBuffers.assign(numTensors, std::vector<Value>());
  this->valBuffer.assign(numTensors, nullptr);
  this->isSparseSlices.assign(numTensors, false);
  this->sliceOffsets.assign(numTensors, std::vector<Value>());
  this->sliceStrides.assign(numTensors, std::vector<Value>());

  const LoopOrd numLoops = topSort.size();
  // These zeros will be overwritten below, but we need to initialize
  // them to something since we'll need random-access assignment.
  this->loopIdToOrd.assign(numLoops, 0);
  this->loopStack.reserve(numLoops);
  this->loopSeqStack.reserve(numLoops);

  // Index-reduction related fields.
  this->dependentLvlMap.assign(
      numTensors, std::vector<std::vector<std::pair<TensorLevel, unsigned>>>());
  this->slicePosBuffer.assign(numTensors, std::vector<std::vector<Value>>());
  this->sliceMeta.assign(
      numTensors, std::vector<std::vector<std::pair<Value, unsigned>>>());
  this->sliceStack.assign(numTensors, std::vector<SliceInfo>());
  this->levelReducedDep.assign(numTensors, std::vector<unsigned>());

  // Initialize nested types of `TensorId`-indexed fields.
  for (TensorId tid = 0; tid < numTensors; tid++) {
    Level lvlRank;
    if (tid == synTensorId) {
      // Synthetic tensor (conceptually) is an all-dense tensor with rank equal
      // to the total number of loops (each level can potentially be mapped to
      // one of the loop being generated).
      lvlRank = numLoops;
      lvlTypes[tid].assign(lvlRank, DimLevelType::Dense);
    } else {
      const Value t = tensors[tid];
      // a scalar or 0-dimension tensors
      if (isZeroRankedTensorOrScalar(t.getType()))
        continue;

      auto rtp = getRankedTensorType(t);
      const SparseTensorType stt(rtp);
      lvlRank = stt.getLvlRank();

      // We always treat sparse output tensor as dense so that we always iterate
      // it based on lvl size.
      if (stt.hasEncoding() && !(isOutputTensor(tid) && isSparseOut)) {
        const auto enc = stt.getEncoding();
        isSparseSlices[tid] = enc.isSlice();
        for (auto lvlTp : enc.getLvlTypes())
          lvlTypes[tid].push_back(lvlTp);
      } else {
        lvlTypes[tid].assign(lvlRank, DimLevelType::Dense);
      }
    }

    // Initialize using empty value.
    lvlSizes[tid].assign(lvlRank, Value());
    highs[tid].assign(lvlRank, Value());
    segHi[tid].assign(lvlRank, Value());
    posits[tid].assign(lvlRank, Value());
    coords[tid].assign(lvlRank, Value());
    positionsBuffers[tid].assign(lvlRank, Value());
    coordinatesBuffers[tid].assign(lvlRank, Value());
    sliceOffsets[tid].assign(lvlRank, Value());
    sliceStrides[tid].assign(lvlRank, Value());

    // Slice-driven loops related initialization.
    levelReducedDep[tid].assign(lvlRank, 0);
    dependentLvlMap[tid].assign(
        lvlRank, std::vector<std::pair<TensorLevel, unsigned>>());
    slicePosBuffer[tid].assign(lvlRank, std::vector<Value>());
    sliceMeta[tid].assign(lvlRank, std::vector<std::pair<Value, unsigned>>());
    sliceStack[tid].emplace_back(/*minCrd=*/Value(),
                                 /*offset=*/Value(), /*isNonEmpty*/ Value(),
                                 std::nullopt, 0);
    if (dimGetter && !isSynTensor(tid)) {
      for (Level l = 0; l < lvlRank; l++) {
        dependentLvlMap[tid][l] = dimGetter(tid, l);
        unsigned depends = dependentLvlMap[tid][l].size();
        if (depends == 0)
          continue;
        sliceMeta[tid][l].assign(depends, std::make_pair(nullptr, 0));
        // We need `depends - 1` slices to fully reduce the affine expression.
        slicePosBuffer[tid][l].assign(depends - 1, nullptr);
      }
    }
  }

  // Construct the inverse of the `topSort` from the sparsifier.
  // This is needed to map `AffineDimExpr`s back to the `LoopOrd`
  // used in loop emitter.
  // FIXME: This map should be maintained outside loop emitter.
  for (LoopOrd n = 0; n < numLoops; n++)
    loopIdToOrd[topSort[n]] = n;
}

void LoopEmitter::initializeLoopEmit(
    OpBuilder &builder, Location loc, LoopEmitter::OutputUpdater updater,
    LoopEmitter::SynTensorBoundSetter synSetter) {

  // For every synthetic tensor, set the high bound by calling the callback.
  if (synSetter)
    for (unsigned i = 0, e = highs[getSynTensorId()].size(); i < e; i++)
      highs[getSynTensorId()][i] = synSetter(builder, loc, i);

  // For every manifest tensor:
  // * get the values buffer.
  // * For every level:
  //   * get the positions and coordinates buffers
  //   * get/compute the level-size, which is also used as the upper-bound
  //     on positions.
  for (TensorId t = 0, numTensors = getNumManifestTensors(); t < numTensors;
       t++) {
    const Value tensor = tensors[t];
    const auto rtp = dyn_cast<RankedTensorType>(tensor.getType());
    if (!rtp)
      // Skips only scalar, zero ranked tensor still need to be bufferized and
      // (probably) filled with zeros by users.
      continue;
    // FIXME: the definition of `lvlRank` looks more like a dim-rank;
    // but the variable is used as a level everywhere below, which
    // suggests there may be some dim/lvl confusion going on here.
    const Level lvlRank = rtp.getRank();
    const auto shape = rtp.getShape();
    const auto enc = getSparseTensorEncoding(rtp);
    const Level cooStart = enc ? getCOOStart(enc) : lvlRank;
    // Scan all levels of current tensor.
    for (Level l = 0; l < lvlRank; l++) {
      // This should be called only once at beginning.
      assert(!positionsBuffers[t][l] && !coordinatesBuffers[t][l] &&
             !highs[t][l]);
      const auto lvlTp = lvlTypes[t][l];
      // Handle sparse storage schemes.
      if (isCompressedDLT(lvlTp) || isCompressedWithHiDLT(lvlTp)) {
        // Generate sparse primitives to obtain positions and coordinates.
        positionsBuffers[t][l] = genToPositions(builder, loc, tensor, l);
        coordinatesBuffers[t][l] =
            genToCoordinates(builder, loc, tensor, l, cooStart);
      } else if (isSingletonDLT(lvlTp)) {
        // Singleton level, fetch coordinates.
        coordinatesBuffers[t][l] =
            genToCoordinates(builder, loc, tensor, l, cooStart);
      } else {
        // Dense level, nothing to fetch.
        assert(isDenseDLT(lvlTp));
      }

      // FIXME: `toOrigDim` is deprecated.  For now this relies on the
      // 1:1 mapping between levels and dimensions, since nowhere else
      // in the code supports non-permutations yet either.
      Value lvlSz = mlir::linalg::createOrFoldDimOp(builder, loc, tensor,
                                                    toOrigDim(enc, l));
      // Find upper bound in current dimension.
      highs[t][l] = lvlSizes[t][l] = lvlSz;
      if (isSparseSlices[t]) {
        sliceOffsets[t][l] = genSliceOffset(builder, loc, tensors[t], l);
        sliceStrides[t][l] = genSliceStride(builder, loc, tensors[t], l);
      }
    }

    // Perform the required bufferization. Dense inputs materialize
    // from the input tensors. Sparse inputs use sparse primitives to obtain the
    // values.
    // Delegates extra output initialization to clients.
    bool isOutput = isOutputTensor(t);
    Type elementType = rtp.getElementType();
    if (!enc) {
      // Non-annotated dense tensors.
      BaseMemRefType denseTp = MemRefType::get(shape, elementType);

      // TODO: if we unconditionally use fully dynamic layout here, it breaks
      // some vectorization passes which requires static stride = 1.
      // Is it possible to call vectorization pass after bufferization?
      if (llvm::isa_and_nonnull<tensor::ExtractSliceOp>(tensor.getDefiningOp()))
        denseTp = bufferization::getMemRefTypeWithFullyDynamicLayout(rtp);

      Value denseVal =
          builder.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
      // Dense outputs need special handling.
      if (isOutput && updater)
        denseVal = updater(builder, loc, denseVal, tensor);

      valBuffer[t] = denseVal;
    } else {
      // Annotated sparse tensors.
      // We also need the value buffer for all-dense annotated "sparse" tensors.
      valBuffer[t] = genToValues(builder, loc, tensor);
    }
    // NOTE: we can also prepare for 0 lvl here in advance, this will hoist
    // some loop preparation from tensor iteration, but will also (undesirably)
    // hoist the code ouside if-conditions.
  }

  Type indexType = builder.getIndexType();
  Value c0 = constantZero(builder, loc, indexType);
  for (TensorId t = 0, e = tensors.size(); t < e; t++) {
    auto rtp = dyn_cast<RankedTensorType>(tensors[t].getType());
    if (!rtp)
      continue;

    Level lvlRank = SparseTensorType(rtp).getLvlRank();
    for (Level lvl = 0; lvl < lvlRank; lvl++) {
      if (!dependentLvlMap[t][lvl].empty()) {
        ArrayRef<std::pair<TensorLevel, unsigned>> depLvls =
            dependentLvlMap[t][lvl];
        // Needs at least two operands to form a non-trivial affine expression.
        assert(depLvls.size() == sliceMeta[t][lvl].size());

        Value size = c0;
        for (int e = depLvls.size() - 1; e >= 0; e--) {
          auto [dt, dl] = unpackTensorLevel(depLvls[e].first);
          unsigned stride = depLvls[e].second;
          Value stridedSize = lvlSizes[dt][dl];
          if (stride != 1)
            stridedSize = MULI(stridedSize, C_IDX(stride));
          size = ADDI(size, stridedSize);
          sliceMeta[t][lvl][e] = std::make_pair(size, stride);
        }
      }
    }
  }
  localInsertPos = builder.getInsertionPoint()->getPrevNode();
}

void LoopEmitter::categorizeLoopCondition(
    ArrayRef<TensorLevel> tidLvls, SmallVectorImpl<TensorLvlCond> &dnConds,
    SmallVectorImpl<TensorLvlCond> &spConds) {
  // Finds out the tensor level that we should use to generate loops. Amongs all
  // the tensor levels, there is at most one sparse tensor level.
  for (auto [t, l] : unpackTensorLevelRange(tidLvls)) {
    assert(lvlTypes[t].size() > l); // Must be a valid tid, dim pair
    auto lvlType = lvlTypes[t][l];
    // Must be a recognizable DLT.
    assert(isDenseDLT(lvlType) || isCompressedDLT(lvlType) ||
           isCompressedWithHiDLT(lvlType) || isSingletonDLT(lvlType));

    bool isSparse = !isDenseDLT(lvlType);
    bool isSlice = isSparseSlices[t];
    bool isAffine = !dependentLvlMap[t][l].empty();
    bool isUnRedu = false;
    // TODO: Supports affine index expression on sparse tensor slices.
    assert(!isSlice || !isAffine);

    // Whether the affine index expression has been fully reduced or not.
    if (!dependentLvlMap[t][l].empty())
      isUnRedu = !depFullyReduced(t, l);

    auto &dstVec = isSparse ? spConds : dnConds;
    dstVec.emplace_back(
        makeTensorLevel(t, l),
        makeLoopCondKind(isSparse, isSlice, isAffine, isUnRedu));
  }

  std::stable_sort(spConds.begin(), spConds.end(), [](auto lhs, auto rhs) {
    // AffineUnRed > Affine > Slice > Trivial
    return static_cast<uint8_t>(lhs.second) > static_cast<uint8_t>(rhs.second);
  });
}

void LoopEmitter::enterNewLoopSeq(OpBuilder &builder, Location loc,
                                  ArrayRef<TensorLevel> tidLvls) {
  // TODO: sort
  assert(loopSeqStack.size() == loopStack.size());
  // Prepares for all the tensors used in the current loop sequence.
  std::vector<std::tuple<TensorId, Level, bool>> slicedTids;

  for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
    if (!dependentLvlMap[tid][lvl].empty()) {
      bool fullyRed = genSliceBegin(builder, loc, tid, lvl);
      slicedTids.emplace_back(tid, lvl, fullyRed);
    } else if (!isSynTensor(tid)) {
      prepareLoopOverTensorAtLvl(builder, loc, tid, lvl);
    }
  }

  // Universal Index starts from 0.
  loopSeqStack.emplace_back(C_IDX(0), std::move(slicedTids));
}

void LoopEmitter::exitCurrentLoopSeq(OpBuilder &builder, Location loc) {
  assert(loopSeqStack.size() == loopStack.size() + 1);

  const auto &slicedTids = loopSeqStack.back().second;

  // Depending on whether the slice is resolved or not at current loop sequence,
  // end them in different ways.
  for (auto [tid, lvl, res] : slicedTids) {
    if (!res) {
      // If this is a unresolved-slice-driven loop, pops out the slice.
      assert(sliceStack[tid].back().slicedOnLvl == lvl);
      sliceStack[tid].pop_back();
    }
  }
  loopSeqStack.pop_back();
}

Value LoopEmitter::genAffine(OpBuilder &builder, Location loc, AffineExpr a) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    // FIXME: since the one callsite in Sparsification passes in a
    // level-expression, the `getPosition` must in fact be a `Dimension`.
    // However, elsewhere we have been lead to expect that `loopIdToOrd`
    // should be indexed by `LoopId`...
    const auto loopId = a.cast<AffineDimExpr>().getPosition();
    assert(loopId < loopIdToOrd.size());
    return loopStack[loopIdToOrd[loopId]].iv;
  }
  case AffineExprKind::Add: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return ADDI(genAffine(builder, loc, binOp.getLHS()),
                genAffine(builder, loc, binOp.getRHS()));
  }
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return MULI(genAffine(builder, loc, binOp.getLHS()),
                genAffine(builder, loc, binOp.getRHS()));
  }
  case AffineExprKind::Constant: {
    int64_t c = a.cast<AffineConstantExpr>().getValue();
    return C_IDX(c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

std::pair<Operation *, Value> LoopEmitter::emitForLoopOverTensorAtLvl(
    OpBuilder &builder, Location loc, TensorId tid, Level lvl, Value lo,
    Value hi, MutableArrayRef<Value> reduc, bool isParallel) {
  bool isSparseCond = isCompressedDLT(lvlTypes[tid][lvl]) ||
                      isCompressedWithHiDLT(lvlTypes[tid][lvl]) ||
                      isSingletonDLT(lvlTypes[tid][lvl]);
  // TODO: support dynamic slices.
  // Uses the first dimension here to build the loop bound (which is also the
  // biggest range).
  Value step = C_IDX(1);
  Operation *loop = nullptr;
  Value iv;
  if (isParallel) {
    scf::ParallelOp parOp =
        builder.create<scf::ParallelOp>(loc, lo, hi, step, reduc);
    builder.setInsertionPointToStart(parOp.getBody());
    assert(parOp.getNumReductions() == reduc.size());
    iv = parOp.getInductionVars()[0];

    // In-place update on the reduction variable vector.
    // Note that the init vals is not the actual reduction variables but instead
    // used as a "special handle" to (temporarily) represent them. The
    // expression on init vals will be moved into scf.reduce and replaced with
    // the block arguments when exiting the loop (see exitForLoop). This is
    // needed as we can not build the actual reduction block and get the actual
    // reduction varaible before users fill parallel loop body.
    for (int i = 0, e = reduc.size(); i < e; i++)
      reduc[i] = parOp.getInitVals()[i];
    loop = parOp;
  } else {
    scf::ForOp forOp = builder.create<scf::ForOp>(loc, lo, hi, step, reduc);
    builder.setInsertionPointToStart(forOp.getBody());
    iv = forOp.getInductionVar();

    // In-place update on the reduction variable vector.
    assert(forOp.getNumRegionIterArgs() == reduc.size());
    for (int i = 0, e = reduc.size(); i < e; i++)
      reduc[i] = forOp.getRegionIterArg(i);
    loop = forOp;
  }
  assert(loop && iv);

  Value crd;
  if (isSparseCond) {
    // For COO, the position is the same across consecutive levels.
    /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
    posits[tid][lvl] = iv;
    crd = genSparseCrd(builder, loc, tid, lvl);
  } else {
    // Dense tensor, the coordinate is the inducation variable.
    crd = iv;
  }

  if (isSparseSlices[tid] && isSparseCond) {
    // For sparse level slices, we need to filter out invalid coordinates that
    // are not included in the slice.
    SmallVector<Type> types;
    for (Value red : reduc)
      types.push_back(red.getType());

    auto [trans, pred] = genSliceLegitPredicate(builder, loc, crd, tid, lvl);
    bool hasReduc = !types.empty();
    scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, pred,
                                               /*else*/ hasReduc);
    if (hasReduc) {
      // scf.for (a) -> v
      //  %s = scf.if (a) -> v
      //    user-generated code.
      //  else
      //    yield a
      //  yield %s
      YIELD(ifOp.getResults());
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      // On mismatch.
      YIELD(reduc);
    }
    // Set the insertion point to matched branch.
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    crd = trans;
  }

  assert(crd);
  coords[tid][lvl] = crd;
  return {loop, crd};
}

Value LoopEmitter::genWhileLoopConditions(OpBuilder &builder, Location loc,
                                          ValueRange ivs, TensorLvlCond cond) {
  auto [tid, lvl] = unpackTensorLevel(cond.first);

  switch (cond.second) {
  case LoopCondKind::SparseCond: {
    assert(ivs.size() == 1);
    // We used the first level bound as the bound the collapsed set of levels.
    return CMPI(ult, ivs.back(), highs[tid][lvl]);
  }
  case LoopCondKind::SparseSliceCond: {
    assert(ivs.size() == 1);
    return CMPI(ult, ivs.back(), highs[tid][lvl]);
  }
  case LoopCondKind::SparseAffineCond: {
    assert(ivs.size() == 1);

    Value crdHi; // loop upper bound
    {
      OpBuilder::InsertionGuard guard(builder);
      Operation *loop = builder.getInsertionBlock()->getParentOp();
      // crdHi is a loop invariant, hosit the computation outside the loop.
      if (llvm::isa_and_nonnull<scf::WhileOp>(loop))
        builder.setInsertionPoint(loop);
      auto [remSz, stride] = sliceMeta[tid][lvl].back();
      assert(stride == 1 && "Not yet implemented");
      crdHi = ADDI(getMostRecentSliceOnLvl(tid, lvl).offset, remSz);
    }
    assert(crdHi);
    return genSparseReducedAffineCond(builder, loc,
                                      coordinatesBuffers[tid][lvl], crdHi,
                                      ivs[0], highs[tid][lvl]);
  }
  case LoopCondKind::SparseAffineUnRedCond: {
    assert(ivs.size() == 3);
    return ivs.front(); // isNonEmpty
  }
  default:
    llvm_unreachable("Unhandled LoopCondKind");
  }
  llvm_unreachable("Unhandled LoopCondKind");
}

std::optional<Value> LoopEmitter::genWhileLoopBody(OpBuilder &builder,
                                                   Location loc, ValueRange ivs,
                                                   TensorLvlCond cond) {
  auto [tid, lvl] = unpackTensorLevel(cond.first);

  switch (cond.second) {
  case LoopCondKind::SparseCond: {
    // Updates position. For collapsed COO, the position is the same across
    // consecutive levels.
    posits[tid][lvl] = ivs.back();

    // Update coordinates.
    coords[tid][lvl] = genSparseCrd(builder, loc, tid, lvl);
    return std::nullopt;
  }
  case LoopCondKind::SparseSliceCond: {
    assert(ivs.size() == 1);
    posits[tid][lvl] = ivs.front();
    Value sCrd = genSparseCrd(builder, loc, tid, lvl);
    // Converts the coordinate loaded from the actual sparse tensor to the
    // coordinates in the sparse slice.
    auto [dCrd, pred] = genSliceLegitPredicate(builder, loc, sCrd, tid, lvl);
    coords[tid][lvl] = dCrd;
    return pred;
  }
  case LoopCondKind::SparseAffineCond: {
    assert(ivs.size() == 1);
    // Coord is the relative offset related to its parents.
    assert(sliceStack[tid].back().depth == 1 && "TODO: not yet implement");
    // Update c = absOffset[lvl][depth] - absOffset[lvl][depth - 1]
    Value posit = ivs[0];
    Value crdBuf = coordinatesBuffers[tid][lvl];
    // We need to substract the offset to get relative coordinates.
    // TODO: Maybe assert relC >=0 during runtime in debug build?
    Value absC = genIndexLoad(builder, loc, crdBuf, posit);
    auto relC = SUBI(absC, getFinalSliceOnLvl(tid, lvl).offset);
    posits[tid][lvl] = posit;
    coords[tid][lvl] = relC;
    return std::nullopt;
  }
  case LoopCondKind::SparseAffineUnRedCond: {
    unsigned depth = sliceStack[tid].back().depth;
    unsigned curStride = sliceMeta[tid][lvl][depth - 1].second;
    assert(ivs.size() == 3);

    // Updates the current slice info
    SliceInfo &sliceInfo = sliceStack[tid].back();
    sliceInfo.isNonEmpty = ivs[0];
    sliceInfo.minCrd = ivs[1];
    sliceInfo.offset = ivs[2];

    // Crd (the value we used to coiterate) is the relative offset related to
    // its parents, we can use the absolute offset here because when depth = 1,
    // absOffset[lvl][depth - 1] always equals zero.
    // TODO: Update crd =absOffset[lvl][depth] - absOffset[lvl][depth - 1]
    assert(depth == 1 && "TODO: not yet implement");
    Value crd = sliceInfo.offset;

    Value onStride = constantI1(builder, loc, true);
    if (curStride != 1) {
      Value strideVal = C_IDX(curStride);
      Value rem = REMUI(crd, strideVal);
      crd = DIVUI(crd, strideVal);
      onStride = CMPI(eq, rem, C_IDX(0));
    }
    coords[tid][lvl] = crd;
    // No extra check is needed before accessing the tensor level.
    return onStride;
  }
  default:
    llvm_unreachable("Unhandled LoopCondKind");
  }
  llvm_unreachable("Unhandled LoopCondKind");
}

ValueRange LoopEmitter::genCheckedValue(OpBuilder &builder, Location loc,
                                        Value pred, ValueRange curArgs,
                                        TensorLvlCond cond) {
  assert(isSparseCond(cond.second));
  auto [tid, lvl] = unpackTensorLevel(cond.first);
  if (isAffineIdxUnRedCond(cond.second)) {
    unsigned depth = sliceStack[tid].back().depth;
    unsigned curStride = sliceMeta[tid][lvl][depth - 1].second;
    if (curStride == 1)
      return curArgs;
    // Build
    // if (onStride) {
    //    yield curSlice
    // } else {
    //    yield nxSlice.
    //}
    assert(curArgs.size() == 3);
    auto ifOp = builder.create<scf::IfOp>(loc, curArgs.getTypes(), pred, true);
    {
      OpBuilder::InsertionGuard guard(builder);
      // If not all slices are legit, yield the updated value.
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

      YIELD(curArgs);
      // If not all slices are legit, yield the updated value.
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto [nonEmpty, minCrd, offset] =
          genSliceNextInduction(builder, loc, tid, lvl);
      SmallVector<Value> nxSlice{nonEmpty, minCrd, offset};
      YIELD(nxSlice);
    }
    // If all slices are legit, start the user generated code.
    return ifOp.getResults();
  } else {
    // Currently only sparse slice condition need extra check.
    assert(isSliceCond(cond.second) && isSparseCond(cond.second));
    assert(curArgs.size() == 1);
    Value nextPos = ADDI(curArgs.front(), C_IDX(1));
    return SELECT(pred, curArgs.front(), nextPos)->getResults();
  }
  llvm_unreachable("unhandled case");
}

std::pair<Operation *, Value> LoopEmitter::emitWhileLoopOverTensorsAtLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorLvlCond> spConds,
    MutableArrayRef<Value> reduc, bool needsUniv) {
  // NOTE: the slice driven tensor-related reduction variable must
  // appear before normal tensors.
  assert(!spConds.empty());

  // The set of induction variables for the while loop.
  SmallVector<Value> ivs;
  // Segement sizes for induction variables used for different kinds of loop
  // conditions.
  SmallVector<unsigned> opSegSize;

  // Construct the while-loop with a parameter for each coordinate.
  for (auto [tl, cKind] : spConds) {
    auto [tid, lvl] = unpackTensorLevel(tl);
    const auto lvlTp = lvlTypes[tid][lvl];
    // Dense level are handled by the shared univeral index.
    assert(!isDenseCond(cKind));
    // Must be a recognizable sparse level.
    assert(isCompressedDLT(lvlTp) || isCompressedWithHiDLT(lvlTp) ||
           isSingletonDLT(lvlTp));
    (void)lvlTp;

    unsigned prevSz = ivs.size();
    if (isAffineIdxCond(cKind)) {
      // TODO: Support view-based reshape on sparse levels with affine index
      // expressions.
      if (isAffineIdxUnRedCond(cKind)) {
        SliceInfo &sliceInfo = sliceStack[tid].back();
        // The order matters!
        ivs.push_back(sliceInfo.isNonEmpty);
        ivs.push_back(sliceInfo.minCrd);
        ivs.push_back(sliceInfo.offset);
      } else {
        ivs.push_back(posits[tid][lvl]); // loop lower bound (pos low).
      }
      // We reduced one more dependency after entering the loop.
      levelReducedDep[tid][lvl]++;
    } else {
      assert(dependentLvlMap[tid][lvl].empty());
      const Value pos = posits[tid][lvl];
      ivs.push_back(pos);
    }
    opSegSize.push_back(ivs.size() - prevSz);
  }

  // The position where user-supplied reduction variable starts.
  ivs.append(reduc.begin(), reduc.end());
  // Update universal index.
  if (needsUniv)
    ivs.push_back(loopSeqStack.back().first);

  // Ensures all operands are valid.
  assert(llvm::all_of(ivs, [](Value v) { return v != nullptr; }));
  TypeRange types = ValueRange(ivs).getTypes();
  auto whileOp = builder.create<scf::WhileOp>(loc, types, ivs);

  SmallVector<Location> locs(types.size(), loc);
  Block *before = builder.createBlock(&whileOp.getBefore(), {}, types, locs);
  Block *after = builder.createBlock(&whileOp.getAfter(), {}, types, locs);

  // Generates loop conditions.
  builder.setInsertionPointToStart(before);
  ValueRange bArgs = before->getArguments();
  Value whileCond = nullptr; // bool values for loop condition.
  for (auto [c, segSz] : llvm::zip_equal(spConds, opSegSize)) {
    Value cv = genWhileLoopConditions(builder, loc, bArgs.take_front(segSz), c);
    bArgs = bArgs.drop_front(segSz);
    whileCond = !whileCond ? cv : ANDI(whileCond, cv);
  }
  // The remaining block arguments are user-provided reduction values and an
  // optional universal index. Make sure their sizes match.
  assert(bArgs.size() == reduc.size() + needsUniv ? 1 : 0);
  builder.create<scf::ConditionOp>(loc, whileCond, before->getArguments());

  // Generates loop body.
  builder.setInsertionPointToStart(after);
  ValueRange aArgs = after->getArguments();
  // Since some LoopCondKind might need extra checks to filter out invalid
  // iterations, we maintains another array to hold the iteration arguments to
  // yield if the checks fails.
  SmallVector<Value> nextArgs(aArgs.begin(), aArgs.end());
  // A mutable alias for convenient slicing.
  MutableArrayRef<Value> nextArgsRef = nextArgs;
  Value extraPred = nullptr;
  for (auto [c, segSz] : llvm::zip_equal(spConds, opSegSize)) {
    ValueRange condArgs = aArgs.take_front(segSz);
    auto pred = genWhileLoopBody(builder, loc, condArgs, c);
    assert(pred.has_value() == isCondWithExtraCheck(c.second));
    if (pred.has_value()) {
      // We need all extra checks to pass.
      extraPred = extraPred == nullptr ? *pred : ANDI(*pred, extraPred);
      ValueRange nxArgs = genCheckedValue(builder, loc, *pred, condArgs, c);
      assert(nxArgs.size() == segSz);
      // Update the value for cases when some check fails.
      for (unsigned i = 0; i < segSz; i++) {
        nextArgsRef[i] = nxArgs[i];
      }
    }
    aArgs = aArgs.drop_front(segSz);
    nextArgsRef = nextArgsRef.drop_front(segSz);
  }

  if (extraPred) {
    auto ifOp = builder.create<scf::IfOp>(loc, types, extraPred, /*else*/ true);
    // Marks this special IfOp so that Sparsification does not finalizing it.
    ifOp->setAttr(getLoopEmitterLoopAttrName(),
                  StringAttr::get(builder.getContext(), "slice"));
    // Links the SSA chain outside the if statement.
    YIELD(ifOp->getResults());

    // If not all slices are legit, yield the updated value.
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    YIELD(nextArgs);

    // If all slices are legit, start the user generated code.
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }

  for (auto [tid, lvl] : unpackTensorLevelFromCondRange(spConds)) {
    // Generates segment high for non-unique level.
    if (!isUniqueDLT(lvlTypes[tid][lvl])) {
      segHi[tid][lvl] = genSegmentHigh(builder, loc, tid, lvl, posits[tid][lvl],
                                       highs[tid][lvl]);
    }
  }

  // In-place update on reduction variable.
  assert(aArgs.size() == reduc.size() + needsUniv ? 1 : 0);
  for (unsigned i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = aArgs[i];

  Value min;
  // Finds the minimum coordinate
  if (!needsUniv) {
    for (auto [tid, lvl] : unpackTensorLevelFromCondRange(spConds)) {
      const auto lvlTp = lvlTypes[tid][lvl];
      if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp) ||
          isCompressedWithHiDLT(lvlTp)) {
        const auto crd = coords[tid][lvl];
        if (min) {
          Value cmp = CMPI(ult, coords[tid][lvl], min);
          min = SELECT(cmp, coords[tid][lvl], min);
        } else {
          min = crd;
        }
      }
    }
  } else {
    assert(!min);
    // Otherwise, universal index is the minimal pos.
    min = whileOp.getAfterArguments().back();
  }

  return {whileOp, min};
}

bool LoopEmitter::shouldIteratedByForLoop(ArrayRef<TensorLvlCond> sparseConds,
                                          bool genDedup) {
  assert(llvm::all_of(sparseConds,
                      [](TensorLvlCond c) { return isSparseCond(c.second); }));

  // If we need to co-iterate over two sparse tensors, we need a while loop
  if (sparseConds.size() > 1)
    return false;

  // We also need a while loop for levels with affine index expression and
  // non-unique levels when deduplication is required.
  if (sparseConds.size() == 1) {
    auto [tid, lvl] = unpackTensorLevel(sparseConds.back().first);
    return !isAffineIdxCond(sparseConds.back().second) &&
           !(genDedup && !isUniqueDLT(lvlTypes[tid][lvl]));
  }

  return true;
}

Operation *LoopEmitter::enterCoIterationOverTensorsAtLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorLevel> tidLvls,
    MutableArrayRef<Value> reduc, bool tryParallel, bool genDedup,
    bool needsUniv) {
#ifndef NDEBUG
  // Sanity checks.
  assert(!tidLvls.empty());
  for (auto [t, l] : unpackTensorLevelRange(tidLvls)) {
    assert(!coords[t][l] ||                 // We cannot re-enter the same level
           !dependentLvlMap[t][l].empty()); // unless it is a slice-driver loop
  }
#endif
  // TODO: support multiple return on parallel for?
  tryParallel = tryParallel && reduc.size() <= 1;

  SmallVector<TensorLvlCond> spConds;
  SmallVector<TensorLvlCond> dnConds;
  categorizeLoopCondition(tidLvls, dnConds, spConds);

  // Only when there is at least one sparse conditions, do we really need the
  // universal index.
  // TODO: Maybe we should instead requires merger to pass in a valid value at
  // the first place instead of adjusting it in LoopEmitter?
  needsUniv = !spConds.empty() && needsUniv;
  // The TensorLevel used for loop conditions.
  // If there is any sparse level, we need to use the sparse condition.
  // If all levels are dense, we can pick arbitary one (dense slice-driven loop
  // can be generated using a simple ForOp as well).
  Operation *l = nullptr;
  Value iv = nullptr;
  SmallVector<SliceLoopInfo> sliceDrivenInfo;
  SmallVector<TensorLevel> trivialLvls;

  // Generates loops differently depending on whether we need a slice-driven
  // loop or a simple level traversal loop.
  if (shouldIteratedByForLoop(spConds, genDedup) && !needsUniv) {
    assert(spConds.size() <= 1);
    TensorLvlCond tlCond = spConds.empty() ? dnConds.front() : spConds.front();
    auto loopCondKind = tlCond.second;
    auto [tid, lvl] = unpackTensorLevel(tlCond.first);
    Value lo = isSparseCond(loopCondKind)
                   ? posits[tid][lvl]           // current offset
                   : loopSeqStack.back().first; // universal index
    Value hi = highs[tid][lvl];
    if (isDenseCond(loopCondKind) && isAffineIdxCond(loopCondKind)) {
      bool unReduc = isAffineIdxUnRedCond(loopCondKind);
      assert(unReduc == !depFullyReduced(tid, lvl));
      unsigned depth = sliceStack[tid].back().depth;
      assert(depth >= 1);
      // The *next* slice size after reducing the current index variable.
      auto [nxSz, nxStride] = sliceMeta[tid][lvl][depth];
      // The *current* stride to reduce the current index variable.
      // E.g., for 2 * i, stride = 2.
      unsigned stride = sliceMeta[tid][lvl][depth - 1].second;
      hi = nxSz;
      if (unReduc) {
        // Adjust for loop hi for dense slice-driven loop.
        hi = SUBI(lvlSizes[tid][lvl], hi);
        hi = ADDI(hi, C_IDX(1));
        hi = DIVUI(hi, C_IDX(stride));
      } else {
        // TODO: dialuted convolution.
        assert(nxStride == 1 && "Not yet implemented.");
      }
    }
    std::tie(l, iv) = emitForLoopOverTensorAtLvl(builder, loc, tid, lvl, lo, hi,
                                                 reduc, tryParallel);
    // For loop condition must be a trivial condition (levels without affine
    // index expression).
    trivialLvls.push_back(tlCond.first);
  } else {
    for (auto [tl, cKind] : spConds) {
      if (isAffineIdxCond(cKind)) {
        auto [tid, lvl] = unpackTensorLevel(tl);
        bool unReduc = isAffineIdxUnRedCond(cKind);
        assert(unReduc == !depFullyReduced(tid, lvl));
        sliceDrivenInfo.emplace_back(tid, lvl, /*fullyReduced=*/!unReduc);
      } else {
        trivialLvls.push_back(tl);
      }
    }

    std::tie(l, iv) =
        emitWhileLoopOverTensorsAtLvls(builder, loc, spConds, reduc, needsUniv);
  }

  // Enter dense tensor levels.
  enterTensorsAtDenseLvls(builder, loc, dnConds, iv, sliceDrivenInfo);
  // NOTE: we can also prepare for next dim here in advance

  // Pushes the loop into stack.
  loopStack.emplace_back(trivialLvls, sliceDrivenInfo, l,
                         builder.getInsertionBlock(), iv, loopTag);
  return l;
}

Operation *LoopEmitter::enterFilterLoopOverTensorAtLvl(
    OpBuilder &builder, Location loc, TensorId tid, Level lvl,
    AffineExpr affine, MutableArrayRef<Value> reduc) {
  assert(isValidLevel(tid, lvl));
  assert(!affine.isa<AffineDimExpr>() && !isDenseDLT(lvlTypes[tid][lvl]));
  // We can not re-enter the same level.
  assert(!coords[tid][lvl]);

  // TODO: We should instead use a whileOp for filter loop to allow early
  // break when exceeding (for ordered levels).
  // TODO: There are many other potiential opportunities that we might apply in
  // the future. E.g., we could use binary search to locate positions.
  const Value step = C_IDX(1);
  const Value pLo = posits[tid][lvl];
  const Value pHi = highs[tid][lvl];
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, pLo, pHi, step, reduc);

  // In-place update on the reduction variable vector.
  assert(forOp.getNumRegionIterArgs() == reduc.size());
  for (int i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = forOp.getRegionIterArg(i);

  builder.setInsertionPointToStart(forOp.getBody());
  // The induction variable gives the position.
  const Value pos = forOp.getInductionVar();
  posits[tid][lvl] = pos;
  // Generating a load on the coordinates array yields the crd.
  const Value mem = coordinatesBuffers[tid][lvl];
  const Value crd = genIndexLoad(builder, loc, mem, pos);
  coords[tid][lvl] = crd;

  // Generate an if-condition to filter out coordinates that are not
  // equal to the result of the affine expression.
  Value expected = genAffine(builder, loc, affine);
  auto pred = CMPI(eq, crd, expected);
  SmallVector<Type> types;
  for (Value red : reduc) {
    types.push_back(red.getType());
  }

  bool hasReduc = !types.empty();
  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, types, pred, /*else*/ hasReduc);
  if (hasReduc) {
    // scf.for (a) -> v
    //  %s = scf.if (a) -> v
    //    user-generated code.
    //  else
    //    yield a
    //  yield %s
    YIELD(ifOp.getResults());
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    // On mismatch.
    YIELD(reduc);
  }
  // Set the insert point to matched branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // NOTE: we can also prepare for next lvl here in advance
  // Push the loop into stack
  loopStack.emplace_back(ArrayRef<TensorLevel>(makeTensorLevel(tid, lvl)),
                         ArrayRef<SliceLoopInfo>(), forOp,
                         builder.getInsertionBlock(), coords[tid][lvl],
                         nullptr);
  return forOp;
}

void LoopEmitter::genDenseAffineAddress(OpBuilder &builder, Location loc,
                                        TensorLevel tidLvl,
                                        AffineExpr lvlExpr) {
  auto [tid, lvl] = unpackTensorLevel(tidLvl);
  assert(isDenseDLT(lvlTypes[tid][lvl]));
  // For dense levels, the vel-coordinate also serves as the position.
  Value lvlCrd = genAffine(builder, loc, lvlExpr);
  posits[tid][lvl] = genAddress(builder, loc, tid, lvl, lvlCrd);
}

void LoopEmitter::prepareLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                             TensorId tid, Level lvl) {
  assert(isValidLevel(tid, lvl));
  const auto lvlTp = lvlTypes[tid][lvl];

  if (isDenseDLT(lvlTp))
    return;

  const Value c0 = C_IDX(0);
  const Value c1 = C_IDX(1);
  // Either the first level, or the previous level has been set.
  /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
  assert(lvl == 0 || posits[tid][lvl - 1]);
  if (isCompressedDLT(lvlTp) || isCompressedWithHiDLT(lvlTp)) {
    const Value mem = positionsBuffers[tid][lvl];

    Value pLo = lvl == 0 ? c0 : posits[tid][lvl - 1];
    if (isCompressedWithHiDLT(lvlTp))
      pLo = builder.create<arith::MulIOp>(loc, pLo, C_IDX(2));
    posits[tid][lvl] = genIndexLoad(builder, loc, mem, pLo);

    const Value pHi = ADDI(pLo, c1);
    highs[tid][lvl] = genIndexLoad(builder, loc, mem, pHi);
    return;
  }
  if (isSingletonDLT(lvlTp)) {
    const Value pLo = lvl == 0 ? c0 : posits[tid][lvl - 1];
    posits[tid][lvl] = pLo;

    // If we are coiterating non-unique levels, then use pHi=segHi;
    // otherwise use pHi=pLo+1.
    // NOTE: Just because the level is non-unique, that does not
    // guarantee that segHi is defined: because we only generate segHi
    // whenever coiterating, in order to improve code quality for the
    // non-coiterating cases.
    const auto parentSegHi = segHi[tid][lvl - 1];
    highs[tid][lvl] = (!isUniqueDLT(lvlTypes[tid][lvl - 1]) && parentSegHi)
                          ? parentSegHi
                          : ADDI(pLo, c1);
    return;
  }

  llvm_unreachable("Unrecognized level-type!");
}

void LoopEmitter::enterTensorsAtDenseLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorLvlCond> dnConds, Value iv,
    SmallVectorImpl<SliceLoopInfo> &sliceInfo) {
  for (auto [dnTidLvl, denseLoopCond] : dnConds) {
    auto [tid, lvl] = unpackTensorLevel(dnTidLvl);
    assert(isDenseDLT(lvlTypes[tid][lvl]));

    if (isAffineIdxCond(denseLoopCond)) {
      // Pushes sliced levels to build correct LoopInfo.
      bool unReduc = isAffineIdxUnRedCond(denseLoopCond);
      SliceInfo &info = sliceStack[tid].back();
      // Pushes sliced dense loop info to tell LoopEmitter how to exit it.
      sliceInfo.emplace_back(tid, lvl, /*fullyReduced=*/!unReduc);
      // FIXME: The offset and position iterator need to be adjusted when the
      // slice is strided.
      if (unReduc) {
        assert(*info.slicedOnLvl == lvl);
        unsigned depth = sliceStack[tid].back().depth;
        assert(depth >= 1);
        unsigned stride = sliceMeta[tid][lvl][depth - 1].second;
        // Update the slice information as we enter the new loop.
        info.minCrd = info.offset = MULI(iv, C_IDX(stride));
        info.isNonEmpty = constantI1(builder, loc, true);
        levelReducedDep[tid][lvl]++;
      } else {
        posits[tid][lvl] =
            genAddress(builder, loc, tid, lvl, ADDI(info.offset, iv));
      }
    } else {
      // Skips the synthetic tensor
      if (isSynTensor(tid))
        continue;
      // A dense level with trivial index expression.
      assert(dependentLvlMap[tid][lvl].empty());
      auto enc = getSparseTensorEncoding(tensors[tid].getType());
      if (enc && !isSparseOutput(tid)) {
        bool validPos = lvl == 0 || posits[tid][lvl - 1];
        if (!validPos) {
          // We might not find the pos for the sparse output tensor as it is
          // unconditionally required by the sparsification.
          assert(isOutputTensor(tid));
          continue;
        }
        posits[tid][lvl] = genAddress(builder, loc, tid, lvl, iv);
        // NOTE: we can also prepare for next lvl here in advance
      }
    }
  }
}

void LoopEmitter::exitForLoop(RewriterBase &rewriter, Location loc,
                              MutableArrayRef<Value> reduc) {
  const LoopInfo &loopInfo = loopStack.back();
  for (auto [tid, lvl, reduced] : loopInfo.sliceDrivenInfo) {
    if (!reduced) {
      SliceInfo &info = sliceStack[tid].back();
      assert(isDenseDLT(lvlTypes[tid][lvl]));
      assert(*info.slicedOnLvl == lvl);
      (void)reduced;
      // Resets slices pointers as the resolved slices are invalidated after we
      // moves forward to the next slice.
      invalidateSliceIterIdx(rewriter, loc, tid, lvl);
      info.minCrd = info.offset = info.isNonEmpty = Value();
      levelReducedDep[tid][lvl]--;
    } else {
      forwardsReducedSliceLevelTreeIt(rewriter, loc, tid, lvl,
                                      constantIndex(rewriter, loc, 1));
    }
  }
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(loopInfo.loop)) {
    if (!reduc.empty()) {
      assert(reduc.size() == forOp.getNumResults());
      rewriter.create<scf::YieldOp>(loc, reduc);
    }
    // Exit the loop.
    rewriter.setInsertionPointAfter(forOp);
    // In-place update reduction variables.
    for (unsigned i = 0, e = forOp.getResults().size(); i < e; i++)
      reduc[i] = forOp.getResult(i);
  } else {
    auto parOp = llvm::cast<scf::ParallelOp>(loopInfo.loop);
    if (!reduc.empty()) {
      assert(reduc.size() == parOp.getInitVals().size() && reduc.size() == 1);
      Operation *redExp = reduc.front().getDefiningOp();
      // Reduction expression should have no use.
      assert(redExp->getUses().empty());
      // This must be a binary operation.
      // NOTE: This is users' responsibility to ensure the operation are
      // commutative.
      assert(redExp->getNumOperands() == 2 && redExp->getNumResults() == 1);

      Value redVal = parOp.getInitVals().front();
      Value curVal;
      if (redExp->getOperand(0) == redVal)
        curVal = redExp->getOperand(1);
      else if (redExp->getOperand(1) == redVal)
        curVal = redExp->getOperand(0);
      // One of the operands must be the init value (which is also the
      // previous reduction value).
      assert(curVal);
#ifndef NDEBUG
      // The reduction expression should be the only user of the reduction val
      // inside the parallel for.
      unsigned numUsers = 0;
      for (Operation *op : redVal.getUsers()) {
        if (op->getParentOp() == parOp)
          numUsers++;
      }
      assert(numUsers == 1);
#endif // NDEBUG

      rewriter.setInsertionPointAfter(redExp);
      auto redOp = rewriter.create<scf::ReduceOp>(loc, curVal);
      // Attach to the reduction op.
      Block *redBlock = &redOp.getRegion().getBlocks().front();
      rewriter.setInsertionPointToEnd(redBlock);
      Operation *newRed = rewriter.clone(*redExp);
      // Replaces arguments of the reduction expression by using the block
      // arguments from scf.reduce.
      rewriter.updateRootInPlace(
          newRed, [&]() { newRed->setOperands(redBlock->getArguments()); });
      // Erases the out-dated reduction expression.
      rewriter.eraseOp(redExp);
      rewriter.setInsertionPointToEnd(redBlock);
      rewriter.create<scf::ReduceReturnOp>(loc, newRed->getResult(0));
    }
    rewriter.setInsertionPointAfter(parOp);
    // In-place update reduction variables.
    for (unsigned i = 0, e = parOp.getResults().size(); i < e; i++)
      reduc[i] = parOp.getResult(i);
  }

  // Finished iterating a tensor, clean up
  // We only do the clean up on for loop as while loops do not necessarily
  // finish the iteration on a sparse tensor
  for (auto [tid, lvl] : unpackTensorLevelRange(loopInfo.trivialTidLvls)) {
    // Reset to null.
    coords[tid][lvl] = Value();
    posits[tid][lvl] = Value();
    // Dense level, high is fixed.
    if (!isDenseDLT(lvlTypes[tid][lvl]))
      highs[tid][lvl] = Value();
  }
}

void LoopEmitter::forwardsReducedSliceLevelTreeIt(OpBuilder &builder,
                                                  Location loc, TensorId tid,
                                                  Level rootLvl, Value fcnt) {
  auto stt = getSparseTensorType(tensors[tid]);

  // Finds a [Lvl, leafLvl) range, and all level in between are fully reduced
  // level (but not resolved). Since we forward an iterator at higher level of
  // the tree, the subtree need to be pruned.
  Level leafLvl = rootLvl + 1;
  while (leafLvl < stt.getLvlRank() && !dependentLvlMap[tid][leafLvl].empty()) {
    assert(depFullyReduced(tid, leafLvl));
    leafLvl++;
  }

  Level curLvl = rootLvl + 1;
  // Prunes all denses subtree.
  while (curLvl < leafLvl && isDenseDLT(lvlTypes[tid][curLvl])) {
    // One step forward in parent level results in forwarding `slice.size` step
    // in child dense level.
    auto [size, stride] = sliceMeta[tid][curLvl].back();
    assert(stride == 1 && "Not yet implemented");
    fcnt = MULI(size, fcnt);
    curLvl++;
  }

  Value nxPosPtr = nullptr;
  if (curLvl < leafLvl) {
    assert(!isDenseDLT(lvlTypes[tid][curLvl]));
    // The first compressed level, setting up the position pointer for it.
    Value sPosBuf = slicePosBuffer[tid][curLvl].back();
    // One step forwards in the parent level result in forwarding one `segment`
    // (kSliceIterWidth) in the child sparse level.
    Value fPosPtr = MULI(fcnt, C_IDX(kSliceIterWidth));     // forward ptr
    Value pPosPtr = loadSlicePosPtr(builder, loc, sPosBuf); // previous ptr
    Value cPosPtr = ADDI(fPosPtr, pPosPtr);                 // current ptr
    updateSlicePosPtr(builder, loc, sPosBuf, cPosPtr);
    // Loads the position pointer start for next level.
    nxPosPtr = loadSliceNextPosPtrStart(builder, loc, sPosBuf, cPosPtr);
    curLvl++;
  }

  // TODO: This is not always needed, but we did it unconditionally for now for
  // simplicity.
  // It is only needed when `curLvl` is forwarded without traversing its child
  // level (e.g., the level is in a conjunctive lattices and got pruned), such
  // that the position pointer is not forwarded inside the loop.
  for (; curLvl < leafLvl; curLvl++) {
    assert(nxPosPtr);
    if (!isDenseDLT(lvlTypes[tid][curLvl])) {
      nxPosPtr = MULI(nxPosPtr, C_IDX(kSliceIterWidth));
      Value sPosBuf = slicePosBuffer[tid][curLvl].back();
      updateSlicePosPtr(builder, loc, sPosBuf, nxPosPtr);
      nxPosPtr = loadSliceNextPosPtrStart(builder, loc, sPosBuf, nxPosPtr);
    }
  }
}

void LoopEmitter::exitWhileLoop(OpBuilder &builder, Location loc,
                                MutableArrayRef<Value> reduc) {
  const LoopInfo &loopInfo = loopStack.back();
  auto whileOp = llvm::cast<scf::WhileOp>(loopInfo.loop);
  Value iv = loopInfo.iv;
  Value one = C_IDX(1);

  // Finalize the induction. Note that the induction could be performed
  // in the individual if-branches to avoid re-evaluating the conditions.
  // However, that would result in a rather elaborate forest of yield
  // instructions during code generation. Moreover, performing the induction
  // after the if-statements more closely resembles code generated by TACO.
  unsigned o = 0;
  SmallVector<Value> operands;
  unsigned delta = 0;
  for (auto [tid, lvl, resolved] : loopInfo.sliceDrivenInfo) {
    // TODO: handle dense.
    assert(isCompressedDLT(lvlTypes[tid][lvl]));
    levelReducedDep[tid][lvl]--;
    if (!resolved) {
      // TODO: support coiterating multiple slices
      assert(loopInfo.trivialTidLvls.empty() &&
             loopInfo.sliceDrivenInfo.size() == 1);
      auto [nxNonEmpty, nxMinCrd, nxAbsOffset] =
          genSliceNextInduction(builder, loc, tid, lvl);
      // Update while loop induction operands.
      operands.push_back(nxNonEmpty);
      operands.push_back(nxMinCrd);
      operands.push_back(nxAbsOffset);

      // Update the slice stack.
      SliceInfo &info = sliceStack[tid].back();
      info.isNonEmpty = whileOp.getResult(o++);
      info.minCrd = whileOp.getResult(o++);
      info.offset = whileOp.getResult(o++);
      continue;
    }

    Value forwarded = nullptr;
    if (loopInfo.trivialTidLvls.empty() &&
        loopInfo.sliceDrivenInfo.size() == 1) {
      // Forwards the position iterator.
      operands.push_back(ADDI(posits[tid][lvl], one));
      forwarded = constantI1(builder, loc, true);
    } else {
      const Value pos = posits[tid][lvl];
      const Value nxPos = ADDI(posits[tid][lvl], one);
      forwarded = CMPI(eq, coords[tid][lvl], iv);
      operands.push_back(SELECT(forwarded, nxPos, pos));
    }
    {
      OpBuilder::InsertionGuard guard(builder);
      auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, forwarded,
                                            /*else=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      forwardsReducedSliceLevelTreeIt(builder, loc, tid, lvl, one);
    }
    // The coordinate is invalid now.
    coords[tid][lvl] = nullptr;

    // Update the position iterator as we exit the while loop.
    posits[tid][lvl] = whileOp->getResult(o++);
  };

  for (auto [tid, lvl] : unpackTensorLevelRange(loopInfo.trivialTidLvls)) {
    const auto lvlTp = lvlTypes[tid][lvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp) ||
        isCompressedWithHiDLT(lvlTp)) {
      const Value crd = coords[tid][lvl];
      const Value pos = posits[tid][lvl];
      Value cmp = CMPI(eq, crd, iv);
      // If the loop contains a coiteration with non-unique level, we fast
      // forward all the duplicated coords by setting the position to the
      // segment high.
      Value add =
          !isUniqueDLT(lvlTypes[tid][lvl]) ? segHi[tid][lvl] : ADDI(pos, one);

      operands.push_back(SELECT(cmp, add, pos));
      // Following loops continue iteration from the break point of the
      // current while loop.
      const Value newPos = whileOp->getResult(o++);
      // We need to define a new local variable for `tid` to avoid
      // warnings about "captured structured bindings are a C++20 extension".
      // FIXME(wrengr): define a helper function to capture this idiom!
      const TensorId newTid = tid;
      posits[newTid][lvl] = newPos;

      // The coordinate is invalid now.
      coords[tid][lvl] = nullptr;
      // The segment high is invalid now.
      segHi[tid][lvl] = nullptr;
      // highs remains unchanged.
    }
  }

  // Reduction value from users.
  for (auto &i : reduc) {
    operands.push_back(i);
    // In place update reduction variable.
    i = whileOp->getResult(o++);
  }

  // An (optional) universal index.
  if (operands.size() + delta < whileOp.getNumResults()) {
    assert(operands.size() + delta + 1 == whileOp.getNumResults());
    // The last one is the universial index.
    operands.push_back(ADDI(iv, one));
    // update the loop starting point of current loop sequence
    loopSeqStack.back().first = whileOp->getResult(o++);
  }

  assert(o == operands.size() + delta);
  if (!operands.empty())
    YIELD(operands);

  builder.setInsertionPointAfter(whileOp);
}

void LoopEmitter::exitCurrentLoop(RewriterBase &rewriter, Location loc,
                                  MutableArrayRef<Value> reduc) {
  // Clean up the values, it would help use to discover potential bug at a
  // earlier stage (instead of silently using a wrong value).
  const LoopInfo &loopInfo = loopStack.back();

  // Sets the insertion point to the right position.
  rewriter.setInsertionPointToEnd(loopInfo.userCodeBlock);
  if (!loopInfo.userCodeBlock->empty() &&
      llvm::isa<scf::YieldOp>(&loopInfo.userCodeBlock->back())) {
    // scf::While/For inserts an implicit yield op when there is no loop
    // iter args. In this case, we need to insert the code before the yield.
    assert(loopInfo.userCodeBlock->back().getNumResults() == 0);
    rewriter.setInsertionPoint(&loopInfo.userCodeBlock->back());
  }

  if (llvm::isa<scf::WhileOp>(loopInfo.loop)) {
    exitWhileLoop(rewriter, loc, reduc);
  } else {
    exitForLoop(rewriter, loc, reduc);
  }

  assert(loopStack.size() == loopSeqStack.size());
  loopStack.pop_back();
}

//===----------------------------------------------------------------------===//
// Slice-driven loop related methods.
//===----------------------------------------------------------------------===//

unsigned LoopEmitter::remDepOnLevel(TensorId tid, Level lvl) const {
  unsigned totalDependencies = dependentLvlMap[tid][lvl].size();
  if (totalDependencies != 0) {
    assert(totalDependencies >= 2);
    return totalDependencies - levelReducedDep[tid][lvl];
  }
  return totalDependencies;
}

const LoopEmitter::SliceInfo &LoopEmitter::getMostRecentSliceOnLvl(TensorId tid,
                                                                   Level lvl) {
  // Finds the most-recent slice using a reverse iteration.
  for (auto it = sliceStack[tid].rbegin(), ie = sliceStack[tid].rend(); it < ie;
       it++) {
    if (it->slicedOnLvl == lvl) { // the level matched
      return *it;
    }
  }
  llvm_unreachable("Failed to find sliceInfo");
}

// Generates a while loop to iterate over a slice sparse level as follows.
//
// while(coords[loopLo] < offset + size) {
//   body_builder
//   loopLo ++;
// }
std::pair<Operation *, ValueRange> LoopEmitter::genSliceLvlTraverseLoop(
    OpBuilder &builder, Location loc, Value posLo, Value posHi, Value offset,
    Value size, TensorId tid, Level lvl, ValueRange userReduc,
    LoopBodyBuilder bodyBuilder) {
  Value c1 = C_IDX(1);
  auto [sliceSz, stride] = sliceMeta[tid][lvl].back();
  assert(stride == 1 && "Not yet implemented");
  Value sliceHi = ADDI(offset, sliceSz);

  SmallVector<Value> reduc{posLo}; // loop lower bounds
  const unsigned numMetaReduc = reduc.size();

  // Append user required reduction value.
  reduc.append(userReduc.begin(), userReduc.end());
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(
      loc, ValueRange(reduc).getTypes(), reduc,
      /*beforeBuilder=*/
      [this, posHi, sliceHi, tid, lvl](OpBuilder &builder, Location loc,
                                       ValueRange args) {
        Value cond = genSparseReducedAffineCond(builder, loc,
                                                coordinatesBuffers[tid][lvl],
                                                sliceHi, args[0], posHi);
        // continue if not yet break nor out of bound.
        builder.create<scf::ConditionOp>(loc, cond, args);
      },
      /*afterBuilder=*/
      [c1, numMetaReduc, bodyBuilder](OpBuilder &builder, Location loc,
                                      ValueRange args) {
        Value iv = args[0];
        TypeRange types = args.drop_front(numMetaReduc).getTypes();
        // The coordinate must be in bound as guaranteed by the loop
        // condition. We generate a fake if operation here only to hide the
        // extra loop induction variables maintained by us from users, which
        // will be removed by later optimization pass.
        auto ifOp = builder.create<scf::IfOp>(loc, types,
                                              constantI1(builder, loc, true),
                                              /*withElseBlock=*/!types.empty());
        {
          // 2 reduction variable maintained by us.
          SmallVector<Value> ifRet = args.drop_front(numMetaReduc);
          assert(ifRet.size() == args.size() - 1);

          OpBuilder::InsertionGuard guard(builder);
          // If coord >= sliceHi.
          if (!ifRet.empty()) {
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            YIELD(ifRet);
          }

          // If coord < sliceHi.
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          // Delegates to users' callback.
          bodyBuilder(builder, loc, iv, ifRet);
        }
        // Marks this speical ifOp to avoid sparisification finalizing it.
        ifOp->setAttr(getLoopEmitterLoopAttrName(),
                      StringAttr::get(builder.getContext(), "slice"));
        // Insertion point restored to after ifOp.
        SmallVector<Value> yields;
        // Increase induction variable.
        yields.push_back(ADDI(iv, c1));
        yields.append(ifOp.getResults().begin(), ifOp.getResults().end());
        YIELD(yields);
      });

  builder.setInsertionPointAfter(whileOp);
  return std::make_pair(whileOp, whileOp.getResults().drop_front(numMetaReduc));
}

// Generates a loop nest that traverse all the unresolved levels in between.
//
// for(int i = 0; i < slicePos.size(); i+=2) {
//   loopLo = slicePos[i];
//   loopHi = slicePos[i + 1];
//
//   // Then the same loop generated by genSliceLvlTraverse above.
//   while (loopLo < loopHI) {
//     if (pos[loopLo] < sliceHi) {
//       bodyBuilder();
//     } else {
//       break;
//     }
//     loopLo ++;
//   }
// }
ValueRange LoopEmitter::genUnResolvedSliceTreeTraverse(
    OpBuilder &builder, Location loc, TensorId tid,
    ArrayRef<const SliceInfo *> unResLvls,
    std::optional<std::pair<TensorId, Level>> firstResLvl, ValueRange userReduc,
    LoopBodyBuilder bodyBuilder) {

  Value c0 = C_IDX(0), c1 = C_IDX(1), c2 = C_IDX(2);
  Value pos = c0;
  OpBuilder::InsertPoint ip;
  SmallVector<Value> innerArgs(userReduc.begin(), userReduc.end());
  scf::ForOp outerMost = nullptr; // the outtermost loop.

  // Wraps body builder and inserts a extra counting instruction at the end.
  auto wrapped = [bodyBuilder](OpBuilder &builder, Location loc, Value iv,
                               MutableArrayRef<Value> reduc) {
    bodyBuilder(builder, loc, iv, reduc.drop_back());
    // Increments the counter.
    reduc.back() = ADDI(reduc.back(), C_IDX(1));
  };

  // FIXME: Need special handling when the previous unresolved slice is strided:
  // We probably need to filter out coordinates that is not on stride.
  if (firstResLvl.has_value()) {
    // Overwrite position when the first level is fully resolved.
    pos = posits[firstResLvl->first][firstResLvl->second];
    ip = builder.saveInsertionPoint();
  } else {
    const SliceInfo &frontSlice = *unResLvls.back();
    Level firstLvl = *frontSlice.slicedOnLvl;
    if (!lvlFullyResolved(tid, firstLvl)) {
      if (isCompressedDLT(lvlTypes[tid][firstLvl])) {
        // An extra counter that tracks how many segments are there in the child
        // compressed level.
        innerArgs.push_back(c0);
        // Overrides the user-provided builder.
        bodyBuilder = wrapped;
        unsigned depth = frontSlice.depth - 1;
        Value offset = frontSlice.offset;
        Value sPtrBuf = slicePosBuffer[tid][firstLvl][depth];
        Value mSz = genIndexLoad(builder, loc, sPtrBuf, c0); // memSize
        outerMost = builder.create<scf::ForOp>(
            loc, c2, mSz, C_IDX(kSliceIterWidth), innerArgs,
            [this, c1, c2, tid, firstLvl, offset, sPtrBuf, &ip, &pos,
             &innerArgs](OpBuilder &builder, Location loc, Value iv,
                         ValueRange iterArgs) {
              // generate traversal for each level.
              Value loopLo = genIndexLoad(builder, loc, sPtrBuf, iv);
              Value loopHi = genIndexLoad(builder, loc, sPtrBuf, ADDI(iv, c1));
              // We need to remember the starting index for next level's
              // position, because slice-driven loop breaks the level into
              // non-consecutive segments.
              builder.create<memref::StoreOp>(loc, iterArgs.back(), sPtrBuf,
                                              ADDI(iv, c2).getResult());

              auto [size, stride] = sliceMeta[tid][firstLvl].back();
              assert(stride == 1 && "Not yet implemented");
              ValueRange itArgs =
                  genSliceLvlTraverseLoop(
                      builder, loc, loopLo, loopHi, offset, size, tid, firstLvl,
                      iterArgs,
                      [&](OpBuilder &builder, Location, Value iv,
                          MutableArrayRef<Value> reduc) {
                        ip = builder.saveInsertionPoint();
                        pos = iv;
                        innerArgs.assign(reduc.begin(), reduc.end());
                      })
                      .second;
              YIELD(itArgs);
            });
      } else if (isDenseDLT(lvlTypes[tid][firstLvl])) {
        assert(firstLvl == 0); // This must be the first level.
        Value lb = frontSlice.offset;
        auto [sliceSz, stride] =
            sliceMeta[tid][*frontSlice.slicedOnLvl][frontSlice.depth];
        assert(stride == 1 && "Not yet implemented");
        Value ub = ADDI(lb, sliceSz);
        outerMost = builder.create<scf::ForOp>(
            loc, lb, ub, c1, innerArgs,
            [&](OpBuilder &builder, Location loc, Value iv,
                ValueRange iterArgs) {
              ip = builder.saveInsertionPoint();
              pos = iv;
              innerArgs.assign(iterArgs.begin(), iterArgs.end());
            });
      }
      // We generated the loop for the first slice above, now remove it.
      unResLvls = unResLvls.drop_back();
    }
  }
  // Reset the insertion point into the loop body.
  builder.restoreInsertionPoint(ip);
  if (!unResLvls.empty()) {
    // Fills in dense slices levels in between.
    SmallVector<Value> lbs, ubs, steps, lvlSzs;
    for (const SliceInfo *slice : llvm::reverse(unResLvls)) {
      Level sliceLvl = *slice->slicedOnLvl;
      assert(isDenseDLT(lvlTypes[tid][sliceLvl]));
      Value offset = slice->offset;
      auto [sliceSz, stride] = sliceMeta[tid][sliceLvl][slice->depth];
      assert(stride == 1 && "Not yet implemented");
      lbs.push_back(offset);
      ubs.push_back(ADDI(offset, sliceSz));
      steps.push_back(c1);
      lvlSzs.push_back(lvlSizes[tid][sliceLvl]);
    }
    auto denseNest =
        scf::buildLoopNest(builder, loc, lbs, ubs, steps, innerArgs,
                           [&innerArgs, &lvlSzs, &pos, bodyBuilder](
                               OpBuilder &builder, Location loc, ValueRange ivs,
                               ValueRange iterArgs) -> scf::ValueVector {
                             for (auto em : llvm::enumerate(ivs)) {
                               // Linearizes postion: pos = (pos * lvlsize) +
                               // iv;
                               pos = MULI(pos, lvlSzs[em.index()]);
                               pos = ADDI(pos, em.value());
                             }
                             innerArgs.assign(iterArgs.begin(), iterArgs.end());
                             // Generates user request loop body.
                             bodyBuilder(builder, loc, pos, innerArgs);
                             return innerArgs;
                           });

    if (!outerMost) {
      // If the outermost loop has not been set, this is the outermost loop.
      outerMost = denseNest.loops.front();
    } else {
      // Otherwise we need to generate yield operations to link the SSA chain.
      YIELD(denseNest.results);
    }
  } else {
    assert(outerMost);
    // Generates user request loop body.
    bodyBuilder(builder, loc, pos, innerArgs);
    YIELD(innerArgs);
  }
  assert(outerMost);
  // Insert after current while operation.
  builder.setInsertionPointAfter(outerMost);
  return outerMost.getResults();
}

void LoopEmitter::genResolvedSliceBegin(OpBuilder &builder, Location loc,
                                        TensorId tid, Level lvl) {
  Value c0 = C_IDX(0), c1 = C_IDX(1), c2 = C_IDX(2), c3 = C_IDX(3),
        c4 = C_IDX(4);
  if (isDenseDLT(lvlTypes[tid][lvl])) {
    // Dense slice begin is trivial.
    sliceStack[tid].emplace_back(/*minCoord=*/c0, /*offset=*/c0,
                                 /*nonEmpty=*/constantI1(builder, loc, true),
                                 lvl, /*depth=*/1);
    return;
  }
  auto [nxSz, stride] = sliceMeta[tid][lvl][1];
  assert(stride == 1 && "Not yet implemented");
  Value sPtrBuf = slicePosBuffer[tid][lvl][0];
  Value pHi, pLo;
  if (lvl == 0) {
    pLo = c0;
    pHi = genIndexLoad(builder, loc, positionsBuffers[tid][0], c1);
  } else {
    pLo = genIndexLoad(builder, loc, positionsBuffers[tid][lvl],
                       posits[tid][lvl - 1]);
    pHi = genIndexLoad(builder, loc, positionsBuffers[tid][lvl],
                       ADDI(posits[tid][lvl - 1], c1));
  }
  // Fills out pIdxBuffer[tid][lvl][0] with [/*memSize =*/4, 0, pLo, pHi]
  builder.create<memref::StoreOp>(loc, c4, sPtrBuf, c0);  // memSize = 4
  builder.create<memref::StoreOp>(loc, c0, sPtrBuf, c1);  // index = 0
  builder.create<memref::StoreOp>(loc, pLo, sPtrBuf, c2); // pLo
  builder.create<memref::StoreOp>(loc, pHi, sPtrBuf, c3); // pHi

  // This is an non empty tensor if pLo < pHi.
  Value isNonEmpty = CMPI(ult, pLo, pHi);
  // The minimal coord must be at the first on ordered level.
  // FIXME: Technically we should load the coord only when the slice is
  // nonempty. though we assume that even on empty sparse tensors, a non-empty
  // ptr/idx buffer is allocated for each level so it would not cause OOB to
  // avoid generating a ifOp here.
  Value minCrd = genIndexLoad(builder, loc, coordinatesBuffers[tid][lvl], pLo);

  // FIXME: We need the relative offset related to the base slice.
  Value absOffset = offsetFromMinCoord(builder, loc, minCrd, nxSz, isNonEmpty);
  sliceStack[tid].emplace_back(minCrd, absOffset, isNonEmpty, lvl,
                               /*depth=*/1);
}

// Fills in the slicePosBuffer before slice-driven loop begin.
// TODO: it can only handle all compressed tensors.
//
// // Loop generated by `genUnResolvedSliceTreeTraverse`
// for(int i = 0; i < slicePos.size(); i+=2) {
//   loopLo = slicePos[i];
//   loopHi = slicePos[i + 1];
//   minCrd = max;
//   while (loopLo < loopHi) {
//     if (pos[loopLo] < sliceHi) {
//       // bodyBuilder
//       slicePos[tid].push_back(pos[loopLo]);
//       slicePos[tid].push_back(pos[loopLo + 1]);
//       minCrd = min(minCrd, crd[pos[loopLo]]);
//     } else {
//       break;
//     }
//     loopLo ++;
//   }
// }
void LoopEmitter::genUnResolvedSliceBegin(OpBuilder &builder, Location loc,
                                          TensorId tid, Level lvl) {
  Value c0 = C_IDX(0), c1 = C_IDX(1), c2 = C_IDX(2);
  unsigned depth = levelReducedDep[tid][lvl];
  // The remaining slice size after reduction.
  Value remSz = sliceMeta[tid][lvl][depth + 1].first;
  // Dense slice begin is trivial
  if (isDenseDLT(lvlTypes[tid][lvl])) {
    sliceStack[tid].emplace_back(c0, c0, constantI1(builder, loc, false), lvl,
                                 depth + 1);
    return;
  }

  assert(isCompressedDLT(lvlTypes[tid][lvl]));
  // Unhandled Cases:
  //
  // 1st, lvl = prevSlicedLvl, i.e., t[d0 + d1 + d2,...] (more than one
  // variable need to be reduced on the same level).
  //
  // 2nd, lvl > prevSliceLvl + 1, i.e., t[..., d2, d3 + d4] (having a
  // simple dim expression in between).
  assert(lvl == *sliceStack[tid].back().slicedOnLvl + 1);

  // Check slice stack integrity.
  assert(slicePosBuffer[tid][lvl - 1].size() == sliceStack[tid].back().depth);

  SmallVector<const SliceInfo *> unResSlices;
  std::optional<std::pair<TensorId, Level>> firstResLvl;
  for (Level curLvl = lvl; curLvl >= 1; curLvl--) {
    Level prevLvl = curLvl - 1;
    if (lvlFullyResolved(tid, prevLvl)) {
      firstResLvl = std::make_pair(tid, prevLvl);
      break;
    }
    unResSlices.push_back(&getMostRecentSliceOnLvl(tid, prevLvl));
    if (!isDenseDLT(lvlTypes[tid][prevLvl])) {
      break;
    }
  }

  assert(!unResSlices.empty() &&
         !lvlFullyResolved(tid, *unResSlices.front()->slicedOnLvl));

  Value sPtrBuf = slicePosBuffer[tid][lvl].back();
  SmallVector<Value, 3> reduc = {
      constantI1(builder, loc, false), // isNonEmpty
      lvlSizes[tid][lvl],              // minCoord
      c2,                              // memSize
  };

  ValueRange result = genUnResolvedSliceTreeTraverse(
      builder, loc, tid, unResSlices, firstResLvl, reduc,
      [this, c1, tid, lvl, sPtrBuf](OpBuilder &builder, Location loc, Value iv,
                                    MutableArrayRef<Value> reduc) {
        Value &nonEmpty = reduc[0];
        Value &minCrd = reduc[1];
        Value &curMemSz = reduc[2];

        Value pHi = ADDI(iv, c1);
        Value sPLo = genIndexLoad(builder, loc, positionsBuffers[tid][lvl], iv);
        Value sPHi =
            genIndexLoad(builder, loc, positionsBuffers[tid][lvl], pHi);

        // isNonEmpty = isNonEmpty || lvlNonEmpty, i.e., as long as there is
        // one non-empty lvl, the slice is non-empty.
        Value lvlNonEmpty = CMPI(ult, sPLo, sPHi);
        nonEmpty = builder.create<arith::OrIOp>(loc, lvlNonEmpty, nonEmpty);

        // Update the minimum coordinate.
        auto ifNonEmpty = builder.create<scf::IfOp>(loc, builder.getIndexType(),
                                                    lvlNonEmpty, true);
        {
          // Generate Code as follows.
          //
          // if (nonEmpty) {
          //   minCrd = min(minCrd, crd[pos[pLo]]);
          // }
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(ifNonEmpty.thenBlock());
          Value curC =
              genIndexLoad(builder, loc, coordinatesBuffers[tid][lvl], sPLo);
          Value isSmaller = CMPI(ult, curC, minCrd);
          Value newMin = SELECT(isSmaller, curC, minCrd);
          YIELD(newMin);
          builder.setInsertionPointToStart(ifNonEmpty.elseBlock());
          YIELD(minCrd);
        }
        minCrd = ifNonEmpty.getResult(0);
        builder.create<memref::StoreOp>(loc, sPLo, sPtrBuf, curMemSz);
        Value nxtMemSize = ADDI(curMemSz, c1);
        builder.create<memref::StoreOp>(loc, sPHi, sPtrBuf, nxtMemSize);
        // curMemSize += kSliceIterWidth
        curMemSz = ADDI(curMemSz, C_IDX(kSliceIterWidth));
      });

  Value isNonEmpty = result[0];
  Value minCrd = result[1];
  // Two metadata [memSize, idx].
  // TODO: Can use an SSA value for these two metadata
  builder.create<memref::StoreOp>(loc, result[2], sPtrBuf, c0);
  builder.create<memref::StoreOp>(loc, c0, sPtrBuf, c1);
  // FIXME: we need the relative offset related to the base slice.
  Value absOffset = offsetFromMinCoord(builder, loc, minCrd, remSz, isNonEmpty);
  sliceStack[tid].emplace_back(minCrd, absOffset, isNonEmpty, lvl, depth + 1);
}

bool LoopEmitter::genSliceBegin(OpBuilder &builder, Location loc, TensorId tid,
                                Level lvl) {
  Value c1 = C_IDX(1), c2 = C_IDX(2);

  if (depFullyReduced(tid, lvl)) {
    // Do not need to prepare for slice driven loop on dense level after it is
    // fully reduced.
    if (isDenseDLT(lvlTypes[tid][lvl]))
      return true;
    // If constraints on the tensor is fully resolved. We do not need to
    // generates slice begin any more, instead we fall back to TACO-based
    // algorithm to (co)iterates over the slice.
    Value pLoPtr =
        loadSlicePosPtr(builder, loc, slicePosBuffer[tid][lvl].back());
    pLoPtr = ADDI(pLoPtr, c2);
    Value pHiPtr = ADDI(pLoPtr, c1);
    posits[tid][lvl] =
        genIndexLoad(builder, loc, slicePosBuffer[tid][lvl].back(), pLoPtr);
    highs[tid][lvl] =
        genIndexLoad(builder, loc, slicePosBuffer[tid][lvl].back(), pHiPtr);
    return true;
  }

  // Only when the level is sorted, the next-non-empty slice can be computed
  // efficiently.
  const DimLevelType lvlType = lvlTypes[tid][lvl];
  assert(isOrderedDLT(lvlType));
  if (isSingletonDLT(lvlType)) {
    llvm_unreachable("TODO: dense level should be easy to support, while "
                     "singleton level requres more efforts");
  }

  assert(!dependentLvlMap[tid][lvl].empty());
  assert(!sliceStack[tid].empty());

  const SliceInfo &sliceInfo = sliceStack[tid].back();
  auto baseEnc = getSparseTensorEncoding(tensors[tid].getType());
  if (baseEnc.isSlice())
    llvm_unreachable("TODO: not yet implemented");

  // Generate caches required to fast compute next-non-empty slices with
  // increasing offset for slice-base loop.
  // We do not need cache for dense levels.
  if (slicePosBuffer[tid][lvl][0] == nullptr && !isDenseDLT(lvlType)) {
    OpBuilder::InsertionGuard guard(builder);
    // The buffer can be reused, and the size is loop invariant: it only
    // depends on the iteration graph's toposort.
    builder.setInsertionPointAfter(localInsertPos);
    Value bufSize = C_IDX(1);
    Value c2 = C_IDX(2);
    // Accumlates the size required to cache the pLo for the slice.
    // E.g., if we want to cache the pIdx for slice<d0xd1xf64> on the second
    // level. We at most need to a memref<d0xindex>.
    // NOTE: this is apperantly an over-approximation when the previous
    // level is compressed, and we can compute a precise memory size
    // inside the loops. But that would also requires us to allocate/free
    // memorys in loops.
    // TODO: Maybe using allocaScopeOp inside the loop to resolve the issue?
    for (Level curLevel = lvl;
         curLevel >= 1 && !lvlFullyResolved(tid, curLevel - 1); curLevel--) {
      // We only handle cases when all the previously unresolved levels are
      // fully reduced.
      assert(depFullyReduced(tid, curLevel - 1));
      assert(!sliceMeta[tid][curLevel - 1].empty());
      auto [sz, stride] = sliceMeta[tid][curLevel - 1].back();
      assert(stride == 1 && "Not yet implemented");
      bufSize = MULI(bufSize, sz);
    }
    // For a triple of [pLo, pHi, pPtr]. Note that we can not compress pHi
    // because slice creates segments in the index buffer so that the pHi for
    // the current level is no longer the pLo for the next level.
    bufSize = MULI(bufSize, C_IDX(kSliceIterWidth));
    // Additional two metadata {memSize, idx} at head.
    bufSize = ADDI(bufSize, c2);
    for (Value &cache : slicePosBuffer[tid][lvl])
      cache = genAlloca(builder, loc, bufSize, builder.getIndexType());
  }

  if (sliceInfo.isInitialTensor() ||
      (lvl >= 1 && lvlFullyResolved(tid, lvl - 1))) {
    // First level or previous level has been full resolved.
    genResolvedSliceBegin(builder, loc, tid, lvl);
  } else {
    // The previous level has not been full resolved.
    genUnResolvedSliceBegin(builder, loc, tid, lvl);
  }
  return false;
}

void LoopEmitter::invalidateSliceIterIdx(OpBuilder &builder, Location loc,
                                         TensorId tid, Level lvl) {
  for (unsigned i = 0; i <= lvl; i++) {
    if (!isDenseDLT(lvlTypes[tid][i]) && !dependentLvlMap[tid][i].empty()) {
      updateSlicePosPtr(builder, loc, slicePosBuffer[tid][i].back(), C_IDX(0));
    }
  }
}

std::tuple<Value, Value, Value>
LoopEmitter::genSliceNextInduction(OpBuilder &builder, Location loc,
                                   TensorId tid, Level lvl) {
  if (!isCompressedDLT(lvlTypes[tid][lvl]))
    llvm_unreachable("TODO");

  // else generate code to compute next non empty slice.
  Value c0 = C_IDX(0), c1 = C_IDX(1), c2 = C_IDX(2);

  SliceInfo &info = sliceStack[tid].back();
  assert(info.slicedOnLvl == lvl);
  //
  // We forward to the next non empty slice by
  // if (minCrd > offset) {
  //   offset += 1
  // } else {
  //    minCrd = nextMinInSlice();
  //    offset = minCrd - size + 1;
  // }
  //
  // if (offset + size > parents.size)
  //   isNonEmpty = false;
  //
  Value absOffset = info.offset;
  // Resets slices pointers as the resolved slices are invalidated after we
  // moves forward to the next slice.
  invalidateSliceIterIdx(builder, loc, tid, lvl);

  SmallVector<Value, 3> reduc = {info.minCrd, info.isNonEmpty, absOffset};
  Value sPtrBuf = slicePosBuffer[tid][lvl][info.depth - 1];
  Value fastPathP = CMPI(ugt, info.minCrd, absOffset);
  auto ifOp = builder.create<scf::IfOp>(loc, ValueRange(reduc).getTypes(),
                                        fastPathP, true);
  {
    OpBuilder::InsertionGuard guard(builder);
    // Take the fast path
    // if (minCrd > offset) {
    //   return offset += 1
    // }
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    reduc[2] = ADDI(absOffset, c1);
    // Yield offset + 1.
    YIELD(reduc);

    // else /*minCrd == offset*/ {
    //    for (i = 0; i < slicePos.size(); i+=kSliceIterWidth) {
    //       if (crd[pos[slicePos[i]]] == minCrd) {
    //          slicePos[i]++;
    //       }
    //       minCrd=min(minCrd, crd[pos[slicePos[i]]]);
    //    }
    //    offset = minCrd - size + 1;
    // }
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    reduc[2] = absOffset; // restore value.
    Value pSt = c2;       // pointer starting index
    Value mSz = genIndexLoad(builder, loc, sPtrBuf, c0); // memSize
    reduc[0] = lvlSizes[tid][lvl];                       // next min coord
    reduc[1] = constantI1(builder, loc, false);          // isNonEmpty
    auto loopArgs = static_cast<ValueRange>(reduc).drop_back();
    auto forOp = scf::buildLoopNest(
        builder, loc, pSt, mSz, C_IDX(kSliceIterWidth), loopArgs,
        [this, tid, lvl, c1, sPtrBuf,
         &info](OpBuilder &builder, Location loc, ValueRange ivs,
                ValueRange iterArgs) -> scf::ValueVector {
          Value curMinCrd = iterArgs[0];
          Value isNonEmpty = iterArgs[1];

          Type idxTp = builder.getIndexType();
          Value pLo = genIndexLoad(builder, loc, sPtrBuf, ivs.front());
          Value pHi =
              genIndexLoad(builder, loc, sPtrBuf, ADDI(ivs.front(), c1));
          //
          // if (pLo < pHi) // Only loads when inbound.
          //   coord = load[pLo]
          //   if coord == minCrd
          //     pLo += 1
          //
          // if (pLo < pHi)
          //   curMinCrd = min(curMinCrd, load[pLo])
          //
          Value pred = CMPI(ult, pLo, pHi);
          auto advPLo = builder.create<scf::IfOp>(loc, idxTp, pred, true);
          /* if pLo < pHi */ {
            builder.setInsertionPointToStart(&advPLo.getThenRegion().front());
            // coord = load[pLo]
            Value coord =
                genIndexLoad(builder, loc, coordinatesBuffers[tid][lvl], pLo);
            Value pred = CMPI(eq, coord, info.minCrd);
            auto ifEqual = builder.create<scf::IfOp>(loc, idxTp, pred, true);
            /* if coord == minCrd */ {
              builder.setInsertionPointToStart(
                  &ifEqual.getThenRegion().front());
              Value newPlo = ADDI(pLo, c1);
              // Updates the cache.
              builder.create<memref::StoreOp>(loc, newPlo, sPtrBuf,
                                              ivs.front());
              YIELD(newPlo);
            }
            /* else coord != minCrd */ {
              builder.setInsertionPointToStart(
                  &ifEqual.getElseRegion().front());
              YIELD(pLo);
            }
            builder.setInsertionPointAfter(ifEqual);
            YIELD(ifEqual.getResults());
          }
          /* else pLo >= pHi */ {
            builder.setInsertionPointToStart(&advPLo.getElseRegion().front());
            YIELD(pLo);
          }

          builder.setInsertionPointAfter(advPLo);
          pLo = advPLo.getResult(0);
          Value lvlNonEmpty = CMPI(ult, pLo, pHi);
          // Update minCrds
          auto newMin =
              builder.create<scf::IfOp>(loc, idxTp, lvlNonEmpty, true);
          builder.setInsertionPointToStart(&newMin.getThenRegion().front());
          YIELD(genIndexLoad(builder, loc, coordinatesBuffers[tid][lvl], pLo));

          builder.setInsertionPointToStart(&newMin.getElseRegion().front());
          YIELD(curMinCrd);
          builder.setInsertionPointAfter(newMin);

          // isNonEmpty = isNonEmpty || lvlNonEmpty
          isNonEmpty =
              builder.create<arith::OrIOp>(loc, lvlNonEmpty, isNonEmpty);
          curMinCrd = builder.create<arith::SelectOp>(
              loc, CMPI(ult, newMin.getResult(0), curMinCrd),
              newMin.getResult(0), curMinCrd);
          return {curMinCrd, isNonEmpty};
        });

    builder.setInsertionPointAfter(forOp.loops.front());
    // minOffset = minCrd + 1 >= size ? minCrd + 1 - size : c0
    Value tmp = ADDI(forOp.results.front(), c1);
    auto [size, stride] = sliceMeta[tid][lvl][info.depth];
    assert(stride == 1 && "Not yet implemented");
    Value minOffset = SUBI(tmp, size);
    Value p = CMPI(uge, tmp, size);
    minOffset = SELECT(p, minOffset, c0);

    SmallVector<Value, 3> yields;
    yields.assign(forOp.results.begin(), forOp.results.end());
    yields.push_back(minOffset);
    YIELD(yields);
  }

  Value nextMinCrd = ifOp.getResults()[0];
  Value nextNonEmpty = ifOp.getResults()[1];

  // The next offset should at least be offset + 1;
  Value minOffset = ifOp.getResults()[2];
  Value nxOffset = ADDI(info.offset, c1);
  Value maxPred = CMPI(ugt, minOffset, nxOffset);
  Value nextAbsOffset = SELECT(maxPred, minOffset, nxOffset);

  auto [size, stride] = sliceMeta[tid][lvl][info.depth];
  assert(stride == 1 && "Not yet implemented");
  Value sliceUB = ADDI(nextAbsOffset, size);

  // FIXME: this only works if there is only one parent.
  assert(info.depth - 1 == 0);
  // nextNonEmpty = nextNonEmpty && slice upper bound <= parent upperbound.
  nextNonEmpty = ANDI(nextNonEmpty, CMPI(ule, sliceUB, lvlSizes[tid][lvl]));

  // FIXME: compute relative offset.
  assert(info.depth - 1 == 0);
  return std::make_tuple(nextNonEmpty, nextMinCrd, nextAbsOffset);
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
