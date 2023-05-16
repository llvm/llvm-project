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
#define SELECT(c, l, r) (builder.create<arith::SelectOp>(loc, (c), (l), (r)))

//===----------------------------------------------------------------------===//
// File local helper functions.
//===----------------------------------------------------------------------===//

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
  Value rem = builder.create<arith::RemUIOp>(loc, crd, stride);
  crd = builder.create<arith::DivUIOp>(loc, crd, stride);
  return std::make_pair(crd, rem);
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
                                Level dstLvl) {
  Value crd = C_IDX(0);
  const auto reassoc = getCollapseReassociation(tid, dstLvl);
  const unsigned reassocSize = reassoc.size();
  for (unsigned i = 0; i < reassocSize; i++) {
    const Level srcLvl = reassoc[i];
    // A load on the coordinates array yields the coordinate.
    const Value mem = coordinatesBuffers[tid][srcLvl];
    /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
    const Value pos = posits[tid][dstLvl];
    const Value off = genIndexLoad(builder, loc, mem, pos);
    // Linearized the coordinates within the same collapse reassociation.
    crd = ADDI(crd, off);
    if (i != reassocSize - 1) {
      crd = MULI(crd, this->lvlSizes[tid][reassoc[i + 1]]);
    }
  }
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

  const unsigned numTensors = ts.size();
  this->tensors.assign(ts.begin(), ts.end());
  this->lvlTypes.assign(numTensors, std::vector<DimLevelType>());
  this->lvlSizes.assign(numTensors, std::vector<Value>());
  this->highs.assign(numTensors, std::vector<Value>());
  this->segHi.assign(numTensors, std::vector<Value>());
  this->posits.assign(numTensors, std::vector<Value>());
  this->coords.assign(numTensors, std::vector<Value>());
  this->positionsBuffers.assign(numTensors, std::vector<Value>());
  this->coordinatesBuffers.assign(numTensors, std::vector<Value>());
  this->valBuffer.assign(numTensors, nullptr);
  this->collapseReassoc.assign(numTensors, nullptr);
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
      numTensors, std::vector<std::vector<std::pair<TensorId, Level>>>());
  this->slicePosBuffer.assign(numTensors, std::vector<std::vector<Value>>());
  this->sliceSizes.assign(numTensors, std::vector<std::vector<Value>>());
  this->sliceStack.assign(numTensors, std::vector<SliceInfo>());
  this->levelReducedDep.assign(numTensors, std::vector<unsigned>());

  // Initialize nested types of `TensorId`-indexed fields.
  for (TensorId tid = 0; tid < numTensors; tid++) {
    const Value t = tensors[tid];
    // a scalar or 0-dimension tensors
    if (isZeroRankedTensorOrScalar(t.getType()))
      continue;

    auto rtp = getRankedTensorType(t);
    if (auto reshape = t.getDefiningOp<tensor::CollapseShapeOp>();
        isUniqueCOOType(rtp) && reshape) {
      // TODO: Supports more kinds of sparse tensors.
      // FIXME: We should instead lower reshape operations on sparse tensors to
      // view change.
      collapseReassoc[tid] = reshape.getReassociation();
      rtp = reshape.getSrcType();
      // Overwrites the tensor to the source tensor of reshape operations.
      tensors[tid] = reshape.getSrc();
    }
    const SparseTensorType stt(rtp);
    const Level lvlRank = stt.getLvlRank();
    // We always treat sparse output tensor as dense so that we always iterate
    // it based on lvl size.
    if (stt.hasEncoding() && !(isOutputTensor(tid) && isSparseOut)) {
      const auto enc = stt.getEncoding();
      isSparseSlices[tid] = enc.isSlice();
      for (auto lvlTp : enc.getDimLevelType())
        lvlTypes[tid].push_back(lvlTp);
    } else {
      lvlTypes[tid].assign(lvlRank, DimLevelType::Dense);
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
    dependentLvlMap[tid].assign(lvlRank,
                                std::vector<std::pair<TensorId, Level>>());
    slicePosBuffer[tid].assign(lvlRank, std::vector<Value>());
    sliceSizes[tid].assign(lvlRank, std::vector<Value>());
    sliceStack[tid].emplace_back(/*minCrd=*/Value(),
                                 /*offset=*/Value(), /*isNonEmpty*/ Value(),
                                 std::nullopt, 0);
    if (dimGetter) {
      auto reassoc = collapseReassoc[tid];
      Level dstRank = reassoc ? reassoc.size() : lvlRank;
      for (Level l = 0; l < dstRank; l++) {
        dependentLvlMap[tid][l] = dimGetter(tid, l);
        unsigned depends = dependentLvlMap[tid][l].size();
        if (depends == 0)
          continue;
        // TODO: View-base collapse and dependent index reduction are not
        // compatible right now.
        assert(!reassoc);
        // We need `depends - 1` slices to fully  the affine expression.
        sliceSizes[tid][l].assign(depends - 1, nullptr);
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

void LoopEmitter::initializeLoopEmit(OpBuilder &builder, Location loc,
                                     LoopEmitter::OutputUpdater updater) {
  // For every tensor:
  // * get the values buffer.
  // * For every level:
  //   * get the positions and coordinates buffers
  //   * get/compute the level-size, which is also used as the upper-bound
  //     on positions.
  for (TensorId t = 0, numTensors = getNumTensors(); t < numTensors; t++) {
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
      // in the code supports HigherOrdering yet either.
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
        ArrayRef<std::pair<TensorId, Level>> depLvls = dependentLvlMap[t][lvl];
        // Needs at least two operands to form a non-trivial affine expression.
        assert(depLvls.size() > 1);

        Value size = c0;
        for (unsigned e = depLvls.size() - 1; e >= 1; e--) {
          auto [dt, dd] = depLvls[e];
          size = ADDI(size, lvlSizes[dt][dd]);
          sliceSizes[t][lvl][e - 1] = size;
        }
      }
    }
  }
  localInsertPos = builder.getInsertionPoint()->getPrevNode();
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
    } else {
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
    } else {
      if (!isDenseDLT(lvlTypes[tid][lvl])) {
        // Else this is a resolved-slice, and advance posit similar to TACO.
        Value c1 = C_IDX(1), c2 = C_IDX(2);
        // pIdx += 2, we finished the current lvl, advance the pointer index of
        // the previous level by two to skip the [pLo, pHi] for current level.
        Value sPtrBuf = slicePosBuffer[tid][lvl].back();
        Value curP = genIndexLoad(builder, loc, sPtrBuf, c1);
        // TODO: we could probably use an SSA value for it.
        Value nexP = ADDI(curP, c2);
        builder.create<memref::StoreOp>(loc, nexP, sPtrBuf, c1);
      }
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

Operation *LoopEmitter::emitForLoopOverTensorAtLvl(
    OpBuilder &builder, Location loc, TensorId tid, Level dstLvl, Value lo,
    Value hi, MutableArrayRef<Value> reduc, bool isParallel) {
  bool isSparseCond = isCompressedDLT(lvlTypes[tid][dstLvl]) ||
                      isCompressedWithHiDLT(lvlTypes[tid][dstLvl]) ||
                      isSingletonDLT(lvlTypes[tid][dstLvl]);

  const auto reassoc = getCollapseReassociation(tid, dstLvl);
  // TODO: support dynamic slices.
  // Uses the first dimension here to build the loop bound (which is also the
  // biggest range).
  const Level srcLvl = reassoc.front();
  Value step = C_IDX(1);

  Operation *loop = nullptr;
  Value iv;
  if (isParallel) {
    assert(collapseReassoc[tid] == nullptr);
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
    assert(reassoc.size() == 1 || isUniqueCOOType(tensors[tid].getType()));
    // For COO, the position is the same across consecutive levels.
    /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
    llvm::for_each(reassoc,
                   [this, tid, iv](Level srcLvl) { posits[tid][srcLvl] = iv; });
    crd = genSparseCrd(builder, loc, tid, dstLvl);
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

    auto [trans, pred] = genSliceLegitPredicate(builder, loc, crd, tid, srcLvl);
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
  coords[tid][dstLvl] = crd;
  return loop;
}

Operation *LoopEmitter::emitWhileLoopOverSliceAtSparseLvl(
    OpBuilder &builder, Location loc, Value pLo, Value pHi, Value offset,
    Value sliceSize, TensorId tid, Level lvl, MutableArrayRef<Value> reduc) {
  // TODO: we should generalize the method to support iteration over for
  // normal slices as well to allow early break.
  Operation *insertPoint = nullptr;
  Operation *loop =
      genSliceLvlTraverseLoop(
          builder, loc, pLo, pHi, offset, sliceSize, tid, lvl, reduc,
          /*genYield=*/false, // unaware of the yield values from user yet
          [this, tid, lvl, reduc, offset,
           &insertPoint](OpBuilder &builder, Location loc, Value iv,
                         MutableArrayRef<Value> innerReduc) {
            assert(innerReduc.size() == reduc.size());
            // Updates users' reduction variable inplace
            for (unsigned i = 0, e = reduc.size(); i < e; i++)
              reduc[i] = innerReduc[i];
            // Loads the coordinates.
            Value absC =
                genIndexLoad(builder, loc, coordinatesBuffers[tid][lvl], iv);

            // We need to substract the offset to get relative coordinates.
            // TODO: how to assert relC >=0 during runtime?
            insertPoint = builder.create<arith::SubIOp>(loc, absC, offset);
            posits[tid][lvl] = iv;
            coords[tid][lvl] = insertPoint->getResult(0);
          })
          .first;
  // Sets the insertionn pointer inside loop body.
  builder.setInsertionPointAfter(insertPoint);
  return loop;
}

Operation *LoopEmitter::enterLoopOverTensorAtLvl(OpBuilder &builder,
                                                 Location loc,
                                                 ArrayRef<TensorLevel> tidLvls,
                                                 MutableArrayRef<Value> reduc,
                                                 bool isParallel) {
  // TODO: support multiple return on parallel for?
  assert(!isParallel || reduc.size() <= 1);
  bool isSparseCond = false, isSparseSliceCond = false;
  auto [tid, lvl] = unpackTensorLevel(tidLvls.front());

  // Finds out the tensor level that we should use to generate loops. Amongs all
  // the tensor levels, there is at most one sparse tensor level.
  for (auto [t, l] : unpackTensorLevelRange(tidLvls)) {
    assert(lvlTypes[t].size() > l);         // Must be a valid tid, dim pair
    assert(!coords[t][l] ||                 // We cannot re-enter the same level
           !dependentLvlMap[t][l].empty()); // unless it is a slice-driver loop
    auto lvlType = lvlTypes[t][l];
    // Must be a recognizable DLT.
    assert(isDenseDLT(lvlType) || isCompressedDLT(lvlType) ||
           isCompressedWithHiDLT(lvlType) || isSingletonDLT(lvlType));

    // This is a slice-driven loop on sparse level.
    if (!dependentLvlMap[t][l].empty() && !isDenseDLT(lvlType)) {
      assert(!isSparseSliceCond && !isSparseCond);
      isSparseSliceCond = true;
      tid = t;
      lvl = l;
      continue;
    }

    bool isSparse = isCompressedDLT(lvlType) || isSingletonDLT(lvlType) ||
                    isCompressedWithHiDLT(lvlType);
    // We can at most have one sparse input, otherwise, a while loop is
    // required to co-iterate multiple sparse tensors.
    assert(!isSparseCond || !isSparse);
    assert(!isSparseSliceCond || !isSparseCond);
    if (isSparse) {
      tid = t;
      lvl = l;
    }
    isSparseCond = isSparseCond || isSparse;
  }

  DimLevelType lvlType = lvlTypes[tid][lvl];
  // TODO: Dense slice driven loop can be generated using for loop as well.
  assert(!isSparseSliceCond || !isDenseDLT(lvlType));
  bool isDenseSliceCond =
      isDenseDLT(lvlType) && !dependentLvlMap[tid][lvl].empty();
  // if the slice is fully reduced, we can now use TACO-based algorithm to
  // iterate it.

  Operation *l = nullptr;

  // At most one tensor used as condition in for loop;
  SmallVector<TensorLevel, 1> condTidLvl;
  // There might be multiple dense slice driven tensor.
  SmallVector<SliceLoopInfo> sliceDrivenInfo;

  // Generates loops differently depending on whether we need a slice-driven
  // loop or a simple level traversal loop.
  if (isSparseSliceCond) {
    bool fullyReduced = depFullyReduced(tid, lvl);
    if (!fullyReduced) {
      l = emitSliceDrivenLoopOverTensorAtLvl(builder, loc, tid, lvl, reduc);
    } else {
      // If the slice is fully reduced, we can now use TACO-based algorithm to
      // iterate it.
      l = emitWhileLoopOverSliceAtSparseLvl(
          builder, loc, posits[tid][lvl], highs[tid][lvl],
          getFinalSliceOnLvl(tid, lvl).offset, sliceSizes[tid][lvl].back(), tid,
          lvl, reduc);
    }
    levelReducedDep[tid][lvl]++;
    sliceDrivenInfo.emplace_back(tid, lvl, fullyReduced);
  } else {
    Value lo = isSparseCond ? posits[tid][lvl]           // current offset
                            : loopSeqStack.back().first; // universal index
    Value hi = highs[tid][lvl];
    if (isDenseSliceCond) {
      bool fullyReduced = depFullyReduced(tid, lvl);
      Value sliceSz = sliceSizes[tid][lvl][sliceStack[tid].back().depth - 1];
      // Adjust for loop hi for dense slice-driven loop.
      if (fullyReduced) {
        hi = sliceSz;
        condTidLvl.push_back(makeTensorLevel(tid, lvl));
      } else {
        hi = SUBI(lvlSizes[tid][lvl], sliceSz);
        hi = ADDI(hi, C_IDX(1));
      }
    } else {
      condTidLvl.push_back(makeTensorLevel(tid, lvl));
    }
    l = emitForLoopOverTensorAtLvl(builder, loc, tid, lvl, lo, hi, reduc,
                                   isParallel);
  }
  Value iv = coords[tid][lvl];
  for (auto [t, l] : unpackTensorLevelRange(tidLvls)) {
    // We only need to handle slice-driven loops on dense level here.
    // If it is a slice-driven loop on sparse level, it needs a while loop to
    // insert break statements, and it must have been handled correctly in L692.
    if (!dependentLvlMap[t][l].empty() && isDenseDLT(lvlTypes[t][l])) {
      // Pushes sliced levels to build correct LoopInfo.
      bool fullyReduc = depFullyReduced(t, l);
      SliceInfo &info = sliceStack[t].back();
      if (fullyReduc) {
        posits[t][l] = genAddress(builder, loc, t, l, ADDI(info.offset, iv));
      } else {
        // Puts sliced dense loop into LoopInfo so that LoopEmitter knows how to
        // exit it.
        sliceDrivenInfo.emplace_back(t, l, fullyReduc);
        // Update the slice information as we enter the new loop.
        assert(*info.slicedOnLvl == l);
        info.minCrd = info.offset = iv;
        info.isNonEmpty = constantI1(builder, loc, true);
        levelReducedDep[t][l]++;
      }
    }
  }
  // NOTE: we can also prepare for next dim here in advance
  // Pushes the loop into stack.
  loopStack.emplace_back(condTidLvl, sliceDrivenInfo, l,
                         builder.getInsertionBlock(), iv, loopTag);
  // Emit extra locals.
  emitExtraLocalsForTensorsAtDenseLvls(builder, loc, tidLvls);
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
  // For dense levels, the level-coordinate also serves as the position.
  Value lvlCrd = genAffine(builder, loc, lvlExpr);
  posits[tid][lvl] = genAddress(builder, loc, tid, lvl, lvlCrd);
}

Operation *LoopEmitter::enterCoIterationOverTensorsAtLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorLevel> tidLvls,
    bool needsUniv, MutableArrayRef<Value> reduc) {
  // NOTE: the slice driven tensor-related reduction variable must
  // appear before normal tensors.
  SmallVector<Type> types;
  SmallVector<Value> operands;
  // Construct the while-loop with a parameter for each coordinate.
  const Type indexType = builder.getIndexType();
  for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
    // TODO: support coiteration with slice driven tensors.
    const auto lvlTp = lvlTypes[tid][lvl];
    assert(dependentLvlMap[tid][lvl].empty() && "TODO: not yet implemented");
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp) ||
        isCompressedWithHiDLT(lvlTp)) {
      const auto reassoc = getCollapseReassociation(tid, lvl);
      for (unsigned i = 0, e = reassoc.size() - 1; i < e; i++) {
        if (!isUniqueDLT(lvlTypes[tid][reassoc[i]])) {
          // This is the segment high for each non-unique levels.
          types.push_back(indexType);
          operands.push_back(C_IDX(0));
        }
      }
      const auto pos = posits[tid][reassoc.front()];
      assert(pos);
      types.push_back(indexType);
      operands.push_back(pos);
    }
  }
  // The position where user-supplied reduction variable starts.
  for (Value rec : reduc) {
    types.push_back(rec.getType());
    operands.push_back(rec);
  }
  if (needsUniv) {
    types.push_back(indexType);
    // Update universal index.
    operands.push_back(loopSeqStack.back().first);
  }
  assert(types.size() == operands.size());
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(loc, types, operands);

  SmallVector<Location> locs(types.size(), loc);
  Block *before = builder.createBlock(&whileOp.getBefore(), {}, types, locs);
  Block *after = builder.createBlock(&whileOp.getAfter(), {}, types, locs);

  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  builder.setInsertionPointToStart(&whileOp.getBefore().front());
  Value cond;
  unsigned o = 0;
  for (auto [t, lvl] : unpackTensorLevelRange(tidLvls)) {
    const TensorId tid = t; // Why `t` can not be captured by lambda?
    const auto lvlTp = lvlTypes[tid][lvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp) ||
        isCompressedWithHiDLT(lvlTp)) {
      const auto reassoc = getCollapseReassociation(tid, lvl);
      assert(reassoc.size() == 1 || isUniqueCOOType(tensors[tid].getType()));
      for (unsigned i = 0, e = reassoc.size() - 1; i < e; i++) {
        if (!isUniqueDLT(lvlTypes[tid][reassoc[i]])) {
          // Links the SSA chain for segHi.
          segHi[tid][reassoc[i]] = after->getArgument(o++);
        }
      }
      Value op1 = before->getArgument(o);
      // We used the first level bound as the bound the collapsed set of levels.
      Value op2 = highs[tid][reassoc.front()];
      Value opc = CMPI(ult, op1, op2);
      cond = cond ? ANDI(cond, opc) : opc;
      // Update positions
      Value pos = after->getArgument(o++);
      // For COO, the position is the same across consecutive levels.
      /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
      llvm::for_each(reassoc, [this, tid, pos](Level srcLvl) {
        posits[tid][srcLvl] = pos;
      });
    }
  }
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());

  // Generates while body.
  builder.setInsertionPointToStart(&whileOp.getAfter().front());

  SmallVector<std::pair<Value, unsigned>> slicesPreds;
  unsigned i = 0;
  for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
    // Prepares for next level.
    const auto lvlTp = lvlTypes[tid][lvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp) ||
        isCompressedWithHiDLT(lvlTp)) {
      coords[tid][lvl] = genSparseCrd(builder, loc, tid, lvl);
      if (isSparseSlices[tid]) {
        auto [trans, pred] =
            genSliceLegitPredicate(builder, loc, coords[tid][lvl], tid, lvl);
        slicesPreds.emplace_back(pred, i);
        // Updates to the relative coordinate to the slice.
        coords[tid][lvl] = trans;
      }
      i++;
    }
  }

  if (!slicesPreds.empty()) {
    // Skips invalid loop iteration when slice coordinate is inapplicable.
    SmallVector<Value> yields(after->getArguments());
    // Generates a list of if statments
    //  pos = in_slice ? pos : pos + 1
    // TODO: instead of always picking pos + 1, we should set pos = high to
    // break to loop if the coordinates are larger than the slice size.
    //
    // This "idx" is the index into `llvm::zip(tids, lvls)`
    for (auto [pred, idx] : slicesPreds) {
      Value nextPos = ADDI(yields[idx], C_IDX(1));
      yields[idx] = SELECT(pred, yields[idx], nextPos);
    }

    Value pred = slicesPreds.front().first;
    for (int i = 1, e = slicesPreds.size(); i < e; i++) {
      pred = ANDI(pred, slicesPreds[i].first);
    }
    auto ifOp = builder.create<scf::IfOp>(loc, types, pred, /*else*/ true);
    ifOp->setAttr(getLoopEmitterLoopAttrName(),
                  StringAttr::get(builder.getContext(), "slice"));
    YIELD(ifOp->getResults());
    assert(types.size() == yields.size());
    // If not all slices are legit
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    YIELD(yields);

    // If all slices are legit, start the user generated code.
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }

  Value min;
  // Finds the minimum coordinate
  if (!needsUniv) {
    for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
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
    min = after->getArguments().back();
  }

  // Sets up the loop stack.
  loopStack.emplace_back(tidLvls, ArrayRef<SliceLoopInfo>(), whileOp,
                         builder.getInsertionBlock(), min, loopTag);
  assert(loopStack.size() == loopSeqStack.size());

  for (auto [tid, dstLvl] : unpackTensorLevelRange(tidLvls)) {
    const auto reassoc = getCollapseReassociation(tid, dstLvl);
    assert(reassoc.size() == 1 || isUniqueCOOType(tensors[tid].getType()));
    // TODO: Refactors this into smaller functions.
    // NOTE: For all the collapsed level (except for the last one, that is why
    // the loop ends with `reassoc.size() - 1`), as each iteration is advanced
    // by the segment size of the last level, which does not always invalidate
    // the segment size for the previous levels, thus we need to propagate the
    // segment sizes across loop iterations and only forward if needed.
    //
    // E.g., for a COO tensor with the following coordinates array.
    // (0, 0, 1),
    // (0, 0, 2),
    // (1, 1, 1),
    // segHi[lvl=0] = segHi[lvl=1] = 2
    // segHi[lvl=2] = 1,
    // the first iteration does not invalidate segHi[0] and segHi[1]
    for (unsigned i = 0, e = reassoc.size() - 1; i < e; i++) {
      const Level srcLvl = reassoc[i];
      if (!isUniqueDLT(lvlTypes[tid][srcLvl])) {
        const Value pos = posits[tid][srcLvl];
        const auto oldSegHi = segHi[tid][srcLvl];
        assert(oldSegHi);
        Value newSegHi = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::uge, pos, oldSegHi);
        auto ifNewSegHi = builder.create<scf::IfOp>(loc, builder.getIndexType(),
                                                    newSegHi, true);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(ifNewSegHi.thenBlock());
          YIELD(genSegmentHigh(builder, loc, tid, srcLvl, pos,
                               highs[tid][srcLvl]));
          // Else, resues the same segment high.
          builder.setInsertionPointToStart(ifNewSegHi.elseBlock());
          YIELD(oldSegHi);
        }
        highs[tid][srcLvl + 1] = segHi[tid][srcLvl] = ifNewSegHi.getResult(0);
      }
    };
    const auto srcLvl = reassoc.back();
    if (!isUniqueDLT(lvlTypes[tid][srcLvl])) {
      segHi[tid][srcLvl] = genSegmentHigh(
          builder, loc, tid, srcLvl, posits[tid][srcLvl], highs[tid][srcLvl]);
    }
  }

  // Emits extra locals
  emitExtraLocalsForTensorsAtDenseLvls(builder, loc, tidLvls);

  // Updates reduction variables
  assert(after->getNumArguments() == o + reduc.size() + (needsUniv ? 1 : 0));
  // In-place update on reduction variable.
  for (unsigned i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = after->getArgument(o + i);

  return whileOp;
}

void LoopEmitter::prepareLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                             TensorId tid, Level dstLvl) {
  assert(isValidLevel(tid, dstLvl));
  const auto lvlTp = lvlTypes[tid][dstLvl];

  if (isDenseDLT(lvlTp))
    return;

  const Value c0 = C_IDX(0);
  const Value c1 = C_IDX(1);
  for (const Level srcLvl : getCollapseReassociation(tid, dstLvl)) {
    // Either the first level, or the previous level has been set.
    /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
    assert(srcLvl == 0 || posits[tid][srcLvl - 1]);
    if (isDenseDLT(lvlTp))
      continue;
    if (isCompressedDLT(lvlTp) || isCompressedWithHiDLT(lvlTp)) {
      const Value mem = positionsBuffers[tid][srcLvl];

      Value pLo = srcLvl == 0 ? c0 : posits[tid][srcLvl - 1];
      if (isCompressedWithHiDLT(lvlTp))
        pLo = builder.create<arith::MulIOp>(loc, pLo, C_IDX(2));
      posits[tid][srcLvl] = genIndexLoad(builder, loc, mem, pLo);

      const Value pHi = ADDI(pLo, c1);
      highs[tid][srcLvl] = genIndexLoad(builder, loc, mem, pHi);
      return;
    }
    if (isSingletonDLT(lvlTp)) {
      const Value pLo = srcLvl == 0 ? c0 : posits[tid][srcLvl - 1];
      posits[tid][srcLvl] = pLo;

      // If we are coiterating non-unique levels, then use pHi=segHi;
      // otherwise use pHi=pLo+1.
      // NOTE: Just because the level is non-unique, that does not
      // guarantee that segHi is defined: because we only generate segHi
      // whenever coiterating, in order to improve code quality for the
      // non-coiterating cases.
      const auto parentSegHi = segHi[tid][srcLvl - 1];
      highs[tid][srcLvl] =
          (!isUniqueDLT(lvlTypes[tid][srcLvl - 1]) && parentSegHi)
              ? parentSegHi
              : ADDI(pLo, c1);
      return;
    }
  }

  llvm_unreachable("Unrecognized level-type!");
}

void LoopEmitter::emitExtraLocalsForTensorsAtDenseLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorLevel> tidLvls) {
  // Initialize dense positions. Note that we generate dense coordinates of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
    if (isDenseDLT(lvlTypes[tid][lvl])) {
      // Slice-driven dense level should have be handled already.
      if (!dependentLvlMap[tid][lvl].empty())
        continue;

      auto enc = getSparseTensorEncoding(tensors[tid].getType());
      if (enc && !isSparseOutput(tid)) {
        bool validPos = lvl == 0 || posits[tid][lvl - 1];
        if (!validPos) {
          // We might not find the pos for the sparse output tensor as it is
          // unconditionally required by the sparsification.
          assert(isOutputTensor(tid));
          continue;
        }
        posits[tid][lvl] =
            genAddress(builder, loc, tid, lvl, loopStack.back().iv);
        // NOTE: we can also prepare for next lvl here in advance
      }
    }
  }
}

void LoopEmitter::exitForLoop(RewriterBase &rewriter, Location loc,
                              MutableArrayRef<Value> reduc) {
  const LoopInfo &loopInfo = loopStack.back();
  rewriter.setInsertionPointToEnd(loopInfo.userCodeBlock);
  for (auto [tid, lvl, reduced] : loopInfo.sliceDrivenInfo) {
    SliceInfo &info = sliceStack[tid].back();
    assert(isDenseDLT(lvlTypes[tid][lvl]));
    assert(*info.slicedOnLvl == lvl && !reduced);
    (void)reduced;
    // Resets slices pointers as the resolved slices are invalidated after we
    // moves forward to the next slice.
    invalidateSliceIterIdx(rewriter, loc, tid, lvl);
    info.minCrd = info.offset = info.isNonEmpty = Value();
    levelReducedDep[tid][lvl]--;
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
      // NOTE: This is users' responsibilty to ensure the operation are
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
  for (auto [tid, lvl] : unpackTensorLevelRange(loopInfo.tidLvls)) {
    // Reset to null.
    coords[tid][lvl] = Value();
    posits[tid][lvl] = Value();
    // Dense level, high is fixed.
    if (!isDenseDLT(lvlTypes[tid][lvl]))
      highs[tid][lvl] = Value();
  }
}

void LoopEmitter::exitWhileLoop(OpBuilder &builder, Location loc,
                                MutableArrayRef<Value> reduc) {
  const LoopInfo &loopInfo = loopStack.back();
  auto whileOp = llvm::cast<scf::WhileOp>(loopInfo.loop);
  builder.setInsertionPointToEnd(loopInfo.userCodeBlock);
  Value iv = loopInfo.iv;

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
      genSliceNextInduction(builder, loc, whileOp, tid, lvl, operands, o);
      continue;
    }
    // TODO: We need to distinguish coiterate loop with slice-driven loop and
    // fully reduced while op for iterating one slices.
    // FIXME: since we didn't implement coiteration, this must be iteration
    // just on fully resolved slice.
    assert(loopInfo.sliceDrivenInfo.size() == 1 && loopInfo.tidLvls.empty());
    // The if guard to filter out out-range coordinates.
    assert(llvm::isa<scf::IfOp>(builder.getInsertionBlock()->getParentOp()));
    posits[tid][lvl] = whileOp->getResult(o++);
    // FIXME: we are not using continue here since we do not support
    // coiteration on slices. But it need to be treated similarly as the
    // universal index.
    o++; // skip continue flag.
    // Since we did not push two results from whileOp. The size of the
    // operands vector is smaller than the actual number of return values from
    // the whileOp.
    // It is because we are actually generating yield in the IfOp inside the
    // whileOp to only iterates over inbound coordinates within the slices.
    delta += 2;
  };

  Value one = C_IDX(1);
  for (auto [tid, dstLvl] : unpackTensorLevelRange(loopInfo.tidLvls)) {
    const auto lvlTp = lvlTypes[tid][dstLvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp) ||
        isCompressedWithHiDLT(lvlTp)) {
      const auto reassoc = getCollapseReassociation(tid, dstLvl);
      assert(reassoc.size() == 1 || isUniqueCOOType(tensors[tid].getType()));
      for (unsigned i = 0, e = reassoc.size() - 1; i < e; i++) {
        const Level srcLvl = reassoc[i];
        if (!isUniqueDLT(lvlTypes[tid][srcLvl])) {
          operands.push_back(segHi[tid][srcLvl]);
          o++;
        }
      }
      const Value crd = coords[tid][dstLvl];
      const Value pos = posits[tid][dstLvl];
      Value cmp = CMPI(eq, crd, iv);
      // If the loop contains a coiteration with non-unique level, we fast
      // forward all the duplicated coords by setting the position to the
      // segment high.
      Value add = !isUniqueDLT(lvlTypes[tid][reassoc.back()])
                      ? segHi[tid][reassoc.back()]
                      : ADDI(pos, one);

      operands.push_back(SELECT(cmp, add, pos));
      // Following loops continue iteration from the break point of the
      // current while loop.
      const Value newPos = whileOp->getResult(o++);
      // We need to define a new local variable for `tid` to avoid
      // warnings about "captured structured bindings are a C++20 extension".
      // FIXME(wrengr): define a helper function to capture this idiom!
      const TensorId newTid = tid;
      llvm::for_each(reassoc, [this, newTid, newPos](Level srcLvl) {
        posits[newTid][srcLvl] = newPos;
      });
      // The coordinate is invalid now.
      coords[tid][dstLvl] = nullptr;
      // The segment high is invalid now.
      segHi[tid][dstLvl] = nullptr;
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
  YIELD(operands);
  builder.setInsertionPointAfter(whileOp);
}

void LoopEmitter::exitCurrentLoop(RewriterBase &rewriter, Location loc,
                                  MutableArrayRef<Value> reduc) {
  // Clean up the values, it would help use to discover potential bug at a
  // earlier stage (instead of silently using a wrong value).
  const LoopInfo &loopInfo = loopStack.back();
  SmallVector<Value> red;
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
// while(loopLo < loopHi) {
//   if (coords[loopLo] < offset + size) {
//     body_builder
//   } else {
//    break;
//   }
//   loopLo ++;
// }
std::pair<Operation *, ValueRange> LoopEmitter::genSliceLvlTraverseLoop(
    OpBuilder &builder, Location loc, Value loopLo, Value loopHi, Value offset,
    Value size, TensorId tid, Level lvl, ValueRange userReduc, bool genYield,
    LoopBodyBuilder bodyBuilder) {
  Value c1 = C_IDX(1);
  Value sliceHi = ADDI(offset, sliceSizes[tid][lvl].back());

  SmallVector<Value> reduc = {
      loopLo,                         // loop lower bounds
      constantI1(builder, loc, true), // continue
  };
  // Append user required reduction value.
  reduc.append(userReduc.begin(), userReduc.end());
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(
      loc, ValueRange(reduc).getTypes(), reduc,
      /*beforeBuilder=*/
      [loopHi](OpBuilder &builder, Location loc, ValueRange args) {
        Value lo = args[0];
        Value cont = args[1];
        Value inBound = CMPI(ult, lo, loopHi);
        Value cond = ANDI(cont, inBound);
        // continue if not yet break nor out of bound.
        builder.create<scf::ConditionOp>(loc, cond, args);
      },
      /*afterBuilder=*/
      [this, c1, tid, lvl, sliceHi, genYield,
       bodyBuilder](OpBuilder &builder, Location loc, ValueRange args) {
        Value iv = args[0];
        Value coord =
            genIndexLoad(builder, loc, coordinatesBuffers[tid][lvl], iv);
        Value cont = CMPI(ult, coord, sliceHi);
        TypeRange types = args.drop_front(2).getTypes();

        auto ifOp = builder.create<scf::IfOp>(loc, types, cont, true);
        {
          // 2 reduction variable maintained by us.
          SmallVector<Value> ifRet = args.drop_front(2);
          assert(ifRet.size() == args.size() - 2);

          OpBuilder::InsertionGuard guard(builder);
          // If coord >= sliceHi.
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          YIELD(ifRet);

          // If coord < sliceHi.
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          // Delegates to users' callback.
          bodyBuilder(builder, loc, iv, ifRet);
          if (genYield) {
            builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
            YIELD(ifRet);
          }
        }
        // Marks this speical ifOp to avoid sparisification finalizing it.
        ifOp->setAttr(getLoopEmitterLoopAttrName(),
                      StringAttr::get(builder.getContext(), "slice"));
        // Insertion point restored to after ifOp.
        SmallVector<Value> yields;
        // Increase induction variable.
        yields.push_back(ADDI(iv, c1));
        yields.push_back(cont);
        yields.append(ifOp.getResults().begin(), ifOp.getResults().end());
        YIELD(yields);
      });

  builder.setInsertionPointAfter(whileOp);
  return std::make_pair(whileOp, whileOp.getResults().drop_front(2));
}

// Generates a loop nest that traverse all the unresolved levels in between.
// TODO: it can only handle all compressed tensors.
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
  if (firstResLvl.has_value()) {
    // Overwrite position when the first level is fully resolved.
    pos = posits[firstResLvl->first][firstResLvl->second];
    ip = builder.saveInsertionPoint();
  } else {
    const SliceInfo &frontSlice = *unResLvls.back();
    Level firstLvl = *frontSlice.slicedOnLvl;
    if (!lvlFullyResolved(tid, firstLvl)) {
      if (isCompressedDLT(lvlTypes[tid][firstLvl])) {
        unsigned depth = frontSlice.depth - 1;
        Value offset = frontSlice.offset;
        Value sPtrBuf = slicePosBuffer[tid][firstLvl][depth];
        Value mSz = genIndexLoad(builder, loc, sPtrBuf, c0); // memSize
        outerMost = builder.create<scf::ForOp>(
            loc, c2, mSz, c2, innerArgs,
            [this, c1, tid, firstLvl, offset, sPtrBuf, &ip, &pos,
             &innerArgs](OpBuilder &builder, Location loc, Value iv,
                         ValueRange iterArgs) {
              // generate traversal for each level.
              Value loopLo = genIndexLoad(builder, loc, sPtrBuf, iv);
              Value loopHi = genIndexLoad(builder, loc, sPtrBuf, ADDI(iv, c1));
              ValueRange itArgs =
                  genSliceLvlTraverseLoop(
                      builder, loc, loopLo, loopHi, offset,
                      sliceSizes[tid][firstLvl].back(), tid, firstLvl, iterArgs,
                      false,
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
        Value sliceSz =
            sliceSizes[tid][*frontSlice.slicedOnLvl][frontSlice.depth - 1];
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
      Value sliceSz = sliceSizes[tid][sliceLvl][slice->depth - 1];
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
  Value size = sliceSizes[tid][lvl][0];
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
  // Fills out pIdxBuffer[tid][lvl][0] with [/*memSize =*/4, 0, 0, pHi]
  builder.create<memref::StoreOp>(loc, c4, sPtrBuf, c0);  // memSize = 4
  builder.create<memref::StoreOp>(loc, c0, sPtrBuf, c1);  // index = 0
  builder.create<memref::StoreOp>(loc, pLo, sPtrBuf, c2); // pLo
  builder.create<memref::StoreOp>(loc, pHi, sPtrBuf, c3); // pHi

  // This is an non empty tensor if 0 < pHi.
  Value isNonEmpty = CMPI(ult, c0, pHi);
  // The minimal coord must be at the first on ordered level.
  // FIXME: Technically we should load the coord only when the slice is
  // nonempty. though we assume that even on empty sparse tensors, a non-empty
  // ptr/idx buffer is allocated for each level so it would not cause OOB to
  // avoid generating a ifOp here.
  Value minCrd = genIndexLoad(builder, loc, coordinatesBuffers[tid][0], c0);

  // FIXME: We need the relative offset related to the base slice.
  Value absOffset = offsetFromMinCoord(builder, loc, minCrd, size, isNonEmpty);
  sliceStack[tid].emplace_back(minCrd, absOffset, isNonEmpty, lvl, /*depth=*/1);
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
  Value size = sliceSizes[tid][lvl][depth];
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
      [this, c1, c2, tid, lvl, sPtrBuf](OpBuilder &builder, Location loc,
                                        Value iv,
                                        MutableArrayRef<Value> reduc) {
        Value &nonEmpty = reduc[0];
        Value &minCrd = reduc[1];
        Value &curMemSz = reduc[2];

        Value pHi = ADDI(iv, c1);
        Value sPLo = genIndexLoad(builder, loc, positionsBuffers[tid][lvl], iv);
        Value sPHi =
            genIndexLoad(builder, loc, positionsBuffers[tid][lvl], pHi);

        // isNonEmpty = isNonEmpty || lvlNonEmpty, i.e., as long as there is one
        // non-empty lvl, the slice is non-empty.
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
        // curMemSize += 2
        curMemSz = ADDI(curMemSz, c2);
      });

  Value isNonEmpty = result[0];
  Value minCrd = result[1];
  // Two metadata [memSize, idx].
  // TODO: Can use an SSA value for these two metadata
  builder.create<memref::StoreOp>(loc, result[2], sPtrBuf, c0);
  builder.create<memref::StoreOp>(loc, c0, sPtrBuf, c1);
  // FIXME: we need the relative offset related to the base slice.
  Value absOffset = offsetFromMinCoord(builder, loc, minCrd, size, isNonEmpty);
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
        genIndexLoad(builder, loc, slicePosBuffer[tid][lvl].back(), c1);
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
    // The buffer can be reused, and the size is loop invariant: it only depends
    // on the iteration graph's toposort.
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
      auto depth = remDepOnLevel(tid, curLevel - 1);
      assert(sliceSizes[tid][lvl].size() >= depth);
      Value sz = *(sliceSizes[tid][lvl].rbegin() + depth - 1);
      bufSize = MULI(bufSize, sz);
    }
    // For a pair of [pLo, pHi]. Note that we can not compress pHi because slice
    // creates segments in the index buffer so that the pHi for the current
    // level is no longer the pLo for the next level.
    bufSize = MULI(bufSize, c2);
    // Additional two metadata {memSize, idx} at head.
    bufSize = ADDI(bufSize, c2);
    llvm::for_each(
        slicePosBuffer[tid][lvl], [bufSize, loc, &builder](Value &cache) {
          cache = genAlloca(builder, loc, bufSize, builder.getIndexType());
        });
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
      builder.create<memref::StoreOp>(loc, C_IDX(0),
                                      slicePosBuffer[tid][i].back(), C_IDX(1));
    }
  }
}

void LoopEmitter::genSliceNextInduction(OpBuilder &builder, Location loc,
                                        const Operation *op, TensorId tid,
                                        Level lvl,
                                        SmallVectorImpl<Value> &operands,
                                        unsigned &retIdx) {
  if (!isCompressedDLT(lvlTypes[tid][lvl]))
    llvm_unreachable("TODO");

  // else generate code to compute next non empty slice.
  Value c0 = C_IDX(0), c1 = C_IDX(1), c2 = C_IDX(2);

  auto whileOp = llvm::cast<scf::WhileOp>(op);
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
    //    for (i = 0; i < slicePos.size(); i+=2) {
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
        builder, loc, pSt, mSz, c2, loopArgs,
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
    Value minOffset = SUBI(tmp, sliceSizes[tid][lvl][info.depth - 1]);
    Value p = CMPI(uge, tmp, sliceSizes[tid][lvl][info.depth - 1]);
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

  Value sliceUB = ADDI(nextAbsOffset, sliceSizes[tid][lvl][info.depth - 1]);

  // FIXME: this only works if there is only one parent.
  assert(info.depth - 1 == 0);
  // nextNonEmpty = nextNonEmpty && slice upper bound <= parent upperbound.
  nextNonEmpty = ANDI(nextNonEmpty, CMPI(ule, sliceUB, lvlSizes[tid][lvl]));

  // FIXME: compute relative offset.
  assert(info.depth - 1 == 0);
  Value nextRelOffset = nextAbsOffset;
  nextRelOffset = SELECT(nextNonEmpty, nextRelOffset, c0);

  operands.push_back(nextNonEmpty);
  operands.push_back(nextMinCrd);
  operands.push_back(nextAbsOffset); // we push the absolute offset.

  // Update the slice stack.
  info.isNonEmpty = whileOp.getResult(retIdx++);
  info.minCrd = whileOp.getResult(retIdx++);
  info.offset = whileOp.getResult(retIdx++);
}

Operation *LoopEmitter::emitSliceDrivenLoopOverTensorAtLvl(
    OpBuilder &builder, Location loc, TensorId tid, Level lvl,
    MutableArrayRef<Value> reduc) {
  assert(!depFullyReduced(tid, lvl));
  SliceInfo &sliceInfo = sliceStack[tid].back();
  assert(sliceInfo.slicedOnLvl == lvl);

  // The order matters!
  SmallVector<Value, 3> operands{sliceInfo.isNonEmpty, sliceInfo.minCrd,
                                 sliceInfo.offset};
  // number of reduction maintained by us.
  size_t numMetaReduc = operands.size();

  // Append user-required reduction values.
  operands.append(reduc.begin(), reduc.end());
  assert(operands.size() == numMetaReduc + reduc.size());

  // while (slice.nonEmpty()) {
  //   bodyBuilder();
  //   SliceNext();
  // }
  auto whileOp = builder.create<scf::WhileOp>(
      loc, ValueRange(operands).getTypes(), operands,
      /*beforeBuilder=*/
      [](OpBuilder &builder, Location loc, ValueRange args) {
        builder.create<scf::ConditionOp>(loc, /*isNonEmpty*/ args[0], args);
      },
      /*afterBuilder=*/
      [this, tid, lvl, reduc, numMetaReduc,
       &sliceInfo](OpBuilder &builder, Location loc, ValueRange args) {
        assert(args.size() == reduc.size() + numMetaReduc);
        sliceInfo.isNonEmpty = args[0];
        sliceInfo.minCrd = args[1];
        sliceInfo.offset = args[2];
        // The slice offset is used to coiterate with other tensors'
        // coordinates.
        Value c = sliceInfo.offset;
        if (sliceInfo.depth > 1) {
          // Coord is the relative offset related to its parents.
          // Update c = absOffset[lvl][depth] - absOffset[lvl][depth - 1]
          llvm_unreachable("TODO: not yet implement");
        }
        coords[tid][lvl] = c;

        for (unsigned i = 0, e = reduc.size(); i < e; i++)
          reduc[i] = args[i + numMetaReduc];
      });

  // Set the insertion point to while loop body.
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
  return whileOp;
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
