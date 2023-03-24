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
// File local helper functions.
//===----------------------------------------------------------------------===//

/// Generates a position/coordinate load from the sparse storage scheme.
/// Narrower data types need to be zero extended before casting the
/// value into the `Index` type used for looping and indexing.
static Value genIndexLoad(OpBuilder &builder, Location loc, Value mem,
                          Value s) {
  // For the scalar case, we simply zero extend narrower indices into 64-bit
  // values before casting to index without a performance penalty. Here too,
  // however, indices that already are 64-bit, in theory, cannot express the
  // full range as explained above.
  Value load = builder.create<memref::LoadOp>(loc, mem, s);
  if (!load.getType().isa<IndexType>()) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), load);
    load =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), load);
  }
  return load;
}

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
  crd = builder.create<arith::MulIOp>(loc, crd, stride);
  crd = builder.create<arith::AddIOp>(loc, crd, offset);
  return crd;
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
  crd = builder.create<arith::SubIOp>(loc, crd, offset);
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
    auto geOffset = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, crd, offset);
    conds.push_back(geOffset);
  }

  // Second, coord_in_slice < length
  auto ltLength = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                newCrd, lvlSizes[tid][lvl]);
  conds.push_back(ltLength);

  // Third, rem == 0 (skip the check if stride is known to be 1).
  if (auto staticStride = enc.getStaticLvlSliceStride(lvl);
      !(staticStride.has_value() && *staticStride == 1)) {
    auto fitStride = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, crdRem, constantIndex(builder, loc, 0));
    conds.push_back(fitStride);
  }

  // Must meet all condition to be a valid coordinate in slice.
  auto pred = conds.front();
  for (auto cond : ValueRange(conds).drop_front())
    pred = builder.create<arith::AndIOp>(loc, pred, cond);

  return {newCrd, pred};
}

//===----------------------------------------------------------------------===//
// Sparse tensor loop emitter class implementations
//===----------------------------------------------------------------------===//

Value LoopEmitter::genAddress(OpBuilder &builder, Location loc, TensorId tid,
                              Level lvl, Value crd) {
  Value pos = lvl == 0 ? constantIndex(builder, loc, 0) : posits[tid][lvl - 1];
  Value mul = builder.create<arith::MulIOp>(loc, highs[tid][lvl], pos);
  if (isSparseSlices[tid])
    crd = toSliceCrd(builder, loc, crd, sliceOffsets[tid][lvl],
                     sliceStrides[tid][lvl], tensors[tid], lvl);
  Value add = builder.create<arith::AddIOp>(loc, mul, crd);
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
          builder.create<scf::YieldOp>(loc, isSameCrd);
          // Else, the position is out of bound, yield false to terminate the
          // loop.
          builder.setInsertionPointToStart(ifInBound.elseBlock());
          builder.create<scf::YieldOp>(loc, constantI1(builder, loc, false));
        }
        builder.create<scf::ConditionOp>(loc, ifInBound.getResults()[0], ivs);
      },
      /*afterBuilder=*/
      [](OpBuilder &builder, Location loc, ValueRange ivs) {
        // pos ++
        Value nextPos = builder.create<arith::AddIOp>(
            loc, ivs[0], constantIndex(builder, loc, 1));
        builder.create<scf::YieldOp>(loc, nextPos);
      });
  // Return the segment high.
  return whileOp.getResult(0);
}

Value LoopEmitter::genSparseCrd(OpBuilder &builder, Location loc, TensorId tid,
                                Level dstLvl) {
  Value crd = constantIndex(builder, loc, 0);
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
    crd = builder.create<arith::AddIOp>(loc, crd, off);
    if (i != reassocSize - 1) {
      crd = builder.create<arith::MulIOp>(loc, crd,
                                          this->lvlSizes[tid][reassoc[i + 1]]);
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

  this->dependentLvlMap.assign(
      numTensors, std::vector<std::vector<std::pair<TensorId, Level>>>());

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
    dependentLvlMap[tid].assign(lvlRank,
                                std::vector<std::pair<TensorId, Level>>());
    if (dimGetter) {
      auto reassoc = collapseReassoc[tid];
      Level dstRank = reassoc ? reassoc.size() : lvlRank;
      for (Level l = 0; l < dstRank; l++) {
        dependentLvlMap[tid][l] = dimGetter(tid, l);
        // TODO: View-base collapse and dependent index reduction are not
        // compatible right now.
        assert(!reassoc || dependentLvlMap[tid][l].empty());
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
    const auto rtp = tensor.getType().dyn_cast<RankedTensorType>();
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
      if (isCompressedDLT(lvlTp)) {
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
}

void LoopEmitter::enterNewLoopSeq(OpBuilder &builder, Location loc,
                                  ArrayRef<TensorId> tids,
                                  ArrayRef<Level> lvls) {
  // TODO: sort
  assert(loopSeqStack.size() == loopStack.size());
  // Universal Index starts from 0.
  loopSeqStack.emplace_back(constantIndex(builder, loc, 0));
  // Prepares for all the tensors used in the current loop sequence.
  assert(tids.size() == lvls.size());
  for (auto [tid, lvl] : llvm::zip(tids, lvls))
    prepareLoopOverTensorAtLvl(builder, loc, tid, lvl);
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
    return builder.create<arith::AddIOp>(
        loc, genAffine(builder, loc, binOp.getLHS()),
        genAffine(builder, loc, binOp.getRHS()));
  }
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::MulIOp>(
        loc, genAffine(builder, loc, binOp.getLHS()),
        genAffine(builder, loc, binOp.getRHS()));
  }
  case AffineExprKind::Constant: {
    int64_t c = a.cast<AffineConstantExpr>().getValue();
    return constantIndex(builder, loc, c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

Operation *LoopEmitter::enterLoopOverTensorAtLvl(
    OpBuilder &builder, Location loc, ArrayRef<TensorId> tids,
    ArrayRef<Level> lvls, MutableArrayRef<Value> reduc, bool isParallel) {
  // TODO: support multiple return on parallel for?
  assert(!isParallel || reduc.size() <= 1);
  bool isSparseInput = false;
  TensorId tid = tids.front();
  Level dstLvl = lvls.front();
  assert(tids.size() == lvls.size());
  for (auto [t, l] : llvm::zip(tids, lvls)) {
    // TODO: this check for validity of the (t,l) pairs should be
    // checked/enforced at the callsites, if possible.
    assert(isValidLevel(t, l));
    assert(!coords[t][l]); // We cannot re-enter the same level
    const auto lvlTp = lvlTypes[t][l];
    const bool isSparse = isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp);
    // Must be a recognizable level-type.
    assert(isSparse || isDenseDLT(lvlTp));
    // We can at most have one sparse input, otherwise, a while loop is required
    // to co-iterate multiple sparse tensors.
    assert(!isSparseInput || !isSparse);
    if (isSparse) {
      tid = t;
      dstLvl = l;
    }
    isSparseInput = isSparseInput || isSparse;
  }

  const auto reassoc = getCollapseReassociation(tid, dstLvl);
  // TODO: support dynamic slices.
  // Use the first source-level here to build the loop bound (which is
  // also the biggest range).
  const Level srcLvl = reassoc.front();
  const Value step = constantIndex(builder, loc, 1);
  /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
  const Value lo = isSparseInput ? posits[tid][srcLvl]  // current position
                                 : loopSeqStack.back(); // universal index
  const Value hi = highs[tid][srcLvl];

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
  if (isSparseInput) {
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

  if (isSparseSlices[tid] && isSparseInput) {
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
      builder.create<scf::YieldOp>(loc, ifOp.getResults());
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      // On mismatch.
      builder.create<scf::YieldOp>(loc, reduc);
    }
    // Set the insertion point to matched branch.
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    crd = trans;
  }

  assert(crd);
  coords[tid][srcLvl] = crd;
  // NOTE: we can also prepare for next level here in advance
  // Push the loop into stack
  loopStack.emplace_back(ArrayRef<TensorId>(tid), ArrayRef<Level>(srcLvl), loop,
                         builder.getInsertionBlock(), crd, loopTag);
  // Emit extra locals.
  emitExtraLocalsForTensorsAtDenseLvls(builder, loc, tids, lvls);

  return loop;
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
  const Value step = constantIndex(builder, loc, 1);
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
  auto pred = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, crd,
                                            expected);
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
    builder.create<scf::YieldOp>(loc, ifOp.getResults());
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    // On mismatch.
    builder.create<scf::YieldOp>(loc, reduc);
  }
  // Set the insert point to matched branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // NOTE: we can also prepare for next lvl here in advance
  // Push the loop into stack
  loopStack.emplace_back(ArrayRef<TensorId>(tid), ArrayRef<Level>(lvl), forOp,
                         builder.getInsertionBlock(), crd, nullptr);
  return forOp;
}

void LoopEmitter::genDenseAffineAddress(OpBuilder &builder, Location loc,
                                        TensorId tid, Level lvl,
                                        AffineExpr lvlExpr) {
  assert(isDenseDLT(lvlTypes[tid][lvl]));
  // For dense levels, the level-coordinate also serves as the position.
  Value lvlCrd = genAffine(builder, loc, lvlExpr);
  posits[tid][lvl] = genAddress(builder, loc, tid, lvl, lvlCrd);
}

Operation *LoopEmitter::enterCoIterationOverTensorsAtLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorId> tids,
    ArrayRef<Level> lvls, bool needsUniv, MutableArrayRef<Value> reduc) {
  assert(tids.size() == lvls.size());
  SmallVector<Type> types;
  SmallVector<Value> operands;
  // Construct the while-loop with a parameter for each coordinate.
  const Type indexType = builder.getIndexType();
  for (auto [tid, lvl] : llvm::zip(tids, lvls)) {
    const auto lvlTp = lvlTypes[tid][lvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp)) {
      const auto reassoc = getCollapseReassociation(tid, lvl);
      for (unsigned i = 0, e = reassoc.size() - 1; i < e; i++) {
        if (!isUniqueDLT(lvlTypes[tid][reassoc[i]])) {
          // This is the segment high for each non-unique levels.
          types.push_back(indexType);
          operands.push_back(constantIndex(builder, loc, 0));
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
    operands.push_back(loopSeqStack.back());
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
  for (auto [t, lvl] : llvm::zip(tids, lvls)) {
    const TensorId tid = t; // Why `t` can not be captured by lambda?
    const auto lvlTp = lvlTypes[tid][lvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp)) {
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
      Value opc = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                op1, op2);
      cond = cond ? builder.create<arith::AndIOp>(loc, cond, opc) : opc;
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
  for (auto [tid, lvl] : llvm::zip(tids, lvls)) {
    // Prepares for next level.
    const auto lvlTp = lvlTypes[tid][lvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp)) {
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
      Value nextPos = builder.create<arith::AddIOp>(
          loc, yields[idx], constantIndex(builder, loc, 1));
      yields[idx] =
          builder.create<arith::SelectOp>(loc, pred, yields[idx], nextPos);
    }

    Value pred = slicesPreds.front().first;
    for (int i = 1, e = slicesPreds.size(); i < e; i++) {
      pred = builder.create<arith::AndIOp>(loc, pred, slicesPreds[i].first);
    }
    auto ifOp = builder.create<scf::IfOp>(loc, types, pred, /*else*/ true);
    ifOp->setAttr(getLoopEmitterLoopAttrName(),
                  StringAttr::get(builder.getContext(), "slice"));
    builder.create<scf::YieldOp>(loc, ifOp->getResults());
    assert(types.size() == yields.size());
    // If not all slices are legit
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, yields);

    // If all slices are legit, start the user generated code.
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }

  Value min;
  // Finds the minimum coordinate
  if (!needsUniv) {
    for (auto [tid, lvl] : llvm::zip(tids, lvls)) {
      const auto lvlTp = lvlTypes[tid][lvl];
      if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp)) {
        const auto crd = coords[tid][lvl];
        if (min) {
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, crd, min);
          min = builder.create<arith::SelectOp>(loc, cmp, crd, min);
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
  loopStack.emplace_back(tids, lvls, whileOp, builder.getInsertionBlock(), min,
                         loopTag);
  assert(loopStack.size() == loopSeqStack.size());

  for (auto [tid, dstLvl] : llvm::zip(tids, lvls)) {
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
          builder.create<scf::YieldOp>(loc,
                                       genSegmentHigh(builder, loc, tid, srcLvl,
                                                      pos, highs[tid][srcLvl]));
          // Else, resues the same segment high.
          builder.setInsertionPointToStart(ifNewSegHi.elseBlock());
          builder.create<scf::YieldOp>(loc, oldSegHi);
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
  emitExtraLocalsForTensorsAtDenseLvls(builder, loc, tids, lvls);

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

  const Value c0 = constantIndex(builder, loc, 0);
  const Value c1 = constantIndex(builder, loc, 1);
  for (const Level srcLvl : getCollapseReassociation(tid, dstLvl)) {
    // Either the first level, or the previous level has been set.
    /// FIXME: See the [CLARIFY_POSITS_LVL] note in the header.
    assert(srcLvl == 0 || posits[tid][srcLvl - 1]);
    if (!isCompressedDLT(lvlTp) && !isSingletonDLT(lvlTp))
      continue;
    if (isCompressedDLT(lvlTp)) {
      const Value mem = positionsBuffers[tid][srcLvl];

      const Value pLo = srcLvl == 0 ? c0 : posits[tid][srcLvl - 1];
      posits[tid][srcLvl] = genIndexLoad(builder, loc, mem, pLo);

      const Value pHi = builder.create<arith::AddIOp>(loc, pLo, c1);
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
              : builder.create<arith::AddIOp>(loc, pLo, c1);
      return;
    }
  }

  llvm_unreachable("Unrecognized level-type!");
}

void LoopEmitter::emitExtraLocalsForTensorsAtDenseLvls(OpBuilder &builder,
                                                       Location loc,
                                                       ArrayRef<TensorId> tids,
                                                       ArrayRef<Level> lvls) {
  // Initialize dense positions. Note that we generate dense coordinates of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  assert(tids.size() == lvls.size());
  for (auto [tid, lvl] : llvm::zip(tids, lvls)) {
    if (isDenseDLT(lvlTypes[tid][lvl])) {
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
  for (auto [tid, lvl] : llvm::zip(loopInfo.tids, loopInfo.lvls)) {
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
  Value one = constantIndex(builder, loc, 1);
  for (auto [tid, dstLvl] : llvm::zip(loopInfo.tids, loopInfo.lvls)) {
    const auto lvlTp = lvlTypes[tid][dstLvl];
    if (isCompressedDLT(lvlTp) || isSingletonDLT(lvlTp)) {
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
      Value cmp =
          builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, crd, iv);
      // If the loop contains a coiteration with non-unique level, we fast
      // forward all the duplicated coords by setting the position to the
      // segment high.
      Value add = !isUniqueDLT(lvlTypes[tid][reassoc.back()])
                      ? segHi[tid][reassoc.back()]
                      : builder.create<arith::AddIOp>(loc, pos, one);

      operands.push_back(builder.create<arith::SelectOp>(loc, cmp, add, pos));
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
  if (operands.size() < whileOp.getNumResults()) {
    assert(operands.size() + 1 == whileOp.getNumResults());
    // The last one is the universial index.
    operands.push_back(builder.create<arith::AddIOp>(loc, iv, one));
    // update the loop starting point of current loop sequence
    loopSeqStack.back() = whileOp->getResult(o++);
  }

  assert(o == operands.size());
  builder.create<scf::YieldOp>(loc, operands);
  builder.setInsertionPointAfter(whileOp);
}

void LoopEmitter::exitCurrentLoop(RewriterBase &rewriter, Location loc,
                                  MutableArrayRef<Value> reduc) {
  // Clean up the values, it would help use to discover potential bug at a
  // earlier stage (instead of silently using a wrong value).
  const LoopInfo &loopInfo = loopStack.back();
  assert(loopInfo.tids.size() == loopInfo.lvls.size());
  SmallVector<Value> red;
  if (llvm::isa<scf::WhileOp>(loopInfo.loop)) {
    exitWhileLoop(rewriter, loc, reduc);
  } else {
    exitForLoop(rewriter, loc, reduc);
  }

  assert(loopStack.size() == loopSeqStack.size());
  loopStack.pop_back();
}
