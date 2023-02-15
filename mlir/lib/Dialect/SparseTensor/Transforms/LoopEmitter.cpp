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
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// File local helper functions.
//===----------------------------------------------------------------------===//

/// Generates a pointer/index load from the sparse storage scheme. Narrower
/// data types need to be zero extended before casting the value into the
/// index type used for looping and indexing.
static Value genIndexLoad(OpBuilder &builder, Location loc, Value ptr,
                          Value s) {
  // For the scalar case, we simply zero extend narrower indices into 64-bit
  // values before casting to index without a performance penalty. Here too,
  // however, indices that already are 64-bit, in theory, cannot express the
  // full range as explained above.
  Value load = builder.create<memref::LoadOp>(loc, ptr, s);
  if (!load.getType().isa<IndexType>()) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), load);
    load =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), load);
  }
  return load;
}

// TODO: Support dynamic sized slice.
static Value getSliceOffset(OpBuilder &builder, Location loc,
                            SparseTensorEncodingAttr enc, unsigned lvl) {
  return constantIndex(builder, loc, *enc.getStaticLvlSliceOffset(lvl));
}

static Value getSliceSize(OpBuilder &builder, Location loc,
                          SparseTensorEncodingAttr enc, unsigned lvl) {
  return constantIndex(builder, loc, *enc.getStaticLvlSliceSize(lvl));
}

static Value getSliceStride(OpBuilder &builder, Location loc,
                            SparseTensorEncodingAttr enc, unsigned lvl) {
  return constantIndex(builder, loc, *enc.getStaticLvlSliceStride(lvl));
}

// Converts a coordinate relative to the slice to the coordinate relative
// to the underlying tensor.
static Value toSliceCoord(OpBuilder &builder, Location loc, Value v,
                          SparseTensorEncodingAttr enc, unsigned lvl) {

  Value stride = getSliceStride(builder, loc, enc, lvl);
  Value offset = getSliceOffset(builder, loc, enc, lvl);
  // iv = iv * stride + offset
  v = builder.create<arith::MulIOp>(loc, v, stride);
  v = builder.create<arith::AddIOp>(loc, v, offset);
  return v;
}

// Converts a coordinate relative to the underlying tensor to the coordinate
// relative to the slice, returns a extra reminder value
static std::pair<Value, Value> fromSliceCoord(OpBuilder &builder, Location loc,
                                              Value v,
                                              SparseTensorEncodingAttr enc,
                                              unsigned lvl) {
  Value stride = getSliceStride(builder, loc, enc, lvl);
  Value offset = getSliceOffset(builder, loc, enc, lvl);
  // iv = (iv - offset) / stride
  v = builder.create<arith::SubIOp>(loc, v, offset);
  Value rem = builder.create<arith::RemUIOp>(loc, v, stride);
  v = builder.create<arith::DivUIOp>(loc, v, stride);
  return std::make_pair(v, rem);
}

//===----------------------------------------------------------------------===//
// Sparse tensor loop emitter class implementations
//===----------------------------------------------------------------------===//

Value LoopEmitter::genAddress(OpBuilder &builder, Location loc, size_t tid,
                              size_t dim, Value iv) {
  Value p = dim == 0 ? constantIndex(builder, loc, 0) : pidxs[tid][dim - 1];
  Value mul = builder.create<arith::MulIOp>(loc, highs[tid][dim], p);
  if (isSparseSlices[tid]) {
    auto enc = getSparseTensorEncoding(tensors[tid].getType());
    iv = toSliceCoord(builder, loc, iv, enc, dim);
  }
  Value add = builder.create<arith::AddIOp>(loc, mul, iv);
  return add;
}

LoopEmitter::LoopEmitter(ValueRange tensors, StringAttr loopTag, bool hasOutput,
                         bool isSparseOut, ArrayRef<unsigned> topSort) {
  initialize(tensors, loopTag, hasOutput, isSparseOut, topSort);
}

void LoopEmitter::initialize(ValueRange tensors, StringAttr loopTag,
                             bool hasOutput, bool isSparseOut,
                             ArrayRef<unsigned> topSort) {
  // First initializes fields.
  this->loopTag = loopTag;
  this->hasOutput = hasOutput;
  this->isSparseOut = isSparseOut;
  this->tensors.assign(tensors.begin(), tensors.end());
  this->isSparseSlices.assign(tensors.size(), false);
  this->dimTypes.assign(tensors.size(), std::vector<DimLevelType>());
  this->pidxs.assign(tensors.size(), std::vector<Value>());
  this->coord.assign(tensors.size(), std::vector<Value>());
  this->highs.assign(tensors.size(), std::vector<Value>());
  this->ptrBuffer.assign(tensors.size(), std::vector<Value>());
  this->idxBuffer.assign(tensors.size(), std::vector<Value>());
  this->valBuffer.assign(tensors.size(), nullptr);
  this->loopStack.reserve(topSort.size());
  this->sparsiferLoopLvlMap.assign(topSort.size(), 0);

  for (size_t tid = 0, e = tensors.size(); tid < e; tid++) {
    auto t = tensors[tid];
    // a scalar or 0-dimension tensors
    if (isZeroRankedTensorOrScalar(t.getType()))
      continue;
    auto rtp = getRankedTensorType(t);
    auto rank = static_cast<size_t>(rtp.getRank());
    auto enc = getSparseTensorEncoding(rtp);
    // We always treat sparse output tensor as dense so that we always iterate
    // it based on dim size.
    if (enc && !(isOutputTensor(tid) && isSparseOut)) {
      isSparseSlices[tid] = enc.isSlice();
      for (auto dimTp : enc.getDimLevelType())
        dimTypes[tid].push_back(dimTp);
    } else
      dimTypes[tid].assign(rank, DimLevelType::Dense);

    // Initialize using empty value.
    pidxs[tid].assign(rank, Value());
    coord[tid].assign(rank, Value());
    highs[tid].assign(rank, Value());
    ptrBuffer[tid].assign(rank, Value());
    idxBuffer[tid].assign(rank, Value());
  }

  // FIXME: This map should be maintained outside loop emitter.
  for (unsigned i = 0, e = topSort.size(); i < e; i++) {
    // This is an inverse map of the topologically sorted loop index from
    // sparsifier. This is needed to map the AffineDimExpr back to the loopStack
    // index used in loop emitter.
    sparsiferLoopLvlMap[topSort[i]] = i;
  }
}

void LoopEmitter::initializeLoopEmit(OpBuilder &builder, Location loc,
                                     LoopEmitter::OutputUpdater updater) {
  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and obtain dense or sparse buffer(s).
  for (size_t t = 0, e = tensors.size(); t < e; t++) {
    const auto tensor = tensors[t];
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
      assert(!ptrBuffer[t][l] && !idxBuffer[t][l] && !highs[t][l]);
      const auto dlt = dimTypes[t][l];
      // Handle sparse storage schemes.
      if (isCompressedDLT(dlt)) {
        // Generate sparse primitives to obtains pointer and indices.
        ptrBuffer[t][l] = genToPointers(builder, loc, tensor, l);
        idxBuffer[t][l] = genToIndices(builder, loc, tensor, l, cooStart);
      } else if (isSingletonDLT(dlt)) {
        // Singleton dimension, fetch indices.
        idxBuffer[t][l] = genToIndices(builder, loc, tensor, l, cooStart);
      } else {
        // Dense dimension, nothing to fetch.
        assert(isDenseDLT(dlt));
      }

      // Find upper bound in current dimension.
      // FIXME: `toOrigDim` is deprecated
      const Dimension d = toOrigDim(enc, l);
      highs[t][l] = mlir::linalg::createOrFoldDimOp(builder, loc, tensor, d);
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
      // We also need the value buffer for annotated all dense `sparse` tensor.
      valBuffer[t] = genToValues(builder, loc, tensor);
    }
    // NOTE: we can also prepare for 0 dim here in advance, this will hosit
    // some loop preparation from tensor iteration, but will also (undesirably)
    // hosit the code ouside if conditions.
  }
}

void LoopEmitter::enterNewLoopSeq(OpBuilder &builder, Location loc,
                                  ArrayRef<size_t> tids,
                                  ArrayRef<size_t> dims) {
  assert(loopSeqStack.size() == loopStack.size());
  // Universal Index starts from 0.
  loopSeqStack.emplace_back(constantIndex(builder, loc, 0));
  // Prepares for all the tensors used in the current loop sequence.
  for (auto [tid, dim] : llvm::zip(tids, dims))
    prepareLoopOverTensorAtDim(builder, loc, tid, dim);
}

Value LoopEmitter::genAffine(OpBuilder &builder, AffineExpr a, Location loc) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    return loopStack[sparsiferLoopLvlMap[idx]].iv;
  }
  case AffineExprKind::Add: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::AddIOp>(
        loc, genAffine(builder, binOp.getLHS(), loc),
        genAffine(builder, binOp.getRHS(), loc));
  }
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::MulIOp>(
        loc, genAffine(builder, binOp.getLHS(), loc),
        genAffine(builder, binOp.getRHS(), loc));
  }
  case AffineExprKind::Constant: {
    int64_t c = a.cast<AffineConstantExpr>().getValue();
    return constantIndex(builder, loc, c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

Operation *LoopEmitter::enterLoopOverTensorAtDim(
    OpBuilder &builder, Location loc, ArrayRef<size_t> tids,
    ArrayRef<size_t> dims, MutableArrayRef<Value> reduc, bool isParallel) {
  // TODO: support multiple return on parallel for?
  assert(!isParallel || reduc.size() <= 1);
  bool isSparseInput = false;
  size_t tid = tids.front(), dim = dims.front();
  for (auto [t, d] : llvm::zip(tids, dims)) {
    assert(dimTypes[t].size() > d); // Must be a valid tid, dim pair
    assert(!coord[t][d]);           // We cannot re-enter the same level
    auto dimType = dimTypes[t][d];
    // Must be a recognizable DLT.
    assert(isDenseDLT(dimType) || isCompressedDLT(dimType) ||
           isSingletonDLT(dimType));
    bool isSparse = isCompressedDLT(dimType) || isSingletonDLT(dimType);
    // We can at most have one sparse input, otherwise, a while loop is required
    // to co-iterate multiple sparse tensors.
    assert(!isSparseInput || !isSparse);
    if (isSparse) {
      tid = t;
      dim = d;
    }
    isSparseInput = isSparseInput || isSparse;
  }

  auto enc = getSparseTensorEncoding(tensors[tid].getType());
  // TODO: support dynamic slices.
  Value step = constantIndex(builder, loc, 1);
  Value lo = isSparseInput ? pidxs[tid][dim]      // current offset
                           : loopSeqStack.back(); // universal index
  Value hi = highs[tid][dim];

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
    // used as a `special handle` to (temporarily) represent them. The
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

  Value c;
  if (isSparseInput) {
    pidxs[tid][dim] = iv;
    // Generating a load on the indices array yields the coordinate.
    Value ptr = idxBuffer[tid][dim];
    c = genIndexLoad(builder, loc, ptr, iv);
  } else {
    // Dense tensor, the coordinates is the inducation variable.
    c = iv;
  }

  if (isSparseSlices[tid] && isSparseInput) {
    // For sparse level slices, we need to filter out invalid coordinates that
    // are not included in the slice.
    std::pair<Value, Value> trans = fromSliceCoord(builder, loc, c, enc, dim);
    SmallVector<Type> types;
    for (Value red : reduc)
      types.push_back(red.getType());

    // First, coord >= offset (TODO: seems unsigned >= 0 won't be folded, skip
    // the check if the offset is zero).
    auto geOff =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, c,
                                      getSliceOffset(builder, loc, enc, dim));
    // Second, coords < length
    auto ltLen = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, trans.first,
        getSliceSize(builder, loc, enc, dim));

    // Third, rem == 0; confirmed that (a % 1) will be folded to 0
    auto fitStride = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, trans.second,
        constantIndex(builder, loc, 0));

    auto pred = builder.create<arith::AndIOp>(loc, geOff, ltLen);
    pred = builder.create<arith::AndIOp>(loc, pred, fitStride);
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
    // Set the insertion point to matched branch.
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    c = trans.first;
  }

  assert(c);
  coord[tid][dim] = c;
  // NOTE: we can also prepare for next dim here in advance
  // Push the loop into stack
  loopStack.emplace_back(ArrayRef<size_t>(tid), ArrayRef<size_t>(dim), loop,
                         coord[tid][dim], loopTag);
  // Emit extra locals.
  emitExtraLocalsForTensorsAtDenseDims(builder, loc, tids, dims);

  return loop;
}

Operation *LoopEmitter::enterFilterLoopOverTensorAtDim(
    OpBuilder &builder, Location loc, size_t tid, size_t dim, AffineExpr affine,
    MutableArrayRef<Value> reduc) {
  assert(!affine.isa<AffineDimExpr>() && !isDenseDLT(dimTypes[tid][dim]));
  assert(dimTypes[tid].size() > dim);
  // We can not re-enter the same level.
  assert(!coord[tid][dim]);

  Value step = constantIndex(builder, loc, 1);

  Value lo = pidxs[tid][dim];
  Value hi = highs[tid][dim];

  // TODO: We should instead use a whileOp for filter loop to allow early
  // break when exceeding (for ordered dimensions).
  // TODO: There are many other potiential opportunities that we might apply in
  // the future. E.g., we could use binary search to located the pointer index.
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lo, hi, step, reduc);

  // In-place update on the reduction variable vector.
  assert(forOp.getNumRegionIterArgs() == reduc.size());
  for (int i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = forOp.getRegionIterArg(i);

  builder.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();

  pidxs[tid][dim] = iv;
  // Generating a load on the indices array yields the coordinate.
  Value ptr = idxBuffer[tid][dim];
  coord[tid][dim] = genIndexLoad(builder, loc, ptr, iv);

  // Generate an if condition to filter out indices that is not equal to the
  // result of the affine expression.
  Value expected = genAffine(builder, affine, loc);
  auto pred = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            coord[tid][dim], expected);
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

  // NOTE: we can also prepare for next dim here in advance
  // Push the loop into stack
  loopStack.emplace_back(ArrayRef<size_t>(tid), ArrayRef<size_t>(dim), forOp,
                         coord[tid][dim], nullptr);
  return forOp;
}

void LoopEmitter::genDenseAffineAddressAtCurLevel(OpBuilder &builder,
                                                  Location loc, size_t tid,
                                                  size_t dim,
                                                  AffineExpr affine) {
  Value affineV = genAffine(builder, affine, loc);
  pidxs[tid][dim] = genAddress(builder, loc, tid, dim, affineV);
}

Operation *LoopEmitter::enterCoIterationOverTensorsAtDims(
    OpBuilder &builder, Location loc, ArrayRef<size_t> tids,
    ArrayRef<size_t> dims, bool needsUniv, MutableArrayRef<Value> reduc) {
  assert(tids.size() == dims.size());
  SmallVector<Type> types;
  SmallVector<Value> operands;
  // Construct the while-loop with a parameter for each index.
  Type indexType = builder.getIndexType();
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    if (isCompressedDLT(dimTypes[tid][dim]) ||
        isSingletonDLT(dimTypes[tid][dim])) {
      assert(pidxs[tid][dim]);
      types.push_back(indexType);
      operands.push_back(pidxs[tid][dim]);
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
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    if (isCompressedDLT(dimTypes[tid][dim]) ||
        isSingletonDLT(dimTypes[tid][dim])) {
      Value op1 = before->getArgument(o);
      Value op2 = highs[tid][dim];
      Value opc = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                op1, op2);
      cond = cond ? builder.create<arith::AndIOp>(loc, cond, opc) : opc;
      // Update
      pidxs[tid][dim] = after->getArgument(o++);
    }
  }
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());

  // Generates while body.
  builder.setInsertionPointToStart(&whileOp.getAfter().front());
  Value min;
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    // Prepares for next level.
    if (isCompressedDLT(dimTypes[tid][dim]) ||
        isSingletonDLT(dimTypes[tid][dim])) {
      Value ptr = idxBuffer[tid][dim];
      Value s = pidxs[tid][dim];
      Value load = genIndexLoad(builder, loc, ptr, s);
      coord[tid][dim] = load;
      if (!needsUniv) {
        if (min) {
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, load, min);
          min = builder.create<arith::SelectOp>(loc, cmp, load, min);
        } else {
          min = load;
        }
      }
    }
  }

  if (needsUniv) {
    assert(!min);
    // Otherwise, universal index is the minimal pidx.
    min = after->getArguments().back();
  }

  // Sets up the loop stack.
  loopStack.emplace_back(tids, dims, whileOp, min, loopTag);
  assert(loopStack.size() == loopSeqStack.size());

  // Emits extra locals
  emitExtraLocalsForTensorsAtDenseDims(builder, loc, tids, dims);

  // Updates reduction variables
  assert(after->getNumArguments() == o + reduc.size() + (needsUniv ? 1 : 0));
  // In-place update on reduction variable.
  for (unsigned i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = after->getArgument(o + i);

  return whileOp;
}

void LoopEmitter::prepareLoopOverTensorAtDim(OpBuilder &builder, Location loc,
                                             size_t tid, size_t dim) {
  assert(dimTypes[tid].size() > dim);
  auto dimType = dimTypes[tid][dim];

  if (isDenseDLT(dimType))
    return;

  // Either the first dimension, or the previous dimension has been set.
  assert(dim == 0 || pidxs[tid][dim - 1]);
  Value c0 = constantIndex(builder, loc, 0);
  Value c1 = constantIndex(builder, loc, 1);
  if (isCompressedDLT(dimType)) {
    Value ptr = ptrBuffer[tid][dim];

    Value pLo = dim == 0 ? c0 : pidxs[tid][dim - 1];
    pidxs[tid][dim] = genIndexLoad(builder, loc, ptr, pLo);

    Value pHi = builder.create<arith::AddIOp>(loc, pLo, c1);
    highs[tid][dim] = genIndexLoad(builder, loc, ptr, pHi);
    return;
  }
  if (isSingletonDLT(dimType)) {
    Value pLo = dim == 0 ? c0 : pidxs[tid][dim - 1];
    Value pHi = builder.create<arith::AddIOp>(loc, pLo, c1);

    pidxs[tid][dim] = pLo;
    highs[tid][dim] = pHi;
    return;
  }

  llvm_unreachable("Unrecognizable dimesion type!");
}

void LoopEmitter::emitExtraLocalsForTensorsAtDenseDims(OpBuilder &builder,
                                                       Location loc,
                                                       ArrayRef<size_t> tids,
                                                       ArrayRef<size_t> dims) {
  // Initialize dense positions. Note that we generate dense indices of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    if (isDenseDLT(dimTypes[tid][dim])) {
      auto enc = getSparseTensorEncoding(tensors[tid].getType());
      if (enc && !isSparseOutput(tid)) {
        bool validPidx = dim == 0 || pidxs[tid][dim - 1];
        if (!validPidx) {
          // We might not find the pidx for the sparse output tensor as it is
          // unconditionally required by the sparsification.
          assert(isOutputTensor(tid));
          continue;
        }
        pidxs[tid][dim] =
            genAddress(builder, loc, tid, dim, loopStack.back().iv);
        // NOTE: we can also prepare for next dim here in advance
      }
    }
  }
}

void LoopEmitter::exitForLoop(RewriterBase &rewriter, Location loc,
                              MutableArrayRef<Value> reduc) {
  LoopLevelInfo &loopInfo = loopStack.back();
  auto &dims = loopStack.back().dims;
  auto &tids = loopStack.back().tids;
  auto forOp = llvm::dyn_cast<scf::ForOp>(loopInfo.loop);
  if (forOp) {
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
      // The reduction expression should be the only user of the reduction val
      // inside the parallel for.
      unsigned numUsers = 0;
      for (Operation *op : redVal.getUsers()) {
        if (op->getParentOp() == parOp)
          numUsers++;
      }
      assert(numUsers == 1);
      (void)numUsers; // to silence unused variable warning in release build

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
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    // Reset to null.
    coord[tid][dim] = Value();
    pidxs[tid][dim] = Value();
    // Dense dimension, high is fixed.
    if (!isDenseDLT(dimTypes[tid][dim]))
      highs[tid][dim] = Value();
  }
}

void LoopEmitter::exitCoIterationLoop(OpBuilder &builder, Location loc,
                                      MutableArrayRef<Value> reduc) {
  auto whileOp = llvm::cast<scf::WhileOp>(loopStack.back().loop);
  auto &dims = loopStack.back().dims;
  auto &tids = loopStack.back().tids;
  Value iv = loopStack.back().iv;
  // Generation while loop induction at the end.
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
  // Finalize the induction. Note that the induction could be performed
  // in the individual if-branches to avoid re-evaluating the conditions.
  // However, that would result in a rather elaborate forest of yield
  // instructions during code generation. Moreover, performing the induction
  // after the if-statements more closely resembles code generated by TACO.
  unsigned o = 0;
  SmallVector<Value> operands;
  Value one = constantIndex(builder, loc, 1);
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    if (isCompressedDLT(dimTypes[tid][dim]) ||
        isSingletonDLT(dimTypes[tid][dim])) {
      Value op1 = coord[tid][dim];
      Value op3 = pidxs[tid][dim];
      Value cmp =
          builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, op1, iv);
      Value add = builder.create<arith::AddIOp>(loc, op3, one);
      operands.push_back(builder.create<arith::SelectOp>(loc, cmp, add, op3));
      // Following loops continue iteration from the break point of the
      // current while loop.
      pidxs[tid][dim] = whileOp->getResult(o++);
      // The coordinates are invalid now.
      coord[tid][dim] = nullptr;
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
  LoopLevelInfo &loopInfo = loopStack.back();
  assert(loopInfo.tids.size() == loopInfo.dims.size());
  SmallVector<Value> red;
  if (llvm::isa<scf::WhileOp>(loopInfo.loop)) {
    exitCoIterationLoop(rewriter, loc, reduc);
  } else {
    exitForLoop(rewriter, loc, reduc);
  }

  assert(loopStack.size() == loopSeqStack.size());
  loopStack.pop_back();
}
