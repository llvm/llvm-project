//===- CodegenUtils.cpp - Utilities for generating MLIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

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

/// If the tensor is a sparse constant, generates and returns the pair of
/// the constants for the indices and the values.
static Optional<std::pair<Value, Value>>
genSplitSparseConstant(OpBuilder &builder, Location loc, Value tensor) {
  if (auto constOp = tensor.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = constOp.getValue().dyn_cast<SparseElementsAttr>()) {
      DenseElementsAttr indicesAttr = attr.getIndices();
      Value indices = builder.create<arith::ConstantOp>(loc, indicesAttr);
      DenseElementsAttr valuesAttr = attr.getValues();
      Value values = builder.create<arith::ConstantOp>(loc, valuesAttr);
      return std::make_pair(indices, values);
    }
  }
  return {};
}

/// Generates the code to copy the index at indices[ivs] to ind, and return
/// the value at value[ivs].
static Value genIndexAndValueForSparse(OpBuilder &builder, Location loc,
                                       Value indices, Value values,
                                       SmallVectorImpl<Value> &indicesArray,
                                       ValueRange ivs, unsigned rank) {
  for (unsigned i = 0; i < rank; i++) {
    Value idx = constantIndex(builder, loc, i);
    Value val = builder.create<tensor::ExtractOp>(loc, indices,
                                                  ValueRange{ivs[0], idx});
    val = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
    // builder.create<memref::StoreOp>(loc, val, ind, idx);
    indicesArray.push_back(val);
  }
  return builder.create<tensor::ExtractOp>(loc, values, ivs[0]);
}

/// Generates the code to read the value from tensor[ivs], and conditionally
/// stores the indices ivs to the memory in ind. The generated code looks like
/// the following and the insertion point after this routine is inside the
/// if-then branch behind the assignment to ind. This is to ensure that the
/// code that uses the ind, such as an addEltX call generated after, is inside
/// the if-then branch.
///    if (tensor[ivs] != 0)
///      ind = ivs
static Value genIndexAndValueForDense(OpBuilder &builder, Location loc,
                                      Value tensor,
                                      SmallVectorImpl<Value> &indicesArray,
                                      ValueRange ivs) {
  Value val = genValueForDense(builder, loc, tensor, ivs);
  indicesArray.append(ivs.begin(), ivs.end());
  return val;
}

//===----------------------------------------------------------------------===//
// Sparse tensor loop emitter class implementations
//===----------------------------------------------------------------------===//

SparseTensorLoopEmitter::SparseTensorLoopEmitter(ValueRange tensors,
                                                 bool hasOutput,
                                                 bool isSparseOut)
    : hasOutput(hasOutput), isSparseOut(isSparseOut),
      tensors(tensors.begin(), tensors.end()), dimTypes(tensors.size()),
      pidxs(tensors.size()), coord(tensors.size()), highs(tensors.size()),
      ptrBuffer(tensors.size()), idxBuffer(tensors.size()),
      valBuffer(tensors.size()), loopStack() {
  for (size_t tid = 0, e = tensors.size(); tid < e; tid++) {
    auto t = tensors[tid];
    // a scalar or 0-dimension tensors
    if (isZeroRankedTensorOrScalar(t.getType()))
      continue;
    auto rtp = t.getType().cast<RankedTensorType>();
    auto rank = static_cast<size_t>(rtp.getRank());
    auto enc = getSparseTensorEncoding(rtp);
    // We always treat sparse output tensor as dense so that we always iterate
    // it based on dim size.
    if (enc && !(isOutputTensor(tid) && isSparseOut))
      for (auto dimTp : enc.getDimLevelType())
        dimTypes[tid].push_back(dimTp);
    else
      dimTypes[tid].assign(rank, DimLevelType::Dense);

    // Initialize using empty value.
    pidxs[tid].assign(rank, Value());
    coord[tid].assign(rank, Value());
    highs[tid].assign(rank, Value());
    ptrBuffer[tid].assign(rank, Value());
    idxBuffer[tid].assign(rank, Value());
  }
}

void SparseTensorLoopEmitter::initializeLoopEmit(
    OpBuilder &builder, Location loc,
    SparseTensorLoopEmitter::OutputUpdater updater) {
  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and obtain dense or sparse buffer(s).
  for (size_t t = 0, e = tensors.size(); t < e; t++) {
    auto tensor = tensors[t];
    auto rtp = tensor.getType().dyn_cast<RankedTensorType>();
    if (!rtp)
      // Skips only scalar, zero ranked tensor still need to be bufferized and
      // (probably) filled with zeros by users.
      continue;
    auto rank = rtp.getRank();
    auto shape = rtp.getShape();
    auto enc = getSparseTensorEncoding(rtp);
    auto dynShape = {ShapedType::kDynamicSize};
    // Scan all dimensions of current tensor.
    for (int64_t d = 0; d < rank; d++) {
      // This should be called only once at beginning.
      assert(!ptrBuffer[t][d] && !idxBuffer[t][d] && !highs[t][d]);
      // Handle sparse storage schemes.
      if (isCompressedDLT(dimTypes[t][d])) {
        auto ptrTp =
            MemRefType::get(dynShape, getPointerOverheadType(builder, enc));
        auto indTp =
            MemRefType::get(dynShape, getIndexOverheadType(builder, enc));
        auto dim = builder.getIndexAttr(d);
        // Generate sparse primitives to obtains pointer and indices.
        ptrBuffer[t][d] = builder.create<ToPointersOp>(loc, ptrTp, tensor, dim);
        idxBuffer[t][d] = builder.create<ToIndicesOp>(loc, indTp, tensor, dim);
      } else if (isSingletonDLT(dimTypes[t][d])) {
        // Singleton dimension, fetch indices.
        auto indTp =
            MemRefType::get(dynShape, getIndexOverheadType(builder, enc));
        auto dim = builder.getIndexAttr(d);
        idxBuffer[t][d] = builder.create<ToIndicesOp>(loc, indTp, tensor, dim);
      } else {
        // Dense dimension, nothing to fetch.
        assert(isDenseDLT(dimTypes[t][d]));
      }

      // Find upper bound in current dimension.
      unsigned p = toOrigDim(enc, d);
      Value up = mlir::linalg::createOrFoldDimOp(builder, loc, tensor, p);
      highs[t][d] = up;
    }

    // Perform the required bufferization. Dense inputs materialize
    // from the input tensors. Sparse inputs use sparse primitives to obtain the
    // values.
    // Delegates extra output initialization to clients.
    bool isOutput = isOutputTensor(t);
    Type elementType = rtp.getElementType();
    if (!enc) {
      // Non-annotated dense tensors.
      auto denseTp = MemRefType::get(shape, elementType);
      Value denseVal =
          builder.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
      // Dense outputs need special handling.
      if (isOutput && updater)
        denseVal = updater(builder, loc, denseVal, tensor);

      valBuffer[t] = denseVal;
    } else {
      // Annotated sparse tensors.
      // We also need the value buffer for annotated all dense `sparse` tensor.
      auto dynShape = {ShapedType::kDynamicSize};
      auto sparseTp = MemRefType::get(dynShape, elementType);
      valBuffer[t] = builder.create<ToValuesOp>(loc, sparseTp, tensor);
    }
    // NOTE: we can also prepares for 0 dim here in advance, this will hosit
    // some loop preparation from tensor iteration, but will also (undesirably)
    // hosit the code ouside if conditions.
  }
}

void SparseTensorLoopEmitter::enterNewLoopSeq(OpBuilder &builder, Location loc,
                                              ArrayRef<size_t> tids,
                                              ArrayRef<size_t> dims) {
  // Universal Index start from 0
  assert(loopSeqStack.size() == loopStack.size());
  // Universal index starts from 0
  loopSeqStack.emplace_back(constantIndex(builder, loc, 0));
  // Prepares for all the tensors used in the current loop sequence.
  for (auto [tid, dim] : llvm::zip(tids, dims))
    prepareLoopOverTensorAtDim(builder, loc, tid, dim);
}

Operation *SparseTensorLoopEmitter::enterLoopOverTensorAtDim(
    OpBuilder &builder, Location loc, size_t tid, size_t dim,
    MutableArrayRef<Value> reduc, bool isParallel, ArrayRef<size_t> extraTids,
    ArrayRef<size_t> extraDims) {

  assert(dimTypes[tid].size() > dim);
  // We can not re-enter the same level.
  assert(!coord[tid][dim]);
  // TODO: support multiple return on parallel for?
  assert(!isParallel || reduc.size() <= 1);

  Value step = constantIndex(builder, loc, 1);
  auto dimType = dimTypes[tid][dim];
  bool isSparseInput = isCompressedDLT(dimType) || isSingletonDLT(dimType);
  assert(isDenseDLT(dimType) || isCompressedDLT(dimType) ||
         isSingletonDLT(dimType));

  Value lo = isSparseInput ? pidxs[tid][dim]      // current offset
                           : loopSeqStack.back(); // univeral tid
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

  if (isSparseInput) {
    pidxs[tid][dim] = iv;
    // Generating a load on the indices array yields the coordinate.
    Value ptr = idxBuffer[tid][dim];
    coord[tid][dim] = genIndexLoad(builder, loc, ptr, iv);
  } else {
    // Dense tensor, the coordinates is the inducation variable.
    coord[tid][dim] = iv;
    // generate pidx for dense dim (pidx = i * sz + j)
    auto enc = getSparseTensorEncoding(tensors[tid].getType());
    if (enc && !isSparseOutput(tid))
      pidxs[tid][dim] = genAddress(builder, loc, tid, dim, iv);
  }

  // NOTE: we can also prepares for next dim here in advance
  // Push the loop into stack
  loopStack.emplace_back(ArrayRef<size_t>(tid), ArrayRef<size_t>(dim), loop,
                         coord[tid][dim]);
  // Emit extra locals.
  emitExtraLocalsForTensorsAtDenseDims(builder, loc, extraTids, extraDims);

  return loop;
}

Operation *SparseTensorLoopEmitter::enterCoIterationOverTensorsAtDims(
    OpBuilder &builder, Location loc, ArrayRef<size_t> tids,
    ArrayRef<size_t> dims, bool needsUniv, MutableArrayRef<Value> reduc,
    ArrayRef<size_t> extraTids, ArrayRef<size_t> extraDims) {
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

  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    // All dense dim (as well as sparse output tensor) shared the same pidx in
    // the while loop.
    if (isDenseDLT(dimTypes[tid][dim])) {
      pidxs[tid][dim] = min;
      // generate pidx for dense dim (pidx = i * sz + j)
      auto enc = getSparseTensorEncoding(tensors[tid].getType());
      if (enc && !isSparseOutput(tid))
        pidxs[tid][dim] = genAddress(builder, loc, tid, dim, min);
    }
    // NOTE: we can also prepares for next dim here in advance
  }
  // Sets up the loop stack.
  loopStack.emplace_back(tids, dims, whileOp, min);
  assert(loopStack.size() == loopSeqStack.size());

  // Emits extra locals
  emitExtraLocalsForTensorsAtDenseDims(builder, loc, extraTids, extraDims);

  // Updates reduction variables
  assert(after->getNumArguments() == o + reduc.size() + (needsUniv ? 1 : 0));
  // In-place update on reduction variable.
  for (unsigned i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = after->getArgument(o + i);

  return whileOp;
}

void SparseTensorLoopEmitter::prepareLoopOverTensorAtDim(OpBuilder &builder,
                                                         Location loc,
                                                         size_t tid,
                                                         size_t dim) {
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

void SparseTensorLoopEmitter::emitExtraLocalsForTensorsAtDenseDims(
    OpBuilder &builder, Location loc, ArrayRef<size_t> tids,
    ArrayRef<size_t> dims) {
  // Initialize dense positions. Note that we generate dense indices of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  for (auto [tid, dim] : llvm::zip(tids, dims)) {
    assert(isDenseDLT(dimTypes[tid][dim]));
    auto enc = getSparseTensorEncoding(tensors[tid].getType());
    if (enc && !isSparseOutput(tid)) {
      bool validPidx = dim == 0 || pidxs[tid][dim - 1];
      if (!validPidx) {
        // We might not find the pidx for the sparse output tensor as it is
        // unconditionally required by the sparsification.
        assert(isOutputTensor(tid));
        continue;
      }
      pidxs[tid][dim] = genAddress(builder, loc, tid, dim, loopStack.back().iv);
      // NOTE: we can also prepares for next dim here in advance
    }
  }
}

void SparseTensorLoopEmitter::exitForLoop(RewriterBase &rewriter, Location loc,
                                          MutableArrayRef<Value> reduc) {
  LoopLevelInfo &loopInfo = loopStack.back();
  auto &dims = loopStack.back().dims;
  auto &tids = loopStack.back().tids;
  auto forOp = llvm::dyn_cast<scf::ForOp>(loopInfo.loop);
  if (forOp) {
    if (!reduc.empty()) {
      assert(reduc.size() == forOp.getNumResults());
      rewriter.setInsertionPointToEnd(forOp.getBody());
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

void SparseTensorLoopEmitter::exitCoIterationLoop(
    OpBuilder &builder, Location loc, MutableArrayRef<Value> reduc) {
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

void SparseTensorLoopEmitter::exitCurrentLoop(RewriterBase &rewriter,
                                              Location loc,
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

//===----------------------------------------------------------------------===//
// ExecutionEngine/SparseTensorUtils helper functions.
//===----------------------------------------------------------------------===//

OverheadType mlir::sparse_tensor::overheadTypeEncoding(unsigned width) {
  switch (width) {
  case 64:
    return OverheadType::kU64;
  case 32:
    return OverheadType::kU32;
  case 16:
    return OverheadType::kU16;
  case 8:
    return OverheadType::kU8;
  case 0:
    return OverheadType::kIndex;
  }
  llvm_unreachable("Unsupported overhead bitwidth");
}

OverheadType mlir::sparse_tensor::overheadTypeEncoding(Type tp) {
  if (tp.isIndex())
    return OverheadType::kIndex;
  if (auto intTp = tp.dyn_cast<IntegerType>())
    return overheadTypeEncoding(intTp.getWidth());
  llvm_unreachable("Unknown overhead type");
}

Type mlir::sparse_tensor::getOverheadType(Builder &builder, OverheadType ot) {
  switch (ot) {
  case OverheadType::kIndex:
    return builder.getIndexType();
  case OverheadType::kU64:
    return builder.getIntegerType(64);
  case OverheadType::kU32:
    return builder.getIntegerType(32);
  case OverheadType::kU16:
    return builder.getIntegerType(16);
  case OverheadType::kU8:
    return builder.getIntegerType(8);
  }
  llvm_unreachable("Unknown OverheadType");
}

OverheadType mlir::sparse_tensor::pointerOverheadTypeEncoding(
    const SparseTensorEncodingAttr &enc) {
  return overheadTypeEncoding(enc.getPointerBitWidth());
}

OverheadType mlir::sparse_tensor::indexOverheadTypeEncoding(
    const SparseTensorEncodingAttr &enc) {
  return overheadTypeEncoding(enc.getIndexBitWidth());
}

Type mlir::sparse_tensor::getPointerOverheadType(
    Builder &builder, const SparseTensorEncodingAttr &enc) {
  return getOverheadType(builder, pointerOverheadTypeEncoding(enc));
}

Type mlir::sparse_tensor::getIndexOverheadType(
    Builder &builder, const SparseTensorEncodingAttr &enc) {
  return getOverheadType(builder, indexOverheadTypeEncoding(enc));
}

// TODO: Adjust the naming convention for the constructors of
// `OverheadType` so we can use the `MLIR_SPARSETENSOR_FOREVERY_O` x-macro
// here instead of `MLIR_SPARSETENSOR_FOREVERY_FIXED_O`; to further reduce
// the possibility of typo bugs or things getting out of sync.
StringRef mlir::sparse_tensor::overheadTypeFunctionSuffix(OverheadType ot) {
  switch (ot) {
  case OverheadType::kIndex:
    return "0";
#define CASE(ONAME, O)                                                         \
  case OverheadType::kU##ONAME:                                                \
    return #ONAME;
    MLIR_SPARSETENSOR_FOREVERY_FIXED_O(CASE)
#undef CASE
  }
  llvm_unreachable("Unknown OverheadType");
}

StringRef mlir::sparse_tensor::overheadTypeFunctionSuffix(Type tp) {
  return overheadTypeFunctionSuffix(overheadTypeEncoding(tp));
}

PrimaryType mlir::sparse_tensor::primaryTypeEncoding(Type elemTp) {
  if (elemTp.isF64())
    return PrimaryType::kF64;
  if (elemTp.isF32())
    return PrimaryType::kF32;
  if (elemTp.isF16())
    return PrimaryType::kF16;
  if (elemTp.isBF16())
    return PrimaryType::kBF16;
  if (elemTp.isInteger(64))
    return PrimaryType::kI64;
  if (elemTp.isInteger(32))
    return PrimaryType::kI32;
  if (elemTp.isInteger(16))
    return PrimaryType::kI16;
  if (elemTp.isInteger(8))
    return PrimaryType::kI8;
  if (auto complexTp = elemTp.dyn_cast<ComplexType>()) {
    auto complexEltTp = complexTp.getElementType();
    if (complexEltTp.isF64())
      return PrimaryType::kC64;
    if (complexEltTp.isF32())
      return PrimaryType::kC32;
  }
  llvm_unreachable("Unknown primary type");
}

StringRef mlir::sparse_tensor::primaryTypeFunctionSuffix(PrimaryType pt) {
  switch (pt) {
#define CASE(VNAME, V)                                                         \
  case PrimaryType::k##VNAME:                                                  \
    return #VNAME;
    MLIR_SPARSETENSOR_FOREVERY_V(CASE)
#undef CASE
  }
  llvm_unreachable("Unknown PrimaryType");
}

StringRef mlir::sparse_tensor::primaryTypeFunctionSuffix(Type elemTp) {
  return primaryTypeFunctionSuffix(primaryTypeEncoding(elemTp));
}

//===----------------------------------------------------------------------===//
// Misc code generators.
//===----------------------------------------------------------------------===//

mlir::Attribute mlir::sparse_tensor::getOneAttr(Builder &builder, Type tp) {
  if (tp.isa<FloatType>())
    return builder.getFloatAttr(tp, 1.0);
  if (tp.isa<IndexType>())
    return builder.getIndexAttr(1);
  if (auto intTp = tp.dyn_cast<IntegerType>())
    return builder.getIntegerAttr(tp, APInt(intTp.getWidth(), 1));
  if (tp.isa<RankedTensorType, VectorType>()) {
    auto shapedTp = tp.cast<ShapedType>();
    if (auto one = getOneAttr(builder, shapedTp.getElementType()))
      return DenseElementsAttr::get(shapedTp, one);
  }
  llvm_unreachable("Unsupported attribute type");
}

Value mlir::sparse_tensor::genIsNonzero(OpBuilder &builder, mlir::Location loc,
                                        Value v) {
  Type tp = v.getType();
  Value zero = constantZero(builder, loc, tp);
  if (tp.isa<FloatType>())
    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v,
                                         zero);
  if (tp.isIntOrIndex())
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                         zero);
  if (tp.dyn_cast<ComplexType>())
    return builder.create<complex::NotEqualOp>(loc, v, zero);
  llvm_unreachable("Non-numeric type");
}

void mlir::sparse_tensor::genReshapeDstShape(
    Location loc, PatternRewriter &rewriter, SmallVectorImpl<Value> &dstShape,
    ArrayRef<Value> srcShape, ArrayRef<int64_t> staticDstShape,
    ArrayRef<ReassociationIndices> reassociation) {
  // Collapse shape.
  if (reassociation.size() < srcShape.size()) {
    unsigned start = 0;
    for (const auto &map : llvm::enumerate(reassociation)) {
      auto dstDim = constantIndex(rewriter, loc, 1);
      for (unsigned i = start; i < start + map.value().size(); i++) {
        dstDim = rewriter.create<arith::MulIOp>(loc, dstDim, srcShape[i]);
      }
      dstShape.push_back(dstDim);
      start = start + map.value().size();
    }
    assert(start == srcShape.size());
    return;
  }

  // Expand shape.
  assert(reassociation.size() == srcShape.size());
  unsigned start = 0;
  // Expand the i-th dimension in srcShape.
  for (unsigned i = 0, size = srcShape.size(); i < size; i++) {
    const auto &map = reassociation[i];
    auto srcDim = srcShape[i];
    // Iterate through dimensions expanded from the i-th dimension.
    for (unsigned j = start; j < start + map.size(); j++) {
      // There can be only one dynamic sized dimension among dimensions
      // expanded from the i-th dimension in srcShape.
      // For example, if srcDim = 8, then the expanded shape could be <2x?x2>,
      // but not <2x?x?>.
      if (staticDstShape[j] == ShapedType::kDynamicSize) {
        // The expanded dimension has dynamic size. We compute the dimension
        // by dividing srcDim by the product of the static dimensions.
        int64_t product = 1;
        for (unsigned k = start; k < start + map.size(); k++) {
          if (staticDstShape[k] != ShapedType::kDynamicSize) {
            product *= staticDstShape[k];
          }
        }
        // Compute the dynamic dimension size.
        Value productVal = constantIndex(rewriter, loc, product);
        Value dynamicSize =
            rewriter.create<arith::DivUIOp>(loc, srcDim, productVal);
        dstShape.push_back(dynamicSize);
      } else {
        // The expanded dimension is statically known.
        dstShape.push_back(constantIndex(rewriter, loc, staticDstShape[j]));
      }
    }
    start = start + map.size();
  }
  assert(start == staticDstShape.size());
}

void mlir::sparse_tensor::translateIndicesArray(
    OpBuilder &builder, Location loc,
    ArrayRef<ReassociationIndices> reassociation, ValueRange srcIndices,
    ArrayRef<Value> srcShape, ArrayRef<Value> dstShape,
    SmallVectorImpl<Value> &dstIndices) {
  unsigned i = 0;
  unsigned start = 0;
  unsigned dstRank = dstShape.size();
  unsigned srcRank = srcShape.size();
  assert(srcRank == srcIndices.size());
  bool isCollapse = srcRank > dstRank;
  ArrayRef<Value> shape = isCollapse ? srcShape : dstShape;
  // Iterate over reassociation map.
  for (const auto &map : llvm::enumerate(reassociation)) {
    // Prepare strides information in dimension slice.
    Value linear = constantIndex(builder, loc, 1);
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::MulIOp>(loc, linear, shape[j]);
    }
    // Start expansion.
    Value val;
    if (!isCollapse)
      val = srcIndices[i];
    // Iterate over dimension slice.
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::DivUIOp>(loc, linear, shape[j]);
      if (isCollapse) {
        Value old = srcIndices[j];
        Value mul = builder.create<arith::MulIOp>(loc, old, linear);
        val = val ? builder.create<arith::AddIOp>(loc, val, mul) : mul;
      } else {
        Value old = val;
        val = builder.create<arith::DivUIOp>(loc, val, linear);
        assert(dstIndices.size() == j);
        dstIndices.push_back(val);
        val = builder.create<arith::RemUIOp>(loc, old, linear);
      }
    }
    // Finalize collapse.
    if (isCollapse) {
      assert(dstIndices.size() == i);
      dstIndices.push_back(val);
    }
    start += map.value().size();
    i++;
  }
  assert(dstIndices.size() == dstRank);
}

FlatSymbolRefAttr mlir::sparse_tensor::getFunc(ModuleOp module, StringRef name,
                                               TypeRange resultType,
                                               ValueRange operands,
                                               EmitCInterface emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (static_cast<bool>(emitCInterface))
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

func::CallOp mlir::sparse_tensor::createFuncCall(
    OpBuilder &builder, Location loc, StringRef name, TypeRange resultType,
    ValueRange operands, EmitCInterface emitCInterface) {
  auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  FlatSymbolRefAttr fn =
      getFunc(module, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(loc, resultType, fn, operands);
}

Type mlir::sparse_tensor::getOpaquePointerType(OpBuilder &builder) {
  return LLVM::LLVMPointerType::get(builder.getI8Type());
}

Value mlir::sparse_tensor::genAlloca(OpBuilder &builder, Location loc,
                                     unsigned sz, Type tp) {
  return genAlloca(builder, loc, constantIndex(builder, loc, sz), tp);
}

Value mlir::sparse_tensor::genAlloca(OpBuilder &builder, Location loc, Value sz,
                                     Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  return builder.create<memref::AllocaOp>(loc, memTp, ValueRange{sz});
}

Value mlir::sparse_tensor::genAllocaScalar(OpBuilder &builder, Location loc,
                                           Type tp) {
  return builder.create<memref::AllocaOp>(loc, MemRefType::get({}, tp));
}

Value mlir::sparse_tensor::allocDenseTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType tensorTp,
                                            ValueRange sizes) {
  Type elemTp = tensorTp.getElementType();
  auto shape = tensorTp.getShape();
  auto memTp = MemRefType::get(shape, elemTp);
  SmallVector<Value> dynamicSizes;
  for (unsigned i = 0, rank = tensorTp.getRank(); i < rank; i++) {
    if (shape[i] == ShapedType::kDynamicSize)
      dynamicSizes.push_back(sizes[i]);
  }
  Value mem = builder.create<memref::AllocOp>(loc, memTp, dynamicSizes);
  Value zero = constantZero(builder, loc, elemTp);
  builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{mem});
  return mem;
}

void mlir::sparse_tensor::deallocDenseTensor(OpBuilder &builder, Location loc,
                                             Value buffer) {
  builder.create<memref::DeallocOp>(loc, buffer);
}

Value mlir::sparse_tensor::genValueForDense(OpBuilder &builder, Location loc,
                                            Value tensor, ValueRange ivs) {
  Value val = builder.create<tensor::ExtractOp>(loc, tensor, ivs);
  Value cond = genIsNonzero(builder, loc, val);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, cond, /*else*/ false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return val;
}

// FIXME:
// 1. Dense tensors loop should be generated by loop emitter.
// 2. Support reduction variables to propagate SSA chains properly.
void mlir::sparse_tensor::genDenseTensorOrSparseConstantIterLoop(
    OpBuilder &builder, Location loc, Value src, unsigned rank,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder) {
  SmallVector<Value> indicesArray;
  SmallVector<Value> lo;
  SmallVector<Value> hi;
  SmallVector<Value> st;
  Value zero = constantIndex(builder, loc, 0);
  Value one = constantIndex(builder, loc, 1);
  auto indicesValues = genSplitSparseConstant(builder, loc, src);
  bool isCOOConstant = indicesValues.has_value();
  Value indices;
  Value values;
  if (isCOOConstant) {
    indices = indicesValues->first;
    values = indicesValues->second;
    lo.push_back(zero);
    hi.push_back(linalg::createOrFoldDimOp(builder, loc, values, 0));
    st.push_back(one);
  } else {
    for (unsigned i = 0; i < rank; i++) {
      lo.push_back(zero);
      hi.push_back(linalg::createOrFoldDimOp(builder, loc, src, i));
      st.push_back(one);
    }
  }

  scf::buildLoopNest(
      builder, loc, lo, hi, st, {},
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange args) -> scf::ValueVector {
        Value val;
        if (isCOOConstant)
          val = genIndexAndValueForSparse(builder, loc, indices, values,
                                          indicesArray, ivs, rank);
        else
          val = genIndexAndValueForDense(builder, loc, src, indicesArray, ivs);
        bodyBuilder(builder, loc, val, indicesArray);
        return {};
      });
}

void mlir::sparse_tensor::sizesFromSrc(OpBuilder &builder,
                                       SmallVectorImpl<Value> &sizes,
                                       Location loc, Value src) {
  unsigned rank = src.getType().cast<ShapedType>().getRank();
  for (unsigned i = 0; i < rank; i++)
    sizes.push_back(linalg::createOrFoldDimOp(builder, loc, src, i));
}

Operation *mlir::sparse_tensor::getTop(Operation *op) {
  for (; isa<scf::ForOp>(op->getParentOp()) ||
         isa<scf::WhileOp>(op->getParentOp()) ||
         isa<scf::ParallelOp>(op->getParentOp()) ||
         isa<scf::IfOp>(op->getParentOp());
       op = op->getParentOp())
    ;
  return op;
}
