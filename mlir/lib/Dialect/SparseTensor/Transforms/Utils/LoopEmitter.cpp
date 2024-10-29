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
static Value genSliceOffset(OpBuilder &builder, Location loc, Value tensor,
                            Level lvl) {
  auto enc = getSparseTensorEncoding(tensor.getType());
  return createOrFoldSliceOffsetOp(builder, loc, tensor, toDim(enc, lvl));
}

static Value genSliceStride(OpBuilder &builder, Location loc, Value tensor,
                            Level lvl) {
  auto enc = getSparseTensorEncoding(tensor.getType());
  return createOrFoldSliceStrideOp(builder, loc, tensor, toDim(enc, lvl));
}

static bool isIntOrFPZero(Attribute attr) {
  if (auto f = llvm::dyn_cast<FloatAttr>(attr); f && f.getValue().isZero())
    return true;
  if (auto i = llvm::dyn_cast<IntegerAttr>(attr); i && i.getValue().isZero())
    return true;
  return false;
}

static Value unFoldOpIntResult(OpBuilder &builder, Location loc,
                               OpFoldResult ofr) {
  if (std::optional<int64_t> i = getConstantIntValue(ofr); i.has_value())
    return constantIndex(builder, loc, *i);
  return ofr.get<Value>();
}

static Value tryFoldTensors(Value t) {
  // TODO: this should be done through a folding pass after switching to
  // `sparse_tensor.iterate`-based sparsification.
  auto stt = tryGetSparseTensorType(t);
  auto padOp = t.getDefiningOp<tensor::PadOp>();
  if (padOp && stt.has_value() && stt->hasEncoding() &&
      padOp.getSourceType().getEncoding() == stt->getEncoding() &&
      stt->getEncoding().isIdentity()) {
    // Try fusing padOp with zeros.
    Attribute padCst;
    if (matchPattern(padOp.getBody()->getTerminator(),
                     m_Op<tensor::YieldOp>(m_Constant(&padCst))) &&
        isIntOrFPZero(padCst)) {
      return padOp.getSource();
    }
  }
  return t;
}

//===----------------------------------------------------------------------===//
// Sparse tensor loop emitter class implementations
//===----------------------------------------------------------------------===//

LoopEmitter::LoopEmitter(ValueRange tensors, StringAttr loopTag, bool hasOutput,
                         bool isSparseOut, unsigned numLoops,
                         DependentLvlGetter dimGetter,
                         SparseEmitStrategy emitStrategy) {
  initialize(tensors, loopTag, hasOutput, isSparseOut, numLoops, dimGetter);
}

void LoopEmitter::initialize(ValueRange ts, StringAttr loopTag, bool hasOutput,
                             bool isSparseOut, unsigned numLoops,
                             DependentLvlGetter dimGetter,
                             SparseEmitStrategy emitStrategy) {
  // First initialize the top-level type of the fields.
  this->loopTag = loopTag;
  this->hasOutput = hasOutput;
  this->isSparseOut = isSparseOut;
  this->emitStrategy = emitStrategy;

  const unsigned numManifestTensors = ts.size();
  const unsigned synTensorId = numManifestTensors;
  const unsigned numTensors = numManifestTensors + 1;
  // tensors array (len == numManifestTensor).
  this->tensors.assign(ts.begin(), ts.end());
  // Arrays with len == numTensor.
  this->valBuffer.assign(numTensors, nullptr);
  this->lvls.resize(numTensors);
  this->iters.resize(numTensors);
  this->spIterVals.resize(numTensors);

  // These zeros will be overwritten below, but we need to initialize
  // them to something since we'll need random-access assignment.
  this->loopStack.reserve(numLoops);
  this->loopSeqStack.reserve(numLoops);

  // Index-reduction related fields.
  this->dependentLvlMap.assign(
      numTensors, std::vector<std::vector<std::pair<TensorLevel, unsigned>>>());
  this->sliceMeta.assign(
      numTensors, std::vector<std::vector<std::pair<Value, unsigned>>>());
  this->levelReducedDep.assign(numTensors, std::vector<unsigned>());

  // Initialize nested types of `TensorId`-indexed fields.
  for (TensorId tid = 0; tid < numTensors; tid++) {
    Level lvlRank;
    if (tid == synTensorId) {
      // Synthetic tensor (conceptually) is an all-dense tensor with rank equal
      // to the total number of loops (each level can potentially be mapped to
      // one of the loop being generated).
      lvlRank = numLoops;
    } else {
      const Value t = tensors[tid];
      // a scalar or 0-dimension tensors
      if (isZeroRankedTensorOrScalar(t.getType()))
        continue;

      auto rtp = getRankedTensorType(t);
      const SparseTensorType stt(rtp);
      lvlRank = stt.getLvlRank();
    }

    lvls[tid].resize(lvlRank);
    iters[tid].resize(lvlRank);
    spIterVals[tid].resize(lvlRank);
    loopHighs.assign(numLoops, nullptr);

    // Slice-driven loops related initialization.
    levelReducedDep[tid].assign(lvlRank, 0);
    dependentLvlMap[tid].assign(
        lvlRank, std::vector<std::pair<TensorLevel, unsigned>>());
    sliceMeta[tid].assign(lvlRank, std::vector<std::pair<Value, unsigned>>());
    if (dimGetter && !isSynTensor(tid)) {
      for (Level l = 0; l < lvlRank; l++) {
        std::vector<std::pair<LoopId, unsigned>> deps = dimGetter(tid, l);
        // Sort the loop by order.
        llvm::sort(deps, llvm::less_first());

        dependentLvlMap[tid][l] = std::move(deps);
        unsigned depends = dependentLvlMap[tid][l].size();
        if (depends == 0)
          continue;
        sliceMeta[tid][l].reserve(depends);
      }
    }
  }
}

std::unique_ptr<SparseIterator>
LoopEmitter::makeLevelIterator(OpBuilder &builder, Location loc, TensorId t,
                               Level l) {
  Value tensor = tensors[t];
  auto stt = getSparseTensorType(tensor);
  auto it = makeSimpleIterator(*lvls[t][l], emitStrategy);

  Value folded = tryFoldTensors(tensor);
  if (folded != tensor) {
    auto padOp = tensor.getDefiningOp<tensor::PadOp>();
    assert(padOp);
    if (padOp.getPaddedDims().test(l)) {
      Value low = unFoldOpIntResult(builder, loc, padOp.getMixedLowPad()[l]);
      Value high = unFoldOpIntResult(builder, loc, padOp.getMixedHighPad()[l]);
      auto padIt = makePaddedIterator(std::move(it), low, high, emitStrategy);
      return padIt;
    }
  }

  if (stt.hasEncoding() && stt.getEncoding().isSlice()) {
    Value offset = genSliceOffset(builder, loc, tensor, l);
    Value stride = genSliceStride(builder, loc, tensor, l);
    auto slicedIt = makeSlicedLevelIterator(
        std::move(it), offset, stride, lvls[t][l]->getSize(), emitStrategy);
    return slicedIt;
  }

  return it;
}

void LoopEmitter::initializeLoopEmit(
    OpBuilder &builder, Location loc, LoopEmitter::OutputUpdater updater,
    LoopEmitter::SynTensorBoundSetter synSetter) {

  // For every manifest tensor, set up the values buffer.
  for (TensorId t = 0, numTensors = getNumManifestTensors(); t < numTensors;
       t++) {
    // TODO: this should be done through a folding pass after switching to
    // `sparse_tensor.iterate`-based sparsification.
    const Value tensor = tryFoldTensors(tensors[t]);
    const auto rtp = dyn_cast<RankedTensorType>(tensor.getType());
    // Skips only scalar, zero ranked tensor still need to be bufferized and
    // (probably) filled with zeros by users.
    if (!rtp)
      continue;

    auto stt = getSparseTensorType(tensor);
    const auto shape = rtp.getShape();

    // Perform the required bufferization. Dense inputs materialize from the
    // input tensors. Sparse inputs use sparse primitives to obtain the values.
    // Delegates extra output initialization to clients.
    bool isOutput = isOutputTensor(t);
    Type elementType = stt.getElementType();
    if (!stt.hasEncoding()) {
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
      // We also need the value buffer for all-dense annotated "sparse"
      // tensors.
      valBuffer[t] = builder.create<ToValuesOp>(loc, tensor);
    }
  }

  // The sparse iterator values will only be available after the loop is
  // constructed.
  if (emitStrategy == SparseEmitStrategy::kSparseIterator)
    return;

  // For every synthetic tensor, set the high bound by calling the callback.
  if (synSetter) {
    TensorId synId = getSynTensorId();
    for (unsigned i = 0, e = loopHighs.size(); i < e; i++) {
      Value sz = loopHighs[i] = synSetter(builder, loc, i);
      auto [stl, it] = makeSynLevelAndIterator(sz, synId, i, emitStrategy);
      lvls[synId][i] = std::move(stl);
      iters[synId][i].emplace_back(std::move(it));
    }
  }

  // For every manifest tensor:
  // * For every level:
  //   * get the positions and coordinates buffers
  //   * get/compute the level-size, which is also used as the upper-bound
  //     on positions.
  for (TensorId t = 0, numTensors = getNumManifestTensors(); t < numTensors;
       t++) {
    // TODO: this should be done through a folding pass after switching to
    // `sparse_tensor.iterate`-based sparsification.
    const Value tensor = tryFoldTensors(tensors[t]);
    const auto rtp = dyn_cast<RankedTensorType>(tensor.getType());
    if (!rtp)
      // Skips only scalar, zero ranked tensor still need to be bufferized and
      // (probably) filled with zeros by users.
      continue;

    auto stt = getSparseTensorType(tensor);
    const Level lvlRank = stt.getLvlRank();

    // Scan all levels of current tensor.
    for (Level l = 0; l < lvlRank; l++) {
      // Find upper bound in current dimension.
      lvls[t][l] = makeSparseTensorLevel(builder, loc, tensor, t, l);
      if (!dependentLvlMap[t][l].empty())
        continue;

      auto it = makeLevelIterator(builder, loc, t, l);
      iters[t][l].emplace_back(std::move(it));
    }
    // NOTE: we can also prepare for 0 lvl here in advance, this will hoist
    // some loop preparation from tensor iteration, but will also (undesirably)
    // hoist the code ouside if-conditions.
  }
  // TODO: avoid treating subsection iterator as a special case.
  initSubSectIterator(builder, loc);
}

void LoopEmitter::initSubSectIterator(OpBuilder &builder, Location loc) {
  Value c0 = C_IDX(0);
  for (TensorId t = 0, e = tensors.size(); t < e; t++) {
    auto rtp = dyn_cast<RankedTensorType>(tensors[t].getType());
    if (!rtp)
      continue;

    Level lvlRank = SparseTensorType(rtp).getLvlRank();

    // Compute the dependency reduction order.
    auto remDepStack = dependentLvlMap;
    std::vector<std::tuple<LoopId, TensorId, Level>> depRedOrder;
    for (Level lvl = 0; lvl < lvlRank; lvl++) {
      // Reverse queue into a stack.
      std::reverse(remDepStack[t][lvl].begin(), remDepStack[t][lvl].end());
      for (auto [loop, coeff] : dependentLvlMap[t][lvl])
        depRedOrder.emplace_back(std::make_tuple(loop, t, lvl));
    }

    if (depRedOrder.empty())
      continue;

    std::sort(depRedOrder.begin(), depRedOrder.end(),
              [](auto &l, auto &r) { return std::get<0>(l) < std::get<0>(r); });

    SmallVector<SparseIterator *> lastIter(tensors.size(), nullptr);
    for (auto [loop, t, lvl] : depRedOrder) {
      std::pair<LoopId, unsigned> curDep = remDepStack[t][lvl].back();
      assert(curDep.first == loop);
      remDepStack[t][lvl].pop_back();

      auto lvlIt = makeLevelIterator(builder, loc, t, lvl);
      const SparseIterator *parent = lastIter[t];
      if (!parent && lvl > 0) {
        if (dependentLvlMap[t][lvl - 1].empty()) {
          parent = iters[t][lvl - 1].back().get();
        }
      }

      std::unique_ptr<SparseIterator> it;
      if (!remDepStack[t][lvl].empty()) {
        // Compute the subsection size.
        Value size = c0;
        for (auto [loop, stride] : remDepStack[t][lvl]) {
          Value idxMax = SUBI(loopHighs[loop], C_IDX(1));
          size = ADDI(size, ADDI(MULI(idxMax, C_IDX(stride)), C_IDX(1)));
        }
        it = makeNonEmptySubSectIterator(builder, loc, parent, loopHighs[loop],
                                         std::move(lvlIt), size, curDep.second,
                                         emitStrategy);
      } else {
        const SparseIterator &subSectIter = *iters[t][lvl].back();
        it = makeTraverseSubSectIterator(builder, loc, subSectIter, *parent,
                                         std::move(lvlIt), loopHighs[loop],
                                         curDep.second, emitStrategy);
      }
      lastIter[t] = it.get();
      iters[t][lvl].emplace_back(std::move(it));
    }
  }
}

void LoopEmitter::categorizeIterators(
    ArrayRef<TensorLevel> tidLvls, SmallVectorImpl<SparseIterator *> &raIters,
    SmallVectorImpl<SparseIterator *> &spIters) {
  // Finds out the tensor level that we should use to generate loops. Amongs all
  // the tensor levels, there is at most one sparse tensor level.
  for (auto [t, l] : unpackTensorLevelRange(tidLvls)) {
    SparseIterator *it = &getCurIterator(t, l);
    if (it->randomAccessible())
      raIters.push_back(it);
    else
      spIters.push_back(it);
  }

  std::stable_sort(spIters.begin(), spIters.end(), [](auto lhs, auto rhs) {
    // AffineUnRed > Affine > Slice > Trivial
    return static_cast<uint8_t>(lhs->kind) > static_cast<uint8_t>(rhs->kind);
  });
}

void LoopEmitter::enterNewLoopSeq(OpBuilder &builder, Location loc,
                                  ArrayRef<TensorLevel> tidLvls) {
  // TODO: sort
  assert(loopSeqStack.size() == loopStack.size());

  if (emitStrategy != SparseEmitStrategy::kSparseIterator) {
    // Prepares for all the tensors used in the current loop sequence.
    for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
      levelReducedDep[tid][lvl]++;
      prepareLoopOverTensorAtLvl(builder, loc, tid, lvl);
    }
  }

  // Universal Index starts from 0.
  loopSeqStack.emplace_back(C_IDX(0), tidLvls.vec());
}

void LoopEmitter::exitCurrentLoopSeq(OpBuilder &builder, Location loc) {
  assert(loopSeqStack.size() == loopStack.size() + 1);

  // Depending on whether the slice is resolved or not at current loop sequence,
  // end them in different ways.
  for (auto [tid, lvl] : unpackTensorLevelRange(loopSeqStack.back().second))
    levelReducedDep[tid][lvl]--;

  loopSeqStack.pop_back();
}

Value LoopEmitter::genAffine(OpBuilder &builder, Location loc, AffineExpr a) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    // FIXME: since the one callsite in Sparsification passes in a
    // level-expression, the `getPosition` must in fact be a `Dimension`.
    // However, elsewhere we have been lead to expect that `loopIdToOrd`
    // should be indexed by `LoopId`...
    const auto loopId = cast<AffineDimExpr>(a).getPosition();
    return loopStack[loopId].iv;
  }
  case AffineExprKind::Add: {
    auto binOp = cast<AffineBinaryOpExpr>(a);
    return ADDI(genAffine(builder, loc, binOp.getLHS()),
                genAffine(builder, loc, binOp.getRHS()));
  }
  case AffineExprKind::Mul: {
    auto binOp = cast<AffineBinaryOpExpr>(a);
    return MULI(genAffine(builder, loc, binOp.getLHS()),
                genAffine(builder, loc, binOp.getRHS()));
  }
  case AffineExprKind::Constant: {
    int64_t c = cast<AffineConstantExpr>(a).getValue();
    return C_IDX(c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

std::pair<Operation *, Value> LoopEmitter::emitForLoopOverTensorAtLvl(
    OpBuilder &builder, Location loc, SparseIterator &iter,
    MutableArrayRef<Value> reduc, bool isParallel) {

  // TODO: support dynamic slices.
  // Uses the first dimension here to build the loop bound (which is also the
  // biggest range).

  Value step = C_IDX(1);
  auto [lo, hi] = iter.genForCond(builder, loc);
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
    // reduction variable before users fill parallel loop body.
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

  Value crd = iv;
  if (!iter.randomAccessible()) {
    iter.linkNewScope(iv);
    crd = iter.deref(builder, loc);
  } else {
    iter.locate(builder, loc, iv);
  }

  return {loop, crd};
}

std::pair<Operation *, Value> LoopEmitter::emitWhileLoopOverTensorsAtLvls(
    OpBuilder &builder, Location loc, ArrayRef<SparseIterator *> spIters,
    MutableArrayRef<Value> reduc, bool needsUniv) {
  return genCoIteration(builder, loc, spIters, reduc,
                        needsUniv ? loopSeqStack.back().first : nullptr);
}

bool LoopEmitter::shouldIteratedByForLoop(ArrayRef<SparseIterator *> spIters) {
  // If we need to co-iterate over two sparse tensors, we need a while loop
  if (spIters.size() > 1)
    return false;

  if (spIters.size() == 1)
    return spIters.front()->iteratableByFor();

  return true;
}

Region *LoopEmitter::enterCurrentCoIterationCase(OpBuilder &builder,
                                                 Location loc,
                                                 I64BitSet caseBit,
                                                 unsigned caseIdx,
                                                 MutableArrayRef<Value> reduc) {
  auto coIterOp = cast<CoIterateOp>(loopStack.back().loop);
  SmallVector<Attribute> cases(coIterOp.getCases().getAsRange<Attribute>());
  cases[caseIdx] = builder.getI64IntegerAttr(caseBit);

  coIterOp.setCasesAttr(builder.getArrayAttr(cases));
  Region &caseRegion = coIterOp.getRegion(caseIdx);
  assert(caseRegion.getBlocks().empty() &&
         "re-initialize the same coiteration case region.");

  // Each block starts with by a list of user-provided iteration arguments.
  TypeRange iterArgsTps = coIterOp.getInitArgs().getTypes();
  // Followed by a list of used coordinates of index type.
  SmallVector<Type> blockArgTps(coIterOp.getCrdUsedLvls().count(),
                                builder.getIndexType());

  blockArgTps.append(iterArgsTps.begin(), iterArgsTps.end());
  // Ends with a set of iterators that defines the actually iteration space.
  for (auto i : caseBit.bits()) {
    blockArgTps.push_back(
        cast<IterSpaceType>(coIterOp.getIterSpaces()[i].getType())
            .getIteratorType());
  }
  SmallVector<Location> locs(blockArgTps.size(), loc);
  caseRegion.emplaceBlock().addArguments(blockArgTps, locs);

  // Entering the new region scope, updating the SSA chain.
  builder.setInsertionPointToStart(&caseRegion.front());
  // Update the coordinates.
  loopStack.back().iv = coIterOp.getCrds(caseIdx).front();
  // Updates loop iteration arguments.
  ValueRange iterArgs = coIterOp.getRegionIterArgs(caseIdx);
  llvm::copy(iterArgs, reduc.begin());
  // Updates sparse iterator values.
  ValueRange iters = coIterOp.getRegionIterators(caseIdx);
  ArrayRef<TensorLevel> tidLvls = loopStack.back().tidLvls;
  for (auto [i, tl] : llvm::enumerate(unpackTensorLevelRange(tidLvls))) {
    if (caseBit[i]) {
      spIterVals[tl.first][tl.second] = iters.front();
      iters = iters.drop_front();
    } else {
      spIterVals[tl.first][tl.second] = nullptr;
    }
  }
  // Must have consumed all iterator SSA values.
  assert(iters.empty());
  return &caseRegion;
}

Operation *LoopEmitter::enterCoIterationOverTensorsAtLvls(
    OpBuilder &builder, Location loc, ArrayRef<TensorLevel> tidLvls,
    unsigned numCases, MutableArrayRef<Value> reduc, bool tryParallel,
    bool needsUniv) {
  // TODO: Argument `numCases` only used when generating iterator-based sparse
  // loops. Simplify the code upon feature complete.
  // TODO: handle coiteration with sparse iterator.
  if (emitStrategy == SparseEmitStrategy::kSparseIterator) {
    if (tidLvls.size() == 1) {
      auto [tid, lvl] = unpackTensorLevel(tidLvls.front());
      Value t = tensors[tid];

      // Extract and iterate over the iteration space.
      ExtractIterSpaceOp extractSpaceOp =
          lvl == 0 ? builder.create<ExtractIterSpaceOp>(loc, t)
                   : builder.create<ExtractIterSpaceOp>(
                         loc, t, spIterVals[tid][lvl - 1], lvl);

      IterateOp iterOp = builder.create<IterateOp>(
          loc, extractSpaceOp.getExtractedSpace(), reduc);
      spIterVals[tid][lvl] = iterOp.getIterator();

      // Update the reduction varaibles.
      llvm::copy(iterOp.getRegionIterArgs(), reduc.begin());
      // Set the insertion point to loop body.
      builder.setInsertionPointToStart(iterOp.getBody());
      loopStack.emplace_back(tidLvls, iterOp, builder.getInsertionBlock(),
                             iterOp.getCrds().front(), loopTag);
      return iterOp;
    }

    // CoIteration Loops.
    SmallVector<Value> spaces;
    for (auto [tid, lvl] : unpackTensorLevelRange(tidLvls)) {
      Value t = tensors[tid];
      ExtractIterSpaceOp extractSpaceOp =
          lvl == 0 ? builder.create<ExtractIterSpaceOp>(loc, t)
                   : builder.create<ExtractIterSpaceOp>(
                         loc, t, spIterVals[tid][lvl - 1], lvl);
      spaces.push_back(extractSpaceOp.getExtractedSpace());
    }
    auto coIterOp = builder.create<CoIterateOp>(loc, spaces, reduc, numCases);
    // The CoIterationOp does not have insertion block nor induction variable.
    // TODO: the `struct LoopInfo` should be simplied after full migration.
    loopStack.emplace_back(tidLvls, coIterOp, /*insertion block*/ nullptr,
                           /*induction variable*/ nullptr, loopTag);
    return coIterOp;
  }

  // TODO: support multiple return on parallel for?
  tryParallel = tryParallel && reduc.size() <= 1;

  SmallVector<SparseIterator *> raIters;
  SmallVector<SparseIterator *> spIters;
  categorizeIterators(tidLvls, raIters, spIters);

  // Only when there is at least one sparse conditions, do we really need the
  // universal index.
  // TODO: Maybe we should instead requires merger to pass in a valid value at
  // the first place instead of adjusting it in LoopEmitter?
  needsUniv = !spIters.empty() && needsUniv;
  // The TensorLevel used for loop conditions.
  // If there is any sparse level, we need to use the sparse condition.
  // If all levels are dense, we can pick arbitrary one (dense slice-driven loop
  // can be generated using a simple ForOp as well).
  Operation *l = nullptr;
  Value iv = nullptr;
  SmallVector<TensorLevel> tls;

  // Generates loops differently depending on whether we need a slice-driven
  // loop or a simple level traversal loop.
  if (shouldIteratedByForLoop(spIters) && !needsUniv) {
    assert(spIters.size() <= 1);
    SparseIterator &it = spIters.empty() ? *raIters.front() : *spIters.front();
    std::tie(l, iv) =
        emitForLoopOverTensorAtLvl(builder, loc, it, reduc, tryParallel);
    tls.push_back(makeTensorLevel(it.tid, it.lvl));
  } else {
    for (auto *it : spIters) {
      tls.push_back(makeTensorLevel(it->tid, it->lvl));
    }

    if (needsUniv)
      for (auto *it : raIters)
        tls.push_back(makeTensorLevel(it->tid, it->lvl));

    std::tie(l, iv) =
        emitWhileLoopOverTensorsAtLvls(builder, loc, spIters, reduc, needsUniv);
  }

  // Enter dense tensor levels.
  for (SparseIterator *it : raIters)
    it->locate(builder, loc, iv);

  // NOTE: we can also prepare for next dim here in advance
  // Pushes the loop into stack.
  loopStack.emplace_back(tls, l, builder.getInsertionBlock(), iv, loopTag);
  return l;
}

void LoopEmitter::locateLvlAtAffineAddress(OpBuilder &builder, Location loc,
                                           TensorLevel tidLvl,
                                           AffineExpr lvlExpr) {
  auto [tid, lvl] = unpackTensorLevel(tidLvl);

  const SparseIterator *parent =
      lvl == 0 ? nullptr : iters[tid][lvl - 1].back().get();
  auto &it = getCurIterator(tid, lvl);
  it.genInit(builder, loc, parent);

  assert(it.kind == IterKind::kTrivial && it.randomAccessible());
  Value lvlCrd = genAffine(builder, loc, lvlExpr);
  it.locate(builder, loc, lvlCrd);
}

void LoopEmitter::prepareLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                             TensorId tid, Level lvl) {
  // if this is the first level, there is no parent iterator for the current
  // iterator.
  // If the current iterator is a subsection-based iterator, the parent iterator
  // is memorized by the iterator.
  bool hasParent = lvl == 0 || !dependentLvlMap[tid][lvl].empty();

  const SparseIterator *parent =
      hasParent ? nullptr : iters[tid][lvl - 1].back().get();
  auto &it = getCurIterator(tid, lvl);
  it.genInit(builder, loc, parent);

  // Locates the randon accessible iterator to 0.
  if (it.randomAccessible())
    it.locate(builder, loc, C_IDX(0));
}

void LoopEmitter::exitForLoop(RewriterBase &rewriter, Location loc,
                              MutableArrayRef<Value> reduc) {
  const LoopInfo &loopInfo = loopStack.back();
  if (emitStrategy == SparseEmitStrategy::kSparseIterator) {
    auto iterateOp = llvm::cast<IterateOp>(loopInfo.loop);
    assert(reduc.size() == iterateOp.getNumResults());
    rewriter.create<sparse_tensor::YieldOp>(loc, reduc);
    // Exit the loop.
    rewriter.setInsertionPointAfter(iterateOp);
    // In-place update reduction variables.
    llvm::copy(iterateOp.getResults(), reduc.begin());
    return;
  }
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(loopInfo.loop)) {
    if (!reduc.empty()) {
      assert(reduc.size() == forOp.getNumResults());
      rewriter.create<scf::YieldOp>(loc, reduc);
    }
    // Exit the loop.
    rewriter.setInsertionPointAfter(forOp);
    // In-place update reduction variables.
    llvm::copy(forOp.getResults(), reduc.begin());
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
      Block *redBlock = &redOp.getReductions().front().front();
      rewriter.setInsertionPointToEnd(redBlock);
      Operation *newRed = rewriter.clone(*redExp);
      // Replaces arguments of the reduction expression by using the block
      // arguments from scf.reduce.
      rewriter.modifyOpInPlace(
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
  SmallVector<Value> operands;
  ValueRange whileRes = whileOp.getResults();

  for (auto [tid, lvl] : unpackTensorLevelRange(loopInfo.tidLvls)) {
    SparseIterator &it = getCurIterator(tid, lvl);
    if (!it.randomAccessible()) {
      // Forward the sparse iterator.
      Value cmp = CMPI(eq, it.getCrd(), iv);
      it.forwardIf(builder, loc, cmp);
      operands.append(it.getCursor().begin(), it.getCursor().end());
      // const Value newPos = whileOp->getResult(o++);
      // Following loops continue iteration from the break point of the
      // current while loop.
      whileRes = it.linkNewScope(whileRes);
    } else {
      // Make sure randomly accessible (dense) iterator is set to the right
      // position according to the universal index.
      Value uniIdx = whileOp.getResults().back();
      it.locate(builder, loc, uniIdx);
    }
  }

  // Reduction value from users.
  for (auto &i : reduc) {
    operands.push_back(i);
    // Update user reduction variables.
    i = whileRes.front();
    whileRes = whileRes.drop_front();
  }

  // An (optional) universal index.
  if (operands.size() < whileOp.getNumResults()) {
    assert(operands.size() + 1 == whileOp.getNumResults());
    // The last one is the universial index.
    operands.push_back(ADDI(iv, one));
    // update the loop starting point of current loop sequence
    loopSeqStack.back().first = whileOp->getResults().back();
  }

  if (!operands.empty())
    YIELD(operands);

  builder.setInsertionPointAfter(whileOp);
}

void LoopEmitter::exitCurrentLoop(RewriterBase &rewriter, Location loc,
                                  MutableArrayRef<Value> reduc) {
  // Clean up the values, it would help use to discover potential bug at a
  // earlier stage (instead of silently using a wrong value).
  const LoopInfo &loopInfo = loopStack.back();
  if (emitStrategy == SparseEmitStrategy::kSparseIterator) {
    Operation *p = loopInfo.loop;
    if (isa<IterateOp>(p))
      rewriter.create<sparse_tensor::YieldOp>(loc, reduc);

    // Exit the loop.
    rewriter.setInsertionPointAfter(p);
    // In-place update reduction variables.
    llvm::copy(p->getResults(), reduc.begin());
    loopStack.pop_back();
    return;
  }

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
// Loop generation utils
//===----------------------------------------------------------------------===//

std::pair<Operation *, Value> sparse_tensor::genCoIteration(
    OpBuilder &builder, Location loc, ArrayRef<SparseIterator *> spIters,
    MutableArrayRef<Value> reduc, Value uniIdx, bool userReducFirst) {
  // NOTE: the slice driven tensor-related reduction variable must
  // appear before normal tensors.

  // The set of induction variables for the while loop.
  SmallVector<Value> ivs;

  // TODO: remove the flag after full migration. Currently
  // `sparse_tensor.coiterate` operation (must) put user provided reduction
  // values at the front of the block list, while direct sparsification to scf
  // loops put them at the end.
  if (userReducFirst)
    ivs.append(reduc.begin(), reduc.end());

  // Construct the while-loop with a parameter for each coordinate.
  for (SparseIterator *it : spIters) {
    ValueRange itVals = it->getCursor();
    ivs.append(itVals.begin(), itVals.end());
  }

  if (!userReducFirst)
    ivs.append(reduc.begin(), reduc.end());

  // Update universal index.
  if (uniIdx)
    ivs.push_back(uniIdx);

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

  for (SparseIterator *it : spIters) {
    auto [cond, remArgs] = it->genWhileCond(builder, loc, bArgs);
    whileCond = !whileCond ? cond : ANDI(whileCond, cond);
    bArgs = remArgs;
  }
  // The remaining block arguments are user-provided reduction values and an
  // optional universal index. Make sure their sizes match.
  assert(bArgs.size() == reduc.size() + (uniIdx ? 1 : 0));
  builder.create<scf::ConditionOp>(loc, whileCond, before->getArguments());

  // Generates loop body.
  builder.setInsertionPointToStart(after);
  ValueRange aArgs = after->getArguments();
  // Since some LoopCondKind might need extra checks to filter out invalid
  // iterations, we maintains another array to hold the iteration arguments to
  // yield if the checks fails.
  SmallVector<Value> nextArgs(aArgs.begin(), aArgs.end());

  for (SparseIterator *it : spIters) {
    aArgs = it->linkNewScope(aArgs);
    // Dereference the iterator to cache the coordinate.
    it->deref(builder, loc);
  }

  // In-place update on reduction variable.
  for (unsigned i = 0, e = reduc.size(); i < e; i++)
    reduc[i] = aArgs[i];

  Value min;
  // Finds the minimum coordinate
  if (!uniIdx) {
    for (SparseIterator *it : spIters) {
      if (min) {
        Value cmp = CMPI(ult, it->getCrd(), min);
        min = SELECT(cmp, it->getCrd(), min);
      } else {
        min = it->getCrd();
      }
    }
  } else {
    // Otherwise, universal index is the minimal pos.
    min = whileOp.getAfterArguments().back();
  }

  return {whileOp, min};
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
