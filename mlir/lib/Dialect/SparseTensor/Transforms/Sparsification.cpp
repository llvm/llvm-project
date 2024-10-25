//===- Sparsification.cpp - Implementation of sparsification --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements converting sparse tensor types to actual sparse code.
//
//===----------------------------------------------------------------------===//

#include "Utils/CodegenEnv.h"
#include "Utils/CodegenUtils.h"
#include "Utils/LoopEmitter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"

#include <optional>

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Sparsifier analysis methods.
//===----------------------------------------------------------------------===//

/// Returns true iff affine expression is invariant. Sets the
/// parameter `isCurrentLoop` when expression just became invariant.
static bool isInvariantAffine(AffineExpr a, LoopId curr, bool &isCurrentLoop) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    const LoopId i = cast<AffineDimExpr>(a).getPosition();
    if (i + 1 == curr) {
      isCurrentLoop = true;
      return true; // becomes invariant at current loop
    }
    return i < curr; // invariant when already generated
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = cast<AffineBinaryOpExpr>(a);
    return isInvariantAffine(binOp.getLHS(), curr, isCurrentLoop) &&
           isInvariantAffine(binOp.getRHS(), curr, isCurrentLoop);
  }
  default: {
    assert(isa<AffineConstantExpr>(a));
    return true;
  }
  }
}

/// Helper method to inspect affine expressions. Rejects cases where the
/// same index is used more than once. Also rejects compound affine
/// expressions in sparse dimensions.
static bool findAffine(Merger &merger, TensorId tid, Level lvl, AffineExpr a,
                       LevelType lt, bool setLvlFormat = true) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    const LoopId idx = merger.makeLoopId(cast<AffineDimExpr>(a).getPosition());
    if (!isUndefLT(merger.getLvlType(tid, idx)))
      return false; // used more than once
    if (setLvlFormat)
      merger.setLevelAndType(tid, idx, lvl, lt);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Constant: {
    assert(lt.hasDenseSemantic());
    if (auto binOp = dyn_cast<AffineBinaryOpExpr>(a)) {
      // We do not set dim level format for affine expression like d0 + d1 on
      // either loop index at d0 or d1. We continue the recursion merely to
      // check whether current affine is admissible or not.
      return findAffine(merger, tid, lvl, binOp.getLHS(), lt, false) &&
             findAffine(merger, tid, lvl, binOp.getRHS(), lt, false);
    }
    // Falls through when it is a constant Affine
    return true;
  }
  default:
    return false;
  }
}

/// Helper method to inspect affine expressions for index variable reduction
/// based codegen. It finds the dependent index set for all tensor levels in the
/// current expression we are generating.
///
/// For example, when handling A[i+j][j+k], we build the two way mapping in
/// merger between (tensor, level) pairs and their dependent index variable set:
/// A_0 <=> [i, j] and A_1 <=> [j, k]
///
/// It rejects cases (returns false)
/// 1st, when the same index is used more than once, e.g., A[i+j][i]
/// 2nd, when multiplication is used in the non-trivial index expression.
/// 3rd, when a constant operand is used in the non-trivial index expression.
///
/// TODO: constant should be easy to handle.
static bool findDepIdxSet(Merger &merger, TensorId tensor, Level lvl,
                          AffineExpr a, LevelType lt, bool isSubExp = false,
                          int64_t coefficient = 1) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    // Only allow positive coefficients on AffineDimExpr.
    if (coefficient <= 0)
      return false;

    const LoopId idx = merger.makeLoopId(cast<AffineDimExpr>(a).getPosition());
    if (!isUndefLT(merger.getLvlType(tensor, idx)))
      return false; // used more than once, e.g., A[i][i]

    // TODO: Generalizes the following two cases. A[i] (with trivial index
    // expression) can be treated as a special affine index expression. We do
    // not necessarily need to differentiate them.
    if (!isSubExp) {
      assert(coefficient == 1);
      merger.setLevelAndType(tensor, idx, lvl, lt);
    }

    if (isSubExp) {
      // The current loops appears in more than one affine expressions on the
      // same tensor. We can not handle this case. e.g., A[i+j][i+k], `i` is
      // used twice.
      if (merger.hasDependentLvl(idx, tensor)) {
        // TODO: This can be supported by coiterate slices if the loop idx is
        // appeared on affine index for different tensor, or take slice on
        // multiple dimensions when it is on the same tensor.
        // E.g.,
        // `d0 + d1` for indexing t0[lvl0] and `d0 + d2` for indexing t1[lvl0]
        // d0_1 = getNextSliceOffset t0 along lvl0
        // d0_2 = getNextSliceOffset t1 along lvl0
        // if d0_1 == d0_2 then d0 = d0_1 = d0_1
        // else increase min(d0_1, d0_2).
        return false;
      }
      merger.setLoopDependentTensorLevel(idx, tensor, lvl, lt, coefficient);
    }
    return true;
  }
  case AffineExprKind::Constant:
  case AffineExprKind::Mul: {
    // TODO: Support index expression like `2 * d0`, we now only support more
    // complicated cases like `2 * d0 + d1`.
    if (!isSubExp)
      return false;

    // TODO: Support Constant AffineExp for slice-based codegen
    if (isa<AffineConstantExpr>(a))
      llvm_unreachable("Not yet implemented");

    auto binOp = cast<AffineBinaryOpExpr>(a);
    auto lhs = binOp.getLHS(), rhs = binOp.getRHS();
    if (isa<AffineConstantExpr>(rhs))
      std::swap(lhs, rhs);
    // Must be in form of `constant * d`.
    assert(isa<AffineConstantExpr>(lhs) && isa<AffineDimExpr>(rhs));
    int64_t coefficient = cast<AffineConstantExpr>(lhs).getValue();
    return findDepIdxSet(merger, tensor, lvl, rhs, lt, isSubExp, coefficient);
  }
  case AffineExprKind::Add: {
    auto binOp = cast<AffineBinaryOpExpr>(a);
    return findDepIdxSet(merger, tensor, lvl, binOp.getLHS(), lt, true) &&
           findDepIdxSet(merger, tensor, lvl, binOp.getRHS(), lt, true);
  }
  default:
    return false;
  }
}

/// Gets the total number of compound affine expressions in the
/// `getMatchingIndexingMap` for the given tensor.  For the following inputs:
///
/// map = (d0, d1, d2) => (d0 + d1 : compressed, d2 : compressed)
///
/// Returns 1 (because the first level is compressed and its corresponding
/// indexing-expression is `d0 + d1`)
static unsigned getNumNonTrivialIdxExpOnSparseLvls(AffineMap map,
                                                   Value tensor) {
  // The `tensor` is not guaranteed to have `RankedTensorType`, therefore
  // we can't use `getRankedTensorType`/`getSparseTensorType` here.
  // However, we don't need to handle `StorageSpecifierType`, so we
  // can use `SparseTensorType` once we guard against non-tensors.
  const auto rtp = dyn_cast<RankedTensorType>(tensor.getType());
  if (!rtp)
    return 0;
  const SparseTensorType stt(rtp);

  const Level lvlRank = stt.getLvlRank();
  const auto exprs = map.getResults();
  assert(static_cast<Dimension>(exprs.size()) == lvlRank &&
         "AffineMap does not have dimension-rank many results");
  unsigned num = 0;
  for (Level l = 0; l < lvlRank; l++) {
    if (!isa<AffineDimExpr>(exprs[l]) && !stt.getLvlType(l).hasDenseSemantic())
      num++;
  }
  return num;
}

/// Gets the total number of sparse levels with compound affine
/// expressions, summed over all operands of the `GenericOp`.
static unsigned getNumNonTrivialIdxExpOnSparseLvls(linalg::GenericOp op) {
  unsigned num = 0;
  for (OpOperand &t : op->getOpOperands())
    num += getNumNonTrivialIdxExpOnSparseLvls(op.getMatchingIndexingMap(&t),
                                              t.get());
  return num;
}

// Returns true iff output has nontrivial affine indices.
static bool hasNonTrivialAffineOnSparseOut(linalg::GenericOp op) {
  OpOperand *out = op.getDpsInitOperand(0);
  if (getSparseTensorType(out->get()).isAllDense())
    return false;
  return getNumNonTrivialIdxExpOnSparseLvls(op.getMatchingIndexingMap(out),
                                            out->get());
}

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
/// Returns true if the sparse annotations and affine subscript
/// expressions of all tensors are admissible. Returns false if
/// no annotations are found or inadmissible constructs occur.
/// We currently support two different ways to handle non-trivial index
/// expression on sparse tensors, and they accept different affine expressions.
/// When using dependent index reducton-based approach, it currently only
/// supports affine addition index expression.
static bool findSparseAnnotations(CodegenEnv &env, bool idxReducBased) {
  bool annotated = false;
  for (OpOperand &t : env.op()->getOpOperands()) {
    const TensorId tid = env.makeTensorId(t.getOperandNumber());
    const auto map = env.op().getMatchingIndexingMap(&t);
    const auto enc = getSparseTensorEncoding(t.get().getType());
    if (enc)
      annotated = true;
    const Level lvlRank = map.getNumResults();
    assert(!enc || lvlRank == enc.getLvlRank());
    assert(static_cast<Level>(env.op().getRank(&t)) == lvlRank);
    // We only need to do index reduction if there is at least one
    // non-trivial index expression on sparse levels. If all non-trivial
    // index expression is on dense levels, we can efficiently rely on
    // the random access to locate the element.
    bool needIdxReduc =
        enc && getNumNonTrivialIdxExpOnSparseLvls(map, t.get()) != 0;
    // If then current tensor being inspected requires affine index, it need
    // to be sliced.
    for (Level l = 0; l < lvlRank; l++) {
      const AffineExpr a = map.getResult(l);
      const LevelType lt = enc.getLvlType(l);
      if (idxReducBased && needIdxReduc) {
        if (!findDepIdxSet(env.merger(), tid, l, a, lt))
          return false; // inadmissible affine expression
      } else {
        if (!findAffine(env.merger(), tid, l, a, lt))
          return false; // inadmissible affine expression
      }
    }
  }
  return annotated;
}

//===----------------------------------------------------------------------===//
// Sparsifier synthesis methods (statements and expressions).
//===----------------------------------------------------------------------===//

/// Local bufferization of all dense and sparse data structures.
static void genBuffers(CodegenEnv &env, OpBuilder &builder) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  assert(op.getNumOperands() == op.getNumDpsInputs() + 1);

  SmallVector<Range, 4> loopRange =
      llvm::cast<linalg::LinalgOp>(op.getOperation())
          .createLoopRanges(builder, loc);

  env.emitter().initializeLoopEmit(
      builder, loc,
      /// Generates buffer for the output tensor.
      /// Note that all sparse kernels assume that when all elements are written
      /// to (viz. x(i) = y(i) * z(i)), the output buffer is already initialized
      /// to all zeroes and only nonzeroes values are computed and written out.
      /// For updates (viz. x(i) += y(i) * z(i)), only nonzeroes values are used
      /// for the updates and no assumption on the original contents of the
      /// output buffer is necessary.
      [&op](OpBuilder &builder, Location loc, Value memref,
            Value tensor) -> Value {
        // Must not be a sparse tensor.
        assert(!getSparseTensorEncoding(tensor.getType()));
        // Two output tensor references should point to the same object.
        OpOperand *lhs = op.getDpsInitOperand(0);
        assert(lhs->get() == tensor);
        // An output tensor can simply materialize from the buffer of the tensor
        // that appears in the outs() clause. For updates, this has the
        // advantage that only the nonzero value are involved in the
        // computation, keeping the operation O(nnz). In all other cases, we are
        // forced to zero out the buffer to enforce the assumption above, which
        // may negatively impact running complexity (viz. O(n^2 + nnz) vs.
        // O(nnz) for matrices).
        // TODO: use better analysis to avoid zeroing out the buffer?
        bool isInit = op.isInitTensor(lhs);
        Value init = memref;
        if (!isInit) {
          Value zero = constantZero(builder, loc,
                                    getElementTypeOrSelf(tensor.getType()));
          builder.create<linalg::FillOp>(loc, ValueRange{zero},
                                         ValueRange{init});
        }
        return init;
      },
      [&loopRange](OpBuilder &b, Location loc, Level l) {
        assert(l < loopRange.size());
        return mlir::getValueOrCreateConstantIndexOp(b, loc, loopRange[l].size);
      });
}

/// Generates index for load/store on sparse tensor.
static Value genIndex(CodegenEnv &env, OpOperand *t) {
  const auto map = env.op().getMatchingIndexingMap(t);
  const auto stt = getSparseTensorType(t->get());
  const Level lvlRank = stt.getLvlRank();
  assert(static_cast<Level>(map.getNumResults()) == lvlRank);
  const AffineExpr a = map.getResult(lvlRank - 1);
  assert(a.getKind() == AffineExprKind::DimId);
  const LoopId idx = env.makeLoopId(cast<AffineDimExpr>(a).getPosition());
  return env.getLoopVar(idx);
}

/// Generates subscript for load/store on a dense or sparse tensor.
static Value genSubscript(CodegenEnv &env, OpBuilder &builder, OpOperand *t,
                          SmallVectorImpl<Value> &args) {
  const Location loc = env.op().getLoc();
  const TensorId tid = env.makeTensorId(t->getOperandNumber());
  const auto map = env.op().getMatchingIndexingMap(t);
  const auto stt = getSparseTensorType(t->get());
  if (stt.hasEncoding()) {
    // For sparse tensors we only push the last-level's position onto `args`.
    const auto pos = env.emitter().getValPosits(tid);
    assert(!pos.empty());
    args.append(pos);
    // Simply returns the tensor to extract value using iterators.
    if (env.options().sparseEmitStrategy == SparseEmitStrategy::kSparseIterator)
      return t->get();
  } else {
    // For dense tensors we push all level's coordinates onto `args`.
    const Level lvlRank = stt.getLvlRank();
    assert(static_cast<Level>(map.getNumResults()) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      const auto lvlExpr = map.getResult(l);
      const auto lvlCrd = env.emitter().genAffine(builder, loc, lvlExpr);
      args.push_back(lvlCrd);
    }
  }
  return env.emitter().getValBuffer()[tid];
}

/// Generates insertion code to implement dynamic tensor load.
static Value genInsertionLoad(CodegenEnv &env, OpBuilder &builder,
                              OpOperand *t) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  // Direct lexicographic coordinate order, tensor loads as zero.
  if (!env.isExpand()) {
    Type tp = getElementTypeOrSelf(t->get().getType());
    return constantZero(builder, loc, tp);
  }
  // Load from expanded access pattern.
  Value index = genIndex(env, t);
  return builder.create<memref::LoadOp>(loc, env.getExpandValues(), index);
}

/// Generates insertion code to implement dynamic tensor load for reduction.
static Value genInsertionLoadReduce(CodegenEnv &env, OpBuilder &builder,
                                    OpOperand *t) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  Value identity = env.getCustomRedId();
  // Direct lexicographic coordinate order, tensor loads as identity.
  if (!env.isExpand())
    return identity;
  // Load from expanded access pattern if filled, identity otherwise.
  Value values = env.getExpandValues();
  Value filled = env.getExpandFilled();
  Value index = genIndex(env, t);
  Value isFilled = builder.create<memref::LoadOp>(loc, filled, index);
  Value valAtIndex = builder.create<memref::LoadOp>(loc, values, index);
  return builder.create<arith::SelectOp>(loc, isFilled, valAtIndex, identity);
}

static Value genConditionalInsert(Location loc, OpBuilder &builder, Value cond,
                                  Value sparseOut, ValueRange ivs, Value v) {
  scf::IfOp condInsert =
      builder.create<scf::IfOp>(loc, sparseOut.getType(), cond, true);
  // True branch.
  builder.setInsertionPointToStart(condInsert.thenBlock());
  Value res = builder.create<tensor::InsertOp>(loc, v, sparseOut, ivs);
  builder.create<scf::YieldOp>(loc, res);
  // False branch.
  builder.setInsertionPointToStart(condInsert.elseBlock());
  builder.create<scf::YieldOp>(loc, sparseOut);
  // Value assignment.
  builder.setInsertionPointAfter(condInsert);
  return condInsert.getResult(0);
}

/// Generates insertion code to implement dynamic tensor store.
static void genInsertionStore(CodegenEnv &env, OpBuilder &builder, OpOperand *t,
                              Value rhs) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  // Direct insertion in lexicographic coordinate order.
  if (!env.isExpand()) {
    const LoopId numLoops = op.getRank(t);
    // Retrieves the first `numLoop` induction variables.
    SmallVector<Value> ivs = llvm::to_vector(llvm::drop_end(
        env.emitter().getLoopIVsRange(), env.getCurrentDepth() - numLoops));
    Value chain = env.getInsertionChain();
    if (env.isValidLexInsert()) {
      // Generates runtime check for a valid lex during reduction,
      // to avoid inserting the identity value for empty reductions.
      //   if (validLexInsert) then
      //     insert(rhs) into chain
      //     return updated chain
      //   else
      //     return unmodified chain
      Value out = genConditionalInsert(loc, builder, env.getValidLexInsert(),
                                       chain, ivs, rhs);
      env.updateInsertionChain(out);
    } else {
      Value sparseOut;
      if (!hasAnySparseType(env.op().getInputs().getTypes())) {
        // This is an all-dense -> sparse kernel, test rhs != 0 before
        // insertion.
        Value nz = genIsNonzero(builder, loc, rhs);
        sparseOut = genConditionalInsert(loc, builder, nz, chain, ivs, rhs);
      } else {
        sparseOut = builder.create<tensor::InsertOp>(loc, rhs, chain, ivs);
      }
      // Generates regular insertion chain.
      env.updateInsertionChain(sparseOut);
    }
    return;
  }
  // Generates insertion code along expanded access pattern.
  //   if (!expFilled[i]) then
  //     expFilled[i] = true
  //     expAdded[inserts++] = i
  //   endif
  //   values[i] = rhs
  Value values = env.getExpandValues();
  Value filled = env.getExpandFilled();
  Value added = env.getExpandAdded();
  Value count = env.getExpandCount();
  Value index = genIndex(env, t);
  Value fval = constantI1(builder, loc, false);
  Value tval = constantI1(builder, loc, true);
  // If statement.
  Value isFilled = builder.create<memref::LoadOp>(loc, filled, index);
  Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             isFilled, fval);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), cond,
                                             /*else=*/true);
  // True branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<memref::StoreOp>(loc, tval, filled, index);
  builder.create<memref::StoreOp>(loc, index, added, count);
  Value one = constantIndex(builder, loc, 1);
  Value add = builder.create<arith::AddIOp>(loc, count, one);
  builder.create<scf::YieldOp>(loc, add);
  // False branch.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, count);
  builder.setInsertionPointAfter(ifOp);
  // Value assignment.
  env.updateExpandCount(ifOp.getResult(0));
  builder.create<memref::StoreOp>(loc, rhs, values, index);
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(CodegenEnv &env, OpBuilder &builder, ExprId exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = env.exp(exp).val;
  if (val)
    return val;
  // Get tensor operand.
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  OpOperand *t = &op->getOpOperand(env.exp(exp).tensor);
  // Fold binary-valued tensor into explicit value.
  const auto stt = getSparseTensorType(t->get());
  if (auto explVal = stt.getExplicitVal())
    return genValFromAttr(builder, loc, explVal);
  // Load during insertion.
  if (env.isSparseOutput(t)) {
    if (env.isCustomReduc())
      return genInsertionLoadReduce(env, builder, t);
    return genInsertionLoad(env, builder, t);
  }

  // Actual load.
  SmallVector<Value> args;
  Value ptr = genSubscript(env, builder, t, args);
  if (llvm::isa<TensorType>(ptr.getType())) {
    assert(env.options().sparseEmitStrategy ==
               SparseEmitStrategy::kSparseIterator &&
           args.size() == 1);
    return builder.create<ExtractValOp>(loc, ptr, args.front());
  }
  return builder.create<memref::LoadOp>(loc, ptr, args);
}

/// Generates a store on a dense or sparse tensor.
static void genTensorStore(CodegenEnv &env, OpBuilder &builder, ExprId exp,
                           Value rhs) {
  // Only unary and binary are allowed to return an uninitialized rhs
  // to indicate missing output. Or otherwise a custom reduction that
  // received no value to accumulate.
  if (!rhs) {
    assert(env.exp(exp).kind == TensorExp::Kind::kUnary ||
           env.exp(exp).kind == TensorExp::Kind::kBinary ||
           env.exp(exp).kind == TensorExp::Kind::kReduce);
    return;
  }
  // Test if this is a scalarized reduction.
  if (env.isReduc()) {
    env.updateReduc(rhs);
    return;
  }
  // Regular store.
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  OpOperand *t = op.getDpsInitOperand(0);
  if (!env.isSparseOutput(t)) {
    SmallVector<Value> args;
    Value ptr = genSubscript(env, builder, t, args);
    builder.create<memref::StoreOp>(loc, rhs, ptr, args);
    return;
  }
  // Store during sparse insertion.
  if (env.exp(exp).kind != TensorExp::Kind::kSelect) {
    genInsertionStore(env, builder, t, rhs);
    return;
  }
  // Select operation insertion.
  Value chain = env.getInsertionChain();
  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, chain.getType(), rhs, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  // Existing value was preserved to be used here.
  assert(env.exp(exp).val);
  Value v0 = env.exp(exp).val;
  genInsertionStore(env, builder, t, v0);
  env.merger().clearExprValue(exp);
  // Yield modified insertion chain along true branch.
  Value mchain = env.getInsertionChain();
  builder.create<scf::YieldOp>(op.getLoc(), mchain);
  // Yield original insertion chain along false branch.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, chain);
  // Done with if statement.
  env.updateInsertionChain(ifOp->getResult(0));
  builder.setInsertionPointAfter(ifOp);
}

/// Generates an invariant value.
inline static Value genInvariantValue(CodegenEnv &env, ExprId exp) {
  return env.exp(exp).val;
}

/// Semi-ring branches are simply inlined by the sparsifier. Prior
/// analysis has verified that all computations are "local" to the inlined
/// branch or otherwise invariantly defined outside the loop nest, with the
/// exception of index computations, which need to be relinked to actual
/// inlined cloned code.
static Value relinkBranch(CodegenEnv &env, RewriterBase &rewriter, Block *block,
                          Value e) {
  if (auto arg = dyn_cast<BlockArgument>(e)) {
    // Direct arguments of the original linalg op must be converted
    // into dense tensor loads. Note that we should not encounter
    // anything else. This needs to be verified by semi-ring ops.
    linalg::GenericOp op = env.op();
    if (arg.getOwner()->getParentOp() == op) {
      const TensorId tid = env.makeTensorId(arg.getArgNumber());
      OpOperand *t = &op->getOpOperand(tid);
      assert(!getSparseTensorType(t->get()).hasEncoding()); // dense!
      SmallVector<Value> args;
      Value ptr = genSubscript(env, rewriter, t, args);
      return rewriter.create<memref::LoadOp>(op.getLoc(), ptr, args);
    }
  } else if (Operation *def = e.getDefiningOp()) {
    // Handle index computation.
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return env.getLoopVar(env.makeLoopId(indexOp.getDim()));
    // When still defined in new body, recurse into operands.
    if (def->getBlock() == block) {
      rewriter.setInsertionPoint(def);
      for (unsigned i = 0, n = def->getNumOperands(); i < n; i++) {
        rewriter.modifyOpInPlace(def, [&]() {
          def->setOperand(
              i, relinkBranch(env, rewriter, block, def->getOperand(i)));
        });
      }
    }
  }
  return e;
}

/// Recursively generates tensor expression.
static Value genExp(CodegenEnv &env, RewriterBase &rewriter, ExprId e) {
  if (e == ::mlir::sparse_tensor::detail::kInvalidId)
    return Value();

  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  const TensorExp &exp = env.exp(e);
  const auto kind = exp.kind;
  if (kind == TensorExp::Kind::kTensor)
    return genTensorLoad(env, rewriter, e);
  if (kind == TensorExp::Kind::kInvariant)
    return genInvariantValue(env, e);
  if (kind == TensorExp::Kind::kLoopVar)
    return env.getLoopVar(exp.loop);

  if (kind == TensorExp::Kind::kReduce)
    env.startCustomReduc(e); // enter custom

  // If either lhs/rhs is a synthetic zero, we infer the type for the zero value
  // based on the type of the other operand.
  Value v0, v1;
  if (exp.children.e0 != ::mlir::sparse_tensor::detail::kInvalidId &&
      env.exp(exp.children.e0).kind == TensorExp::Kind::kSynZero) {
    v1 = genExp(env, rewriter, exp.children.e1);
    v0 = constantZero(rewriter, loc, v1.getType());
  } else if (exp.children.e1 != ::mlir::sparse_tensor::detail::kInvalidId &&
             env.exp(exp.children.e1).kind == TensorExp::Kind::kSynZero) {
    v0 = genExp(env, rewriter, exp.children.e0);
    v1 = constantZero(rewriter, loc, v0.getType());
  } else {
    v0 = genExp(env, rewriter, exp.children.e0);
    v1 = genExp(env, rewriter, exp.children.e1);
  }

  Value ee;
  if (kind == TensorExp::Kind::kReduce && (!v0 || !v1)) {
    // custom reduce did not receive a value
  } else {
    ee = env.merger().buildExp(rewriter, loc, e, v0, v1);
    if (ee &&
        (kind == TensorExp::Kind::kUnary || kind == TensorExp::Kind::kBinary ||
         kind == TensorExp::Kind::kBinaryBranch ||
         kind == TensorExp::Kind::kReduce ||
         kind == TensorExp::Kind::kSelect)) {
      OpBuilder::InsertionGuard guard(rewriter);
      ee = relinkBranch(env, rewriter, ee.getParentBlock(), ee);
    }
  }

  if (kind == TensorExp::Kind::kReduce)
    env.endCustomReduc(); // exit custom

  if (kind == TensorExp::Kind::kSelect)
    env.merger().setExprValue(e, v0); // Preserve value for later use.

  return ee;
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(CodegenEnv &env, OpBuilder &builder, ExprId exp,
                          LoopId curr, bool isStart) {
  if (exp == ::mlir::sparse_tensor::detail::kInvalidId)
    return;
  if (env.exp(exp).kind == TensorExp::Kind::kTensor) {
    // Inspect tensor indices.
    linalg::GenericOp op = env.op();
    OpOperand &t = op->getOpOperand(env.exp(exp).tensor);
    const auto map = op.getMatchingIndexingMap(&t);
    const auto stt = getSparseTensorType(t.get());
    const Level lvlRank = stt.getLvlRank();
    assert(static_cast<Level>(map.getNumResults()) == lvlRank);
    bool isCurrentLoop = curr == 0; // for scalar tensors
    for (Level l = 0; l < lvlRank; l++) {
      const AffineExpr a = map.getResult(l);
      if (!isInvariantAffine(a, curr, /*out*/ isCurrentLoop))
        return; // still in play
    }
    // All exhausted at current level.
    if (!isCurrentLoop)
      return;
    // Generate code for a scalarized reduction or invariant. Note that
    // because custom reduction lhs may occur several times in the IR,
    // we have a built-in safety for only initializing and wrapping-up
    // the scalarized reduction once.
    OpOperand *lhs = op.getDpsInitOperand(0);
    if (lhs == &t) {
      // Start or end a scalarized reduction.
      if (isStart) {
        if (env.isCustomReduc()) {
          if (!env.isReduc())
            env.startReduc(exp, env.getCustomRedId());
        } else {
          env.startReduc(exp, genTensorLoad(env, builder, exp));
        }
        if (env.hasSparseOutput())
          env.startValidLexInsert(
              constantI1(builder, env.op().getLoc(), false));
      } else {
        if (!env.isCustomReduc() || env.isReduc())
          genTensorStore(env, builder, exp, env.endReduc());
        if (env.hasSparseOutput())
          env.endValidLexInsert();
      }
    } else {
      // Start or end loop invariant hoisting of a tensor load.
      if (isStart) {
        env.merger().setExprValue(exp, genTensorLoad(env, builder, exp));
      } else {
        env.merger().clearExprValue(exp);
      }
    }
  } else if (env.exp(exp).kind != TensorExp::Kind::kInvariant &&
             env.exp(exp).kind != TensorExp::Kind::kLoopVar &&
             env.exp(exp).kind != TensorExp::Kind::kSynZero) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    if (env.exp(exp).kind == TensorExp::Kind::kReduce)
      env.startCustomReduc(exp); // enter custom
    const ExprId e0 = env.exp(exp).children.e0;
    const ExprId e1 = env.exp(exp).children.e1;
    genInvariants(env, builder, e0, curr, isStart);
    genInvariants(env, builder, e1, curr, isStart);
    if (env.exp(exp).kind == TensorExp::Kind::kReduce)
      env.endCustomReduc(); // exit custom
  }
}

/// Generates an expanded access pattern in innermost dimension.
static void genExpand(CodegenEnv &env, OpBuilder &builder, LoopId curr,
                      bool isStart) {
  linalg::GenericOp op = env.op();
  OpOperand *lhs = op.getDpsInitOperand(0);
  if (!env.atExpandLevel(lhs, op.getRank(lhs), curr))
    return; // not needed at current level
  assert(!env.isReduc());
  // Generate start or end of an expanded access pattern. Note that because
  // an expansion does not rely on the ongoing contents of the sparse storage
  // scheme, we can use the original tensor as incoming SSA value (which
  // simplifies codegen a bit). If expansion on the actual contents is ever
  // needed, we will need to use the SSA value in the insertion chain instead.
  Value tensor = lhs->get();
  Location loc = op.getLoc();
  if (isStart) {
    auto dynShape = {ShapedType::kDynamic};
    Type etp = cast<ShapedType>(tensor.getType()).getElementType();
    Type t1 = MemRefType::get(dynShape, etp);
    Type t2 = MemRefType::get(dynShape, builder.getI1Type());
    Type t3 = MemRefType::get(dynShape, builder.getIndexType());
    Type t4 = builder.getIndexType();
    auto r = builder.create<ExpandOp>(loc, TypeRange({t1, t2, t3, t4}), tensor);
    assert(r.getNumResults() == 4);
    env.startExpand(r.getResult(0), r.getResult(1), r.getResult(2),
                    r.getResult(3));
  } else {
    SmallVector<Value> indices;
    for (LoopId i = 0; i < curr; i++)
      indices.push_back(env.emitter().getLoopIV(i));
    Value values = env.getExpandValues();
    Value filled = env.getExpandFilled();
    Value added = env.getExpandAdded();
    Value count = env.getExpandCount();
    Value chain = env.getInsertionChain();
    Value compress = builder.create<CompressOp>(loc, values, filled, added,
                                                count, chain, indices);
    env.updateInsertionChain(compress);
    env.endExpand();
  }
}

/// Returns parallelization strategy. Any implicit loop in the Linalg
/// operation that is marked "parallel" is a candidate. Whether it is actually
/// converted to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodegenEnv &env, bool isOuter, bool isSparse) {
  // Reject parallelization of sparse output.
  if (env.hasSparseOutput())
    return false;
  // Parallel loops on tensor expansion can cause data races.
  if (env.isExpand())
    return false;
  // Inspect strategy.
  switch (env.options().parallelizationStrategy) {
  case SparseParallelizationStrategy::kNone:
    return false;
  case SparseParallelizationStrategy::kDenseOuterLoop:
    return isOuter && !isSparse;
  case SparseParallelizationStrategy::kAnyStorageOuterLoop:
    return isOuter;
  case SparseParallelizationStrategy::kDenseAnyLoop:
    return !isSparse;
  case SparseParallelizationStrategy::kAnyStorageAnyLoop:
    return true;
  }
  llvm_unreachable("unexpected parallelization strategy");
}

/// Whether or not the current loop being generated should be parallized (if
/// possible) according to the configuration.
static bool shouldTryParallize(CodegenEnv &env, LoopId curr,
                               ArrayRef<TensorLevel> tidLvls) {
  linalg::GenericOp op = env.op();
  auto iteratorTypes = op.getIteratorTypesArray();
  bool isSparse = llvm::any_of(tidLvls, [curr, &env](TensorLevel tidLvl) {
    // Queries the LT based on the tensor and loop id, as requested by
    // `CodegenEnv::lt(TensorId, LoopId)`. The returned LT from CodegenEnv
    // should be consistent with the LT indexed by <TensorId, Level>.
    const auto lt = env.lt(env.unpackTensorLevel(tidLvl).first, curr);
    return lt.hasSparseSemantic();
  });
  return isParallelFor(env, /*isOuter=*/curr == 0, isSparse);
}

/// Emit a loop to coiterate over the list of tensor levels. The generated loop
/// can either be a for loop or while loop depending on whether there is at most
/// one sparse level in the list.
static Operation *genCoIteration(CodegenEnv &env, OpBuilder &builder,
                                 ArrayRef<TensorLevel> tidLvls,
                                 unsigned numCases, bool tryParallel,
                                 bool needsUniv) {
  Operation *loop = *env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    // Construct while-loop with a parameter for each index.
    return env.emitter().enterCoIterationOverTensorsAtLvls(
        builder, env.op().getLoc(), tidLvls, numCases, reduc, tryParallel,
        needsUniv);
  });
  assert(loop);
  return loop;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(CodegenEnv &env, OpBuilder &builder, LoopId curr,
                          unsigned numCases, bool needsUniv,
                          ArrayRef<TensorLevel> tidLvls) {
  bool tryParallel = shouldTryParallize(env, curr, tidLvls);
  return genCoIteration(env, builder, tidLvls, numCases, tryParallel,
                        needsUniv);
}

/// Generates the induction structure for a while-loop.
static void finalizeWhileOp(CodegenEnv &env, OpBuilder &builder,
                            bool needsUniv) {
  Location loc = env.op().getLoc();
  // Finalize each else branch of all if statements.
  if (env.isReduc() || env.isExpand() || env.getInsertionChain()) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               builder.getInsertionBlock()->getParentOp())) {
      // Break on IfOp for slicing filtering.
      if (ifOp->getAttr(LoopEmitter::getLoopEmitterLoopAttrName()) ==
          StringAttr::get(ifOp->getContext(), "slice"))
        break;

      unsigned y = 0;
      SmallVector<Value> yields;
      if (env.isReduc()) {
        yields.push_back(env.getReduc());
        env.updateReduc(ifOp.getResult(y++));
        if (env.isValidLexInsert()) {
          yields.push_back(env.getValidLexInsert());
          env.updateValidLexInsert(ifOp.getResult(y++));
        }
      }
      if (env.isExpand()) {
        yields.push_back(env.getExpandCount());
        env.updateExpandCount(ifOp->getResult(y++));
      }
      if (env.getInsertionChain()) {
        yields.push_back(env.getInsertionChain());
        env.updateInsertionChain(ifOp->getResult(y++));
      }
      assert(y == yields.size());
      builder.create<scf::YieldOp>(loc, yields);
      builder.setInsertionPointAfter(ifOp);
    }
  }
  // No need to set the insertion point here as LoopEmitter keeps track of the
  // basic block where scf::Yield should be inserted.
}

/// Generates a case region in the coiterate operation.
static void genCoIterationCase(CodegenEnv &env, OpBuilder &builder,
                               unsigned caseIdx, LatPointId allCase,
                               LatPointId curCase,
                               MutableArrayRef<Value> reduc) {
  assert(allCase == curCase || env.merger().latGT(allCase, curCase));
  const BitVector &allCaseBits = env.merger().lat(allCase).simple;
  const BitVector &curCaseBits = env.merger().lat(curCase).simple;

  /// Computes the subset of iterators that are valid in the current case being
  /// generated.
  I64BitSet caseBit(0);
  for (auto [idx, set] : llvm::enumerate(allCaseBits.set_bits()))
    if (curCaseBits.test(set))
      caseBit.set(idx);

  env.emitter().enterCurrentCoIterationCase(builder, env.op().getLoc(), caseBit,
                                            caseIdx, reduc);
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(CodegenEnv &env, OpBuilder &builder, LoopId curr,
                       LatPointId p) {
  Location loc = env.op().getLoc();
  SmallVector<Type> types;
  Value cond;
  env.merger().foreachTensorLoopId(
      p, /*simple=*/true,
      [&](TensorLoopId b, TensorId tid, std::optional<Level> lvl, LevelType lt,
          bool isIdxRed) {
        if (isIdxRed) {
          // Since there is no 1:1 mapping from loop to level (multiple loops
          // are required to resolve one level with non-trivial index
          // expression), we need to reconstruct the tensor level types if this
          // loop requires index reduction condition.
          assert(lvl.has_value() && isUndefLT(lt));
          auto stt = getSparseTensorType(env.op().getInputs()[tid]);
          lt = stt.getLvlType(*lvl);
        }
        assert(curr == env.merger().loop(b));
        Value clause;
        if (lt.hasSparseSemantic()) {
          assert(lvl.has_value());
          const Value crd = env.emitter().getCoord(tid, *lvl);
          const Value lvar = env.getLoopVar(curr);
          clause = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 crd, lvar);
        } else {
          assert(lt.hasDenseSemantic() || isUndefLT(lt));
          clause = constantI1(builder, loc, true);
        }
        cond = cond ? builder.create<arith::AndIOp>(loc, cond, clause) : clause;
      });
  if (env.isReduc()) {
    types.push_back(env.getReduc().getType());
    if (env.isValidLexInsert())
      types.push_back(env.getValidLexInsert().getType());
  }
  if (env.isExpand())
    types.push_back(builder.getIndexType());
  if (env.getInsertionChain())
    types.push_back(env.getInsertionChain().getType());
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
static void endIf(CodegenEnv &env, OpBuilder &builder, scf::IfOp ifOp,
                  Value redInput, Value cntInput, Value insInput,
                  Value validIns) {
  SmallVector<Value> operands;
  if (env.isReduc()) {
    operands.push_back(env.getReduc());
    env.updateReduc(redInput);
    if (env.isValidLexInsert()) {
      // Any overlapping indices during a reduction creates a valid lex insert.
      operands.push_back(constantI1(builder, env.op().getLoc(), true));
      env.updateValidLexInsert(validIns);
    }
  }
  if (env.isExpand()) {
    operands.push_back(env.getExpandCount());
    env.updateExpandCount(cntInput);
  }
  if (env.getInsertionChain()) {
    operands.push_back(env.getInsertionChain());
    env.updateInsertionChain(insInput);
  }
  if (!operands.empty())
    builder.create<scf::YieldOp>(env.op().getLoc(), operands);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
}

//===----------------------------------------------------------------------===//
// Sparsifier synthesis methods (loop sequence).
//===----------------------------------------------------------------------===//

static bool getAllTidLvlsInLatPoints(
    CodegenEnv &env, LatPointId li, LoopId curr,
    llvm::function_ref<void(TensorLevel, AffineExpr)> callback) {
  const BitVector &simple = env.lat(li).simple;
  const TensorId outTid = env.merger().getOutTensorID();
  const std::optional<Level> outLvl = env.merger().getLvl(outTid, curr);

  unsigned numloopCond = 0;
  bool hasNonUnique = false;
  env.merger().foreachTensorLoopId(
      li, [&, curr](TensorLoopId b, TensorId tid, std::optional<Level> lvl,
                    LevelType lt, bool isIdxReduc) {
        if (simple[b]) {
          if (isIdxReduc) {
            callback(env.makeTensorLevel(tid, *lvl), nullptr);
            numloopCond++;
            return;
          }
          if (isUndefLT(lt)) {
            // An undefined lt in the lattices, we probably mean to
            // generate a dense loop according to the synthetic tensor (for
            // invariants and sparse output tensor).
            if (env.merger().getSynTensorID() == tid) {
              // Coiterating with an invariant
              // e.g., out = prod(in[i][j] op invariant);
              // or a broadcast
              // e.g., out[i][j] = in[i] (j is undef for input)
              //
              // The level of the synthetic tensor is the current loop depth;
              // the rank of the synthetic tensor equals to number of loops.
              assert(curr == env.getCurrentDepth());
              lvl = curr;
            } else if (!lvl) {
              // Skips invalid lvl (e.g., when this is a zero ranked tensor).
              return;
            }
          }
          hasNonUnique = !isUniqueLT(lt) || hasNonUnique;
          callback(env.makeTensorLevel(tid, *lvl), nullptr);
          numloopCond++;
        } else if (lt.hasDenseSemantic() || isIdxReduc) {
          callback(env.makeTensorLevel(tid, *lvl), nullptr);
        } else {
          assert(isUndefLT(lt));
          linalg::GenericOp op = env.op();
          if (tid >= op.getNumDpsInputs())
            // We only handle affine expression on input tensors (for now).
            return;
          OpOperand *operand = &op->getOpOperand(tid);
          const auto stt = getSparseTensorType(operand->get());
          // Non-annotated dense tensors requires no special handling.
          if (!stt.hasEncoding())
            return;

          ArrayRef<AffineExpr> affines =
              op.getMatchingIndexingMap(operand).getResults();
          const Level lvlRank = stt.getLvlRank();
          assert(affines.size() == static_cast<size_t>(lvlRank));
          for (Level l = 0; l < lvlRank; l++) {
            AffineExpr exp = affines[l];
            // Skip simple affine expression and non-dense levels (which
            // have their own filter loop).
            LevelType lt = stt.getLvlType(l);
            if (isa<AffineDimExpr>(exp) || !lt.hasDenseSemantic())
              continue;

            // Constant affine expression are handled in genLoop.
            if (!isa<AffineConstantExpr>(exp)) {
              bool isCurrentLoop = false;
              assert(curr == env.getCurrentDepth());
              if (isInvariantAffine(exp, curr + 1, /*out*/ isCurrentLoop) &&
                  isCurrentLoop) {
                // If the compound affine is invariant and we are right at the
                // level. We need to generate the address according to the
                // affine expression. This is also the best place we can do it
                // to avoid putting it inside inner loops.
                callback(env.makeTensorLevel(tid, l), exp);
              }
            }
          }
        }
      });

  if (isDenseLT(env.lt(outTid, curr))) {
    auto stt = getSparseTensorType(env.op().getOutputs().front());
    // Note that we generate dense indices of the output tensor unconditionally,
    // since they may not appear in the lattice, but may be needed for
    // linearized env.
    // TODO: we should avoid introducing corner cases for all-dense sparse
    // tensors.
    if (stt.hasEncoding() && stt.isAllDense())
      callback(env.makeTensorLevel(outTid, *outLvl), nullptr);
  }

  if (numloopCond == 0) {
    // Corner cases where the loop bound is defined by a *unused* operand, in
    // this case, we just generate a dense "fake" loop by iterating over the
    // synthetic tensor.
    callback(env.makeTensorLevel(env.merger().getSynTensorID(), curr), nullptr);
    numloopCond++;
  }
  // If we just need to one loop conditions and the conditions is not imposed on
  // non-unique level, the loop can be generated by a for loop.
  // Or, if we are generating sparse-iterator-based loops, we always generate
  // `sparse_tensor.iterate` regardless whether the level is unique or not.
  return numloopCond == 1 &&
         (!hasNonUnique || env.options().sparseEmitStrategy ==
                               SparseEmitStrategy::kSparseIterator);
}

/// Starts a loop sequence at given level. Returns true if
/// the universal loop index must be maintained at this level.
static bool startLoopSeq(CodegenEnv &env, OpBuilder &builder, ExprId exp,
                         LoopId curr, LatSetId lts) {
  assert(!env.getLoopVar(curr));
  // Emit invariants at this loop sequence level.
  genInvariants(env, builder, exp, curr, /*isStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpand(env, builder, curr, /*isStart=*/true);
  // Emit further initialization at this loop sequence level.
  const LatPointId l0 = env.set(lts)[0];

  SmallVector<TensorLevel> tidLvls;
  getAllTidLvlsInLatPoints(env, l0, curr, [&](TensorLevel tl, AffineExpr) {
    // TODO: remove this! The same tensor level might be added for multiple
    // times due to the special handling for all-dense "sparse" output tensor
    // (see L1038).
    if (llvm::find(tidLvls, tl) != tidLvls.end())
      return;
    tidLvls.emplace_back(tl);
  });

  env.emitter().enterNewLoopSeq(builder, env.op().getLoc(), tidLvls);

  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  for (const LatPointId li : env.set(lts).drop_front())
    if (!env.merger().hasAnySparse(env.lat(li).simple))
      return true;

  return false;
}

// Generates dense affine address for encoding.
static void genConstantDenseAddressFromLevel(CodegenEnv &env,
                                             OpBuilder &builder, TensorId tid,
                                             Level startLvl) {
  // TODO: Handle affine expression on output tensor.
  linalg::GenericOp op = env.op();
  assert(tid < op.getNumDpsInputs());
  OpOperand *input = op.getDpsInputOperands()[tid];
  const auto lvlExprs = op.getMatchingIndexingMap(input).getResults();
  const auto enc = getSparseTensorEncoding(input->get().getType());
  if (enc) {
    const Location loc = op.getLoc();
    const TensorId tid = env.makeTensorId(input->getOperandNumber());
    const Level lvlRank = enc.getLvlRank();
    assert(lvlExprs.size() == static_cast<size_t>(lvlRank));
    for (Level l = startLvl; l < lvlRank; l++) {
      AffineExpr lvlExpr = lvlExprs[l];
      if (enc.getLvlType(l).hasDenseSemantic() &&
          isa<AffineConstantExpr>(lvlExpr))
        env.emitter().locateLvlAtAffineAddress(
            builder, loc, env.makeTensorLevel(tid, l), lvlExpr);
      else
        return; // break on first non-dense non-constant level
    }
  }
}

// We can generate address for constant affine expression before any loops
// starting from the first level as they do not depend on anything.
// E.g., [Dense, Dense, Sparse] -> (1, 2, d0), the addresses for the first two
// levels can be determined before loops.
static void genInitConstantDenseAddress(CodegenEnv &env,
                                        RewriterBase &rewriter) {
  for (TensorId tid = 0, e = env.op().getNumDpsInputs(); tid < e; tid++)
    genConstantDenseAddressFromLevel(env, rewriter, tid, 0);
}

/// Returns true if the lattice bit can be iterated by a for loop.
static bool translateBitsToTidLvlPairs(
    CodegenEnv &env, LatPointId li, LoopId curr,
    SmallVectorImpl<TensorLevel> &tidLvls,
    SmallVectorImpl<std::pair<TensorLevel, AffineExpr>> &affineTidLvls) {
  return getAllTidLvlsInLatPoints(env, li, curr,
                                  [&](TensorLevel tl, AffineExpr exp) {
                                    if (exp)
                                      affineTidLvls.emplace_back(tl, exp);
                                    else
                                      tidLvls.emplace_back(tl);
                                  });
}

/// Starts a single loop in current sequence.
static std::pair<Operation *, bool> startLoop(CodegenEnv &env,
                                              OpBuilder &builder, LoopId curr,
                                              LatPointId li, unsigned numCases,
                                              bool needsUniv) {
  // TODO: numCases only used when generating iterator-based loops. Cleanup
  // after fully migration.
  // The set of tensors + lvls to generate loops on
  SmallVector<TensorLevel> tidLvls;

  // The set of dense tensors with non-trivial affine expression that just
  // becomes invariant and the address are generated at the current level.
  SmallVector<std::pair<TensorLevel, AffineExpr>> affineTidLvls;
  bool isSingleCond =
      translateBitsToTidLvlPairs(env, li, curr, tidLvls, affineTidLvls);

  // Emit the for/while-loop control.
  Operation *loop = genLoop(env, builder, curr, numCases, needsUniv, tidLvls);
  Location loc = env.op().getLoc();
  for (auto [tidLvl, exp] : affineTidLvls) {
    env.emitter().locateLvlAtAffineAddress(builder, loc, tidLvl, exp);
  }

  // Until now, we have entered every <tid, lvl> pair in {cond, extra,
  // affine}Tids/Lvls. The addresses of the upcoming levels which are dependent
  // on constant affines expression may now be determined.
  auto allTidLvls =
      llvm::concat<TensorLevel>(tidLvls, llvm::make_first_range(affineTidLvls));
  for (auto [tid, lvl] : env.unpackTensorLevelRange(allTidLvls)) {
    if (tid != env.merger().getOutTensorID() &&
        tid != env.merger().getSynTensorID())
      genConstantDenseAddressFromLevel(env, builder, tid, lvl + 1);
  }

  return std::make_pair(loop, isSingleCond);
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(CodegenEnv &env, RewriterBase &rewriter, Operation *loop,
                    LatPointId li, bool needsUniv, bool isSingleCond) {
  // Either a for-loop or a while-loop that iterates over a slice.
  if (isSingleCond) {
    // Any iteration creates a valid lex insert.
    if (env.isReduc() && env.isValidLexInsert())
      env.updateValidLexInsert(constantI1(rewriter, env.op().getLoc(), true));
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    // End a while-loop.
    finalizeWhileOp(env, rewriter, needsUniv);
  } else {
    needsUniv = false;
  }
  env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    env.emitter().exitCurrentLoop(rewriter, env.op().getLoc(), reduc);
    return std::nullopt;
  });
  return needsUniv;
}

/// Ends a loop sequence at given level.
static void endLoopSeq(CodegenEnv &env, OpBuilder &builder, unsigned exp,
                       unsigned at) {
  assert(!env.getLoopVar(at));
  env.emitter().exitCurrentLoopSeq(builder, env.op().getLoc());
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(env, builder, exp, at, /*isStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
  genExpand(env, builder, at, /*isStart=*/false);
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(CodegenEnv &env, RewriterBase &rewriter, ExprId exp,
                    LoopId curr) {
  assert(curr == env.getCurrentDepth());

  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (curr == env.getLoopNum()) {
    Value rhs = genExp(env, rewriter, exp);
    genTensorStore(env, rewriter, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index.
  const LatSetId lts =
      env.merger().optimizeSet(env.merger().buildLattices(exp, curr));

  // Start a loop sequence.
  bool needsUniv = startLoopSeq(env, rewriter, exp, curr, lts);

  // When using sparse-iterator-based loops, we only need one loops, as
  // opposed to a loop sequence, to cover all the iterator spaces.
  const unsigned lsize = env.set(lts).size();
  if (env.generatingSparseIterator()) {
    // Get the largest lattice point and start a loop.
    const LatPointId li = env.set(lts)[0];
    auto [loop, isSingleCond] =
        startLoop(env, rewriter, curr, li, lsize, needsUniv);
    assert(isSingleCond == llvm::isa<IterateOp>(loop));
    // We cannot change this to `for (const LatPointId li : env.set(lts))`
    // because the loop body causes data-movement which invalidates
    // the iterator.
    for (unsigned j = 0; j < lsize; j++) {
      const LatPointId lj = env.set(lts)[j];
      const ExprId ej = env.lat(lj).exp;
      // Recurse into body of each branch.
      if (!isSingleCond) {
        env.genLoopBoundary([&, curr, j, li, lj](MutableArrayRef<Value> reduc) {
          genCoIterationCase(env, rewriter, /*caseIdx*/ j, li, lj, reduc);
          genStmt(env, rewriter, ej, curr + 1);
          // TODO: handle yield values.
          assert(reduc.empty() && "Not Implemented");
          rewriter.create<sparse_tensor::YieldOp>(env.op().getLoc());
          return std::nullopt;
        });
        // endIf(env, rewriter, ifOp, redInput, cntInput, insInput, validIns);
      } else {
        genStmt(env, rewriter, ej, curr + 1);
      }
    }
    // End a loop.
    needsUniv = endLoop(env, rewriter, loop, curr, needsUniv, isSingleCond);
  } else {
    // Emit a loop for every lattice point L0 >= Li in this loop sequence.
    for (unsigned i = 0; i < lsize; i++) {
      const LatPointId li = env.set(lts)[i];
      // Start a loop.
      auto [loop, isSingleCond] =
          startLoop(env, rewriter, curr, li, lsize, needsUniv);

      // Visit all lattices points with Li >= Lj to generate the
      // loop-body, possibly with if statements for coiteration.
      Value redInput = env.getReduc();
      Value cntInput = env.getExpandCount();
      Value insInput = env.getInsertionChain();
      Value validIns = env.getValidLexInsert();
      // We cannot change this to `for (const LatPointId lj : env.set(lts))`
      // because the loop body causes data-movement which invalidates the
      // iterator.
      for (unsigned j = 0; j < lsize; j++) {
        const LatPointId lj = env.set(lts)[j];
        const ExprId ej = env.lat(lj).exp;
        if (li == lj || env.merger().latGT(li, lj)) {
          // Recurse into body of each branch.
          if (!isSingleCond) {
            scf::IfOp ifOp = genIf(env, rewriter, curr, lj);
            genStmt(env, rewriter, ej, curr + 1);
            endIf(env, rewriter, ifOp, redInput, cntInput, insInput, validIns);
          } else {
            genStmt(env, rewriter, ej, curr + 1);
          }
        }
      }

      // End a loop.
      needsUniv = endLoop(env, rewriter, loop, curr, needsUniv, isSingleCond);
    }
  }

  // End a loop sequence.
  endLoopSeq(env, rewriter, exp, curr);
  assert(curr == env.getCurrentDepth());
}

/// Converts the result computed by the sparse kernel into the required form.
static void genResult(CodegenEnv &env, RewriterBase &rewriter) {
  linalg::GenericOp op = env.op();
  OpOperand *lhs = op.getDpsInitOperand(0);
  Value tensor = lhs->get();
  Type resType = tensor.getType();
  if (getSparseTensorEncoding(resType)) {
    // The sparse tensor rematerializes from the original sparse tensor's
    // underlying sparse storage format. For an insertion chain, the
    // tensor materializes from the chain with 'hasInserts' enabled.
    bool hasInserts = false;
    if (Value chain = env.getInsertionChain()) {
      hasInserts = true;
      tensor = chain;
    }
    rewriter.replaceOpWithNewOp<LoadOp>(op, resType, tensor, hasInserts);
  } else {
    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
    Value val = env.emitter().getValBuffer()[env.merger().getOutTensorID()];
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, val);
  }
}

//===----------------------------------------------------------------------===//
// Sparsifier rewriting methods.
//===----------------------------------------------------------------------===//

namespace {

/// Sparse rewriting rule for generic Lingalg operation.
struct GenericOpSparsifier : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparsifier(MLIRContext *context, SparsificationOptions o)
      : OpRewritePattern<linalg::GenericOp>(context), options(o) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Only accept single output operations with pure tensor semantics.
    if (op.getNumDpsInits() != 1 || !op.hasPureTensorSemantics())
      return failure();

    // Only accept trivial affine indices.
    if (hasNonTrivialAffineOnSparseOut(op))
      return failure();

    // Only accept scheduled loops.
    if (!op->hasAttr("sorted")) {
      return rewriter.notifyMatchFailure(
          op, "Loops not yet scheduled, try run --sparse-reinterpret-map "
              "before sparsification.");
    }

    // Must have been demapped as well if the generic op is sorted.
    assert(!hasAnyNonIdentityOperandsOrResults(op));

    // Sets up a code generation environment.
    const unsigned numTensors = op->getNumOperands();
    const unsigned numLoops = op.getNumLoops();
    bool needIdxRed = getNumNonTrivialIdxExpOnSparseLvls(op) != 0;
    // If we have indexing map like (d0) -> (0, d0), there might be more
    // levels then loops because of the constant index, that means we can not
    // use numLoops as the upper bound for ranks of all tensors.
    // TODO: Constant indices are currently not support on sparse tensor, but
    // are allowed in non-annotated dense tensor. Support it, it would be
    // required for sparse tensor slice rank reducing too.
    Level maxLvlRank = 0;
    for (auto operand : op.getOperands()) {
      if (auto rtp = dyn_cast<RankedTensorType>(operand.getType())) {
        maxLvlRank = std::max(maxLvlRank, SparseTensorType(rtp).getLvlRank());
      }
    }

    // Detects sparse annotations and translates the per-level sparsity
    // information for all tensors to loop indices in the kernel.
    CodegenEnv env(op, options, numTensors, numLoops, maxLvlRank);
    if (!findSparseAnnotations(env, needIdxRed))
      return failure();

    // Only standard reduction operations (add, sub, or, xor) that can be
    // sparsified by merely reducing the stored values are admissible. More
    // elaborate reduction operations (such as mul, and, min, max) would need
    // to know whether implicit zeros occur as well. They can still be
    // implemented with a custom reduction operation, accepted here as well.
    if (op.getNumReductionLoops() > 0) {
      Operation *yield = op.getRegion().front().getTerminator();
      assert(isa<linalg::YieldOp>(yield));
      Operation *redop = yield->getOperand(0).getDefiningOp();
      if (!isa<arith::AddFOp>(redop) && !isa<complex::AddOp>(redop) &&
          !isa<arith::AddIOp>(redop) && !isa<arith::SubFOp>(redop) &&
          !isa<complex::SubOp>(redop) && !isa<arith::SubIOp>(redop) &&
          !isa<arith::OrIOp>(redop) && !isa<arith::XOrIOp>(redop) &&
          !isa<ReduceOp>(redop)) {
        return failure();
      }
    }

    // Constructs the tensor expressions tree from `op`, returns failure if the
    // tree can not be built or the tensor expression is inadmissible.
    if (failed(env.initTensorExp()))
      return failure();

    // Recursively generates code if admissible.
    env.startEmit(options.sparseEmitStrategy);
    genBuffers(env, rewriter);
    // TODO: Constant affine expression should be handled differently when using
    // slice-based codegen, it does not matter now because we already reject the
    // constant expression at an earlier stage.
    genInitConstantDenseAddress(env, rewriter);
    genStmt(env, rewriter, env.getExprId(), 0);
    genResult(env, rewriter);
    return success();
  }

private:
  /// Options to control sparse code generation.
  SparsificationOptions options;
};

} // namespace

/// Populates the given patterns list with rewriting rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparsificationPatterns(
    RewritePatternSet &patterns, const SparsificationOptions &options) {
  patterns.add<GenericOpSparsifier>(patterns.getContext(), options);
}
