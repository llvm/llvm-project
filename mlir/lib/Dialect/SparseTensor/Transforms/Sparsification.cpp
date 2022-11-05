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

#include "CodegenUtils.h"

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
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Declarations of data structures.
//===----------------------------------------------------------------------===//

namespace {

// Iteration graph sorting.
enum SortMask {
  kSparseOnly = 0x0,
  kIncludeDense = 0x1,
  kIncludeUndef = 0x2,
  kIncludeAll = 0x3
};

// Reduction kinds.
enum Reduction { kNoReduc, kSum, kProduct, kAnd, kOr, kXor, kCustom };

// Code generation.
struct CodeGen {
  CodeGen(SparsificationOptions o, ValueRange tensors, unsigned numTensors,
          unsigned numLoops, OpOperand *op, unsigned nest,
          std::vector<unsigned> &ts)
      : options(o), loopEmitter(tensors, /*isLastOutput=*/true,
                                /*isSparseOut=*/op != nullptr),
        sparseOut(op), outerParNest(nest), topSort(ts) {
    if (op)
      insChain = op->get();
  }
  /// Sparsification options.
  SparsificationOptions options;
  /// Loop emitter helper class.
  SparseTensorLoopEmitter loopEmitter;
  /// Current reduction, updated during code generation. When indices of a
  /// reduction are exhausted, all inner loops can use a scalarized reduction.
  unsigned redExp = -1u;
  Value redVal;
  Reduction redKind = kNoReduc;
  unsigned redCustom = -1u;
  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order or through access pattern expansion
  // in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  unsigned outerParNest;
  Value insChain; // bookkeeping for insertion chain
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;
  // Topsort (reference should remain in scope).
  std::vector<unsigned> &topSort;

  Value getLoopIdxValue(size_t loopIdx) const {
    for (unsigned lv = 0; lv < topSort.size(); lv++)
      if (topSort[lv] == loopIdx)
        return loopEmitter.getLoopIV(lv);

    llvm_unreachable("invalid loop index");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse compiler analysis methods.
//===----------------------------------------------------------------------===//

/// Helper method to construct a permuted dimension ordering
/// that adheres to the given topological sort.
static AffineMap permute(MLIRContext *context, AffineMap m,
                         std::vector<unsigned> &topSort) {
  unsigned sz = topSort.size();
  assert(m.getNumResults() == sz && "TopoSort/AffineMap size mismatch");
  // Construct the inverse of `m`; to avoid the asymptotic complexity
  // of calling `m.getPermutedPosition` repeatedly.
  SmallVector<unsigned, 4> inv(sz);
  for (unsigned i = 0; i < sz; i++)
    inv[i] = m.getDimPosition(i);
  // Construct the permutation.
  SmallVector<unsigned, 4> perm(sz);
  for (unsigned i = 0; i < sz; i++)
    perm[i] = inv[topSort[i]];
  return AffineMap::getPermutationMap(perm, context);
}

/// Helper method to inspect affine expressions. Rejects cases where the
/// same index is used more than once. Also rejects compound affine
/// expressions in sparse dimensions.
static bool findAffine(Merger &merger, unsigned tensor, unsigned dim,
                       AffineExpr a, DimLevelType dlt,
                       bool setLvlFormat = true) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (!isUndefDLT(merger.getDimLevelType(tensor, idx)))
      return false; // used more than once

    if (setLvlFormat)
      merger.setDimAndDimLevelType(tensor, idx, dim, dlt);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    if (!isDenseDLT(dlt))
      return false; // compound only in dense dim
    auto binOp = a.cast<AffineBinaryOpExpr>();
    // We do not set dim level format for affine expresssion like d0 + d1 on
    // both loop index at d0 and d1,
    return findAffine(merger, tensor, dim, binOp.getLHS(), dlt, false) &&
           findAffine(merger, tensor, dim, binOp.getRHS(), dlt, false);
  }
  case AffineExprKind::Constant:
    return isDenseDLT(dlt); // const only in dense dim
  default:
    return false;
  }
}

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
/// Returns true if the sparse annotations and affine subscript
/// expressions of all tensors are admissible. Returns false if
/// no annotations are found or inadmissible constructs occur.
static bool findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  bool annotated = false;
  for (OpOperand &t : op->getOpOperands()) {
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    if (enc)
      annotated = true;
    assert(map.getNumResults() == op.getRank(&t));
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned tensor = t.getOperandNumber();
      AffineExpr a = map.getResult(toOrigDim(enc, d));
      if (!findAffine(merger, tensor, d, a, getDimLevelType(enc, d)))
        return false; // inadmissible affine expression
    }
  }
  return annotated;
}

/// A helper to compute a topological sort. O(n^2) time complexity
/// as we use adj matrix for the graph.
/// The sorted result will put the first Reduction iterator to the
/// latest possible index.
static bool topSortOptimal(unsigned n, ArrayRef<StringRef> iteratorTypes,
                           std::vector<unsigned> &topSort,
                           std::vector<unsigned> &inDegree,
                           std::vector<std::vector<bool>> &adjM) {
  std::vector<unsigned> redIt; // reduce iterator with 0 degree
  std::vector<unsigned> parIt; // parallel iterator with 0 degree
  for (unsigned i = 0; i < n; i++) {
    if (inDegree[i] == 0) {
      if (linalg::isReductionIterator(iteratorTypes[i]))
        redIt.push_back(i);
      else
        parIt.push_back(i);
    }
  }

  while (!redIt.empty() || !parIt.empty()) {
    // We always choose parallel iterator if there is any.
    auto &it = !parIt.empty() ? parIt : redIt;
    auto src = it.back();
    topSort.push_back(src);
    it.pop_back();
    // Update in-degree, and push 0-degree node into worklist.
    for (unsigned dst = 0; dst < n; dst++)
      if (adjM[src][dst] && --inDegree[dst] == 0) {
        if (linalg::isReductionIterator(iteratorTypes[dst]))
          redIt.push_back(dst);
        else
          parIt.push_back(dst);
      }
  }
  return topSort.size() == n;
}

/// Helper method to add all constraints from the indices in one affine
/// expression before all indices in the other affine expression. For
/// example i0+i1 < i2+i3+1 yields i0<i2, i0<i3, i1<i2, and i1<i3.
static void addAffineOrderings(std::vector<std::vector<bool>> &adjM,
                               std::vector<unsigned> &inDegree, AffineExpr a,
                               AffineExpr b, unsigned fidx) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (b)
      addAffineOrderings(adjM, inDegree, b, AffineExpr(), idx);
    else if (!adjM[fidx][idx]) {
      adjM[fidx][idx] = true;
      inDegree[idx]++;
    }
    break;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    addAffineOrderings(adjM, inDegree, binOp.getLHS(), b, fidx);
    addAffineOrderings(adjM, inDegree, binOp.getRHS(), b, fidx);
    break;
  }
  default:
    break;
  }
}

/// Computes a topologically sorted iteration graph for the linalg operation.
/// Ensures all tensors are visited in natural index order. This is essential
/// for sparse storage formats since these only support access along fixed
/// dimensions. Even for dense storage formats, however, the natural index
/// order yields innermost unit-stride access with better spatial locality.
static bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort, unsigned mask,
                                  OpOperand *skip = nullptr) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = op.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));
  std::vector<unsigned> inDegree(n, 0); // in-degree of each node.
  auto iteratorTypes = op.getIteratorTypesArray();
  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand &t : op->getOpOperands()) {
    // Skip tensor during cycle resolution.
    if (&t == skip)
      continue;
    // Get map and encoding.
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    assert(map.getNumDims() == n);
    // Skip dense tensor constraints when not requested.
    if (!(mask & SortMask::kIncludeDense) && !enc)
      continue;
    // Each tensor expression and optional dimension ordering (row-major
    // by default) puts an ordering constraint on the loop indices. For
    // example, the tensor expresion A_ijk forces the ordering i < j < k
    // on the loop indices if no explicit dimension ordering is given.
    for (unsigned d = 1, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr f = map.getResult(toOrigDim(enc, d - 1));
      AffineExpr t = map.getResult(toOrigDim(enc, d));
      addAffineOrderings(adjM, inDegree, f, t, 0);
    }
    // Push unrelated loops into sparse iteration space, so these
    // will be skipped more often.
    if (mask & SortMask::kIncludeUndef) {
      unsigned tensor = t.getOperandNumber();
      for (unsigned i = 0; i < n; i++)
        if (isCompressedDLT(merger.getDimLevelType(tensor, i)) ||
            isSingletonDLT(merger.getDimLevelType(tensor, i))) {
          for (unsigned j = 0; j < n; j++)
            if (isUndefDLT(merger.getDimLevelType(tensor, j))) {
              adjM[i][j] = true;
              inDegree[j]++;
            }
        } else {
          assert(isDenseDLT(merger.getDimLevelType(tensor, i)) ||
                 isUndefDLT(merger.getDimLevelType(tensor, i)));
        }
    }
  }
  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  topSort.clear();
  topSort.reserve(n);
  return topSortOptimal(n, iteratorTypes, topSort, inDegree, adjM);
}

/// Returns true if tensor materializes uninitialized into the computation.
static bool isMaterializing(Value val) {
  return val.getDefiningOp<tensor::EmptyOp>() ||
         val.getDefiningOp<bufferization::AllocTensorOp>();
}

/// Returns true when the tensor expression is admissible for codegen.
/// Since all sparse input tensors are admissible, we just need to check
/// whether the out tensor in the tensor expression codegen is admissible.
/// Sets `sparseOut` to the tensor and `outerParNest` to the outer injective
/// nesting depth when a "truly dynamic" sparse tensor output occurs.
static bool isAdmissibleTensorExp(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort, unsigned exp,
                                  OpOperand **sparseOut,
                                  unsigned &outerParNest) {
  OpOperand *lhs = op.getDpsInitOperand(0);
  unsigned tensor = lhs->getOperandNumber();
  auto enc = getSparseTensorEncoding(lhs->get().getType());
  // An non-annotated output tensor is assumed dense, and becomes a random
  // access n-dim memref. Admissible since insertions cannot occur.
  if (!enc)
    return true;
  // An all-dense annotated "sparse" output tensor becomes a linearized random
  // access 1-dim memref. Also admissible since insertions cannot occur.
  bool allDense = true;
  auto iteratorTypes = op.getIteratorTypesArray();
  unsigned numLoops = iteratorTypes.size();
  for (unsigned i = 0; i < numLoops; i++)
    if (isCompressedDLT(merger.getDimLevelType(tensor, i)) ||
        isSingletonDLT(merger.getDimLevelType(tensor, i))) {
      allDense = false;
      break;
    } else {
      assert(isDenseDLT(merger.getDimLevelType(tensor, i)) ||
             isUndefDLT(merger.getDimLevelType(tensor, i)));
    }
  if (allDense)
    return true;
  // A tensor expression with a sparse output tensor that changes its values
  // but not its nonzero structure, an operation called "simply dynamic" in
  // [Bik96,Ch9], is also admissible without special codegen.
  if (merger.isSingleCondition(tensor, exp))
    return true;
  // Accept "truly dynamic" if the output tensor materializes uninitialized
  // into the computation and insertions occur in lexicographic index order.
  if (isMaterializing(lhs->get())) {
    unsigned nest = 0;
    for (unsigned i = 0; i < numLoops; i++) {
      if (linalg::isReductionIterator(iteratorTypes[topSort[i]]))
        break; // terminate at first reduction
      nest++;
    }
    // Determine admissible dynamic insertion situations:
    // (1) fully injective, since there are no reductions,
    // (2) admissible 1-d expansion in innermost dimension.
    if (nest >= op.getRank(lhs) - 1) {
      *sparseOut = lhs;
      outerParNest = nest;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (reductions).
//===----------------------------------------------------------------------===//

/// Maps operation to reduction.
static Reduction getReduction(Kind kind) {
  switch (kind) {
  case Kind::kAddF:
  case Kind::kAddC:
  case Kind::kAddI:
  case Kind::kSubF:
  case Kind::kSubC:
  case Kind::kSubI:
    return kSum;
  case Kind::kMulF:
  case Kind::kMulC:
  case Kind::kMulI:
    return kProduct;
  case Kind::kAndI:
    return kAnd;
  case Kind::kOrI:
    return kOr;
  case Kind::kXorI:
    return kXor;
  case Kind::kReduce:
    return kCustom;
  default:
    llvm_unreachable("unexpected reduction operator");
  }
}

/// Updates scalarized reduction value.
static void updateReduc(Merger &merger, CodeGen &codegen, Value reduc) {
  assert(codegen.redKind != kNoReduc);
  codegen.redVal = merger.exp(codegen.redExp).val = reduc;
}

/// Extracts identity from custom reduce.
static Value getCustomRedId(Operation *op) {
  return dyn_cast<sparse_tensor::ReduceOp>(op).getIdentity();
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (statements and expressions).
//===----------------------------------------------------------------------===//

/// Generates loop boundary statements (entering/exiting loops). The function
/// passes and updates the reduction value.
static Optional<Operation *> genLoopBoundary(
    CodeGen &codegen, Merger &merger,
    function_ref<Optional<Operation *>(MutableArrayRef<Value> reduc)>
        callback) {
  SmallVector<Value, 4> reduc;
  if (codegen.redVal)
    reduc.push_back(codegen.redVal);
  if (codegen.expValues)
    reduc.push_back(codegen.expCount);
  if (codegen.insChain)
    reduc.push_back(codegen.insChain);

  auto r = callback(reduc);

  // Callback should do in-place update on reduction value vector.
  unsigned i = 0;
  if (codegen.redVal)
    updateReduc(merger, codegen, reduc[i++]);
  if (codegen.expValues)
    codegen.expCount = reduc[i++];
  if (codegen.insChain)
    codegen.insChain = reduc[i];

  return r;
}

/// Local bufferization of all dense and sparse data structures.
static void genBuffers(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op) {
  Location loc = op.getLoc();
  assert(op.getNumOperands() == op.getNumDpsInputs() + 1);

  codegen.loopEmitter.initializeLoopEmit(
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
        OpOperand *lhs = op.getDpsInitOperand(0);
        // Two output tensors references should pointed to the same object.
        assert(lhs->get() == tensor);
        bool isInit = op.isInitTensor(lhs);
        // An output tensor can simply materialize from the buffer of the tensor
        // that appears in the outs() clause. For updates, this has the
        // advantage that only the nonzero value are involved in the
        // computation, keeping the operation O(nnz). In all other cases, we are
        // forced to zero out the buffer to enforce the assumption above, which
        // may negatively impact running complexity (viz. O(n^2 + nnz) vs.
        // O(nnz) for matrices).
        // TODO: use better analysis to avoid zeroing out the buffer?
        Value init = memref;
        if (!isInit) {
          Value zero = constantZero(builder, loc,
                                    getElementTypeOrSelf(tensor.getType()));
          builder.create<linalg::FillOp>(loc, ValueRange{zero},
                                         ValueRange{init});
        }
        return init;
      });
}

/// Generates an affine expression.
//
// TODO: generalize for sparse tensor subscripts
//
static Value genAffine(CodeGen &codegen, OpBuilder &builder, AffineExpr a,
                       Location loc) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    return codegen.getLoopIdxValue(idx); // universal dense index
  }
  case AffineExprKind::Add: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::AddIOp>(
        loc, genAffine(codegen, builder, binOp.getLHS(), loc),
        genAffine(codegen, builder, binOp.getRHS(), loc));
  }
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::MulIOp>(
        loc, genAffine(codegen, builder, binOp.getLHS(), loc),
        genAffine(codegen, builder, binOp.getRHS(), loc));
  }
  case AffineExprKind::Constant: {
    int64_t c = a.cast<AffineConstantExpr>().getValue();
    return constantIndex(builder, loc, c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

/// Generates index for load/store on sparse tensor.
static Value genIndex(CodeGen &codegen, linalg::GenericOp op, OpOperand *t) {
  auto map = op.getMatchingIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  AffineExpr a = map.getResult(toOrigDim(enc, map.getNumResults() - 1));
  assert(a.getKind() == AffineExprKind::DimId);
  unsigned idx = a.cast<AffineDimExpr>().getPosition();
  return codegen.getLoopIdxValue(idx);
}

/// Generates subscript for load/store on a dense or sparse tensor.
static Value genSubscript(CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, OpOperand *t,
                          SmallVector<Value, 4> &args) {
  unsigned tensor = t->getOperandNumber();
  auto map = op.getMatchingIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  unsigned rank = map.getNumResults();
  if (enc) {
    // Note that currently, all sparse subscripts are simple.
    // TODO: accept affine too?
    assert(map.getResult(toOrigDim(enc, rank - 1)).getKind() ==
           AffineExprKind::DimId);
    Value pidx = codegen.loopEmitter.getPidxs()[tensor].back();
    assert(pidx);
    args.push_back(pidx); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      AffineExpr a = map.getResult(d);
      args.push_back(genAffine(codegen, builder, a, op.getLoc()));
    }
  }
  return codegen.loopEmitter.getValBuffer()[tensor];
}

/// Generates insertion code to implement dynamic tensor load.
static Value genInsertionLoad(CodeGen &codegen, OpBuilder &builder,
                              linalg::GenericOp op, OpOperand *t) {
  Location loc = op.getLoc();
  // Direct lexicographic index order, tensor loads as zero.
  if (!codegen.expValues) {
    Type tp = getElementTypeOrSelf(t->get().getType());
    return constantZero(builder, loc, tp);
  }
  // Load from expanded access pattern.
  Value index = genIndex(codegen, op, t);
  return builder.create<memref::LoadOp>(loc, codegen.expValues, index);
}

/// Generates insertion code to implement dynamic tensor load for reduction.
static Value genInsertionLoadReduce(Merger &merger, CodeGen &codegen,
                                    OpBuilder &builder, linalg::GenericOp op,
                                    OpOperand *t) {
  Location loc = op.getLoc();
  Value identity = getCustomRedId(merger.exp(codegen.redCustom).op);
  // Direct lexicographic index order, tensor loads as identity.
  if (!codegen.expValues) {
    return identity;
  }
  // Load from expanded access pattern if filled, identity otherwise.
  Value index = genIndex(codegen, op, t);
  Value isFilled =
      builder.create<memref::LoadOp>(loc, codegen.expFilled, index);
  Value valAtIndex =
      builder.create<memref::LoadOp>(loc, codegen.expValues, index);
  return builder.create<arith::SelectOp>(loc, isFilled, valAtIndex, identity);
}

/// Generates insertion code to implement dynamic tensor store.
static void genInsertionStore(CodeGen &codegen, OpBuilder &builder,
                              linalg::GenericOp op, OpOperand *t, Value rhs) {
  Location loc = op.getLoc();
  // Direct insertion in lexicographic index order.
  if (!codegen.expValues) {
    unsigned rank = op.getRank(t);
    SmallVector<Value, 4> indices;
    for (unsigned i = 0; i < rank; i++) {
      assert(codegen.loopEmitter.getLoopIV(i));
      indices.push_back(codegen.loopEmitter.getLoopIV(i));
    }
    codegen.insChain =
        builder.create<InsertOp>(loc, rhs, codegen.insChain, indices);
    return;
  }
  // Generates insertion code along expanded access pattern.
  //   if (!expFilled[i]) then
  //     expFilled[i] = true
  //     expAdded[inserts++] = i
  //   endif
  //   values[i] = rhs
  Value index = genIndex(codegen, op, t);
  Value fval = constantI1(builder, loc, false);
  Value tval = constantI1(builder, loc, true);
  // If statement.
  Value filled = builder.create<memref::LoadOp>(loc, codegen.expFilled, index);
  Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             filled, fval);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), cond,
                                             /*else=*/true);
  // True branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<memref::StoreOp>(loc, tval, codegen.expFilled, index);
  builder.create<memref::StoreOp>(loc, index, codegen.expAdded,
                                  codegen.expCount);
  Value one = constantIndex(builder, loc, 1);
  Value add = builder.create<arith::AddIOp>(loc, codegen.expCount, one);
  builder.create<scf::YieldOp>(loc, add);
  // False branch.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, codegen.expCount);
  builder.setInsertionPointAfter(ifOp);
  // Value assignment.
  codegen.expCount = ifOp.getResult(0);
  builder.create<memref::StoreOp>(loc, rhs, codegen.expValues, index);
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = merger.exp(exp).val;
  if (val)
    return val;

  // Load during insertion.
  OpOperand &t = op->getOpOperand(merger.exp(exp).tensor);
  if (&t == codegen.sparseOut) {
    if (codegen.redCustom != -1u)
      return genInsertionLoadReduce(merger, codegen, builder, op, &t);
    return genInsertionLoad(codegen, builder, op, &t);
  }
  // Actual load.
  SmallVector<Value, 4> args;
  Value ptr = genSubscript(codegen, builder, op, &t, args);
  return builder.create<memref::LoadOp>(op.getLoc(), ptr, args);
}

/// Generates a store on a dense or sparse tensor.
static void genTensorStore(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned exp, Value rhs) {
  Location loc = op.getLoc();
  // Test if this is a scalarized reduction.
  if (codegen.redVal) {
    updateReduc(merger, codegen, rhs);
    return;
  }
  // Store during insertion.
  OpOperand *t = op.getDpsInitOperand(0);
  if (t == codegen.sparseOut) {
    if (!rhs) {
      // Only unary and binary are allowed to return uninitialized rhs
      // to indicate missing output.
      assert(merger.exp(exp).kind == kUnary || merger.exp(exp).kind == kBinary);
    } else if (merger.exp(exp).kind == kSelect) {
      // Select operation insertion.
      Value insChain = codegen.insChain;
      assert(insChain);
      SmallVector<Type, 1> types;
      types.push_back(codegen.insChain.getType());
      scf::IfOp ifOp =
          builder.create<scf::IfOp>(loc, types, rhs, /*else=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // Existing value was preserved to be used here.
      assert(merger.exp(exp).val);
      Value v0 = merger.exp(exp).val;
      genInsertionStore(codegen, builder, op, t, v0);
      merger.exp(exp).val = Value();
      // Yield modified insertion chain along true branch.
      builder.create<scf::YieldOp>(op.getLoc(), codegen.insChain);
      // Yield original insertion chain along false branch.
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<scf::YieldOp>(loc, insChain);
      // Done with if statement.
      codegen.insChain = ifOp->getResult(0);
      builder.setInsertionPointAfter(ifOp);
    } else {
      genInsertionStore(codegen, builder, op, t, rhs);
    }
    return;
  }
  // Actual store.
  SmallVector<Value, 4> args;
  Value ptr = genSubscript(codegen, builder, op, t, args);
  builder.create<memref::StoreOp>(loc, rhs, ptr, args);
}

/// Generates an invariant value.
inline static Value genInvariantValue(Merger &merger, CodeGen &codegen,
                                      OpBuilder &builder, unsigned exp) {
  return merger.exp(exp).val;
}

/// Generates an index value.
inline static Value genIndexValue(CodeGen &codegen, OpBuilder &builder,
                                  unsigned idx) {
  return codegen.getLoopIdxValue(idx);
}

/// Semi-ring branches are simply inlined by the sparse compiler. Prior
/// analysis has verified that all computations are "local" to the inlined
/// branch or otherwise invariantly defined outside the loop nest, with the
/// exception of index computations, which need to be relinked to actual
/// inlined cloned code.
static Value relinkBranch(CodeGen &codegen, RewriterBase &rewriter,
                          Block *block, Value e, unsigned ldx) {
  if (Operation *def = e.getDefiningOp()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return genIndexValue(codegen, rewriter, indexOp.getDim());
    if (def->getBlock() == block) {
      for (unsigned i = 0, n = def->getNumOperands(); i < n; i++)
        def->setOperand(
            i, relinkBranch(codegen, rewriter, block, def->getOperand(i), ldx));
    }
  }
  return e;
}

/// Recursively generates tensor expression.
static Value genExp(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, unsigned exp, unsigned ldx) {
  Location loc = op.getLoc();
  if (exp == -1u)
    return Value();
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, exp);
  if (merger.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(merger, codegen, rewriter, exp);
  if (merger.exp(exp).kind == Kind::kIndex)
    return genIndexValue(codegen, rewriter, merger.exp(exp).index);

  if (merger.exp(exp).kind == Kind::kReduce) {
    // Make custom reduction identity accessible for expanded access pattern.
    assert(codegen.redCustom == -1u);
    codegen.redCustom = exp;
  }

  Value v0 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e0, ldx);
  Value v1 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e1, ldx);
  Value ee = merger.buildExp(rewriter, loc, exp, v0, v1);
  if (ee && (merger.exp(exp).kind == Kind::kUnary ||
             merger.exp(exp).kind == Kind::kBinary ||
             merger.exp(exp).kind == Kind::kBinaryBranch ||
             merger.exp(exp).kind == Kind::kReduce ||
             merger.exp(exp).kind == Kind::kSelect))
    ee = relinkBranch(codegen, rewriter, ee.getParentBlock(), ee, ldx);

  if (merger.exp(exp).kind == kSelect) {
    assert(!merger.exp(exp).val);
    merger.exp(exp).val = v0; // Preserve value for later use.
  }

  if (merger.exp(exp).kind == Kind::kReduce) {
    assert(codegen.redCustom != -1u);
    codegen.redCustom = -1u;
  }

  return ee;
}

/// Determines if affine expression is invariant.
static bool isInvariantAffine(const CodeGen &codegen, AffineExpr a,
                              unsigned ldx, bool &atLevel) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (idx == ldx)
      atLevel = true;
    return codegen.getLoopIdxValue(idx) != nullptr; // no longer in play?
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return isInvariantAffine(codegen, binOp.getLHS(), ldx, atLevel) &&
           isInvariantAffine(codegen, binOp.getRHS(), ldx, atLevel);
  }
  default:
    return true;
  }
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, unsigned exp, unsigned ldx,
                          bool atStart, unsigned last = -1u) {
  if (exp == -1u)
    return;
  if (merger.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    OpOperand &t = op->getOpOperand(merger.exp(exp).tensor);
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(toOrigDim(enc, d));
      if (!isInvariantAffine(codegen, a, ldx, atLevel))
        return; // still in play
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    if (!atLevel)
      return;
    OpOperand *lhs = op.getDpsInitOperand(0);
    if (lhs == &t) {
      // Start or end a scalarized reduction
      if (atStart) {
        Kind kind = merger.exp(last).kind;
        Value load = kind == Kind::kReduce
                         ? getCustomRedId(merger.exp(last).op)
                         : genTensorLoad(merger, codegen, builder, op, exp);
        codegen.redKind = getReduction(kind);
        codegen.redExp = exp;
        updateReduc(merger, codegen, load);
      } else {
        Value redVal = codegen.redVal;
        updateReduc(merger, codegen, Value());
        codegen.redExp = -1u;
        codegen.redKind = kNoReduc;
        genTensorStore(merger, codegen, builder, op, exp, redVal);
      }
    } else {
      // Start or end loop invariant hoisting of a tensor load.
      merger.exp(exp).val =
          atStart ? genTensorLoad(merger, codegen, builder, op, exp) : Value();
    }
  } else if (merger.exp(exp).kind != Kind::kInvariant &&
             merger.exp(exp).kind != Kind::kIndex) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    unsigned e0 = merger.exp(exp).children.e0;
    unsigned e1 = merger.exp(exp).children.e1;
    genInvariants(merger, codegen, builder, op, e0, ldx, atStart, exp);
    genInvariants(merger, codegen, builder, op, e1, ldx, atStart, exp);
  }
}

/// Generates an expanded access pattern in innermost dimension.
static void genExpansion(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, unsigned at, bool atStart) {
  OpOperand *lhs = codegen.sparseOut;
  if (!lhs || codegen.outerParNest != op.getRank(lhs) - 1 ||
      at != codegen.outerParNest)
    return; // not needed at this level
  assert(codegen.redVal == nullptr);
  // Generate start or end of an expanded access pattern. Note that because
  // an expension does not rely on the ongoing contents of the sparse storage
  // scheme, we can use the original tensor as incoming SSA value (which
  // simplifies codegen a bit). If expansion on the actual contents is ever
  // needed, we will need to use the SSA value in the insertion chain instead.
  Value tensor = lhs->get();
  Location loc = op.getLoc();
  if (atStart) {
    auto dynShape = {ShapedType::kDynamicSize};
    Type etp = tensor.getType().cast<ShapedType>().getElementType();
    Type t1 = MemRefType::get(dynShape, etp);
    Type t2 = MemRefType::get(dynShape, builder.getI1Type());
    Type t3 = MemRefType::get(dynShape, builder.getIndexType());
    Type t4 = builder.getIndexType();
    auto res =
        builder.create<ExpandOp>(loc, TypeRange({t1, t2, t3, t4}), tensor);
    assert(res.getNumResults() == 4);
    assert(!codegen.expValues);
    codegen.expValues = res.getResult(0);
    codegen.expFilled = res.getResult(1);
    codegen.expAdded = res.getResult(2);
    codegen.expCount = res.getResult(3);
  } else {
    assert(codegen.expValues);
    SmallVector<Value, 4> indices;
    for (unsigned i = 0; i < at; i++) {
      assert(codegen.loopEmitter.getLoopIV(i));
      indices.push_back(codegen.loopEmitter.getLoopIV(i));
    }
    codegen.insChain = builder.create<CompressOp>(
        loc, codegen.expValues, codegen.expFilled, codegen.expAdded,
        codegen.expCount, codegen.insChain, indices);
    codegen.expValues = codegen.expFilled = codegen.expAdded =
        codegen.expCount = Value();
  }
}

/// Returns parallelization strategy. Any implicit loop in the Linalg
/// operation that is marked "parallel" is a candidate. Whether it is actually
/// converted to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodeGen &codegen, bool isOuter, bool isSparse) {
  // Reject parallelization of sparse output.
  if (codegen.sparseOut)
    return false;
  // Parallel loops on tensor expansion can cause data races.
  if (codegen.expCount)
    return false;
  // Inspect strategy.
  switch (codegen.options.parallelizationStrategy) {
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

/// Generates a for-loop on a single index.
static Operation *genFor(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, bool isOuter, bool isInner,
                         unsigned idx, size_t tid, size_t dim,
                         ArrayRef<size_t> extraTids,
                         ArrayRef<size_t> extraDims) {
  Location loc = op.getLoc();
  auto iteratorTypes = op.getIteratorTypesArray();
  bool isSparse = isCompressedDLT(merger.getDimLevelType(tid, idx)) ||
                  isSingletonDLT(merger.getDimLevelType(tid, idx));
  bool isParallel = isParallelFor(codegen, isOuter, isSparse);

  Operation *loop =
      genLoopBoundary(codegen, merger, [&](MutableArrayRef<Value> reduc) {
        return codegen.loopEmitter.enterLoopOverTensorAtDim(
            builder, loc, tid, dim, reduc, isParallel, extraTids, extraDims);
      }).value();
  assert(loop);
  return loop;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned idx, bool needsUniv,
                           ArrayRef<size_t> condTids, ArrayRef<size_t> condDims,
                           ArrayRef<size_t> extraTids,
                           ArrayRef<size_t> extraDims) {

  Operation *loop =
      genLoopBoundary(codegen, merger, [&](MutableArrayRef<Value> reduc) {
        // Construct the while-loop with a parameter for each index.
        return codegen.loopEmitter.enterCoIterationOverTensorsAtDims(
            builder, op.getLoc(), condTids, condDims, needsUniv, reduc,
            extraTids, extraDims);
      }).value();
  assert(loop);
  return loop;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, unsigned at, bool needsUniv,
                          ArrayRef<size_t> condTids, ArrayRef<size_t> condDims,
                          ArrayRef<size_t> extraTids,
                          ArrayRef<size_t> extraDims) {
  assert(condTids.size() == condDims.size());
  assert(extraTids.size() == extraDims.size());
  unsigned idx = codegen.topSort[at];
  if (condTids.size() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == codegen.topSort.size() - 1;
    return genFor(merger, codegen, builder, op, isOuter, isInner, idx,
                  condTids.front(), condDims.front(), extraTids, extraDims);
  }
  return genWhile(merger, codegen, builder, op, idx, needsUniv, condTids,
                  condDims, extraTids, extraDims);
}

/// Generates the induction structure for a while-loop.
static void finalizeWhileOp(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            unsigned idx, bool needsUniv, BitVector &induction,
                            scf::WhileOp whileOp) {
  Location loc = op.getLoc();
  // Finalize each else branch of all if statements.
  if (codegen.redVal || codegen.expValues || codegen.insChain) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               builder.getInsertionBlock()->getParentOp())) {
      unsigned y = 0;
      SmallVector<Value, 4> yields;
      if (codegen.redVal) {
        yields.push_back(codegen.redVal);
        updateReduc(merger, codegen, ifOp.getResult(y++));
      }
      if (codegen.expValues) {
        yields.push_back(codegen.expCount);
        codegen.expCount = ifOp->getResult(y++);
      }
      if (codegen.insChain) {
        yields.push_back(codegen.insChain);
        codegen.insChain = ifOp->getResult(y++);
      }
      assert(y == yields.size());
      builder.create<scf::YieldOp>(loc, yields);
      builder.setInsertionPointAfter(ifOp);
    }
  }
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned idx,
                       BitVector &conditions) {
  Location loc = op.getLoc();
  SmallVector<Type, 4> types;
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (!conditions[b])
      continue;
    unsigned tensor = merger.tensor(b);
    assert(idx == merger.index(b));
    Value clause;
    if (isCompressedDLT(merger.getDimLevelType(b)) ||
        isSingletonDLT(merger.getDimLevelType(b))) {
      auto dim = merger.getDimNum(tensor, idx).value();
      Value op1 = codegen.loopEmitter.getCoord()[tensor][dim];
      Value op2 = codegen.getLoopIdxValue(idx);
      clause = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, op1,
                                             op2);
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
      clause = constantI1(builder, loc, true);
    }
    cond = cond ? builder.create<arith::AndIOp>(loc, cond, clause) : clause;
  }
  if (codegen.redVal)
    types.push_back(codegen.redVal.getType());
  if (codegen.expValues)
    types.push_back(builder.getIndexType());
  if (codegen.insChain)
    types.push_back(codegen.insChain.getType());
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
static void endIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                  linalg::GenericOp op, scf::IfOp ifOp, Operation *loop,
                  Value redInput, Value cntInput, Value insInput) {
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, redInput);
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = cntInput;
  }
  if (codegen.insChain) {
    operands.push_back(codegen.insChain);
    codegen.insChain = insInput;
  }
  if (!operands.empty())
    builder.create<scf::YieldOp>(op.getLoc(), operands);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (loop sequence).
//===----------------------------------------------------------------------===//

/// Starts a loop sequence at given level. Returns true if
/// the universal loop index must be maintained at this level.
static bool startLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, unsigned exp, unsigned at,
                         unsigned idx, unsigned ldx, unsigned lts) {
  assert(!codegen.getLoopIdxValue(idx));
  // Emit invariants at this loop sequence level.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, builder, op, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  unsigned l0 = merger.set(lts)[0];
  bool needsUniv = false;

  SmallVector<size_t> tids;
  SmallVector<size_t> dims;
  merger.foreachTidDimPairInBits(
      merger.lat(l0).bits,
      [&](unsigned b, unsigned tid, Optional<unsigned> dim, DimLevelType dlt) {
        assert(merger.index(b) == idx);
        if (isDenseDLT(dlt) || isUndefDLT(dlt)) {
          needsUniv = true;
        } else {
          // sparse/singleton dim levels.
          tids.push_back(tid);
          dims.push_back(dim.value());
        }
      });

  codegen.loopEmitter.enterNewLoopSeq(builder, op.getLoc(), tids, dims);

  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  if (needsUniv) {
    unsigned lsize = merger.set(lts).size();
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = merger.set(lts)[i];
      if (!merger.hasAnySparse(merger.lat(li).simple))
        return true;
    }
  }
  return false;
}

static void translateBitsToTidDimPairs(Merger &merger, CodeGen &codegen,
                                       unsigned li, unsigned idx,
                                       SmallVectorImpl<size_t> &condTids,
                                       SmallVectorImpl<size_t> &condDims,
                                       SmallVectorImpl<size_t> &extraTids,
                                       SmallVectorImpl<size_t> &extraDims) {
  const BitVector &all = merger.lat(li).bits;
  const BitVector &simple = merger.lat(li).simple;

  // Converts bits to array + dim pair
  merger.foreachTidDimPairInBits(all, [&, idx](unsigned b, unsigned tid,
                                               Optional<unsigned> dim,
                                               DimLevelType dlt) {
    if (simple.test(b)) {
      if (isUndefDLT(dlt)) {
        // An undefined dlt in the lattices, we probably mean to iterate based
        // on the dim of output tensor.
        // E.g., this could be a synthetic tensor (for invariants and sparse
        // output tensor).
        // out[i][j] = invariant; or a broadcast
        // out[i][j] = in[i] (j is undef for input)
        tid = merger.getOutTensorID();
        dim = merger.getDimNum(tid, idx);
        // Skips invalid dim (e.g., when this is a zero ranked tensor).
        if (!dim)
          return;
      }
      condTids.push_back(tid);
      condDims.push_back(dim.value());
    } else if (isDenseDLT(dlt)) {
      // TODO: get rid of extraTids and extraDims.
      extraTids.push_back(tid);
      extraDims.push_back(dim.value());
    }
  });

  if (isDenseDLT(merger.getDimLevelType(merger.getOutTensorID(), idx))) {
    // Note that we generate dense indices of the output tensor
    // unconditionally, since they may not appear in the lattice, but may be
    // needed for linearized codegen.
    // Only dense dimensions should be optimized from conditions.
    auto dim = merger.getDimNum(merger.getOutTensorID(), idx).value();
    extraTids.push_back(merger.getOutTensorID());
    extraDims.push_back(dim);
  }
}

/// Starts a single loop in current sequence.
static Operation *startLoop(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            unsigned at, unsigned li, bool needsUniv) {
  // The set of tensors + dims to generate loops on
  SmallVector<size_t, 4> condTids, condDims;
  // The set of (dense) tensors that is optimized from condition, yet still
  // need extra locals to iterate on them.
  SmallVector<size_t, 4> extraTids, extraDims;

  translateBitsToTidDimPairs(merger, codegen, li, codegen.topSort[at], condTids,
                             condDims, extraTids, extraDims);
  // Emit the for/while-loop control.
  Operation *loop = genLoop(merger, codegen, builder, op, at, needsUniv,
                            condTids, condDims, extraTids, extraDims);
  return loop;
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, Operation *loop, unsigned idx,
                    unsigned li, bool needsUniv) {
  // End a while-loop.
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    finalizeWhileOp(merger, codegen, rewriter, op, idx, needsUniv,
                    merger.lat(li).bits, whileOp);
  } else {
    needsUniv = false;
  }

  genLoopBoundary(codegen, merger, [&](MutableArrayRef<Value> reduc) {
    codegen.loopEmitter.exitCurrentLoop(rewriter, op.getLoc(), reduc);
    return llvm::None;
  });

  return needsUniv;
}

/// Ends a loop sequence at given level.
static void endLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned exp, unsigned at,
                       unsigned idx, unsigned ldx) {
  assert(codegen.getLoopIdxValue(idx) == nullptr);
  codegen.loopEmitter.exitCurrentLoopSeq();
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, builder, op, at, /*atStart=*/false);
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == codegen.topSort.size()) {
    unsigned ldx = codegen.topSort[at - 1];
    Value rhs = genExp(merger, codegen, rewriter, op, exp, ldx);
    genTensorStore(merger, codegen, rewriter, op, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  unsigned idx = codegen.topSort[at];
  unsigned ldx = at == 0 ? -1u : codegen.topSort[at - 1];
  unsigned lts = merger.optimizeSet(merger.buildLattices(exp, idx));

  // TODO: sort
  // TODO: dedup

  // Start a loop sequence.
  bool needsUniv =
      startLoopSeq(merger, codegen, rewriter, op, exp, at, idx, ldx, lts);

  // Emit a loop for every lattice point L0 >= Li in this loop sequence.
  unsigned lsize = merger.set(lts).size();
  for (unsigned i = 0; i < lsize; i++) {
    // Start a loop.
    unsigned li = merger.set(lts)[i];
    Operation *loop =
        startLoop(merger, codegen, rewriter, op, at, li, needsUniv);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    Value redInput = codegen.redVal;
    Value cntInput = codegen.expCount;
    Value insInput = codegen.insChain;
    bool isWhile = dyn_cast<scf::WhileOp>(loop) != nullptr;
    for (unsigned j = 0; j < lsize; j++) {
      unsigned lj = merger.set(lts)[j];
      unsigned ej = merger.lat(lj).exp;
      if (li == lj || merger.latGT(li, lj)) {
        // Recurse into body of each branch.
        if (isWhile) {
          scf::IfOp ifOp =
              genIf(merger, codegen, rewriter, op, idx, merger.lat(lj).simple);
          genStmt(merger, codegen, rewriter, op, ej, at + 1);
          endIf(merger, codegen, rewriter, op, ifOp, loop, redInput, cntInput,
                insInput);
        } else {
          genStmt(merger, codegen, rewriter, op, ej, at + 1);
        }
      }
    }

    // End a loop.
    needsUniv =
        endLoop(merger, codegen, rewriter, op, loop, idx, li, needsUniv);
  }

  // End a loop sequence.
  endLoopSeq(merger, codegen, rewriter, op, exp, at, idx, ldx);
}

/// Converts the result computed by the sparse kernel into the required form.
static void genResult(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                      linalg::GenericOp op) {
  OpOperand *lhs = op.getDpsInitOperand(0);
  Value tensor = lhs->get();
  Type resType = tensor.getType();
  if (getSparseTensorEncoding(resType)) {
    // The sparse tensor rematerializes from the original sparse tensor's
    // underlying sparse storage format. For an insertion chain, the
    // tensor materializes from the chain with 'hasInserts' enabled.
    bool hasInserts = codegen.sparseOut == lhs;
    if (hasInserts)
      tensor = codegen.insChain;
    rewriter.replaceOpWithNewOp<LoadOp>(op, resType, tensor, hasInserts);
  } else {
    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
    Value val = codegen.loopEmitter.getValBuffer().back(); // value array
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, val);
  }
}

//===----------------------------------------------------------------------===//
// Sparse compiler rewriting methods.
//===----------------------------------------------------------------------===//

namespace {

/// Sparse rewriting rule for generic Lingalg operation.
struct GenericOpSparsifier : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparsifier(MLIRContext *context, SparsificationOptions o)
      : OpRewritePattern<linalg::GenericOp>(context), options(o) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Detects sparse annotations and translate the per-dimension sparsity
    // information for all tensors to loop indices in the kernel.
    if (op.getNumDpsInits() != 1)
      return failure();
    unsigned numTensors = op->getNumOperands();
    unsigned numLoops = op.getNumLoops();
    Merger merger(numTensors, numLoops);
    if (!findSparseAnnotations(merger, op))
      return failure();

    // Builds the tensor expression for the Linalg operation in SSA form.
    Optional<unsigned> optExp = merger.buildTensorExpFromLinalg(op);
    if (!optExp.has_value())
      return failure();

    unsigned exp = optExp.value();
    OpOperand *sparseOut = nullptr;
    unsigned outerParNest = 0;
    // Computes a topologically sorted iteration graph to ensure tensors
    // are visited in natural index order. Gradually relaxes the considered
    // constraints until an acyclic iteration graph results, such that sparse
    // code generation can proceed. As a last resort, an attempt is made
    // to resolve cycles by inserting a conversion.
    std::vector<unsigned> topSort;
    // Whether the current GenericOp is admissible.
    bool isAdmissible = false;
    bool hasCycle = true;
    // An const list of all masks that we used for interation graph
    // computation. Must be ordered from strict -> loose.
    const auto allMask = {SortMask::kIncludeAll, SortMask::kIncludeUndef,
                          SortMask::kIncludeDense, SortMask::kSparseOnly};
    for (auto mask : allMask)
      if (computeIterationGraph(merger, op, topSort, mask)) {
        hasCycle = false;
        if (isAdmissibleTensorExp(merger, op, topSort, exp, &sparseOut,
                                  outerParNest)) {
          isAdmissible = true;
          break;
        }
        // else try a set of less strict constraints.
      }

    if (hasCycle)
      // Give it one last shot to resolve the cycle.
      return resolveCycle(merger, rewriter, op);
    if (!isAdmissible)
      // Inadmissible expression, reject.
      return failure();

    merger.setHasSparseOut(sparseOut != nullptr);

    SmallVector<Value, 4> tensors;
    for (OpOperand &t : op->getOpOperands())
      tensors.push_back(t.get());

    // Recursively generates code if admissible.
    CodeGen codegen(options, tensors, numTensors, numLoops, sparseOut,
                    outerParNest, topSort);
    genBuffers(merger, codegen, rewriter, op);
    genStmt(merger, codegen, rewriter, op, exp, 0);
    genResult(merger, codegen, rewriter, op);
    return success();
  }

private:
  // Last resort cycle resolution.
  LogicalResult resolveCycle(Merger &merger, PatternRewriter &rewriter,
                             linalg::GenericOp op) const {
    // Compute topological sort while leaving out every
    // sparse input tensor in succession until an acylic
    // iteration graph results.
    std::vector<unsigned> topSort;
    for (OpOperand *t : op.getDpsInputOperands()) {
      unsigned tensor = t->getOperandNumber();
      Value tval = t->get();
      auto srcEnc = getSparseTensorEncoding(tval.getType());
      if (!srcEnc ||
          !computeIterationGraph(merger, op, topSort, SortMask::kSparseOnly, t))
        continue;
      // Found an input tensor that resolves the cycle by inserting a
      // conversion into a sparse tensor that adheres to the iteration
      // graph order. Also releases the temporary sparse tensor.
      //
      // TODO: investigate fusing the conversion with computation,
      //       especially if it is a direct yield!
      //
      auto srcTp = tval.getType().cast<RankedTensorType>();
      auto dstEnc = SparseTensorEncodingAttr::get(
          op->getContext(), srcEnc.getDimLevelType(),
          permute(getContext(), op.getMatchingIndexingMap(t),
                  topSort), // new order
          srcEnc.getHigherOrdering(), srcEnc.getPointerBitWidth(),
          srcEnc.getIndexBitWidth());
      auto dstTp = RankedTensorType::get(srcTp.getShape(),
                                         srcTp.getElementType(), dstEnc);
      auto convert = rewriter.create<ConvertOp>(tval.getLoc(), dstTp, tval);
      op->setOperand(tensor, convert);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<bufferization::DeallocTensorOp>(tval.getLoc(), convert);
      return success();
    }
    // Cannot be resolved with a single conversion.
    // TODO: convert more than one?
    return failure();
  }

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
