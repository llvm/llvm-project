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
  CodeGen(SparsificationOptions o, unsigned numTensors, unsigned numLoops,
          OpOperand *op, unsigned nest, std::vector<unsigned> &ts)
      : options(o), loops(numLoops), sizes(numLoops), buffers(numTensors),
        pointers(numTensors, std::vector<Value>(numLoops)),
        indices(numTensors, std::vector<Value>(numLoops)),
        highs(numTensors, std::vector<Value>(numLoops)),
        pidxs(numTensors, std::vector<Value>(numLoops)),
        idxs(numTensors, std::vector<Value>(numLoops)), sparseOut(op),
        outerParNest(nest), topSort(ts) {}
  /// Sparsification options.
  SparsificationOptions options;
  /// Universal dense indices and upper bounds (by index). The loops array
  /// is updated with the value of the universal dense index in the current
  /// loop. The sizes array is set once with the inferred dimension sizes.
  std::vector<Value> loops;
  std::vector<Value> sizes;
  /// Buffers for storing dense and sparse numerical values (by tensor).
  /// This array is set once during bufferization of all tensors.
  std::vector<Value> buffers;
  /// Sparse storage schemes (1-D): pointers and indices (by tensor and index).
  /// This array is set once during bufferization of all sparse tensors.
  std::vector<std::vector<Value>> pointers;
  std::vector<std::vector<Value>> indices;
  /// Sparse iteration information (by tensor and index). These arrays
  /// are updated to remain current within the current loop.
  std::vector<std::vector<Value>> highs;
  std::vector<std::vector<Value>> pidxs;
  std::vector<std::vector<Value>> idxs;
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
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;
  // Topsort (reference should remain in scope).
  std::vector<unsigned> &topSort;
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
static bool findAffine(Merger &merger, unsigned tensor, AffineExpr a,
                       DimLevelType dim) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (!isUndefDLT(merger.getDimLevelType(tensor, idx)))
      return false; // used more than once
    merger.setDimLevelType(tensor, idx, dim);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    if (!isDenseDLT(dim))
      return false; // compound only in dense dim
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return findAffine(merger, tensor, binOp.getLHS(), dim) &&
           findAffine(merger, tensor, binOp.getRHS(), dim);
  }
  case AffineExprKind::Constant:
    return isDenseDLT(dim); // const only in dense dim
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
      if (!findAffine(merger, tensor, a, getDimLevelType(enc, d)))
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
  OpOperand *lhs = op.getOutputOperand(0);
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

/// Generates buffer for the output tensor. Note that all sparse kernels
/// assume that when all elements are written to (viz. x(i) = y(i) * z(i)),
/// the output buffer is already initialized to all zeroes and only nonzeroes
/// values are computed and written out. For updates (viz. x(i) += y(i) * z(i)),
/// only nonzeroes values are used for the updates and no assumption on the
/// original contents of the output buffer is necessary.
static Value genOutputBuffer(CodeGen &codegen, OpBuilder &builder,
                             linalg::GenericOp op, MemRefType denseTp,
                             ArrayRef<Value> args) {
  Location loc = op.getLoc();
  OpOperand *lhs = op.getOutputOperand(0);
  Value tensor = lhs->get();
  bool isInit = op.isInitTensor(lhs);
  // An output tensor can simply materialize from the buffer of the tensor that
  // appears in the outs() clause. For updates, this has the advantage that only
  // the nonzero value are involved in the computation, keeping the operation
  // O(nnz). In all other cases, we are forced to zero out the buffer to enforce
  // the assumption above, which may negatively impact running complexity
  // (viz. O(n^2 + nnz) vs. O(nnz) for matrices).
  // TODO: use better analysis to avoid zeroing out the buffer?
  Value init = builder.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
  if (!isInit) {
    Value zero = constantZero(builder, loc, denseTp.getElementType());
    builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init});
  }
  return init;
}

/// Local bufferization of all dense and sparse data structures.
static void genBuffers(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op) {
  Location loc = op.getLoc();
  assert(op->getNumOperands() == op.getNumInputs() + 1);
  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and obtain dense or sparse buffer(s).
  auto dynShape = {ShapedType::kDynamicSize};
  SmallVector<Value, 4> args;
  for (OpOperand &t : op->getOpOperands()) {
    unsigned tensor = t.getOperandNumber();
    auto shape = op.getShape(&t);
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    // Scan all dimensions of current tensor.
    args.clear();
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(toOrigDim(enc, d));
      if (a.getKind() != AffineExprKind::DimId)
        continue; // compound
      unsigned idx = a.cast<AffineDimExpr>().getPosition();
      // Handle the different storage schemes.
      if (isCompressedDLT(merger.getDimLevelType(tensor, idx))) {
        // Compressed dimension, fetch pointer and indices.
        auto ptrTp =
            MemRefType::get(dynShape, getPointerOverheadType(builder, enc));
        auto indTp =
            MemRefType::get(dynShape, getIndexOverheadType(builder, enc));
        auto dim = builder.getIndexAttr(d);
        codegen.pointers[tensor][idx] =
            builder.create<ToPointersOp>(loc, ptrTp, t.get(), dim);
        codegen.indices[tensor][idx] =
            builder.create<ToIndicesOp>(loc, indTp, t.get(), dim);
      } else if (isSingletonDLT(merger.getDimLevelType(tensor, idx))) {
        // Singleton dimension, fetch indices.
        auto indTp =
            MemRefType::get(dynShape, getIndexOverheadType(builder, enc));
        auto dim = builder.getIndexAttr(d);
        codegen.indices[tensor][idx] =
            builder.create<ToIndicesOp>(loc, indTp, t.get(), dim);
      } else {
        // Dense dimension, nothing to fetch.
        assert(isDenseDLT(merger.getDimLevelType(tensor, idx)));
      }
      // Find upper bound in current dimension.
      unsigned p = toOrigDim(enc, d);
      Value up = linalg::createOrFoldDimOp(builder, loc, t.get(), p);
      if (ShapedType::isDynamic(shape[p]))
        args.push_back(up);
      assert(codegen.highs[tensor][idx] == nullptr);
      codegen.sizes[idx] = codegen.highs[tensor][idx] = up;
    }
    // Perform the required bufferization. Dense inputs materialize
    // from the input tensors. Dense outputs need special handling.
    // Sparse inputs use sparse primitives to obtain the values.
    Type elementType = getElementTypeOrSelf(t.get().getType());
    if (!enc) {
      // Non-annotated dense tensors.
      auto denseTp = MemRefType::get(shape, elementType);
      if (tensor < op.getNumInputs())
        codegen.buffers[tensor] =
            builder.create<bufferization::ToMemrefOp>(loc, denseTp, t.get());
      else
        codegen.buffers[tensor] =
            genOutputBuffer(codegen, builder, op, denseTp, args);
    } else if (&t != codegen.sparseOut) {
      // Annotated sparse tensors (not involved in output).
      auto sparseTp = MemRefType::get(dynShape, elementType);
      codegen.buffers[tensor] =
          builder.create<ToValuesOp>(loc, sparseTp, t.get());
    }
  }
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
    return codegen.loops[idx]; // universal dense index
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
  return codegen.loops[idx];
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
    AffineExpr a = map.getResult(toOrigDim(enc, rank - 1));
    assert(a.getKind() == AffineExprKind::DimId);
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    assert(codegen.pidxs[tensor][idx] != nullptr);
    args.push_back(codegen.pidxs[tensor][idx]); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      AffineExpr a = map.getResult(d);
      args.push_back(genAffine(codegen, builder, a, op.getLoc()));
    }
  }
  return codegen.buffers[tensor];
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
      assert(codegen.loops[codegen.topSort[i]]);
      indices.push_back(codegen.loops[codegen.topSort[i]]);
    }
    builder.create<InsertOp>(loc, rhs, t->get(), indices);
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
  OpOperand *t = op.getOutputOperand(0);
  if (t == codegen.sparseOut) {
    if (!rhs) {
      // Only unary and binary are allowed to return uninitialized rhs
      // to indicate missing output.
      assert(merger.exp(exp).kind == kUnary || merger.exp(exp).kind == kBinary);
    } else if (merger.exp(exp).kind == kSelect) {
      scf::IfOp ifOp = builder.create<scf::IfOp>(loc, rhs);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // Existing value was preserved to be used here.
      assert(merger.exp(exp).val);
      Value v0 = merger.exp(exp).val;
      genInsertionStore(codegen, builder, op, t, v0);
      merger.exp(exp).val = Value();
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

/// Generates a pointer/index load from the sparse storage scheme. Narrower
/// data types need to be zero extended before casting the value into the
/// index type used for looping and indexing.
static Value genLoad(CodeGen &codegen, OpBuilder &builder, Location loc,
                     Value ptr, Value s) {
  // Simply zero extends narrower indices into 64-bit values before casting to
  // index without a performance penalty.
  Value load = builder.create<memref::LoadOp>(loc, ptr, s);
  if (!load.getType().isa<IndexType>()) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), load);
    load =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), load);
  }
  return load;
}

/// Generates an invariant value.
static Value genInvariantValue(Merger &merger, CodeGen &codegen,
                               OpBuilder &builder, unsigned exp) {
  Value val = merger.exp(exp).val;
  return val;
}

/// Generates an address computation "sz * p + i".
static Value genAddress(CodeGen &codegen, OpBuilder &builder, Location loc,
                        Value size, Value p, Value i) {
  Value mul = builder.create<arith::MulIOp>(loc, size, p);
  return builder.create<arith::AddIOp>(loc, mul, i);
}

/// Generates an index value.
static Value genIndexValue(CodeGen &codegen, OpBuilder &builder, unsigned idx,
                           unsigned ldx) {
  Value ival = codegen.loops[idx];
  return ival;
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
      return genIndexValue(codegen, rewriter, indexOp.getDim(), ldx);
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
    return genIndexValue(codegen, rewriter, merger.exp(exp).index, ldx);

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
    return codegen.loops[idx] != nullptr; // no longer in play?
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
    OpOperand *lhs = op.getOutputOperand(0);
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
  // Generate start or end of an expanded access pattern.
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
      assert(codegen.loops[codegen.topSort[i]]);
      indices.push_back(codegen.loops[codegen.topSort[i]]);
    }
    builder.create<CompressOp>(loc, codegen.expValues, codegen.expFilled,
                               codegen.expAdded, codegen.expCount, tensor,
                               indices);
    codegen.expValues = codegen.expFilled = codegen.expAdded =
        codegen.expCount = Value();
  }
}

/// Generates initialization code for the subsequent loop sequence at
/// current index level. Returns true if the loop sequence needs to
/// maintain the universal index.
static bool genInit(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                    linalg::GenericOp op, unsigned at, BitVector &inits) {
  std::vector<unsigned> &topSort(codegen.topSort);
  bool needsUniv = false;
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse positions.
  for (unsigned b = 0, be = inits.size(); b < be; b++) {
    if (!inits[b])
      continue;
    unsigned tensor = merger.tensor(b);
    assert(idx == merger.index(b));
    if (isCompressedDLT(merger.getDimLevelType(b))) {
      // Initialize sparse index that will implement the iteration:
      //   for pidx_idx = pointers(pidx_idx-1), pointers(1+pidx_idx-1)
      unsigned pat = at;
      for (; pat != 0; pat--) {
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      }
      Value ptr = codegen.pointers[tensor][idx];
      Value one = constantIndex(builder, loc, 1);
      Value p0 = (pat == 0) ? constantIndex(builder, loc, 0)
                            : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = genLoad(codegen, builder, loc, ptr, p0);
      Value p1 = builder.create<arith::AddIOp>(loc, p0, one);
      codegen.highs[tensor][idx] = genLoad(codegen, builder, loc, ptr, p1);
    } else if (isSingletonDLT(merger.getDimLevelType(b))) {
      // Initialize sparse index that will implement the "iteration":
      //   for pidx_idx = pidx_idx-1, 1+pidx_idx-1
      // We rely on subsequent loop unrolling to get rid of the loop
      // if it is not involved in co-iteration with anything else.
      unsigned pat = at;
      for (; pat != 0; pat--) {
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      }
      Value one = constantIndex(builder, loc, 1);
      Value p0 = (pat == 0) ? constantIndex(builder, loc, 0)
                            : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = p0;
      codegen.highs[tensor][idx] = builder.create<arith::AddIOp>(loc, p0, one);
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
      // Dense index still in play.
      needsUniv = true;
    }
  }

  // Initialize the universal dense index.
  codegen.loops[idx] = constantIndex(builder, loc, 0);
  return needsUniv;
}

/// Returns parallelization strategy. Any implicit loop in the Linalg operation
/// that is marked "parallel" is a candidate. Whether it is actually converted
/// to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodeGen &codegen, bool isOuter, bool isReduction,
                          bool isSparse) {
  // Reject parallelization of sparse output.
  if (codegen.sparseOut)
    return false;
  // Inspect strategy.
  switch (codegen.options.parallelizationStrategy) {
  case SparseParallelizationStrategy::kNone:
    return false;
  case SparseParallelizationStrategy::kDenseOuterLoop:
    return isOuter && !isSparse && !isReduction;
  case SparseParallelizationStrategy::kAnyStorageOuterLoop:
    return isOuter && !isReduction;
  case SparseParallelizationStrategy::kDenseAnyLoop:
    return !isSparse && !isReduction;
  case SparseParallelizationStrategy::kAnyStorageAnyLoop:
    return !isReduction;
  }
  llvm_unreachable("unexpected parallelization strategy");
}

/// Generates a for-loop on a single index.
static Operation *genFor(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, bool isOuter, bool isInner,
                         unsigned idx, BitVector &indices) {
  unsigned fb = indices.find_first();
  unsigned tensor = merger.tensor(fb);
  assert(idx == merger.index(fb));
  auto iteratorTypes = op.getIteratorTypesArray();
  bool isReduction = linalg::isReductionIterator(iteratorTypes[idx]);
  bool isSparse = isCompressedDLT(merger.getDimLevelType(fb)) ||
                  isSingletonDLT(merger.getDimLevelType(fb));
  bool isParallel = isParallelFor(codegen, isOuter, isReduction, isSparse);

  // Loop bounds and increment.
  Location loc = op.getLoc();
  Value lo = isSparse ? codegen.pidxs[tensor][idx] : codegen.loops[idx];
  Value hi = isSparse ? codegen.highs[tensor][idx] : codegen.sizes[idx];
  Value step = constantIndex(builder, loc, 1);

  // Emit a parallel loop.
  if (isParallel) {
    scf::ParallelOp parOp = builder.create<scf::ParallelOp>(loc, lo, hi, step);
    if (isSparse)
      codegen.pidxs[tensor][idx] = parOp.getInductionVars()[0];
    else
      codegen.loops[idx] = parOp.getInductionVars()[0];
    builder.setInsertionPointToStart(parOp.getBody());
    return parOp;
  }

  // Emit a sequential or vector loop.
  SmallVector<Value, 4> operands;
  if (codegen.redVal)
    operands.push_back(codegen.redVal);
  if (codegen.expValues)
    operands.push_back(codegen.expCount);

  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lo, hi, step, operands);

  if (codegen.redVal)
    updateReduc(merger, codegen, forOp.getRegionIterArgs().front());
  if (codegen.expValues)
    codegen.expCount = forOp.getRegionIterArgs().back();
  // Assign induction variable to sparse or dense index.
  Value iv = forOp.getInductionVar();
  if (isSparse)
    codegen.pidxs[tensor][idx] = iv;
  else
    codegen.loops[idx] = iv;

  builder.setInsertionPointToStart(forOp.getBody());
  return forOp;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned idx, bool needsUniv,
                           BitVector &indices) {
  SmallVector<Type, 4> types;
  SmallVector<Value, 4> operands;
  // Construct the while-loop with a parameter for each index.
  Type indexType = builder.getIndexType();
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (!indices[b])
      continue;
    if (isCompressedDLT(merger.getDimLevelType(b)) ||
        isSingletonDLT(merger.getDimLevelType(b))) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      types.push_back(indexType);
      operands.push_back(codegen.pidxs[tensor][idx]);
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
    }
  }
  if (codegen.redVal) {
    types.push_back(codegen.redVal.getType());
    operands.push_back(codegen.redVal);
  }
  if (codegen.expValues) {
    types.push_back(indexType);
    operands.push_back(codegen.expCount);
  }
  if (needsUniv) {
    types.push_back(indexType);
    operands.push_back(codegen.loops[idx]);
  }
  assert(types.size() == operands.size());
  Location loc = op.getLoc();
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(loc, types, operands);

  SmallVector<Location> locs(types.size(), loc);
  Block *before = builder.createBlock(&whileOp.getBefore(), {}, types, locs);
  Block *after = builder.createBlock(&whileOp.getAfter(), {}, types, locs);

  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  builder.setInsertionPointToStart(&whileOp.getBefore().front());
  Value cond;
  unsigned o = 0;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (!indices[b])
      continue;
    if (isCompressedDLT(merger.getDimLevelType(b)) ||
        isSingletonDLT(merger.getDimLevelType(b))) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = before->getArgument(o);
      Value op2 = codegen.highs[tensor][idx];
      Value opc = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                op1, op2);
      cond = cond ? builder.create<arith::AndIOp>(loc, cond, opc) : opc;
      codegen.pidxs[tensor][idx] = after->getArgument(o++);
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
    }
  }
  if (codegen.redVal)
    updateReduc(merger, codegen, after->getArgument(o++));
  if (codegen.expValues)
    codegen.expCount = after->getArgument(o++);
  if (needsUniv)
    codegen.loops[idx] = after->getArgument(o++);
  assert(o == operands.size());
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());
  builder.setInsertionPointToStart(&whileOp.getAfter().front());
  return whileOp;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, unsigned at, bool needsUniv,
                          BitVector &indices) {
  unsigned idx = codegen.topSort[at];
  if (indices.count() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == codegen.topSort.size() - 1;
    return genFor(merger, codegen, builder, op, isOuter, isInner, idx, indices);
  }
  return genWhile(merger, codegen, builder, op, idx, needsUniv, indices);
}

/// Generates the local variables for this loop, consisting of the sparse
/// indices, restored universal dense index, and dense positions.
static void genLocals(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                      linalg::GenericOp op, unsigned at, bool needsUniv,
                      BitVector &locals) {
  std::vector<unsigned> &topSort(codegen.topSort);
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse indices.
  Value min;
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if (!locals[b])
      continue;
    if (isCompressedDLT(merger.getDimLevelType(b)) ||
        isSingletonDLT(merger.getDimLevelType(b))) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value ptr = codegen.indices[tensor][idx];
      Value s = codegen.pidxs[tensor][idx];
      Value load = genLoad(codegen, builder, loc, ptr, s);
      codegen.idxs[tensor][idx] = load;
      if (!needsUniv) {
        if (min) {
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, load, min);
          min = builder.create<arith::SelectOp>(loc, cmp, load, min);
        } else {
          min = load;
        }
      }
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
    }
  }

  // Merge dense universal index over minimum.
  if (min) {
    assert(!needsUniv);
    codegen.loops[idx] = min;
  }

  // Initialize dense positions. Note that we generate dense indices of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if ((locals[b] || merger.isOutTensor(b, idx)) &&
        isDenseDLT(merger.getDimLevelType(b))) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      unsigned pat = at;
      for (; pat != 0; pat--)
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      Value p = (pat == 0) ? constantIndex(builder, loc, 0)
                           : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = genAddress(
          codegen, builder, loc, codegen.sizes[idx], p, codegen.loops[idx]);
    }
  }
}

/// Generates the induction structure for a while-loop.
static void genWhileInduction(Merger &merger, CodeGen &codegen,
                              OpBuilder &builder, linalg::GenericOp op,
                              unsigned idx, bool needsUniv,
                              BitVector &induction, scf::WhileOp whileOp) {
  Location loc = op.getLoc();
  // Finalize each else branch of all if statements.
  if (codegen.redVal || codegen.expValues) {
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
      assert(y == yields.size());
      builder.create<scf::YieldOp>(loc, yields);
      builder.setInsertionPointAfter(ifOp);
    }
  }
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
  // Finalize the induction. Note that the induction could be performed
  // in the individual if-branches to avoid re-evaluating the conditions.
  // However, that would result in a rather elaborate forest of yield
  // instructions during code generation. Moreover, performing the induction
  // after the if-statements more closely resembles code generated by TACO.
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  Value one = constantIndex(builder, loc, 1);
  for (unsigned b = 0, be = induction.size(); b < be; b++) {
    if (!induction[b])
      continue;
    if (isCompressedDLT(merger.getDimLevelType(b)) ||
        isSingletonDLT(merger.getDimLevelType(b))) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = codegen.idxs[tensor][idx];
      Value op2 = codegen.loops[idx];
      Value op3 = codegen.pidxs[tensor][idx];
      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                op1, op2);
      Value add = builder.create<arith::AddIOp>(loc, op3, one);
      operands.push_back(builder.create<arith::SelectOp>(loc, cmp, add, op3));
      codegen.pidxs[tensor][idx] = whileOp->getResult(o++);
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
    }
  }
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, whileOp->getResult(o++));
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = whileOp->getResult(o++);
  }
  if (needsUniv) {
    operands.push_back(
        builder.create<arith::AddIOp>(loc, codegen.loops[idx], one));
    codegen.loops[idx] = whileOp->getResult(o++);
  }
  assert(o == operands.size());
  builder.create<scf::YieldOp>(loc, operands);
  builder.setInsertionPointAfter(whileOp);
}

/// Generates the induction structure for a for-loop.
static void genForInduction(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            Operation *loop) {
  Location loc = op.getLoc();
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, loop->getResult(o++));
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = loop->getResult(o++);
  }
  assert(o == operands.size());
  if (o > 0)
    builder.create<scf::YieldOp>(loc, operands);
  builder.setInsertionPointAfter(loop);
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
      Value op1 = codegen.idxs[tensor][idx];
      Value op2 = codegen.loops[idx];
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
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
static void endIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                  linalg::GenericOp op, scf::IfOp ifOp, Operation *loop,
                  Value redInput, Value cntInput) {
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, redInput);
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = cntInput;
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
  assert(!codegen.loops[idx]);
  // Emit invariants at this loop sequence level.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, builder, op, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  unsigned l0 = merger.set(lts)[0];
  bool needsUniv =
      genInit(merger, codegen, builder, op, at, merger.lat(l0).bits);
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

/// Starts a single loop in current sequence.
static Operation *startLoop(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            unsigned at, unsigned li, bool needsUniv) {
  // Emit the for/while-loop control.
  Operation *loop = genLoop(merger, codegen, builder, op, at, needsUniv,
                            merger.lat(li).simple);
  // Emit the locals for this loop.
  genLocals(merger, codegen, builder, op, at, needsUniv, merger.lat(li).bits);
  return loop;
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                    linalg::GenericOp op, Operation *loop, unsigned idx,
                    unsigned li, bool needsUniv) {
  // End a while-loop.
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    genWhileInduction(merger, codegen, builder, op, idx, needsUniv,
                      merger.lat(li).bits, whileOp);
    return needsUniv;
  }
  // End a for-loop.
  genForInduction(merger, codegen, builder, op, loop);
  return false;
}

/// Ends a loop sequence at given level.
static void endLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned exp, unsigned at,
                       unsigned idx, unsigned ldx) {
  assert(codegen.loops[idx]);
  codegen.loops[idx] = Value();
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
          endIf(merger, codegen, rewriter, op, ifOp, loop, redInput, cntInput);
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
  OpOperand *lhs = op.getOutputOperand(0);
  Type resType = lhs->get().getType();
  if (getSparseTensorEncoding(resType)) {
    // The sparse tensor rematerializes from the original sparse tensor's
    // underlying sparse storage format.
    rewriter.replaceOpWithNewOp<LoadOp>(op, resType, lhs->get(),
                                        codegen.sparseOut == lhs);
  } else {
    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
    Value val = codegen.buffers.back(); // value array
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
    if (op.getNumOutputs() != 1)
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

    // Recursively generates code if admissible.
    merger.setHasSparseOut(sparseOut != nullptr);
    CodeGen codegen(options, numTensors, numLoops, sparseOut, outerParNest,
                    topSort);
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
    for (OpOperand *t : op.getInputOperands()) {
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
