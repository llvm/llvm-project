//===- Sparsification.cpp - Implementation of linalg sparsification -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering annotated linalg dialect to sparse code.
//
// The concept of letting a compiler generate sparse code automatically was
// pioneered for dense linear algebra code in Fortran by [Bik96] in MT1 and
// formalized to tensor algebra by [Kjolstad17,20] for the Sparse Tensor
// Algebra Compiler (TACO). The implementation in this file closely follows
// the "sparse iteration theory" that forms the foundation of TACO. A rewriting
// rule is applied to each tensor expression in linalg (MLIR's tensor index
// notation) where the sparsity of tensors is indicated with annotation using
// a per-dimension specification of sparse/dense storage together with a
// specification of the order on the dimensions. Subsequently, a topologically
// sorted iteration graph, reflecting the required order on indices with respect
// to the dimensions of each tensor, is constructed to ensure that all tensors
// are visited in natural index order. Next, iteration lattices are constructed
// for the tensor expression for every index in topological order. Each
// iteration lattice point consists of a conjunction of tensor indices together
// with a tensor (sub)expression that needs to be evaluated for that
// conjunction. Within the lattice, iteration points are ordered according to
// the way indices are exhausted. As such these iteration lattices drive actual
// sparse code generation, which consists of a tedious but relatively
// straightforward one-to-one mapping from iteration lattices to combinations
// of for-loops, while-loops, and if-statements.
//
// [Bik96] Aart J.C. Bik. Compiler Support for Sparse Matrix Computations.
// PhD thesis, Leiden University, May 1996 (aartbik.com/sparse.php).
// [Kjolstad17] Fredrik Berg Kjolstad, Shoaib Ashraf Kamil, Stephen Chou,
// David Lugato, and Saman Amarasinghe. The Tensor Algebra Compiler.
// Proceedings of the ACM on Programming Languages, October 2017.
// [Kjolstad20] Fredrik Berg Kjolstad. Sparse Tensor Algebra Compilation.
// PhD thesis, MIT, February, 2020 (tensor-compiler.org).
//
// Implementation detail: We use llvm::SmallVector for vectors with
// variable lengths and std::vector for vectors with fixed lengths.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

namespace {

enum class Kind { kTensor, kInvariant, kMulF, kMulI, kAddF, kAddI };

/// Tensor expression. Represents a MLIR expression in tensor index notation.
/// For tensors and invariants, e0 denotes the tensor index. For all binary
/// operations, e0 and e1 denote the index of the children tensor expressions.
struct TensorExp {
  TensorExp(Kind k, unsigned x, unsigned y) : kind(k), e0(x), e1(y) {}
  Kind kind;
  unsigned e0;
  unsigned e1;
};

/// Lattice point. Each lattice point consist of a conjunction of tensor
/// loop indices (encoded in a bitvector) and the index of the corresponding
/// tensor expression.
struct LatPoint {
  LatPoint(unsigned n, unsigned e, unsigned b) : bits(n, false), exp(e) {
    bits.set(b);
  }
  LatPoint(const llvm::BitVector &b, unsigned e) : bits(b), exp(e) {}
  llvm::BitVector bits;
  unsigned exp;
};

/// A class to handle all iteration lattice operations. This class abstracts
/// away from some implementation details of storing iteration lattices and
/// tensor expressions. This allows for fine-tuning performance characteristics
/// independently from the basic algorithm if bottlenecks are identified.
class Merger {
public:
  Merger(unsigned t, unsigned l)
      : numTensors(t), numLoops(l), isSparse(t, std::vector<bool>(l, false)) {}

  /// Adds a tensor expression. Returns its index.
  unsigned addExp(Kind k, unsigned e0, unsigned e1 = -1u) {
    unsigned e = tensorExps.size();
    tensorExps.push_back(TensorExp(k, e0, e1));
    return e;
  }

  /// Adds an iteration lattice point. Returns its index.
  unsigned addLat(unsigned t, unsigned i, unsigned e) {
    assert(t < numTensors && i < numLoops);
    unsigned p = latPoints.size();
    latPoints.push_back(LatPoint(numLoops * numTensors, e, numTensors * i + t));
    return p;
  }

  /// Adds a new, initially empty, set. Returns its index.
  unsigned addSet() {
    unsigned s = latSets.size();
    latSets.emplace_back(SmallVector<unsigned, 16>());
    return s;
  }

  /// Computes a single conjunction of two lattice points by taking the "union"
  /// of loop indices (effectively constucting a larger "intersection" of those
  /// indices) with a newly constructed tensor (sub)expression of given kind.
  /// Returns the index of the new lattice point.
  unsigned conjLatPoint(Kind kind, unsigned p0, unsigned p1) {
    unsigned p = latPoints.size();
    llvm::BitVector nb = llvm::BitVector(latPoints[p0].bits);
    nb |= latPoints[p1].bits;
    unsigned e = addExp(kind, latPoints[p0].exp, latPoints[p1].exp);
    latPoints.push_back(LatPoint(nb, e));
    return p;
  }

  /// Conjunctive merge of L1 and L2 is conjunction of cartesian product.
  /// Returns the index of the new set.
  unsigned takeConj(Kind kind, unsigned s0, unsigned s1) {
    unsigned s = addSet();
    for (unsigned p0 : latSets[s0])
      for (unsigned p1 : latSets[s1])
        latSets[s].push_back(conjLatPoint(kind, p0, p1));
    return s;
  }

  /// Disjunctive merge of L0 and L1 is (L0 /\_op L1, L0, L1).
  /// Returns the index of the new set.
  unsigned takeDisj(Kind kind, unsigned s0, unsigned s1) {
    unsigned s = takeConj(kind, s0, s1);
    for (unsigned p : latSets[s0])
      latSets[s].push_back(p);
    for (unsigned p : latSets[s1])
      latSets[s].push_back(p);
    return s;
  }

  /// Optimizes the iteration lattice points in the given set.
  unsigned optimize(unsigned s0) {
    unsigned s = addSet();
    assert(latSets[s0].size() != 0);
    unsigned p0 = latSets[s0][0];
    for (unsigned p1 : latSets[s0]) {
      bool add = true;
      if (p0 != p1) {
        llvm::BitVector tmp = latPoints[p1].bits;
        tmp ^= latPoints[p0].bits;
        if (hasAnyOf(tmp, false))
          continue; // dense exhausted?
        for (unsigned p2 : latSets[s]) {
          tmp = latPoints[p1].bits;
          tmp ^= latPoints[p2].bits;
          if (tmp.count() == 0) {
            add = false; // direct dup?
            break;
          }
        }
        assert(!add || latGT(p0, p1));
      }
      if (add)
        latSets[s].push_back(p1);
    }
    return s;
  }

  // Returns true if Li > Lj.
  bool latGT(unsigned i, unsigned j) const {
    const llvm::BitVector &bitsi = latPoints[i].bits;
    const llvm::BitVector &bitsj = latPoints[j].bits;
    assert(bitsi.size() == bitsj.size());
    if (bitsi.count() > bitsj.count()) {
      for (unsigned b = 0, be = bitsj.size(); b < be; b++)
        if (bitsj[b] && !bitsi[b])
          return false;
      return true;
    }
    return false;
  }

  // Bit translation.
  unsigned tensor(unsigned b) const { return b % numTensors; }
  unsigned index(unsigned b) const { return b / numTensors; }

  // Returns true if bit corresponds to sparse access.
  bool isSparseBit(unsigned b) const {
    return isSparseAccess(tensor(b), index(b));
  }

  // Returns true if tensor access at given index is sparse.
  bool isSparseAccess(unsigned t, unsigned i) const {
    assert(t < numTensors && i < numLoops);
    return isSparse[t][i];
  }

  // Returns true if any set bit corresponds to sparse/dense access.
  bool hasAnyOf(const llvm::BitVector &bits, bool sparse) const {
    for (unsigned b = 0, be = bits.size(); b < be; b++)
      if (bits[b] && isSparseBit(b) == sparse)
        return true;
    return false;
  }

  // Getters.
  std::vector<std::vector<bool>> &sparse() { return isSparse; }
  TensorExp &exp(unsigned e) { return tensorExps[e]; }
  LatPoint &lat(unsigned l) { return latPoints[l]; }
  SmallVector<unsigned, 16> &set(unsigned s) { return latSets[s]; }

private:
  const unsigned numTensors;
  const unsigned numLoops;

  std::vector<std::vector<bool>> isSparse;
  llvm::SmallVector<TensorExp, 32> tensorExps;
  llvm::SmallVector<LatPoint, 16> latPoints;
  llvm::SmallVector<SmallVector<unsigned, 16>, 8> latSets;
};

// Code generation.
struct CodeGen {
  CodeGen(unsigned numTensors, unsigned numLoops)
      : loops(numLoops), sizes(numLoops), buffers(numTensors),
        pointers(numTensors, std::vector<Value>(numLoops)),
        indices(numTensors, std::vector<Value>(numLoops)),
        highs(numTensors, std::vector<Value>(numLoops)),
        pidxs(numTensors, std::vector<Value>(numLoops)),
        idxs(numTensors, std::vector<Value>(numLoops)) {}
  // Universal dense indices and upper bounds (by index).
  std::vector<Value> loops;
  std::vector<Value> sizes;
  // Buffers for storing dense and sparse numerical values (by tensor).
  std::vector<Value> buffers;
  // Sparse storage schemes (1-D): pointers and indices (by tensor and index).
  std::vector<std::vector<Value>> pointers;
  std::vector<std::vector<Value>> indices;
  // Sparse iteration information (by tensor and index).
  std::vector<std::vector<Value>> highs;
  std::vector<std::vector<Value>> pidxs;
  std::vector<std::vector<Value>> idxs;
};

} // namespace

/// Helper method to inspect sparse annotations in the linalg operation.
/// Fills the per-dimension sparsity information for all tensors.
static void findSparseAnnotations(linalg::GenericOp op,
                                  std::vector<std::vector<bool>> &isSparse) {
  unsigned numTensors = op.getNumInputsAndOutputs();
  ArrayAttr sparseAttr = op.sparseAttr();
  for (unsigned t = 0; t < numTensors; t++) {
    auto map = op.getIndexingMap(t);
    auto dimAttr = sparseAttr[t].cast<ArrayAttr>();
    // For each tensor, we accept a per-dimension Sparse or Dense annotation.
    // This is translated to the loop index that indexes that dimension.
    unsigned rank = op.getShapedType(t).getRank();
    for (unsigned d = 0; d < rank; d++)
      if (isSparseDim(dimAttr[d])) {
        unsigned idx = map.getDimPosition(d);
        isSparse[t][idx] = true;
      } else {
        assert(isDenseDim(dimAttr[d]));
      }
  }
}

/// A DFS helper to compute a topological sort. Note that recursion is
/// bounded by the number of implicit loops, which is always small.
/// Returns false when a cycle is detected.
static bool topSortDFS(unsigned i, std::vector<unsigned> &visit,
                       std::vector<unsigned> &topSort,
                       std::vector<std::vector<bool>> &adjM) {
  if (visit[i] != 0)
    return visit[i] != 1; // 1 denotes cycle!
  visit[i] = 1;
  for (unsigned j = 0, e = visit.size(); j < e; j++)
    if (adjM[i][j])
      if (!topSortDFS(j, visit, topSort, adjM))
        return false;
  visit[i] = 2;
  topSort.push_back(i);
  return true;
}

/// Computes a topologically sorted iteration graph for the linalg operation.
/// Ensures all tensors are visited in natural index order. This is essential
/// for sparse storage formats since these only support access along fixed
/// dimensions. Even for dense storage formats, however, the natural index
/// order yields innermost unit-stride access with better spatial locality.
static bool computeIterationGraph(linalg::GenericOp op,
                                  std::vector<unsigned> &topSort) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = op.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));

  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (auto imap : llvm::enumerate(op.indexing_maps())) {
    auto map = imap.value().template cast<AffineMapAttr>().getValue();
    assert(map.getNumDims() == n);
    // At the moment, we take the index variables in the tensor access
    // expression in the order in which they appear (conceptually a
    // "row-major" layout of every tensor). So, a tensor access A_ijk
    // forces the ordering i < j < k on the loop indices.
    // TODO: support affine map to define alternative dimension orders.
    for (unsigned d = 1, e = map.getNumResults(); d < e; d++) {
      unsigned f = map.getDimPosition(d - 1);
      unsigned t = map.getDimPosition(d);
      adjM[f][t] = true;
    }
  }

  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  topSort.reserve(n);
  std::vector<unsigned> visit(n, 0);
  for (unsigned i = 0; i < n; i++)
    if (visit[i] == 0)
      if (!topSortDFS(i, visit, topSort, adjM))
        return false; // cycle!
  std::reverse(std::begin(topSort), std::end(topSort));
  return true;
}

/// Traverses the SSA tree (possibly a DAG) to build a tensor expression.
/// This simplifies constructing (sub)expressions during iteration lattice
/// building (compared to using the SSA representation everywhere).
static Optional<unsigned> buildTensorExp(Merger &merger, linalg::GenericOp op,
                                         Value val) {
  Operation *def = val.getDefiningOp();
  if (auto arg = val.dyn_cast<BlockArgument>()) {
    unsigned argN = arg.getArgNumber();
    if (arg.getOwner()->getParentOp() == op) {
      // Any parameter of the generic op is considered a tensor,
      // indexed by the implicit loop bounds.
      auto map = op.getIndexingMap(argN);
      if (map.isProjectedPermutation())
        return merger.addExp(Kind::kTensor, argN);
    } else {
      // Any parameter of a higher op is invariant in the tensor expression.
      return merger.addExp(Kind::kInvariant, argN);
    }
  } else if (def->getNumOperands() == 2) {
    // Construct binary operations if subexpressions could be built.
    auto x = buildTensorExp(merger, op, def->getOperand(0));
    auto y = buildTensorExp(merger, op, def->getOperand(1));
    if (x.hasValue() && y.hasValue()) {
      unsigned e0 = x.getValue();
      unsigned e1 = y.getValue();
      if (isa<MulFOp>(def))
        return merger.addExp(Kind::kMulF, e0, e1);
      if (isa<MulIOp>(def))
        return merger.addExp(Kind::kMulI, e0, e1);
      if (isa<AddFOp>(def))
        return merger.addExp(Kind::kAddF, e0, e1);
      if (isa<AddIOp>(def))
        return merger.addExp(Kind::kAddI, e0, e1);
    }
  }
  // Cannot build (yet).
  return None;
}

/// Builds the iteration lattices in a bottom-up traversal given the remaining
/// tensor (sub)expression and the next loop index in the iteration graph.
static unsigned buildLattices(Merger &merger, linalg::GenericOp op,
                              unsigned exp, unsigned idx) {
  Kind kind = merger.exp(exp).kind;
  if (kind == Kind::kTensor || kind == Kind::kInvariant) {
    // Either the index is really used in the tensor expression, or it it
    // set to the "non-existing dense index" in that dimension.
    unsigned s = merger.addSet();
    merger.set(s).push_back(merger.addLat(merger.exp(exp).e0, idx, exp));
    return s;
  }
  unsigned s0 = buildLattices(merger, op, merger.exp(exp).e0, idx);
  unsigned s1 = buildLattices(merger, op, merger.exp(exp).e1, idx);
  switch (kind) {
  case Kind::kTensor:
  case Kind::kInvariant:
    llvm_unreachable("handled above");
  case Kind::kMulF:
  case Kind::kMulI:
    return merger.takeConj(kind, s0, s1);
  case Kind::kAddF:
  case Kind::kAddI:
    return merger.takeDisj(kind, s0, s1);
  }
}

/// Local bufferization of all dense and sparse data structures.
/// This code enables testing the first prototype sparse compiler.
// TODO: replace this with a proliferated bufferization strategy
void genBuffers(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                linalg::GenericOp op) {
  Location loc = op.getLoc();
  unsigned numTensors = op.getNumInputsAndOutputs();
  unsigned numInputs = op.getNumInputs();
  assert(numTensors == numInputs + 1);
  Type indexType = rewriter.getIndexType();

  // For now, set all unknown dimensions to 999.
  // TODO: compute these values (using sparsity or by reading tensor)
  Value unknown = rewriter.create<ConstantIndexOp>(loc, 999);

  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and allocate dense or sparse buffer(s).
  SmallVector<Value, 4> args;
  for (unsigned t = 0; t < numTensors; t++) {
    auto tensorType = op.getShapedType(t);
    auto shape = tensorType.getShape();
    auto map = op.getIndexingMap(t);
    // Scan all dimensions of current tensor.
    bool allDense = true;
    args.clear();
    for (unsigned d = 0, rank = shape.size(); d < rank; d++) {
      unsigned i = map.getDimPosition(d);
      // Handle sparse storage schemes.
      if (merger.isSparseAccess(t, i)) {
        allDense = false;
        auto dynTp = MemRefType::get({ShapedType::kDynamicSize}, indexType);
        codegen.pointers[t][i] = rewriter.create<AllocaOp>(loc, dynTp, unknown);
        codegen.indices[t][i] = rewriter.create<AllocaOp>(loc, dynTp, unknown);
      }
      // Find lower and upper bound in current dimension.
      Value up;
      if (shape[d] == TensorType::kDynamicSize) {
        // For the output tensor, we may need to infer the upper bound.
        // For all others, we look at the incoming argument.
        if (t == numInputs && !op.getNumInitTensors()) {
          up = codegen.sizes[i];
          assert(up); // TODO: what else?
        } else {
          Value arg = t < numInputs ? op.getInput(t) : op.getInitTensor(0);
          up = rewriter.create<DimOp>(loc, arg, d);
        }
        args.push_back(up);
      } else {
        up = rewriter.create<ConstantIndexOp>(loc, shape[d]);
      }
      codegen.sizes[i] = codegen.highs[t][i] = up;
    }
    // Allocate dense or sparse buffer for numerical values.
    if (allDense) {
      auto denseTp = MemRefType::get(shape, tensorType.getElementType());
      codegen.buffers[t] = rewriter.create<AllocaOp>(loc, denseTp, args);
    } else {
      auto sparseTp = MemRefType::get({ShapedType::kDynamicSize},
                                      tensorType.getElementType());
      codegen.buffers[t] = rewriter.create<AllocaOp>(loc, sparseTp, unknown);
    }
  }
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned tensor) {
  SmallVector<Value, 4> args;
  auto map = op.getIndexingMap(tensor);
  bool sparse = false;
  for (unsigned i = 0, m = map.getNumResults(); i < m; ++i) {
    unsigned idx = map.getDimPosition(i);
    args.push_back(codegen.loops[idx]); // universal dense index
    if (sparse || merger.isSparseAccess(tensor, idx)) {
      sparse = true;
      args.clear();
      args.push_back(codegen.pidxs[tensor][idx]); // position index
    }
  }
  return rewriter.create<LoadOp>(op.getLoc(), codegen.buffers[tensor], args);
}

/// Generates a store on a dense tensor.
static void genTensorStore(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned tensor, Value rhs) {
  SmallVector<Value, 4> args;
  auto map = op.getIndexingMap(tensor);
  for (unsigned i = 0, m = map.getNumResults(); i < m; ++i) {
    unsigned idx = map.getDimPosition(i);
    args.push_back(codegen.loops[idx]); // universal dense index
  }
  rewriter.create<StoreOp>(op.getLoc(), rhs, codegen.buffers[tensor], args);
}

/// Recursively generates tensor expression.
static Value genExp(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, unsigned exp) {
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, merger.exp(exp).e0);
  else if (merger.exp(exp).kind == Kind::kInvariant)
    return op.getParentRegion()->front().getArgument(merger.exp(exp).e0);
  Value v0 = genExp(merger, codegen, rewriter, op, merger.exp(exp).e0);
  Value v1 = genExp(merger, codegen, rewriter, op, merger.exp(exp).e1);
  switch (merger.exp(exp).kind) {
  case Kind::kTensor:
  case Kind::kInvariant:
    llvm_unreachable("handled above");
  case Kind::kMulF:
    return rewriter.create<MulFOp>(op.getLoc(), v0, v1);
  case Kind::kMulI:
    return rewriter.create<MulIOp>(op.getLoc(), v0, v1);
  case Kind::kAddF:
    return rewriter.create<AddFOp>(op.getLoc(), v0, v1);
  case Kind::kAddI:
    return rewriter.create<AddIOp>(op.getLoc(), v0, v1);
  }
}

/// Generates initialization code for the subsequent loop sequence at
/// current index level. Returns true if the loop sequence needs to
/// maintain the universal index.
static bool genInit(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned at, llvm::BitVector &inits) {
  bool needsUniv = false;
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse positions.
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  for (unsigned b = 0, be = inits.size(); b < be; b++) {
    if (inits[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      if (merger.isSparseBit(b)) {
        // Initialize sparse index.
        unsigned pat = at;
        for (; pat != 0; pat--) {
          if (codegen.pidxs[tensor][topSort[pat - 1]])
            break;
        }
        Value ptr = codegen.pointers[tensor][idx];
        Value p = (pat == 0) ? rewriter.create<ConstantIndexOp>(loc, 0)
                             : codegen.pidxs[tensor][topSort[pat - 1]];
        codegen.pidxs[tensor][idx] = rewriter.create<LoadOp>(loc, ptr, p);
        p = rewriter.create<AddIOp>(loc, p, one);
        codegen.highs[tensor][idx] = rewriter.create<LoadOp>(loc, ptr, p);
      } else {
        // Dense index still in play.
        needsUniv = true;
      }
    }
  }

  // Initialize the universal dense index.
  codegen.loops[idx] = rewriter.create<ConstantIndexOp>(loc, 0);
  return needsUniv;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static void genLoop(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, unsigned idx, bool needsUniv,
                    llvm::BitVector &indices, scf::ForOp &forOp,
                    scf::WhileOp &whileOp) {
  Location loc = op.getLoc();

  // Emit a for-loop for a single index.
  if (indices.count() == 1) {
    unsigned fb = indices.find_first();
    unsigned tensor = merger.tensor(fb);
    assert(idx == merger.index(fb));
    // Emit a sparse for-loop or a dense for-loop.
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);
    if (merger.isSparseBit(fb)) {
      forOp = rewriter.create<scf::ForOp>(loc, codegen.pidxs[tensor][idx],
                                          codegen.highs[tensor][idx], one);
      codegen.pidxs[tensor][idx] = forOp.getInductionVar();
    } else {
      forOp = rewriter.create<scf::ForOp>(loc, codegen.loops[idx],
                                          codegen.sizes[idx], one);
      codegen.loops[idx] = forOp.getInductionVar();
    }
    rewriter.setInsertionPointToStart(forOp.getBody());
    return;
  }

  // Otherwise, emit a while-loop for co-iteration.
  Type indexType = rewriter.getIndexType();
  SmallVector<Type, 4> types;
  SmallVector<Value, 4> operands;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isSparseBit(b)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      types.push_back(indexType);
      operands.push_back(codegen.pidxs[tensor][idx]);
    }
  }
  if (needsUniv) {
    types.push_back(indexType);
    operands.push_back(codegen.loops[idx]);
  }
  whileOp = rewriter.create<scf::WhileOp>(loc, types, operands);
  Block *before = rewriter.createBlock(&whileOp.before(), {}, types);
  Block *after = rewriter.createBlock(&whileOp.after(), {}, types);
  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  rewriter.setInsertionPointToStart(&whileOp.before().front());
  Value cond;
  unsigned o = 0;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isSparseBit(b)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = before->getArgument(o);
      Value op2 = codegen.highs[tensor][idx];
      Value opc = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, op1, op2);
      cond = cond ? rewriter.create<AndOp>(loc, cond, opc) : opc;
      codegen.pidxs[tensor][idx] = after->getArgument(o++);
    }
  }
  if (needsUniv)
    codegen.loops[idx] = after->getArgument(o++);
  assert(o == operands.size());
  rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());
  rewriter.setInsertionPointToStart(&whileOp.after().front());
}

/// Generates the local variables for this loop, consisting of the sparse
/// indices, restored universal dense index, and dense positions.
static void genLocals(Merger &merger, CodeGen &codegen,
                      PatternRewriter &rewriter, linalg::GenericOp op,
                      std::vector<unsigned> &topSort, unsigned at,
                      bool needsUniv, llvm::BitVector &locals) {
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse indices.
  Value min;
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if (locals[b] && merger.isSparseBit(b)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value ld = rewriter.create<LoadOp>(loc, codegen.indices[tensor][idx],
                                         codegen.pidxs[tensor][idx]);
      codegen.idxs[tensor][idx] = ld;
      if (!needsUniv) {
        if (min) {
          Value cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, ld, min);
          min = rewriter.create<SelectOp>(loc, cmp, ld, min);
        } else {
          min = ld;
        }
      }
    }
  }

  // Merge dense universal index over minimum.
  if (min) {
    assert(!needsUniv);
    codegen.loops[idx] = min;
  }

  // Initialize dense positions.
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if (locals[b] && !merger.isSparseBit(b)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      if (!codegen.highs[tensor][idx])
        continue; // unused dimension
      unsigned pat = at;
      for (; pat != 0; pat--)
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      Value p = (pat == 0) ? rewriter.create<ConstantIndexOp>(loc, 0)
                           : codegen.pidxs[tensor][topSort[pat - 1]];
      Value m = rewriter.create<MulIOp>(loc, codegen.sizes[idx], p);
      codegen.pidxs[tensor][idx] =
          rewriter.create<AddIOp>(loc, m, codegen.loops[idx]);
    }
  }
}

/// Generates the induction structure for a while-loop.
static void genWhileInduction(Merger &merger, CodeGen &codegen,
                              PatternRewriter &rewriter, linalg::GenericOp op,
                              unsigned idx, bool needsUniv,
                              llvm::BitVector &induction, ResultRange results) {
  Location loc = op.getLoc();
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  for (unsigned b = 0, be = induction.size(); b < be; b++)
    if (induction[b] && merger.isSparseBit(b)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = codegen.idxs[tensor][idx];
      Value op2 = codegen.loops[idx];
      Value op3 = codegen.pidxs[tensor][idx];
      Value cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, op1, op2);
      Value add = rewriter.create<AddIOp>(loc, op3, one);
      operands.push_back(rewriter.create<SelectOp>(loc, cmp, add, op3));
      codegen.pidxs[tensor][idx] = results[o++];
    }
  if (needsUniv) {
    operands.push_back(rewriter.create<AddIOp>(loc, codegen.loops[idx], one));
    codegen.loops[idx] = results[o++];
  }
  assert(o == operands.size());
  rewriter.create<scf::YieldOp>(loc, operands);
}

/// Generates a single if-statement within a while-loop.
static void genIf(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                  linalg::GenericOp op, unsigned idx,
                  llvm::BitVector &conditions, scf::IfOp &ifOp) {
  Location loc = op.getLoc();
  if (ifOp)
    rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (conditions[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value clause;
      if (merger.isSparseBit(b)) {
        Value op1 = codegen.idxs[tensor][idx];
        Value op2 = codegen.loops[idx];
        clause = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, op1, op2);
      } else {
        clause = rewriter.create<ConstantIntOp>(loc, 1, 1); // true
      }
      cond = cond ? rewriter.create<AndOp>(loc, cond, clause) : clause;
    }
  }
  ifOp = rewriter.create<scf::IfOp>(loc, cond, /*else*/ true);
  rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
}

/// Optimize the loop indices of Li with two rules rules:
/// (1) convert multiple dense to single dense, and
/// (2) convert singleton sparse/dense to sparse/random access.
static void optimizeIndices(Merger merger, unsigned lsize,
                            llvm::BitVector &indices) {
  if (merger.hasAnyOf(indices, false)) {
    bool reset = lsize == 1 && merger.hasAnyOf(indices, true);
    for (unsigned b = 0, be = indices.size(); b < be; b++) {
      if (indices[b] && !merger.isSparseBit(b)) {
        if (reset)
          indices.reset(b);
        reset = true;
      }
    }
  }
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == topSort.size()) {
    unsigned lhs = op.getNumInputsAndOutputs() - 1;
    Value rhs = genExp(merger, codegen, rewriter, op, exp);
    genTensorStore(merger, codegen, rewriter, op, lhs, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  // Then emit initialization code for the loop sequence at this level.
  // We maintain the universal dense index if dense indices are still
  // in play for a non-singleton loop sequence.
  unsigned idx = topSort[at];
  unsigned lts = merger.optimize(buildLattices(merger, op, exp, idx));
  unsigned lsize = merger.set(lts).size();
  assert(lsize != 0);
  unsigned l0 = merger.set(lts)[0];
  LatPoint lat0 = merger.lat(l0);
  bool needsUniv =
      genInit(merger, codegen, rewriter, op, topSort, at, lat0.bits) &&
      lsize > 1;

  // Emit a loop for every lattice point L0 >= Li.
  for (unsigned li : merger.set(lts)) {
    LatPoint lati = merger.lat(li);

    // Emit loop.
    scf::ForOp forOp;
    scf::WhileOp whileOp;
    llvm::BitVector indices = lati.bits;
    optimizeIndices(merger, lsize, indices);
    genLoop(merger, codegen, rewriter, op, idx, needsUniv, indices, forOp,
            whileOp);
    genLocals(merger, codegen, rewriter, op, topSort, at, needsUniv, lati.bits);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    scf::IfOp ifOp;
    for (unsigned lj : merger.set(lts)) {
      if (li == lj || merger.latGT(li, lj)) {
        LatPoint latj = merger.lat(lj);
        llvm::BitVector tmp = latj.bits;
        tmp ^= lati.bits;
        if (merger.hasAnyOf(tmp, false))
          continue; // dense exhausted within if/else
        // Recurse into body of each branch.
        if (whileOp)
          genIf(merger, codegen, rewriter, op, idx, latj.bits, ifOp);
        genStmt(merger, codegen, rewriter, op, topSort, latj.exp, at + 1);
      }
    }

    // Wrap-up induction and restore insertion point.
    if (forOp) {
      needsUniv = false;
      rewriter.setInsertionPointAfter(forOp);
    } else {
      rewriter.setInsertionPointToEnd(&whileOp.after().front());
      genWhileInduction(merger, codegen, rewriter, op, idx, needsUniv,
                        lati.bits, whileOp.results());
      rewriter.setInsertionPointAfter(whileOp);
    }
  }
}

namespace {

/// Sparse rewriting rule for generic Lingalg operation.
struct GenericOpSparsifier : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    unsigned numTensors = op.getNumInputsAndOutputs();
    unsigned numLoops = op.iterator_types().getValue().size();
    Merger merger(numTensors, numLoops);

    // Detects sparse annotations and translate the per-dimension sparsity
    // information for all tensors to loop indices in the kernel.
    if (!op.hasSparseSemantics())
      return failure();
    findSparseAnnotations(op, merger.sparse());

    // Accept only single, dense result.
    if (op.getNumOutputs() != 1 ||
        std::any_of(merger.sparse().back().begin(),
                    merger.sparse().back().end(), [](bool b) { return b; }))
      return failure();

    // Computes a topologically sorted iteration graph to ensure
    // tensors are visited in natural index order. Fails on cycles.
    // This assumes that higher-level passes have already put the
    // tensors in each tensor expression in a feasible order.
    // TODO: try again without *dense* constraints on failure or
    //       even try to insert sparse reorderings to resolve cycles
    std::vector<unsigned> topSort;
    if (!computeIterationGraph(op, topSort))
      return failure();

    // Finds the terminating yield statement and builds the tensor
    // expression for the Linalg operation in SSA form.
    auto &region = op.region();
    if (!llvm::hasSingleElement(region))
      return failure(); // single block only
    Operation *yield = region.front().getTerminator();
    Optional<unsigned> exp = buildTensorExp(merger, op, yield->getOperand(0));
    if (!exp.hasValue())
      return failure(); // build failure

    // Recursively generates code.
    CodeGen codegen(numTensors, numLoops);
    genBuffers(merger, codegen, rewriter, op);
    genStmt(merger, codegen, rewriter, op, topSort, exp.getValue(), 0);
    Value result =
        rewriter.create<TensorLoadOp>(op.getLoc(), codegen.buffers.back());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

/// Populates the given patterns list with rewriting rules required for
/// the sparsification of linear algebra operations.
void mlir::linalg::populateSparsificationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<GenericOpSparsifier>(context);
}
