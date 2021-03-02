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
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

enum class Kind { kTensor, kInvariant, kMulF, kMulI, kAddF, kAddI };
enum class Dim { kSparse, kDense, kUndef };

/// Tensor expression. Represents a MLIR expression in tensor index notation.
/// For tensors, e0 denotes the tensor index. For invariants, the IR value is
/// stored directly. For binary operations, e0 and e1 denote the index of the
/// children tensor expressions.
struct TensorExp {
  TensorExp(Kind k, unsigned x, unsigned y, Value v)
      : kind(k), e0(x), e1(y), val(v) {
    assert((kind == Kind::kTensor && e0 != -1u && e1 == -1u && !val) ||
           (kind == Kind::kInvariant && e0 == -1u && e1 == -1u && val) ||
           (kind >= Kind::kMulF && e0 != -1u && e1 != -1u && !val));
  }
  Kind kind;
  /// Indices of children expression(s).
  unsigned e0;
  unsigned e1;
  /// Direct link to IR for an invariant. During code generation,
  /// field is used to cache "hoisted" loop invariant tensor loads.
  Value val;
};

/// Lattice point. Each lattice point consists of a conjunction of tensor
/// loop indices (encoded in a bitvector) and the index of the corresponding
/// tensor expression.
struct LatPoint {
  LatPoint(unsigned n, unsigned e, unsigned b) : bits(n, false), exp(e) {
    bits.set(b);
  }
  LatPoint(const llvm::BitVector &b, unsigned e) : bits(b), exp(e) {}
  /// Conjunction of tensor loop indices as bitvector. This represents
  /// all indices involved in the tensor expression
  llvm::BitVector bits;
  /// Simplified conjunction of tensor loop indices as bitvector. This
  /// represents a simplified condition under which this tensor expression
  /// must execute. Pre-computed during codegen to avoid repeated eval.
  llvm::BitVector simple;
  /// Index of the tensor expresssion.
  unsigned exp;
};

/// A class to handle all iteration lattice operations. This class abstracts
/// away from some implementation details of storing iteration lattices and
/// tensor expressions. This allows for fine-tuning performance characteristics
/// independently from the basic algorithm if bottlenecks are identified.
class Merger {
public:
  /// Constructs a merger for the given number of tensors and loops. The
  /// user supplies the number of tensors involved in the kernel, with the
  /// last tensor in this set denoting the output tensor. The merger adds an
  /// additional synthetic tensor at the end of this set to represent all
  /// invariant expressions in the kernel.
  Merger(unsigned t, unsigned l)
      : outTensor(t - 1), numTensors(t + 1), numLoops(l),
        dims(t + 1, std::vector<Dim>(l, Dim::kUndef)) {}

  /// Adds a tensor expression. Returns its index.
  unsigned addExp(Kind k, unsigned e0, unsigned e1 = -1u, Value v = Value()) {
    unsigned e = tensorExps.size();
    tensorExps.push_back(TensorExp(k, e0, e1, v));
    return e;
  }
  unsigned addExp(Kind k, Value v) { return addExp(k, -1u, -1u, v); }

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
  /// of loop indices (effectively constructing a larger "intersection" of those
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

  /// Conjunctive merge of two lattice sets L0 and L1 is conjunction of
  /// cartesian product. Returns the index of the new set.
  unsigned takeConj(Kind kind, unsigned s0, unsigned s1) {
    unsigned s = addSet();
    for (unsigned p0 : latSets[s0])
      for (unsigned p1 : latSets[s1])
        latSets[s].push_back(conjLatPoint(kind, p0, p1));
    return s;
  }

  /// Disjunctive merge of two lattice sets L0 and L1 is (L0 /\_op L1, L0, L1).
  /// Returns the index of the new set.
  unsigned takeDisj(Kind kind, unsigned s0, unsigned s1) {
    unsigned s = takeConj(kind, s0, s1);
    for (unsigned p : latSets[s0])
      latSets[s].push_back(p);
    for (unsigned p : latSets[s1])
      latSets[s].push_back(p);
    return s;
  }

  /// Optimizes the iteration lattice points in the given set. This
  /// method should be called right before code generation to avoid
  /// generating redundant loops and conditions.
  unsigned optimizeSet(unsigned s0) {
    unsigned s = addSet();
    assert(latSets[s0].size() != 0);
    unsigned p0 = latSets[s0][0];
    for (unsigned p1 : latSets[s0]) {
      bool add = true;
      if (p0 != p1) {
        // Is this a straightforward copy?
        unsigned e = latPoints[p1].exp;
        if (exp(e).kind == Kind::kTensor && exp(e).e0 == outTensor)
          continue;
        // Conjunction already covered?
        for (unsigned p2 : latSets[s]) {
          assert(!latGT(p1, p2)); // Lj => Li would be bad
          if (onlyDenseDiff(p2, p1)) {
            add = false;
            break;
          }
        }
        assert(!add || latGT(p0, p1));
      }
      if (add)
        latSets[s].push_back(p1);
    }
    for (unsigned p : latSets[s])
      latPoints[p].simple = simplifyCond(s, p);
    return s;
  }

  /// Simplifies the conditions in a conjunction of a given lattice point
  /// within the given set using just two basic rules:
  /// (1) multiple dense conditions are reduced to single dense, and
  /// (2) a *singleton* sparse/dense is reduced to sparse/random access.
  llvm::BitVector simplifyCond(unsigned s, unsigned p0) {
    // First determine if this lattice point is a *singleton*, i.e.,
    // the last point in a lattice, no other is less than this one.
    bool isSingleton = true;
    for (unsigned p1 : latSets[s]) {
      if (p0 != p1 && latGT(p0, p1)) {
        isSingleton = false;
        break;
      }
    }
    // Now apply the two basic rules.
    llvm::BitVector simple = latPoints[p0].bits;
    bool reset = isSingleton && hasAnyDimOf(simple, Dim::kSparse);
    for (unsigned b = 0, be = simple.size(); b < be; b++) {
      if (simple[b] && !isDim(b, Dim::kSparse)) {
        if (reset)
          simple.reset(b);
        reset = true;
      }
    }
    return simple;
  }

  /// Returns true if Li > Lj.
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

  /// Returns true if Li and Lj only differ in dense.
  bool onlyDenseDiff(unsigned i, unsigned j) {
    llvm::BitVector tmp = latPoints[j].bits;
    tmp ^= latPoints[i].bits;
    return !hasAnyDimOf(tmp, Dim::kSparse);
  }

  /// Bit translation.
  unsigned tensor(unsigned b) const { return b % numTensors; }
  unsigned index(unsigned b) const { return b / numTensors; }

  /// Returns true if bit corresponds to queried dim.
  bool isDim(unsigned b, Dim d) const { return isDim(tensor(b), index(b), d); }

  /// Returns true if tensor access at given index has queried dim.
  bool isDim(unsigned t, unsigned i, Dim d) const {
    assert(t < numTensors && i < numLoops);
    return dims[t][i] == d;
  }

  /// Returns true if any set bit corresponds to queried dim.
  bool hasAnyDimOf(const llvm::BitVector &bits, Dim d) const {
    for (unsigned b = 0, be = bits.size(); b < be; b++)
      if (bits[b] && isDim(b, d))
        return true;
    return false;
  }

  /// Returns true if tensor has any sparse dimension.
  bool isSparseTensor(unsigned t) const {
    return llvm::any_of(dims[t], [](Dim d) { return d == Dim::kSparse; });
  }

  /// Setter
  void setDim(unsigned t, unsigned i, Dim d) { dims[t][i] = d; }

  /// Getters.
  TensorExp &exp(unsigned e) { return tensorExps[e]; }
  LatPoint &lat(unsigned l) { return latPoints[l]; }
  SmallVector<unsigned, 16> &set(unsigned s) { return latSets[s]; }

private:
  const unsigned outTensor;
  const unsigned numTensors;
  const unsigned numLoops;

  std::vector<std::vector<Dim>> dims;
  llvm::SmallVector<TensorExp, 32> tensorExps;
  llvm::SmallVector<LatPoint, 16> latPoints;
  llvm::SmallVector<SmallVector<unsigned, 16>, 8> latSets;
};

// Code generation.
struct CodeGen {
  CodeGen(linalg::SparsificationOptions o, unsigned numTensors,
          unsigned numLoops)
      : options(o), loops(numLoops), sizes(numLoops), buffers(numTensors),
        pointers(numTensors, std::vector<Value>(numLoops)),
        indices(numTensors, std::vector<Value>(numLoops)),
        highs(numTensors, std::vector<Value>(numLoops)),
        pidxs(numTensors, std::vector<Value>(numLoops)),
        idxs(numTensors, std::vector<Value>(numLoops)), redExp(-1u), redVal(),
        curVecLength(1), curVecMask() {}
  /// Sparsification options.
  linalg::SparsificationOptions options;
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
  /// reduction are exhausted,  all inner loops can "scalarize" the reduction.
  // TODO: currently only done for (a chain of) innermost for-loops, where it
  // is most effective; we could generalize to more outer and while-loops.
  unsigned redExp;
  Value redVal;
  // Current vector length and mask.
  unsigned curVecLength;
  Value curVecMask;
};

} // namespace

/// Helper method to inspect sparse annotations in the linalg operation.
/// Fills the per-dimension sparsity information for all tensors.
static void findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  unsigned numTensors = op.getNumShapedOperands();
  ArrayAttr sparseAttr = op.sparseAttr();
  for (unsigned t = 0; t < numTensors; t++) {
    auto map = op.getIndexingMap(t);
    auto dimAttr = sparseAttr[t].cast<ArrayAttr>();
    // For each tensor, we accept a per-dimension Sparse or Dense annotation.
    // This is translated to the loop index that indexes that dimension.
    unsigned rank = op.getShapedType(t).getRank();
    for (unsigned d = 0; d < rank; d++) {
      unsigned idx = map.getDimPosition(d);
      if (isSparseDim(dimAttr[d])) {
        merger.setDim(t, idx, Dim::kSparse);
      } else {
        assert(isDenseDim(dimAttr[d]));
        merger.setDim(t, idx, Dim::kDense);
      }
    }
  }
}

/// Returns true if tensor was set up with sparse storage scheme.
static bool linkedSparse(linalg::GenericOp op, unsigned tensor) {
  if (tensor < op.getNumInputs())
    return isa_and_nonnull<linalg::SparseTensorFromPointerOp>(
        op.getInput(tensor).getDefiningOp());
  return false;
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
static bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort,
                                  bool sparseOnly) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = op.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));

  // Iterate over the indexing maps of every tensor in the tensor expression.
  unsigned numTensors = op.getNumShapedOperands();
  for (unsigned t = 0; t < numTensors; t++) {
    auto map = op.getIndexingMap(t);
    assert(map.getNumDims() == n);
    // Skip dense tensor constraints when sparse only is requested.
    if (sparseOnly && !merger.isSparseTensor(t) && !linkedSparse(op, t))
      continue;
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
  topSort.clear();
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
  if (auto arg = val.dyn_cast<BlockArgument>()) {
    unsigned argN = arg.getArgNumber();
    if (arg.getOwner()->getParentOp() == op) {
      // Any parameter of the generic op is considered a tensor,
      // indexed by the implicit loop bounds.
      auto map = op.getIndexingMap(argN);
      if (map.isProjectedPermutation())
        return merger.addExp(Kind::kTensor, argN);
      // Cannot handle (yet).
      return None;
    }
    // Any parameter of a higher op is invariant.
    return merger.addExp(Kind::kInvariant, val);
  }
  Operation *def = val.getDefiningOp();
  if (def->getBlock() != &op.region().front()) {
    // Something defined outside is invariant.
    return merger.addExp(Kind::kInvariant, val);
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
    // Either the index is really used in the tensor expression, or it is
    // set to the undefined index in that dimension. An invariant expression
    // is set to a synthetic tensor with undefined indices only.
    unsigned s = merger.addSet();
    unsigned t =
        kind == Kind::kTensor ? merger.exp(exp).e0 : op.getNumShapedOperands();
    merger.set(s).push_back(merger.addLat(t, idx, exp));
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
  llvm_unreachable("unexpected expression kind");
}

/// Maps sparse integer option to actual integral storage type.
static Type genIntType(PatternRewriter &rewriter, linalg::SparseIntType tp) {
  switch (tp) {
  case linalg::SparseIntType::kNative:
    return rewriter.getIndexType();
  case linalg::SparseIntType::kI64:
    return rewriter.getIntegerType(64);
  case linalg::SparseIntType::kI32:
    return rewriter.getIntegerType(32);
  case linalg::SparseIntType::kI16:
    return rewriter.getIntegerType(16);
  case linalg::SparseIntType::kI8:
    return rewriter.getIntegerType(8);
  }
  llvm_unreachable("unexpected SparseIntType");
}

/// Generates buffer for the output tensor.
static Value genOutputBuffer(CodeGen &codegen, PatternRewriter &rewriter,
                             linalg::GenericOp op, MemRefType denseTp,
                             ArrayRef<Value> args) {
  Location loc = op.getLoc();
  Value tensor = op.getOutput(0);
  // The output tensor simply could materialize from the buffer that will
  // be generated for the tensor present in the outs() clause. This has
  // the major advantage that the sparse kernel only updates the nonzero
  // positions for the output tensor. Currently this results in functional,
  // but slightly imprecise IR, so it is put under an experimental option.
  if (codegen.options.fastOutput)
    return rewriter.create<TensorToMemrefOp>(loc, denseTp, tensor);
  // By default, a new buffer is allocated which is initialized to the
  // tensor defined in the outs() clause. This is always correct but
  // introduces a dense initialization component that may negatively
  // impact the running complexity of the sparse kernel.
  Value init = rewriter.create<TensorToMemrefOp>(loc, denseTp, tensor);
  Value alloc = rewriter.create<AllocOp>(loc, denseTp, args);
  rewriter.create<linalg::CopyOp>(loc, init, alloc);
  return alloc;
}

/// Local bufferization of all dense and sparse data structures.
/// This code enables testing the first prototype sparse compiler.
// TODO: replace this with a proliferated bufferization strategy
static void genBuffers(Merger &merger, CodeGen &codegen,
                       PatternRewriter &rewriter, linalg::GenericOp op) {
  Location loc = op.getLoc();
  unsigned numTensors = op.getNumShapedOperands();
  unsigned numInputs = op.getNumInputs();
  assert(numTensors == numInputs + 1);
  // For every tensor, find lower and upper bound on dimensions, set the
  // same bounds on loop indices, and obtain dense or sparse buffer(s).
  SmallVector<Value, 4> args;
  for (unsigned t = 0; t < numTensors; t++) {
    Value tensor = t < numInputs ? op.getInput(t) : op.getOutput(0);
    auto tensorType = op.getShapedType(t);
    auto shape = tensorType.getShape();
    auto map = op.getIndexingMap(t);
    // Scan all dimensions of current tensor.
    bool dense = !linkedSparse(op, t);
    args.clear();
    for (unsigned d = 0, rank = shape.size(); d < rank; d++) {
      unsigned i = map.getDimPosition(d);
      // Handle sparse storage schemes.
      if (merger.isDim(t, i, Dim::kSparse)) {
        dense = false;
        auto dynShape = {ShapedType::kDynamicSize};
        auto ptrTp = MemRefType::get(
            dynShape, genIntType(rewriter, codegen.options.ptrType));
        auto indTp = MemRefType::get(
            dynShape, genIntType(rewriter, codegen.options.indType));
        Value dim = rewriter.create<ConstantIndexOp>(loc, d);
        // Generate sparse primitives to obtains pointer and indices.
        codegen.pointers[t][i] =
            rewriter.create<linalg::SparseTensorToPointersMemRefOp>(
                loc, ptrTp, tensor, dim);
        codegen.indices[t][i] =
            rewriter.create<linalg::SparseTensorToIndicesMemRefOp>(loc, indTp,
                                                                   tensor, dim);
      }
      // Find lower and upper bound in current dimension.
      Value up;
      if (shape[d] == TensorType::kDynamicSize) {
        up = rewriter.create<DimOp>(loc, tensor, d);
        args.push_back(up);
      } else {
        up = rewriter.create<ConstantIndexOp>(loc, shape[d]);
      }
      codegen.sizes[i] = codegen.highs[t][i] = up;
    }
    // Perform the required bufferization. All dense inputs materialize
    // from the input tensor. The dense output tensor needs special
    // handling. Sparse inputs use a sparse primitive to obtain the values.
    if (dense) {
      auto denseTp = MemRefType::get(shape, tensorType.getElementType());
      if (t < numInputs)
        codegen.buffers[t] =
            rewriter.create<TensorToMemrefOp>(loc, denseTp, tensor);
      else
        codegen.buffers[t] =
            genOutputBuffer(codegen, rewriter, op, denseTp, args);
    } else {
      auto dynShape = {ShapedType::kDynamicSize};
      auto sparseTp = MemRefType::get(dynShape, tensorType.getElementType());
      codegen.buffers[t] =
          rewriter.create<linalg::SparseTensorToValuesMemRefOp>(loc, sparseTp,
                                                                tensor);
    }
  }
}

/// Constructs vector type from pointer.
static VectorType vectorType(CodeGen &codegen, Value ptr) {
  Type etp = ptr.getType().cast<MemRefType>().getElementType();
  return VectorType::get(codegen.curVecLength, etp);
}

/// Constructs vector iteration mask.
static Value genVectorMask(CodeGen &codegen, PatternRewriter &rewriter,
                           Value iv, Value lo, Value hi, Value step) {
  Location loc = iv.getLoc();
  VectorType mtp =
      VectorType::get(codegen.curVecLength, rewriter.getIntegerType(1));
  // Special case if the vector length evenly divides the trip count (for
  // example, "for i = 0, 128, 16"). A constant all-true mask is generated
  // so that all subsequent masked memory operations are immediately folded
  // into unconditional memory operations.
  IntegerAttr loInt, hiInt, stepInt;
  if (matchPattern(lo, m_Constant(&loInt)) &&
      matchPattern(hi, m_Constant(&hiInt)) &&
      matchPattern(step, m_Constant(&stepInt))) {
    if (((hiInt.getInt() - loInt.getInt()) % stepInt.getInt()) == 0)
      return rewriter.create<vector::ConstantMaskOp>(
          loc, mtp, rewriter.getI64ArrayAttr(codegen.curVecLength));
  }
  // Otherwise, generate a vector mask that avoids overrunning the upperbound
  // during vector execution. Here we rely on subsequent loop optimizations to
  // avoid executing the mask in all iterations, for example, by splitting the
  // loop into an unconditional vector loop and a scalar cleanup loop.
  Value end = rewriter.create<SubIOp>(loc, hi, iv);
  return rewriter.create<vector::CreateMaskOp>(loc, mtp, end);
}

/// Generates a vectorized load lhs = a[ind[lo:hi]] or lhs = a[lo:hi].
static Value genVectorLoad(CodeGen &codegen, PatternRewriter &rewriter,
                           Value ptr, ArrayRef<Value> args) {
  Location loc = ptr.getLoc();
  VectorType vtp = vectorType(codegen, ptr);
  Value pass = rewriter.create<ConstantOp>(loc, vtp, rewriter.getZeroAttr(vtp));
  if (args.back().getType().isa<VectorType>()) {
    SmallVector<Value, 4> scalarArgs(args.begin(), args.end());
    Value indexVec = args.back();
    scalarArgs.back() = rewriter.create<ConstantIndexOp>(loc, 0);
    return rewriter.create<vector::GatherOp>(
        loc, vtp, ptr, scalarArgs, indexVec, codegen.curVecMask, pass);
  }
  return rewriter.create<vector::MaskedLoadOp>(loc, vtp, ptr, args,
                                               codegen.curVecMask, pass);
}

/// Generates a vectorized store a[ind[lo:hi]] = rhs or a[lo:hi] = rhs.
static void genVectorStore(CodeGen &codegen, PatternRewriter &rewriter,
                           Value rhs, Value ptr, ArrayRef<Value> args) {
  Location loc = ptr.getLoc();
  if (args.back().getType().isa<VectorType>()) {
    SmallVector<Value, 4> scalarArgs(args.begin(), args.end());
    Value indexVec = args.back();
    scalarArgs.back() = rewriter.create<ConstantIndexOp>(loc, 0);
    rewriter.create<vector::ScatterOp>(loc, ptr, scalarArgs, indexVec,
                                       codegen.curVecMask, rhs);
    return;
  }
  rewriter.create<vector::MaskedStoreOp>(loc, ptr, args, codegen.curVecMask,
                                         rhs);
}

/// Generates a vectorized invariant. Here we rely on subsequent loop
/// optimizations to hoist the invariant broadcast out of the vector loop.
static Value genVectorInvariantValue(CodeGen &codegen,
                                     PatternRewriter &rewriter, Value val) {
  VectorType vtp = VectorType::get(codegen.curVecLength, val.getType());
  return rewriter.create<vector::BroadcastOp>(val.getLoc(), vtp, val);
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = merger.exp(exp).val;
  if (val) {
    if (codegen.curVecLength > 1 && !val.getType().isa<VectorType>())
      return genVectorInvariantValue(codegen, rewriter, val);
    return val;
  }
  // Actual load.
  SmallVector<Value, 4> args;
  unsigned tensor = merger.exp(exp).e0;
  auto map = op.getIndexingMap(tensor);
  bool sparse = linkedSparse(op, tensor);
  for (unsigned i = 0, m = map.getNumResults(); i < m; ++i) {
    unsigned idx = map.getDimPosition(i);
    args.push_back(codegen.loops[idx]); // universal dense index
    if (sparse || merger.isDim(tensor, idx, Dim::kSparse)) {
      sparse = true;
      args.clear();
      args.push_back(codegen.pidxs[tensor][idx]); // position index
    }
  }
  Location loc = op.getLoc();
  Value ptr = codegen.buffers[tensor];
  if (codegen.curVecLength > 1)
    return genVectorLoad(codegen, rewriter, ptr, args);
  return rewriter.create<LoadOp>(loc, ptr, args);
}

/// Generates a store on a dense tensor.
static void genTensorStore(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned tensor, Value rhs) {
  // Test if this is a scalarized reduction.
  unsigned lhs = op.getNumShapedOperands() - 1;
  if (lhs == tensor && codegen.redVal) {
    codegen.redVal = rhs;
    return;
  }
  // Actual store.
  SmallVector<Value, 4> args;
  auto map = op.getIndexingMap(tensor);
  for (unsigned i = 0, m = map.getNumResults(); i < m; ++i) {
    unsigned idx = map.getDimPosition(i);
    args.push_back(codegen.loops[idx]); // universal dense index
  }
  Location loc = op.getLoc();
  Value ptr = codegen.buffers[tensor];
  if (codegen.curVecLength > 1)
    genVectorStore(codegen, rewriter, rhs, ptr, args);
  else
    rewriter.create<StoreOp>(loc, rhs, ptr, args);
}

/// Generates a pointer/index load from the sparse storage scheme.
static Value genLoad(CodeGen &codegen, PatternRewriter &rewriter, Location loc,
                     Value ptr, Value s) {
  if (codegen.curVecLength > 1)
    return genVectorLoad(codegen, rewriter, ptr, {s});
  Value load = rewriter.create<LoadOp>(loc, ptr, s);
  return load.getType().isa<IndexType>()
             ? load
             : rewriter.create<IndexCastOp>(loc, load, rewriter.getIndexType());
}

/// Generates an invariant value.
static Value genInvariantValue(Merger &merger, CodeGen &codegen,
                               PatternRewriter &rewriter, unsigned exp) {
  Value val = merger.exp(exp).val;
  if (codegen.curVecLength > 1)
    return genVectorInvariantValue(codegen, rewriter, val);
  return val;
}

/// Generates an address computation "sz * p + i".
static Value genAddress(CodeGen &codegen, PatternRewriter &rewriter,
                        Location loc, Value size, Value p, Value i) {
  Value mul = rewriter.create<MulIOp>(loc, size, p);
  if (auto vtp = i.getType().dyn_cast<VectorType>()) {
    Value inv = rewriter.create<IndexCastOp>(loc, mul, vtp.getElementType());
    mul = genVectorInvariantValue(codegen, rewriter, inv);
  }
  return rewriter.create<AddIOp>(loc, mul, i);
}

/// Recursively generates tensor expression.
static Value genExp(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, unsigned exp) {
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, exp);
  else if (merger.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(merger, codegen, rewriter, exp);
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
  llvm_unreachable("unexpected expression kind");
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(Merger &merger, CodeGen &codegen,
                          PatternRewriter &rewriter, linalg::GenericOp op,
                          unsigned exp, unsigned ldx, bool hoist) {
  if (merger.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    unsigned tensor = merger.exp(exp).e0;
    auto map = op.getIndexingMap(tensor);
    for (unsigned i = 0, m = map.getNumResults(); i < m; ++i) {
      unsigned idx = map.getDimPosition(i);
      if (!codegen.loops[idx])
        return; // still in play
      else if (idx == ldx)
        atLevel = true;
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    unsigned lhs = op.getNumShapedOperands() - 1;
    if (lhs == tensor) {
      codegen.redExp = hoist ? exp : -1u;
    } else if (atLevel) {
      merger.exp(exp).val =
          hoist ? genTensorLoad(merger, codegen, rewriter, op, exp) : Value();
    }
  } else if (merger.exp(exp).kind != Kind::kInvariant) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    unsigned e0 = merger.exp(exp).e0;
    unsigned e1 = merger.exp(exp).e1;
    genInvariants(merger, codegen, rewriter, op, e0, ldx, hoist);
    genInvariants(merger, codegen, rewriter, op, e1, ldx, hoist);
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
  for (unsigned b = 0, be = inits.size(); b < be; b++) {
    if (inits[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      if (merger.isDim(b, Dim::kSparse)) {
        // Initialize sparse index.
        unsigned pat = at;
        for (; pat != 0; pat--) {
          if (codegen.pidxs[tensor][topSort[pat - 1]])
            break;
        }
        Value ptr = codegen.pointers[tensor][idx];
        Value one = rewriter.create<ConstantIndexOp>(loc, 1);
        Value p0 = (pat == 0) ? rewriter.create<ConstantIndexOp>(loc, 0)
                              : codegen.pidxs[tensor][topSort[pat - 1]];
        codegen.pidxs[tensor][idx] = genLoad(codegen, rewriter, loc, ptr, p0);
        Value p1 = rewriter.create<AddIOp>(loc, p0, one);
        codegen.highs[tensor][idx] = genLoad(codegen, rewriter, loc, ptr, p1);
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

/// Returns vectorization strategy. Any implicit inner loop in the Linalg
/// operation is a candidate. Whether it is actually converted to SIMD code
/// depends on the requested strategy.
static bool isVectorFor(CodeGen &codegen, bool isInner, bool isSparse) {
  switch (codegen.options.vectorizationStrategy) {
  case linalg::SparseVectorizationStrategy::kNone:
    return false;
  case linalg::SparseVectorizationStrategy::kDenseInnerLoop:
    return isInner && !isSparse;
  case linalg::SparseVectorizationStrategy::kAnyStorageInnerLoop:
    return isInner;
  }
  llvm_unreachable("unexpected vectorization strategy");
}

/// Returns parallelization strategy. Any implicit loop in the Linalg operation
/// that is marked "parallel" is a candidate. Whether it is actually converted
/// to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodeGen &codegen, bool isOuter, bool isReduction,
                          bool isSparse, bool isVector) {
  switch (codegen.options.parallelizationStrategy) {
  case linalg::SparseParallelizationStrategy::kNone:
    return false;
  case linalg::SparseParallelizationStrategy::kDenseOuterLoop:
    return isOuter && !isSparse && !isReduction && !isVector;
  case linalg::SparseParallelizationStrategy::kAnyStorageOuterLoop:
    return isOuter && !isReduction && !isVector;
  case linalg::SparseParallelizationStrategy::kDenseAnyLoop:
    return !isSparse && !isReduction && !isVector;
  case linalg::SparseParallelizationStrategy::kAnyStorageAnyLoop:
    return !isReduction && !isVector;
  }
  llvm_unreachable("unexpected parallelization strategy");
}

/// Generates a for-loop on a single index.
static Operation *genFor(Merger &merger, CodeGen &codegen,
                         PatternRewriter &rewriter, linalg::GenericOp op,
                         bool isOuter, bool isInner, unsigned idx,
                         llvm::BitVector &indices) {
  unsigned fb = indices.find_first();
  unsigned tensor = merger.tensor(fb);
  assert(idx == merger.index(fb));
  auto iteratorTypes = op.iterator_types().getValue();
  bool isReduction = linalg::isReductionIteratorType(iteratorTypes[idx]);
  bool isSparse = merger.isDim(fb, Dim::kSparse);
  bool isVector = isVectorFor(codegen, isInner, isSparse);
  bool isParallel =
      isParallelFor(codegen, isOuter, isReduction, isSparse, isVector);

  // Prepare vector length.
  if (isVector)
    codegen.curVecLength = codegen.options.vectorLength;

  // Loop bounds and increment.
  Location loc = op.getLoc();
  Value lo = isSparse ? codegen.pidxs[tensor][idx] : codegen.loops[idx];
  Value hi = isSparse ? codegen.highs[tensor][idx] : codegen.sizes[idx];
  Value step = rewriter.create<ConstantIndexOp>(loc, codegen.curVecLength);

  // Emit a parallel loop.
  if (isParallel) {
    assert(!isVector);
    scf::ParallelOp parOp = rewriter.create<scf::ParallelOp>(loc, lo, hi, step);
    if (isSparse)
      codegen.pidxs[tensor][idx] = parOp.getInductionVars()[0];
    else
      codegen.loops[idx] = parOp.getInductionVars()[0];
    rewriter.setInsertionPointToStart(parOp.getBody());
    return parOp;
  }

  // Emit a sequential loop, potentially with a scalarized reduction.
  bool scalarRed = isInner && codegen.redExp != -1u;
  SmallVector<Value, 4> operands;
  if (scalarRed) {
    Value load;
    if (codegen.redVal) {
      load = codegen.redVal; // chained with previous for-loop
    } else if (isVector) {
      // TODO: assumes + reductions for now
      VectorType vtp = vectorType(codegen, codegen.buffers[codegen.redExp]);
      load = rewriter.create<ConstantOp>(loc, vtp, rewriter.getZeroAttr(vtp));
    } else {
      load = genTensorLoad(merger, codegen, rewriter, op, codegen.redExp);
    }
    operands.push_back(load);
  }
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, lo, hi, step, operands);
  if (scalarRed) {
    codegen.redVal = merger.exp(codegen.redExp).val =
        forOp.getRegionIterArgs().front();
  }
  // Assign induction variable to sparse or dense index.
  Value iv = forOp.getInductionVar();
  if (isSparse)
    codegen.pidxs[tensor][idx] = iv;
  else
    codegen.loops[idx] = iv;
  rewriter.setInsertionPointToStart(forOp.getBody());
  // Share vector iteration mask between all subsequent loads/stores.
  if (isVector)
    codegen.curVecMask = genVectorMask(codegen, rewriter, iv, lo, hi, step);
  return forOp;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(Merger &merger, CodeGen &codegen,
                           PatternRewriter &rewriter, linalg::GenericOp op,
                           unsigned idx, bool needsUniv,
                           llvm::BitVector &indices) {
  SmallVector<Type, 4> types;
  SmallVector<Value, 4> operands;
  // Construct the while-loop with a parameter for each index.
  Type indexType = rewriter.getIndexType();
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      types.push_back(indexType);
      assert(codegen.pidxs[tensor][idx].getType().isa<IndexType>() &&
             "type mismatch for sparse index");
      operands.push_back(codegen.pidxs[tensor][idx]);
    }
  }
  if (needsUniv) {
    types.push_back(indexType);
    assert(codegen.loops[idx].getType().isa<IndexType>() &&
           "type mismatch for universal index");
    operands.push_back(codegen.loops[idx]);
  }
  Location loc = op.getLoc();
  scf::WhileOp whileOp = rewriter.create<scf::WhileOp>(loc, types, operands);
  Block *before = rewriter.createBlock(&whileOp.before(), {}, types);
  Block *after = rewriter.createBlock(&whileOp.after(), {}, types);

  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  rewriter.setInsertionPointToStart(&whileOp.before().front());
  Value cond;
  unsigned o = 0;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
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
  return whileOp;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(Merger &merger, CodeGen &codegen,
                          PatternRewriter &rewriter, linalg::GenericOp op,
                          std::vector<unsigned> &topSort, unsigned at,
                          bool needsUniv, llvm::BitVector &indices) {
  unsigned idx = topSort[at];
  if (indices.count() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == topSort.size() - 1;
    return genFor(merger, codegen, rewriter, op, isOuter, isInner, idx,
                  indices);
  }
  return genWhile(merger, codegen, rewriter, op, idx, needsUniv, indices);
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
    if (locals[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value ptr = codegen.indices[tensor][idx];
      Value s = codegen.pidxs[tensor][idx];
      Value load = genLoad(codegen, rewriter, loc, ptr, s);
      codegen.idxs[tensor][idx] = load;
      if (!needsUniv) {
        if (min) {
          Value cmp =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, load, min);
          min = rewriter.create<SelectOp>(loc, cmp, load, min);
        } else {
          min = load;
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
    if (locals[b] && merger.isDim(b, Dim::kDense)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      unsigned pat = at;
      for (; pat != 0; pat--)
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      Value p = (pat == 0) ? rewriter.create<ConstantIndexOp>(loc, 0)
                           : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = genAddress(
          codegen, rewriter, loc, codegen.sizes[idx], p, codegen.loops[idx]);
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
  for (unsigned b = 0, be = induction.size(); b < be; b++) {
    if (induction[b] && merger.isDim(b, Dim::kSparse)) {
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
  }
  if (needsUniv) {
    operands.push_back(rewriter.create<AddIOp>(loc, codegen.loops[idx], one));
    codegen.loops[idx] = results[o++];
  }
  assert(o == operands.size());
  rewriter.create<scf::YieldOp>(loc, operands);
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(Merger &merger, CodeGen &codegen,
                       PatternRewriter &rewriter, linalg::GenericOp op,
                       unsigned idx, llvm::BitVector &conditions) {
  Location loc = op.getLoc();
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (conditions[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value clause;
      if (merger.isDim(b, Dim::kSparse)) {
        Value op1 = codegen.idxs[tensor][idx];
        Value op2 = codegen.loops[idx];
        clause = rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, op1, op2);
      } else {
        clause = rewriter.create<ConstantIntOp>(loc, 1, 1); // true
      }
      cond = cond ? rewriter.create<AndOp>(loc, cond, clause) : clause;
    }
  }
  scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, cond, /*else*/ true);
  rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
  return ifOp;
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(Merger &merger, CodeGen &codegen, PatternRewriter &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == topSort.size()) {
    unsigned lhs = op.getNumShapedOperands() - 1;
    Value rhs = genExp(merger, codegen, rewriter, op, exp);
    genTensorStore(merger, codegen, rewriter, op, lhs, rhs);
    return;
  }
  assert(codegen.curVecLength == 1);

  // Construct iteration lattices for current loop index, with L0 at top.
  // Then emit initialization code for the loop sequence at this level.
  // We maintain the universal dense index if dense indices are still
  // in play for a non-singleton loop sequence.
  Location loc = op.getLoc();
  unsigned idx = topSort[at];
  unsigned lts = merger.optimizeSet(buildLattices(merger, op, exp, idx));
  unsigned lsize = merger.set(lts).size();
  assert(lsize != 0);
  unsigned l0 = merger.set(lts)[0];
  unsigned ldx = at == 0 ? -1u : topSort[at - 1];
  genInvariants(merger, codegen, rewriter, op, exp, ldx, /*hoist=*/true);
  bool needsUniv = false;
  if (genInit(merger, codegen, rewriter, op, topSort, at,
              merger.lat(l0).bits)) {
    // Maintain the universal index only if it is actually
    // consumed by a subsequent lattice point.
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = merger.set(lts)[i];
      if (!merger.hasAnyDimOf(merger.lat(li).simple, Dim::kSparse)) {
        needsUniv = true;
        break;
      }
    }
  }

  // Emit a loop for every lattice point L0 >= Li.
  for (unsigned i = 0; i < lsize; i++) {
    unsigned li = merger.set(lts)[i];

    // Emit loop.
    codegen.curVecLength = 1;
    llvm::BitVector indices = merger.lat(li).simple;
    Operation *loop =
        genLoop(merger, codegen, rewriter, op, topSort, at, needsUniv, indices);
    genLocals(merger, codegen, rewriter, op, topSort, at, needsUniv,
              merger.lat(li).bits);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    bool isWhile = dyn_cast<scf::WhileOp>(loop) != nullptr;
    for (unsigned j = 0; j < lsize; j++) {
      unsigned lj = merger.set(lts)[j];
      unsigned ej = merger.lat(lj).exp;
      if (li == lj || merger.latGT(li, lj)) {
        // Recurse into body of each branch.
        if (isWhile) {
          scf::IfOp ifOp =
              genIf(merger, codegen, rewriter, op, idx, merger.lat(lj).simple);
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
          rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
        } else {
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
        }
      }
    }

    // Wrap-up induction and restore insertion point.
    if (isWhile) {
      scf::WhileOp whileOp = cast<scf::WhileOp>(loop);
      rewriter.setInsertionPointToEnd(&whileOp.after().front());
      genWhileInduction(merger, codegen, rewriter, op, idx, needsUniv,
                        merger.lat(li).bits, whileOp.results());
    } else {
      needsUniv = false;
      if (codegen.redVal) {
        rewriter.create<scf::YieldOp>(loc, codegen.redVal);
        codegen.redVal = loop->getResult(0);
      }
    }
    rewriter.setInsertionPointAfter(loop);
  }

  // Wrap-up loop sequence.
  Value red = codegen.redVal;
  if (red) {
    codegen.redVal = merger.exp(codegen.redExp).val = Value(); // end chain
    unsigned lhs = op.getNumShapedOperands() - 1;
    if (codegen.curVecLength > 1) {
      codegen.curVecLength = 1;
      Value ld = genTensorLoad(merger, codegen, rewriter, op, codegen.redExp);
      red = rewriter.create<vector::ReductionOp>(
          loc, ld.getType(), rewriter.getStringAttr("add"), red, ld);
    }
    genTensorStore(merger, codegen, rewriter, op, lhs, red);
  }
  genInvariants(merger, codegen, rewriter, op, exp, ldx, /*hoist=*/false);
  codegen.loops[idx] = Value();
  codegen.curVecLength = 1;
}

namespace {

/// Sparse rewriting rule for generic Lingalg operation.
struct GenericOpSparsifier : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparsifier(MLIRContext *context, linalg::SparsificationOptions o)
      : OpRewritePattern<linalg::GenericOp>(context), options(o) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Detects sparse annotations and translate the per-dimension sparsity
    // information for all tensors to loop indices in the kernel.
    if (!op.hasSparseSemantics())
      return failure();
    assert(op.getNumOutputs() == 1);
    unsigned numTensors = op.getNumShapedOperands();
    unsigned numLoops = op.iterator_types().getValue().size();
    Merger merger(numTensors, numLoops);
    findSparseAnnotations(merger, op);

    // Computes a topologically sorted iteration graph to ensure
    // tensors are visited in natural index order. Fails on cycles.
    // This assumes that higher-level passes have already put the
    // tensors in each tensor expression in a feasible order.
    std::vector<unsigned> topSort;
    if (!computeIterationGraph(merger, op, topSort, /*sparseOnly=*/false) &&
        !computeIterationGraph(merger, op, topSort, /*sparseOnly=*/true))
      return failure();

    // Finds the terminating yield statement and builds the tensor
    // expression for the Linalg operation in SSA form.
    Operation *yield = op.region().front().getTerminator();
    Optional<unsigned> exp = buildTensorExp(merger, op, yield->getOperand(0));
    if (!exp.hasValue())
      return failure(); // build failure

    // Recursively generates code.
    CodeGen codegen(options, numTensors, numLoops);
    genBuffers(merger, codegen, rewriter, op);
    genStmt(merger, codegen, rewriter, op, topSort, exp.getValue(), 0);
    Value result =
        rewriter.create<TensorLoadOp>(op.getLoc(), codegen.buffers.back());
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Options to control sparse code generation.
  linalg::SparsificationOptions options;
};

} // namespace

/// Populates the given patterns list with rewriting rules required for
/// the sparsification of linear algebra operations.
void linalg::populateSparsificationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    const SparsificationOptions &options) {
  patterns.insert<GenericOpSparsifier>(context, options);
}
