//===- Merger.h - Utilities for defining lattices ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for dealing with iteration lattices.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_
#define MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"

namespace mlir {
namespace sparse_tensor {

/// Dimension level type for a tensor (undef means index does not appear).
enum Dim { kSparse, kDense, kSingle, kUndef };

/// Tensor expression kind.
enum Kind {
  // Leaf.
  kTensor = 0,
  kInvariant,
  // Unary operations.
  kAbsF,
  kCeilF,
  kFloorF,
  kNegF,
  kNegI,
  kTruncF,
  kExtF,
  kCastFS, // signed
  kCastFU, // unsigned
  kCastSF, // signed
  kCastUF, // unsigned
  kCastS,  // signed
  kCastU,  // unsigned
  kTruncI,
  kBitCast,
  // Binary operations.
  kMulF,
  kMulI,
  kDivF,
  kDivS, // signed
  kDivU, // unsigned
  kAddF,
  kAddI,
  kSubF,
  kSubI,
  kAndI,
  kOrI,
  kXorI,
  kShrS, // signed
  kShrU, // unsigned
  kShlI,
};

/// Children subexpressions of tensor operations.
struct Children {
  unsigned e0;
  unsigned e1;
};

/// Tensor expression. Represents a MLIR expression in tensor index notation.
struct TensorExp {
  TensorExp(Kind k, unsigned x, unsigned y, Value v);

  /// Tensor expression kind.
  Kind kind;

  union {
    /// Expressions representing tensors simply have a tensor number.
    unsigned tensor;

    /// Tensor operations hold the indices of their children.
    Children children;
  };

  /// Direct link to IR for an invariant or the destination value (to
  /// infer destination type) of a cast operation During code generation,
  /// this field may be used to cache "hoisted" loop invariant tensor loads.
  Value val;
};

/// Lattice point. Each lattice point consists of a conjunction of tensor
/// loop indices (encoded in a bitvector) and the index of the corresponding
/// tensor expression.
struct LatPoint {
  LatPoint(unsigned n, unsigned e, unsigned b);
  LatPoint(const llvm::BitVector &b, unsigned e);

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
      : outTensor(t - 1), syntheticTensor(t), numTensors(t + 1), numLoops(l),
        hasSparseOut(false), dims(t + 1, std::vector<Dim>(l, Dim::kUndef)) {}

  /// Adds a tensor expression. Returns its index.
  unsigned addExp(Kind k, unsigned e0, unsigned e1 = -1u, Value v = Value());
  unsigned addExp(Kind k, unsigned e, Value v) { return addExp(k, e, -1u, v); }
  unsigned addExp(Kind k, Value v) { return addExp(k, -1u, -1u, v); }

  /// Adds an iteration lattice point. Returns its index.
  unsigned addLat(unsigned t, unsigned i, unsigned e);

  /// Adds a new, initially empty, set. Returns its index.
  unsigned addSet();

  /// Computes a single conjunction of two lattice points by taking the "union"
  /// of loop indices (effectively constructing a larger "intersection" of those
  /// indices) with a newly constructed tensor (sub)expression of given kind.
  /// Returns the index of the new lattice point.
  unsigned conjLatPoint(Kind kind, unsigned p0, unsigned p1);

  /// Conjunctive merge of two lattice sets L0 and L1 is conjunction of
  /// cartesian product. Returns the index of the new set.
  unsigned takeConj(Kind kind, unsigned s0, unsigned s1);

  /// Disjunctive merge of two lattice sets L0 and L1 is (L0 /\_op L1, L0, L1).
  /// Returns the index of the new set.
  unsigned takeDisj(Kind kind, unsigned s0, unsigned s1);

  /// Maps the unary operator over the lattice set of the operand, i.e. each
  /// lattice point on an expression E is simply copied over, but with OP E
  /// as new expression. Returns the index of the new set.
  unsigned mapSet(Kind kind, unsigned s0, Value v = Value());

  /// Optimizes the iteration lattice points in the given set. This
  /// method should be called right before code generation to avoid
  /// generating redundant loops and conditions.
  unsigned optimizeSet(unsigned s0);

  /// Simplifies the conditions in a conjunction of a given lattice point
  /// within the given set using just two basic rules:
  /// (1) multiple dense conditions are reduced to single dense, and
  /// (2) a *singleton* sparse/dense is reduced to sparse/random access.
  llvm::BitVector simplifyCond(unsigned s0, unsigned p0);

  /// Returns true if Li > Lj.
  bool latGT(unsigned i, unsigned j) const;

  /// Returns true if Li and Lj only differ in dense.
  bool onlyDenseDiff(unsigned i, unsigned j);

  /// Bit translation.
  unsigned tensor(unsigned b) const { return b % numTensors; }
  unsigned index(unsigned b) const { return b / numTensors; }

  /// Returns true if bit corresponds to queried dim.
  bool isDim(unsigned b, Dim d) const { return isDim(tensor(b), index(b), d); }

  /// Returns true if bit corresponds to index of output tensor.
  bool isOutTensor(unsigned b, unsigned i) const {
    return tensor(b) == outTensor && index(b) == i;
  }

  /// Returns true if tensor access at given index has queried dim.
  bool isDim(unsigned t, unsigned i, Dim d) const {
    assert(t < numTensors && i < numLoops);
    return dims[t][i] == d;
  }

  /// Returns true if any set bit corresponds to queried dim.
  bool hasAnyDimOf(const llvm::BitVector &bits, Dim d) const;

  /// Returns true if given tensor iterates *only* in the given tensor
  /// expression. For the output tensor, this defines a "simply dynamic"
  /// operation [Bik96]. For instance: a(i) *= 2.0 or a(i) += a(i) for
  /// sparse vector a.
  bool isSingleCondition(unsigned t, unsigned e) const;

  /// Dimension setter.
  void setDim(unsigned t, unsigned i, Dim d) { dims[t][i] = d; }

  // Has sparse output tensor setter.
  void setHasSparseOut(bool s) { hasSparseOut = s; }

  /// Convenience getters to immediately access the stored nodes.
  /// Typically it is inadvisible to keep the reference around, as in
  /// "TensorExpr &te = merger.exp(e))", since insertions into the merger
  /// may cause data movement and invalidate the underlying memory address.
  TensorExp &exp(unsigned e) { return tensorExps[e]; }
  LatPoint &lat(unsigned l) { return latPoints[l]; }
  SmallVector<unsigned, 16> &set(unsigned s) { return latSets[s]; }

#ifndef NDEBUG
  /// Print methods (for debugging).
  void dumpExp(unsigned e) const;
  void dumpLat(unsigned p) const;
  void dumpSet(unsigned s) const;
  void dumpBits(const llvm::BitVector &bits) const;
#endif

  /// Builds the iteration lattices in a bottom-up traversal given the remaining
  /// tensor (sub)expression and the next loop index in the iteration graph.
  /// Returns index of the root expression.
  unsigned buildLattices(unsigned e, unsigned i);

  /// Builds a tensor expression from the given Linalg operation.
  /// Returns index of the root expression on success.
  Optional<unsigned> buildTensorExpFromLinalg(linalg::GenericOp op);

  /// Rebuilds SSA format from a tensor expression.
  Value buildExp(PatternRewriter &rewriter, Location loc, unsigned e, Value v0,
                 Value v1);

private:
  /// Private helpers.
  bool maybeZero(unsigned e) const;
  bool isInvariant(unsigned e) const;
  Type inferType(unsigned e, Value src);

  /// Traverses the SSA tree (possibly a DAG) to build a tensor expression.
  Optional<unsigned> buildTensorExp(linalg::GenericOp op, Value v);

  /// Merger data structures.
  const unsigned outTensor;
  const unsigned syntheticTensor;
  const unsigned numTensors;
  const unsigned numLoops;
  bool hasSparseOut;
  std::vector<std::vector<Dim>> dims;
  llvm::SmallVector<TensorExp, 32> tensorExps;
  llvm::SmallVector<LatPoint, 16> latPoints;
  llvm::SmallVector<SmallVector<unsigned, 16>, 8> latSets;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_
