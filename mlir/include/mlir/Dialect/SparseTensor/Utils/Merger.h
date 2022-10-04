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
enum class DimLvlType { kDense, kCompressed, kSingleton, kUndef };

/// Per-dimension level format (type and properties). Dense and undefined
/// level types should always be marked ordered and unique.
struct DimLevelFormat {
  DimLevelFormat(DimLvlType tp, bool o = true, bool u = true)
      : levelType(tp), isOrdered(o), isUnique(u) {
    assert((tp == DimLvlType::kCompressed || tp == DimLvlType::kSingleton) ||
           (o && u));
  }
  DimLvlType levelType;
  bool isOrdered;
  bool isUnique;
};

/// Tensor expression kind.
enum Kind {
  // Leaf.
  kTensor = 0,
  kInvariant,
  kIndex,
  // Unary operations.
  kAbsF,
  kAbsC,
  kAbsI,
  kCeilF,
  kFloorF,
  kSqrtF,
  kSqrtC,
  kExpm1F,
  kExpm1C,
  kLog1pF,
  kLog1pC,
  kSinF,
  kSinC,
  kTanhF,
  kTanhC,
  kNegF,
  kNegC,
  kNegI,
  kTruncF,
  kExtF,
  kCastFS, // signed
  kCastFU, // unsigned
  kCastSF, // signed
  kCastUF, // unsigned
  kCastS,  // signed
  kCastU,  // unsigned
  kCastIdx,
  kTruncI,
  kCIm, // complex.im
  kCRe, // complex.re
  kBitCast,
  kBinaryBranch, // semiring unary branch created from a binary op
  kUnary,        // semiring unary op
  kSelect,       // custom selection criteria
  // Binary operations.
  kMulF,
  kMulC,
  kMulI,
  kDivF,
  kDivC, // complex
  kDivS, // signed
  kDivU, // unsigned
  kAddF,
  kAddC,
  kAddI,
  kSubF,
  kSubC,
  kSubI,
  kAndI,
  kOrI,
  kXorI,
  kShrS, // signed
  kShrU, // unsigned
  kShlI,
  kBinary, // semiring binary op
  kReduce, // semiring reduction op
};

/// Children subexpressions of tensor operations.
struct Children {
  unsigned e0;
  unsigned e1;
};

/// Tensor expression. Represents a MLIR expression in tensor index notation.
struct TensorExp {
  TensorExp(Kind k, unsigned x, unsigned y, Value v, Operation *operation);

  /// Tensor expression kind.
  Kind kind;

  union {
    /// Expressions representing tensors simply have a tensor number.
    unsigned tensor;

    /// Indices hold the index number.
    unsigned index;

    /// Tensor operations hold the indices of their children.
    Children children;
  };

  /// Direct link to IR for an invariant or the destination value (to
  /// infer destination type) of a cast operation During code generation,
  /// this field may be used to cache "hoisted" loop invariant tensor loads.
  Value val;

  /// Code blocks used by semirings. For the case of kUnary, kBinary, kReduce,
  /// and kSelect, this holds the original operation with all regions. For
  /// kBinaryBranch, this holds the YieldOp for the left or right half
  /// to be merged into a nested scf loop.
  Operation *op;
};

/// Lattice point. Each lattice point consists of a conjunction of tensor
/// loop indices (encoded in a bitvector) and the index of the corresponding
/// tensor expression.
struct LatPoint {
  LatPoint(unsigned n, unsigned e, unsigned b);
  LatPoint(const BitVector &b, unsigned e);

  /// Conjunction of tensor loop indices as bitvector. This represents
  /// all indices involved in the tensor expression
  BitVector bits;

  /// Simplified conjunction of tensor loop indices as bitvector. This
  /// represents a simplified condition under which this tensor expression
  /// must execute. Pre-computed during codegen to avoid repeated eval.
  BitVector simple;

  /// Index of the tensor expression.
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
        hasSparseOut(false),
        dims(t + 1, std::vector<DimLevelFormat>(
                        l, DimLevelFormat(DimLvlType::kUndef))) {}

  /// Adds a tensor expression. Returns its index.
  unsigned addExp(Kind k, unsigned e0, unsigned e1 = -1u, Value v = Value(),
                  Operation *op = nullptr);
  unsigned addExp(Kind k, unsigned e, Value v, Operation *op = nullptr) {
    return addExp(k, e, -1u, v, op);
  }
  unsigned addExp(Kind k, Value v, Operation *op = nullptr) {
    return addExp(k, -1u, -1u, v, op);
  }

  /// Adds an iteration lattice point. Returns its index.
  unsigned addLat(unsigned t, unsigned i, unsigned e);

  /// Adds a new, initially empty, set. Returns its index.
  unsigned addSet();

  /// Computes a single conjunction of two lattice points by taking the "union"
  /// of loop indices (effectively constructing a larger "intersection" of those
  /// indices) with a newly constructed tensor (sub)expression of given kind.
  /// Returns the index of the new lattice point.
  unsigned conjLatPoint(Kind kind, unsigned p0, unsigned p1,
                        Operation *op = nullptr);

  /// Conjunctive merge of two lattice sets L0 and L1 is conjunction of
  /// cartesian product. Returns the index of the new set.
  unsigned takeConj(Kind kind, unsigned s0, unsigned s1,
                    Operation *op = nullptr);

  /// Disjunctive merge of two lattice sets L0 and L1 is (L0 /\_op L1, L0, L1).
  /// Returns the index of the new set.
  unsigned takeDisj(Kind kind, unsigned s0, unsigned s1,
                    Operation *op = nullptr);

  /// Disjunctive merge of two lattice sets L0 and L1 with custom handling of
  /// the overlap, left, and right regions. Any region may be left missing in
  /// the output. Returns the index of the new set.
  unsigned takeCombi(Kind kind, unsigned s0, unsigned s1, Operation *orig,
                     bool includeLeft, Kind ltrans, Operation *opleft,
                     bool includeRight, Kind rtrans, Operation *opright);

  /// Maps the unary operator over the lattice set of the operand, i.e. each
  /// lattice point on an expression E is simply copied over, but with OP E
  /// as new expression. Returns the index of the new set.
  unsigned mapSet(Kind kind, unsigned s0, Value v = Value(),
                  Operation *op = nullptr);

  /// Optimizes the iteration lattice points in the given set. This
  /// method should be called right before code generation to avoid
  /// generating redundant loops and conditions.
  unsigned optimizeSet(unsigned s0);

  /// Simplifies the conditions in a conjunction of a given lattice point
  /// within the given set using just two basic rules:
  /// (1) multiple dense conditions are reduced to single dense, and
  /// (2) a *singleton* sparse/dense is reduced to sparse/random access.
  BitVector simplifyCond(unsigned s0, unsigned p0);

  /// Returns true if Li > Lj.
  bool latGT(unsigned i, unsigned j) const;

  /// Returns true if Li and Lj only differ in dense.
  bool onlyDenseDiff(unsigned i, unsigned j);

  /// Bit translation.
  unsigned tensor(unsigned b) const { return b % numTensors; }
  unsigned index(unsigned b) const { return b / numTensors; }

  /// Returns true if bit corresponds to index of output tensor.
  bool isOutTensor(unsigned b, unsigned i) const {
    return tensor(b) == outTensor && index(b) == i;
  }

  /// Returns true if given tensor iterates *only* in the given tensor
  /// expression. For the output tensor, this defines a "simply dynamic"
  /// operation [Bik96]. For instance: a(i) *= 2.0 or a(i) += a(i) for
  /// sparse vector a.
  bool isSingleCondition(unsigned t, unsigned e) const;

  /// Returns true if bit corresponds to given dimension level type.
  bool isDimLevelType(unsigned b, DimLvlType tp) const {
    return isDimLevelType(tensor(b), index(b), tp);
  }

  /// Returns true if tensor access at index has given dimension level type.
  bool isDimLevelType(unsigned t, unsigned i, DimLvlType tp) const {
    return getDimLevelFormat(t, i).levelType == tp;
  }

  /// Returns true if any set bit corresponds to sparse dimension level type.
  bool hasAnySparse(const BitVector &bits) const;

  /// Dimension level format getter.
  DimLevelFormat getDimLevelFormat(unsigned t, unsigned i) const {
    assert(t < numTensors && i < numLoops);
    return dims[t][i];
  }

  /// Dimension level format setter.
  void setDimLevelFormat(unsigned t, unsigned i, DimLevelFormat d) {
    dims[t][i] = d;
  }

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
  void dumpBits(const BitVector &bits) const;
#endif

  /// Builds the iteration lattices in a bottom-up traversal given the remaining
  /// tensor (sub)expression and the next loop index in the iteration graph.
  /// Returns index of the root expression.
  unsigned buildLattices(unsigned e, unsigned i);

  /// Builds a tensor expression from the given Linalg operation.
  /// Returns index of the root expression on success.
  Optional<unsigned> buildTensorExpFromLinalg(linalg::GenericOp op);

  /// Rebuilds SSA format from a tensor expression.
  Value buildExp(RewriterBase &rewriter, Location loc, unsigned e, Value v0,
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
  std::vector<std::vector<DimLevelFormat>> dims;
  llvm::SmallVector<TensorExp, 32> tensorExps;
  llvm::SmallVector<LatPoint, 16> latPoints;
  llvm::SmallVector<SmallVector<unsigned, 16>, 8> latSets;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_
