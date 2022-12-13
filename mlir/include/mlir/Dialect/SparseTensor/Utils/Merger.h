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
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"

namespace mlir {
namespace sparse_tensor {

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
  /// Constructs a merger for the given number of tensors, native loops, and
  /// filter loops. The user supplies the number of tensors involved in the
  /// kernel, with the last tensor in this set denoting the output tensor. The
  /// merger adds an additional synthetic tensor at the end of this set to
  /// represent all invariant expressions in the kernel.
  /// In addition to natives
  /// loops (which are specified by the GenericOp), extra filter loops are
  /// needed in order to handle affine expressions on sparse dimensions.
  /// E.g., (d0, d1, d2) => (d0 + d1, d2), a naive implementation of the filter
  /// loop could be generated as:
  /// for (coord : sparse_dim[0])
  ///   if (coord == d0 + d1) {
  ///      generated_code;
  ///   }
  /// }
  /// to filter out coordinates that are not equal to the affine expression
  /// result.
  /// TODO: we want to make the filter loop more efficient in the future, e.g.,
  /// by avoiding scanning the full stored index sparse (keeping the last
  /// position in ordered list) or even apply binary search to find the index.
  Merger(unsigned t, unsigned l, unsigned fl)
      : outTensor(t - 1), syntheticTensor(t), numTensors(t + 1),
        numNativeLoops(l), numLoops(l + fl), hasSparseOut(false),
        dimTypes(numTensors,
                 std::vector<DimLevelType>(numLoops, DimLevelType::Undef)),
        loopIdxToDim(numTensors,
                     std::vector<Optional<unsigned>>(numLoops, std::nullopt)),
        dimToLoopIdx(numTensors,
                     std::vector<Optional<unsigned>>(numLoops, std::nullopt)) {}

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

  /// Bit translation (get tensor ID).
  unsigned tensor(unsigned b) const { return b % numTensors; }
  /// Bit translation (get loop index).
  unsigned index(unsigned b) const { return b / numTensors; }

  /// Get the number of total loops (native loops + filter loops).
  unsigned getNumLoops() const { return numLoops; }
  /// Get the number of native loops.
  unsigned getNumNativeLoops() const { return numNativeLoops; }
  /// Get the number of filter loops.
  unsigned getNumFilterLoops() const { return numLoops - numNativeLoops; }
  /// Get the starting filter loop index.
  unsigned getFilterLoopStartingIdx() const { return getNumNativeLoops(); }

  /// Returns true if bit corresponds to index of output tensor.
  bool isOutTensor(unsigned b, unsigned i) const {
    return tensor(b) == outTensor && index(b) == i;
  }

  /// Gets tensor ID for the output tensor.
  unsigned getOutTensorID() const { return outTensor; }
  /// Gets tensor ID for the synthetic tensor (used for all invariant tensor
  /// expressions).
  unsigned getSynTensorID() const { return syntheticTensor; }

  bool isFilterLoop(unsigned ldx) const {
    assert(ldx < numLoops);
    return ldx >= numNativeLoops;
  }

  /// Returns true if the expression contains the `t` as an operand.
  bool expContainsTensor(unsigned e, unsigned t) const;

  /// Returns true if the expression contains a negation on output tensor.
  /// I.e., `- outTensor` or `exp - outputTensor`
  /// NOTE: this is an trivial tests in that it does not handle recursive
  /// negation, i.e., it returns true when the expression is `-(-tensor)`.
  bool hasNegateOnOut(unsigned e) const;

  /// Returns true if given tensor iterates *only* in the given tensor
  /// expression. For the output tensor, this defines a "simply dynamic"
  /// operation [Bik96]. For instance: a(i) *= 2.0 or a(i) += a(i) for
  /// sparse vector a.
  bool isSingleCondition(unsigned t, unsigned e) const;

  /// Returns true if any set bit corresponds to sparse dimension level type.
  bool hasAnySparse(const BitVector &bits) const;

  /// Gets the dimension level type of the `t`th tensor on `i`th loop.
  DimLevelType getDimLevelType(unsigned t, unsigned i) const {
    assert(t < numTensors && i < numLoops);
    return dimTypes[t][i];
  }

  /// Gets the dimension level type of `b`.
  DimLevelType getDimLevelType(unsigned b) const {
    return getDimLevelType(tensor(b), index(b));
  }

  Optional<unsigned> getLoopIdx(unsigned t, unsigned dim) const {
    assert(t < numTensors && dim < numLoops);
    return dimToLoopIdx[t][dim];
  }

  /// Gets the dimension number of the the `t`th tensor on `i`th loop.
  Optional<unsigned> getDimNum(unsigned t, unsigned i) const {
    assert(t < numTensors && i < numLoops);
    return loopIdxToDim[t][i];
  }

  /// Gets the dimension number of `b`.
  Optional<unsigned> getDimNum(unsigned b) const {
    return getDimNum(tensor(b), index(b));
  }

  /// Sets the dimension and dimension level type of the `t`th tensor on `i`th
  /// loop.
  void setDimAndDimLevelType(unsigned t, unsigned i, unsigned dim,
                             DimLevelType dlt) {
    assert(isValidDLT(dlt));
    dimTypes[t][i] = dlt;
    loopIdxToDim[t][i] = dim;
    assert(dim < numLoops);
    dimToLoopIdx[t][dim] = i;
  }

  // Iterates the bits of a lattice, for each set bit, converts it into the
  // corresponding tensor dimension and invokes the callback.
  void foreachTidDimPairInBits(
      const BitVector &bits,
      function_ref<void(unsigned b, unsigned tid, Optional<unsigned> dim,
                        DimLevelType dlt)>
          cb) {
    for (unsigned b : bits.set_bits())
      cb(b, tensor(b), getDimNum(b), getDimLevelType(b));
  }

  // Has sparse output tensor setter.
  void setHasSparseOut(bool s) { hasSparseOut = s; }

  /// Convenience getters to immediately access the stored nodes.
  /// Typically it is inadvisible to keep the reference around, as in
  /// "TensorExpr &te = merger.exp(e))", since insertions into the merger
  /// may cause data movement and invalidate the underlying memory address.
  TensorExp &exp(unsigned e) { return tensorExps[e]; }
  LatPoint &lat(unsigned l) { return latPoints[l]; }
  SmallVector<unsigned> &set(unsigned s) { return latSets[s]; }

#ifndef NDEBUG
  /// Print methods (for debugging).
  void dumpExp(unsigned e) const;
  void dumpLat(unsigned p) const;
  void dumpSet(unsigned s) const;
  void dumpBits(const BitVector &bits) const;
#endif

  /// Builds the iteration lattices in a bottom-up traversal given the
  /// remaining tensor (sub)expression and the next loop index in the
  /// iteration graph. Returns index of the root expression.
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
  const unsigned numNativeLoops;
  const unsigned numLoops;
  bool hasSparseOut;
  // Map that converts pair<tensor id, loop id> to the corresponding dimension
  // level type.
  std::vector<std::vector<DimLevelType>> dimTypes;
  // Map that converts pair<tensor id, loop id> to the corresponding
  // dimension.
  std::vector<std::vector<Optional<unsigned>>> loopIdxToDim;
  // Map that converts pair<tensor id, dim> to the corresponding loop id.
  std::vector<std::vector<Optional<unsigned>>> dimToLoopIdx;
  llvm::SmallVector<TensorExp> tensorExps;
  llvm::SmallVector<LatPoint> latPoints;
  llvm::SmallVector<SmallVector<unsigned>> latSets;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_
