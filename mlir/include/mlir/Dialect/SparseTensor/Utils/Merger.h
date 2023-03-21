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
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include <optional>

namespace mlir {
namespace sparse_tensor {

// TODO: These type aliases currently only serve to make the code more
// self-documenting, however because they are not type-checked they can
// do nothing to prevent mixups.  We should really change them from mere
// aliases to actual struct definitions, so that we can type-check them.

/// Tensor identifiers.  The valid set of identifiers is defined by the
/// first argument passed to the `Merger` ctor.
using TensorId = unsigned;

/// Loop identifiers.  The valid set of identifiers is defined by the
/// second two arguments to the `Merger` ctor.
///
/// These identifiers serve as proxies for the `$dim` argument to
/// `linalg::IndexOp`, however the numerical value of a `LoopId` should
/// not necessarily be equated with the numerical value of the corresponding
/// `$dim` argument.  The `$dim` arguments are De Bruijn indices: that
/// is, they identify the loop which binds the loop-variable by counting
/// the enclosing loops from innermost to outermost, starting from zero.
/// Whereas `LoopId` are considered to be arbitrary names for identifying
/// loops; since the `Merger` does not care about the actual ordering of
/// loops, and leaves it up to the `LoopEmitter` to specify the actual
/// loop ordering (`LoopOrd`).
///
/// TODO: Despite the above claim that `$dim` and `LoopId` need not be
/// numerically equal, some code in the `Merger` class does equate them
/// (e.g., `buildTensorExp`).  So we need to explicate the exact relationship
/// between `$dim`, `LoopId`, and `LoopOrd`; especially with regards to their
/// providence.  If `LoopId` really is supposed to be equated with `$dim`,
/// then we should change the name to `LoopIdx` or similar, to capture the
/// fact that its numerical value is not invariant when entering/exiting
/// loops (unlike `TensorId`, `ExprId`, `LatPointId`, and `LatSetId` which
/// are invariant identifiers).
using LoopId = unsigned;

/// A compressed representation of `std::pair<TensorId, LoopId>`.
/// The compression scheme is such that this also serves as an index
/// into the bitvector stored in `LatPoint` (since that bitvector is
/// just the implementation for a set of `TensorLoopId` values).
using TensorLoopId = unsigned;

/// `TensorExp` identifiers.  These are allocated by `Merger::addExp`,
/// and serve as unique identifiers for the corresponding `TensorExp` object.
using ExprId = unsigned;

/// `LatPoint` identifiers.  These are allocated by `Merger::addLat`,
/// and serve as unique identifiers for the corresponding `LatPoint` object.
using LatPointId = unsigned;

/// `LatSet` identifiers.  These are allocated by `Merger::addSet` (and
/// by other methods calling that one), and serve as unique identifiers
/// for the corresponding `SmallVector<LatPointId>` object.
using LatSetId = unsigned;

/// A constant serving as the canonically invalid identifier, regardless
/// of the identifier type.
static constexpr unsigned kInvalidId = -1u;

/// Children subexpressions of tensor operations.
struct Children {
  ExprId e0;
  ExprId e1;
};

/// Tensor expression. Represents a MLIR expression in tensor index notation.
struct TensorExp {
  enum class Kind;

  // The `x` parameter has different types depending on the value of the
  // `k` parameter.  The correspondences are:
  // * `kTensor`    -> `TensorId`
  // * `kInvariant` -> `kInvalidId`
  // * `kLoopVar`   -> `LoopId`
  // * else         -> `ExprId`
  //
  // The `y`, `v`, and `op` parameters either must or must not be
  // `kInvalidId`/`nullptr`, depending on the value of the `k` parameter;
  // however, they have uniform C++ types regardless of the value of `k`.
  TensorExp(Kind k, unsigned x, ExprId y, Value v, Operation *op);

  /// Tensor expression kind.
  Kind kind;

  union {
    /// `kTensor` expressions simply have a tensor identifier.
    TensorId tensor;

    /// `kLoopVar` expressions simply have a loop identifier.
    LoopId loop;

    /// All other expressions hold the `ExprId`s of their children.
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

/// Tensor expression kind.
///
/// The `kLoopVar` leaf kind is for representing `linalg::IndexOp`.
/// That is, its argument is a `LoopId` identifying the loop-variable
/// in question, and its value will be the current iteration's value
/// of that loop-variable.  See the `LoopId` documentation for more details.
//
// TODO: Modify this definition so that the numeric values already encode
// the `ExpArity` (while extending the notion of "arity" to include not
// just the number of `ExprId` children the node has, but also whether the
// node has a `Value` and/or `Operation*`).  Doing this will avoid needing
// to enumerate all the kinds in `getExpArity` and in the `TensorExp` ctor,
// and should help clean up a few other places as well.
enum class TensorExp::Kind {
  // Leaf.
  kTensor = 0,
  kInvariant,
  kLoopVar,
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

/// Lattice point.  Each lattice point consists of a formal conjunction
/// of `TensorLoopId`s, together with the identifier of the corresponding
/// tensor expression.  The formal conjunction is represented as a set of
/// `TensorLoopId`, where that set is implemented as a `BitVector`.
struct LatPoint {
  /// Construct the lattice point from a given set of `TensorLoopId`s.
  LatPoint(const BitVector &bits, ExprId e);

  /// Construct a lattice point with `(t,i)` as the only `TensorLoopId`,
  /// where `(t,i) < (numTensors,numLoops)`.
  LatPoint(unsigned numTensors, unsigned numLoops, TensorId t, LoopId i,
           ExprId e);

  /// Conjunction of all `TensorLoopId`s involved in the tensor expression.
  BitVector bits;

  /// Simplified conjunction of `TensorLoopId` as bitvector.  This
  /// represents a simplified condition under which this tensor expression
  /// must execute. Pre-computed during codegen to avoid repeated eval.
  BitVector simple;

  /// Identifier of the tensor expression.
  ExprId exp;
};

/// A class to handle all iteration lattice operations. This class abstracts
/// away from some implementation details of storing iteration lattices and
/// tensor expressions. This allows for fine-tuning performance characteristics
/// independently from the basic algorithm if bottlenecks are identified.
class Merger {
public:
  /// Constructs a merger for the given number of tensors, native loops, and
  /// filter loops. The user supplies the number of tensors involved in the
  /// kernel, with the last tensor in this set denoting the output tensor.
  /// The merger adds an additional synthetic tensor at the end of this set
  /// to represent all invariant expressions in the kernel.
  ///
  /// In addition to natives loops (which are specified by the GenericOp),
  /// extra filter loops are needed in order to handle affine expressions on
  /// sparse levels.  E.g., (d0, d1, d2) => (d0 + d1, d2), a naive
  /// implementation of the filter loop could be generated as
  ///
  /// for (const auto c0 : coordinates[0]) {
  ///   if (c0 == d0 + d1) {
  ///      generated_code;
  ///   }
  /// }
  ///
  /// to filter out coordinates that are not equal to the affine expression.
  //
  // TODO: we want to make the filter loop more efficient in the future,
  // e.g., by avoiding scanning the full list of stored coordinates (keeping
  // the last position in ordered list) or even apply binary search to find
  // the coordinate.
  //
  // TODO: would be cleaner to understand/document if the first argument
  // gave the number of input tensors, instead of the current number of
  // input+output tensors.
  Merger(unsigned numInputOutputTensors, unsigned numNativeLoops,
         unsigned numFilterLoops);

  /// Constructs a new tensor expression, and returns its identifier.
  /// The type of the `e0` argument varies according to the value of the
  /// `k` argument, as described by the `TensorExp` ctor.
  ExprId addExp(TensorExp::Kind k, unsigned e0, ExprId e1 = kInvalidId,
                Value v = Value(), Operation *op = nullptr);
  ExprId addExp(TensorExp::Kind k, ExprId e, Value v, Operation *op = nullptr) {
    return addExp(k, e, kInvalidId, v, op);
  }
  ExprId addExp(TensorExp::Kind k, Value v, Operation *op = nullptr) {
    return addExp(k, kInvalidId, kInvalidId, v, op);
  }

  /// Constructs a new iteration lattice point, and returns its identifier.
  LatPointId addLat(TensorId t, LoopId i, ExprId e);

  /// Constructs a new (initially empty) set, and returns its identifier.
  LatSetId addSet();

  /// Computes a single conjunction of two lattice points by taking the "union"
  /// of `LoopId` (effectively constructing a larger "intersection" of those
  /// loops) with a newly constructed tensor (sub)expression of given kind.
  /// Returns the identifier of the new lattice point.
  LatPointId conjLat(TensorExp::Kind kind, LatPointId p0, LatPointId p1,
                     Operation *op = nullptr);

  /// Conjunctive merge of two lattice sets: `(s0 /\_op s1)`.
  /// Returns the identifier of the new set.
  LatSetId conjSet(TensorExp::Kind kind, LatSetId s0, LatSetId s1,
                   Operation *op = nullptr);

  /// Disjunctive merge of two lattice sets: `(s0 /\_op s1, s0, s1)`.
  /// Returns the identifier of the new set.
  LatSetId disjSet(TensorExp::Kind kind, LatSetId s0, LatSetId s1,
                   Operation *op = nullptr);

  /// Disjunctive merge of two lattice sets with custom handling of the
  /// overlap, left, and right regions.  Any region may be left missing
  /// in the output.  Returns the identifier of the new set.
  LatSetId combiSet(TensorExp::Kind kind, LatSetId s0, LatSetId s1,
                    Operation *orig, bool includeLeft, TensorExp::Kind ltrans,
                    Operation *opleft, bool includeRight,
                    TensorExp::Kind rtrans, Operation *opright);

  /// Maps the unary operator over the lattice set of the operand, i.e. each
  /// lattice point on an expression E is simply copied over, but with OP E
  /// as new expression. Returns the identifier of the new set.
  LatSetId mapSet(TensorExp::Kind kind, LatSetId s, Value v = Value(),
                  Operation *op = nullptr);

  /// Optimizes the iteration lattice points in the given set. This
  /// method should be called right before code generation to avoid
  /// generating redundant loops and conditions.
  LatSetId optimizeSet(LatSetId s);

  /// Simplifies the conditions in a conjunction of a given lattice point
  /// within the given set using just two basic rules:
  /// (1) multiple dense conditions are reduced to single dense, and
  /// (2) a *singleton* sparse/dense is reduced to sparse/random access.
  BitVector simplifyCond(LatSetId s, LatPointId p);

  /// Returns true if p0 > p1.
  bool latGT(LatPointId p0, LatPointId p1) const;

  /// Returns true if p0 and p1 only differ in dense.
  bool onlyDenseDiff(LatPointId p0, LatPointId p1) const;

  /// Gets the tensor-identifier of the `TensorLoopId`.
  TensorId tensor(TensorLoopId b) const { return b % numTensors; }
  /// Gets the loop-identifier of the `TensorLoopId`.
  LoopId loop(TensorLoopId b) const { return b / numTensors; }

  /// Get the total number of tensors (including the output-tensor and
  /// synthetic-tensor).  The result is given the type `TensorId` since
  /// the result is primarily used as an upper bound for `TensorId`s.
  TensorId getNumTensors() const { return numTensors; }

  /// Get the total number of loops (native loops + filter loops).
  /// The result is given the type `LoopId` since the result will
  /// generally be used as a for-loop upper bound.
  LoopId getNumLoops() const { return numLoops; }
  /// Get the number of native loops.  The result is given the type
  /// `LoopId` since the result will generally be used as a for-loop
  /// upper bound.
  LoopId getNumNativeLoops() const { return numNativeLoops; }
  /// Get the number of filter loops.  The result is given the type
  /// `LoopId` since the result will generally be used as a for-loop
  /// upper bound.
  LoopId getNumFilterLoops() const { return numLoops - numNativeLoops; }
  /// Get the identifier of the first filter-loop.
  LoopId getStartingFilterLoopId() const { return getNumNativeLoops(); }

  /// Returns true if `b` is the `i`th loop of the output tensor.
  bool isOutTensor(TensorLoopId b, LoopId i) const {
    assert(i < numLoops);
    return b == numTensors * i + outTensor;
  }

  /// Get the output tensor's identifier.
  TensorId getOutTensorID() const { return outTensor; }
  /// Get the synthetic tensor's identifier (used for all invariant
  /// tensor expressions).
  TensorId getSynTensorID() const { return syntheticTensor; }

  bool isFilterLoop(LoopId i) const {
    assert(i < numLoops);
    return i >= numNativeLoops;
  }

  /// Returns true if the expression is `(kTensor t)`.
  bool expIsTensor(ExprId e, TensorId t) const {
    return tensorExps[e].kind == TensorExp::Kind::kTensor &&
           tensorExps[e].tensor == t;
  }

  /// Returns true if the expression contains the tensor as an operand.
  bool expContainsTensor(ExprId e, TensorId t) const;

  /// Returns true if the expression contains a negation on output tensor.
  /// I.e., `- outTensor` or `exp - outputTensor`
  /// NOTE: this is an trivial tests in that it does not handle recursive
  /// negation, i.e., it returns true when the expression is `-(-tensor)`.
  bool hasNegateOnOut(ExprId e) const;

  /// Returns true if given tensor iterates *only* in the given tensor
  /// expression. For the output tensor, this defines a "simply dynamic"
  /// operation [Bik96]. For instance: a(i) *= 2.0 or a(i) += a(i) for
  /// sparse vector a.
  bool isSingleCondition(TensorId t, ExprId e) const;

  /// Returns true if any `TensorLoopId` in the bitvector corresponds
  /// to sparse level-type.
  bool hasAnySparse(const BitVector &bits) const;

  /// Returns true if bits contains a dependent index reduction condition on
  /// sparse levels.
  bool hasSparseIdxReduction(const BitVector &bits) const;

  /// Gets the level-type of the `t`th tensor on `i`th loop.
  DimLevelType getDimLevelType(TensorId t, LoopId i) const {
    assert(t < numTensors && i < numLoops);
    return lvlTypes[t][i];
  }

  /// Gets the level-type of the TensorLoopId.
  DimLevelType getDimLevelType(TensorLoopId b) const {
    return getDimLevelType(tensor(b), loop(b));
  }

  /// Gets the loop identifier for the `lvl`th level of the `t`th tensor.
  std::optional<LoopId> getLoopId(TensorId t, Level lvl) const {
    assert(t < numTensors && lvl < lvlToLoop[t].size());
    return lvlToLoop[t][lvl];
  }

  /// Gets the level number of the the `t`th tensor on `i`th loop.
  std::optional<Level> getLvl(TensorId t, LoopId i) const {
    assert(t < numTensors && i < numLoops);
    return loopToLvl[t][i];
  }
  std::optional<Level> getLvl(TensorLoopId b) const {
    return getLvl(tensor(b), loop(b));
  }

  /// Sets the level number and level-type of the `t`th tensor on
  /// `i`th loop.
  void setLevelAndType(TensorId t, LoopId i, Level lvl, DimLevelType dlt) {
    assert(t < numTensors && i < numLoops && lvl < lvlToLoop[t].size() &&
           isValidDLT(dlt));
    lvlTypes[t][i] = dlt;
    loopToLvl[t][i] = lvl;
    lvlToLoop[t][lvl] = i;
  }

  /// Iterates over a set of `TensorLoopId`s, invoking the callback
  /// for each `TensorLoopId` and passing it the corresponding tensor
  /// identifier, level, and level-type, following with a boolean value
  /// indicating whether it is a dependent index reduction loop condition.
  void foreachTensorLoopId(
      LatPointId p, function_ref<void(TensorLoopId, TensorId,
                                      std::optional<Level>, DimLevelType, bool)>
                        callback) {
    for (const TensorLoopId b : latPoints[p].bits.set_bits()) {
      TensorId t = tensor(b);
      if (isLvlWithNonTrivialIdxExp(b)) {
        // This must be an undefined level.
        assert(!getLvl(b).has_value());
        // Slice the tid along the dependent level to iterate current loop.
        callback(b, t, loopToDependencies[loop(b)][t], getDimLevelType(b),
                 /*isIdxReduc=*/true);
      } else {
        callback(b, t, getLvl(b), getDimLevelType(b), /*isIdxReduc=*/false);
      }
    }
  }

  /// Sets whether the output tensor is sparse or not.
  void setHasSparseOut(bool s) { hasSparseOut = s; }

  /// Establishes the two-way map that i <-> <t, lvl>.
  void setLoopDependentTensorLevel(LoopId i, TensorId t, Level lvl) {
    assert(lvl < numLoops);
    loopToDependencies[i][t] = lvl;
    levelToDependentIdx[t][lvl].push_back(i);
  }

  /// Whether the loop has dependent slice.
  bool hasDependentLvl(LoopId i, TensorId tid) {
    return loopToDependencies[i][tid].has_value();
  }

  /// Returns the list of loop indices which appear in the non-trivial index
  /// expression on t_l, e.g., A[i+j] => {i, j}
  std::vector<LoopId> &getDependentLoops(TensorId t, Level lvl) {
    return levelToDependentIdx[t][lvl];
  }

  /// Returns the defining [tid, lvl] for the loop.
  std::pair<TensorId, Level> getLoopDefiningLvl(LoopId i) const {
    return loopBounds[i];
  }

  /// Checks whether the TensorLoopId represents a tensor level with
  /// non-trivial index expression on it.
  bool isLvlWithNonTrivialIdxExp(TensorLoopId b) const {
    return loopToDependencies[loop(b)][tensor(b)].has_value();
  }

  /// Convenience getters to immediately access the stored nodes.
  /// Typically it is inadvisible to keep the reference around, as in
  /// `TensorExpr &te = merger.exp(e)`, since insertions into the merger
  /// may cause data movement and invalidate the underlying memory address.
  TensorExp &exp(ExprId e) { return tensorExps[e]; }
  LatPoint &lat(LatPointId p) { return latPoints[p]; }
  SmallVector<LatPointId> &set(LatSetId s) { return latSets[s]; }

#ifndef NDEBUG
  /// Print methods (for debugging).
  void dumpExp(ExprId e) const;
  void dumpLat(LatPointId p) const;
  void dumpSet(LatSetId s) const;
  void dumpBits(const BitVector &bits) const;
#endif

  /// Builds the iteration lattices in a bottom-up traversal given the
  /// remaining tensor (sub)expression and the next loop in the iteration
  /// graph.  Returns the identifier of the root set.
  LatSetId buildLattices(ExprId e, LoopId i);

  /// Builds a tensor expression from the given Linalg operation.
  /// On success, returns the identifier of the root expression.
  std::optional<ExprId> buildTensorExpFromLinalg(linalg::GenericOp op);

  /// Rebuilds SSA format from a tensor expression.
  Value buildExp(RewriterBase &rewriter, Location loc, ExprId e, Value v0,
                 Value v1) const;

private:
  /// Private helpers.
  bool maybeZero(ExprId e) const;
  bool isInvariant(ExprId e) const;
  Type inferType(ExprId e, Value src) const;

  /// Traverses the SSA tree (possibly a DAG) to build a tensor expression.
  std::optional<ExprId> buildTensorExp(linalg::GenericOp op, Value v);

  /// Merger data structures.
  const TensorId outTensor;
  const TensorId syntheticTensor;
  const unsigned numTensors;
  const unsigned numNativeLoops;
  const unsigned numLoops;
  bool hasSparseOut;

  // Below we use `std::vector` for things which have a priori fixed
  // sizes, whereas we use `llvm::SmallVector` for things with variable
  // size.  Do beware that these two classes differ in the semantics of
  // `operator[]`: `SmallVector` performs OOB checks, whereas `std::vector`
  // does not.

  // Map that converts pair<TensorId, LoopId> to the corresponding
  // level-type.
  std::vector<std::vector<DimLevelType>> lvlTypes;

  // Map that converts pair<TensorId, LoopId> to the corresponding
  // level.
  std::vector<std::vector<std::optional<Level>>> loopToLvl;

  // Map that converts pair<TensorId, Level> to the corresponding LoopId.
  std::vector<std::vector<std::optional<LoopId>>> lvlToLoop;

  // Map from a loop to its dependencies if any.
  // The dependencies of a loop is a set of (tensor, level) pairs.
  // It is currently only set for non-trivial index expressions.
  // E.g., A[i+j] => i and j will have dependencies {A0} to indicate that
  // i and j are used in the non-trivial index expression on A0.
  std::vector<std::vector<std::optional<Level>>> loopToDependencies;
  // The inverse map of ldxToDependencies from tensor level -> dependent loop
  // E.g., A[i+j], we have A0 => {i, j}, to indicate that A0 uses both {i, j}
  // to compute its indices.
  std::vector<std::vector<std::vector<LoopId>>> levelToDependentIdx;

  // Map from a loop to the [tid, lvl] pair that defines the loop boundary.
  std::vector<std::pair<TensorId, Level>> loopBounds;

  llvm::SmallVector<TensorExp> tensorExps;
  llvm::SmallVector<LatPoint> latPoints;
  llvm::SmallVector<SmallVector<LatPointId>> latSets;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_
