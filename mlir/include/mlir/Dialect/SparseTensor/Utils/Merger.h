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

namespace detail {
/// A constant serving as the canonically invalid identifier,
/// regardless of the identifier type.
static constexpr unsigned kInvalidId = -1u;
} // namespace detail

/// Tensor identifiers, chosen to be the `BlockArgument::getArgNumber`
/// of the value passed to `Merger::buildTensorExp`.
using TensorId = unsigned;

/// Loop identifiers.
using LoopId = unsigned;

/// A compressed representation of `std::pair<TensorId, LoopId>`.
/// The compression scheme is such that this also serves as an index
/// into the bitvector stored in `LatPoint` (since that bitvector is
/// just the implementation for a set of `TensorLoopId` values).
using TensorLoopId = unsigned;

/// `TensorExp` identifiers. These are allocated by `Merger::addExp`,
/// and serve as unique identifiers for the corresponding `TensorExp` object.
using ExprId = unsigned;

/// `LatPoint` identifiers. These are allocated by `Merger::addLat`,
/// and serve as unique identifiers for the corresponding `LatPoint` object.
using LatPointId = unsigned;

/// `LatSet` identifiers.  These are allocated by `Merger::addSet` (and
/// by other methods calling that one), and serve as unique identifiers
/// for the corresponding `SmallVector<LatPointId>` object.
using LatSetId = unsigned;

/// A pair of level and its corresponding LevelType of a tensor.
using LvlLTPair = std::pair<Level, LevelType>;

/// A pair of loop id and its coefficients. E.g., for affine expression in the
/// affine map `2 * d0`, loop id = 0, coefficient = 2.
using LoopCoeffPair = std::pair<LoopId, unsigned>;

/// Tensor expression. Represents an MLIR expression in tensor index notation.
struct TensorExp final {
  enum class Kind;

  /// Child subexpressions for non-leaf expressions.
  struct Children final {
    ExprId e0;
    ExprId e1;
  };

  /// The `x` parameter has different types depending on the value of the
  /// `k` parameter.  The correspondences are:
  /// * `kTensor`    -> `TensorId`
  /// * `kInvariant` -> `kInvalidId`
  /// * `kLoopVar`   -> `LoopId`
  /// * else         -> `ExprId`
  ///
  /// The `y`, `v`, and `op` parameters either must or must not be
  /// `kInvalidId`/`nullptr`, depending on the value of the `k` parameter;
  /// however, they have uniform C++ types regardless of the value of `k`.
  TensorExp(Kind k, unsigned x, ExprId y, Value v, Operation *op, Attribute a);

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
  ///
  /// Or the actual operation that we can not sparsify but having all dense
  /// operands for kDenseOp.
  Operation *op;

  /// An optional attribute that is required to determine the semantics of the
  /// operations. E.g., CmpPredicateAttr for CmpI/CmpF operations.
  Attribute attr;
};

/// Tensor expression kind.
///
/// The `kLoopVar` leaf kind is for representing `linalg::IndexOp`.
/// That is, its argument is a `LoopId` identifying the loop-variable
/// in question, and its value will be the current iteration's value.
/// The `kSynZero` leaf kind is for representing a synthetic zero value,
/// which can be introduced when sparsifying operations like `arith::cmp`
/// to generate `arith::cmp %lhs, %syn_zero` when the rhs operand is absent.
enum class TensorExp::Kind {
  // Leaf.
  kTensor = 0,
  kSynZero,
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
  kCmpI,
  kCmpF,
  kShrS, // signed
  kShrU, // unsigned
  kShlI,
  kBinary,  // semiring binary op
  kReduce,  // semiring reduction op
  kDenseOp, // special category of operations requiring all dense operands
};

/// Lattice point.  Each lattice point consists of a formal conjunction
/// of `TensorLoopId`s, together with the identifier of the corresponding
/// tensor expression.  The formal conjunction is represented as a set of
/// `TensorLoopId`, where that set is implemented as a `BitVector`.
struct LatPoint final {
  /// Construct a lattice point with the empty set of `TensorLoopId`s.
  LatPoint(unsigned size, ExprId e) : bits(size, false), exp(e) {}

  /// Construct a lattice point from the given set of `TensorLoopId`s.
  LatPoint(const BitVector &bits, ExprId e) : bits(bits), exp(e) {}

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
  /// Constructs a merger for the given number of tensors and loops. The user
  /// supplies the number of tensors involved in the kernel, with the last
  /// tensor in this set denoting the output tensor. The merger adds an
  /// additional synthetic tensor at the end of this set to represent all
  /// invariant expressions in the kernel.
  ///
  /// The maxLvlRank specifies the max level rank of all inputs/output tensors.
  /// It is used to pre-allocate sufficient memory for internal storage.
  Merger(unsigned numInputOutputTensors, unsigned numLoops,
         unsigned maxLvlRank);

  //
  // Constructing valid tensor and loop identifiers.
  //

  /// Safely converts the argument to a tensor identifier.
  constexpr TensorId makeTensorId(unsigned t) const {
    assert(isValidTensorId(t));
    return t;
  }

  /// Safely converts the argument to a loop identifier.
  constexpr LoopId makeLoopId(unsigned i) const {
    assert(isValidLoopId(i));
    return i;
  }

  /// Safely converts the arguments to a pair of (tensor,loop) identifiers.
  constexpr TensorLoopId makeTensorLoopId(unsigned t, unsigned i) const {
    assert(isValidTensorId(t) && isValidLoopId(i));
    return numTensors * i + t;
  }

  //
  // Allocating new expressions, points, and sets.
  //

  /// Constructs a new tensor expression, and returns its identifier.
  ExprId addTensorExp(TensorId t);
  /// Constructs a new loop-variable expression, and returns its identifier.
  ExprId addLoopVarExp(LoopId i);
  /// Constructs a new invariant expression, and returns its identifier.
  ExprId addInvariantExp(Value v);
  /// Constructs a new synthetic zero expression.
  ExprId addSynZeroExp();
  /// Constructs a new unary or binary expression, and returns its identifier.
  ExprId addExp(TensorExp::Kind k, ExprId e0, ExprId e1 = detail::kInvalidId,
                Operation *op = nullptr, Attribute attr = nullptr);
  /// Constructs a new sesquinary expression, and returns its identifier.
  /// Currently no sesquinary `Kind` allows specifying the `op`, but we
  /// allow it anyways because `mapSet` is designed to allow it.
  ExprId addExp(TensorExp::Kind k, ExprId e, Value v, Operation *op = nullptr,
                Attribute attr = nullptr);

  /// Constructs a new iteration lattice point, and returns its identifier.
  LatPointId addLat(TensorId t, LoopId i, ExprId e);
  LatPointId addLat(const BitVector &bits, ExprId e);

  /// Constructs a new (initially empty) set, and returns its identifier.
  LatSetId addSet();

  /// Computes a single conjunction of two lattice points by taking the "union"
  /// of `LoopId` (effectively constructing a larger "intersection" of those
  /// loops) with a newly constructed tensor (sub)expression of given kind.
  /// Returns the identifier of the new lattice point.
  LatPointId conjLat(ExprId e, LatPointId p0, LatPointId p1,
                     Operation *op = nullptr);

  /// Conjunctive merge of two lattice sets: `(s0 /\_op s1)`.
  /// Returns the identifier of the new set.
  LatSetId conjSet(ExprId e, LatSetId s0, LatSetId s1, Operation *op = nullptr);

  /// Disjunctive merge of two lattice sets: `(s0 /\_op s1, s0, s1)`.
  /// Returns the identifier of the new set.
  LatSetId disjSet(ExprId e, LatSetId s0, LatSetId s1, Operation *op = nullptr);

  /// Disjunctive merge of two lattice sets and also set one of the operand to
  /// zero: `(s0 /\_op s1 (e0 op e1), s0 (0 op e0), s1 (e1 op 0))`.
  /// Returns the identifier of the new set.
  LatSetId disjSetWithZero(ExprId e, LatSetId s0, LatSetId s1);

  /// Disjunctive merge of two lattice sets with custom handling of the
  /// overlap, left, and right regions.  Any region may be left missing
  /// in the output.  Returns the identifier of the new set.
  LatSetId combiSet(ExprId e, LatSetId s0, LatSetId s1, Operation *orig,
                    bool includeLeft, TensorExp::Kind ltrans, Operation *opleft,
                    bool includeRight, TensorExp::Kind rtrans,
                    Operation *opright);

  /// Maps the unary operator over the lattice set of the operand, i.e. each
  /// lattice point on an expression E is simply copied over, but with OP E
  /// as new expression. Returns the identifier of the new set.
  LatSetId mapSet(TensorExp::Kind kind, LatSetId s, Value v = Value(),
                  Operation *op = nullptr);

  /// Maps the binary operator to the same operation but with one of its operand
  /// set to zero, i.e. each lattice point on an expression E is simply copied
  /// over, but with `OP 0 E` (if lhsZero == true) or `OP E 0` (if lhsZero ==
  /// false) as new expression. Returns the identifier of the new set.
  LatSetId mapBinWithSynZeroSet(ExprId e, LatSetId s, bool lhsZero);

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
  constexpr TensorId tensor(TensorLoopId b) const { return b % numTensors; }
  /// Gets the loop-identifier of the `TensorLoopId`.
  constexpr LoopId loop(TensorLoopId b) const { return b / numTensors; }

  /// Gets the total number of tensors (including the output-tensor and
  /// synthetic-tensor).
  constexpr unsigned getNumTensors() const { return numTensors; }

  /// Gets the total number of loops (native loops + filter loops).
  constexpr unsigned getNumLoops() const { return numLoops; }

  /// Returns true if `b` is the `i`th loop of the output tensor.
  constexpr bool isOutTensor(TensorLoopId b, LoopId i) const {
    return b == makeTensorLoopId(outTensor, i);
  }

  /// Gets the output tensor's identifier.
  constexpr TensorId getOutTensorID() const { return outTensor; }

  /// Gets the synthetic tensor's identifier (used for all invariant
  /// tensor expressions).
  constexpr TensorId getSynTensorID() const { return syntheticTensor; }

  /// Returns true if the expression is `(kTensor t)`.
  bool expIsTensor(ExprId e, TensorId t) const {
    const auto &expr = exp(e);
    return expr.kind == TensorExp::Kind::kTensor && expr.tensor == t;
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
  LevelType getLvlType(TensorId t, LoopId i) const {
    assert(isValidTensorId(t) && isValidLoopId(i));
    return lvlTypes[t][i];
  }

  /// Gets the level-type of the TensorLoopId.
  LevelType getLvlType(TensorLoopId b) const {
    return getLvlType(tensor(b), loop(b));
  }

  /// Gets the loop identifier for the `lvl`th level of the `t`th tensor.
  std::optional<LoopId> getLoopId(TensorId t, Level lvl) const {
    assert(isValidLevel(t, lvl));
    return lvlToLoop[t][lvl];
  }

  /// Gets the level number of the the `t`th tensor on `i`th loop.
  std::optional<Level> getLvl(TensorId t, LoopId i) const {
    assert(isValidTensorId(t) && isValidLoopId(i));
    return loopToLvl[t][i];
  }
  std::optional<Level> getLvl(TensorLoopId b) const {
    return getLvl(tensor(b), loop(b));
  }

  /// Sets the level number and level-type of the `t`th tensor on
  /// `i`th loop.
  void setLevelAndType(TensorId t, LoopId i, Level lvl, LevelType lt) {
    assert(isValidLevel(t, lvl) && isValidLoopId(i) && isValidLT(lt));
    lvlTypes[t][i] = lt;
    loopToLvl[t][i] = lvl;
    lvlToLoop[t][lvl] = i;
    // TODO: favor a constant loop bound when there are multiple choices.
    loopBounds[i] = std::make_pair(t, lvl);
  }

  using ForeachTensorLoopIdCallback = function_ref<void(
      TensorLoopId, TensorId, std::optional<Level>, LevelType, bool)>;

  /// Iterates over a set of `TensorLoopId`s, invoking the callback
  /// for each `TensorLoopId` and passing it the corresponding tensor
  /// identifier, level, and level-type, following with a boolean value
  /// indicating whether it is a dependent index reduction loop condition.
  void foreachTensorLoopId(LatPointId p,
                           ForeachTensorLoopIdCallback callback) const {
    // TODO: the default ought to be simple=true; but we'll need to make
    // sure to update all the tests to make sure they do the right thing.
    foreachTensorLoopId(p, /*simple=*/false, callback);
  }
  void foreachTensorLoopId(LatPointId p, bool simple,
                           ForeachTensorLoopIdCallback callback) const {
    const auto &point = lat(p);
    const auto &bits = simple ? point.simple : point.bits;
    for (const TensorLoopId b : bits.set_bits()) {
      const TensorId t = tensor(b);
      const auto optLvl = getLvl(b);
      const auto lvlTp = getLvlType(b);
      if (isLvlWithNonTrivialIdxExp(b)) {
        // This must be an undefined level.
        assert(!optLvl.has_value());
        // Slice the tid along the dependent level to iterate current loop.
        callback(b, t, getLoopDependentLevel(b), lvlTp,
                 /*isIdxReduc=*/true);
      } else {
        callback(b, t, optLvl, lvlTp, /*isIdxReduc=*/false);
      }
    }
  }

  /// Sets whether the output tensor is sparse or not.
  void setHasSparseOut(bool s) { hasSparseOut = s; }

  /// Establishes the two-way map that i <-> <t, lvl, lt>.
  void setLoopDependentTensorLevel(LoopId i, TensorId t, Level lvl,
                                   LevelType lt, unsigned coefficient) {
    assert(isValidLoopId(i) && isValidLevel(t, lvl));
    assert(!loopToUnresolvedLvls[i][t].has_value()); // must be the first def
    loopToUnresolvedLvls[i][t] = std::make_pair(lvl, lt);
    levelToDependentLoop[t][lvl].emplace_back(i, coefficient);
  }

  /// Whether the loop has dependent slice.
  bool hasDependentLvl(LoopId i, TensorId t) {
    assert(isValidTensorId(t) && isValidLoopId(i));
    return loopToUnresolvedLvls[i][t].has_value();
  }

  /// Returns the list of loop indices which appear in the non-trivial index
  /// expression on t_l, e.g., A[i+j] => {i, j}
  std::vector<LoopCoeffPair> &getDependentLoops(TensorId t, Level lvl) {
    assert(isValidLevel(t, lvl));
    return levelToDependentLoop[t][lvl];
  }

  /// Returns the defining [tid, lvl] for the loop.
  std::pair<TensorId, Level> getLoopDefiningLvl(LoopId i) const {
    assert(isValidLoopId(i));
    return loopBounds[i];
  }

  /// Checks whether the TensorLoopId represents a tensor level contains
  /// non-trivial index expression.
  bool isLvlWithNonTrivialIdxExp(TensorLoopId b) const {
    const TensorId t = tensor(b);
    const LoopId i = loop(b);
    assert(isValidTensorId(t) && isValidLoopId(i));
    return loopToUnresolvedLvls[i][t].has_value();
  }

  /// Checks whether the TensorLoopId represents a sparse tensor level contains
  /// non-trivial index expression.
  bool isSparseLvlWithNonTrivialIdxExp(TensorLoopId b) const {
    if (isLvlWithNonTrivialIdxExp(b)) {
      auto lt = getLoopDependentLevelType(b);
      return isCompressedLT(lt) || isSingletonLT(lt) ||
             isLooseCompressedLT(lt) || is2OutOf4LT(lt);
    }
    return false;
  }

  Level getLoopDependentLevel(TensorLoopId b) const {
    assert(isLvlWithNonTrivialIdxExp(b));
    return loopToUnresolvedLvls[loop(b)][tensor(b)]->first;
  }

  LevelType getLoopDependentLevelType(TensorLoopId b) const {
    assert(isLvlWithNonTrivialIdxExp(b));
    return loopToUnresolvedLvls[loop(b)][tensor(b)]->second;
  }

  /// Convenience getters to immediately access the stored nodes.
  /// These methods return `const&` because the underlying objects must
  /// not be mutated by client code.  The only exception is for mutating
  /// the value associated with an expression, for which there are
  /// dedicated methods below.
  ///
  /// NOTE: It is inadvisable to keep the reference alive for a long
  /// time (e.g., as in `TensorExpr &te = merger.exp(e)`), since insertions
  /// into the merger can cause data movement which will invalidate the
  /// underlying memory address.  This isn't just a problem with the `&`
  /// references, but also applies to the `ArrayRef`.  In particular,
  /// using `for (LatPointId p : merger.set(s))` will run into the same
  /// dangling-reference problems if the loop body inserts new sets.
  const TensorExp &exp(ExprId e) const {
    assert(isValidExprId(e));
    return tensorExps[e];
  }
  const LatPoint &lat(LatPointId p) const {
    assert(isValidLatPointId(p));
    return latPoints[p];
  }
  ArrayRef<LatPointId> set(LatSetId s) const {
    assert(isValidLatSetId(s));
    return latSets[s];
  }

  /// Checks whether the given expression has an associated value.
  bool hasExprValue(ExprId e) const { return static_cast<bool>(exp(e).val); }

  /// Sets the expression to have the associated value. Asserts that the new
  /// value is defined, and that the expression does not already have a value.
  void setExprValue(ExprId e, Value v) {
    assert(!exp(e).val && "Expression already has an associated value");
    assert(v && "Trying to assign an undefined value");
    tensorExps[e].val = v;
  }

  /// Clears the value associated with the expression. Asserts that the
  /// expression does indeed have an associated value before clearing it.
  void clearExprValue(ExprId e) {
    assert(exp(e).val && "Expression does not have an associated value");
    tensorExps[e].val = Value();
  }

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
  constexpr bool isValidTensorId(TensorId t) const { return t < numTensors; }
  constexpr bool isValidLoopId(LoopId i) const {
    return i != detail::kInvalidId && i < numLoops;
  }
  bool isValidLevel(TensorId t, Level lvl) const {
    assert(levelToDependentLoop[t].size() == lvlToLoop[t].size());
    return isValidTensorId(t) && lvl < lvlToLoop[t].size();
  }
  bool isValidExprId(ExprId e) const {
    return e != detail::kInvalidId && e < tensorExps.size();
  }
  bool isValidLatPointId(LatPointId p) const {
    return p != detail::kInvalidId && p < latPoints.size();
  }
  bool isValidLatSetId(LatSetId s) const {
    return s != detail::kInvalidId && s < latSets.size();
  }
  bool maybeZero(ExprId e) const;
  bool isInvariant(ExprId e) const {
    return exp(e).kind == TensorExp::Kind::kInvariant;
  }
  Type inferType(ExprId e, Value src) const;

  /// Traverses the SSA tree (possibly a DAG) to build a tensor expression.
  /// The boolean value returned indicates whether the result of the current
  /// operation being built depends on any value that is loaded from a sparse
  /// tensor.
  std::pair<std::optional<ExprId>, bool> buildTensorExp(linalg::GenericOp op,
                                                        Value v);

  /// Merger data structures.
  const TensorId outTensor;
  const TensorId syntheticTensor;
  const unsigned numTensors;
  const unsigned numLoops;
  bool hasSparseOut;

  // Below we use `std::vector` for things which have a priori fixed
  // sizes, whereas we use `llvm::SmallVector` for things with variable
  // size.  Do beware that these two classes differ in the semantics of
  // `operator[]`: `SmallVector` performs OOB checks, whereas `std::vector`
  // does not.

  /// Map that converts pair<TensorId, LoopId> to the corresponding lvl-type.
  std::vector<std::vector<LevelType>> lvlTypes;

  /// Map that converts pair<TensorId, LoopId> to the corresponding lvl.
  std::vector<std::vector<std::optional<Level>>> loopToLvl;

  /// Map that converts pair<TensorId, Level> to the corresponding LoopId.
  std::vector<std::vector<std::optional<LoopId>>> lvlToLoop;

  /// Map from a loop to its dependencies if any.
  /// The dependencies of a loop is a set of (tensor, level) pairs.
  /// It is currently only set for non-trivial index expressions.
  /// E.g., A[i+j] => i and j will have dependencies {A0, lt(A0)} to indicate
  /// that i and j are used in the non-trivial index expression on A0.
  std::vector<std::vector<std::optional<LvlLTPair>>> loopToUnresolvedLvls;

  /// The inverse map of ldxToDependencies from tensor level -> dependent loop
  /// E.g., A[2i+j], we have A0 => {(2, i), (1, j)}, to indicate that A0 uses
  /// both {i, j} to compute its indices and the coefficients on the loop id are
  /// 2 and 1 respectively.
  std::vector<std::vector<std::vector<LoopCoeffPair>>> levelToDependentLoop;

  /// Map from a loop to the [tid, lvl] pair that defines the loop boundary.
  std::vector<std::pair<TensorId, Level>> loopBounds;

  llvm::SmallVector<TensorExp> tensorExps;
  llvm::SmallVector<LatPoint> latPoints;
  llvm::SmallVector<SmallVector<LatPointId>> latSets;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_UTILS_MERGER_H_
