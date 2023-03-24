#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "llvm/Support/Compiler.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

using namespace mlir;
using namespace mlir::sparse_tensor;

// Silence 'warning C4002: 'too many arguments for function-liked macro
//                          invocation'
// as MSVC handles ##__VA_ARGS__ differently as gcc/clang

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4002)
#endif

namespace {

///
/// Defines macros to iterate binary and the combination of binary operations.
///

#define FOREVERY_BINOP(DO)                                                     \
  DO(mulf, TensorExp::Kind::kMulF)                                             \
  DO(mulc, TensorExp::Kind::kMulC)                                             \
  DO(muli, TensorExp::Kind::kMulI)                                             \
  DO(addf, TensorExp::Kind::kAddF)                                             \
  DO(addc, TensorExp::Kind::kAddC)                                             \
  DO(addi, TensorExp::Kind::kAddI)                                             \
  DO(subf, TensorExp::Kind::kSubF)                                             \
  DO(subc, TensorExp::Kind::kSubC)                                             \
  DO(subi, TensorExp::Kind::kSubI)                                             \
  DO(andi, TensorExp::Kind::kAndI)                                             \
  DO(xori, TensorExp::Kind::kXorI)                                             \
  DO(ori, TensorExp::Kind::kOrI)

// TODO: Disjunctive binary operations that need special handling are not
// included, e.g., Division are not tested (for now) as it need a constant
// non-zero dividend.
// ##__VA_ARGS__ handles cases when __VA_ARGS__ is empty.
#define FOREVERY_COMMON_DISJ_BINOP(TEST, ...)                                  \
  TEST(addf, ##__VA_ARGS__)                                                    \
  TEST(addc, ##__VA_ARGS__)                                                    \
  TEST(addi, ##__VA_ARGS__)                                                    \
  TEST(xori, ##__VA_ARGS__)                                                    \
  TEST(ori, ##__VA_ARGS__)

// TODO: Conjunctive binary operations that need special handling are not
// included, e.g., substraction yields a different pattern as it is mapped to
// negate operation.
#define FOREVERY_COMMON_CONJ_BINOP(TEST, ...)                                  \
  TEST(mulf, ##__VA_ARGS__)                                                    \
  TEST(mulc, ##__VA_ARGS__)                                                    \
  TEST(muli, ##__VA_ARGS__)                                                    \
  TEST(andi, ##__VA_ARGS__)

#define FOREVERY_PAIR_OF_COMMON_CONJ_DISJ_BINOP(TEST)                          \
  FOREVERY_COMMON_CONJ_BINOP(TEST, addf)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, addc)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, addi)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, xori)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, ori)

#define FOREVERY_PAIR_OF_COMMON_CONJ_CONJ_BINOP(TEST)                          \
  FOREVERY_COMMON_CONJ_BINOP(TEST, mulf)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, mulc)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, muli)                                       \
  FOREVERY_COMMON_CONJ_BINOP(TEST, andi)

#define FOREVERY_PAIR_OF_COMMON_DISJ_DISJ_BINOP(TEST)                          \
  FOREVERY_COMMON_DISJ_BINOP(TEST, addf)                                       \
  FOREVERY_COMMON_DISJ_BINOP(TEST, addc)                                       \
  FOREVERY_COMMON_DISJ_BINOP(TEST, addi)                                       \
  FOREVERY_COMMON_DISJ_BINOP(TEST, ori)                                        \
  FOREVERY_COMMON_DISJ_BINOP(TEST, xori)

///
/// Helper classes/functions for testing Merger.
///

/// Simple recursive data structure used to match expressions in `Merger`.
struct Pattern;
/// Since the patterns we need are rather small and short-lived, we use
/// `Pattern const&` for "pointers" to patterns, rather than using
/// something more elaborate like `std::shared_ptr<Pattern> const&`.
/// (But since we use a typedef rather than spelling it out everywhere,
/// that's easy enough to swap out if we need something more elaborate
/// in the future.)
using PatternRef = const Pattern &;
struct Pattern {
  struct Children {
    Children(PatternRef e0, PatternRef e1) : e0(e0), e1(e1) {}
    PatternRef e0;
    PatternRef e1;
  };

  TensorExp::Kind kind;

  union {
    /// Expressions representing tensors simply have a tensor number.
    TensorId tid;

    /// Tensor operations point to their children.
    Children children;
  };

  /// Constructors.
  /// Rather than using these, please use the readable helper constructor
  /// functions below to make tests more readable.
  Pattern(TensorId tid) : kind(TensorExp::Kind::kTensor), tid(tid) {}
  Pattern(TensorExp::Kind kind, PatternRef e0, PatternRef e1)
      : kind(kind), children(e0, e1) {
    assert(kind >= TensorExp::Kind::kMulF);
  }
};

///
/// Readable Pattern builder functions.
/// These should be preferred over the actual constructors.
///

static Pattern tensorPattern(TensorId tid) { return Pattern(tid); }

#define IMPL_BINOP_PATTERN(OP, KIND)                                           \
  LLVM_ATTRIBUTE_UNUSED static Pattern OP##Pattern(PatternRef e0,              \
                                                   PatternRef e1) {            \
    return Pattern(KIND, e0, e1);                                              \
  }

FOREVERY_BINOP(IMPL_BINOP_PATTERN)

#undef IMPL_BINOP_PATTERN

class MergerTestBase : public ::testing::Test {
protected:
  MergerTestBase(unsigned numTensors, unsigned numLoops)
      : merger(numTensors, numLoops, /*numFilterLoops=*/0,
               /*maxRank=*/numLoops) {
    tensors.reserve(numTensors);
    for (unsigned t = 0; t < numTensors; t++)
      tensors.push_back(merger.addExp(TensorExp::Kind::kTensor, tid(t)));
  }

  ///
  /// Expression construction helpers.
  ///

  TensorId tid(unsigned t) const {
    assert(t < merger.getNumTensors());
    return t;
  }
  LoopId lid(unsigned i) const {
    assert(i < merger.getNumLoops());
    return i;
  }
  ExprId tensor(unsigned t) const {
    assert(t < tensors.size());
    return tensors[t];
  }

#define IMPL_BINOP_EXPR(OP, KIND)                                              \
  LLVM_ATTRIBUTE_UNUSED ExprId OP##Expr(ExprId e0, ExprId e1) {                \
    return merger.addExp(KIND, e0, e1);                                        \
  }

  FOREVERY_BINOP(IMPL_BINOP_EXPR)

#undef IMPL_BINOP_EXPR

  ///
  /// Comparison helpers.
  ///

  /// Returns true if any lattice point with an expression matching
  /// the given `pattern` and bits matching the given `bits` is present
  /// in the `[lo, lo+n)` slice of the lattice set `s`.  This is useful
  /// for testing partial ordering constraints between lattice points.
  /// We generally know how contiguous groups of lattice points should
  /// be ordered with respect to other groups, but there is no required
  /// ordering within groups.  If `simple` is true, then compare the
  /// `lat.simple` field instead to test the result after optimization.
  bool latPointWithinRange(LatSetId s, unsigned lo, unsigned n,
                           PatternRef pattern, const BitVector &bits,
                           bool simple) {
    for (unsigned k = lo, hi = lo + n; k < hi; ++k) {
      if (compareExpression(merger.lat(merger.set(s)[k]).exp, pattern) &&
          compareBits(s, k, bits, simple))
        return true;
    }
    return false;
  }

  /// Wrapper over latPointWithinRange for readability of tests.
  void expectLatPointWithinRange(LatSetId s, unsigned lo, unsigned n,
                                 PatternRef pattern, const BitVector &bits,
                                 bool simple = false) {
    EXPECT_TRUE(latPointWithinRange(s, lo, n, pattern, bits, simple));
  }

  /// Wrapper over expectLatPointWithinRange for a single lat point.
  void expectLatPoint(LatSetId s, unsigned lo, PatternRef pattern,
                      const BitVector &bits, bool simple = false) {
    EXPECT_TRUE(latPointWithinRange(s, lo, 1, pattern, bits, simple));
  }

  /// Converts a vector of (loop, tensor) pairs to a bitvector with the
  /// corresponding bits set.
  BitVector loopsToBits(const std::vector<std::pair<LoopId, TensorId>> &loops) {
    // NOTE: this `numTensors` includes both the output- and synthetic-tensors.
    const auto numTensors = merger.getNumTensors();
    BitVector testBits = BitVector(numTensors, false);
    for (auto [loop, tensor] : loops)
      testBits.set(numTensors * loop + tensor);
    return testBits;
  }

  /// Returns true if the bits of the `k`th point in set `s` matches
  /// the given `bits`.  If `simple` is true, then compares the `lat.simple`
  /// field instead, to test the result after optimization
  bool compareBits(LatSetId s, unsigned k, const BitVector &bits, bool simple) {
    const auto &point = merger.lat(merger.set(s)[k]);
    return (simple ? point.simple : point.bits) == bits;
  }

  /// Check that there are n lattice points in set s.
  void expectNumLatPoints(LatSetId s, unsigned n) {
    EXPECT_THAT(merger.set(s).size(), n);
  }

  /// Compares expressions for equality. Equality is defined recursively as:
  /// - Operations are equal if they have the same kind and children.
  /// - Leaf tensors are equal if they refer to the same tensor.
  bool compareExpression(ExprId e, PatternRef pattern) {
    const auto &tensorExp = merger.exp(e);
    if (tensorExp.kind != pattern.kind)
      return false;
    switch (tensorExp.kind) {
    // Leaf.
    case TensorExp::Kind::kTensor:
      return tensorExp.tensor == pattern.tid;
    case TensorExp::Kind::kInvariant:
      llvm_unreachable("invariant not handled yet");
    case TensorExp::Kind::kLoopVar:
      llvm_unreachable("loop-variables not handled yet");
    // Unary operations.
    case TensorExp::Kind::kAbsF:
    case TensorExp::Kind::kAbsC:
    case TensorExp::Kind::kAbsI:
    case TensorExp::Kind::kCeilF:
    case TensorExp::Kind::kFloorF:
    case TensorExp::Kind::kSqrtF:
    case TensorExp::Kind::kSqrtC:
    case TensorExp::Kind::kExpm1F:
    case TensorExp::Kind::kExpm1C:
    case TensorExp::Kind::kLog1pF:
    case TensorExp::Kind::kLog1pC:
    case TensorExp::Kind::kSinF:
    case TensorExp::Kind::kSinC:
    case TensorExp::Kind::kTanhF:
    case TensorExp::Kind::kTanhC:
    case TensorExp::Kind::kNegF:
    case TensorExp::Kind::kNegC:
    case TensorExp::Kind::kNegI:
    case TensorExp::Kind::kTruncF:
    case TensorExp::Kind::kExtF:
    case TensorExp::Kind::kCastFS:
    case TensorExp::Kind::kCastFU:
    case TensorExp::Kind::kCastSF:
    case TensorExp::Kind::kCastUF:
    case TensorExp::Kind::kCastS:
    case TensorExp::Kind::kCastU:
    case TensorExp::Kind::kCastIdx:
    case TensorExp::Kind::kTruncI:
    case TensorExp::Kind::kCIm:
    case TensorExp::Kind::kCRe:
    case TensorExp::Kind::kBitCast:
    case TensorExp::Kind::kSelect:
    case TensorExp::Kind::kBinaryBranch:
    case TensorExp::Kind::kUnary:
      return compareExpression(tensorExp.children.e0, pattern.children.e0);
    // Binary operations.
    case TensorExp::Kind::kMulF:
    case TensorExp::Kind::kMulC:
    case TensorExp::Kind::kMulI:
    case TensorExp::Kind::kDivF:
    case TensorExp::Kind::kDivC:
    case TensorExp::Kind::kDivS:
    case TensorExp::Kind::kDivU:
    case TensorExp::Kind::kAddF:
    case TensorExp::Kind::kAddC:
    case TensorExp::Kind::kAddI:
    case TensorExp::Kind::kSubF:
    case TensorExp::Kind::kSubC:
    case TensorExp::Kind::kSubI:
    case TensorExp::Kind::kAndI:
    case TensorExp::Kind::kOrI:
    case TensorExp::Kind::kXorI:
    case TensorExp::Kind::kShrS:
    case TensorExp::Kind::kShrU:
    case TensorExp::Kind::kShlI:
    case TensorExp::Kind::kBinary:
    case TensorExp::Kind::kReduce:
      return compareExpression(tensorExp.children.e0, pattern.children.e0) &&
             compareExpression(tensorExp.children.e1, pattern.children.e1);
    }
    llvm_unreachable("unexpected kind");
  }

  // This field is public for convenience.
  Merger merger;

private:
  // This field is private to prevent mutation after the ctor.
  SmallVector<ExprId> tensors;
};

///
/// Tests with all sparse inputs.
///

/// Three tensors (two inputs, one output); and a single loop.
class MergerTest3T1L : public MergerTestBase {
protected:
  MergerTest3T1L() : MergerTestBase(3, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == tid(2));
    // Tensor 0: sparse input vector.
    merger.setLevelAndType(tid(0), lid(0), 0, DimLevelType::Compressed);
    // Tensor 1: sparse input vector.
    merger.setLevelAndType(tid(1), lid(0), 0, DimLevelType::Compressed);
    // Tensor 2: dense output vector.
    merger.setLevelAndType(tid(2), lid(0), 0, DimLevelType::Dense);
  }
};

/// Four tensors (three inputs, one output); and a single loop.
class MergerTest4T1L : public MergerTestBase {
protected:
  MergerTest4T1L() : MergerTestBase(4, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == tid(3));
    // Tensor 0: sparse input vector.
    merger.setLevelAndType(tid(0), lid(0), 0, DimLevelType::Compressed);
    // Tensor 1: sparse input vector.
    merger.setLevelAndType(tid(1), lid(0), 0, DimLevelType::Compressed);
    // Tensor 2: sparse input vector
    merger.setLevelAndType(tid(2), lid(0), 0, DimLevelType::Compressed);
    // Tensor 3: dense output vector
    merger.setLevelAndType(tid(3), lid(0), 0, DimLevelType::Dense);
  }
};

///
/// Tests with both sparse and dense input.
///

/// Three tensors (two inputs, one output); and a single loop.
class MergerTest3T1LD : public MergerTestBase {
protected:
  MergerTest3T1LD() : MergerTestBase(3, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == tid(2));
    // Tensor 0: sparse input vector.
    merger.setLevelAndType(tid(0), lid(0), 0, DimLevelType::Compressed);
    // Tensor 1: dense input vector.
    merger.setLevelAndType(tid(1), lid(0), 0, DimLevelType::Dense);
    // Tensor 2: dense output vector.
    merger.setLevelAndType(tid(2), lid(0), 0, DimLevelType::Dense);
  }
};

///
/// Tests with both undef and dense input.
///

/// Three tensors (three inputs, one output); and a single loop.
class MergerTest4T1LU : public MergerTestBase {
protected:
  MergerTest4T1LU() : MergerTestBase(4, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == tid(3));
    // Tensor 0: undef input vector.
    merger.setLevelAndType(tid(0), lid(0), 0, DimLevelType::Undef);
    // Tensor 1: dense input vector.
    merger.setLevelAndType(tid(1), lid(0), 0, DimLevelType::Dense);
    // Tensor 2: undef input vector.
    merger.setLevelAndType(tid(2), lid(0), 0, DimLevelType::Undef);
    // Tensor 3: dense output vector.
    merger.setLevelAndType(tid(3), lid(0), 0, DimLevelType::Dense);
  }
};

///
/// Tests with operation on sparse output.
///

/// Three tensors (two inputs, one output, one synthetic); and a single loop.
class MergerTest3T1LSo : public MergerTestBase {
protected:
  MergerTest3T1LSo() : MergerTestBase(3, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == tid(2));
    EXPECT_TRUE(merger.getSynTensorID() == tid(3));
    merger.setHasSparseOut(true);
    // Tensor 0: undef input vector.
    merger.setLevelAndType(tid(0), lid(0), 0, DimLevelType::Undef);
    // Tensor 1: undef input vector.
    merger.setLevelAndType(tid(1), lid(0), 0, DimLevelType::Undef);
    // Tensor 2: sparse output vector.
    merger.setLevelAndType(tid(2), lid(0), 0, DimLevelType::Compressed);
  }
};

} // namespace

/// Vector multiplication (conjunction) of 3 vectors, i.e.;
///   a(i) = b(i) * c(i) * d(i)
/// which should form the single lattice point
/// {
///   lat( i_00_U i_01_D i_02_U / (tensor_0 * tensor_1 * tensor2) )
/// }
/// after optimization, the dense dimesion should be kept, despite it appears
/// in the middle
/// {
///   lat( i_01_D / (tensor_0 * tensor_1 * tensor2) )
/// }
#define IMPL_MERGER_TEST_CONJ_CONJ_UNDEF(CONJ1, CONJ2)                         \
  TEST_F(MergerTest4T1LU, vector_##CONJ1##_##CONJ2) {                          \
    const auto em = CONJ1##Expr(tensor(0), tensor(1));                         \
    const auto e = CONJ2##Expr(em, tensor(2));                                 \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const auto t2 = tid(2);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    const PatternRef p2 = tensorPattern(t2);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t1}}), true);                             \
  }

FOREVERY_PAIR_OF_COMMON_CONJ_CONJ_BINOP(IMPL_MERGER_TEST_CONJ_CONJ_UNDEF)

#undef IMPL_MERGER_TEST_CONJ_CONJ_UNDEF

/// Vector multiplication (conjunction) of 2 vectors, i.e.;
///   o(i) = b(i) * c(i) * o(i)
/// which should form the single lattice point (note how a synthetic tensor
/// i_03_U is created for the sparse output)
/// {
///   lat( i_00_U i_01_U i_03_U / (tensor_0 * tensor_1 * output_tensor_2) )
/// }
/// after optimization, the synthetic tensor should be preserved.
/// {
///   lat( i_03_U / (tensor_0 * tensor_1 * output_tensor2) )
/// }
#define IMPL_MERGER_TEST_CONJ_CONJ_SPARSE_OUT(CONJ1, CONJ2)                    \
  TEST_F(MergerTest3T1LSo, vector_##CONJ1##_##CONJ2) {                         \
    const auto em = CONJ1##Expr(tensor(0), tensor(1));                         \
    const auto e = CONJ2##Expr(em, tensor(2));                                 \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const auto t2 = tid(2);                                                    \
    const auto t3 = tid(3);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    const PatternRef p2 = tensorPattern(t2);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t3}}));               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t3}}), true);                             \
  }

FOREVERY_PAIR_OF_COMMON_CONJ_CONJ_BINOP(IMPL_MERGER_TEST_CONJ_CONJ_SPARSE_OUT)

#undef IMPL_MERGER_TEST_CONJ_CONJ_SPARSE_OUT

/// Vector addition (disjunction) of 2 vectors. i.e.;
///   a(i) = b(i) + c(i)
/// which should form the 3 lattice points
/// {
///   lat( i_00 i_01 / (tensor_0 + tensor_1) )
///   lat( i_00 / tensor_0 )
///   lat( i_01 / tensor_1 )
/// }
/// and after optimization, the lattice points do not change (as there is no
/// duplicated point and all input vectors are sparse vector).
/// {
///   lat( i_00 i_01 / (tensor_0 + tensor_1) )
///   lat( i_00 / tensor_0 )
///   lat( i_01 / tensor_1 )
/// }
#define IMPL_MERGER_TEST_DISJ(OP)                                              \
  TEST_F(MergerTest3T1L, vector_##OP) {                                        \
    const auto e = OP##Expr(tensor(0), tensor(1));                             \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
    expectLatPointWithinRange(s, 1, 2, p0, loopsToBits({{l0, t0}}));           \
    expectLatPointWithinRange(s, 1, 2, p1, loopsToBits({{l0, t1}}));           \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}), true);                   \
    expectLatPointWithinRange(s, 1, 2, p0, loopsToBits({{l0, t0}}), true);     \
    expectLatPointWithinRange(s, 1, 2, p1, loopsToBits({{l0, t1}}), true);     \
  }

FOREVERY_COMMON_DISJ_BINOP(IMPL_MERGER_TEST_DISJ)

#undef IMPL_MERGER_TEST_DISJ

/// Vector multiplication (conjunction) of 2 vectors, i.e.;
///   a(i) = b(i) * c(i)
/// which should form the single lattice point
/// {
///   lat( i_00 i_01 / (tensor_0 * tensor_1) )
/// }
#define IMPL_MERGER_TEST_CONJ(OP)                                              \
  TEST_F(MergerTest3T1L, vector_##OP) {                                        \
    const auto e = OP##Expr(tensor(0), tensor(1));                             \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}), true);                   \
  }

FOREVERY_COMMON_CONJ_BINOP(IMPL_MERGER_TEST_CONJ)

#undef IMPL_MERGER_TEST_CONJ

/// Vector multiplication (conjunction) then addition (disjunction), i.e.;
///   a(i) = b(i) * c(i) + d(i);
/// which should form
/// {
///    lat( i_00 i_01 i_02 / (tensor_0 * tensor_1) + tensor_2 )
///    lat( i_00 i_01 / tensor_0 * tensor_1
///    lat( i_02 / tensor_2 )
/// }
#define IMPL_MERGER_TEST_CONJ_DISJ(CONJ, DISJ)                                 \
  TEST_F(MergerTest4T1L, vector_##CONJ##_##DISJ) {                             \
    const auto em = CONJ##Expr(tensor(0), tensor(1));                          \
    const auto e = DISJ##Expr(em, tensor(2));                                  \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const auto t2 = tid(2);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    const PatternRef p2 = tensorPattern(t2);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, 0, DISJ##Pattern(CONJ##Pattern(p0, p1), p2),             \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, 1, 2, CONJ##Pattern(p0, p1),                  \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, 1, 2, p2, loopsToBits({{l0, t2}}));           \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, 0, DISJ##Pattern(CONJ##Pattern(p0, p1), p2),             \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, 1, 2, CONJ##Pattern(p0, p1),                  \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, 1, 2, p2, loopsToBits({{l0, t2}}));           \
  }

FOREVERY_PAIR_OF_COMMON_CONJ_DISJ_BINOP(IMPL_MERGER_TEST_CONJ_DISJ)

#undef IMPL_MERGER_TEST_CONJ_DISJ

/// Vector addition (disjunction) then addition (disjunction), i.e.;
///   a(i) = b(i) + c(i) + d(i)
/// which should form
/// {
///   lat( i_00 i_01 i_02 / (tensor_0 + tensor_1) + tensor_2 )
///   lat( i_02 i_01 / tensor_2 + tensor_1 )
///   lat( i_02 i_00 / tensor_2 + tensor_0 )
///   lat( i_01 i_00 / tensor_1 + tensor_0 )
///   lat( i_02 / tensor_2 )
///   lat( i_01 / tensor_1 )
///   lat( i_00 / tensor_0 )
/// }
#define IMPL_MERGER_TEST_DISJ_DISJ(DISJ1, DISJ2)                               \
  TEST_F(MergerTest4T1L, Vector_##DISJ1##_##DISJ2) {                           \
    const auto em = DISJ1##Expr(tensor(0), tensor(1));                         \
    const auto e = DISJ2##Expr(em, tensor(2));                                 \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const auto t2 = tid(2);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    const PatternRef p2 = tensorPattern(t2);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 7);                                                  \
    expectLatPoint(s, 0, DISJ2##Pattern(DISJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, 1, 6, DISJ2##Pattern(p1, p2),                 \
                              loopsToBits({{l0, t1}, {l0, t2}}));              \
    expectLatPointWithinRange(s, 1, 6, DISJ2##Pattern(p0, p2),                 \
                              loopsToBits({{l0, t0}, {l0, t2}}));              \
    expectLatPointWithinRange(s, 1, 6, DISJ1##Pattern(p0, p1),                 \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, 1, 6, p2, loopsToBits({{l0, t2}}));           \
    expectLatPointWithinRange(s, 1, 6, p1, loopsToBits({{l0, t1}}));           \
    expectLatPointWithinRange(s, 1, 6, p0, loopsToBits({{l0, t0}}));           \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 7);                                                  \
    expectLatPoint(s, 0, DISJ2##Pattern(DISJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, 1, 6, DISJ2##Pattern(p1, p2),                 \
                              loopsToBits({{l0, t1}, {l0, t2}}));              \
    expectLatPointWithinRange(s, 1, 6, DISJ2##Pattern(p0, p2),                 \
                              loopsToBits({{l0, t0}, {l0, t2}}));              \
    expectLatPointWithinRange(s, 1, 6, DISJ1##Pattern(p0, p1),                 \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, 1, 6, p2, loopsToBits({{l0, t2}}));           \
    expectLatPointWithinRange(s, 1, 6, p1, loopsToBits({{l0, t1}}));           \
    expectLatPointWithinRange(s, 1, 6, p0, loopsToBits({{l0, t0}}));           \
  }

FOREVERY_PAIR_OF_COMMON_DISJ_DISJ_BINOP(IMPL_MERGER_TEST_DISJ_DISJ)

#undef IMPL_MERGER_TEST_DISJ_DISJ

/// Vector multiplication (conjunction) then multiplication (conjunction), i.e.;
///   a(i) = b(i) * c(i) * d(i);
/// which should form
/// {
///    lat( i_00 i_01 i_02 / tensor_0 * tensor_1 * tensor_2 )
/// }
#define IMPL_MERGER_TEST_CONJ_CONJ(CONJ1, CONJ2)                               \
  TEST_F(MergerTest4T1L, vector_##CONJ1##_##CONJ2) {                           \
    const auto em = CONJ1##Expr(tensor(0), tensor(1));                         \
    const auto e = CONJ2##Expr(em, tensor(2));                                 \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const auto t2 = tid(2);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    const PatternRef p2 = tensorPattern(t2);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),           \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}), true);         \
  }

FOREVERY_PAIR_OF_COMMON_CONJ_CONJ_BINOP(IMPL_MERGER_TEST_CONJ_CONJ)

#undef IMPL_MERGER_TEST_CONJ_CONJ

/// Vector addition (disjunction) of 2 vectors, i.e.;
///   a(i) = b(i) + c(i)
/// which should form the 3 lattice points
/// {
///   lat( i_00 i_01 / (sparse_tensor_0 + dense_tensor_1) )
///   lat( i_00 / sparse_tensor_0 )
///   lat( i_01 / dense_tensor_1 )
/// }
/// which should be optimized to
/// {
///   lat( i_00 i_01 / (sparse_tensor_0 + dense_tensor_1) ) (not singleton)
///   lat( i_01 / dense_tensor_0 ) (no sparse dimension)
/// }
///
/// lat( i_00 / sparse_tensor_0 ) should be opted out as it only has dense diff
/// with lat( i_00 i_01 / (sparse_tensor_0 + dense_tensor_1) ).
#define IMPL_MERGER_TEST_OPTIMIZED_DISJ(OP)                                    \
  TEST_F(MergerTest3T1LD, vector_opted_##OP) {                                 \
    const auto e = OP##Expr(tensor(0), tensor(1));                             \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
    expectLatPointWithinRange(s, 1, 2, p0, loopsToBits({{l0, t0}}));           \
    expectLatPointWithinRange(s, 1, 2, p1, loopsToBits({{l0, t1}}));           \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 2);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}), true);                   \
    expectLatPoint(s, 1, p1, loopsToBits({{l0, t1}}), true);                   \
  }

FOREVERY_COMMON_DISJ_BINOP(IMPL_MERGER_TEST_OPTIMIZED_DISJ)

#undef IMPL_MERGER_TEST_OPTIMIZED_CONJ

/// Vector multiplication (conjunction) of 2 vectors, i.e.:
///   a(i) = b(i) * c(i)
/// which should form the single lattice point
/// {
///   lat( i_00 i_01 / (sparse_tensor_0 * dense_tensor_1) )
/// }
/// it should be optimized to
/// {
///   lat( i_00 / (sparse_tensor_0 * dense_tensor_1) )
/// }
/// since i_01 is a dense dimension.
#define IMPL_MERGER_TEST_OPTIMIZED_CONJ(OP)                                    \
  TEST_F(MergerTest3T1LD, vector_opted_##OP) {                                 \
    const auto e = OP##Expr(tensor(0), tensor(1));                             \
    const auto l0 = lid(0);                                                    \
    const auto t0 = tid(0);                                                    \
    const auto t1 = tid(1);                                                    \
    const PatternRef p0 = tensorPattern(t0);                                   \
    const PatternRef p1 = tensorPattern(t1);                                   \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1),                                  \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, 0, OP##Pattern(p0, p1), loopsToBits({{l0, t0}}), true);  \
  }

FOREVERY_COMMON_CONJ_BINOP(IMPL_MERGER_TEST_OPTIMIZED_CONJ)

#undef IMPL_MERGER_TEST_OPTIMIZED_CONJ

// TODO: mult-dim tests

// restore warning status
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
