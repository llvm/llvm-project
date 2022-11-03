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
  DO(mulf, Kind::kMulF)                                                        \
  DO(mulc, Kind::kMulC)                                                        \
  DO(muli, Kind::kMulI)                                                        \
  DO(addf, Kind::kAddF)                                                        \
  DO(addc, Kind::kAddC)                                                        \
  DO(addi, Kind::kAddI)                                                        \
  DO(subf, Kind::kSubF)                                                        \
  DO(subc, Kind::kSubC)                                                        \
  DO(subi, Kind::kSubI)                                                        \
  DO(andi, Kind::kAndI)                                                        \
  DO(xori, Kind::kXorI)                                                        \
  DO(ori, Kind::kOrI)

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

/// Simple recursive data structure used to match expressions in Mergers.
struct Pattern {
  Kind kind;

  /// Expressions representing tensors simply have a tensor number.
  unsigned tensorNum;

  /// Tensor operations point to their children.
  std::shared_ptr<Pattern> e0;
  std::shared_ptr<Pattern> e1;

  /// Constructors.
  /// Rather than using these, please use the readable helper constructor
  /// functions below to make tests more readable.
  Pattern(unsigned tensorNum) : kind(Kind::kTensor), tensorNum(tensorNum) {}
  Pattern(Kind kind, const std::shared_ptr<Pattern> &e0,
          const std::shared_ptr<Pattern> &e1)
      : kind(kind), e0(e0), e1(e1) {
    assert(kind >= Kind::kMulF);
    assert(e0 && e1);
  }
};

///
/// Readable Pattern builder functions.
/// These should be preferred over the actual constructors.
///

static std::shared_ptr<Pattern> tensorPattern(unsigned tensorNum) {
  return std::make_shared<Pattern>(tensorNum);
}

#define IMPL_BINOP_PATTERN(OP, KIND)                                           \
  LLVM_ATTRIBUTE_UNUSED static std::shared_ptr<Pattern> OP##Pattern(           \
      const std::shared_ptr<Pattern> &e0,                                      \
      const std::shared_ptr<Pattern> &e1) {                                    \
    return std::make_shared<Pattern>(KIND, e0, e1);                            \
  }

FOREVERY_BINOP(IMPL_BINOP_PATTERN)

#undef IMPL_BINOP_PATTERN

class MergerTestBase : public ::testing::Test {
protected:
  MergerTestBase(unsigned numTensors, unsigned numLoops)
      : numTensors(numTensors), numLoops(numLoops),
        merger(numTensors, numLoops) {}

  ///
  /// Expression construction helpers.
  ///

  unsigned tensor(unsigned tensor) {
    return merger.addExp(Kind::kTensor, tensor);
  }

#define IMPL_BINOP_EXPR(OP, KIND)                                              \
  LLVM_ATTRIBUTE_UNUSED unsigned OP##Expr(unsigned e0, unsigned e1) {          \
    return merger.addExp(KIND, e0, e1);                                        \
  }

  FOREVERY_BINOP(IMPL_BINOP_EXPR)

#undef IMPL_BINOP_EXPR

  ///
  /// Comparison helpers.
  ///

  /// For readability of tests.
  unsigned lat(unsigned lat) { return lat; }

  /// Returns true if a lattice point with an expression matching the given
  /// pattern and bits matching the given bits is present in lattice points
  /// [p, p+n) of lattice set s. This is useful for testing partial ordering
  /// constraints between lattice points. We generally know how contiguous
  /// groups of lattice points should be ordered with respect to other groups,
  /// but there is no required ordering within groups.
  /// If simple is true, then compare the lat.simple field instead to test the
  /// result after optimization
  bool latPointWithinRange(unsigned s, unsigned p, unsigned n,
                           const std::shared_ptr<Pattern> &pattern,
                           const BitVector &bits, bool simple) {
    for (unsigned i = p; i < p + n; ++i) {
      if (compareExpression(merger.lat(merger.set(s)[i]).exp, pattern) &&
          compareBits(s, i, bits, simple))
        return true;
    }
    return false;
  }

  /// Wrapper over latPointWithinRange for readability of tests.
  void expectLatPointWithinRange(unsigned s, unsigned p, unsigned n,
                                 const std::shared_ptr<Pattern> &pattern,
                                 const BitVector &bits, bool simple = false) {
    EXPECT_TRUE(latPointWithinRange(s, p, n, pattern, bits, simple));
  }

  /// Wrapper over expectLatPointWithinRange for a single lat point.
  void expectLatPoint(unsigned s, unsigned p,
                      const std::shared_ptr<Pattern> &pattern,
                      const BitVector &bits, bool simple = false) {
    EXPECT_TRUE(latPointWithinRange(s, p, 1, pattern, bits, simple));
  }

  /// Converts a vector of (loop, tensor) pairs to a bitvector with the
  /// corresponding bits set.
  BitVector
  loopsToBits(const std::vector<std::pair<unsigned, unsigned>> &loops) {
    BitVector testBits = BitVector(numTensors + 1, false);
    for (auto l : loops) {
      auto loop = std::get<0>(l);
      auto tensor = std::get<1>(l);
      testBits.set(numTensors * loop + tensor);
    }
    return testBits;
  }

  /// Returns true if the bits of lattice point p in set s match the given bits.
  /// If simple is true, then compare the lat.simple field instead to test the
  /// result after optimization
  bool compareBits(unsigned s, unsigned p, const BitVector &bits, bool simple) {
    if (simple)
      return merger.lat(merger.set(s)[p]).simple == bits;
    return merger.lat(merger.set(s)[p]).bits == bits;
  }

  /// Check that there are n lattice points in set s.
  void expectNumLatPoints(unsigned s, unsigned n) {
    EXPECT_THAT(merger.set(s).size(), n);
  }

  /// Compares expressions for equality. Equality is defined recursively as:
  /// - Operations are equal if they have the same kind and children.
  /// - Leaf tensors are equal if they refer to the same tensor.
  bool compareExpression(unsigned e, const std::shared_ptr<Pattern> &pattern) {
    auto tensorExp = merger.exp(e);
    if (tensorExp.kind != pattern->kind)
      return false;
    switch (tensorExp.kind) {
    // Leaf.
    case kTensor:
      return tensorExp.tensor == pattern->tensorNum;
    case kInvariant:
    case kIndex:
      llvm_unreachable("invariant not handled yet");
    // Unary operations.
    case kAbsF:
    case kAbsC:
    case kAbsI:
    case kCeilF:
    case kFloorF:
    case kSqrtF:
    case kSqrtC:
    case kExpm1F:
    case kExpm1C:
    case kLog1pF:
    case kLog1pC:
    case kSinF:
    case kSinC:
    case kTanhF:
    case kTanhC:
    case kNegF:
    case kNegC:
    case kNegI:
    case kTruncF:
    case kExtF:
    case kCastFS:
    case kCastFU:
    case kCastSF:
    case kCastUF:
    case kCastS:
    case kCastU:
    case kCastIdx:
    case kTruncI:
    case kCIm:
    case kCRe:
    case kBitCast:
    case kSelect:
    case kBinaryBranch:
    case kUnary:
      return compareExpression(tensorExp.children.e0, pattern->e0);
    // Binary operations.
    case kMulF:
    case kMulC:
    case kMulI:
    case kDivF:
    case kDivC:
    case kDivS:
    case kDivU:
    case kAddF:
    case kAddC:
    case kAddI:
    case kSubF:
    case kSubC:
    case kSubI:
    case kAndI:
    case kOrI:
    case kXorI:
    case kShrS:
    case kShrU:
    case kShlI:
    case kBinary:
    case kReduce:
      return compareExpression(tensorExp.children.e0, pattern->e0) &&
             compareExpression(tensorExp.children.e1, pattern->e1);
    }
    llvm_unreachable("unexpected kind");
  }

  unsigned numTensors;
  unsigned numLoops;
  Merger merger;
};

///
/// Tests with all sparse inputs.
///

class MergerTest3T1L : public MergerTestBase {
protected:
  // Our three tensors (two inputs, one output).
  const unsigned t0 = 0, t1 = 1, t2 = 2;

  // Our single loop.
  const unsigned l0 = 0;

  MergerTest3T1L() : MergerTestBase(3, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == t2);

    // Tensor 0: sparse input vector.
    merger.addExp(Kind::kTensor, t0, -1u);
    merger.setDimAndDimLevelType(t0, l0, 0, DimLevelType::Compressed);

    // Tensor 1: sparse input vector.
    merger.addExp(Kind::kTensor, t1, -1u);
    merger.setDimAndDimLevelType(t1, l0, 0, DimLevelType::Compressed);

    // Tensor 2: dense output vector.
    merger.addExp(Kind::kTensor, t2, -1u);
    merger.setDimAndDimLevelType(t2, l0, 0, DimLevelType::Dense);
  }
};

class MergerTest4T1L : public MergerTestBase {
protected:
  // Our four tensors (three inputs, one output).
  const unsigned t0 = 0, t1 = 1, t2 = 2, t3 = 3;

  // Our single loop.
  const unsigned l0 = 0;

  MergerTest4T1L() : MergerTestBase(4, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == t3);

    // Tensor 0: sparse input vector.
    merger.addExp(Kind::kTensor, t0, -1u);
    merger.setDimAndDimLevelType(t0, l0, 0, DimLevelType::Compressed);

    // Tensor 1: sparse input vector.
    merger.addExp(Kind::kTensor, t1, -1u);
    merger.setDimAndDimLevelType(t1, l0, 0, DimLevelType::Compressed);

    // Tensor 2: sparse input vector
    merger.addExp(Kind::kTensor, t2, -1u);
    merger.setDimAndDimLevelType(t2, l0, 0, DimLevelType::Compressed);

    // Tensor 3: dense output vector
    merger.addExp(Kind::kTensor, t3, -1u);
    merger.setDimAndDimLevelType(t3, l0, 0, DimLevelType::Dense);
  }
};

///
/// Tests with both sparse and dense input.
///

class MergerTest3T1LD : public MergerTestBase {
protected:
  // Our three tensors (two inputs, one output).
  const unsigned t0 = 0, t1 = 1, t2 = 2;

  // Our single loop.
  const unsigned l0 = 0;

  MergerTest3T1LD() : MergerTestBase(3, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == t2);

    // Tensor 0: sparse input vector.
    merger.addExp(Kind::kTensor, t0, -1u);
    merger.setDimAndDimLevelType(t0, l0, 0, DimLevelType::Compressed);

    // Tensor 1: dense input vector.
    merger.addExp(Kind::kTensor, t1, -1u);
    merger.setDimAndDimLevelType(t1, l0, 0, DimLevelType::Dense);

    // Tensor 2: dense output vector.
    merger.addExp(Kind::kTensor, t2, -1u);
    merger.setDimAndDimLevelType(t2, l0, 0, DimLevelType::Dense);
  }
};

///
/// Tests with both undef and dense input.
///

class MergerTest4T1LU : public MergerTestBase {
protected:
  // Our three tensors (three inputs, one output).
  const unsigned t0 = 0, t1 = 1, t2 = 2, t3 = 3;

  // Our single loop.
  const unsigned l0 = 0;

  MergerTest4T1LU() : MergerTestBase(4, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == t3);

    // Tensor 0: undef input vector.
    merger.addExp(Kind::kTensor, t0, -1u);
    merger.setDimAndDimLevelType(t0, l0, 0, DimLevelType::Undef);

    // Tensor 1: dense input vector.
    merger.addExp(Kind::kTensor, t1, -1u);
    merger.setDimAndDimLevelType(t1, l0, 0, DimLevelType::Dense);

    // Tensor 2: undef input vector.
    merger.addExp(Kind::kTensor, t2, -1u);
    merger.setDimAndDimLevelType(t2, l0, 0, DimLevelType::Undef);

    // Tensor 3: dense output vector.
    merger.addExp(Kind::kTensor, t3, -1u);
    merger.setDimAndDimLevelType(t3, l0, 0, DimLevelType::Dense);
  }
};

///
/// Tests with operation on sparse output.
///

class MergerTest3T1LSo : public MergerTestBase {
protected:
  // Our three tensors (two inputs, one output, one synthetic).
  const unsigned t0 = 0, t1 = 1, t2 = 2, t3 = 3;

  // Our single loop.
  const unsigned l0 = 0;

  MergerTest3T1LSo() : MergerTestBase(3, 1) {
    EXPECT_TRUE(merger.getOutTensorID() == t2);
    EXPECT_TRUE(merger.getSynTensorID() == t3);

    merger.setHasSparseOut(true);

    // Tensor 0: undef input vector.
    merger.addExp(Kind::kTensor, t0, -1u);
    merger.setDimAndDimLevelType(t0, l0, 0, DimLevelType::Undef);

    // Tensor 1: undef input vector.
    merger.addExp(Kind::kTensor, t1, -1u);
    merger.setDimAndDimLevelType(t1, l0, 0, DimLevelType::Undef);

    // Tensor 2: sparse output vector.
    merger.addExp(Kind::kTensor, t2, -1u);
    merger.setDimAndDimLevelType(t2, l0, 0, DimLevelType::Compressed);
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
    auto em = CONJ1##Expr(t0, t1);                                             \
    auto e = CONJ2##Expr(em, t2);                                              \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto p2 = tensorPattern(t2);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),      \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),      \
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
    auto em = CONJ1##Expr(t0, t1);                                             \
    auto e = CONJ2##Expr(em, t2);                                              \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto p2 = tensorPattern(t2);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),      \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t3}}));               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),      \
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
    auto e = OP##Expr(tensor(t0), tensor(t1));                                 \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
    expectLatPointWithinRange(s, lat(1), 2, p0, loopsToBits({{l0, t0}}));      \
    expectLatPointWithinRange(s, lat(1), 2, p1, loopsToBits({{l0, t1}}));      \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
                   loopsToBits({{l0, t0}, {l0, t1}}), true);                   \
    expectLatPointWithinRange(s, lat(1), 2, p0, loopsToBits({{l0, t0}}),       \
                              true);                                           \
    expectLatPointWithinRange(s, lat(1), 2, p1, loopsToBits({{l0, t1}}),       \
                              true);                                           \
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
    auto e = OP##Expr(t0, t1);                                                 \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
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
    auto em = CONJ##Expr(t0, t1);                                              \
    auto e = DISJ##Expr(em, t2);                                               \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto p2 = tensorPattern(t2);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, lat(0), DISJ##Pattern(CONJ##Pattern(p0, p1), p2),        \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, lat(1), 2, CONJ##Pattern(p0, p1),             \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, lat(1), 2, p2, loopsToBits({{l0, t2}}));      \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, lat(0), DISJ##Pattern(CONJ##Pattern(p0, p1), p2),        \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, lat(1), 2, CONJ##Pattern(p0, p1),             \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, lat(1), 2, p2, loopsToBits({{l0, t2}}));      \
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
    auto em = DISJ1##Expr(t0, t1);                                             \
    auto e = DISJ2##Expr(em, t2);                                              \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto p2 = tensorPattern(t2);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 7);                                                  \
    expectLatPoint(s, lat(0), DISJ2##Pattern(DISJ1##Pattern(p0, p1), p2),      \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, lat(1), 6, DISJ2##Pattern(p1, p2),            \
                              loopsToBits({{l0, t1}, {l0, t2}}));              \
    expectLatPointWithinRange(s, lat(1), 6, DISJ2##Pattern(p0, p2),            \
                              loopsToBits({{l0, t0}, {l0, t2}}));              \
    expectLatPointWithinRange(s, lat(1), 6, DISJ1##Pattern(p0, p1),            \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, lat(1), 6, p2, loopsToBits({{l0, t2}}));      \
    expectLatPointWithinRange(s, lat(1), 6, p1, loopsToBits({{l0, t1}}));      \
    expectLatPointWithinRange(s, lat(1), 6, p0, loopsToBits({{l0, t0}}));      \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 7);                                                  \
    expectLatPoint(s, lat(0), DISJ2##Pattern(DISJ1##Pattern(p0, p1), p2),      \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    expectLatPointWithinRange(s, lat(1), 6, DISJ2##Pattern(p1, p2),            \
                              loopsToBits({{l0, t1}, {l0, t2}}));              \
    expectLatPointWithinRange(s, lat(1), 6, DISJ2##Pattern(p0, p2),            \
                              loopsToBits({{l0, t0}, {l0, t2}}));              \
    expectLatPointWithinRange(s, lat(1), 6, DISJ1##Pattern(p0, p1),            \
                              loopsToBits({{l0, t0}, {l0, t1}}));              \
    expectLatPointWithinRange(s, lat(1), 6, p2, loopsToBits({{l0, t2}}));      \
    expectLatPointWithinRange(s, lat(1), 6, p1, loopsToBits({{l0, t1}}));      \
    expectLatPointWithinRange(s, lat(1), 6, p0, loopsToBits({{l0, t0}}));      \
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
    auto em = CONJ1##Expr(t0, t1);                                             \
    auto e = CONJ2##Expr(em, t2);                                              \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto p2 = tensorPattern(t2);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),      \
                   loopsToBits({{l0, t0}, {l0, t1}, {l0, t2}}));               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), CONJ2##Pattern(CONJ1##Pattern(p0, p1), p2),      \
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
    auto e = OP##Expr(tensor(t0), tensor(t1));                                 \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 3);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
    expectLatPointWithinRange(s, lat(1), 2, p0, loopsToBits({{l0, t0}}));      \
    expectLatPointWithinRange(s, lat(1), 2, p1, loopsToBits({{l0, t1}}));      \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 2);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
                   loopsToBits({{l0, t0}, {l0, t1}}), true);                   \
    expectLatPoint(s, lat(1), p1, loopsToBits({{l0, t1}}), true);              \
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
    auto e = OP##Expr(t0, t1);                                                 \
    auto p0 = tensorPattern(t0);                                               \
    auto p1 = tensorPattern(t1);                                               \
    auto s = merger.buildLattices(e, l0);                                      \
                                                                               \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1),                             \
                   loopsToBits({{l0, t0}, {l0, t1}}));                         \
                                                                               \
    s = merger.optimizeSet(s);                                                 \
    expectNumLatPoints(s, 1);                                                  \
    expectLatPoint(s, lat(0), OP##Pattern(p0, p1), loopsToBits({{l0, t0}}),    \
                   true);                                                      \
  }

FOREVERY_COMMON_CONJ_BINOP(IMPL_MERGER_TEST_OPTIMIZED_CONJ)

#undef IMPL_MERGER_TEST_OPTIMIZED_CONJ

// TODO: mult-dim tests

// restore warning status
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
