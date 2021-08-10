#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

using namespace mlir::sparse_tensor;

namespace {

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
  Pattern(Kind kind, std::shared_ptr<Pattern> e0, std::shared_ptr<Pattern> e1)
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

static std::shared_ptr<Pattern> addfPattern(std::shared_ptr<Pattern> e0,
                                            std::shared_ptr<Pattern> e1) {
  return std::make_shared<Pattern>(Kind::kAddF, e0, e1);
}

static std::shared_ptr<Pattern> mulfPattern(std::shared_ptr<Pattern> e0,
                                            std::shared_ptr<Pattern> e1) {
  return std::make_shared<Pattern>(Kind::kMulF, e0, e1);
}

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

  unsigned addf(unsigned e0, unsigned e1) {
    return merger.addExp(Kind::kAddF, e0, e1);
  }

  unsigned mulf(unsigned e0, unsigned e1) {
    return merger.addExp(Kind::kMulF, e0, e1);
  }

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
  bool latPointWithinRange(unsigned s, unsigned p, unsigned n,
                           std::shared_ptr<Pattern> pattern,
                           llvm::BitVector bits) {
    for (unsigned i = p; i < p + n; ++i) {
      if (compareExpression(merger.lat(merger.set(s)[i]).exp, pattern) &&
          compareBits(s, i, bits))
        return true;
    }
    return false;
  }

  /// Wrapper over latPointWithinRange for readability of tests.
  void expectLatPointWithinRange(unsigned s, unsigned p, unsigned n,
                                 std::shared_ptr<Pattern> pattern,
                                 llvm::BitVector bits) {
    EXPECT_TRUE(latPointWithinRange(s, p, n, pattern, bits));
  }

  /// Wrapper over expectLatPointWithinRange for a single lat point.
  void expectLatPoint(unsigned s, unsigned p, std::shared_ptr<Pattern> pattern,
                      llvm::BitVector bits) {
    EXPECT_TRUE(latPointWithinRange(s, p, 1, pattern, bits));
  }

  /// Converts a vector of (loop, tensor) pairs to a bitvector with the
  /// corresponding bits set.
  llvm::BitVector
  loopsToBits(std::vector<std::pair<unsigned, unsigned>> loops) {
    llvm::BitVector testBits = llvm::BitVector(numTensors + 1, false);
    for (auto l : loops) {
      auto loop = std::get<0>(l);
      auto tensor = std::get<1>(l);
      testBits.set(numTensors * loop + tensor);
    }
    return testBits;
  }

  /// Returns true if the bits of lattice point p in set s match the given bits.
  bool compareBits(unsigned s, unsigned p, llvm::BitVector bits) {
    return merger.lat(merger.set(s)[p]).bits == bits;
  }

  /// Check that there are n lattice points in set s.
  void expectNumLatPoints(unsigned s, unsigned n) {
    EXPECT_THAT(merger.set(s).size(), n);
  }

  /// Compares expressions for equality. Equality is defined recursively as:
  /// - Two expressions can only be equal if they have the same Kind.
  /// - Two binary expressions are equal if they have the same Kind and their
  ///     children are equal.
  /// - Expressions with Kind invariant or tensor are equal if they have the
  ///     same expression id.
  bool compareExpression(unsigned e, std::shared_ptr<Pattern> pattern) {
    auto tensorExp = merger.exp(e);
    if (tensorExp.kind != pattern->kind)
      return false;
    assert(tensorExp.kind != Kind::kInvariant &&
           "Invariant comparison not yet supported");
    switch (tensorExp.kind) {
    case Kind::kTensor:
      return tensorExp.tensor == pattern->tensorNum;
    case Kind::kAbsF:
    case Kind::kCeilF:
    case Kind::kFloorF:
    case Kind::kNegF:
    case Kind::kNegI:
      return compareExpression(tensorExp.children.e0, pattern->e0);
    case Kind::kMulF:
    case Kind::kMulI:
    case Kind::kDivF:
    case Kind::kDivS:
    case Kind::kDivU:
    case Kind::kAddF:
    case Kind::kAddI:
    case Kind::kSubF:
    case Kind::kSubI:
    case Kind::kAndI:
    case Kind::kOrI:
    case Kind::kXorI:
      return compareExpression(tensorExp.children.e0, pattern->e0) &&
             compareExpression(tensorExp.children.e1, pattern->e1);
    default:
      llvm_unreachable("Unhandled Kind");
    }
  }

  unsigned numTensors;
  unsigned numLoops;
  Merger merger;
};

class MergerTest3T1L : public MergerTestBase {
protected:
  // Our three tensors (two inputs, one output).
  const unsigned t0 = 0, t1 = 1, t2 = 2;

  // Our single loop.
  const unsigned l0 = 0;

  MergerTest3T1L() : MergerTestBase(3, 1) {
    // Tensor 0: sparse input vector.
    merger.addExp(Kind::kTensor, t0, -1u);
    merger.setDim(t0, l0, Dim::kSparse);

    // Tensor 1: sparse input vector.
    merger.addExp(Kind::kTensor, t1, -1u);
    merger.setDim(t1, l0, Dim::kSparse);

    // Tensor 2: dense output vector.
    merger.addExp(Kind::kTensor, t2, -1u);
    merger.setDim(t2, l0, Dim::kDense);
  }
};

} // anonymous namespace

/// Vector addition of 2 vectors, i.e.:
///   a(i) = b(i) + c(i)
/// which should form the 3 lattice points
/// {
///   lat( i_00 i_01 / (tensor_0 + tensor_1) )
///   lat( i_00 / tensor_0 )
///   lat( i_01 / tensor_1 )
/// }
/// and after optimization, will reduce to the 2 lattice points
/// {
///   lat( i_00 i_01 / (tensor_0 + tensor_1) )
///   lat( i_00 / tensor_0 )
/// }
TEST_F(MergerTest3T1L, VectorAdd2) {
  // Construct expression.
  auto e = addf(tensor(t0), tensor(t1));

  // Build lattices and check.
  auto s = merger.buildLattices(e, l0);
  expectNumLatPoints(s, 3);
  expectLatPoint(s, lat(0), addfPattern(tensorPattern(t0), tensorPattern(t1)),
                 loopsToBits({{l0, t0}, {l0, t1}}));
  expectLatPointWithinRange(s, lat(1), 2, tensorPattern(t0),
                            loopsToBits({{l0, t0}}));
  expectLatPointWithinRange(s, lat(1), 2, tensorPattern(t1),
                            loopsToBits({{l0, t1}}));

  // Optimize lattices and check.
  s = merger.optimizeSet(s);
  expectNumLatPoints(s, 3);
  expectLatPoint(s, lat(0), addfPattern(tensorPattern(t0), tensorPattern(t1)),
                 loopsToBits({{l0, t0}, {l0, t1}}));
  expectLatPointWithinRange(s, lat(1), 2, tensorPattern(t0),
                            loopsToBits({{l0, t0}}));
  expectLatPointWithinRange(s, lat(1), 2, tensorPattern(t1),
                            loopsToBits({{l0, t1}}));
}

/// Vector multiplication of 2 vectors, i.e.:
///   a(i) = b(i) * c(i)
/// which should form the single lattice point
/// {
///   lat( i_00 i_01 / (tensor_0 * tensor_1) )
/// }
TEST_F(MergerTest3T1L, VectorMul2) {
  // Construct expression.
  auto e = mulf(t0, t1);

  // Build lattices and check.
  auto s = merger.buildLattices(e, l0);
  expectNumLatPoints(s, 1);
  expectLatPoint(s, lat(0), mulfPattern(tensorPattern(t0), tensorPattern(t1)),
                 loopsToBits({{l0, t0}, {l0, t1}}));

  // Optimize lattices and check.
  s = merger.optimizeSet(s);
  expectNumLatPoints(s, 1);
  expectLatPoint(s, lat(0), mulfPattern(tensorPattern(t0), tensorPattern(t1)),
                 loopsToBits({{l0, t0}, {l0, t1}}));
}
