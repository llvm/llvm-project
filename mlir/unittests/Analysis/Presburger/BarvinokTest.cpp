#include "mlir/Analysis/Presburger/Barvinok.h"
#include "./Utils.h"
#include "Parser.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;
using namespace mlir::presburger::detail;

/// The following are 3 randomly generated vectors with 4
/// entries each and define a cone's H-representation
/// using these numbers. We check that the dual contains
/// the same numbers.
/// We do the same in the reverse case.
TEST(BarvinokTest, getDual) {
  ConeH cone1 = defineHRep(4);
  cone1.addInequality({1, 2, 3, 4, 0});
  cone1.addInequality({3, 4, 2, 5, 0});
  cone1.addInequality({6, 2, 6, 1, 0});

  ConeV dual1 = getDual(cone1);

  EXPECT_EQ_INT_MATRIX(
      dual1, makeIntMatrix(3, 4, {{1, 2, 3, 4}, {3, 4, 2, 5}, {6, 2, 6, 1}}));

  ConeV cone2 = makeIntMatrix(3, 4, {{3, 6, 1, 5}, {3, 1, 7, 2}, {9, 3, 2, 7}});

  ConeH dual2 = getDual(cone2);

  ConeH expected = defineHRep(4);
  expected.addInequality({3, 6, 1, 5, 0});
  expected.addInequality({3, 1, 7, 2, 0});
  expected.addInequality({9, 3, 2, 7, 0});

  EXPECT_TRUE(dual2.isEqual(expected));
}

/// We randomly generate a nxn matrix to use as a cone
/// with n inequalities in n variables and check for
/// the determinant being equal to the index.
TEST(BarvinokTest, getIndex) {
  ConeV cone = makeIntMatrix(3, 3, {{4, 2, 1}, {5, 2, 7}, {4, 1, 6}});
  EXPECT_EQ(getIndex(cone), cone.determinant());

  cone = makeIntMatrix(
      4, 4, {{4, 2, 5, 1}, {4, 1, 3, 6}, {8, 2, 5, 6}, {5, 2, 5, 7}});
  EXPECT_EQ(getIndex(cone), cone.determinant());
}

// The following cones and vertices are randomly generated
// (s.t. the cones are unimodular) and the generating functions
// are computed. We check that the results contain the correct
// matrices.
TEST(BarvinokTest, unimodularConeGeneratingFunction) {
  ConeH cone = defineHRep(2);
  cone.addInequality({0, -1, 0});
  cone.addInequality({-1, -2, 0});

  ParamPoint vertex =
      makeFracMatrix(2, 3, {{2, 2, 0}, {-1, -Fraction(1, 2), 1}});

  GeneratingFunction gf =
      computeUnimodularConeGeneratingFunction(vertex, 1, cone);

  EXPECT_EQ_REPR_GENERATINGFUNCTION(
      gf, GeneratingFunction(
              2, {1},
              {makeFracMatrix(3, 2, {{-1, 0}, {-Fraction(1, 2), 1}, {1, 2}})},
              {{{2, -1}, {-1, 0}}}));

  cone = defineHRep(3);
  cone.addInequality({7, 1, 6, 0});
  cone.addInequality({9, 1, 7, 0});
  cone.addInequality({8, -1, 1, 0});

  vertex = makeFracMatrix(3, 2, {{5, 2}, {6, 2}, {7, 1}});

  gf = computeUnimodularConeGeneratingFunction(vertex, 1, cone);

  EXPECT_EQ_REPR_GENERATINGFUNCTION(
      gf,
      GeneratingFunction(
          1, {1}, {makeFracMatrix(2, 3, {{-83, -100, -41}, {-22, -27, -15}})},
          {{{8, 47, -17}, {-7, -41, 15}, {1, 5, -2}}}));
}

// The following vectors are randomly generated.
// We then check that the output of the function has non-zero
// dot product with all non-null vectors.
TEST(BarvinokTest, getNonOrthogonalVector) {
  std::vector<Point> vectors = {Point({1, 2, 3, 4}), Point({-1, 0, 1, 1}),
                                Point({2, 7, 0, 0}), Point({0, 0, 0, 0})};
  Point nonOrth = getNonOrthogonalVector(vectors);

  for (unsigned i = 0; i < 3; i++)
    EXPECT_NE(dotProduct(nonOrth, vectors[i]), 0);

  vectors = {Point({0, 1, 3}), Point({-2, -1, 1}), Point({6, 3, 0}),
             Point({0, 0, -3}), Point({5, 0, -1})};
  nonOrth = getNonOrthogonalVector(vectors);

  for (const Point &vector : vectors)
    EXPECT_NE(dotProduct(nonOrth, vector), 0);
}

// The following polynomials are randomly generated and the
// coefficients are computed by hand.
// Although the function allows the coefficients of the numerator
// to be arbitrary quasipolynomials, we stick to constants for simplicity,
// as the relevant arithmetic operations on quasipolynomials
// are tested separately.
TEST(BarvinokTest, getCoefficientInRationalFunction) {
  std::vector<QuasiPolynomial> numerator = {
      QuasiPolynomial(0, 2), QuasiPolynomial(0, 3), QuasiPolynomial(0, 5)};
  std::vector<Fraction> denominator = {Fraction(1), Fraction(0), Fraction(4),
                                       Fraction(3)};
  QuasiPolynomial coeff =
      getCoefficientInRationalFunction(1, numerator, denominator);
  EXPECT_EQ(coeff.getConstantTerm(), 3);

  numerator = {QuasiPolynomial(0, -1), QuasiPolynomial(0, 4),
               QuasiPolynomial(0, -2), QuasiPolynomial(0, 5),
               QuasiPolynomial(0, 6)};
  denominator = {Fraction(8), Fraction(4), Fraction(0), Fraction(-2)};
  coeff = getCoefficientInRationalFunction(3, numerator, denominator);
  EXPECT_EQ(coeff.getConstantTerm(), Fraction(55, 64));
}

TEST(BarvinokTest, computeNumTermsCone) {
  // The following test is taken from
  // Verdoolaege, Sven, et al. "Counting integer points in parametric
  // polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
  // 37-66.
  // It represents a right-angled triangle with right angle at the origin,
  // with height and base lengths (p/2).
  GeneratingFunction gf(
      1, {1, 1, 1},
      {makeFracMatrix(2, 2, {{0, Fraction(1, 2)}, {0, 0}}),
       makeFracMatrix(2, 2, {{0, Fraction(1, 2)}, {0, 0}}),
       makeFracMatrix(2, 2, {{0, 0}, {0, 0}})},
      {{{-1, 1}, {-1, 0}}, {{1, -1}, {0, -1}}, {{1, 0}, {0, 1}}});

  QuasiPolynomial numPoints = computeNumTerms(gf).collectTerms();

  // First, we make sure that all the affine functions are of the form ⌊p/2⌋.
  for (const std::vector<SmallVector<Fraction>> &term : numPoints.getAffine()) {
    for (const SmallVector<Fraction> &aff : term) {
      EXPECT_EQ(aff.size(), 2u);
      EXPECT_EQ(aff[0], Fraction(1, 2));
      EXPECT_EQ(aff[1], Fraction(0, 1));
    }
  }

  // Now, we can gather the like terms because we know there's only
  // either ⌊p/2⌋^2, ⌊p/2⌋, or constants.
  // The total coefficient of ⌊p/2⌋^2 is the sum of coefficients of all
  // terms with 2 affine functions, and
  // the coefficient of total ⌊p/2⌋ is the sum of coefficients of all
  // terms with 1 affine function,
  Fraction pSquaredCoeff = 0, pCoeff = 0, constantTerm = 0;
  SmallVector<Fraction> coefficients = numPoints.getCoefficients();
  for (unsigned i = 0; i < numPoints.getCoefficients().size(); i++)
    if (numPoints.getAffine()[i].size() == 2)
      pSquaredCoeff = pSquaredCoeff + coefficients[i];
    else if (numPoints.getAffine()[i].size() == 1)
      pCoeff = pCoeff + coefficients[i];

  // We expect the answer to be (1/2)⌊p/2⌋^2 + (3/2)⌊p/2⌋ + 1.
  EXPECT_EQ(pSquaredCoeff, Fraction(1, 2));
  EXPECT_EQ(pCoeff, Fraction(3, 2));
  EXPECT_EQ(numPoints.getConstantTerm(), Fraction(1, 1));

  // The following generating function corresponds to a cuboid
  // with length M (x-axis), width N (y-axis), and height P (z-axis).
  // There are eight terms.
  gf = GeneratingFunction(
      3, {1, 1, 1, 1, 1, 1, 1, 1},
      {makeFracMatrix(4, 3, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{1, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{1, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}),
       makeFracMatrix(4, 3, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}})},
      {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
       {{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
       {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
       {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
       {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
       {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
       {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
       {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}});

  numPoints = computeNumTerms(gf);
  numPoints = numPoints.collectTerms().simplify();

  // First, we make sure all the affine functions are either
  // M, N, P, or 1.
  for (const std::vector<SmallVector<Fraction>> &term : numPoints.getAffine()) {
    for (const SmallVector<Fraction> &aff : term) {
      // First, ensure that the coefficients are all nonnegative integers.
      for (const Fraction &c : aff) {
        EXPECT_TRUE(c >= 0);
        EXPECT_EQ(c, c.getAsInteger());
      }
      // Now, if the coefficients add up to 1, we can be sure the term is
      // either M, N, P, or 1.
      EXPECT_EQ(aff[0] + aff[1] + aff[2] + aff[3], 1);
    }
  }

  // We store the coefficients of M, N and P in this array.
  Fraction count[2][2][2];
  coefficients = numPoints.getCoefficients();
  for (unsigned i = 0, e = coefficients.size(); i < e; i++) {
    unsigned mIndex = 0, nIndex = 0, pIndex = 0;
    for (const SmallVector<Fraction> &aff : numPoints.getAffine()[i]) {
      if (aff[0] == 1)
        mIndex = 1;
      if (aff[1] == 1)
        nIndex = 1;
      if (aff[2] == 1)
        pIndex = 1;
      EXPECT_EQ(aff[3], 0);
    }
    count[mIndex][nIndex][pIndex] += coefficients[i];
  }

  // We expect the answer to be
  // (⌊M⌋ + 1)(⌊N⌋ + 1)(⌊P⌋ + 1) =
  // ⌊M⌋⌊N⌋⌊P⌋ + ⌊M⌋⌊N⌋ + ⌊N⌋⌊P⌋ + ⌊M⌋⌊P⌋ + ⌊M⌋ + ⌊N⌋ + ⌊P⌋ + 1.
  for (unsigned i = 0; i < 2; i++)
    for (unsigned j = 0; j < 2; j++)
      for (unsigned k = 0; k < 2; k++)
        EXPECT_EQ(count[i][j][k], 1);
}

/// We define some simple polyhedra with unimodular tangent cones and verify
/// that the returned generating functions correspond to those calculated by
/// hand.
TEST(BarvinokTest, computeNumTermsPolytope) {
  // A cube of side 1.
  PolyhedronH poly =
      parseRelationFromSet("(x, y, z) : (x >= 0, y >= 0, z >= 0, -x + 1 >= 0, "
                           "-y + 1 >= 0, -z + 1 >= 0)",
                           0);

  std::vector<std::pair<PresburgerSet, GeneratingFunction>> count =
      computePolytopeGeneratingFunction(poly);
  // There is only one chamber, as it is non-parametric.
  EXPECT_EQ(count.size(), 9u);

  GeneratingFunction gf = count[0].second;
  EXPECT_EQ_REPR_GENERATINGFUNCTION(
      gf,
      GeneratingFunction(
          0, {1, 1, 1, 1, 1, 1, 1, 1},
          {makeFracMatrix(1, 3, {{1, 1, 1}}), makeFracMatrix(1, 3, {{0, 1, 1}}),
           makeFracMatrix(1, 3, {{0, 1, 1}}), makeFracMatrix(1, 3, {{0, 0, 1}}),
           makeFracMatrix(1, 3, {{0, 1, 1}}), makeFracMatrix(1, 3, {{0, 0, 1}}),
           makeFracMatrix(1, 3, {{0, 0, 1}}),
           makeFracMatrix(1, 3, {{0, 0, 0}})},
          {{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
           {{0, 0, 1}, {-1, 0, 0}, {0, -1, 0}},
           {{0, 1, 0}, {-1, 0, 0}, {0, 0, -1}},
           {{0, 1, 0}, {0, 0, 1}, {-1, 0, 0}},
           {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
           {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}},
           {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
           {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}));

  // A right-angled triangle with side p.
  poly =
      parseRelationFromSet("(x, y)[N] : (x >= 0, y >= 0, -x - y + N >= 0)", 0);

  count = computePolytopeGeneratingFunction(poly);
  // There is only one chamber: p ≥ 0
  EXPECT_EQ(count.size(), 4u);

  gf = count[0].second;
  EXPECT_EQ_REPR_GENERATINGFUNCTION(
      gf, GeneratingFunction(
              1, {1, 1, 1},
              {makeFracMatrix(2, 2, {{0, 1}, {0, 0}}),
               makeFracMatrix(2, 2, {{0, 1}, {0, 0}}),
               makeFracMatrix(2, 2, {{0, 0}, {0, 0}})},
              {{{-1, 1}, {-1, 0}}, {{1, -1}, {0, -1}}, {{1, 0}, {0, 1}}}));

  // Cartesian product of a cube with side M and a right triangle with side N.
  poly = parseRelationFromSet(
      "(x, y, z, w, a)[M, N] : (x >= 0, y >= 0, z >= 0, -x + M >= 0, -y + M >= "
      "0, -z + M >= 0, w >= 0, a >= 0, -w - a + N >= 0)",
      0);

  count = computePolytopeGeneratingFunction(poly);

  EXPECT_EQ(count.size(), 25u);

  gf = count[0].second;
  EXPECT_EQ(gf.getNumerators().size(), 24u);
}
