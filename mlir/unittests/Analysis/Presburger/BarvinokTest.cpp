#include "mlir/Analysis/Presburger/Barvinok.h"
#include "./Utils.h"
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

  GeneratingFunction gf = unimodularConeGeneratingFunction(vertex, 1, cone);

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

  gf = unimodularConeGeneratingFunction(vertex, 1, cone);

  EXPECT_EQ_REPR_GENERATINGFUNCTION(
      gf,
      GeneratingFunction(
          1, {1}, {makeFracMatrix(2, 3, {{-83, -100, -41}, {-22, -27, -15}})},
          {{{8, 47, -17}, {-7, -41, 15}, {1, 5, -2}}}));
}
