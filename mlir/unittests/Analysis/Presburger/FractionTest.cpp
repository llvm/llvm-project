#include "mlir/Analysis/Presburger/Fraction.h"
#include "./Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(FractionTest, getAsInteger) {
  Fraction f(3, 1);
  EXPECT_EQ(f.getAsInteger(), MPInt(3));
}

TEST(FractionTest, nearIntegers) {
  Fraction f(52, 14);

  EXPECT_EQ(floor(f), 3);
  EXPECT_EQ(ceil(f), 4);
}

TEST(FractionTest, reduce) {
  Fraction f(20, 35), g(-56, 63);
  EXPECT_EQ(f, Fraction(4, 7));
  EXPECT_EQ(g, Fraction(-8, 9));
}

TEST(FractionTest, arithmetic) {
  Fraction f(3, 4), g(-2, 3);

  EXPECT_EQ(f / g, Fraction(-9, 8));
  EXPECT_EQ(f * g, Fraction(-1, 2));
  EXPECT_EQ(f + g, Fraction(1, 12));
  EXPECT_EQ(f - g, Fraction(17, 12));

  f /= g;
  EXPECT_EQ(f, Fraction(-9, 8));
  f *= g;
  EXPECT_EQ(f, Fraction(3, 4));
  f += g;
  EXPECT_EQ(f, Fraction(Fraction(1, 12)));
  f -= g;
  EXPECT_EQ(f, Fraction(3, 4));
}

TEST(FractionTest, relational) {
  Fraction f(2, 5), g(3, 7);
  EXPECT_TRUE(f < g);
  EXPECT_FALSE(g < f);

  EXPECT_EQ(f, Fraction(4, 10));
}
