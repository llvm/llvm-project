//===- TestPoly3D.cpp - Poly3D unit tests------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using namespace llvm;

class Poly3D;

namespace llvm {
template <> struct LinearPolyBaseTypeTraits<Poly3D> {
  using ScalarTy = int64_t;
  static const unsigned Dimensions = 3;
};
}

using Poly3DBase = LinearPolyBase<Poly3D>;
class Poly3D : public Poly3DBase {
public:
  using ScalarTy = Poly3DBase::ScalarTy;
  Poly3D(ScalarTy x, ScalarTy y, ScalarTy z) : Poly3DBase({x, y, z}) {}
  Poly3D(const Poly3DBase &Convert) : Poly3DBase(Convert) {}
};

TEST(LinearPolyBase, Poly3D_isZero) {
  EXPECT_TRUE(Poly3D(0, 0, 0).isZero());
  EXPECT_TRUE(Poly3D(0, 0, 1).isNonZero());
  EXPECT_TRUE(Poly3D(0, 0, 1));
}

TEST(LinearPolyBase, Poly3D_Equality) {
  EXPECT_EQ(Poly3D(1, 2, 3), Poly3D(1, 2, 3));
  EXPECT_NE(Poly3D(1, 2, 3), Poly3D(1, 2, 4));
}

TEST(LinearPolyBase, Poly3D_GetValue) {
  EXPECT_EQ(Poly3D(1, 2, 3).getValue(0), 1);
  EXPECT_EQ(Poly3D(1, 2, 3).getValue(1), 2);
  EXPECT_EQ(Poly3D(1, 2, 3).getValue(2), 3);
}

TEST(LinearPolyBase, Poly3D_Add) {
  // Test operator+
  EXPECT_EQ(Poly3D(42, 0, 0) + Poly3D(0, 42, 0) + Poly3D(0, 0, 42),
            Poly3D(42, 42, 42));

  // Test operator+=
  Poly3D X(42, 0, 0);
  X += Poly3D(0, 42, 0);
  X += Poly3D(0, 0, 42);
  EXPECT_EQ(X, Poly3D(42, 42, 42));
}

TEST(LinearPolyBase, Poly3D_Sub) {
  // Test operator-
  EXPECT_EQ(Poly3D(42, 42, 42) - Poly3D(42, 0, 0) - Poly3D(0, 42, 0) -
                Poly3D(0, 0, 42),
            Poly3D(0, 0, 0));

  // Test operator-=
  Poly3D X(42, 42, 42);
  X -= Poly3D(42, 0, 0);
  X -= Poly3D(0, 42, 0);
  X -= Poly3D(0, 0, 42);
  EXPECT_EQ(X, Poly3D(0, 0, 0));
}

TEST(LinearPolyBase, Poly3D_Scale) {
  // Test operator*
  EXPECT_EQ(Poly3D(1, 2, 4) * 2, Poly3D(2, 4, 8));
  EXPECT_EQ(Poly3D(1, 2, 4) * -2, Poly3D(-2, -4, -8));
}

TEST(LinearPolyBase, Poly3D_Invert) {
  // Test operator-
  EXPECT_EQ(-Poly3D(2, 4, 8), Poly3D(-2, -4, -8));
}

class Univariate3D;
namespace llvm {
template <> struct LinearPolyBaseTypeTraits<Univariate3D> {
  using ScalarTy = int64_t;
  static const unsigned Dimensions = 3;
};
}

using Univariate3DBase = UnivariateLinearPolyBase<Univariate3D>;
class Univariate3D : public Univariate3DBase {
public:
  using ScalarTy = Univariate3DBase::ScalarTy;
  Univariate3D(ScalarTy x, unsigned Dim) : Univariate3DBase(x, Dim) {}
  Univariate3D(const Univariate3DBase &Convert) : Univariate3DBase(Convert) {}
};

TEST(UnivariateLinearPolyBase, Univariate3D_isZero) {
  EXPECT_TRUE(Univariate3D(0, 0).isZero());
  EXPECT_TRUE(Univariate3D(0, 1).isZero());
  EXPECT_TRUE(Univariate3D(0, 2).isZero());
  EXPECT_TRUE(Univariate3D(1, 0).isNonZero());
  EXPECT_TRUE(Univariate3D(1, 1).isNonZero());
  EXPECT_TRUE(Univariate3D(1, 2).isNonZero());
  EXPECT_TRUE(Univariate3D(1, 0));
}

TEST(UnivariateLinearPolyBase, Univariate3D_Equality) {
  EXPECT_EQ(Univariate3D(1, 0), Univariate3D(1, 0));
  EXPECT_NE(Univariate3D(1, 0), Univariate3D(1, 2));
  EXPECT_NE(Univariate3D(1, 0), Univariate3D(1, 1));
  EXPECT_NE(Univariate3D(1, 0), Univariate3D(2, 0));
  EXPECT_NE(Univariate3D(1, 0), Univariate3D(0, 0));
}

TEST(UnivariateLinearPolyBase, Univariate3D_GetValue) {
  EXPECT_EQ(Univariate3D(42, 0).getValue(0), 42);
  EXPECT_EQ(Univariate3D(42, 0).getValue(1), 0);
  EXPECT_EQ(Univariate3D(42, 0).getValue(2), 0);

  EXPECT_EQ(Univariate3D(42, 1).getValue(0), 0);
  EXPECT_EQ(Univariate3D(42, 1).getValue(1), 42);
  EXPECT_EQ(Univariate3D(42, 1).getValue(2), 0);

  EXPECT_EQ(Univariate3D(42, 0).getValue(), 42);
  EXPECT_EQ(Univariate3D(42, 1).getValue(), 42);
}

TEST(UnivariateLinearPolyBase, Univariate3D_Add) {
  // Test operator+
  EXPECT_EQ(Univariate3D(42, 0) + Univariate3D(42, 0), Univariate3D(84, 0));
  EXPECT_EQ(Univariate3D(42, 1) + Univariate3D(42, 1), Univariate3D(84, 1));
  EXPECT_DEBUG_DEATH(Univariate3D(42, 0) + Univariate3D(42, 1),
                     "Invalid dimensions");

  // Test operator+=
  Univariate3D X(42, 0);
  X += Univariate3D(42, 0);
  EXPECT_EQ(X, Univariate3D(84, 0));

  // Test 'getWithIncrement' method
  EXPECT_EQ(Univariate3D(42, 0).getWithIncrement(1), Univariate3D(43, 0));
  EXPECT_EQ(Univariate3D(42, 1).getWithIncrement(2), Univariate3D(44, 1));
  EXPECT_EQ(Univariate3D(42, 2).getWithIncrement(3), Univariate3D(45, 2));
}

TEST(UnivariateLinearPolyBase, Univariate3D_Sub) {
  // Test operator+
  EXPECT_EQ(Univariate3D(84, 0) - Univariate3D(42, 0), Univariate3D(42, 0));
  EXPECT_EQ(Univariate3D(84, 1) - Univariate3D(42, 1), Univariate3D(42, 1));
  EXPECT_DEBUG_DEATH(Univariate3D(84, 0) - Univariate3D(42, 1),
                     "Invalid dimensions");

  // Test operator+=
  Univariate3D X(84, 0);
  X -= Univariate3D(42, 0);
  EXPECT_EQ(X, Univariate3D(42, 0));

  // Test 'getWithDecrement' method
  EXPECT_EQ(Univariate3D(43, 0).getWithDecrement(1), Univariate3D(42, 0));
  EXPECT_EQ(Univariate3D(44, 1).getWithDecrement(2), Univariate3D(42, 1));
  EXPECT_EQ(Univariate3D(45, 2).getWithDecrement(3), Univariate3D(42, 2));
}

TEST(UnivariateLinearPolyBase, Univariate3D_Scale) {
  // Test operator*
  EXPECT_EQ(Univariate3D(4, 0) * 2, Univariate3D(8, 0));
  EXPECT_EQ(Univariate3D(4, 1) * -2, Univariate3D(-8, 1));
}

TEST(UnivariateLinearPolyBase, Univariate3D_Invert) {
  // Test operator-
  EXPECT_EQ(-Univariate3D(4, 0), Univariate3D(-4, 0));
  EXPECT_EQ(-Univariate3D(4, 1), Univariate3D(-4, 1));
}

